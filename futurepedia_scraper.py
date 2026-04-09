import asyncio
import csv
import json
import re
from urllib.parse import urljoin, urlsplit, urlunsplit, urlparse, parse_qs

from crawlee.playwright_crawler import PlaywrightCrawler, PlaywrightCrawlingContext
from crawlee.configuration import Configuration
from bs4 import BeautifulSoup

URL = "https://www.futurepedia.io"
OUTPUT_CSV = "AI_tools.csv"

PATHS = [
    "personal-assistant", "research-assistant", "spreadsheet-assistant",
    "translators", "presentations", "website-builders", "marketing",
    "finance", "project-management", "social-media", "design-generators",
    "image-generators", "image-editing", "text-to-image", "workflows",
    "ai-agents", "cartoon-generators", "portrait-generators",
    "avatar-generator", "logo-generator", "3D-generator", "audio-editing",
    "text-to-speech", "music-generator", "transcriber", "fitness",
    "religion", "students", "fashion-assistant", "gift-ideas",
    "code-assistant", "no-code", "sql-assistant",
]

def norm(u: str) -> str:
    u = urljoin(URL, u)
    s = urlsplit(u)
    path = s.path.rstrip("/")
    return urlunsplit((s.scheme, s.netloc, path, "", ""))

# Shared state
seen_tools: set[str] = set()
tool_links: list[str] = []
tool_rows: list[dict] = []


# ─────────────────────────────────────────────
# Phase 1: Crawl category pages → collect tool links
# ─────────────────────────────────────────────
async def collect_tool_links():
    crawler = PlaywrightCrawler(
        configuration=Configuration(persist_storage=False),
        max_requests_per_crawl=2000,
        request_handler_timeout=60,
        headless=True,
        browser_type="chromium",
    )

    @crawler.router.default_handler
    async def category_handler(context: PlaywrightCrawlingContext):
        page = context.page
        await page.wait_for_load_state("domcontentloaded")
        await page.wait_for_timeout(2500)

        html = await page.content()
        new_on_page = 0

        # Method 1: <a> tags with /tool/
        links = await page.query_selector_all("a[href*='/tool/']")
        for link in links:
            href = await link.get_attribute("href")
            if not href:
                continue
            href = norm(href.split("?", 1)[0])
            if href not in seen_tools:
                seen_tools.add(href)
                tool_links.append(href)
                new_on_page += 1

        # Method 2: regex over raw HTML + __NEXT_DATA__
        for m in re.finditer(r'"/tool/([a-z0-9-]+)"', html, flags=re.I):
            href = norm("/tool/" + m.group(1))
            if href not in seen_tools:
                seen_tools.add(href)
                tool_links.append(href)
                new_on_page += 1

        req_url = context.request.url
        print(f"[category] {req_url}: +{new_on_page} (total {len(tool_links)})")

        # Enqueue next page if we found tools
        if new_on_page > 0:
            parsed = urlparse(req_url)
            qs = parse_qs(parsed.query)
            current_page = int(qs.get("page", [1])[0])
            next_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}?page={current_page + 1}"
            await context.add_requests([next_url])

    start_urls = [f"{URL}/ai-tools/{path}?page=1" for path in PATHS]
    await crawler.run(start_urls)
    print(f"\nTOTAL TOOL LINKS FOUND: {len(tool_links)}")


# ─────────────────────────────────────────────
# Phase 2: Scrape each tool page → extract data
# ─────────────────────────────────────────────
async def scrape_tool_pages():
    crawler = PlaywrightCrawler(
        configuration=Configuration(persist_storage=False),
        max_requests_per_crawl=len(tool_links) + 50,
        request_handler_timeout=60,
        max_concurrency=5,
        headless=True,
        browser_type="chromium",
    )

    @crawler.router.default_handler
    async def tool_handler(context: PlaywrightCrawlingContext):
        page = context.page
        await page.wait_for_load_state("domcontentloaded")
        await page.wait_for_timeout(2500)

        html = await page.content()
        soup = BeautifulSoup(html, "lxml")
        url = context.request.url

        # ── Tool Name (multi-fallback) ──────────────────────────────────────
        ToolName = ""

        og = soup.find("meta", property="og:title")
        if og and og.get("content"):
            ToolName = re.split(r"\s+(AI\s+)?Reviews", og["content"])[0].strip()

        if not ToolName:
            title_tag = soup.find("title")
            if title_tag:
                raw_title = title_tag.get_text(strip=True)
                ToolName = re.split(r"\s+(AI\s+)?Reviews", raw_title)[0].strip()

        if not ToolName:
            h1 = soup.find("h1")
            if h1:
                ToolName = h1.get_text(" ", strip=True)

        if not ToolName:
            slug = url.rstrip("/").split("/tool/")[-1]
            ToolName = slug.replace("-", " ").title()

        if not ToolName:
            print(f"Skipping {url} — no name found")
            return

        print(f"Scraping: {ToolName}")

        # ── Rating ─────────────────────────────────────────────────────────
        h3_rating = soup.find(id="how-we-rated-it")
        rating_ul = h3_rating.find_next("ul") if h3_rating else None
        rating_items = (
            [li.get_text(" ", strip=True) for li in rating_ul.find_all("li")]
            if rating_ul else []
        )
        Rating = " | ".join(rating_items)

        # ── Description ────────────────────────────────────────────────────
        ToolDescription = ""
        section = soup.select_one('[id^="what-is-"]')
        if section:
            for p in section.find_all("p", recursive=True):
                txt = p.get_text(" ", strip=True)
                if txt and not re.match(r"^\s*what is\b", txt, re.I):
                    ToolDescription = txt
                    break

        if not ToolDescription:
            heading = soup.find(
                lambda t: t.name in ("h2", "h3", "p")
                and re.match(r"^\s*what is\b", t.get_text(" ", strip=True), re.I)
            )
            if heading:
                node = heading.find_next()
                while node:
                    if node.name in ("h2", "h3"):
                        break
                    if node.name == "p":
                        txt = node.get_text(" ", strip=True)
                        if txt and not re.match(r"^\s*what is\b", txt, re.I):
                            ToolDescription = txt
                            break
                    node = node.find_next()

        if not ToolDescription:
            meta = soup.find("meta", attrs={"name": "description"})
            if meta and meta.get("content"):
                ToolDescription = meta["content"].strip()

        # ── Features ───────────────────────────────────────────────────────
        features_h3 = (
            soup.find("h3", string=lambda s: s and s.strip() == "Key Features:")
            or soup.find("h3", id="key-features")
        )
        features_ul = features_h3.find_next("ul") if features_h3 else None
        Features = (
            [li.get_text(" ", strip=True) for li in features_ul.select("li")]
            if features_ul else []
        )

        # ── Pros / Cons ────────────────────────────────────────────────────
        cons_h3 = soup.find("h3", string=lambda s: s and s.strip().lower() == "cons")
        cons = []
        if cons_h3:
            cons_ul = cons_h3.find_next("ul")
            if cons_ul:
                cons = [li.get_text(" ", strip=True) for li in cons_ul.select("li")]

        pros = []
        if cons_h3:
            pros_ul = cons_h3.find_previous("ul")
            if pros_ul:
                pros = [li.get_text(" ", strip=True) for li in pros_ul.select("li")]

        # ── Use Cases ──────────────────────────────────────────────────────
        Uses = []
        h3_uses = soup.select_one('h2[id^="who-is-using-"], h3[id^="who-is-using-"]')
        if h3_uses:
            uses_ul = h3_uses.find_next("ul")
            if uses_ul:
                Uses = [li.get_text(" ", strip=True) for li in uses_ul.select("li")]

        # ── Pricing ────────────────────────────────────────────────────────
        h3_price = soup.find(("h3", "h2"), id="pricing")
        price_ul = h3_price.find_next("ul") if h3_price else None
        Price = (
            [li.get_text(" ", strip=True) for li in price_ul.select("li")]
            if price_ul else []
        )

        # ── Tool Link ──────────────────────────────────────────────────────
        link_el = soup.find("a", attrs={"data-tool-name": ToolName})
        ToolLink = link_el.get("href") if link_el else None
        if ToolLink:
            ToolLink = ToolLink.split("?", 1)[0]

        # ── Categories ─────────────────────────────────────────────────────
        cat_p = soup.select_one("p.mt-2.text-ice-700")
        categories = (
            [a.get_text(strip=True).lower() for a in cat_p.select("a")]
            if cat_p else []
        )

        # ── Unique Value ───────────────────────────────────────────────────
        UniqueValue = " "
        h3_value = soup.select_one(
            'h2[id^="what-makes-"][id$="-unique"], h3[id^="what-makes-"][id$="-unique"]'
        )
        if not h3_value:
            h3_value = soup.find(
                lambda t: t and t.name in ("h2", "h3")
                and re.search(
                    r"^\s*what\s+makes\b.*\bunique\??\s*$",
                    t.get_text(" ", strip=True), re.I,
                )
            )
        if h3_value:
            p = h3_value.find_next("p")
            if p:
                UniqueValue = p.get_text(" ", strip=True)
            else:
                ul = h3_value.find_next(["ul", "ol"])
                if ul:
                    UniqueValue = "; ".join(
                        li.get_text(" ", strip=True) for li in ul.find_all("li")[:3]
                    )

        tool_rows.append({
            "Name": ToolName,
            "Source_URL": url,
            "Rating": Rating,
            "Description": ToolDescription,
            "Features": "|".join(Features),
            "Pros": "|".join(pros),
            "Cons": "|".join(cons),
            "Use_cases": "|".join(Uses),
            "Price": "|".join(Price),
            "Tool_link": ToolLink,
            "Categories": "|".join(categories),
            "Unique_Value": UniqueValue,
        })

    await crawler.run(tool_links)


# ─────────────────────────────────────────────
# Phase 3: Write CSV
# ─────────────────────────────────────────────
def write_csv():
    fieldnames = [
        "Name", "Source_URL", "Rating", "Description", "Features",
        "Pros", "Cons", "Use_cases", "Price", "Tool_link",
        "Categories", "Unique_Value",
    ]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in tool_rows:
            writer.writerow(row)
    print(f"\nWrote {len(tool_rows)} tools to {OUTPUT_CSV}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
async def main():
    print("=" * 60)
    print("Futurepedia AI Tools Scraper — Crawlee + Playwright")
    print("=" * 60)

    print("\nPhase 1: Collecting tool links from category pages...\n")
    await collect_tool_links()

    print(f"\nPhase 2: Scraping {len(tool_links)} tool pages...\n")
    await scrape_tool_pages()

    print("\nPhase 3: Writing CSV...\n")
    write_csv()

    print(f"\nDone! Total tools scraped: {len(tool_rows)}")
    print(f"Output saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    asyncio.run(main())
