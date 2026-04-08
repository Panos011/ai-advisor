import json, re
import os
import csv
import random
import certifi
import time
import requests
from requests.adapters import HTTPAdapter
from urllib.parse import quote_plus
from urllib3.util.retry import Retry
from urllib.parse import urljoin, urlsplit, urlunsplit
from bs4 import BeautifulSoup
SCRAPER_KEY = os.getenv("SCRAPERAPI_KEY")
OUTPUT_CSV = "AI_tools.csv"
URL = "https://www.futurepedia.io"
LOGO_DIR = "logos"
os.makedirs(LOGO_DIR, exist_ok=True)

UA_POOL = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.1 Safari/605.1.15",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36", ]

headers = {
    "User-Agent": random.choice(UA_POOL),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Referer": "https://www.futurepedia.io/ai-tools",

}
session = requests.Session()
session.verify = certifi.where()
session.headers.update(headers)
retries = Retry(
    total=6, connect=6, read=6,
    backoff_factor=0.6,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods={"GET"},
)

adapter = HTTPAdapter(max_retries=retries)
session.mount("https://", adapter)
session.mount("http://", adapter)

def fetch(url: str, tries: int = 4):
    """
    Fetch a URL via ScraperAPI if SCRAPERAPI_KEY is set, otherwise direct.
    Retries on 403/5xx with small backoff and UA rotation.
    """
    last_err = None
    for i in range(tries):
        try:
            # rotate UA a bit across retries
            session.headers["User-Agent"] = random.choice(UA_POOL)

            if SCRAPER_KEY:
                # API mode (recommended). JS rendering is OFF by default.
                # Turn it on by adding &render=true if needed.
                wrapped = (
                    "https://api.scraperapi.com/"
                    f"?api_key={SCRAPER_KEY}"
                    "&keep_headers=true"
                    "&country_code=us"
                    f"&url={quote_plus(url)}"
                )
                r = session.get(wrapped, timeout=40, allow_redirects=True)
            else:
                r = session.get(url, timeout=30, allow_redirects=True)

            if r.status_code in (403, 429, 502, 503, 504):
                time.sleep(1.5 + i)
                continue

            r.raise_for_status()
            return r

        except requests.RequestException as e:
            last_err = e
            time.sleep(1.5 + i)

    raise last_err if last_err else RuntimeError("fetch failed")
def get_category_slugs() -> list[str]:
    try:
        r = fetch(f"{URL}/ai-tools")
        soup = BeautifulSoup(r.content, "lxml")
        slugs = set()
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if not href.startswith("/ai-tools/"):
                continue
            if "?page=" in href:
                continue
            # keep only /ai-tools/<slug> (one segment)
            path = urlsplit(urljoin(URL, href)).path.rstrip("/")
            parts = path.split("/")
            if len(parts) == 3 and parts[1] == "ai-tools":
                slugs.add(parts[2])
        return sorted(slugs)
    except Exception:
        return []

PATHS = [
    "personal-assistant",
    "research-assistant",
    "spreadsheet-assistant",
    "translators",
    "presentations",
    "website-builders",
    "marketing",
    "finance",
    "project-management",
    "social-media",
    "design-generators",
    "image-generators",
    "image-editing",
    "text-to-image",
    "workflows",
    "ai-agents",
    "cartoon-generators",
    "portrait-generators",
    "avatar-generator",
    "logo-generator",
    "3D-generator",
    "audio-editing",
    "text-to-speech",
    "music-generator",
    "transcriber",
    "fitness",
    "religion",
    "students",
    "fashion-assistant",
    "gift-ideas",
    "code-assistant",
    "no-code",
    "sql-assistant"
]

def norm(u: str) -> str:
    u = urljoin(URL, u)
    s = urlsplit(u)
    path = s.path.rstrip("/")
    return urlunsplit((s.scheme, s.netloc, path, "", ""))

toollinks, seen = [], set()
for path in PATHS:
    page = 1
    while True:
        r = fetch(f"{URL}/ai-tools/{path}?page={page}")
        soup = BeautifulSoup(r.content, "lxml")

        new_on_page = 0
        for a in soup.find_all("a", href=True):
            href_raw = a["href"].strip()
            if "/tool/" not in href_raw:
                continue
            href = norm(href_raw.split("?", 1)[0])  # normalize + drop query
            if href not in seen:
                seen.add(href)
                toollinks.append(href)
                new_on_page += 1

        if new_on_page < 10:
            raw = r.text
            # try __NEXT_DATA__ block first
            data_tag = soup.find("script", id="__NEXT_DATA__", type="application/json")
            if data_tag and data_tag.string:
                try:
                    raw += "\n" + data_tag.string
                    j = json.loads(data_tag.string)
                    raw = json.dumps(j)  # search within JSON text too
                except Exception:
                    pass

            for m in re.finditer(r'"/tool/([a-z0-9-]+)"', raw, flags=re.I):
                href = norm("/tool/" + m.group(1))
                if href not in seen:
                    seen.add(href)
                    toollinks.append(href)
                    new_on_page += 1

        print(f"[{path}] page {page}: +{new_on_page} (total {len(toollinks)})")

        if new_on_page == 0:
            break
        page += 1
        time.sleep(0.5)

    print("TOTAL TOOL LINKS FOUND:", len(toollinks))

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = [
        "Name",
        "Source_URL",
        "Rating",
        "Description",
        "Features",
        "Pros",
        "Cons",
        "Use_cases",
        "Price",
        "Tool_link",
        "Categories",
        "Unique_Value",
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    if csvfile.tell() == 0:
        writer.writeheader()

    for url in toollinks:
        r = fetch(url)
        soup = BeautifulSoup(r.content, "lxml")
        # --- Tool Name ---
        ToolName = ""

        # Priority 1: og:title meta tag (server-rendered, no JS needed)
        og = soup.find("meta", property="og:title")
        if og and og.get("content"):
            ToolName = re.split(r"\s+(AI\s+)?Reviews", og["content"])[0].strip()

        # Priority 2: <title> tag
        if not ToolName:
            title_tag = soup.find("title")
            if title_tag:
                raw = title_tag.get_text(strip=True)
                ToolName = re.split(r"\s+(AI\s+)?Reviews", raw)[0].strip()

        # Priority 3: any <h1> on the page
        if not ToolName:
            h1 = soup.find("h1")
            if h1:
                ToolName = h1.get_text(" ", strip=True)

        # Priority 4: extract from URL slug as last resort
        if not ToolName:
            slug = url.rstrip("/").split("/tool/")[-1]
            ToolName = slug.replace("-", " ").title()
            print(f"(name from URL slug: {ToolName})")

        print(f"{ToolName}\n")

        if not ToolName:
            print(f"Skipping {url} — no name found")
            continue
        def slugify(s: str) -> str:
            return re.sub(r'[^a-z0-9]+', '-', (s or "").lower()).strip("-")

        slug = slugify(ToolName)

        # "How We Rated It" list
        h3Rating = soup.find(id="how-we-rated-it")
        rating_ul = h3Rating.find_next("ul") if h3Rating else None
        rating_items = [li.get_text(" ", strip=True) for li in rating_ul.find_all("li")] if rating_ul else []
        Rating = " | ".join(rating_items)
        print(Rating)

        # Tool_Description
        ToolDescription = ""
        section = soup.select_one('[id^="what-is-"]')
        if section:

            for p in section.find_all('p', recursive=True):
                txt = p.get_text(" ", strip=True)
                if txt and not re.match(r'^\s*what is\b', txt, re.I):
                    ToolDescription = txt
                    break

        # Fallback when description is not collected
        if not ToolDescription:
            heading = soup.find(
                lambda t: t.name in ('h2', 'h3', 'p')
                and re.match(r'^\s*what is\b', t.get_text(" ",strip=True), re.I )
            )
            if heading:
                node = heading.find_next()
                while node:
                    if node.name in ('h2', 'h3'):
                        break
                    if node.name == 'p':
                        txt = node.get_text(" ", strip= True)
                        if txt and not re.match(r'^\s*what is\b', txt, re.I):
                            ToolDescription = txt
                            break
                    node = node.find_next()
        # 3) Last fallback: meta description
        if not ToolDescription:
            meta = soup.find('meta', attrs={'name': 'description'})
            if meta and meta.get('content'):
                ToolDescription = meta['content'].strip()
        print("\nTool Description")
        print(f"{ToolDescription}\n")

        # Features
        features_h3 = soup.find("h3", string=lambda s: s and s.strip() == "Key Features:") or soup.find("h3",
                                                                                                        id="key-features")
        features_ul = features_h3.find_next("ul") if features_h3 else None
        Features = [li.get_text(" ", strip=True) for li in features_ul.select("li")] if features_ul else []

        print("Features:")
        print(Features)

        # CONS: <h3>Cons</h3> then the next <ul>
        cons_h3 = soup.find("h3", string=lambda s: s and s.strip().lower() == "cons")
        cons = []
        if cons_h3:
            cons_ul = cons_h3.find_next("ul")
            if cons_ul:
                cons = [li.get_text(" ", strip=True) for li in cons_ul.select("li")]

        # PROS:the <ul> immediately before the Cons heading
        pros = []
        if cons_h3:
            pros_ul = cons_h3.find_previous("ul")
            if pros_ul:
                pros = [li.get_text(" ", strip=True) for li in pros_ul.select("li")]

        print("PROS")
        for p in pros: print("-", p)

        print("\nCONS")
        for c in cons: print("-", c)
        print()

        # Who is using {ToolName}
        Uses = []
        h3Uses = soup.select_one('h2[id^="who-is-using-"], h3[id^="who-is-using-"]')
        if h3Uses:
            uses_ul = h3Uses.find_next(['ul'])
            if uses_ul:
                Uses = [li.get_text(" ", strip=True) for li in uses_ul.select("li")]

        # Pricing
        h3price = soup.find(("h3", "h2"), id="pricing")
        Price_ul = h3price.find_next("ul") if h3price else None
        Price = [li.get_text(" ", strip=True) for li in Price_ul.select("li")] if Price_ul else []

        # Get link to AI tool
        link_class = soup.find("a", attrs={"data-tool-name": ToolName})
        ToolLink = link_class.get("href") if link_class else None

        if ToolLink:
            ToolLink = ToolLink.split('?', 1)[0]
            print("\nTool Link:\n", ToolLink)

        # AI Categories
        # Find the <p> row that contains "AI Categories:" then grab its <a> tags
        cat_p = soup.select_one('p.mt-2.text-ice-700')
        categories = [a.get_text(strip=True).lower() for a in cat_p.select('a')] if cat_p else []
        print("\nCategories:\n", categories)

        # Unique value of Tool
        UniqueValue = " "
        h3_value = soup.select_one('h2[id^="what-makes-"][id$="-unique"], h3[id^="what-makes-"][id$="-unique"]')
        if not h3_value:
            h3_value = soup.find(
                lambda t: t and t.name in ('h2', 'h3')
                and re.search(r'^\s*what\s+makes\b.*\bunique\??\s*$', t.get_text(' ', strip=True), re.I)
            )
        if h3_value:
            p = h3_value.find_next('p')
            if p:
                UniqueValue = p.get_text(' ', strip=True)
            else:
                ul = h3_value.find_next(['ul', 'ol'])
                if ul:
                    UniqueValue = '; '.join(li.get_text(' ', strip=True) for li in ul.find_all('li')[:3])

        if UniqueValue:
            print("\n")
            print(UniqueValue)
        writer.writerow({
            "Name": ToolName,
            "Source_URL": url,
            "Rating": Rating,
            "Description": ToolDescription,
            "Features": Features,
            "Pros": "|".join(pros),
            "Cons": "|".join(cons),
            "Use_cases": "|".join(Uses),
            "Price": "|".join(Price),
            "Tool_link": ToolLink,
            "Categories": categories,
            "Unique_Value": UniqueValue

        })
        csvfile.flush()
print("TOTAL TOOL LINKS FOUND:", len(toollinks))

# DEBUG — paste output back to me
print(f"\n{'='*60}")
print(f"URL: {url}")
print(f"Status: {r.status_code}")
print(f"Content length: {len(r.content)}")
print(f"Title tag: {soup.find('title')}")
print(f"All h1 tags: {[h.get_text(strip=True)[:80] for h in soup.find_all('h1')]}")
print(f"__NEXT_DATA__ present: {bool(soup.find('script', id='__NEXT_DATA__'))}")
# Check if it's a captcha/block page
body_text = soup.get_text()[:500]
print(f"First 500 chars of body:\n{body_text}")
print(f"{'='*60}\n")