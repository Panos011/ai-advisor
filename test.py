# logo_test.py
import csv
import os
import time
import certifi
import requests
from urllib.parse import urljoin, urlsplit
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

INPUT_CSV = "AI_tools.csv"   # your existing file
DOWNLOAD_LOGOS = True        # set False to skip saving images
BASE_URL = "https://www.futurepedia.io"

# ---- session with retries + CA bundle ----
headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.1 Safari/605.1.15",
    "Accept-Language": "en",
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
session.mount("https://", HTTPAdapter(max_retries=retries))
session.mount("http://",  HTTPAdapter(max_retries=retries))


def _norm_src(img):
    if not img:
        return ""
    # Prefer the highest-res candidate from srcset if present
    srcset = (img.get("srcset") or "").strip()
    if srcset:
        last = srcset.split(",")[-1].strip().split()[0]
        return urljoin(BASE_URL, last)
    # Fallback to plain src
    src = (img.get("src") or "").strip()
    if src:
        return urljoin(BASE_URL, src)
    return ""


def pick_logo_url(soup: BeautifulSoup) -> str:
    """
    Heuristics (ordered):
    1) <img> with 'logo' in alt/class (usually the real tool logo)
    2) Square avatar near the card: .aspect-square / .rounded-xl images
    3) OG/Twitter image ONLY if it’s not the /api/og social preview
    """
    # 1) explicit logo
    img = soup.select_one('img[alt*="logo" i], img[class*="logo" i]')
    u = _norm_src(img)
    if u:
        return u

    # 2) common square avatar on Futurepedia cards
    for sel in ("img.aspect-square", "img.rounded-xl"):
        img = soup.select_one(sel)
        u = _norm_src(img)
        if u:
            return u

    # 3) last-resort OG/Twitter image — but avoid the /api/og share image
    meta = soup.select_one('meta[property="og:image"], meta[name="twitter:image"]')
    if meta and meta.get("content"):
        u = meta["content"].strip()
        if "/api/og" not in u:
            return u

    return ""


def _guess_ext(url: str, content_type: str) -> str:
    good = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg"}
    ext = os.path.splitext(urlsplit(url).path)[1].lower()
    if ext in good:
        return ext
    ct = (content_type or "").lower()
    if "png" in ct: return ".png"
    if "webp" in ct: return ".webp"
    if "gif" in ct: return ".gif"
    if "svg" in ct: return ".svg"
    return ".jpg"


def main():
    # 1) load a few tool pages from your CSV
    sample_urls = []
    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            u = row.get("Source_URL") or row.get("Source_url") or row.get("Source url")
            if not u:
                continue
            sample_urls.append(u)

    if not sample_urls:
        print("No tool URLs found in CSV (check the Source_URL column).")
        return

    # 2) fetch each page and try to extract/download the logo
    if DOWNLOAD_LOGOS:
        os.makedirs("logos_test", exist_ok=True)

    for i, url in enumerate(sample_urls, 1):
        try:
            print(f"\n[{i}] Fetching… {url}")
            t0 = time.time()
            r = session.get(url, timeout=30)
            r.raise_for_status()
            dt = time.time() - t0
            print(f"HTTP {r.status_code} in {dt:.2f}s, {len(r.content)} bytes")

            soup = BeautifulSoup(r.content, "lxml")
            logo_url = pick_logo_url(soup)
            print("Logo_URL:", logo_url or "(not found)")

            if DOWNLOAD_LOGOS and logo_url:
                lr = session.get(logo_url, timeout=30)
                lr.raise_for_status()
                slug = url.rstrip("/").split("/")[-1]  # e.g., /tool/promptless -> promptless
                ext = _guess_ext(logo_url, lr.headers.get("Content-Type"))
                outpath = os.path.join("logos_test", f"{slug}{ext}")
                with open(outpath, "wb") as out:
                    out.write(lr.content)
                print("Saved  ->", outpath)

        except Exception as e:
            print("ERROR:", e)


if __name__ == "__main__":
    main()
