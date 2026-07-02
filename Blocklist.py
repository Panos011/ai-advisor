"""Shared helpers for the deleted-tools blocklist.

`deleted_tools.json` is the persistent record of every tool an admin removed
from the platform. Dataset scripts use it to keep those tools out of the
advisor CSVs/index and to flag them if the scraper collects them again.

Each entry looks like:
    {
        "toolId": "magictrips",
        "name": "MagicTrips",
        "nameKey": "magictrips",
        "domainKey": "magictrips.io",
        "sourceUrlKey": "https://www.futurepedia.io/tool/magictrips",
        "action": "soft_delete" | "permanent_delete",
        "requestedBy": "<admin uid>",
        "deletedAt": "2026-07-02T10:00:00Z"
    }
"""

import json
import os
import re
from datetime import datetime, timezone
from urllib.parse import urlparse

BLOCKLIST_FILE = "deleted_tools.json"


def clean(value) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def normalize_name(value) -> str:
    return re.sub(r"[^a-z0-9]", "", clean(value).lower())


def website_domain(value) -> str:
    value = clean(value)
    if not value:
        return ""
    parsed = urlparse(value if "://" in value else "https://" + value)
    return (parsed.hostname or "").lower().removeprefix("www.")


def source_url_key(value) -> str:
    return clean(value).lower().rstrip("/")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_blocklist(path: str = BLOCKLIST_FILE) -> list:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def save_blocklist(entries: list, path: str = BLOCKLIST_FILE) -> None:
    entries = sorted(entries, key=lambda e: str(e.get("toolId") or e.get("nameKey") or ""))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
        f.write("\n")


def entry_keys(entry: dict) -> dict:
    """Normalised match keys for one blocklist/deletion entry."""
    return {
        "toolId": clean(entry.get("toolId")),
        "nameKey": clean(entry.get("nameKey")) or normalize_name(entry.get("name")),
        "domainKey": clean(entry.get("domainKey")) or website_domain(entry.get("website")),
        "sourceUrlKey": source_url_key(entry.get("sourceUrlKey")),
    }


def build_match_keys(entries: list) -> dict:
    keys = {"name_keys": set(), "domains": set(), "source_urls": set()}
    for entry in entries:
        parsed = entry_keys(entry)
        if parsed["nameKey"]:
            keys["name_keys"].add(parsed["nameKey"])
        if parsed["domainKey"]:
            keys["domains"].add(parsed["domainKey"])
        if parsed["sourceUrlKey"]:
            keys["source_urls"].add(parsed["sourceUrlKey"])
    return keys


def row_match(row: dict, keys: dict) -> str:
    """Return which key matched a CSV row ('' when the row is clean)."""
    name_key = normalize_name(row.get("Name"))
    domain = website_domain(row.get("Tool_link"))
    source = source_url_key(row.get("Source_URL"))

    if name_key and name_key in keys["name_keys"]:
        return "nameKey"
    if domain and domain in keys["domains"]:
        return "domainKey"
    if source and source in keys["source_urls"]:
        return "sourceUrlKey"
    return ""


def entries_match(a: dict, b: dict) -> bool:
    """True when two entries refer to the same tool."""
    ka, kb = entry_keys(a), entry_keys(b)
    if ka["toolId"] and ka["toolId"] == kb["toolId"]:
        return True
    if ka["nameKey"] and ka["nameKey"] == kb["nameKey"]:
        return True
    if ka["sourceUrlKey"] and ka["sourceUrlKey"] == kb["sourceUrlKey"]:
        return True
    return False
