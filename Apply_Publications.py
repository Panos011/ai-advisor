"""Apply CommAI publication events to the Advisor's canonical dataset.

The script accepts either one incremental ``upsert`` or one full-catalogue
``backfill`` from a repository_dispatch payload. It never invents product
claims: every searchable field is copied from CommAI's published catalogue.
"""

import argparse
import csv
import gzip
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

from Blocklist import build_match_keys, load_blocklist, row_match, website_domain


DATASET = "AI_tools.csv"
FIELDNAMES = [
    "Name", "Logo_URL", "Logo_File", "Source_URL", "Rating",
    "Description", "Features", "Pros", "Cons", "Use_cases", "Price",
    "Tool_link", "Categories", "Unique_Value", "Tool_ID",
]
MAX_CATALOG_BYTES = 64 * 1024 * 1024
CATALOG_STATE = "index/catalog_publication.json"


def clean(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()


def strings(value):
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        if isinstance(item, dict):
            item = (
                item.get("label") or item.get("name") or
                item.get("capabilityId") or item.get("id")
            )
        item = clean(item)
        if item and item.lower() not in {entry.lower() for entry in result}:
            result.append(item)
    return result


def pipe(values):
    return " | ".join(strings(values))


def tool_row(tool):
    compatibility = tool.get("compatibility")
    compatibility = compatibility if isinstance(compatibility, dict) else {}
    capabilities = strings(tool.get("capabilities"))
    capabilities += strings(compatibility.get("capabilities"))
    features = strings(tool.get("features")) + capabilities
    categories = strings(tool.get("categories"))
    if not categories and clean(tool.get("category")):
        categories = [clean(tool.get("category"))]
    website = clean(tool.get("website"))
    return {
        "Name": clean(tool.get("name")),
        "Logo_URL": clean(tool.get("logoUrl")),
        "Logo_File": "",
        "Source_URL": website,
        "Rating": clean(tool.get("ratingSummary")),
        "Description": clean(tool.get("description")),
        "Features": pipe(features),
        "Pros": pipe(tool.get("pros")),
        "Cons": pipe(tool.get("cons")),
        "Use_cases": pipe(tool.get("useCases")),
        "Price": clean(tool.get("pricing")),
        "Tool_link": website,
        "Categories": pipe(categories),
        "Unique_Value": clean(
            tool.get("uniqueValue") or tool.get("longDescription")
        ),
        "Tool_ID": clean(tool.get("toolId") or tool.get("id")),
    }


def read_rows(path=DATASET):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_rows(rows, path=DATASET):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(
            {field: clean(row.get(field)) for field in FIELDNAMES}
            for row in rows
        )


def same_tool(row, tool):
    tool_id = clean(tool.get("toolId") or tool.get("id"))
    if tool_id and clean(row.get("Tool_ID")) == tool_id:
        return True
    website = clean(tool.get("website"))
    if website and website_domain(row.get("Tool_link")) == website_domain(website):
        return True
    return clean(row.get("Name")).lower() == clean(tool.get("name")).lower()


def apply_upsert(upsert, path=DATASET):
    row = tool_row(upsert)
    if not row["Name"] or not row["Description"]:
        raise ValueError("Incremental Advisor upsert needs a name and description.")
    rows = read_rows(path)
    matches = [index for index, existing in enumerate(rows) if same_tool(existing, upsert)]
    if matches:
        original = rows[matches[0]]
        rows[matches[0]] = {
            field: row.get(field) or clean(original.get(field))
            for field in FIELDNAMES
        }
        for index in reversed(matches[1:]):
            rows.pop(index)
    else:
        rows.append(row)
    write_rows(rows, path)
    return len(rows)


def validate_catalog_url(url):
    parsed = urlparse(url)
    decoded_path = unquote(parsed.path)
    if parsed.scheme != "https" or parsed.hostname != "firebasestorage.googleapis.com":
        raise ValueError("Backfill catalogue must come from Firebase Storage.")
    expected_prefix = (
        "/v0/b/ai-discovery-platform.firebasestorage.app/"
        "o/catalog/tool-catalog-"
    )
    if not decoded_path.startswith(expected_prefix):
        raise ValueError("Backfill URL is not a versioned CommAI catalogue artifact.")


def download_catalog(url):
    validate_catalog_url(url)
    request = Request(url, headers={"User-Agent": "CommAI-Advisor-Publisher/1.0"})
    with urlopen(request, timeout=60) as response:
        body = response.read(MAX_CATALOG_BYTES + 1)
    if len(body) > MAX_CATALOG_BYTES:
        raise ValueError("Catalogue artifact exceeds the 64 MiB safety limit.")
    if body.startswith(b"\x1f\x8b"):
        body = gzip.decompress(body)
    if len(body) > MAX_CATALOG_BYTES:
        raise ValueError("Expanded catalogue exceeds the 64 MiB safety limit.")
    payload = json.loads(body.decode("utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("tools"), list):
        raise ValueError("Catalogue artifact does not contain a tools array.")
    return payload


def apply_backfill(backfill, path=DATASET):
    catalog = download_catalog(clean(backfill.get("catalogDownloadUrl")))
    expected_version = clean(backfill.get("catalogVersion"))
    artifact_version = clean(catalog.get("catalogVersion"))
    legacy_unversioned = (
        not artifact_version
        and backfill.get("allowLegacyUnversioned") is True
    )
    if not legacy_unversioned and artifact_version != expected_version:
        raise ValueError("Downloaded catalogue version does not match the dispatch.")
    expected_count = int(backfill.get("toolCount") or 0)
    if expected_count and len(catalog["tools"]) != expected_count:
        raise ValueError("Downloaded catalogue count does not match the dispatch.")
    blocked = build_match_keys(load_blocklist())
    existing_rows = read_rows(path)
    by_id = {}
    by_name = {}
    by_domain = {}
    for index, existing in enumerate(existing_rows):
        tool_id = clean(existing.get("Tool_ID"))
        name = clean(existing.get("Name")).lower()
        domain = website_domain(
            existing.get("Tool_link") or existing.get("Source_URL")
        )
        if tool_id:
            by_id.setdefault(tool_id, []).append(index)
        if name:
            by_name.setdefault(name, []).append(index)
        if domain:
            by_domain.setdefault(domain, []).append(index)

    rows = []
    consumed_existing = set()
    seen_ids = set()
    for tool in catalog["tools"]:
        if not isinstance(tool, dict):
            raise ValueError("Catalogue tools must be JSON objects.")
        row = tool_row(tool)
        if not row["Name"] or not row["Description"]:
            raise ValueError("Every published tool needs a name and description.")
        blocked_by = row_match(row, blocked)
        if blocked_by:
            raise ValueError(
                f"Published tool {row['Name']} is still blocked by {blocked_by}."
            )
        if not row["Tool_ID"] or row["Tool_ID"] in seen_ids:
            raise ValueError("Published catalogue has a missing or duplicate tool id.")
        seen_ids.add(row["Tool_ID"])
        matches = set(by_id.get(row["Tool_ID"], []))
        matches.update(by_name.get(row["Name"].lower(), []))
        domain = website_domain(row.get("Tool_link") or row.get("Source_URL"))
        if domain:
            matches.update(by_domain.get(domain, []))
        if matches:
            original = existing_rows[min(matches)]
            row = {
                field: row.get(field) or clean(original.get(field))
                for field in FIELDNAMES
            }
            consumed_existing.update(matches)
        rows.append(row)
    published_count = len(rows)
    if published_count != len(catalog["tools"]):
        raise ValueError("Not every published tool was converted for Advisor.")

    # Rows without Tool_ID predate catalogue sync and remain valid long-tail
    # Advisor options. Preserve them unless a published row matched them. Rows
    # with Tool_ID are catalogue-managed, so an absent ID means the tool was
    # archived and must disappear from Advisor on this rebuild.
    rows.extend(
        existing for index, existing in enumerate(existing_rows)
        if index not in consumed_existing and not clean(existing.get("Tool_ID"))
    )
    write_rows(rows, path)
    return len(rows)


def write_catalog_state(backfill, row_count, path=CATALOG_STATE):
    """Record which full catalogue version the committed index represents."""
    version = clean(backfill.get("catalogVersion"))
    if not version:
        raise ValueError("Backfill catalogue version is required for state.")
    state = {
        "schema": 1,
        "catalogVersion": version,
        "catalogGeneratedAt": clean(backfill.get("catalogGeneratedAt")),
        "rows": int(row_count),
        "appliedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def parse_payload(path):
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Publication payload must be a JSON object.")
    if isinstance(payload.get("upsert"), dict):
        return "upsert", payload["upsert"]
    if isinstance(payload.get("backfill"), dict):
        return "backfill", payload["backfill"]
    raise ValueError("Publication payload needs an upsert or backfill object.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=True)
    parser.add_argument("--dataset", default=DATASET)
    args = parser.parse_args()
    kind, data = parse_payload(args.payload)
    if kind == "upsert":
        count = apply_upsert(data, args.dataset)
    else:
        count = apply_backfill(data, args.dataset)
        write_catalog_state(data, count)
    print(f"Applied Advisor {kind}; canonical dataset now has {count} tools.")


if __name__ == "__main__":
    main()
