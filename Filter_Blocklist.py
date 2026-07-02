"""Filter freshly scraped tools against the deleted-tools blocklist.

Runs right after Data_Collection.py during the scheduled data refresh.
Any scraped tool that an admin previously deleted from the platform is:
  1. removed from AI_tools.csv (so it never reaches the advisor index), and
  2. recorded in flagged_tools.csv — full row plus audit columns — so it can
     be inspected. Sync_To_Firebase.py turns these rows into flagged pending
     submissions for the admin dashboard.

Usage:
    python Filter_Blocklist.py [--csv AI_tools.csv] [--flagged flagged_tools.csv]
"""

import argparse
import csv
import os

from Blocklist import (
    BLOCKLIST_FILE,
    build_match_keys,
    load_blocklist,
    row_match,
    utc_now_iso,
)

AUDIT_FIELDS = ["Matched_By", "First_Flagged", "Last_Seen", "Times_Seen"]


def flag_key(row):
    return (row.get("Source_URL") or "").strip() or (row.get("Name") or "").strip()


def load_flagged(path):
    if not os.path.exists(path):
        return {}, []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = [c for c in (reader.fieldnames or []) if c not in AUDIT_FIELDS]
        return {flag_key(row): row for row in reader}, fields


def save_flagged(flagged, data_fields, path):
    fieldnames = list(data_fields) + AUDIT_FIELDS
    rows = sorted(flagged.values(), key=lambda r: (r.get("Name") or "").lower())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="AI_tools.csv")
    parser.add_argument("--blocklist", default=BLOCKLIST_FILE)
    parser.add_argument("--flagged", default="flagged_tools.csv")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise SystemExit(f"Scraped CSV not found: {args.csv}")

    blocklist = load_blocklist(args.blocklist)
    keys = build_match_keys(blocklist)
    with open(args.csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    flagged, previous_fields = load_flagged(args.flagged)
    data_fields = list(dict.fromkeys(previous_fields + fieldnames))

    if not blocklist:
        # Keep the flagged report present (workflows commit it every run).
        save_flagged(flagged, data_fields, args.flagged)
        print("Blocklist is empty; nothing to filter.")
        return
    now = utc_now_iso()

    kept, removed = [], []
    for row in rows:
        matched_by = row_match(row, keys)
        if not matched_by:
            kept.append(row)
            continue
        removed.append(row)
        existing = flagged.get(flag_key(row), {})
        flagged[flag_key(row)] = {
            **row,
            "Matched_By": matched_by,
            "First_Flagged": existing.get("First_Flagged") or now,
            "Last_Seen": now,
            "Times_Seen": str(int(existing.get("Times_Seen") or 0) + 1),
        }

    if removed:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(kept)

    save_flagged(flagged, data_fields, args.flagged)
    print(
        f"Blocklist filter: removed {len(removed)} previously deleted tool(s) "
        f"from {args.csv}; {len(flagged)} tool(s) tracked in {args.flagged}."
    )
    for row in removed[:20]:
        print(f"- {row.get('Name') or 'Unnamed tool'}")


if __name__ == "__main__":
    main()
