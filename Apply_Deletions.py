"""Apply an admin deletion (or restore) to the advisor dataset files.

Triggered by the `advisor-tool-deleted` / `advisor-tool-restored`
repository_dispatch events that the Firebase `adminDeleteTool`,
`adminPermanentlyDeleteTool` and `adminRestoreTool` functions send.

Deletion:  add the tool to `deleted_tools.json` and strip matching rows from
           AI_tools.csv, Clean_AI_tools.csv and Embeddings_AI_tools.csv so the
           advisor can never recommend it again.
Restore:   remove the tool from `deleted_tools.json`. Its rows come back on
           the next scheduled data refresh.

Usage:
    python Apply_Deletions.py --payload payload.json
where payload.json is the dispatch client_payload, e.g.
    {"deletion": {"toolId": "...", "nameKey": "...", ...}}
    {"restore":  {"toolId": "...", "nameKey": "...", ...}}
A bare object or a {"deletions": [...]} array are accepted too.
"""

import argparse
import csv
import json
import os

from Blocklist import (
    BLOCKLIST_FILE,
    build_match_keys,
    entries_match,
    entry_keys,
    load_blocklist,
    row_match,
    save_blocklist,
    utc_now_iso,
)

DATASET_CSVS = ["AI_tools.csv", "Clean_AI_tools.csv", "Embeddings_AI_tools.csv"]


def parse_payload(path):
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)

    deletions, restores = [], []
    if isinstance(payload, dict):
        if isinstance(payload.get("deletion"), dict):
            deletions.append(payload["deletion"])
        if isinstance(payload.get("restore"), dict):
            restores.append(payload["restore"])
        if isinstance(payload.get("deletions"), list):
            deletions.extend(d for d in payload["deletions"] if isinstance(d, dict))
        if not deletions and not restores and payload.get("toolId"):
            deletions.append(payload)
    elif isinstance(payload, list):
        deletions.extend(d for d in payload if isinstance(d, dict))

    deletions = [
        d for d in deletions
        if d.get("syncStatus") != "cancelled"
        and d.get("action") in (None, "soft_delete", "permanent_delete")
    ]
    return deletions, restores


def strip_rows(csv_path, keys):
    if not os.path.exists(csv_path):
        return 0
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    kept = [row for row in rows if not row_match(row, keys)]
    removed = len(rows) - len(kept)
    if removed:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(kept)
    return removed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=True, help="Dispatch client_payload JSON file")
    args = parser.parse_args()

    deletions, restores = parse_payload(args.payload)
    if not deletions and not restores:
        print("No deletion or restore entries in payload; nothing to do.")
        write_output(dataset_changed=False)
        return

    blocklist = load_blocklist()

    for restore in restores:
        before = len(blocklist)
        blocklist = [e for e in blocklist if not entries_match(e, restore)]
        removed = before - len(blocklist)
        print(
            f"Restore {restore.get('toolId') or restore.get('name')}: "
            f"removed {removed} blocklist entr{'y' if removed == 1 else 'ies'}."
        )

    rows_removed = 0
    for deletion in deletions:
        entry = {
            **entry_keys(deletion),
            "name": deletion.get("name") or "",
            "action": deletion.get("action") or "soft_delete",
            "requestedBy": deletion.get("requestedBy") or "unknown",
            "deletedAt": utc_now_iso(),
        }
        already = any(entries_match(e, entry) for e in blocklist)
        if not already:
            blocklist.append(entry)
        keys = build_match_keys([deletion])
        for csv_path in DATASET_CSVS:
            removed = strip_rows(csv_path, keys)
            if removed:
                print(f"{csv_path}: removed {removed} row(s) for {entry['toolId'] or entry['nameKey']}")
                rows_removed += removed

    save_blocklist(blocklist)
    print(f"Blocklist now holds {len(blocklist)} tool(s).")
    write_output(dataset_changed=rows_removed > 0)


def write_output(dataset_changed: bool):
    """Expose whether the embeddings/index need a rebuild to the workflow."""
    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(f"dataset_changed={'true' if dataset_changed else 'false'}\n")


if __name__ == "__main__":
    main()
