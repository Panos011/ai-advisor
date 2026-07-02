"""Two-way sync between the advisor dataset and the CommAI Firebase platform.

Subcommands
-----------
export-blocklist
    Pull active deletion requests from the `advisorDatasetDeletions`
    collection into `deleted_tools.json`. Run before Filter_Blocklist.py so
    the refresh honours deletions even if a repository_dispatch was missed.

sync
    Compare Clean_AI_tools.csv (plus flagged_tools.csv) against Firestore and
    create pending `toolSubmissions` for anything new:
      - a newly collected tool  -> pending submission (auto-collection)
      - a re-collected deleted tool -> pending submission flagged
        `flaggedPreviouslyDeleted` so admins inspect it before it can return.
    Tools already in the catalogue and submissions already reviewed are never
    touched, so curated data stays intact.

Authentication: set FIREBASE_SERVICE_ACCOUNT to the service-account JSON
(string) or GOOGLE_APPLICATION_CREDENTIALS to a key file path.

Requires: pip install firebase-admin
"""

import argparse
import csv
import json
import os
import re
import unicodedata

from Blocklist import (
    BLOCKLIST_FILE,
    clean,
    normalize_name,
    save_blocklist,
    source_url_key,
    website_domain,
)

PROJECT_ID = os.environ.get("FIREBASE_PROJECT_ID", "ai-discovery-platform")
MAX_NEW_SUBMISSIONS_DEFAULT = 250

SPECIAL_WORDS = {
    "ai": "AI", "api": "API", "seo": "SEO", "ui": "UI", "ux": "UX",
    "3d": "3D", "hr": "HR", "pdf": "PDF", "crm": "CRM",
}


# ── Field mapping (mirrors scripts/buildRecommenderImportManifest.py) ────────

def split_pipe(value, limit=40):
    values, seen = [], set()
    for item in re.split(r"\s*\|\s*", value or ""):
        item = clean(item)
        key = item.lower()
        if item and key not in seen:
            seen.add(key)
            values.append(item)
        if len(values) >= limit:
            break
    return values


def title_category(value):
    words = []
    for word in clean(value).lower().split():
        words.append(SPECIAL_WORDS.get(word, word.capitalize()))
    return " ".join(words)


def categories(value):
    result, seen = [], set()
    for item in re.split(r"[|,/]", value or ""):
        formatted = title_category(item)
        key = formatted.lower()
        if formatted and key not in seen:
            seen.add(key)
            result.append(formatted)
    return result


def slugify(value):
    normalized = unicodedata.normalize("NFKD", clean(value))
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_value.lower()).strip("-")
    return slug[:90] or "tool"


def infer_pricing_type(value):
    value = clean(value).lower()
    labels = {section.split(":", 1)[0].strip() for section in value.split("|")}
    free_labels = {
        "free", "free access", "free plan", "free service", "free tier",
        "free usage", "free version", "freemium",
    }
    has_free_plan = bool(labels & free_labels) or bool(re.search(
        r"\b(completely free|fully free|free to use|available for free|"
        r"at no cost|without charge|open[- ]source)\b", value,
    ))
    has_paid_plan = bool(re.search(
        r"[$€£]|\b(paid|pro|premium|subscription|enterprise|per month|"
        r"per year|monthly|annual)\b", value,
    ))
    if "freemium" in value or (has_free_plan and has_paid_plan):
        return "freemium"
    if has_free_plan:
        return "free"
    if "one-time" in value or "lifetime" in value:
        return "one-time"
    if has_paid_plan or "free trial" in value:
        return "subscription"
    return None


def https_only(url):
    url = clean(url)
    return url if url.startswith("https://") else ""


def build_submission(row):
    """Map one dataset CSV row to a toolSubmissions document payload."""
    name = clean(row.get("Name"))
    website = clean(row.get("Tool_link"))
    source_url = clean(row.get("Source_URL"))
    tool_categories = categories(row.get("Categories"))
    primary_category = tool_categories[0] if tool_categories else "Uncategorized"

    return {
        "name": name,
        "nameKey": normalize_name(name),
        "description": clean(row.get("Description")),
        "longDescription": clean(row.get("Unique_Value")),
        "category": primary_category,
        "categoryId": slugify(primary_category),
        "categories": tool_categories,
        "website": website,
        "domainKey": website_domain(website),
        "pricing": clean(row.get("Price")),
        "pricingType": infer_pricing_type(row.get("Price")),
        "features": split_pipe(row.get("Features")),
        "useCases": split_pipe(row.get("Use_cases")),
        "pros": split_pipe(row.get("Pros")),
        "cons": split_pipe(row.get("Cons")),
        "ratingSummary": clean(row.get("Rating")),
        "uniqueValue": clean(row.get("Unique_Value")),
        "capabilities": [],
        "images": [],
        "youtubeVideoId": "",
        "logoUrl": https_only(row.get("Logo_URL")),
        "source": "auto-collection",
        "sourceUrl": source_url,
        "sourceUrlKey": source_url_key(source_url),
    }


# ── Firestore access ─────────────────────────────────────────────────────────

def init_firestore():
    import firebase_admin
    from firebase_admin import credentials, firestore

    if not firebase_admin._apps:
        service_account_json = os.environ.get("FIREBASE_SERVICE_ACCOUNT")
        if service_account_json:
            cred = credentials.Certificate(json.loads(service_account_json))
        else:
            cred = credentials.ApplicationDefault()
        firebase_admin.initialize_app(cred, {"projectId": PROJECT_ID})
    return firestore.client()


def doc_keys(data, name_field="name"):
    """Normalised match keys for a Firestore tool/submission/deletion doc."""
    return {
        "nameKey": clean(data.get("nameKey")) or normalize_name(data.get(name_field)),
        "domainKey": clean(data.get("domainKey")) or website_domain(data.get("website")),
        "sourceUrlKey": source_url_key(data.get("sourceUrlKey") or data.get("sourceUrl")),
    }


class KeyIndex:
    """Lookup of known tools by nameKey / domainKey / sourceUrlKey."""

    def __init__(self):
        self.by_name, self.by_domain, self.by_source = {}, {}, {}

    def add(self, keys, ref_id):
        if keys["nameKey"]:
            self.by_name.setdefault(keys["nameKey"], ref_id)
        if keys["domainKey"]:
            self.by_domain.setdefault(keys["domainKey"], ref_id)
        if keys["sourceUrlKey"]:
            self.by_source.setdefault(keys["sourceUrlKey"], ref_id)

    def find(self, keys):
        return (
            self.by_source.get(keys["sourceUrlKey"]) or
            self.by_name.get(keys["nameKey"]) or
            self.by_domain.get(keys["domainKey"])
        )


# ── Subcommands ──────────────────────────────────────────────────────────────

def export_blocklist(args):
    from firebase_admin import firestore

    db = init_firestore()
    entries = []
    synced = 0
    for doc in db.collection("advisorDatasetDeletions").stream():
        data = doc.to_dict() or {}
        if data.get("syncStatus") == "cancelled":
            continue
        keys = doc_keys(data)
        entries.append({
            "toolId": clean(data.get("toolId")) or doc.id,
            "name": clean(data.get("name")),
            **keys,
            "action": data.get("action") or "soft_delete",
            "requestedBy": data.get("requestedBy") or "unknown",
            "deletedAt": str(data.get("requestedAt") or ""),
        })
        if data.get("syncStatus") == "pending":
            doc.reference.update({
                "syncStatus": "synced",
                "syncedAt": firestore.SERVER_TIMESTAMP,
            })
            synced += 1

    save_blocklist(entries, args.blocklist)
    db.collection("advisorDatasetStatus").document("current").set({
        "blocklistExportedAt": firestore.SERVER_TIMESTAMP,
        "blocklistSize": len(entries),
    }, merge=True)
    print(
        f"Exported {len(entries)} active deletion(s) to {args.blocklist} "
        f"({synced} marked as synced)."
    )


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def sync(args):
    from firebase_admin import firestore

    db = init_firestore()

    tool_index = KeyIndex()
    for doc in db.collection("tools").stream():
        tool_index.add(doc_keys(doc.to_dict() or {}), doc.id)

    submission_index = KeyIndex()
    for doc in db.collection("toolSubmissions").stream():
        submission_index.add(doc_keys(doc.to_dict() or {}), doc.id)

    deletion_index = KeyIndex()
    for doc in db.collection("advisorDatasetDeletions").stream():
        data = doc.to_dict() or {}
        if data.get("syncStatus") == "cancelled":
            continue
        deletion_index.add(doc_keys(data), doc.id)

    candidates = []
    for row in read_csv(args.csv):
        candidates.append((row, False))
    for row in read_csv(args.flagged):
        candidates.append((row, True))

    created = flagged_created = skipped = 0
    batch = db.batch()
    batch_size = 0
    seen_in_run = KeyIndex()

    for row, from_flagged_file in candidates:
        submission = build_submission(row)
        if not submission["name"] or not submission["description"]:
            skipped += 1
            continue
        keys = doc_keys(submission)

        if seen_in_run.find(keys) or tool_index.find(keys) or submission_index.find(keys):
            skipped += 1
            continue

        matched_deletion = deletion_index.find(keys)
        is_flagged = bool(matched_deletion) or from_flagged_file
        if created + flagged_created >= args.max_new:
            print(f"Reached --max-new limit ({args.max_new}); remaining tools postponed.")
            break

        submission.update({
            "submittedBy": None,
            "submittedByEmail": None,
            "submittedAt": firestore.SERVER_TIMESTAMP,
            "status": "pending",
            "reviewedBy": None,
            "reviewedAt": None,
            "adminNotes": None,
            "flaggedPreviouslyDeleted": is_flagged,
            "flagReason": (
                "This tool was previously deleted from the platform and was "
                "collected again by the dataset refresh. Inspect before approving; "
                "approving will also restore it to the AI advisor dataset."
                if is_flagged else None
            ),
            "matchedDeletionIds": [matched_deletion] if matched_deletion else [],
        })

        batch.set(db.collection("toolSubmissions").document(), submission)
        seen_in_run.add(keys, submission["nameKey"] or submission["name"])
        batch_size += 1
        if is_flagged:
            flagged_created += 1
        else:
            created += 1
        if batch_size >= 400:
            batch.commit()
            batch = db.batch()
            batch_size = 0

    if batch_size:
        batch.commit()

    summary = (
        f"{created} new pending submission(s), {flagged_created} flagged "
        f"previously deleted, {skipped} skipped"
    )
    # Surfaced in the admin dashboard's advisor dataset panel.
    db.collection("advisorDatasetStatus").document("current").set({
        "lastSyncAt": firestore.SERVER_TIMESTAMP,
        "lastSyncSummary": summary,
        "lastSyncCreated": created,
        "lastSyncFlagged": flagged_created,
        "lastSyncSkipped": skipped,
    }, merge=True)

    print(
        f"Sync complete: {summary} "
        "(already on the platform, already suggested, or incomplete)."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    export_parser = sub.add_parser("export-blocklist")
    export_parser.add_argument("--blocklist", default=BLOCKLIST_FILE)
    export_parser.set_defaults(func=export_blocklist)

    sync_parser = sub.add_parser("sync")
    sync_parser.add_argument("--csv", default="Clean_AI_tools.csv")
    sync_parser.add_argument("--flagged", default="flagged_tools.csv")
    sync_parser.add_argument("--max-new", type=int, default=MAX_NEW_SUBMISSIONS_DEFAULT)
    sync_parser.set_defaults(func=sync)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
