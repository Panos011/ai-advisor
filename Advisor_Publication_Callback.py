"""Settle CommAI's durable Advisor publication queue after runtime deploy."""

import argparse
import json
import os
from datetime import datetime


def target(payload):
    if isinstance(payload.get("upsert"), dict):
        return "advisorDatasetUpserts", payload["upsert"].get("toolId")
    if isinstance(payload.get("backfill"), dict):
        return "advisorDatasetBackfills", payload["backfill"].get("runId")
    return None, None


def parsed_timestamp(value):
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))


def requested_before_catalog(data, catalog_generated_at):
    requested_at = data.get("requestedAt") if isinstance(data, dict) else None
    cutoff = parsed_timestamp(catalog_generated_at)
    requested = parsed_timestamp(requested_at)
    if not cutoff or not requested:
        return False
    return requested <= cutoff


def settle_pending_publications(db, backfill):
    """Settle queue records provably covered by the deployed catalogue."""
    from google.cloud.firestore_v1.base_query import FieldFilter
    from firebase_admin import firestore

    cutoff = backfill.get("catalogGeneratedAt")
    version = str(backfill.get("catalogVersion") or "")
    update = {
        "syncStatus": "synced",
        "syncError": None,
        "syncedAt": firestore.SERVER_TIMESTAMP,
        "updatedAt": firestore.SERVER_TIMESTAMP,
        "settledByCatalogVersion": version,
    }
    refs = []
    for collection_name in (
        "advisorDatasetUpserts", "advisorDatasetDeletions"
    ):
        query = db.collection(collection_name).where(
            filter=FieldFilter("syncStatus", "==", "pending")
        )
        for document in query.stream():
            if requested_before_catalog(document.to_dict(), cutoff):
                refs.append(document.reference)

    backfills = db.collection("advisorDatasetBackfills").where(
        filter=FieldFilter("syncStatus", "==", "pending")
    )
    for document in backfills.stream():
        if str(document.get("catalogVersion") or "") == version:
            refs.append(document.reference)

    for offset in range(0, len(refs), 400):
        batch = db.batch()
        for reference in refs[offset:offset + 400]:
            batch.set(reference, update, merge=True)
        batch.commit()
    return len(refs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=True)
    parser.add_argument("--outcome", required=True)
    parser.add_argument("--error", default="")
    args = parser.parse_args()
    with open(args.payload, encoding="utf-8") as handle:
        payload = json.load(handle)
    collection, document_id = target(payload)
    if not collection or not document_id:
        raise ValueError("Cannot identify the publication callback target.")

    service_account_json = os.environ.get("FIREBASE_SERVICE_ACCOUNT")
    if not service_account_json:
        raise RuntimeError("FIREBASE_SERVICE_ACCOUNT is required for callbacks.")
    import firebase_admin
    from firebase_admin import credentials, firestore

    if not firebase_admin._apps:
        firebase_admin.initialize_app(
            credentials.Certificate(json.loads(service_account_json)),
            {"projectId": "ai-discovery-platform"},
        )
    succeeded = args.outcome == "success"
    update = {
        "syncStatus": "synced" if succeeded else "failed",
        "syncError": None if succeeded else (
            args.error or "Advisor publication workflow failed."
        )[:500],
        "syncedAt" if succeeded else "failedAt": firestore.SERVER_TIMESTAMP,
        "updatedAt": firestore.SERVER_TIMESTAMP,
    }
    db = firestore.client()
    db.collection(collection).document(str(document_id)).set(
        update, merge=True
    )
    settled = 0
    backfill = payload.get("backfill")
    if (
        succeeded and isinstance(backfill, dict)
        and backfill.get("settlePending") is True
    ):
        settled = settle_pending_publications(db, backfill)
    print(f"Recorded {update['syncStatus']} for {collection}/{document_id}.")
    if settled:
        print(f"Settled {settled} older publication queue records.")


if __name__ == "__main__":
    main()
