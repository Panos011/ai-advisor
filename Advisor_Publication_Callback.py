"""Settle CommAI's durable Advisor publication queue after runtime deploy."""

import argparse
import json
import os


def target(payload):
    if isinstance(payload.get("upsert"), dict):
        return "advisorDatasetUpserts", payload["upsert"].get("toolId")
    if isinstance(payload.get("backfill"), dict):
        return "advisorDatasetBackfills", payload["backfill"].get("runId")
    return None, None


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
    firestore.client().collection(collection).document(str(document_id)).set(
        update, merge=True
    )
    print(f"Recorded {update['syncStatus']} for {collection}/{document_id}.")


if __name__ == "__main__":
    main()
