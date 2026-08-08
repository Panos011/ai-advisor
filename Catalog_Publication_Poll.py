"""Prepare a catalogue backfill when CommAI has published a new version.

This is the recovery half of the publication pipeline. Firebase still sends an
instant GitHub event when its credential is healthy, but this poller reads the
public catalogue pointer directly, so an expired dispatch credential cannot
leave newly published tools missing from Advisor indefinitely.
"""

import argparse
import json
import os
import re
from pathlib import Path
from urllib.request import Request, urlopen


CATALOG_METADATA_URL = (
    "https://firestore.googleapis.com/v1/projects/ai-discovery-platform/"
    "databases/(default)/documents/catalogMetadata/current"
)
DEFAULT_STATE = "index/catalog_publication.json"
VERSION_PATTERN = re.compile(r"^[a-f0-9]{16}$")


def firestore_value(value):
    """Decode the small subset of Firestore REST values used by metadata."""
    if not isinstance(value, dict):
        return None
    for key in (
        "stringValue", "timestampValue", "integerValue", "doubleValue",
        "booleanValue",
    ):
        if key not in value:
            continue
        raw = value[key]
        if key == "integerValue":
            return int(raw)
        if key == "doubleValue":
            return float(raw)
        return raw
    return None


def parse_catalog_metadata(document):
    fields = document.get("fields") if isinstance(document, dict) else None
    if not isinstance(fields, dict):
        raise ValueError("Catalogue metadata response has no Firestore fields.")
    metadata = {key: firestore_value(value) for key, value in fields.items()}
    version = str(metadata.get("version") or "").strip()
    download_url = str(metadata.get("downloadUrl") or "").strip()
    tool_count = int(metadata.get("toolCount") or 0)
    generated_at = str(
        metadata.get("republishedAt") or metadata.get("generatedAt") or ""
    ).strip()
    if not VERSION_PATTERN.fullmatch(version):
        raise ValueError("Catalogue metadata has an invalid version.")
    if not download_url.startswith(
        "https://firebasestorage.googleapis.com/v0/b/"
        "ai-discovery-platform.firebasestorage.app/o/catalog%2Ftool-catalog-"
    ):
        raise ValueError("Catalogue metadata has an unexpected download URL.")
    if tool_count < 1:
        raise ValueError("Catalogue metadata has no active tools.")
    if not generated_at:
        raise ValueError("Catalogue metadata has no generation timestamp.")
    return {
        "version": version,
        "downloadUrl": download_url,
        "toolCount": tool_count,
        "generatedAt": generated_at,
    }


def fetch_catalog_metadata(url=CATALOG_METADATA_URL):
    request = Request(url, headers={
        "Accept": "application/json",
        "User-Agent": "CommAI-Advisor-Catalogue-Poller/1.0",
    })
    with urlopen(request, timeout=30) as response:
        return parse_catalog_metadata(json.load(response))


def read_applied_version(path=DEFAULT_STATE):
    try:
        with open(path, encoding="utf-8") as handle:
            state = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        return ""
    return str(state.get("catalogVersion") or "").strip()


def build_backfill_payload(metadata):
    version = metadata["version"]
    return {
        "backfill": {
            "runId": f"scheduled-{version}",
            "requestedBy": "advisor-scheduled-recovery",
            "catalogVersion": version,
            "catalogDownloadUrl": metadata["downloadUrl"],
            "catalogGeneratedAt": metadata["generatedAt"],
            "toolCount": metadata["toolCount"],
            "settlePending": True,
            "allowLegacyUnversioned": True,
        }
    }


def append_github_output(path, key, value):
    if not path:
        return
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"{key}={value}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=True)
    parser.add_argument("--state", default=DEFAULT_STATE)
    parser.add_argument("--github-output", default=os.getenv("GITHUB_OUTPUT"))
    args = parser.parse_args()

    dispatch_payload = os.getenv("CLIENT_PAYLOAD", "").strip()
    if dispatch_payload and dispatch_payload not in {"null", "{}"}:
        payload = json.loads(dispatch_payload)
        if not isinstance(payload, dict):
            raise ValueError("Repository dispatch payload must be an object.")
        Path(args.payload).write_text(
            json.dumps(payload), encoding="utf-8"
        )
        append_github_output(args.github_output, "needs_sync", "true")
        append_github_output(args.github_output, "source", "dispatch")
        print("Using the instant Firebase publication event.")
        return

    metadata = fetch_catalog_metadata()
    current_version = read_applied_version(args.state)
    needs_sync = metadata["version"] != current_version
    Path(args.payload).write_text(
        json.dumps(build_backfill_payload(metadata)), encoding="utf-8"
    )
    append_github_output(
        args.github_output, "needs_sync", "true" if needs_sync else "false"
    )
    append_github_output(args.github_output, "source", "catalogue_poll")
    append_github_output(args.github_output, "catalog_version", metadata["version"])
    if needs_sync:
        print(
            "CommAI catalogue changed from "
            f"{current_version or 'unrecorded'} to {metadata['version']}."
        )
    else:
        print(f"Advisor already contains catalogue {current_version}.")


if __name__ == "__main__":
    main()
