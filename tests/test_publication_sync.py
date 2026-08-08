import csv
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from Advisor_Publication_Callback import requested_before_catalog
from Apply_Publications import (
    apply_backfill,
    apply_upsert,
    parse_payload,
    validate_catalog_url,
    write_catalog_state,
)
from Catalog_Publication_Poll import (
    build_backfill_payload,
    parse_catalog_metadata,
    read_applied_version,
)
from Normalise_Workflow_Key import normalise_secret


class PublicationSyncTests(unittest.TestCase):
    def test_workflow_key_normalisation_removes_paste_artifacts(self):
        self.assertEqual(
            normalise_secret('  "sk-project-key\n"  '),
            "sk-project-key",
        )
        self.assertEqual(
            normalise_secret("OPENAI_API_KEY='sk-project-key\r\n'"),
            "sk-project-key",
        )

    def test_incremental_upsert_updates_by_tool_id_without_duplicates(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset = Path(folder) / "tools.csv"
            apply_upsert({
                "toolId": "tool-1",
                "name": "Original",
                "description": "First description",
                "website": "https://example.com",
                "features": ["Writing"],
            }, dataset)
            count = apply_upsert({
                "toolId": "tool-1",
                "name": "Renamed",
                "description": "Updated description",
                "website": "https://example.com",
                "features": ["Writing", "Research"],
            }, dataset)
            self.assertEqual(count, 1)
            with open(dataset, newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["Name"], "Renamed")
            self.assertIn("Research", rows[0]["Features"])
            self.assertEqual(rows[0]["Tool_ID"], "tool-1")

    @patch("Apply_Publications.download_catalog")
    def test_backfill_replaces_dataset_and_honours_catalogue_version(self, download):
        download.return_value = {
            "catalogVersion": "v1",
            "tools": [{
                "id": "tool-2",
                "name": "Published",
                "description": "Published description",
                "website": "https://published.example",
                "category": "Writing",
                "capabilities": ["text generation"],
            }],
        }
        with tempfile.TemporaryDirectory() as folder:
            dataset = Path(folder) / "tools.csv"
            count = apply_backfill({
                "catalogVersion": "v1",
                "catalogDownloadUrl": (
                    "https://firebasestorage.googleapis.com/v0/b/"
                    "ai-discovery-platform.firebasestorage.app/o/"
                    "catalog%2Ftool-catalog-v1.json?alt=media&token=secret"
                ),
                "toolCount": 1,
            }, dataset)
            self.assertEqual(count, 1)
            with open(dataset, newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["Tool_ID"], "tool-2")
            self.assertIn("text generation", row["Features"])

    def test_payload_requires_one_supported_publication_shape(self):
        with tempfile.NamedTemporaryFile("w+", suffix=".json") as handle:
            json.dump({"unexpected": {}}, handle)
            handle.flush()
            with self.assertRaises(ValueError):
                parse_payload(handle.name)

    def test_backfill_rejects_non_commai_storage_urls(self):
        with self.assertRaises(ValueError):
            validate_catalog_url("https://attacker.example/catalog.json")

    def test_public_metadata_becomes_a_bounded_recovery_payload(self):
        metadata = parse_catalog_metadata({
            "fields": {
                "version": {"stringValue": "0123456789abcdef"},
                "downloadUrl": {"stringValue": (
                    "https://firebasestorage.googleapis.com/v0/b/"
                    "ai-discovery-platform.firebasestorage.app/o/catalog%2F"
                    "tool-catalog-0123456789abcdef.json?alt=media&token=x"
                )},
                "toolCount": {"integerValue": "2135"},
                "generatedAt": {"timestampValue": "2026-08-01T12:00:00Z"},
                "republishedAt": {
                    "timestampValue": "2026-08-08T12:00:00Z"
                },
            },
        })
        payload = build_backfill_payload(metadata)
        self.assertEqual(
            payload["backfill"]["catalogVersion"], "0123456789abcdef"
        )
        self.assertEqual(payload["backfill"]["toolCount"], 2135)
        self.assertTrue(payload["backfill"]["allowLegacyUnversioned"])
        self.assertEqual(
            payload["backfill"]["catalogGeneratedAt"],
            "2026-08-08T12:00:00Z",
        )
        self.assertTrue(payload["backfill"]["settlePending"])

    def test_catalog_state_prevents_rebuilding_the_same_version(self):
        with tempfile.TemporaryDirectory() as folder:
            state = Path(folder) / "state.json"
            write_catalog_state({
                "catalogVersion": "0123456789abcdef",
                "catalogGeneratedAt": "2026-08-08T12:00:00Z",
            }, 2135, state)
            self.assertEqual(
                read_applied_version(state), "0123456789abcdef"
            )

    def test_recovery_only_settles_requests_covered_by_catalog_snapshot(self):
        cutoff = "2026-08-08T12:00:00Z"
        self.assertTrue(requested_before_catalog({
            "requestedAt": datetime(
                2026, 8, 8, 11, 59, tzinfo=timezone.utc
            ),
        }, cutoff))
        self.assertFalse(requested_before_catalog({
            "requestedAt": datetime(
                2026, 8, 8, 12, 1, tzinfo=timezone.utc
            ),
        }, cutoff))

    @patch("Apply_Publications.download_catalog")
    def test_legacy_unversioned_catalogue_needs_explicit_compatibility(self, download):
        download.return_value = {
            "tools": [{
                "id": "legacy",
                "name": "Legacy",
                "description": "Legacy published description",
            }],
        }
        with tempfile.TemporaryDirectory() as folder:
            dataset = Path(folder) / "tools.csv"
            with self.assertRaises(ValueError):
                apply_backfill({
                    "catalogVersion": "0123456789abcdef",
                    "catalogDownloadUrl": (
                        "https://firebasestorage.googleapis.com/v0/b/"
                        "ai-discovery-platform.firebasestorage.app/o/catalog%2F"
                        "tool-catalog-legacy.json?alt=media&token=x"
                    ),
                    "toolCount": 1,
                }, dataset)
            self.assertEqual(apply_backfill({
                "catalogVersion": "0123456789abcdef",
                "catalogDownloadUrl": (
                    "https://firebasestorage.googleapis.com/v0/b/"
                    "ai-discovery-platform.firebasestorage.app/o/catalog%2F"
                    "tool-catalog-legacy.json?alt=media&token=x"
                ),
                "toolCount": 1,
                "allowLegacyUnversioned": True,
            }, dataset), 1)

    @patch("Apply_Publications.download_catalog")
    def test_backfill_preserves_long_tail_but_removes_archived_managed_rows(
        self, download
    ):
        download.return_value = {
            "catalogVersion": "v2",
            "tools": [{
                "id": "catalog-tool",
                "name": "Catalog Tool",
                "description": "Current catalogue description",
            }],
        }
        with tempfile.TemporaryDirectory() as folder:
            dataset = Path(folder) / "tools.csv"
            with open(dataset, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "Name", "Description", "Tool_ID"
                ])
                writer.writeheader()
                writer.writerows([
                    {
                        "Name": "Long Tail Specialist",
                        "Description": "Useful specialist",
                        "Tool_ID": "",
                    },
                    {
                        "Name": "Archived Managed Tool",
                        "Description": "No longer published",
                        "Tool_ID": "archived-tool",
                    },
                ])
            count = apply_backfill({
                "catalogVersion": "v2",
                "catalogDownloadUrl": (
                    "https://firebasestorage.googleapis.com/v0/b/"
                    "ai-discovery-platform.firebasestorage.app/o/catalog%2F"
                    "tool-catalog-v2.json?alt=media&token=x"
                ),
                "toolCount": 1,
            }, dataset)
            self.assertEqual(count, 2)
            with open(dataset, newline="", encoding="utf-8") as handle:
                names = {row["Name"] for row in csv.DictReader(handle)}
            self.assertEqual(names, {"Catalog Tool", "Long Tail Specialist"})


if __name__ == "__main__":
    unittest.main()
