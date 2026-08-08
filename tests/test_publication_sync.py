import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from Apply_Publications import (
    apply_backfill,
    apply_upsert,
    parse_payload,
    validate_catalog_url,
)


class PublicationSyncTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
