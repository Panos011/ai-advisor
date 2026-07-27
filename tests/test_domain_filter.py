"""Regression tests for the query-domain topicality gate.

off_topic_for_query judged relevance from the Categories field alone. Categories
is a short taxonomy label, not a description, so a domain's vocabulary is often
absent from it even when the tool is an exact fit. For invoice queries that
combination ruled out the entire catalogue — every one of the 2225 rows — and
the advisor answered "I could not find a strong match" no matter how the
question was phrased.
"""

import json
import os
import unittest

import api


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
META_PATH = os.path.join(REPO_ROOT, "index", "meta.jsonl")


def load_meta():
    with open(META_PATH, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class DomainTopicalityTests(unittest.TestCase):
    def test_invoice_query_does_not_empty_the_catalogue(self):
        """The bug in one assertion: an invoice query must leave candidates."""
        meta = load_meta()
        survivors = [
            row for row in meta
            if not api.off_topic_for_query("invoice OCR", str(row.get("Categories", "")), row)
        ]
        self.assertGreater(len(survivors), 0)

    def test_full_record_rescues_a_tool_its_categories_do_not_describe(self):
        # Modelled on real catalogue rows: invoice-capture products filed under
        # "research" or "finance | spreadsheets", with no invoice vocabulary in
        # the category label at all.
        tool = {
            "Name": "Example Capture",
            "Categories": "research",
            "Description": "Extracts totals and VAT from invoices and receipts using OCR.",
        }
        categories = str(tool["Categories"])
        self.assertTrue(api._off_topic_by_categories("invoice OCR", categories))
        self.assertFalse(api.off_topic_for_query("invoice OCR", categories, tool))

    def test_category_verdict_stands_without_meta(self):
        """Callers that cannot supply a record keep the old behaviour exactly."""
        self.assertTrue(api.off_topic_for_query("invoice OCR", "research"))

    def test_unrelated_tool_stays_off_topic(self):
        """The rescue needs domain evidence, not merely the presence of a record."""
        tool = {
            "Name": "Example Painter",
            "Categories": "image",
            "Description": "Generates illustrations and logos from text prompts.",
        }
        self.assertTrue(
            api.off_topic_for_query("invoice OCR", str(tool["Categories"]), tool)
        )

    def test_query_without_a_domain_rule_keeps_its_category_filter(self):
        """Presentations have a category branch but no DOMAIN_RULES entry.

        Absent evidence must not read as approval, or supplying a record would
        silently disable the filter for every such query.
        """
        query = "make slides for a deck"
        self.assertIsNone(api.active_domain_rule(query))
        tool = {
            "Name": "Example Tracker",
            "Categories": "fitness",
            "Description": "Tracks running workouts.",
        }
        categories = str(tool["Categories"])
        self.assertEqual(
            api._off_topic_by_categories(query, categories),
            api.off_topic_for_query(query, categories, tool),
        )

    def test_rescue_never_removes_a_tool_the_categories_accepted(self):
        """The rescue is one-directional: it can only add candidates back."""
        meta = load_meta()
        for query in ("invoice OCR", "marketing campaign tool", "contract review",
                      "meeting notetaker", "customer support chatbot"):
            for row in meta:
                categories = str(row.get("Categories", ""))
                if not api._off_topic_by_categories(query, categories):
                    self.assertFalse(
                        api.off_topic_for_query(query, categories, row),
                        f"{query!r} dropped {row.get('Name')!r} that categories accepted",
                    )


class DomainRuleTableTests(unittest.TestCase):
    def test_matches_query_domain_agrees_with_the_rule_table(self):
        tool = {
            "Name": "Example Capture",
            "Categories": "research",
            "Description": "Extracts totals and VAT from invoices and receipts using OCR.",
        }
        rule = api.active_domain_rule("invoice OCR")
        self.assertIsNotNone(rule)
        self.assertEqual(api.matches_query_domain("invoice OCR", tool), rule(tool))

    def test_query_matching_no_domain_is_always_in_domain(self):
        self.assertIsNone(api.active_domain_rule("make slides for a deck"))
        self.assertTrue(api.matches_query_domain("make slides for a deck", {"Name": "Anything"}))


if __name__ == "__main__":
    unittest.main()
