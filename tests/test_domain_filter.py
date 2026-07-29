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
import pathlib
import re
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


class TranscriptionClassifierTests(unittest.TestCase):
    """A bare stem inside a \\b-terminated group cannot match its inflections.

    `\\b(...|transcrib|...)\\b` requires a word boundary immediately after
    "transcrib", which "transcribe" does not have. The stem was written as a
    prefix but behaved as an exact word, so the domain's most literal phrasing
    missed its own classifier. The query then fell through to the category gate,
    which kept 9 of the 92 tools mentioning transcription and dropped Otter.ai
    and Fireflies.ai outright.
    """

    def test_classifier_matches_its_own_inflections(self):
        for phrasing in (
            "transcribe audio",
            "transcribing a call",
            "transcribed my meeting",
            "transcription service",
            "meeting notes",
        ):
            self.assertTrue(
                api.is_note_or_transcription_query(phrasing),
                f"{phrasing!r} should be recognised as a transcription query",
            )

    def test_transcription_tools_survive_a_transcription_query(self):
        meta = load_meta()
        relevant = [
            row for row in meta
            if re.search(r"\btranscri(be|bes|bing|ption|pt)\b", api.metadata_blob(row))
        ]
        self.assertGreater(len(relevant), 50)
        rule = api.active_domain_rule("transcribe audio")
        self.assertIsNotNone(rule, "transcribe audio must route to a domain rule")
        kept = [row for row in relevant if rule(row)]
        # 9/92 before the stem was fixed.
        self.assertGreater(len(kept) / len(relevant), 0.75)


class StemInflectionCoverageTests(unittest.TestCase):
    """Prefix stems must carry \\w*, but only where measurement supports it.

    Eleven sites wrote a stem as a bare alternative inside a \\b-terminated
    group, where it can never match its own inflections. Fixing all eleven
    together measured WORSE on advisor_semantic_cases — 96.9/96.5/96.9 against
    a 98.7 baseline — and the loss was reproducibly two cases: internal_tool
    100 -> 90 and paper_citations 100 -> 70.

    Isolating showed "summar" was solely responsible. Reverting just those three
    sites restored both cases to 100 and scored 99.2/98.7, at or above baseline.
    So transcrib and advertis are fixed and summar is deliberately left bare:
    widening it pulls summarisation-flavoured queries such as the literature
    review case into a domain filter that costs them their evidence-backed top
    result.
    """

    def test_transcrib_stem_matches_inflections_everywhere(self):
        for phrasing in ("transcribe audio", "transcribing calls", "transcribed notes"):
            self.assertTrue(api.is_note_or_transcription_query(phrasing), phrasing)
            self.assertTrue(
                api.is_note_or_transcription_tool(
                    {"Name": "X", "Description": f"We {phrasing} for teams."}
                ),
                phrasing,
            )

    def test_summar_is_deliberately_not_widened(self):
        """Guards the measured decision, so a future tidy-up has to re-measure."""
        source = pathlib.Path(api.__file__).read_text()
        self.assertNotIn(r"summar\w*|", source,
                         "widening 'summar' regressed internal_tool and "
                         "paper_citations by a reproducible 2.5 points; "
                         "re-run ./experiment.sh before changing this")


if __name__ == "__main__":
    unittest.main()
