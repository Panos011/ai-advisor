import unittest

from claims import extract_evidence_claims


class EvidenceClaimExtractionTests(unittest.TestCase):
    def test_claims_are_exact_source_fragments_with_stable_ids(self):
        meta = {
            "Name": "Buildly",
            "Tool_link": "https://buildly.example",
            "Description": "Turns natural-language prompts into working web applications.",
            "Features": "Code export: Download editable React source code. | Team review: Invite collaborators.",
            "Use_cases": "Rapidly prototype customer portals.",
        }

        first = extract_evidence_claims(meta, 12)
        second = extract_evidence_claims(meta, 12)

        self.assertEqual(first, second)
        self.assertTrue(all(claim["id"].startswith("tool:12:") for claim in first))
        source_text = " ".join(str(value) for value in meta.values())
        self.assertTrue(all(claim["source_quote"] in source_text for claim in first))
        self.assertEqual(
            {claim["source_field"] for claim in first},
            {"description", "features", "use_cases"},
        )

    def test_claim_limit_preserves_field_breadth(self):
        meta = {
            "Name": "ManyClaims",
            "Description": "First description claim is meaningful. Second description claim is also meaningful.",
            "Features": "Creates editable applications from prompts.",
            "Use_cases": "Rapid prototypes for internal operations teams.",
        }
        claims = extract_evidence_claims(meta, 4, max_claims=3)
        self.assertEqual(
            [claim["source_field"] for claim in claims],
            ["description", "features", "use_cases"],
        )


if __name__ == "__main__":
    unittest.main()
