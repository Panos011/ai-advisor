import json
import os
import tempfile
import unittest

import Freshness


class ClassifyStatusTests(unittest.TestCase):
    def test_2xx_3xx_are_alive(self):
        for code in (200, 204, 301, 302, 308):
            self.assertEqual(Freshness.classify_status(code, None), "alive")

    def test_bot_blocked_codes_count_as_alive(self):
        # The domain answered; blocking our probe is not proof the tool is gone.
        for code in (401, 403, 405, 429):
            self.assertEqual(Freshness.classify_status(code, None), "alive")

    def test_404_and_410_are_dead(self):
        self.assertEqual(Freshness.classify_status(404, None), "dead")
        self.assertEqual(Freshness.classify_status(410, None), "dead")

    def test_5xx_and_errors_are_unreachable(self):
        self.assertEqual(Freshness.classify_status(500, None), "unreachable")
        self.assertEqual(Freshness.classify_status(None, "ConnectTimeout"), "unreachable")
        self.assertEqual(Freshness.classify_status(None, None), "unreachable")


class AuditTests(unittest.TestCase):
    def _rows(self):
        return [
            {"Name": "AliveTool", "Tool_link": "https://alive.example"},
            {"Name": "DeadTool", "Tool_link": "https://dead.example"},
            {"Name": "FlakyTool", "Tool_link": "https://flaky.example"},
            {"Name": "NoLinkTool", "Tool_link": ""},
        ]

    def _checker(self):
        table = {
            "https://alive.example": {"status_code": 200, "error": None},
            "https://dead.example": {"status_code": 404, "error": None},
            "https://flaky.example": {"status_code": None, "error": "ConnectTimeout"},
        }

        def checker(url):
            probe = table.get(url, {"status_code": None, "error": "not_checked"})
            return {"url": url, **probe}

        return checker

    def test_audit_classifies_each_row(self):
        results = Freshness.audit(self._rows(), self._checker(), workers=4)
        by_name = {r["name"]: r for r in results}
        self.assertEqual(by_name["AliveTool"]["status"], "alive")
        self.assertEqual(by_name["DeadTool"]["status"], "dead")
        self.assertEqual(by_name["FlakyTool"]["status"], "unreachable")
        # A row with no link is 'unknown' (nothing to check) and never pruned.
        self.assertEqual(by_name["NoLinkTool"]["status"], "unknown")

    def test_audit_populates_domain(self):
        results = Freshness.audit(self._rows(), self._checker(), workers=2)
        by_name = {r["name"]: r for r in results}
        self.assertEqual(by_name["AliveTool"]["domain"], "alive.example")

    def test_summary_counts(self):
        results = Freshness.audit(self._rows(), self._checker(), workers=2)
        summary = Freshness.summarize(results)
        self.assertEqual(summary["total"], 4)
        self.assertEqual(summary["alive"], 1)
        self.assertEqual(summary["dead"], 1)
        self.assertEqual(summary["unreachable"], 1)
        self.assertEqual(summary["unknown"], 1)


class StateAndPruningTests(unittest.TestCase):
    def _result(self, name, status, domain=None):
        return {
            "name": name,
            "url": f"https://{name.lower()}.example",
            "domain": domain if domain is not None else f"{name.lower()}.example",
            "status": status,
            "status_code": None,
            "error": None,
        }

    def test_dead_increments_and_alive_resets(self):
        state = {}
        # First failing run.
        state = Freshness.update_state(state, [self._result("Gone", "dead")], now="t1")
        self.assertEqual(state["gone.example"]["consecutive_failures"], 1)
        # Second failing run.
        state = Freshness.update_state(state, [self._result("Gone", "dead")], now="t2")
        self.assertEqual(state["gone.example"]["consecutive_failures"], 2)
        # Recovery resets to zero.
        state = Freshness.update_state(state, [self._result("Gone", "alive")], now="t3")
        self.assertEqual(state["gone.example"]["consecutive_failures"], 0)
        self.assertIsNone(state["gone.example"]["first_failed"])

    def test_unknown_carries_prior_count(self):
        state = Freshness.update_state({}, [self._result("X", "dead")], now="t1")
        state = Freshness.update_state(state, [self._result("X", "unknown")], now="t2")
        self.assertEqual(state["x.example"]["consecutive_failures"], 1)

    def test_single_failure_is_not_prune_eligible(self):
        # The core safety property: one bad run must never queue a deletion.
        state = Freshness.update_state({}, [self._result("Once", "unreachable")], now="t1")
        self.assertEqual(Freshness.eligible_for_pruning(state, threshold=2), [])

    def test_two_consecutive_failures_are_eligible(self):
        state = {}
        state = Freshness.update_state(state, [self._result("Gone", "dead")], now="t1")
        state = Freshness.update_state(state, [self._result("Gone", "dead")], now="t2")
        self.assertEqual(Freshness.eligible_for_pruning(state, threshold=2), ["gone.example"])

    def test_first_failed_timestamp_is_stable(self):
        state = Freshness.update_state({}, [self._result("Gone", "dead")], now="t1")
        state = Freshness.update_state(state, [self._result("Gone", "dead")], now="t2")
        self.assertEqual(state["gone.example"]["first_failed"], "t1")


class DeletionEntryTests(unittest.TestCase):
    def _results(self):
        return [
            {"name": "Dead One", "url": "https://deadone.example", "domain": "deadone.example",
             "status": "dead", "status_code": 404, "error": None},
            {"name": "Alive One", "url": "https://aliveone.example", "domain": "aliveone.example",
             "status": "alive", "status_code": 200, "error": None},
        ]

    def test_build_deletion_entries_shape(self):
        results = self._results()
        state = {"deadone.example": {"last_status": "dead", "consecutive_failures": 3}}
        entries = Freshness.build_deletion_entries(results, ["deadone.example"], state)
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertEqual(entry["name"], "Dead One")
        self.assertEqual(entry["domainKey"], "deadone.example")
        self.assertEqual(entry["action"], "soft_delete")
        self.assertEqual(entry["requestedBy"], "freshness-bot")
        self.assertIn("link_", entry["reason"])

    def test_merge_into_blocklist_dedupes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "deleted_tools.json")
            entry = {
                "toolId": "deadone", "name": "Dead One", "nameKey": "deadone",
                "domainKey": "deadone.example", "sourceUrlKey": "",
                "action": "soft_delete", "requestedBy": "freshness-bot",
                "deletedAt": "t", "reason": "link_dead",
            }
            added_first = Freshness.merge_into_blocklist([entry], path)
            self.assertEqual(added_first, 1)
            # A second run with the same tool must not double-add it.
            added_again = Freshness.merge_into_blocklist([dict(entry)], path)
            self.assertEqual(added_again, 0)
            with open(path, encoding="utf-8") as f:
                self.assertEqual(len(json.load(f)), 1)


if __name__ == "__main__":
    unittest.main()
