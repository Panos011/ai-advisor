import os
import time
import unittest

import requests


API = os.getenv("API_BASE_URL", "http://localhost:10000").rstrip("/")


class LiveApiTests(unittest.TestCase):
    def test_health(self):
        r = requests.get(f"{API}/health", timeout=30)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("items", data)
        self.assertGreater(data["items"], 0)

    def test_recommend_returns_five_results(self):
        r = requests.post(
            f"{API}/recommend",
            json={"q": "I need a tool to transcribe audio", "retrieve_k": 30, "final_k": 5},
            timeout=60,
        )
        self.assertEqual(r.status_code, 200)
        hits = r.json().get("hits", [])
        self.assertEqual(len(hits), 5)

    def test_recommend_results_have_required_fields(self):
        r = requests.post(
            f"{API}/recommend",
            json={"q": "I need a free image generator", "retrieve_k": 30, "final_k": 5},
            timeout=60,
        )
        self.assertEqual(r.status_code, 200)
        hits = r.json().get("hits", [])
        for hit in hits:
            self.assertIn("meta", hit)
            self.assertIn("why", hit)
            self.assertNotEqual(hit["meta"].get("Name"), "")

    def test_clarify_returns_valid_action(self):
        r = requests.post(
            f"{API}/clarify",
            json={"q": "I need a tool"},
            timeout=30,
        )
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn(data["action"], ("clarify", "search"))

    def test_recommend_empty_query_returns_error(self):
        r = requests.post(
            f"{API}/recommend",
            json={"q": "", "retrieve_k": 30, "final_k": 5},
            timeout=30,
        )
        self.assertEqual(r.status_code, 400)

    def test_recommend_latency(self):
        start = time.time()
        r = requests.post(
            f"{API}/recommend",
            json={"q": "I need a coding assistant", "retrieve_k": 30, "final_k": 5},
            timeout=60,
        )
        self.assertEqual(r.status_code, 200)
        elapsed = time.time() - start
        self.assertLess(elapsed, 7, f"Response took {elapsed:.2f}s, exceeding 7s NFR1 target")


if __name__ == "__main__":
    unittest.main()
