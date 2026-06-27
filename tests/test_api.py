import unittest

try:
    from fastapi.testclient import TestClient

    from api import app
except ModuleNotFoundError:
    TestClient = None
    app = None


class StubRecommender:
    def __init__(self):
        self.chat_calls = []

    def chat(
        self,
        q,
        retrieve_k,
        final_k,
        filters=None,
        mode="balanced",
        conversation_id=None,
        history=None,
        visible_tools=None,
    ):
        self.chat_calls.append({
            "q": q,
            "retrieve_k": retrieve_k,
            "final_k": final_k,
            "filters": filters,
            "mode": mode,
            "conversation_id": conversation_id,
            "history": history,
            "visible_tools": visible_tools,
        })
        return {
            "action": "chat_only",
            "hits": [],
            "message": "I can chat normally and use the advisor tools when you ask for recommendations.",
        }


class ApiTests(unittest.TestCase):
    def setUp(self):
        if TestClient is None or app is None:
            self.skipTest("fastapi is not installed in this local test environment")
        self.recommender = StubRecommender()
        app.state.recommender = self.recommender
        self.client = TestClient(app)

    def tearDown(self):
        if hasattr(app.state, "recommender"):
            delattr(app.state, "recommender")

    def test_chat_endpoint_uses_recommender_chat_and_returns_contract(self):
        response = self.client.post("/chat", json={
            "q": "can we just talk normally?",
            "retrieve_k": 4,
            "final_k": 2,
            "mode": "balanced",
            "conversation_id": "api-chat",
            "history": [{"role": "user", "content": "hello"}],
            "visible_tools": [
                {
                    "score": 0.9,
                    "meta": {
                        "Name": "CodeMate",
                        "Categories": "code assistant",
                        "Price": "Free tier",
                    },
                    "why": "Useful for coding.",
                }
            ],
        })

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["action"], "chat_only")
        self.assertEqual(payload["hits"], [])
        self.assertIn("chat normally", payload["message"])
        self.assertEqual(payload["contract"]["style"], "gpt_wrapper")
        self.assertIn("OpenAI intent planner", payload["contract"]["planner"])
        self.assertIn("OpenAI chat-only responder", payload["contract"]["conversation"])

        self.assertEqual(len(self.recommender.chat_calls), 1)
        call = self.recommender.chat_calls[0]
        self.assertEqual(call["q"], "can we just talk normally?")
        self.assertEqual(call["conversation_id"], "api-chat")
        self.assertEqual(len(call["history"]), 1)
        self.assertEqual(len(call["visible_tools"]), 1)


if __name__ == "__main__":
    unittest.main()
