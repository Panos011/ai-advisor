import json
import unittest
from types import SimpleNamespace

import numpy as np
from pydantic import ValidationError

from backend.metrics import RuntimeMetrics
from backend.retrieval import RecommendationService, ToolStore, local_reason, sanitize_reason
from backend.schemas import RecommendRequest, SearchRequest
from backend.settings import Settings


class FakeIndex:
    d = 2
    ntotal = 2

    def search(self, _vec, k):
        scores = np.array([[0.95, 0.8]], dtype="float32")
        ids = np.array([[0, 1]], dtype="int64")
        return scores[:, :k], ids[:, :k]


class FakeEmbeddings:
    def __init__(self, should_fail=False):
        self.should_fail = should_fail
        self.calls = 0

    def create(self, **_kwargs):
        self.calls += 1
        if self.should_fail:
            raise RuntimeError("embedding failed")
        return SimpleNamespace(
            data=[
                SimpleNamespace(embedding=[1.0, 0.0]),
            ]
        )


class FakeChatCompletions:
    def __init__(self):
        self.calls = 0

    def create(self, **_kwargs):
        self.calls += 1
        content = json.dumps(
            {
                "selected": [
                    {
                        "id": 0,
                        "reason": (
                            "Consultant view: Writerly is a strong writing assistant for blog posts, rewrites, and SEO content. "
                            "It also has a free tier, which makes it easy to try before paying."
                        ),
                        "summary": "Writerly helps marketers draft blog posts, rewrite copy, and prepare SEO-focused content from one workspace.",
                    },
                    {
                        "id": 1,
                        "reason": "Tool B is a backup match for the requested task.",
                        "summary": "ImageBox turns text prompts into images for quick creative drafts.",
                    },
                ]
            }
        )
        message = SimpleNamespace(content=content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeClient:
    def __init__(self, embedding_failure=False):
        self.embeddings = FakeEmbeddings(should_fail=embedding_failure)
        self.chat = SimpleNamespace(completions=FakeChatCompletions())


def make_service(client=None):
    meta = [
        {
            "Name": "Writerly",
            "Categories": "writing | copywriting",
            "Price": "Free tier",
            "Description": "Writing assistant for blog posts and marketing copy. " * 20,
            "Features": "Drafts blog posts and rewrites content.",
            "Pros": "Useful for writing content quickly.",
            "Use_cases": "Blog writing",
        },
        {
            "Name": "ImageBox",
            "Categories": "image generator",
            "Price": "Paid",
            "Description": "Generates images from text prompts.",
            "Features": "Text to image creation.",
            "Pros": "Fast image generation.",
            "Use_cases": "Image generation",
        },
    ]
    store = ToolStore(
        index=FakeIndex(),
        meta=meta,
        vectors=np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32"),
    )
    settings = Settings(cache_ttl_seconds=60, cache_max_entries=8)
    return RecommendationService(store, client or FakeClient(), settings, RuntimeMetrics())


class BackendUnitTests(unittest.TestCase):
    def test_request_validation_rejects_bad_k_values(self):
        with self.assertRaises(ValidationError):
            RecommendRequest(q="find a writing tool", retrieve_k=2, final_k=3)
        with self.assertRaises(ValidationError):
            SearchRequest(q="   ", k=5)

    def test_candidate_embeddings_use_saved_vector_matrix(self):
        service = make_service()
        vectors = service._candidate_embeddings([0, 1])
        self.assertEqual(vectors.shape, (2, 2))
        self.assertGreater(float(np.linalg.norm(vectors[0])), 0.0)
        self.assertGreater(float(np.linalg.norm(vectors[1])), 0.0)

    def test_embedding_failure_uses_keyword_fallback(self):
        service = make_service(client=FakeClient(embedding_failure=True))
        response = service.recommend("I need a free writing tool", retrieve_k=2, final_k=1)
        self.assertEqual(len(response["hits"]), 1)
        self.assertEqual(response["hits"][0]["meta"]["Name"], "Writerly")
        self.assertIn("writing", response["hits"][0]["why"].lower())
        self.assertNotIn("...", response["hits"][0]["why"])

    def test_recommendation_results_are_cached(self):
        client = FakeClient()
        service = make_service(client=client)
        first = service.recommend("I need a writing tool", retrieve_k=2, final_k=2)
        second = service.recommend("I need a writing tool", retrieve_k=2, final_k=2)
        self.assertEqual(first, second)
        self.assertEqual(client.embeddings.calls, 1)
        self.assertEqual(client.chat.completions.calls, 1)
        self.assertIn("Start with", first["message"])
        self.assertNotIn("Consultant view", first["message"])
        self.assertNotIn("Advisor view", first["message"])

    def test_explanation_prompt_is_not_treated_as_search(self):
        service = make_service()
        intent = service.detect_intent("Why these tools?", "I need a writing tool")
        self.assertEqual(intent["intent"], "explain")

        clarify = service.clarify("Why these tools?")
        self.assertEqual(clarify["action"], "explain")

        response = service.recommend("Why these tools?", retrieve_k=2, final_k=2)
        self.assertEqual(response["hits"], [])
        self.assertIn("previous results", response["message"])

    def test_vague_prompt_clarifies_without_openai(self):
        service = make_service()
        decision = service.clarify("I need a tool")
        self.assertEqual(decision["action"], "clarify")
        self.assertIn("What task", decision["question"])

    def test_reasons_do_not_expose_consultant_view_or_prompt_text(self):
        query = (
            "Find a privacy-first AI note taker for meetings. Prioritize tools with privacy, "
            "security, local, self-hosted, or compliance signals. Return recommendations as a "
            "decision shortlist with fit, tradeoffs, and practical next steps."
        )
        reason = sanitize_reason(
            "Consultant view: I treated privacy as the main constraint. Return recommendations as a decision shortlist.",
            name="Hyprnote",
            query=query,
        )
        self.assertNotIn("Consultant view", reason)
        self.assertNotIn("Return recommendations", reason)
        self.assertNotIn("...", reason)

        fallback = local_reason(query, {
            "Name": "Hyprnote",
            "Description": "Private local-first AI notetaker for meetings.",
        })
        self.assertIn("private meeting notes", fallback)
        self.assertIn("Hyprnote is well suited", fallback)
        self.assertNotIn("Return recommendations", fallback)
        self.assertNotIn("...", fallback)

    def test_returned_text_is_compact_without_ellipses(self):
        service = make_service()
        response = service.recommend("I need a writing tool", retrieve_k=2, final_k=2)
        reason = response["hits"][0]["why"]
        description = response["hits"][0]["meta"]["Description"]
        self.assertEqual(description, service.store.meta[0]["Description"])
        self.assertIn("Writerly is a strong writing assistant", reason)
        self.assertIn("free tier", reason)
        self.assertNotIn("...", description)
        self.assertNotIn("...", reason)
        self.assertNotIn("Consultant view", reason)
        self.assertNotIn("Return recommendations", reason)

    def test_original_description_is_preserved_without_ai_summary(self):
        service = make_service()
        original = service.store.meta[0]["Description"]
        response = service.search("writing", k=1)
        self.assertEqual(response["hits"][0]["meta"]["Description"], original)


if __name__ == "__main__":
    unittest.main()
