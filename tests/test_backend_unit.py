import json
import unittest
from types import SimpleNamespace

import numpy as np
from openai import OpenAIError
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
            raise OpenAIError("embedding failed")
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
                    {"id": 0, "reason": "Tool A fits the requested writing task with matching evidence."},
                    {"id": 1, "reason": "Tool B is a backup match for the requested task."},
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

    def test_recommendation_results_are_cached(self):
        client = FakeClient()
        service = make_service(client=client)
        first = service.recommend("I need a writing tool", retrieve_k=2, final_k=2)
        second = service.recommend("I need a writing tool", retrieve_k=2, final_k=2)
        self.assertEqual(first, second)
        self.assertEqual(client.embeddings.calls, 1)
        self.assertEqual(client.chat.completions.calls, 1)

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

        fallback = local_reason(query, {
            "Name": "Hyprnote",
            "Description": "Private local-first AI notetaker for meetings.",
        })
        self.assertIn("private meeting notes", fallback)
        self.assertNotIn("Return recommendations", fallback)

    def test_returned_descriptions_are_compact(self):
        service = make_service()
        response = service.recommend("I need a writing tool", retrieve_k=2, final_k=2)
        description = response["hits"][0]["meta"]["Description"]
        self.assertLessEqual(len(description), 223)
        self.assertTrue(description.endswith("..."))


if __name__ == "__main__":
    unittest.main()
