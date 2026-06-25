import json
import unittest
from types import SimpleNamespace

import numpy as np
from pydantic import ValidationError

from backend.metrics import RuntimeMetrics
from backend.retrieval import (
    ConversationStore,
    RecommendationService,
    ToolStore,
    build_retrieval_query,
    clean_best_for,
    focus_latest_intent,
    has_explicit_task,
    is_free_tool,
    local_reason,
    merge_history_messages,
    normalize_mode,
    recommendation_message,
    request_goal,
    sanitize_reason,
)
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

    def test_free_only_recommendations_exclude_paid_tools(self):
        service = make_service()
        response = service.recommend("I need a free writing tool", retrieve_k=2, final_k=2)
        names = [hit["meta"]["Name"] for hit in response["hits"]]

        self.assertEqual(names, ["Writerly"])
        self.assertTrue(all(is_free_tool(hit["meta"]) for hit in response["hits"]))
        self.assertNotIn("ImageBox", names)
        self.assertNotIn("$", response["hits"][0]["why"])

    def test_free_only_keyword_fallback_does_not_return_paid_task_match(self):
        service = make_service(client=FakeClient(embedding_failure=True))
        response = service.recommend("I need a free image generator", retrieve_k=2, final_k=2)
        names = [hit["meta"]["Name"] for hit in response["hits"]]

        self.assertNotIn("ImageBox", names)
        self.assertTrue(all(is_free_tool(hit["meta"]) for hit in response["hits"]))

    def test_free_only_without_task_asks_for_task(self):
        service = make_service()

        clarify = service.clarify("I said only the free ones")
        self.assertEqual(clarify["action"], "clarify")
        self.assertIn("What task", clarify["question"])

        response = service.recommend("I said only the free ones", retrieve_k=2, final_k=2)
        self.assertEqual(response["hits"], [])
        self.assertIn("What task", response["message"])

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

    def test_spaced_explanation_prompt_is_not_treated_as_search(self):
        service = make_service()
        intent = service.detect_intent("Why these tools ?", "I need a writing tool")
        self.assertEqual(intent["intent"], "explain")

        clarify = service.clarify("Why these tools ?")
        self.assertEqual(clarify["action"], "explain")

        response = service.recommend("Why these tools ?", retrieve_k=2, final_k=2)
        self.assertEqual(response["hits"], [])
        self.assertIn("previous results", response["message"])

    def test_feedback_prompt_is_not_treated_as_search(self):
        service = make_service()
        prompt = "Wtf. Act like a practical software consultant."

        intent = service.detect_intent(prompt, "Find a workflow tool")
        self.assertEqual(intent["intent"], "new")

        clarify = service.clarify(prompt)
        self.assertEqual(clarify["action"], "clarify")
        self.assertIn("feedback", clarify["question"])

        response = service.recommend(prompt, retrieve_k=2, final_k=2)
        self.assertEqual(response["hits"], [])
        self.assertIn("feedback", response["message"])
        self.assertNotIn("Software Ag", response["message"])

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

    def test_normalize_mode_maps_ui_labels(self):
        self.assertEqual(normalize_mode("balanced"), "best_fit")
        self.assertEqual(normalize_mode("Best Fit"), "best_fit")
        self.assertEqual(normalize_mode("One best"), "one_best")
        self.assertEqual(normalize_mode("compare"), "compare")
        self.assertEqual(normalize_mode("nonsense"), "best_fit")

    def test_request_goal_handles_chatbot_and_coding(self):
        self.assertEqual(request_goal("I am creating a chatbot so I need the best AI tool"), "building a chatbot")
        self.assertEqual(request_goal("I need a coding tool"), "coding and development")

    def test_clean_best_for_rejects_query_echo(self):
        meta = {"Name": "DevTool", "Categories": "developer tools"}
        query = "I need a coding tool to build a chatbot"
        echoed = clean_best_for(query, query, meta)
        self.assertNotEqual(echoed.lower(), query.lower())
        self.assertIn("using", echoed)

        short = clean_best_for("Building customer support chatbots", query, meta)
        self.assertEqual(short, "Building customer support chatbots")

    def test_recommendation_message_is_mode_aware(self):
        hits = [
            {"meta": {"Name": "Alpha"}},
            {"meta": {"Name": "Beta"}},
        ]
        best_fit = recommendation_message(hits, "I need a coding tool", "balanced")
        self.assertIn("Start with", best_fit)

        one_best = recommendation_message(hits, "I need a coding tool", "one_best")
        self.assertIn("top pick", one_best)

        compare = recommendation_message(hits, "I need a coding tool", "compare")
        self.assertIn("comparison", compare.lower())
        self.assertIn("Alpha", compare)
        self.assertIn("Beta", compare)

    def test_one_best_mode_returns_single_hit(self):
        service = make_service()
        response = service.recommend("I need a writing tool", retrieve_k=2, final_k=2, mode="one_best")
        self.assertEqual(len(response["hits"]), 1)
        self.assertIn("top pick", response["message"])

    def test_build_retrieval_query_merges_context_without_duplicates(self):
        query = build_retrieval_query(
            "I am creating a chatbot",
            ["I need a coding tool", "I am creating a chatbot"],
        )
        self.assertTrue(query.startswith("I need a coding tool"))
        self.assertTrue(query.endswith("I am creating a chatbot"))
        self.assertEqual(query.lower().count("chatbot"), 1)

    def test_merge_history_messages_keeps_user_turns_only(self):
        history = [
            {"role": "user", "content": "I need a coding tool"},
            {"role": "assistant", "content": "Start with X"},
            {"role": "user", "content": "for a chatbot"},
        ]
        messages = merge_history_messages(history)
        self.assertEqual(messages, ["I need a coding tool", "for a chatbot"])

    def test_conversation_store_caps_turns_and_scopes_by_id(self):
        store = ConversationStore(max_conversations=4, ttl_seconds=60, max_turns=2)
        store.append("c1", "user", "one")
        store.append("c1", "user", "two")
        store.append("c1", "user", "three")
        turns = store.get("c1")
        self.assertEqual([t["content"] for t in turns], ["two", "three"])
        self.assertEqual(store.get("other"), [])
        self.assertEqual(store.get(None), [])

    def test_conversation_context_carries_task_across_turns(self):
        service = make_service()
        service.recommend("I need a writing tool", retrieve_k=2, final_k=2, conversation_id="c1")
        stored = service.conversations.get("c1")
        roles = [turn["role"] for turn in stored]
        self.assertIn("user", roles)
        self.assertIn("assistant", roles)

    def test_focus_latest_intent_follows_pivot(self):
        focused = focus_latest_intent(
            "I need a tool for blog posts. Actually I only need a good AI coding tool"
        )
        self.assertNotIn("blog", focused.lower())
        self.assertIn("coding", focused.lower())
        # No pivot marker leaves the message unchanged.
        self.assertEqual(focus_latest_intent("I need a writing tool"), "I need a writing tool")

    def test_pivot_does_not_inherit_previous_topic(self):
        service = make_service()
        # Previous turn establishes a writing topic in the conversation memory.
        service.recommend("I need a tool for writing blog posts", retrieve_k=2, final_k=2, conversation_id="c2")
        # The user pivots to coding; the goal must not stay on blog/writing.
        response = service.recommend(
            "Actually I only need a good AI coding tool",
            retrieve_k=2,
            final_k=2,
            conversation_id="c2",
        )
        best_for = " ".join(hit.get("best_for", "") for hit in response["hits"]).lower()
        self.assertNotIn("blog", best_for)

    def test_has_explicit_task_distinguishes_tasks_from_filters(self):
        self.assertTrue(has_explicit_task("I only need a good AI coding tool"))
        self.assertTrue(has_explicit_task("I need a writing tool"))
        self.assertFalse(has_explicit_task("free only"))
        self.assertFalse(has_explicit_task("show me cheaper options"))

    def test_explicit_new_task_is_classified_as_new_intent(self):
        service = make_service()
        intent = service.detect_intent(
            "Actually I only need a good AI coding tool",
            "I need a tool for writing blog posts",
        )
        self.assertEqual(intent["intent"], "new")

    def test_refinement_without_task_still_refines(self):
        service = make_service()
        intent = service.detect_intent("free only", "I need a writing tool")
        self.assertEqual(intent["intent"], "refine")


if __name__ == "__main__":
    unittest.main()
