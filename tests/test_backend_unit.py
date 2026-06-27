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
    alternative_requests_new_search,
    build_retrieval_query,
    clean_assistant_message,
    clean_best_for,
    focus_latest_intent,
    has_explicit_task,
    is_coding_query,
    is_free_tool,
    is_pick_best_query,
    is_referential_pick,
    local_reason,
    merge_history_messages,
    normalize_mode,
    off_topic_for_query,
    recommendation_message,
    request_goal,
    sanitize_reason,
)
from backend.schemas import ChatRequest, ChatResponse, RecommendRequest, SearchRequest
from backend.settings import Settings


class FakeIndex:
    d = 2
    ntotal = 2

    def search(self, _vec, k):
        scores = np.array([[0.95, 0.8]], dtype="float32")
        ids = np.array([[0, 1]], dtype="int64")
        return scores[:, :k], ids[:, :k]


class WriterOnlyIndex:
    d = 2
    ntotal = 2

    def search(self, _vec, k):
        scores = np.array([[0.95]], dtype="float32")
        ids = np.array([[0]], dtype="int64")
        return scores[:, :k], ids[:, :k]


class FirstOnlyThreeIndex:
    d = 2
    ntotal = 3

    def search(self, _vec, k):
        scores = np.array([[0.95]], dtype="float32")
        ids = np.array([[0]], dtype="int64")
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


class DecisionChatCompletions:
    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.calls = 0
        self.rank_calls = 0

    def create(self, **kwargs):
        self.calls += 1
        system = (kwargs.get("messages") or [{}])[0].get("content", "")
        if "conversation brain" in system:
            content = json.dumps(self.decisions.pop(0) if self.decisions else {"action": "chat_only", "message": "Okay."})
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

        self.rank_calls += 1
        content = json.dumps({
            "selected": [
                {
                    "id": 0,
                    "reason": "Writerly is a strong writing assistant for blog posts and SEO content.",
                    "best_for": "Writing blog posts",
                    "tradeoff": "Check usage limits on the free plan.",
                }
            ]
        })
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class DecisionClient:
    def __init__(self, decisions):
        self.embeddings = FakeEmbeddings()
        self.chat = SimpleNamespace(completions=DecisionChatCompletions(decisions))


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


def make_dev_service(client=None, index=None):
    meta = [
        {
            "Name": "Writerly",
            "Categories": "writing generators | copywriting | marketing",
            "Price": "Free tier",
            "Description": "Writing assistant for blog posts and marketing copy.",
            "Features": "Drafts blog posts and rewrites content.",
            "Pros": "Useful for writing content quickly.",
            "Use_cases": "Blog writing",
        },
        {
            "Name": "CodeMate",
            "Categories": "developer tools | coding | code assistant",
            "Price": "Free tier",
            "Description": "AI software engineering assistant for code review, debugging, pull requests, and repositories.",
            "Features": "Helps software engineers write, review, and debug code.",
            "Pros": "Useful for developer workflows.",
            "Use_cases": "Software engineering",
        },
    ]
    store = ToolStore(
        index=index or FakeIndex(),
        meta=meta,
        vectors=np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32"),
    )
    settings = Settings(cache_ttl_seconds=60, cache_max_entries=8)
    return RecommendationService(store, client or FakeClient(embedding_failure=True), settings, RuntimeMetrics())


def make_task_switch_service(client=None):
    meta = [
        {
            "Name": "Writerly",
            "Categories": "writing generators | copywriting | marketing",
            "Price": "Free tier",
            "Description": "Writing assistant for blog posts and marketing copy.",
            "Features": "Drafts blog posts and rewrites content.",
            "Pros": "Useful for writing content quickly.",
            "Use_cases": "Blog writing",
        },
        {
            "Name": "CodeMate",
            "Categories": "developer tools | coding | code assistant",
            "Price": "Free tier",
            "Description": "AI coding assistant for software engineers, debugging, code review, pull requests, and repositories.",
            "Features": "Helps developers write, review, and debug code.",
            "Pros": "Useful for software engineering workflows.",
            "Use_cases": "Software engineering",
        },
        {
            "Name": "MusicBox",
            "Categories": "music | audio | voice generator",
            "Price": "Free tier",
            "Description": "AI music generation tool for creating songs, beats, melodies, and audio ideas.",
            "Features": "Generates music, sound, beats, and singing voice drafts.",
            "Pros": "Useful for music production.",
            "Use_cases": "Music creation",
        },
    ]
    store = ToolStore(
        index=FirstOnlyThreeIndex(),
        meta=meta,
        vectors=np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype="float32"),
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
        client = FakeClient(embedding_failure=True)
        service = make_service(client=client)
        response = service.recommend("I need a free writing tool", retrieve_k=2, final_k=1)
        self.assertEqual(len(response["hits"]), 1)
        self.assertEqual(response["hits"][0]["meta"]["Name"], "Writerly")
        self.assertIn("writing", response["hits"][0]["why"].lower())
        self.assertNotIn("...", response["hits"][0]["why"])
        self.assertEqual(client.chat.completions.calls, 0)

    def test_embedding_fallback_stores_shortlist_for_followups(self):
        service = make_service(client=FakeClient(embedding_failure=True))
        conversation_id = "fallback-followup-thread"
        first = service.recommend(
            "I need a writing tool",
            retrieve_k=2,
            final_k=2,
            conversation_id=conversation_id,
        )
        second = service.recommend(
            "Why?",
            retrieve_k=2,
            final_k=2,
            conversation_id=conversation_id,
        )

        self.assertEqual(first["hits"][0]["meta"]["Name"], "Writerly")
        self.assertEqual(second["hits"][0]["meta"]["Name"], "Writerly")
        self.assertIn("I would pick", second["message"])
        self.assertNotIn("need the previous results", second["message"])

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

    def test_cached_recommendation_still_stores_conversation_shortlist(self):
        service = make_service()
        service.recommend("I need a writing tool", retrieve_k=2, final_k=2)

        cached = service.recommend(
            "I need a writing tool",
            retrieve_k=2,
            final_k=2,
            conversation_id="conv-cache",
        )
        self.assertEqual(len(cached["hits"]), 2)
        self.assertEqual(len(service.shortlists["conv-cache"]), 2)

        explained = service.recommend(
            "Why was Writerly chosen?",
            retrieve_k=2,
            final_k=2,
            conversation_id="conv-cache",
        )
        self.assertEqual(explained["hits"][0]["meta"]["Name"], "Writerly")
        self.assertIn("Writerly", explained["message"])

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

    def test_pick_best_followup_is_not_treated_as_new_search(self):
        service = make_service()
        prompt = "Nice which one do you think you is the best out of all them ?"

        self.assertTrue(is_pick_best_query(prompt))
        self.assertTrue(is_referential_pick(prompt))

        intent = service.detect_intent(prompt, "I need a voice tool")
        self.assertEqual(intent["intent"], "explain")

        clarify = service.clarify(prompt)
        self.assertEqual(clarify["action"], "explain")

        response = service.recommend(
            prompt,
            retrieve_k=2,
            final_k=2,
            history=[{"role": "user", "content": "I need a voice tool"}],
        )
        self.assertEqual(response["hits"], [])
        self.assertIn("current shortlist", response["message"])

    def test_pick_best_followup_answers_from_stored_shortlist(self):
        service = make_service()
        conversation_id = "pick-best-thread"
        first = service.recommend(
            "I need a writing tool",
            retrieve_k=2,
            final_k=2,
            conversation_id=conversation_id,
        )
        second = service.recommend(
            "Nice which one do you think you is the best out of all them ?",
            retrieve_k=2,
            final_k=2,
            conversation_id=conversation_id,
        )

        self.assertEqual(second["hits"][0]["meta"]["Name"], first["hits"][0]["meta"]["Name"])
        self.assertIn("I would pick", second["message"])
        self.assertNotIn("Which one", second["message"])

    def test_plain_why_explains_current_shortlist(self):
        service = make_service()
        conversation_id = "why-thread"
        first = service.recommend(
            "I need a writing tool",
            retrieve_k=2,
            final_k=2,
            conversation_id=conversation_id,
        )
        second = service.recommend(
            "Why?",
            retrieve_k=2,
            final_k=2,
            conversation_id=conversation_id,
        )

        self.assertEqual(len(second["hits"]), 1)
        self.assertEqual(second["hits"][0]["meta"]["Name"], first["hits"][0]["meta"]["Name"])
        self.assertIn("I would pick", second["message"])
        self.assertNotIn("Consultant view", second["message"])

    def test_why_these_tools_explains_the_shortlist(self):
        service = make_service()
        conversation_id = "why-these-thread"
        service.recommend(
            "I need a writing tool",
            retrieve_k=2,
            final_k=2,
            conversation_id=conversation_id,
        )
        response = service.recommend(
            "Why these tools?",
            retrieve_k=2,
            final_k=2,
            conversation_id=conversation_id,
        )

        self.assertGreaterEqual(len(response["hits"]), 2)
        self.assertIn("I chose these", response["message"])
        self.assertIn("useful alternatives", response["message"])
        self.assertNotIn("Consultant view", response["message"])

    def test_alternative_followup_uses_alternative_wording(self):
        service = make_service()
        conversation_id = "alternative-thread"
        service.recommend(
            "I need a writing tool",
            retrieve_k=2,
            final_k=2,
            conversation_id=conversation_id,
        )
        response = service.recommend(
            "Show me another one",
            retrieve_k=2,
            final_k=2,
            conversation_id=conversation_id,
        )

        self.assertEqual(response["hits"][0]["meta"]["Name"], "ImageBox")
        self.assertIn("Another option is", response["message"])
        self.assertNotIn("My top pick", response["message"])

    def test_anything_else_followup_returns_next_option(self):
        service = make_service()
        conversation_id = "anything-else-thread"
        service.recommend(
            "I need a writing tool",
            retrieve_k=2,
            final_k=2,
            conversation_id=conversation_id,
        )
        response = service.recommend(
            "Anything else?",
            retrieve_k=2,
            final_k=2,
            conversation_id=conversation_id,
        )

        self.assertEqual(response["hits"][0]["meta"]["Name"], "ImageBox")
        self.assertIn("Another option is", response["message"])

    def test_chat_request_schema_accepts_visible_tools(self):
        request = ChatRequest(
            q="why these?",
            visible_tools=[
                {
                    "score": 0.9,
                    "meta": {"Name": "Writerly", "Categories": "writing", "Price": "Free tier"},
                    "why": "Good for blog posts.",
                }
            ],
        )
        self.assertEqual(request.visible_tools[0].meta["Name"], "Writerly")

    def test_chat_response_contract_documents_gpt_wrapper_pipeline(self):
        response = ChatResponse(
            action="recommend",
            message="Start with Writerly.",
            hits=[],
            contract={},
        )

        self.assertEqual(response.contract.style, "gpt_wrapper")
        self.assertIn("FAISS", response.contract.retrieval)
        self.assertIn("MMR", response.contract.diversification)
        self.assertIn("RAG", response.contract.generation)
        self.assertIn("meta.Name", response.contract.tool_card_fields)

    def test_chat_model_decision_understands_messy_pick_best(self):
        service = make_service(client=DecisionClient([
            {"action": "pick_best"},
        ]))
        conversation_id = "chat-pick"
        service.recommend(
            "Find an AI writing tool for blog posts",
            retrieve_k=2,
            final_k=2,
            conversation_id=conversation_id,
        )
        response = service.chat(
            "nice but which would u actually use?",
            retrieve_k=2,
            final_k=2,
            conversation_id=conversation_id,
        )

        self.assertEqual(response["action"], "pick_best")
        self.assertEqual(response["hits"][0]["meta"]["Name"], "Writerly")
        self.assertIn("I would pick", response["message"])
        self.assertGreater(service.client.chat.completions.calls, 0)

    def test_chat_model_decision_can_refine_to_free_tools(self):
        service = make_service(client=DecisionClient([
            {
                "action": "refine",
                "refined_query": "I need a free writing tool",
                "filters": {"budget": "free"},
            },
        ]))
        conversation_id = "chat-free"
        service.recommend(
            "Find an AI writing tool",
            retrieve_k=2,
            final_k=2,
            conversation_id=conversation_id,
        )
        response = service.chat(
            "nah no paid stuff at all pls",
            retrieve_k=2,
            final_k=2,
            conversation_id=conversation_id,
        )

        self.assertEqual(response["action"], "refine")
        self.assertEqual(response["refined_query"], "I need a free writing tool")
        self.assertTrue(all(is_free_tool(hit["meta"]) for hit in response["hits"]))

    def test_chat_model_decision_can_chat_without_showing_tools(self):
        service = make_service(client=DecisionClient([
            {
                "action": "chat_only",
                "message": "I can help you choose, compare, and filter AI tools.",
            },
        ]))
        response = service.chat(
            "what can you do in this screen?",
            retrieve_k=2,
            final_k=2,
            conversation_id="chat-only",
        )

        self.assertEqual(response["action"], "chat_only")
        self.assertEqual(response["hits"], [])
        self.assertIn("choose", response["message"])

    def test_chat_uses_visible_tools_when_server_memory_is_missing(self):
        service = make_service(client=DecisionClient([
            {"action": "explain_shortlist"},
        ]))
        visible = [
            {
                "score": 0.9,
                "meta": service.store.meta[0],
                "why": "Writerly is useful for blog writing.",
            },
            {
                "score": 0.7,
                "meta": service.store.meta[1],
                "why": "ImageBox is a visual tool.",
            },
        ]
        response = service.chat(
            "why these?",
            retrieve_k=2,
            final_k=2,
            conversation_id="visible-only",
            visible_tools=visible,
        )

        self.assertEqual(response["action"], "explain")
        self.assertGreaterEqual(len(response["hits"]), 2)
        self.assertIn("I chose these", response["message"])

    def test_chat_fallback_free_only_after_greeting_asks_for_task(self):
        service = make_service(client=FakeClient(embedding_failure=True))
        conversation_id = "fallback-free-after-greeting"
        service.chat("hello", retrieve_k=2, final_k=2, conversation_id=conversation_id)
        response = service.chat("only free tools", retrieve_k=2, final_k=2, conversation_id=conversation_id)

        self.assertEqual(response["action"], "clarify")
        self.assertEqual(response["hits"], [])
        self.assertIn("What task", response["message"])

    def test_chat_fallback_handles_pivot_without_old_topic(self):
        service = make_service(client=FakeClient(embedding_failure=True))
        conversation_id = "fallback-pivot"
        service.chat(
            "find a writing tool for blog posts",
            retrieve_k=2,
            final_k=2,
            conversation_id=conversation_id,
        )
        response = service.chat(
            "nah I need something for private meeting notes instead",
            retrieve_k=2,
            final_k=2,
            conversation_id=conversation_id,
        )

        self.assertEqual(response["action"], "recommend")
        refined = (response.get("refined_query") or "").lower()
        self.assertIn("meeting", refined)
        self.assertNotIn("blog", refined)

    def test_chat_fallback_understands_why_tho(self):
        service = make_service(client=FakeClient(embedding_failure=True))
        conversation_id = "fallback-why-tho"
        service.chat(
            "find a writing tool for blog posts",
            retrieve_k=2,
            final_k=2,
            conversation_id=conversation_id,
        )
        response = service.chat("why tho", retrieve_k=2, final_k=2, conversation_id=conversation_id)

        self.assertEqual(response["action"], "explain")
        self.assertIn("I would pick", response["message"])
        self.assertNotIn("Start with", response["message"])

    def test_chat_guardrail_prevents_anything_else_from_restarting_search(self):
        service = make_service()
        conversation_id = "chat-anything-else"
        service.chat(
            "find a writing tool for blog posts",
            retrieve_k=2,
            final_k=2,
            conversation_id=conversation_id,
        )
        response = service.chat(
            "Ok anything else that you are recommending?",
            retrieve_k=2,
            final_k=2,
            conversation_id=conversation_id,
        )

        self.assertEqual(response["action"], "show_alternative")
        self.assertEqual(response["hits"][0]["meta"]["Name"], "ImageBox")
        self.assertIn("Another option is", response["message"])

    def test_chat_new_coding_alternative_runs_fresh_search(self):
        service = make_dev_service()
        conversation_id = "chat-better-coding"
        service.chat(
            "find a writing tool for blog posts",
            retrieve_k=2,
            final_k=1,
            conversation_id=conversation_id,
        )

        self.assertTrue(alternative_requests_new_search(
            "I want another coding tool that is better than the one you gave me"
        ))
        response = service.chat(
            "I want another coding tool that is better than the one you gave me",
            retrieve_k=2,
            final_k=1,
            conversation_id=conversation_id,
        )

        self.assertEqual(response["action"], "recommend")
        self.assertEqual(response["hits"][0]["meta"]["Name"], "CodeMate")
        self.assertNotIn("another distinct option", response["message"].lower())

    def test_chat_planner_tool_routes_without_action_label(self):
        service = make_dev_service(client=DecisionClient([
            {
                "tool": "search_tools",
                "refined_query": "software engineer coding tool",
            },
        ]), index=WriterOnlyIndex())

        response = service.chat(
            "What about a more specific one like a Software Engineer?",
            retrieve_k=1,
            final_k=1,
            conversation_id="planner-tool",
        )

        self.assertEqual(response["action"], "recommend")
        self.assertEqual(response["hits"][0]["meta"]["Name"], "CodeMate")

    def test_chat_specific_task_followup_overrides_visible_shortlist(self):
        service = make_task_switch_service()
        conversation_id = "task-switch-coding"
        service.chat(
            "Find an AI writing tool for blog posts",
            retrieve_k=1,
            final_k=1,
            conversation_id=conversation_id,
        )

        self.assertTrue(has_explicit_task("What about a coding tool"))
        response = service.chat(
            "What about a coding tool",
            retrieve_k=1,
            final_k=1,
            conversation_id=conversation_id,
        )

        self.assertEqual(response["action"], "recommend")
        self.assertEqual(response["hits"][0]["meta"]["Name"], "CodeMate")
        self.assertNotEqual(response["hits"][0]["meta"]["Name"], "Writerly")

    def test_chat_music_request_overrides_coding_shortlist(self):
        service = make_task_switch_service()
        conversation_id = "task-switch-music"
        service.chat(
            "Actually I want a coding tool",
            retrieve_k=1,
            final_k=1,
            conversation_id=conversation_id,
        )

        self.assertTrue(has_explicit_task("Give me the best tools for music"))
        response = service.chat(
            "Give me the best tools for music",
            retrieve_k=1,
            final_k=1,
            conversation_id=conversation_id,
        )

        self.assertEqual(response["action"], "recommend")
        self.assertEqual(response["hits"][0]["meta"]["Name"], "MusicBox")
        self.assertNotEqual(response["hits"][0]["meta"]["Name"], "CodeMate")

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

    def test_small_talk_and_capability_prompts_do_not_search(self):
        service = make_service()

        thanks = service.clarify("thanks")
        self.assertEqual(thanks["action"], "clarify")
        self.assertIn("welcome", thanks["question"].lower())

        hello = service.recommend("hello", retrieve_k=2, final_k=2)
        self.assertEqual(hello["hits"], [])
        self.assertIn("tell me what you need", hello["message"].lower())

        capabilities = service.recommend("what can you do?", retrieve_k=2, final_k=2)
        self.assertEqual(capabilities["hits"], [])
        self.assertIn("recommend AI tools", capabilities["message"])

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

        leaked_alternatives = sanitize_reason(
            "Return alternatives that are meaningfully different from the first result.",
            name="Hyprnote",
            query=query,
        )
        self.assertNotIn("Return alternatives", leaked_alternatives)

        fallback = local_reason(query, {
            "Name": "Hyprnote",
            "Description": "Private local-first AI notetaker for meetings.",
        })
        self.assertIn("private meeting notes", fallback)
        self.assertIn("Hyprnote is well suited", fallback)
        self.assertNotIn("Return recommendations", fallback)
        self.assertNotIn("...", fallback)

    def test_clean_assistant_message_removes_old_template_language(self):
        message = clean_assistant_message(
            "Consultant view: these are alternatives worth comparing, not identical picks. "
            "Start with Copyai. It appears to be the best first test from the current catalogue data."
        )
        self.assertNotIn("Consultant view", message)
        self.assertNotIn("alternatives worth comparing", message)
        self.assertNotIn("current catalogue data", message)
        self.assertEqual(message, "Start with Copyai.")

    def test_clean_assistant_message_removes_screenshot_template_noise(self):
        message = clean_assistant_message(
            "Consultant view: I would start with Copyai. It appears to be the best first test "
            "from the current catalogue data. Best for: Copyai is the best choice for I would "
            "pick Copyai for writing blog posts. Copyai is well suited for writing blog posts "
            "because it combines blog or article drafting with SEO-focused writing support. "
            "It also has a free tier or trial, so you can test it without paying upfront. "
            "because it matches the task, price, and feature signals best.. Use the provider "
            "site to verify pricing and limits before committing."
        )

        self.assertNotIn("Consultant view", message)
        self.assertNotIn("current catalogue data", message)
        self.assertNotIn("Best for:", message)
        self.assertNotIn("because it matches the task", message)

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
        self.assertEqual(request_goal("I need a software engineer tool"), "coding and development")
        self.assertEqual(request_goal("Give me the best tools for music"), "creating music and audio")

    def test_software_engineer_query_blocks_writing_categories(self):
        query = "What about a more specific one like a Software Engineer?"

        self.assertTrue(is_coding_query(query))
        self.assertTrue(off_topic_for_query(query, "copywriting | marketing | writing generators"))
        self.assertFalse(off_topic_for_query(query, "developer tools | coding | code assistant"))

        service = make_dev_service()
        response = service.recommend(query, retrieve_k=2, final_k=2)
        names = [hit["meta"]["Name"] for hit in response["hits"]]
        self.assertEqual(names, ["CodeMate"])
        self.assertNotIn("Writerly", names)

    def test_hybrid_retrieval_adds_keyword_dev_candidate_when_faiss_misses(self):
        service = make_dev_service(client=FakeClient(), index=WriterOnlyIndex())
        response = service.recommend(
            "software engineer code review tool",
            retrieve_k=1,
            final_k=1,
        )

        self.assertEqual(response["hits"][0]["meta"]["Name"], "CodeMate")
        self.assertNotEqual(response["hits"][0]["meta"]["Name"], "Writerly")

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

    def test_pick_best_question_is_classified_as_explain(self):
        # Referential pick-best questions should be answered from the visible shortlist.
        service = make_service()
        intent = service.detect_intent(
            "Which one is the best out of all these?",
            "Find an AI writing tool for blog posts",
        )
        self.assertEqual(intent["intent"], "explain")

    def test_pick_best_question_never_clarifies(self):
        service = make_service()
        decision = service.clarify(
            "Find an AI writing tool for blog posts. Which one is the best out of all these?"
        )
        self.assertEqual(decision["action"], "search")

    def test_pick_best_returns_single_best_from_underlying_task(self):
        # Mirrors production: frontend merges lastQuery + pick-best question.
        service = make_service()
        response = service.recommend(
            "Find an AI writing tool for blog posts. Which one is the best out of all these?",
            retrieve_k=2,
            final_k=5,
            mode="balanced",
        )
        # Single winner, drawn from the writing task, not the literal question.
        self.assertEqual(len(response["hits"]), 1)
        self.assertEqual(response["hits"][0]["meta"]["Name"], "Writerly")
        best_for = (response["hits"][0].get("best_for") or "").lower()
        self.assertNotIn("which", best_for)
        self.assertNotIn("best out of", best_for)

    def test_self_contained_best_query_is_a_normal_search(self):
        # "what's the best writing tool" with no prior shortlist must still search.
        service = make_service()
        response = service.recommend(
            "what's the best writing tool", retrieve_k=2, final_k=5, mode="balanced"
        )
        self.assertTrue(response["hits"])
        self.assertNotIn("earlier results", (response.get("message") or "").lower())

    def test_pick_best_returns_top_of_stored_shortlist(self):
        # With a conversation_id, the backend remembers the shortlist and picks
        # the #1 hit from it — no re-search, no context merging needed.
        service = make_service()
        first = service.recommend(
            "I need a writing tool", retrieve_k=2, final_k=2,
            conversation_id="conv-stateful",
        )
        self.assertEqual(len(first["hits"]), 2)
        # Follow-up: "which is best?" — should return exactly the #1 tool.
        second = service.recommend(
            "Which one is the best out of all these?",
            retrieve_k=2, final_k=5, mode="balanced",
            conversation_id="conv-stateful",
        )
        self.assertEqual(len(second["hits"]), 1)
        self.assertEqual(
            second["hits"][0]["meta"]["Name"],
            first["hits"][0]["meta"]["Name"],
        )
        self.assertNotIn("Which one", second["message"])

    def test_pick_best_detect_intent_uses_stored_shortlist_as_context(self):
        # Even without last_query, a stored shortlist proves there is context.
        service = make_service()
        service.recommend(
            "I need a writing tool", retrieve_k=2, final_k=2,
            conversation_id="conv-intent",
        )
        intent = service.detect_intent(
            "Which one is the best?", "", conversation_id="conv-intent",
        )
        self.assertEqual(intent["intent"], "explain")

    def test_specific_tool_follow_up_returns_named_tool_from_shortlist(self):
        # "What about X?" should return the named tool from the stored shortlist.
        service = make_service()
        first = service.recommend(
            "I need a writing tool", retrieve_k=2, final_k=2,
            conversation_id="conv-specific",
        )
        self.assertEqual(len(first["hits"]), 2)
        second = service.recommend(
            "What about Writerly?", retrieve_k=2, final_k=5,
            conversation_id="conv-specific",
        )
        self.assertEqual(len(second["hits"]), 1)
        self.assertEqual(second["hits"][0]["meta"]["Name"], "Writerly")
        self.assertIn("can work", second["message"])
        # The original shortlist must remain intact so "another" still works.
        self.assertEqual(len(service.shortlists["conv-specific"]), 2)

    def test_specific_tool_follow_up_does_not_replace_original_task(self):
        service = make_service()
        service.recommend(
            "I need a writing tool for blog posts", retrieve_k=2, final_k=2,
            conversation_id="conv-specific-task",
        )
        service.recommend(
            "What about ImageBox?", retrieve_k=2, final_k=2,
            conversation_id="conv-specific-task",
        )
        followup = service.recommend(
            "show me another one", retrieve_k=2, final_k=2,
            conversation_id="conv-specific-task",
        )
        self.assertEqual(followup["hits"], [])
        self.assertIn("another distinct option", followup["message"])
        self.assertNotIn("generating images", followup["message"].lower())

    def test_criterion_pick_returns_cheapest_from_shortlist(self):
        # "Which one is the cheapest?" picks the cheapest tool from the shortlist.
        service = make_service()
        first = service.recommend(
            "I need a writing tool", retrieve_k=2, final_k=2,
            conversation_id="conv-cheap",
        )
        self.assertEqual(len(first["hits"]), 2)
        second = service.recommend(
            "Which one is the cheapest?", retrieve_k=2, final_k=5,
            conversation_id="conv-cheap",
        )
        self.assertEqual(len(second["hits"]), 1)
        self.assertEqual(second["hits"][0]["meta"]["Name"], "Writerly")
        self.assertEqual(len(service.shortlists["conv-cheap"]), 2)

    def test_alternative_query_returns_next_hit_from_shortlist(self):
        # "Is there any other tool?" returns the next hit from the stored shortlist.
        service = make_service()
        first = service.recommend(
            "I need a writing tool", retrieve_k=2, final_k=2,
            conversation_id="conv-alt",
        )
        self.assertEqual(len(first["hits"]), 2)
        self.assertEqual(first["hits"][0]["meta"]["Name"], "Writerly")
        second = service.recommend(
            "Is there any other tool that is probably better than this?",
            retrieve_k=2, final_k=5,
            conversation_id="conv-alt",
        )
        self.assertEqual(len(second["hits"]), 1)
        self.assertEqual(second["hits"][0]["meta"]["Name"], "ImageBox")
        # Asking again should not give the same one back.
        third = service.recommend(
            "You gave me the same one",
            retrieve_k=2, final_k=5,
            conversation_id="conv-alt",
        )
        self.assertEqual(third["hits"], [])
        self.assertIn("another distinct option", third["message"])

    def test_detect_intent_specific_tool_and_alternative_are_refine(self):
        service = make_service()
        service.recommend(
            "I need a writing tool", retrieve_k=2, final_k=2,
            conversation_id="conv-intent2",
        )
        self.assertEqual(
            service.detect_intent("What about Writerly?", "", conversation_id="conv-intent2")["intent"],
            "refine",
        )
        self.assertEqual(
            service.detect_intent("Is there any other tool?", "", conversation_id="conv-intent2")["intent"],
            "refine",
        )

    def test_explanation_query_with_shortlist_returns_top_hit_explained(self):
        # "Why is that the best tool?" should explain the top hit from the stored shortlist.
        service = make_service()
        first = service.recommend(
            "I need a writing tool", retrieve_k=2, final_k=2,
            conversation_id="conv-explain",
        )
        self.assertEqual(len(first["hits"]), 2)
        self.assertEqual(first["hits"][0]["meta"]["Name"], "Writerly")
        second = service.recommend(
            "Why is that the best tool?", retrieve_k=2, final_k=5,
            conversation_id="conv-explain",
        )
        self.assertEqual(len(second["hits"]), 1)
        self.assertEqual(second["hits"][0]["meta"]["Name"], "Writerly")
        self.assertIn("best choice", second["hits"][0].get("best_for", ""))
        self.assertIn("writing", second["hits"][0].get("why", "").lower())
        self.assertIn("Writerly", second["message"])

    def test_named_tool_explanation_is_detected(self):
        service = make_service()
        service.recommend(
            "I need a writing tool", retrieve_k=2, final_k=2,
            conversation_id="conv-named-explain",
        )
        intent = service.detect_intent(
            "Why was Writerly chosen?", "", conversation_id="conv-named-explain",
        )
        self.assertEqual(intent["intent"], "explain")
        response = service.recommend(
            "Why was Writerly chosen?", retrieve_k=2, final_k=2,
            conversation_id="conv-named-explain",
        )
        self.assertEqual(response["hits"][0]["meta"]["Name"], "Writerly")
        self.assertIn("Writerly", response["message"])

    def test_is_it_free_answers_about_current_shortlist_item(self):
        service = make_service()
        service.recommend(
            "I need a writing tool", retrieve_k=2, final_k=2,
            conversation_id="conv-free-question",
        )
        intent = service.detect_intent(
            "is it free?", "", conversation_id="conv-free-question",
        )
        self.assertEqual(intent["intent"], "refine")
        response = service.recommend(
            "is it free?", retrieve_k=2, final_k=2,
            conversation_id="conv-free-question",
        )
        self.assertEqual(response["hits"][0]["meta"]["Name"], "Writerly")
        self.assertIn("Yes", response["message"])
        self.assertIn("free tier", response["message"])


if __name__ == "__main__":
    unittest.main()
