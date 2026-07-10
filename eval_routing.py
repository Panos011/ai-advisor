"""Routing eval harness for the AI-advisor chat planner.

Motivation
----------
Chat routing (does a message become a new search, a refinement, an
explanation of the current cards, an alternative, or plain chat?) is decided
by a large layer of deterministic regex gates in ``chat()`` plus the OpenAI
planner. That layer is where user-facing routing bugs keep appearing, and it
had no offline regression net: every fix was a blind regex tweak.

This module turns the known routing contract into a replayable golden set:

* ``routing_golden.jsonl`` — one case per line: a user message, optional prior
  context (a seeded shortlist and/or a prior task), and the expected response
  ``action`` the frontend should receive.
* ``run_case`` — seed a ``RecommendationService`` with that context and return
  the actual action ``chat()`` produced.
* Offline mode uses a deterministic stub client, so the suite exercises the
  real gate logic without an OpenAI key (the gates decide most of these cases
  before the planner is even consulted).
* ``--live`` swaps in a real OpenAI client to measure how the *planner* routes
  the same cases (this is the number to watch when tuning the planner prompt;
  it is also where ``chat_decision_invalid`` shows up).

Usage
-----
    python eval_routing.py            # offline, deterministic
    python eval_routing.py --live     # requires OPENAI_API_KEY
    python eval_routing.py --verbose  # print every case, not just failures
"""

from __future__ import annotations

import argparse
import json
import os
from types import SimpleNamespace
from typing import Any

import numpy as np

import api

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "routing_golden.jsonl")


# --- Catalog used for offline replay -----------------------------------------
# A small, deterministic catalog whose rows cover the domains the golden cases
# touch (writing, coding, image, notes). Names double as the shortlist tools.
_CATALOG: list[dict[str, Any]] = [
    {
        "Name": "Writerly",
        "Categories": "writing | copywriting | marketing",
        "Price": "Free tier available. Pro $12/month.",
        "Description": "AI writing assistant for blog posts, marketing copy, and SEO content.",
        "Features": "Drafts blog posts, rewrites copy, and generates SEO outlines.",
        "Use_cases": "Blog writing, marketing copy",
        "Pros": "Fast drafting; simple editor.",
    },
    {
        "Name": "Draftly",
        "Categories": "writing | content",
        "Price": "Freemium. Team plan $20/month.",
        "Description": "Long-form content generator for articles and newsletters.",
        "Features": "Outlines, long-form drafts, tone control.",
        "Use_cases": "Article and newsletter writing",
        "Pros": "Good long-form structure.",
    },
    {
        "Name": "CodeMate",
        "Categories": "developer tools | coding | code assistant",
        "Price": "Free tier available.",
        "Description": "AI coding assistant for code review, debugging, and pull requests.",
        "Features": "Code completion, review, and debugging across repositories.",
        "Use_cases": "Software engineering",
        "Pros": "Strong code review.",
    },
    {
        "Name": "ImageBox",
        "Categories": "image generator | design",
        "Price": "Paid. $15/month.",
        "Description": "Text-to-image generator for creative drafts and concept art.",
        "Features": "Text to image, style presets.",
        "Use_cases": "Image generation",
        "Pros": "Fast, varied styles.",
    },
    {
        "Name": "NoteScribe",
        "Categories": "notes | transcription | productivity",
        "Price": "Freemium.",
        "Description": "Meeting note taker and transcription assistant.",
        "Features": "Live transcription, summaries, action items.",
        "Use_cases": "Meeting notes and transcription",
        "Pros": "Accurate transcription.",
    },
]

_NAME_TO_ID = {row["Name"].lower(): i for i, row in enumerate(_CATALOG)}


class _StubEmbeddings:
    """Return a stable unit vector so FAISS-style paths run deterministically."""

    def create(self, **kwargs: Any) -> Any:
        inputs = kwargs.get("input") or [""]
        n = len(inputs) if isinstance(inputs, list) else 1
        return SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, 0.0]) for _ in range(n)])


class _StubChatCompletions:
    """Deterministic planner + ranker.

    The planner returns ``{}`` so ``chat()`` falls through to its own gate and
    fallback routing — which is exactly the deterministic contract we want to
    lock down. The ranker returns the first candidate so recommend paths yield
    a hit.
    """

    def __init__(self) -> None:
        self.calls = 0

    def create(self, **kwargs: Any) -> Any:
        self.calls += 1
        system = (kwargs.get("messages") or [{}])[0].get("content", "")
        if "numbered list of candidate tools" in system or "AI tool recommender" in system:
            payload = {
                "selected": [
                    {"id": 0, "reason": "Matches the requested task.", "tradeoff": "Check limits.", "best_for": "The task"}
                ]
            }
        elif "conversation brain" in system:
            payload = {}
        else:
            payload = {"message": "Sure — tell me the task and any constraints."}
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))])


class _StubClient:
    def __init__(self) -> None:
        self.embeddings = _StubEmbeddings()
        self.chat = SimpleNamespace(completions=_StubChatCompletions())


def _build_index(vectors: np.ndarray) -> Any:
    class _Index:
        d = 2

        def __init__(self, ntotal: int) -> None:
            self.ntotal = ntotal

        def search(self, _vec: np.ndarray, k: int):
            count = min(k, self.ntotal)
            scores = np.array([[0.95 - 0.02 * i for i in range(count)]], dtype="float32")
            ids = np.array([list(range(count))], dtype="int64")
            return scores, ids

    return _Index(len(vectors))


def build_service(live: bool = False) -> "api.RecommendationService":
    vectors = np.eye(len(_CATALOG), 2, dtype="float32")
    for i in range(len(vectors)):
        vectors[i, 0] = 1.0
    store = api.ToolStore(index=_build_index(vectors), meta=list(_CATALOG), vectors=vectors)
    settings = api.get_settings()
    if live:
        if not settings.openai_api_key:
            raise SystemExit("--live requires OPENAI_API_KEY in the environment.")
        from openai import OpenAI

        client: Any = OpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.openai_timeout,
            max_retries=settings.openai_max_retries,
        )
    else:
        client = _StubClient()
    return api.RecommendationService(store, client, settings, api.RuntimeMetrics())


def _hit_for(name: str) -> dict[str, Any]:
    idx = _NAME_TO_ID[name.lower()]
    return {"score": 0.9 - 0.01 * idx, "meta": dict(_CATALOG[idx])}


def _seed_context(svc: "api.RecommendationService", cid: str, case: dict[str, Any]) -> list[dict[str, Any]]:
    """Seed a shortlist + prior task so follow-up cases have real context."""
    shortlist_names = case.get("shortlist") or []
    shortlist = [_hit_for(name) for name in shortlist_names]
    if shortlist:
        svc.shortlists[cid] = shortlist
        svc.shortlist_pointers[cid] = 0
        svc._record_shown(cid, shortlist)
    prior_task = case.get("prior_task")
    if prior_task:
        svc.conversations.append(cid, "user", prior_task)
        svc.conversations.append(cid, "assistant", "Here are some options.")
    return shortlist


def run_case(svc: "api.RecommendationService", case: dict[str, Any], index: int) -> dict[str, Any]:
    """Run one golden case and return {action, hits, message}."""
    cid = f"eval-{index}"
    shortlist = _seed_context(svc, cid, case)
    visible_tools = shortlist if case.get("send_visible", True) else None
    result = svc.chat(
        case["q"],
        retrieve_k=case.get("retrieve_k", 20),
        final_k=case.get("final_k", 3),
        conversation_id=cid,
        visible_tools=visible_tools,
        user_context=case.get("user_context"),
        shown_tools=case.get("shown_tools"),
    )
    return result


def load_golden() -> list[dict[str, Any]]:
    cases = []
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("//"):
                cases.append(json.loads(line))
    return cases


def evaluate(live: bool = False, verbose: bool = False) -> tuple[int, int, int, list[dict[str, Any]]]:
    """Return (passed, total_run, skipped, failures).

    Cases flagged ``requires_planner`` genuinely depend on the LLM planner's
    judgement (e.g. a conversational greeting with extra words that the
    deterministic gates do not hard-match). They are skipped offline — where the
    stub planner returns no decision — and only asserted in ``--live`` mode.
    """
    cases = load_golden()
    failures = []
    passed = 0
    skipped = 0
    for i, case in enumerate(cases):
        if case.get("requires_planner") and not live:
            skipped += 1
            if verbose:
                print(f"SKIP [{case.get('category','')}] {case['q']!r} (requires live planner)")
            continue
        # A fresh service per case keeps conversation state isolated.
        svc = build_service(live=live)
        result = run_case(svc, case, i)
        action = result.get("action")
        expected = case["expected_action"]
        ok = action in expected if isinstance(expected, list) else action == expected
        # Optional anti-regression guard: the returned cards must not include
        # tools the user has already been shown (the "alternatives re-show
        # displayed tools" class of bug).
        forbidden = {name.lower() for name in case.get("forbid_hit_names", [])}
        if forbidden:
            returned = {
                str((hit.get("meta") or {}).get("Name", "")).strip().lower()
                for hit in result.get("hits", [])
            }
            leaked = forbidden & returned
            if leaked:
                ok = False
                action = f"{action} (leaked already-shown: {sorted(leaked)})"
        if ok:
            passed += 1
            if verbose:
                print(f"PASS [{case.get('category','')}] {case['q']!r} -> {action}")
        else:
            record = {"case": case, "got": action, "expected": expected, "message": result.get("message", "")}
            failures.append(record)
            print(f"FAIL [{case.get('category','')}] {case['q']!r}\n     expected {expected}, got {action!r}")
    total_run = passed + len(failures)
    return passed, total_run, skipped, failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay routing golden cases against chat().")
    parser.add_argument("--live", action="store_true", help="Use a real OpenAI client (needs OPENAI_API_KEY).")
    parser.add_argument("--verbose", action="store_true", help="Print every case, not only failures.")
    args = parser.parse_args()

    passed, total, skipped, failures = evaluate(live=args.live, verbose=args.verbose)
    mode = "LIVE planner" if args.live else "offline (deterministic gates)"
    pct = (100 * passed / total) if total else 100.0
    skip_note = f", {skipped} skipped (planner-dependent)" if skipped else ""
    print(f"\nRouting eval [{mode}]: {passed}/{total} passed ({pct:.0f}%){skip_note}.")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
