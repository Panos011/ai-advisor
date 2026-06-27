from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import faiss
except ModuleNotFoundError:  # pragma: no cover - exercised only in missing-dependency test envs
    faiss = None

from backend.cache import TTLCache
from backend.metrics import RuntimeMetrics
from backend.settings import Settings

logger = logging.getLogger(__name__)

MAX_REASON_CHARS = 320
TEXT_FORMAT_VERSION = "display-v8"


INSTRUCTION_LEAK_PATTERNS = (
    r"\bact\s+(?:like|as)\s+(?:a|an)?\s*[^.?!]{0,80}(?:consultant|advisor|expert)[^.?!]*[.?!]?",
    r"\bprioritize tools with[^.?!]*[.?!]?",
    r"\breturn recommendations as[^.?!]*[.?!]?",
    r"\breturn alternatives that[^.?!]*[.?!]?",
    r"\bthese are alternatives worth comparing[^.?!]*[.?!]?",
    r"\bit appears to be the best first test from the current catalogue data[.?!]?",
    r"\bbecause it matches the task,\s*price,\s*and feature signals best[.?!]?",
    r"\bdecision shortlist with[^.?!]*[.?!]?",
    r"\bfit,\s*tradeoffs,\s*and practical next steps:?",
)

FEEDBACK_PATTERNS = (
    r"\bwtf\b",
    r"\bwhat\s+the\s+fuck\b",
    r"\b(?:this|that|it)\s+(?:doesn'?t|does not|isn'?t|is not)\s+(?:work|working|right|correct)\b",
    r"\b(?:wrong|broken|bad|nonsense|useless)\b",
    r"\b(?:you|it)\s+(?:messed|broke)\b",
)

NON_SEARCH_PATTERNS = (
    r"^(?:hi|hello|hey|yo|sup|good\s+(?:morning|afternoon|evening))[\s!.?]*$",
    r"^(?:thanks|thank\s+you|thx|cheers|ok|okay|nice|cool|great|perfect|awesome)[\s!.?]*$",
    r"\bwhat\s+can\s+you\s+do\b",
    r"\bhow\s+do\s+i\s+use\s+this\b",
    r"\bwho\s+are\s+you\b",
)

FREE_FILTER_WORDS = {
    "free", "only", "ones", "one", "option", "options", "tool", "tools",
    "app", "apps", "said", "tier", "trial", "plan", "plans",
}

# Recommendation modes surfaced in the UI (Best fit / One best / Compare).
MODE_BEST_FIT = "best_fit"
MODE_ONE_BEST = "one_best"
MODE_COMPARE = "compare"

CHAT_ACTIONS = {
    "chat_only",
    "clarify",
    "recommend",
    "refine",
    "explain_shortlist",
    "explain_best",
    "pick_best",
    "show_alternative",
    "tool_question",
}

_MODE_ALIASES = {
    "balanced": MODE_BEST_FIT,
    "best_fit": MODE_BEST_FIT,
    "best-fit": MODE_BEST_FIT,
    "bestfit": MODE_BEST_FIT,
    "best fit": MODE_BEST_FIT,
    "shortlist": MODE_BEST_FIT,
    "one_best": MODE_ONE_BEST,
    "one-best": MODE_ONE_BEST,
    "onebest": MODE_ONE_BEST,
    "one best": MODE_ONE_BEST,
    "best": MODE_ONE_BEST,
    "winner": MODE_ONE_BEST,
    "single": MODE_ONE_BEST,
    "compare": MODE_COMPARE,
    "comparison": MODE_COMPARE,
    "side_by_side": MODE_COMPARE,
    "side-by-side": MODE_COMPARE,
}


def normalize_mode(mode: Any) -> str:
    key = " ".join(str(mode or "").split()).lower()
    return _MODE_ALIASES.get(key, MODE_BEST_FIT)


@dataclass
class ToolStore:
    index: Any
    meta: list[dict[str, Any]]
    vectors: np.ndarray | None

    @property
    def ready(self) -> bool:
        return self.index is not None and bool(self.meta)


def load_tool_store(settings: Settings) -> ToolStore:
    if faiss is None:
        raise RuntimeError(
            "FAISS is not installed. Install requirements.txt before loading the production index."
        )

    with open(settings.meta_path, "r", encoding="utf-8") as f:
        meta = [json.loads(line) for line in f]

    index = faiss.read_index(settings.index_path)
    vectors = _load_vectors(settings.vectors_path, index, len(meta))

    if index.ntotal != len(meta):
        raise RuntimeError(
            f"Index item count ({index.ntotal}) does not match metadata rows ({len(meta)})"
        )
    if vectors is not None and len(vectors) != len(meta):
        raise RuntimeError(
            f"Vector row count ({len(vectors)}) does not match metadata rows ({len(meta)})"
        )

    logger.info("Loaded %s tools from %s", len(meta), settings.index_path)
    return ToolStore(index=index, meta=meta, vectors=vectors)


def _load_vectors(path: str, index: Any, expected_rows: int) -> np.ndarray | None:
    if os.path.exists(path):
        vectors = np.load(path).astype("float32")
        normalize_l2(vectors)
        return vectors

    vectors = _extract_vectors_from_flat_index(index, expected_rows)
    if vectors is None:
        logger.warning("No vector matrix found at %s and FAISS vectors could not be extracted", path)
    return vectors


def _extract_vectors_from_flat_index(index: Any, expected_rows: int) -> np.ndarray | None:
    base_index = getattr(index, "index", index)
    try:
        vectors = np.array(
            [base_index.reconstruct(i) for i in range(expected_rows)],
            dtype="float32",
        )
    except Exception as exc:
        logger.warning("Unable to reconstruct vectors for MMR: %s", exc)
        return None
    normalize_l2(vectors)
    return vectors


def normalize_l2(vectors: np.ndarray) -> None:
    if faiss is not None:
        faiss.normalize_L2(vectors)
        return

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    np.divide(vectors, norms, out=vectors, where=norms != 0)


class ConversationStore:
    """In-memory, TTL-bounded store of recent chat turns keyed by conversation id."""

    def __init__(self, max_conversations: int = 500, ttl_seconds: int = 3600, max_turns: int = 12) -> None:
        self._cache: TTLCache[list[dict[str, str]]] = TTLCache(
            max_entries=max_conversations,
            ttl_seconds=ttl_seconds,
        )
        self.max_turns = max_turns

    def get(self, conversation_id: str | None) -> list[dict[str, str]]:
        if not conversation_id:
            return []
        return list(self._cache.get(conversation_id) or [])

    def append(self, conversation_id: str | None, role: str, content: str) -> None:
        if not conversation_id:
            return
        content = normalize_query_text(content)
        if not content:
            return
        turns = list(self._cache.get(conversation_id) or [])
        turns.append({"role": role, "content": content})
        if len(turns) > self.max_turns:
            turns = turns[-self.max_turns:]
        self._cache.set(conversation_id, turns)


def merge_history_messages(
    history: Any,
    stored: list[dict[str, str]] | None = None,
    limit: int = 6,
) -> list[str]:
    """Return de-duplicated recent user messages from client history and server store."""
    messages: list[str] = []
    for source in (stored or [], history or []):
        for item in source:
            role = item.get("role") if isinstance(item, dict) else getattr(item, "role", "user")
            content = item.get("content") if isinstance(item, dict) else getattr(item, "content", "")
            if role != "user":
                continue
            cleaned = normalize_query_text(content)
            if cleaned and cleaned not in messages:
                messages.append(cleaned)
    return messages[-limit:]


def build_retrieval_query(current: str, prior_user_messages: list[str]) -> str:
    """Combine the latest message with recent task context for retrieval/ranking."""
    current_clean = normalize_query_text(current)
    parts: list[str] = []
    for message in prior_user_messages:
        if message and message != current_clean and message not in parts:
            parts.append(message)
    parts.append(current_clean)
    return " ".join(part for part in parts if part).strip()


def normalize_query_text(value: Any) -> str:
    text = " ".join(str(value or "").split())
    text = re.sub(r"\s+([?!.,;:])", r"\1", text)
    return text.strip()


def strip_instruction_text(value: Any) -> str:
    text = normalize_query_text(value)
    for pattern in INSTRUCTION_LEAK_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    return normalize_query_text(text).strip(" -:")


def remove_feedback_text(value: Any) -> str:
    text = strip_instruction_text(value)
    for pattern in FEEDBACK_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    return normalize_query_text(text).strip(" -:")


def is_feedback_only_query(text: str) -> bool:
    cleaned = strip_instruction_text(text)
    if not cleaned:
        return False
    has_feedback_signal = any(
        re.search(pattern, cleaned, flags=re.IGNORECASE)
        for pattern in FEEDBACK_PATTERNS
    )
    if not has_feedback_signal:
        return False

    remaining = remove_feedback_text(cleaned)
    remaining_terms = query_terms(remaining)
    return len(remaining_terms) <= 1


def feedback_clarifying_question() -> str:
    return (
        "That looks like feedback rather than a new tool search. "
        "Do you want me to explain the current tools, change the filters, or start a new search?"
    )


def is_non_search_message(text: str) -> bool:
    normalized = normalize_query_text(text).lower().strip()
    if not normalized:
        return False
    return any(re.search(pattern, normalized) for pattern in NON_SEARCH_PATTERNS)


def non_search_response(text: str) -> str:
    normalized = normalize_query_text(text).lower().strip()
    if re.search(r"^(?:thanks|thank\s+you|thx|cheers)", normalized):
        return "You are welcome. Tell me the next task, or ask me to explain, compare, filter, or pick from the current tools."
    if re.search(r"^(?:hi|hello|hey|yo|sup|good\s+)", normalized):
        return "Hi. Tell me what you need a tool for, including budget, privacy needs, or apps it must connect with."
    return "I can recommend AI tools, explain why a tool was chosen, compare the current shortlist, filter by budget or privacy, and pick the best option."


def requires_free_only(text: str) -> bool:
    normalized = normalize_query_text(text).lower()
    if not normalized:
        return False
    if re.search(r"\b(?:paid\s+only|paid\s+ones|paid\s+tools|paid\s+options|compare\s+paid|free\s+(?:and|or)\s+paid)\b", normalized):
        return False
    return bool(re.search(
        r"\bfree\b|"
        r"\b(?:only|just)\s+(?:the\s+)?free(?:\s+(?:ones|tools|apps|options))?\b|"
        r"\b(?:no|without)\s+paid\b|"
        r"\bdo\s+not\s+show\s+paid\b",
        normalized,
    ))


def non_filter_terms(text: str) -> list[str]:
    return [term for term in query_terms(text) if term not in FREE_FILTER_WORDS]



def is_free_tool(meta: dict[str, Any]) -> bool:
    price = normalize_display_text(meta.get("Price", "")).lower()
    if not price:
        return False
    negative = bool(re.search(
        r"\b(?:no|not|without)\s+(?:a\s+)?free\b|"
        r"\bdoes\s+not\s+offer\s+(?:a\s+)?free\b|"
        r"\bfree\s+(?:tier|plan|trial)\s+(?:is\s+)?(?:not|unavailable)\b",
        price,
    ))
    positive = bool(re.search(
        r"\bfree\b|"
        r"\bno\s+cost\b|"
        r"\bat\s+no\s+cost\b|"
        r"\bwithout\s+paying\b|"
        r"\bno\s+credit\s+card\b|"
        r"\bopen\s+source\b",
        price,
    ))
    return positive and not negative


# === Decision Filter and Enrichment Helpers ===

def filter_value(filters: Any, key: str, default: Any = None) -> Any:
    if filters is None:
        return default
    if isinstance(filters, dict):
        return filters.get(key, default)
    return getattr(filters, key, default)


def metadata_blob(meta: dict[str, Any]) -> str:
    return " ".join(str(value or "") for value in meta.values()).lower()


def candidate_meta(candidate: dict[str, Any], meta_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if "meta" in candidate and isinstance(candidate["meta"], dict):
        return candidate["meta"]
    try:
        return meta_rows[int(candidate["id"])]
    except Exception:
        return {}


def matches_budget_filter(meta: dict[str, Any], budget: str) -> bool:
    price = normalize_display_text(meta.get("Price", "")).lower()
    if budget == "any":
        return True
    if budget == "free":
        return is_free_tool(meta)
    if budget == "freemium":
        return is_free_tool(meta) or "freemium" in price or "free tier" in price
    if budget == "paid":
        return bool(re.search(
            r"[$€£]|\b(paid|pro|premium|subscription|enterprise|per month|per year|monthly|annual)\b",
            price,
        ))
    return True


def apply_decision_filters(
    candidates: list[dict[str, Any]],
    filters: Any,
    meta_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if filters is None:
        return candidates

    budget = filter_value(filters, "budget", "any") or "any"
    privacy = filter_value(filters, "privacy", "standard") or "standard"
    integrations = [str(item).lower() for item in (filter_value(filters, "integrations", []) or [])]
    categories = [str(item).lower() for item in (filter_value(filters, "categories", []) or [])]
    platforms = [str(item).lower() for item in (filter_value(filters, "platforms", []) or [])]
    skill_level = filter_value(filters, "skill_level", None) or filter_value(filters, "skillLevel", "any") or "any"

    filtered: list[dict[str, Any]] = []
    for candidate in candidates:
        meta = candidate_meta(candidate, meta_rows)
        blob = metadata_blob(meta)
        category_text = str(meta.get("Categories", "")).lower()

        if not matches_budget_filter(meta, str(budget)):
            continue

        if privacy == "privacy-first" and not re.search(
            r"\b(privacy|private|secure|security|gdpr|compliance|encryption|data control|soc 2|iso 27001)\b",
            blob,
        ):
            continue

        if privacy == "local-first" and not re.search(
            r"\b(local|on-device|on device|self-hosted|self hosted|private cloud|offline|open source)\b",
            blob,
        ):
            continue

        if integrations and not all(integration in blob for integration in integrations):
            continue

        if categories and not any(category in category_text or category in blob for category in categories):
            continue

        if platforms and not any(platform in blob for platform in platforms):
            continue

        if skill_level == "beginner" and not re.search(
            r"\b(beginner|easy|simple|no-code|no code|template|user-friendly|user friendly)\b",
            blob,
        ):
            continue

        filtered.append(candidate)

    return filtered


def fit_label(score: float) -> str:
    if score >= 0.82:
        return "Strong match"
    if score >= 0.68:
        return "Good match"
    return "Possible match"


def build_tradeoff(meta: dict[str, Any]) -> str:
    cons = normalize_display_text(meta.get("Cons", ""))
    if cons:
        return complete_sentences(cons, 220, max_sentences=1) or cons[:220]

    price = normalize_display_text(meta.get("Price", ""))
    lower = price.lower()
    if "enterprise" in lower:
        return "Pricing may be less transparent and could require contacting sales."
    if "free trial" in lower:
        return "Free access may be limited to a trial, so check the limits before relying on it."
    if "freemium" in lower or is_free_tool(meta):
        return "The free plan may have usage limits, exports limits, or locked advanced features."
    if not price:
        return "Pricing and feature limits should be checked on the official website."
    return "Check the official website because pricing and features can change."


def build_best_for(q: str, meta: dict[str, Any]) -> str:
    goal = request_goal(q)
    categories = category_list(meta, limit=2)
    if categories:
        return f"{goal.capitalize()} using {human_join(categories)} tools"
    return goal.capitalize()


def clean_best_for(value: Any, q: str, meta: dict[str, Any], max_words: int = 14) -> str:
    """Use the model's best_for only when it is a short use case, not a query restatement."""
    text = normalize_display_text(value)
    if not text:
        return build_best_for(q, meta)
    query_clean = strip_instruction_text(q).lower()
    lowered = text.lower()
    if query_clean and (query_clean in lowered or lowered in query_clean):
        return build_best_for(q, meta)
    if len(text.split()) > max_words:
        return build_best_for(q, meta)
    return text


def enrich_hit(hit: dict[str, Any], q: str) -> dict[str, Any]:
    meta = hit.get("meta") or {}
    score = float(hit.get("score", 0.0) or 0.0)
    enriched = dict(hit)
    enriched.setdefault("why", local_reason(q, meta))
    enriched.setdefault("tradeoff", build_tradeoff(meta))
    enriched.setdefault("best_for", build_best_for(q, meta))
    enriched.setdefault("fit_label", fit_label(score))
    return enriched


def query_terms(text: str) -> list[str]:
    normalized = strip_instruction_text(text).lower().strip()
    intent_aliases = {
        "research": "research search summarize market competitor analysis web data insights",
        "create": "create content writing design image video presentation copywriting social media",
        "automate": "automate automation workflow integration no-code agent scheduling productivity",
        "measure": "measure analytics reporting dashboard metrics tracking seo marketing performance",
    }
    expanded = f"{normalized} {intent_aliases.get(normalized, '')}"
    if re.search(r"\b(writ|blog|article|post|copy|content)\w*\b", normalized):
        expanded += " writing writers blog article posts copywriting content seo marketing"
    if re.search(r"\b(meeting|meetings|notetaker|note\s+taker|notes|transcrib|summar)\w*\b", normalized):
        expanded += " meeting notes notetaker transcriber transcription summarizer summary recording"
    if re.search(r"\b(presentation|presentations|slides?|deck)\b", normalized):
        expanded += " presentations slides deck powerpoint pitch"
    if is_coding_query(normalized):
        expanded += " code coding developer programming python javascript debugging"
    if is_chatbot_query(normalized):
        expanded += " chatbot chatbots conversational assistant customer support bot"
    words = re.findall(r"[a-z0-9]+", expanded)
    stopwords = {
        "a", "an", "and", "are", "as", "at", "be", "for", "from", "i", "in",
        "is", "it", "me", "my", "of", "on", "or", "that", "the", "to",
        "ai", "best", "find", "give", "help", "like", "looking", "make",
        "need", "recommend", "show", "tool", "tools", "want", "with", "you",
        "act", "advisor", "consultant", "expert", "practical", "software",
        "wtf", "fuck", "fucking", "shit",
    }
    return [w for w in words if len(w) > 2 and w not in stopwords]


def meta_text(meta: dict[str, Any]) -> str:
    fields = (
        "Name", "Categories", "Description", "Features", "Pros", "Cons",
        "Use_cases", "Price",
    )
    return " ".join(str(meta.get(field, "")) for field in fields).lower()


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def token_count(items: list[str], term: str) -> int:
    return sum(1 for item in items if item == term)


def is_writing_query(q: str) -> bool:
    return bool(re.search(r"\b(writ|blog|article|post|copy|content)\w*\b", q.lower()))


def is_coding_query(q: str) -> bool:
    return bool(re.search(
        r"\b(cod(?:e|ing)|develop(?:er|ment)?|program(?:ming)?|debug|python|javascript|typescript|api|sdk|software\s+development)\b",
        q.lower(),
    ))


def is_chatbot_query(q: str) -> bool:
    return bool(re.search(r"\b(chatbot|chat\s?bot|conversational|virtual\s+assistant|support\s+bot|dialogue|dialog)\b", q.lower()))


# Phrases that signal the user is overriding the previous topic, not refining it.
PIVOT_MARKERS = (
    "actually", "instead", "never mind", "nevermind", "forget that",
    "forget about", "scratch that", "on second thought", "second thoughts",
    "changed my mind", "change of plan", "in fact", "i only need",
    "i just need", "all i need", "all i really need", "what i actually need",
    "let's focus on", "lets focus on", "i really need", "rather than",
)


def focus_latest_intent(q: str) -> str:
    """Return the most recent intent when the message overrides an earlier topic.

    Keeps the text from the last override marker onward so a pivot like
    "... Actually I only need a coding tool" is not polluted by the old topic.
    """
    text = normalize_query_text(q)
    if not text:
        return text
    text = re.sub(r"^(?:nah|nope|no)\b[,\s]+", "", text, flags=re.IGNORECASE).strip()
    lowered = text.lower()
    best_pos = -1
    for marker in PIVOT_MARKERS:
        idx = lowered.rfind(marker)
        if idx >= 0 and len(lowered[idx + len(marker):].split()) < 2:
            continue
        if idx > best_pos:
            best_pos = idx
    if best_pos <= 0:
        return text
    focused = text[best_pos:].strip(" ,.;:-")
    return focused or text


# Categories that are almost never the right answer for a coding or chatbot build.
_DEV_OFF_TOPIC_CATEGORIES = (
    "website builders", "website builder", "image generators", "image generator",
    "video generators", "video", "music", "fitness", "health", "dating", "travel",
    "presentation", "social media", "logo",
)
_DEV_ON_TOPIC_CATEGORIES = (
    "developer tools", "developer", "coding", "code", "api", "chatbot", "chatbots",
    "ai chatbots", "automation", "no-code", "low-code", "productivity", "assistants",
)


def is_explanation_query(text: str) -> bool:
    normalized = normalize_query_text(text).lower().strip()
    if normalized in {"why", "why?", "why this", "why these", "why those"}:
        return True
    if re.fullmatch(r"why\s*(?:tho|though|exactly|again)?\??", normalized):
        return True
    return bool(re.search(
        r"\b("
        r"why\s+(?:is\s+|are\s+|was\s+)?(?:this|that|it|the)\s+(?:the\s+)?best\s+(?:tool|one|option|app|choice|pick)|"
        r"why\s+(?:is\s+|are\s+|was\s+)?(?:this|that|it|the)\s+(?:tool|one|option|app|choice|pick)\s+(?:the\s+)?best|"
        r"why\s+(?:these|this|those)(?:\s+(?:tools?|ones|results?|recommendations?|picks?))?\??|"
        r"why\s+(?:was|is|were|are)\s+[a-z0-9][a-z0-9 .'-]{0,60}\s+(?:chosen|picked|recommended|suggested)|"
        r"why\s+did|"
        r"explain(?:\s+(?:these|this|those|the|why|how))?|"
        r"reason|reasons|"
        r"why\s+(?:recommended|recommend|suggest|suggested|picked|chose|choose)\b|"
        r"why\s+(?:do\s+|did\s+|would\s+|should\s+)?(?:you|it)\s+(?:recommend|suggest|pick|choose|say|think)|"
        r"what\s+makes\s+(?:this|that|it|the)\s+(?:the\s+)?best|"
        r"how\s+(?:is\s+|does\s+)(?:this|that|it|the)\s+(?:tool|one|option|app)\s+(?:the\s+)?best"
        r")\b",
        normalized,
    ))


def is_shortlist_explanation_query(text: str) -> bool:
    normalized = normalize_query_text(text).lower().strip()
    return is_explanation_query(normalized) and bool(re.search(
        r"\b(these|those|results?|recommendations?|picks?|shortlist)\b",
        normalized,
    ))


def is_refinement_query(text: str) -> bool:
    normalized = text.lower().strip()
    return bool(re.search(
        r"\b(free only|free|paid|cheaper|simpler|more specific|more privacy|private|secure|local|self-hosted|show me more|another|alternatives?|compare|better|only)\b",
        normalized,
    ))


_PICK_BEST_PATTERNS = (
    r"\bwhich\s+(?:one|tool|app|of\s+(?:these|those|them))\b",
    r"\bwhich\s+(?:is|are|would|do|should|tool|one)\b.*\bbest\b",
    r"\bbest\s+(?:one|tool|option|pick|choice)\b",
    r"\bbest\s+of\s+(?:these|those|them|the\s+(?:above|options|list))\b",
    r"\b(?:these|those|them|the\s+(?:above|options))\b.*\bbest\b",
    r"\bpick\s+(?:the|a|one|your)\b",
    r"\b(?:your\s+)?top\s+pick\b",
    r"\bsingle\s+best\b",
    r"\bwhich\s+would\s+you\s+recommend\b",
    r"\brecommend\s+(?:the\s+)?most\b",
    r"\bwhat(?:'s| is)\s+the\s+best\b",
)


def is_pick_best_query(text: str) -> bool:
    """Detect 'which one of these is the best and why' style follow-ups."""
    normalized = normalize_query_text(text).lower().strip()
    return any(re.search(pattern, normalized) for pattern in _PICK_BEST_PATTERNS)


def strip_pick_best_clause(text: str) -> str:
    """Remove pick-the-best sentences, leaving any stated task behind."""
    normalized = normalize_query_text(text)
    parts = re.split(r"(?<=[.!?])\s+", normalized)
    kept = [part for part in parts if part.strip() and not is_pick_best_query(part)]
    return " ".join(kept).strip()


def is_referential_pick(text: str) -> bool:
    """True when a pick-best question points at an existing shortlist."""
    normalized = normalize_query_text(text).lower()
    return bool(re.search(
        r"\b(these|those|them|all\s+(?:of\s+)?them|the\s+(?:above|options|list|ones|shortlist|picks)|already|"
        r"you\s+(?:recommended|suggested|listed|showed|gave))\b",
        normalized,
    ))


def pick_best_needs_shortlist_message() -> str:
    return "I can pick the best option from the current shortlist, but I need those results first."


# Specific-tool follow-ups: the user is asking about one named tool in context.
_SPECIFIC_TOOL_PATTERNS = (
    r"\bwhat\s+about\s+([a-z][a-z0-9]*(?:\s+[a-z0-9]+){0,3})",
    r"\bhow\s+about\s+([a-z][a-z0-9]*(?:\s+[a-z0-9]+){0,3})",
    r"\bis\s+([a-z][a-z0-9]*(?:\s+[a-z0-9]+){0,3})\s+(?:good|better|worth|suitable|fit)",
    r"\bwhat\s+do\s+you\s+(?:think|say)\s+about\s+([a-z][a-z0-9]*(?:\s+[a-z0-9]+){0,3})",
    r"\bshould\s+i\s+(?:use|try|consider)\s+([a-z][a-z0-9]*(?:\s+[a-z0-9]+){0,3})",
    r"\bcan\s+i\s+use\s+([a-z][a-z0-9]*(?:\s+[a-z0-9]+){0,3})\s+(?:instead|for)",
    r"\bcompare\s+(?:it\s+with|it\s+to|against)\s+([a-z][a-z0-9]*(?:\s+[a-z0-9]+){0,3})",
    r"\bhow\s+does\s+([a-z][a-z0-9]*(?:\s+[a-z0-9]+){0,3})\s+compare",
)


_NON_TOOL_WORDS = {
    "the", "a", "an", "there", "it", "this", "that", "probably", "maybe",
    "definitely", "so", "too", "very", "really", "quite", "pretty", "more",
    "most", "less", "least", "much", "many", "any", "some", "all", "both",
    "each", "every", "either", "neither", "one", "two", "other", "another",
    "such", "what", "which", "who", "whose", "how", "when", "where", "why",
    "then", "also", "still", "even", "just", "only", "well", "about", "for",
    "with", "from", "into", "onto", "upon", "over", "under", "above", "below",
}


def _specific_tool_name(text: str) -> str | None:
    """Extract the tool name from a specific-tool follow-up, or None if not a valid match."""
    normalized = normalize_query_text(text).lower().strip()
    for pattern in _SPECIFIC_TOOL_PATTERNS:
        match = re.search(pattern, normalized)
        if match:
            name = match.group(1).strip()
            if name and name not in _NON_TOOL_WORDS:
                return name
    return None


def is_specific_tool_query(text: str) -> bool:
    """Detect follow-ups like 'What about Claude?' or 'Is Claude good too?'."""
    return _specific_tool_name(text) is not None


def extract_tool_name(text: str) -> str | None:
    """Extract the tool name from a specific-tool follow-up question."""
    return _specific_tool_name(text)


_CRITERION_PATTERNS = (
    r"\bis\s+(?:it|this|that)\s+(?:actually\s+)?free\b",
    r"\bdoes\s+(?:it|this|that)\s+(?:have|offer)\s+(?:a\s+)?free\b",
    r"\bis\s+(?:it|this|that)\s+paid\b",
    r"\b(?:which\s+(?:one|tool|option)|the)\s+(?:is\s+)?cheapest\b",
    r"\b(?:which\s+(?:one|tool|option)|the)\s+(?:is\s+)?most\s+expensive\b",
    r"\b(?:which\s+(?:one|tool|option)|the)\s+(?:is\s+)?free\b",
    r"\b(?:which\s+(?:one|tool|option)|the)\s+(?:is\s+)?paid\b",
    r"\b(?:which\s+(?:one|tool|option)|the)\s+(?:is\s+)?best\s+value\b",
    r"\b(?:which\s+(?:one|tool|option)|the)\s+(?:is\s+)?most\s+private\b",
    r"\b(?:which\s+(?:one|tool|option)|the)\s+(?:is\s+)?most\s+secure\b",
    r"\b(?:which\s+(?:one|tool|option)|the)\s+(?:is\s+)?beginner\s*[-]?friendly\b",
    r"\b(?:which\s+(?:one|tool|option)|the)\s+(?:is\s+)?easiest\b",
    r"\b(?:show\s+me\s+the\s+|give\s+me\s+the\s+)free\s+(?:one|tool|option)\b",
    r"\b(?:show\s+me\s+the\s+|give\s+me\s+the\s+)cheapest\s+(?:one|tool|option)\b",
    r"\b(?:show\s+me\s+the\s+|give\s+me\s+the\s+)paid\s+(?:one|tool|option)\b",
)


_ALTERNATIVE_PATTERNS = (
    r"\b(?:is\s+there\s+)?any\s+other\s+(?:tool|one|option|app)\b",
    r"\b(?:show\s+me\s+|give\s+me\s+)another\b",
    r"\b(?:show\s+me\s+)?something\s+else\b",
    r"\b(?:a\s+|the\s+)?different\s+(?:tool|one|option)\b",
    r"\bnext\s+best\b",
    r"\bwhat\s+else\b",
    r"\bnot\s+that\s+one\b",
    r"\b(?:i\s+|you\s+)gave\s+me\s+the\s+same\s+one\b",
    r"\b(?:i\s+)?already\s+(?:have|know|use|got)\s+that\b",
    r"\bbetter\s+than\s+(?:this|that|the\s+one|it)\b",
    r"\b(?:can\s+you\s+)?recommend\s+another\b",
    r"\b(?:another\s+|other\s+)alternative\b",
    r"\banything\s+else\b",
    r"\belse\s+(?:that\s+)?(?:you\s+)?(?:recommend|suggest)\b",
)


def is_alternative_query(text: str) -> bool:
    """Detect requests for a different tool from the existing shortlist."""
    normalized = normalize_query_text(text).lower().strip()
    return any(re.search(pattern, normalized) for pattern in _ALTERNATIVE_PATTERNS)


def is_criterion_pick_query(text: str) -> bool:
    """Detect follow-ups like 'which one is the cheapest?' or 'the free one'."""
    normalized = normalize_query_text(text).lower().strip()
    return any(re.search(pattern, normalized) for pattern in _CRITERION_PATTERNS)


def criterion_from_query(text: str) -> str:
    """Return the criterion the user is asking for (cheapest, free, paid, etc.)."""
    normalized = normalize_query_text(text).lower().strip()
    if re.search(r"\b(?:cheapest|least\s+expensive|best\s+value)\b", normalized):
        return "cheapest"
    if re.search(r"\b(?:most\s+expensive|priciest)\b", normalized):
        return "most_expensive"
    if re.search(r"\b(?:free\s+(?:one|tool|option)|no\s+cost)\b|\bis\s+(?:it|this|that)\s+(?:actually\s+)?free\b|\bdoes\s+(?:it|this|that)\s+(?:have|offer)\s+(?:a\s+)?free\b", normalized):
        return "free"
    if re.search(r"\b(?:paid\s+(?:one|tool|option)|most\s+expensive)\b|\bis\s+(?:it|this|that)\s+paid\b", normalized):
        return "paid"
    if re.search(r"\b(?:private|privacy|secure)\b", normalized):
        return "privacy"
    if re.search(r"\b(?:beginner|easiest|simplest)\b", normalized):
        return "beginner"
    return "best"


def _price_sort_key(meta: dict[str, Any]) -> tuple[int, float]:
    """Lower tuple = cheaper. Free/open-source first, then try to parse dollar amount."""
    price = normalize_display_text(meta.get("Price", "")).lower()
    if is_free_tool(meta):
        return (0, 0.0)
    if "open source" in price:
        return (0, 0.0)
    # Extract the lowest dollar amount mentioned.
    amounts = [float(m.replace("$", "").replace(",", ""))
               for m in re.findall(r"\$\s*([\d,]+(?:\.\d{2})?)", price, flags=re.IGNORECASE)]
    if amounts:
        return (1, min(amounts))
    # Paid but no number: sort after known numbers but before unknown.
    if re.search(r"[$€£]|\b(paid|pro|premium|subscription|enterprise)\b", price):
        return (2, 0.0)
    return (3, 0.0)


def _matches_tool_name(meta: dict[str, Any], name: str) -> bool:
    """Case-insensitive fuzzy match of a tool name against the catalog Name."""
    if not name:
        return False
    tool_name = str(meta.get("Name", "")).lower()
    normalized_name = name.lower()
    # Exact match or normalized exact match.
    if normalized_name == tool_name:
        return True
    # Remove common suffixes/prefixes and compare.
    clean = re.sub(r"\b(ai|app|tool|bot|chat|gpt|claude)\b", "", tool_name).strip()
    clean_query = re.sub(r"\b(ai|app|tool|bot|chat|gpt|claude)\b", "", normalized_name).strip()
    if clean_query and clean_query == clean:
        return True
    if clean_query and clean_query in tool_name:
        return True
    if normalized_name in tool_name:
        return True
    # Acronym / token overlap (e.g. "ChatGPT" vs "chatgpt").
    if normalized_name.replace(" ", "") == tool_name.replace(" ", ""):
        return True
    return False


def _sort_hits_by_criterion(hits: list[dict[str, Any]], criterion: str) -> list[dict[str, Any]]:
    """Return a copy of hits sorted by the requested criterion."""
    if criterion == "cheapest":
        return sorted(hits, key=lambda h: _price_sort_key(h.get("meta") or {}))
    if criterion == "most_expensive":
        return sorted(hits, key=lambda h: _price_sort_key(h.get("meta") or {}), reverse=True)
    if criterion == "free":
        free = [h for h in hits if is_free_tool(h.get("meta") or {})]
        return free or hits
    if criterion == "paid":
        paid = [h for h in hits if not is_free_tool(h.get("meta") or {})]
        return paid or hits
    # For privacy / beginner / best, keep original order but move strongest matches first.
    return list(hits)


def criterion_pick_message(hit: dict[str, Any], criterion: str, query: str) -> str:
    meta = hit.get("meta") or {}
    name = str(meta.get("Name", "This tool")).strip() or "This tool"
    price = normalize_display_text(meta.get("Price", ""))
    if criterion == "free":
        if is_free_tool(meta):
            return f"Yes. {name} appears to have a free tier or trial. Check the provider page for current limits before relying on it."
        return f"I do not see a clear free tier for {name}. Check the provider page before choosing it."
    if criterion == "paid":
        if is_free_tool(meta):
            return f"{name} has free access listed, but it may also have paid upgrades. Check the provider page for limits."
        return f"{name} appears to be a paid option. Check the provider page for the current plan details."
    if criterion == "cheapest":
        if price:
            return f"The cheapest-looking option from the shortlist is {name}. Its pricing says: {complete_sentences(price, 180, max_sentences=1)}"
        return f"The cheapest-looking option from the shortlist is {name}, but its pricing is not clear in the catalogue."
    if criterion == "most_expensive":
        return f"The most expensive-looking option from the shortlist is {name}. Verify current pricing on the provider page."
    if criterion == "privacy":
        return f"For privacy or security, I would look first at {name}. Verify its retention, compliance, and data-control details before using sensitive data."
    if criterion == "beginner":
        return f"For ease of use, I would start with {name}. It looks like the simplest fit from the current shortlist."
    return recommendation_message([hit], query, MODE_ONE_BEST)


def specific_tool_message(hit: dict[str, Any], query: str) -> str:
    meta = hit.get("meta") or {}
    name = str(meta.get("Name", "This tool")).strip() or "This tool"
    goal = request_goal(query)
    categories = str(meta.get("Categories", ""))
    if off_topic_for_query(query, categories):
        cats = category_list(meta, limit=2)
        focus = f" It looks more focused on {human_join(cats)}." if cats else ""
        return f"{name} is not my strongest pick for {goal}.{focus} I would only choose it if that side task matters."
    why = complete_sentences(str(hit.get("why") or local_reason(query, meta)), 260, max_sentences=2)
    return f"{name} can work for {goal}. {why}".strip()


def no_more_alternatives_message() -> str:
    return "I do not have another distinct option in the current shortlist. Try a new search or loosen the filters."


def alternative_message(hit: dict[str, Any], query: str) -> str:
    meta = hit.get("meta") or {}
    name = str(meta.get("Name", "This tool")).strip() or "This tool"
    why = complete_sentences(str(hit.get("why") or local_reason(query, meta)), 220, max_sentences=1)
    message = f"Another option is {name}."
    if why:
        message = f"{message} {why}"
    return clean_assistant_message(message)


def needs_clarification(q: str) -> bool:
    if is_feedback_only_query(q):
        return True
    normalized = strip_instruction_text(q).lower().strip()
    terms = query_terms(normalized)
    if is_explanation_query(normalized):
        return False
    if requires_free_only(normalized) and not non_filter_terms(normalized):
        return True
    if len(terms) <= 1 and re.search(r"\b(tool|tools|app|apps|software|ai)\b", normalized):
        return True
    return normalized in {
        "i need a tool",
        "i need an ai tool",
        "recommend a tool",
        "recommend an ai tool",
        "find a tool",
        "find an ai tool",
        "help me choose a tool",
    }


def default_clarifying_question(q: str) -> str:
    if is_feedback_only_query(q):
        return feedback_clarifying_question()
    if requires_free_only(q) and not non_filter_terms(q):
        return "What task should the free tool help with?"
    if "free" not in q.lower() and re.search(r"\b(tool|tools|app|apps|software|ai)\b", q.lower()):
        return "What task do you want the tool to help with, and do you need a free option?"
    return "What exact task should the tool help with?"


def off_topic_for_query(q: str, categories: str) -> bool:
    category_tokens = set(tokens(categories))
    text = categories.lower()
    if is_writing_query(q):
        allowed = {"writing", "generators", "copywriting", "seo", "marketing", "social", "media"}
        blocked = {"fitness", "health", "travel", "dating", "music", "finance", "stock", "trading"}
        if category_tokens & blocked:
            return True
        return not bool(category_tokens & allowed)
    if any(term in q.lower() for term in ("meeting", "meetings", "notetaker", "note taker", "notes", "transcrib")):
        blocked = {"travel", "image", "images", "logo", "website", "dating", "fitness", "health"}
        if category_tokens & blocked:
            return True
        return not bool(re.search(r"\b(meeting|transcrib|transcription|summar|notetaker|note|audio|record)\b", text))
    if re.search(r"\b(presentation|presentations|slides?|deck|powerpoint)\b", q.lower()):
        blocked = {"image", "images", "logo", "video", "audio", "music", "dating", "travel"}
        if category_tokens & blocked and "present" not in text and "slide" not in text:
            return True
        return not bool(re.search(r"\b(presentation|presentations|slides?|deck|powerpoint)\b", text))
    if is_coding_query(q):
        blocked = {"image", "images", "logo", "video", "music", "travel", "dating", "fitness", "health"}
        if category_tokens & blocked:
            return True
        return not bool(re.search(r"\b(code|coding|developer|programming|debug|python|javascript|software|api|sdk)\b", text))
    if is_chatbot_query(q):
        blocked = {"image", "images", "logo", "video", "music", "travel", "dating", "fitness", "health"}
        if category_tokens & blocked:
            return True
        return not bool(re.search(r"\b(chatbot|chatbots|chat\s?bot|conversational|assistant|customer|support|bot)\b", text))
    return False


def shorten_goal_text(cleaned: str, max_words: int = 8) -> str:
    """Trim a free-text goal to a short phrase, avoiding restating the whole query."""
    text = cleaned.strip().rstrip(".")
    if not text:
        return "your task"
    # Prefer the segment after a leading "I need / I want / looking for" lead-in.
    lead = re.sub(
        r"^(?:i\s+(?:need|want|am\s+looking\s+for|would\s+like)|looking\s+for|find|recommend|help\s+me\s+(?:find|with)|i'?m\s+(?:building|creating|making))\s+",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    candidate = lead or text
    words = candidate.split()
    if len(words) > max_words:
        candidate = " ".join(words[:max_words])
    return candidate or "your task"


def request_goal(q: str) -> str:
    cleaned = strip_instruction_text(q)
    text = cleaned.lower()
    if any(term in text for term in ("chatbot", "chat bot", "conversational", "virtual assistant", "support bot")):
        return "building a chatbot"
    if any(term in text for term in ("meeting", "meetings", "notetaker", "note taker", "notes")):
        if any(term in text for term in ("privacy", "private", "local", "self-hosted", "secure", "security")):
            return "private meeting notes"
        return "meeting notes and summaries"
    if is_writing_query(q):
        if "essay" in text or "academic" in text:
            return "essay writing"
        if "blog" in text or "article" in text:
            return "writing blog posts"
        if "social" in text or "post" in text:
            return "creating social media posts"
        return "writing content"
    if "transcrib" in text or "audio" in text:
        return "transcribing audio"
    if "image" in text:
        return "generating images"
    if any(term in text for term in ("video", "youtube")):
        return "creating videos"
    if "presentation" in text or "slides" in text:
        return "creating presentations"
    if re.search(r"\b(cod(?:e|ing)|develop(?:er|ment)?|program(?:ming)?|debug|python|javascript|software)\b", text):
        return "coding and development"
    if any(term in text for term in ("automat", "workflow", "integrat")):
        return "automating workflows"
    if "research" in text or "competitor" in text:
        return "researching information"
    return shorten_goal_text(cleaned)


# Specific goals produced by request_goal (i.e. a recognised, self-sufficient task).
KNOWN_SPECIFIC_GOALS = frozenset({
    "building a chatbot", "private meeting notes", "meeting notes and summaries",
    "essay writing", "writing blog posts", "creating social media posts",
    "writing content", "transcribing audio", "generating images", "creating videos",
    "creating presentations", "coding and development", "automating workflows",
    "researching information",
})


def has_explicit_task(text: str) -> bool:
    """True when the message names a concrete task on its own (not just a filter)."""
    if (
        is_explanation_query(text)
        or is_pick_best_query(text)
        or is_specific_tool_query(text)
        or is_criterion_pick_query(text)
        or is_alternative_query(text)
    ):
        return False
    return request_goal(text) in KNOWN_SPECIFIC_GOALS


def evidence_fragments(value: str) -> list[str]:
    raw_parts = re.split(r"(?<=[.!?])\s+|\|", str(value or ""))
    fragments = []
    for part in raw_parts:
        fragment = " ".join(part.strip().rstrip(".!?").split())
        if 6 <= len(fragment.split()) <= 28:
            fragments.append(fragment)
    return fragments


def best_evidence(q: str, meta: dict[str, Any]) -> str:
    terms = set(query_terms(q))
    fragments = []
    for field in ("Features", "Pros", "Description"):
        fragments.extend(evidence_fragments(str(meta.get(field, ""))))
    if not fragments:
        return ""

    def score(fragment: str) -> int:
        return len(terms & set(tokens(fragment)))

    best = max(fragments, key=score)
    return best if score(best) > 0 else fragments[0]


def local_reason(q: str, meta: dict[str, Any]) -> str:
    name = str(meta.get("Name", "This tool")).strip() or "This tool"
    goal = request_goal(q)
    detail = practical_fit_detail(goal, meta)
    reason = f"{name} is well suited for {goal} because {detail}."
    price_note = price_reason(meta)
    if price_note:
        reason += f" {price_note}"
    return sanitize_reason(reason, name=name, query=q)


def practical_fit_detail(goal: str, meta: dict[str, Any]) -> str:
    text = meta_text(meta)
    categories = set(tokens(str(meta.get("Categories", ""))))

    if "private meeting notes" in goal:
        if any(term in text for term in ("private", "privacy", "local", "self-hosted", "secure", "security", "compliance")):
            return "it emphasizes privacy, local control, or security signals for meeting notes"
        return "it supports meeting notes while matching your privacy-focused search"
    if "meeting notes" in goal:
        if any(term in text for term in ("transcrib", "summar", "record")):
            return "it can help capture, transcribe, or summarize meetings"
        return "it is built around meeting workflows"
    if "writing blog posts" in goal:
        if "seo" in categories or "seo" in text:
            return "it combines blog or article drafting with SEO-focused writing support"
        if any(term in text for term in ("grammar", "style", "clarity", "readability")):
            return "it improves grammar, style, clarity, and readability"
        if any(term in text for term in ("blog", "article", "content")):
            return "it is focused on creating blog posts, articles, or written content"
        return "it has writing features that match blog and article drafting"
    if "essay writing" in goal:
        if any(term in text for term in ("academic", "essay", "citation", "plagiarism", "research")):
            return "it supports academic writing, essays, or research-oriented drafting"
        if any(term in text for term in ("grammar", "style", "clarity", "readability")):
            return "it improves grammar, style, clarity, and readability"
        return "it has writing features that can help draft and improve essays"
    if "creating social media posts" in goal:
        if any(term in text for term in ("schedule", "publish", "calendar")):
            return "it helps create, schedule, and manage social posts"
        return "it supports social media content creation"
    if "writing content" in goal:
        if "seo" in text:
            return "it combines content generation with SEO writing support"
        if "copywriting" in categories:
            return "it is focused on copywriting and content drafts"
        if any(term in text for term in ("grammar", "style", "clarity", "readability")):
            return "it improves grammar, style, clarity, and readability"
        return "it has broad writing features for drafting and improving content"
    if "building a chatbot" in goal:
        if any(term in text for term in ("chatbot", "conversational", "assistant", "dialog", "dialogue", "bot")):
            return "it provides chatbot or conversational assistant capabilities you can build on"
        if any(term in text for term in ("api", "sdk", "llm", "language model", "nlp")):
            return "it exposes language model or API features useful for building a chatbot"
        return "its features align with building conversational chatbot experiences"
    if "transcribing audio" in goal:
        return "it is designed to convert audio into text or meeting notes"
    if "generating images" in goal:
        return "it generates images or visual assets from prompts"
    if "creating videos" in goal:
        return "it helps generate or edit video content"
    if "creating presentations" in goal:
        return "it helps create slides or presentation content faster"
    if "coding and development" in goal:
        return "it assists with writing, completing, or debugging code"
    if "automating workflows" in goal:
        return "it automates tasks or connects apps into workflows"
    if "researching information" in goal:
        return "it supports research, analysis, or information gathering"

    evidence = clean_evidence(best_evidence(goal, meta))
    if evidence:
        return evidence
    cats = category_list(meta, limit=2)
    if cats:
        return f"it focuses on {human_join(cats)}"
    return "its listed features match the request"


def clean_evidence(value: str) -> str:
    evidence = normalize_display_text(value)
    if not evidence:
        return ""
    if ":" in evidence:
        label, rest = evidence.split(":", 1)
        if len(label.split()) <= 5 and rest.strip():
            evidence = rest.strip()
    evidence = evidence[0].lower() + evidence[1:] if evidence else evidence
    if not re.search(r"[.!?]$", evidence):
        return evidence
    return evidence.rstrip(".!?")


def price_reason(meta: dict[str, Any]) -> str:
    price = normalize_display_text(meta.get("Price", ""))
    lower = price.lower()
    if not price:
        return ""
    if is_free_tool(meta):
        return "It also has a free tier or trial, so you can test it without paying upfront."
    money = re.search(r"\$\s?\d+(?:\.\d{2})?(?:\s*/?\s*(?:month|mo|monthly|year|yr|annually))?", price, re.IGNORECASE)
    if money:
        return f"Pricing starts around {money.group(0).replace(' ', '')}."
    if "waitlist" in lower:
        return "Pricing is not public yet, so verify availability before relying on it."
    return ""


def normalize_display_text(value: Any) -> str:
    text = " ".join(str(value or "").split())
    text = re.sub(r"\b(?:consultant|advisor)\s+view\s*[:\-]\s*", "", text, flags=re.IGNORECASE)
    text = strip_instruction_text(text)
    text = text.replace("...", "")
    text = text.replace("..", ".")
    text = re.sub(r"\s+([.,;:])", r"\1", text)
    return " ".join(text.split()).strip(" -:")


def human_join(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def category_list(meta: dict[str, Any], limit: int = 3) -> list[str]:
    raw = str(meta.get("Categories", ""))
    categories = [c.strip() for c in re.split(r"[|,/]", raw) if c.strip()]
    return categories[:limit]


def ensure_terminal_punctuation(text: str) -> str:
    cleaned = normalize_display_text(text)
    if not cleaned:
        return ""
    if not re.search(r"[.!?]$", cleaned):
        cleaned += "."
    return cleaned


def clean_assistant_message(value: Any) -> str:
    """Final cleanup for user-facing assistant bubbles."""
    text = normalize_display_text(value)
    text = re.sub(
        r"\b(?:consultant|advisor)\s+view\s*[:\-]\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(?:these are alternatives worth comparing,\s*not identical picks|"
        r"it appears to be the best first test from the current catalogue data|"
        r"because it matches the task,\s*price,\s*and feature signals best)[.?!]?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\bI would start with\b", "Start with", text, flags=re.IGNORECASE)
    text = re.sub(r"\bBest for:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r";\s*", ". ", text)
    text = normalize_display_text(text)
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text


def compact_meta(meta: dict[str, Any], summary: Any = None) -> dict[str, Any]:
    return dict(meta)


def plain_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return {}


def visible_tool_hits(visible_tools: Any) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for item in visible_tools or []:
        raw = plain_dict(item)
        if not raw:
            continue
        meta = raw.get("meta")
        if not isinstance(meta, dict):
            continue
        hits.append({
            "score": float(raw.get("score", 0.0) or 0.0),
            "meta": compact_meta(meta),
            "why": raw.get("why") or local_reason("", meta),
            "tradeoff": raw.get("tradeoff") or build_tradeoff(meta),
            "best_for": raw.get("best_for") or build_best_for("", meta),
            "fit_label": raw.get("fit_label") or "Good match",
        })
    return hits


def compact_hit_for_prompt(hit: dict[str, Any]) -> dict[str, Any]:
    meta = hit.get("meta") or {}
    return {
        "name": meta.get("Name", ""),
        "categories": meta.get("Categories", ""),
        "price": meta.get("Price", ""),
        "why": hit.get("why", ""),
        "best_for": hit.get("best_for", ""),
    }


def sanitize_reason(reason: Any, name: str = "This tool", query: str = "") -> str:
    text = normalize_display_text(reason)
    if not text:
        return local_reason(query, {"Name": name})

    query_clean = " ".join(str(query or "").split())
    if query_clean:
        text = text.replace(query_clean, request_goal(query_clean))

    for pattern in INSTRUCTION_LEAK_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    text = normalize_display_text(text)

    if not text:
        text = f"{name} matches this request based on its listed features."

    completed = complete_sentences(text, MAX_REASON_CHARS, max_sentences=2)
    if completed:
        return completed

    goal = request_goal(query)
    return f"{name} matches {goal} based on its listed features."


def complete_sentences(value: Any, max_chars: int, max_sentences: int = 2) -> str:
    text = normalize_display_text(value)
    if not text:
        return ""
    sentences = [ensure_terminal_punctuation(s) for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    selected = []
    for sentence in sentences:
        if not sentence:
            continue
        candidate = " ".join([*selected, sentence])
        if len(candidate) <= max_chars:
            selected.append(sentence)
        if len(selected) >= max_sentences:
            break
    if selected:
        return " ".join(selected)
    if len(text) <= max_chars:
        return ensure_terminal_punctuation(text)
    return ""


def keyword_scores(q: str, k: int, meta_rows: list[dict[str, Any]]) -> list[tuple[float, int]]:
    terms = query_terms(q)
    if not terms:
        return []
    free_only = requires_free_only(q)
    ranking_terms = [term for term in terms if term not in FREE_FILTER_WORDS]
    if free_only and not ranking_terms:
        return []

    scored = []
    for idx, meta in enumerate(meta_rows):
        if free_only and not is_free_tool(meta):
            continue

        text_items = tokens(meta_text(meta))
        text_set = set(text_items)
        name = str(meta.get("Name", "")).lower()
        categories = str(meta.get("Categories", "")).lower()
        description = str(meta.get("Description", "")).lower()
        category_items = tokens(categories)
        name_items = tokens(name)
        description_items = tokens(description)

        if off_topic_for_query(q, categories):
            continue

        score = 0.0
        for term in ranking_terms:
            if term in name_items:
                score += 8.0
            if term in category_items:
                score += 6.0
            if term in description_items:
                score += 2.0
            if term in text_set:
                score += min(token_count(text_items, term), 4) * 0.5

        task_score = score
        if free_only and task_score <= 0:
            continue
        if "free" in terms and is_free_tool(meta):
            score += 4.0
        if is_writing_query(q):
            if any(term in category_items for term in ("writing", "copywriting", "seo", "marketing")):
                score += 10.0
        elif is_coding_query(q) or is_chatbot_query(q):
            if any(cat in categories for cat in _DEV_ON_TOPIC_CATEGORIES):
                score += 10.0
            if any(cat in categories for cat in _DEV_OFF_TOPIC_CATEGORIES):
                score -= 10.0
        elif any(term in terms for term in ("business", "marketing", "competitor", "social", "seo")):
            if any(term in categories for term in ("fitness", "health", "fun tools", "dating")):
                score -= 8.0

        if score >= 4.0:
            scored.append((score, idx))

    scored.sort(reverse=True, key=lambda item: item[0])
    return scored[:k]


def keyword_search(q: str, k: int, meta_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "score": float(score),
            "meta": compact_meta(meta_rows[idx]),
            "why": local_reason(q, meta_rows[idx]),
        }
        for score, idx in keyword_scores(q, k, meta_rows)
    ]


def recommendation_message(
    hits: list[dict[str, Any]],
    query: str = "",
    mode: str = "balanced",
    pick_best: bool = False,
) -> str:
    mode = normalize_mode(mode)
    names = [
        str((hit.get("meta") or {}).get("Name", "")).strip()
        for hit in hits
        if (hit.get("meta") or {}).get("Name")
    ]
    names = [name for name in names if name]
    if not names:
        if pick_best:
            return "I need the earlier results first. Run a search, then I can pick the single best option from them."
        if requires_free_only(query):
            return "I could not find a strong free match. Try changing the task or allowing free trials and freemium plans."
        return "I could not find a strong match. Try adding the task, budget, and any must-have integrations."

    goal = request_goal(query)
    first = names[0]

    if pick_best:
        why = complete_sentences(str((hits[0].get("why") or "")).strip(), 260, max_sentences=2)
        lead = f"I would pick {first} for {goal}."
        return clean_assistant_message(f"{lead} {why}".strip() if why else lead)

    if mode == MODE_ONE_BEST:
        why = complete_sentences(str((hits[0].get("why") or "")).strip(), 220, max_sentences=1)
        message = f"My top pick for {goal} is {first}."
        if why:
            message = f"{message} {why}"
        return clean_assistant_message(message)

    if mode == MODE_COMPARE:
        if len(names) >= 2:
            listed = human_join(names[: min(3, len(names))])
            return clean_assistant_message(
                f"Here is a side-by-side comparison for {goal}: {listed}. Check the fit and tradeoff notes on each card."
            )
        return clean_assistant_message(f"I only found {first} as a clear match for {goal}, so there is little to compare yet.")

    if len(names) == 1:
        why = complete_sentences(str((hits[0].get("why") or "")).strip(), 220, max_sentences=1)
        message = f"Start with {first}."
        if why:
            message = f"{message} {why}"
        return clean_assistant_message(message)

    second = names[1]
    return clean_assistant_message(f"Start with {first}. Compare it with {second} if you want another good option.")


def shortlist_explanation_message(hits: list[dict[str, Any]], query: str) -> str:
    names = [
        str((hit.get("meta") or {}).get("Name", "")).strip()
        for hit in hits
        if (hit.get("meta") or {}).get("Name")
    ]
    names = [name for name in names if name]
    if not names:
        return "I can explain the current shortlist after a search, but I need the previous results to do that."

    goal = request_goal(query)
    if len(names) == 1:
        return recommendation_message(hits, query, MODE_ONE_BEST, pick_best=True)

    first = names[0]
    alternatives = human_join(names[1:min(3, len(names))])
    reason = complete_sentences(str(hits[0].get("why") or ""), 180, max_sentences=1)
    message = (
        f"I chose these because they are the strongest matches I found for {goal}. "
        f"{first} is the best first pick"
    )
    if reason:
        message += f": {reason}"
    else:
        message += "."
    if alternatives:
        message += f" {alternatives} are useful alternatives to compare before choosing."
    return clean_assistant_message(message)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


def mmr_rerank(
    candidates: list[dict[str, Any]],
    embeddings: np.ndarray,
    lambda_: float = 0.7,
    top_k: int = 30,
) -> list[dict[str, Any]]:
    n = len(candidates)
    if n == 0:
        return []
    selected_indices: list[int] = []
    remaining_indices: list[int] = list(range(n))

    max_score = max(c["score"] for c in candidates) or 1.0
    relevance = np.array([c["score"] / max_score for c in candidates])

    for _ in range(min(top_k, n)):
        best_idx = None
        best_mmr = -float("inf")

        for idx in remaining_indices:
            rel = relevance[idx]
            if not selected_indices:
                max_sim = 0.0
            else:
                sims = [
                    cosine_similarity(embeddings[idx], embeddings[selected])
                    for selected in selected_indices
                ]
                max_sim = max(sims)
            mmr_score = lambda_ * rel - (1 - lambda_) * max_sim

            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = idx

        if best_idx is None:
            break
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)

    return [candidates[i] for i in selected_indices]


class RecommendationService:
    def __init__(
        self,
        store: ToolStore,
        client: Any,
        settings: Settings,
        metrics: RuntimeMetrics,
    ) -> None:
        self.store = store
        self.client = client
        self.settings = settings
        self.metrics = metrics
        self.embedding_cache: TTLCache[np.ndarray] = TTLCache(
            max_entries=settings.cache_max_entries,
            ttl_seconds=settings.cache_ttl_seconds,
        )
        self.recommend_cache: TTLCache[list[dict[str, Any]]] = TTLCache(
            max_entries=settings.cache_max_entries,
            ttl_seconds=settings.cache_ttl_seconds,
        )
        self.conversations = ConversationStore(
            max_conversations=settings.cache_max_entries,
            ttl_seconds=settings.cache_ttl_seconds,
        )
        self.shortlists: dict[str, list[dict[str, Any]]] = {}
        self.shortlist_pointers: dict[str, int] = {}

    def health(self) -> dict[str, Any]:
        return {
            "ok": self.store.ready,
            "items": len(self.store.meta),
            "openai_configured": bool(self.settings.openai_api_key),
            "vectors_loaded": self.store.vectors is not None,
        }

    def embed(self, texts: list[str]) -> np.ndarray:
        if len(texts) == 1:
            key = f"{self.settings.emb_model}:{texts[0]}"
            cached = self.embedding_cache.get(key)
            if cached is not None:
                self.metrics.increment("embedding_cache_hit")
                return cached.copy()

        with self.metrics.timer("openai.embeddings_ms"):
            resp = self.client.embeddings.create(model=self.settings.emb_model, input=texts)
        vecs = np.array([d.embedding for d in resp.data], dtype="float32")
        normalize_l2(vecs)

        if len(texts) == 1:
            self.embedding_cache.set(f"{self.settings.emb_model}:{texts[0]}", vecs.copy())
        return vecs

    def search(self, q: str, k: int) -> dict[str, Any]:
        self.metrics.increment("search_requests")
        q = strip_instruction_text(q)
        if is_feedback_only_query(q):
            return {"hits": []}
        free_only = requires_free_only(q)

        try:
            vec = self.embed([q])
        except Exception as exc:
            logger.info("Embedding unavailable; served keyword search fallback (%s)", type(exc).__name__)
            self.metrics.increment("embedding_fallbacks")
            return {"hits": keyword_search(q, k, self.store.meta)}

        with self.metrics.timer("faiss.search_ms"):
            scores, ids = self.store.index.search(vec, min(k, len(self.store.meta)))

        hits = []
        for score, id_ in zip(scores[0].tolist(), ids[0].tolist()):
            if id_ == -1:
                continue
            if free_only and not is_free_tool(self.store.meta[id_]):
                continue
            hits.append({"score": float(score), "meta": compact_meta(self.store.meta[id_])})
        return {"hits": hits}

    def recommend(
        self,
        q: str,
        retrieve_k: int,
        final_k: int,
        filters: Any = None,
        mode: str = "balanced",
        conversation_id: str | None = None,
        history: Any = None,
    ) -> dict[str, Any]:
        self.metrics.increment("recommend_requests")
        mode = normalize_mode(mode)
        if is_non_search_message(q):
            self.metrics.increment("recommend_non_search_blocked")
            return {"hits": [], "message": non_search_response(q)}
        if is_feedback_only_query(q):
            self.metrics.increment("recommend_feedback_query_blocked")
            return {
                "hits": [],
                "message": feedback_clarifying_question(),
            }
        if is_explanation_query(q):
            self.metrics.increment("recommend_explain_query_blocked")
            prior_hits = self.shortlists.get(conversation_id) if conversation_id else None
            if prior_hits:
                prior_messages = [
                    m.get("content", "") for m in self.conversations.get(conversation_id) or []
                ]
                prior_task = next(
                    (m for m in reversed(prior_messages) if has_explicit_task(m)), ""
                )
                reason_query = prior_task or q
                if is_shortlist_explanation_query(q) and len(prior_hits) > 1:
                    explained_hits = [
                        enrich_hit(dict(hit), reason_query)
                        for hit in prior_hits[: min(3, len(prior_hits))]
                    ]
                    for explained in explained_hits:
                        meta = explained.get("meta") or {}
                        explained["why"] = local_reason(reason_query, meta)
                    message = shortlist_explanation_message(explained_hits, reason_query)
                    self.conversations.append(conversation_id, "assistant", message)
                    return {"hits": explained_hits, "message": message}

                top = prior_hits[0]
                explained = enrich_hit(dict(top), reason_query)
                explained["why"] = local_reason(reason_query, explained.get("meta") or {})
                explained["best_for"] = (
                    f"{explained.get('meta', {}).get('Name', 'This tool')} is the best choice for "
                    f"{reason_query} because it matches the task, price, and feature signals best."
                )
                message = recommendation_message([explained], reason_query, MODE_ONE_BEST, pick_best=True)
                self.conversations.append(conversation_id, "assistant", message)
                return {"hits": [explained], "message": message}
            return {
                "hits": [],
                "message": "I can explain the current shortlist after a search, but I need the previous results to do that.",
            }
        q = strip_instruction_text(q)
        # If the user pivots ("Actually I only need ..."), follow the latest intent.
        q = focus_latest_intent(q)

        # Pull recent task context so follow-up messages behave like a real conversation,
        # but never drag an old topic into a message that already states its own task.
        stored = self.conversations.get(conversation_id)
        prior_messages = merge_history_messages(history, stored)
        self.conversations.append(conversation_id, "user", q)

        # Recover the task that produced the current shortlist. We use it for
        # contextual reasons when answering follow-ups about specific tools.
        prior_task = next((m for m in reversed(prior_messages) if has_explicit_task(m)), "")
        if not prior_task and prior_messages:
            prior_task = prior_messages[-1]

        # Specific-tool follow-up: "What about Claude?" or "Is Claude good too?"
        # Answer from the existing shortlist (or the catalog) instead of running a new search.
        tool_name = extract_tool_name(q)
        if tool_name and conversation_id and self.shortlists.get(conversation_id):
            self.metrics.increment("recommend_specific_tool_requests")
            prior_hits = self.shortlists[conversation_id]
            hit = self._find_hit_by_name(prior_hits, tool_name)
            meta = (hit.get("meta") if hit else None) or self._find_tool_by_name(tool_name)
            if meta:
                reason_query = prior_task or q
                single_hit = enrich_hit({
                    "score": float(hit.get("score", 0.0)) if hit else 0.0,
                    "meta": compact_meta(meta),
                    "why": local_reason(reason_query, meta),
                }, reason_query)
                message = specific_tool_message(single_hit, reason_query)
                self.conversations.append(conversation_id, "assistant", message)
                if hit:
                    self._set_shortlist_pointer(
                        conversation_id, prior_hits.index(hit)
                    )
                return {"hits": [single_hit], "message": message}

        # Criterion-based pick: "Which one is the cheapest/free/paid?"
        # Re-rank the existing shortlist and return the top match for that criterion.
        criterion = criterion_from_query(q)
        if is_criterion_pick_query(q) and conversation_id and self.shortlists.get(conversation_id):
            self.metrics.increment("recommend_criterion_pick_requests")
            prior_hits = self.shortlists[conversation_id]
            sorted_hits = _sort_hits_by_criterion(prior_hits, criterion)
            if criterion in {"free", "paid"} and re.search(r"\b(?:it|this|that)\b", q.lower()):
                current_idx = self.shortlist_pointers.get(conversation_id, 0)
                best_hit = prior_hits[min(current_idx, len(prior_hits) - 1)]
            else:
                best_hit = sorted_hits[0] if sorted_hits else prior_hits[0]
            if best_hit:
                reason_query = prior_task or q
                single_hit = enrich_hit(dict(best_hit), reason_query)
                message = criterion_pick_message(single_hit, criterion, reason_query)
                self.conversations.append(conversation_id, "assistant", message)
                self._set_shortlist_pointer(
                    conversation_id, prior_hits.index(best_hit)
                )
                return {"hits": [single_hit], "message": message}

        # Alternative request: "Is there any other tool?" / "Show me another" / "Not that one".
        # Return the next hit from the stored shortlist instead of re-searching.
        if is_alternative_query(q) and conversation_id and self.shortlists.get(conversation_id):
            self.metrics.increment("recommend_alternative_requests")
            alt_hit, alt_idx = self._next_alternative_hit(conversation_id)
            if alt_hit:
                reason_query = prior_task or q
                single_hit = enrich_hit(dict(alt_hit), reason_query)
                message = alternative_message(single_hit, reason_query)
                self.conversations.append(conversation_id, "assistant", message)
                self._set_shortlist_pointer(conversation_id, alt_idx)
                return {"hits": [single_hit], "message": message}
            message = no_more_alternatives_message()
            self.conversations.append(conversation_id, "assistant", message)
            return {"hits": [], "message": message}

        # "Which one of these is the best?" -> pick the single best from the
        # shortlist we already returned for this conversation, if we have one.
        pick_best = is_pick_best_query(q)
        if pick_best and conversation_id and self.shortlists.get(conversation_id):
            self.metrics.increment("recommend_pick_best_requests")
            prior_hits = self.shortlists[conversation_id]
            best_hit = prior_hits[0] if prior_hits else None
            if best_hit:
                reason_query = prior_task or q
                single_hit = enrich_hit(dict(best_hit), reason_query)
                message = recommendation_message([single_hit], reason_query, MODE_ONE_BEST, pick_best=True)
                self.conversations.append(conversation_id, "assistant", message)
                self._set_shortlist_pointer(conversation_id, 0)
                return {"hits": [single_hit], "message": message}

        # No stored shortlist: a referential pick-best question is about the
        # currently visible cards, so do not launch a fresh catalogue search.
        if pick_best:
            residual = strip_pick_best_clause(q)
            if is_referential_pick(q) and residual and query_terms(residual):
                prior_task = residual
            elif is_referential_pick(q):
                message = pick_best_needs_shortlist_message()
                self.conversations.append(conversation_id, "assistant", message)
                return {"hits": [], "message": message}
            if not prior_task:
                if residual and query_terms(residual):
                    prior_task = residual
            if not prior_task and prior_messages:
                prior_task = prior_messages[-1]

        if pick_best and prior_task:
            self.metrics.increment("recommend_pick_best_requests")
            q = prior_task
            mode = MODE_ONE_BEST
            prior_messages = []
        else:
            # Self-contained query (e.g. "what's the best coding tool"); treat normally.
            pick_best = False
            if has_explicit_task(q):
                prior_messages = []

        retrieval_query = build_retrieval_query(q, prior_messages)

        if requires_free_only(retrieval_query) and not non_filter_terms(retrieval_query):
            self.metrics.increment("recommend_free_query_missing_task")
            return {
                "hits": [],
                "message": "What task should the free tool help with?",
            }
        if not query_terms(retrieval_query):
            self.metrics.increment("recommend_empty_query_blocked")
            return {
                "hits": [],
                "message": "Tell me the task, budget, or must-have integrations and I will search for better matches.",
            }
        free_only = requires_free_only(retrieval_query)
        if free_only and filters is None:
            filters = {"budget": "free"}

        effective_final_k = final_k
        if mode == MODE_ONE_BEST:
            effective_final_k = 1
        elif mode == MODE_COMPARE:
            effective_final_k = min(max(final_k, 3), 10)

        filter_key = json.dumps(filters.model_dump() if hasattr(filters, "model_dump") else (filters or {}), sort_keys=True)
        cache_key = (
            f"{TEXT_FORMAT_VERSION}:{self.settings.emb_model}:{self.settings.chat_model}:"
            f"{retrieve_k}:{effective_final_k}:{mode}:{filter_key}:{retrieval_query}"
        )
        cached = self.recommend_cache.get(cache_key)
        if cached is not None:
            self.metrics.increment("recommend_cache_hit")
            if conversation_id:
                self.shortlists[conversation_id] = cached
                self.shortlist_pointers[conversation_id] = 0
            message = recommendation_message(cached, q, mode, pick_best=pick_best)
            self.conversations.append(conversation_id, "assistant", message)
            return {"hits": cached, "message": message}

        try:
            vec = self.embed([retrieval_query])
        except Exception as exc:
            logger.info("Embedding unavailable; served keyword recommendation fallback (%s)", type(exc).__name__)
            self.metrics.increment("embedding_fallbacks")
            hits = keyword_search(retrieval_query, min(retrieve_k, len(self.store.meta)), self.store.meta)
            hits = apply_decision_filters(hits, filters, self.store.meta)[:effective_final_k] or hits[:effective_final_k]
            hits = [enrich_hit(hit, q) for hit in hits]
            self.recommend_cache.set(cache_key, hits)
            if conversation_id:
                self.shortlists[conversation_id] = hits
                self.shortlist_pointers[conversation_id] = 0
            message = recommendation_message(hits, q, mode, pick_best=pick_best)
            self.conversations.append(conversation_id, "assistant", message)
            return {"hits": hits, "message": message}

        with self.metrics.timer("faiss.recommend_search_ms"):
            scores, ids = self.store.index.search(vec, min(retrieve_k, len(self.store.meta)))

        candidates = self._build_candidates(scores[0].tolist(), ids[0].tolist())
        filtered_candidates = apply_decision_filters(candidates, filters, self.store.meta)
        if len(filtered_candidates) >= effective_final_k:
            candidates = filtered_candidates
        elif free_only:
            candidates = [
                candidate for candidate in candidates
                if is_free_tool(self.store.meta[int(candidate["id"])])
            ]
        if not candidates:
            return {"hits": [], "message": recommendation_message([], q, mode, pick_best=pick_best)}

        candidate_embeddings = self._candidate_embeddings([int(c["id"]) for c in candidates])
        with self.metrics.timer("rerank.mmr_ms"):
            candidates = mmr_rerank(
                candidates,
                candidate_embeddings,
                lambda_=self.settings.mmr_lambda,
                top_k=retrieve_k,
            )

        selected = self._rank_with_llm(retrieval_query, candidates, effective_final_k, mode=mode)
        final_hits = self._selected_hits(
            selected,
            q,
            effective_final_k,
            allowed_ids={int(candidate["id"]) for candidate in candidates},
        )

        if not final_hits:
            top = [
                (i, s)
                for i, s in zip(ids[0].tolist(), scores[0].tolist())
                if i != -1 and (not free_only or is_free_tool(self.store.meta[i]))
            ]
            final_hits = [
                enrich_hit(
                    {
                        "score": float(s),
                        "meta": compact_meta(self.store.meta[i]),
                        "why": local_reason(q, self.store.meta[i]),
                    },
                    q,
                )
                for i, s in top[:effective_final_k]
            ]
            self.metrics.increment("llm_rank_fallbacks")

        self.recommend_cache.set(cache_key, final_hits)
        if conversation_id:
            self.shortlists[conversation_id] = final_hits
            self.shortlist_pointers[conversation_id] = 0
        message = recommendation_message(final_hits, q, mode, pick_best=pick_best)
        self.conversations.append(conversation_id, "assistant", message)
        return {"hits": final_hits, "message": message}

    def chat(
        self,
        q: str,
        retrieve_k: int,
        final_k: int,
        filters: Any = None,
        mode: str = "balanced",
        conversation_id: str | None = None,
        history: Any = None,
        visible_tools: Any = None,
    ) -> dict[str, Any]:
        self.metrics.increment("chat_requests")
        q = normalize_query_text(q)
        mode = normalize_mode(mode)

        provided_hits = visible_tool_hits(visible_tools)
        if conversation_id and provided_hits:
            self.shortlists[conversation_id] = provided_hits
            self.shortlist_pointers.setdefault(conversation_id, 0)

        decision = self._chat_decision(q, filters, mode, conversation_id, history)
        action = decision.get("action", "recommend")
        if action not in CHAT_ACTIONS:
            action = "recommend"

        if action == "chat_only":
            message = clean_assistant_message(decision.get("message") or non_search_response(q))
            self.conversations.append(conversation_id, "user", q)
            self.conversations.append(conversation_id, "assistant", message)
            return {"action": "chat_only", "hits": [], "message": message}

        if action == "clarify":
            message = clean_assistant_message(
                decision.get("message")
                or decision.get("question")
                or default_clarifying_question(q)
            )
            self.conversations.append(conversation_id, "user", q)
            self.conversations.append(conversation_id, "assistant", message)
            return {"action": "clarify", "hits": [], "message": message}

        if action in {"explain_shortlist", "explain_best", "pick_best"}:
            response = self._chat_explain(action, q, conversation_id, history)
            return {
                "action": "pick_best" if action == "pick_best" else "explain",
                **response,
            }

        if action == "show_alternative":
            response = self._chat_alternative(q, conversation_id, history)
            return {"action": "show_alternative", **response}

        if action == "tool_question":
            response = self.recommend(
                q,
                retrieve_k,
                final_k,
                filters=filters,
                mode=mode,
                conversation_id=conversation_id,
                history=history,
            )
            return {"action": "refine", **response}

        refined_query = normalize_query_text(decision.get("refined_query") or q)
        next_filters = self._merge_chat_filters(filters, decision.get("filters"))
        next_mode = normalize_mode(decision.get("mode") or mode)
        if action == "recommend":
            next_mode = next_mode or mode
        response = self.recommend(
            refined_query,
            retrieve_k,
            final_k,
            filters=next_filters,
            mode=next_mode,
            conversation_id=conversation_id,
            history=history,
        )
        return {
            "action": "refine" if action == "refine" else "recommend",
            "refined_query": refined_query,
            **response,
        }

    def _chat_explain(
        self,
        action: str,
        q: str,
        conversation_id: str | None,
        history: Any,
    ) -> dict[str, Any]:
        prior_hits = self.shortlists.get(conversation_id) if conversation_id else None
        self.conversations.append(conversation_id, "user", q)
        if not prior_hits:
            message = "I can explain the current tools after a search, but I need the visible results first."
            self.conversations.append(conversation_id, "assistant", message)
            return {"hits": [], "message": message}

        reason_query = self._latest_task_query(conversation_id, history) or q
        if action == "explain_shortlist":
            explained_hits = [
                enrich_hit(dict(hit), reason_query)
                for hit in prior_hits[: min(3, len(prior_hits))]
            ]
            for explained in explained_hits:
                explained["why"] = local_reason(reason_query, explained.get("meta") or {})
            message = shortlist_explanation_message(explained_hits, reason_query)
            self.conversations.append(conversation_id, "assistant", message)
            return {"hits": explained_hits, "message": message}

        top = enrich_hit(dict(prior_hits[0]), reason_query)
        top["why"] = local_reason(reason_query, top.get("meta") or {})
        message = recommendation_message([top], reason_query, MODE_ONE_BEST, pick_best=True)
        self._set_shortlist_pointer(conversation_id, 0)
        self.conversations.append(conversation_id, "assistant", message)
        return {"hits": [top], "message": message}

    def _chat_alternative(
        self,
        q: str,
        conversation_id: str | None,
        history: Any,
    ) -> dict[str, Any]:
        self.conversations.append(conversation_id, "user", q)
        alt_hit, alt_idx = self._next_alternative_hit(conversation_id)
        if not alt_hit:
            message = no_more_alternatives_message()
            self.conversations.append(conversation_id, "assistant", message)
            return {"hits": [], "message": message}

        reason_query = self._latest_task_query(conversation_id, history) or q
        single_hit = enrich_hit(dict(alt_hit), reason_query)
        message = alternative_message(single_hit, reason_query)
        self._set_shortlist_pointer(conversation_id, alt_idx)
        self.conversations.append(conversation_id, "assistant", message)
        return {"hits": [single_hit], "message": message}

    def _latest_task_query(self, conversation_id: str | None, history: Any) -> str:
        stored = self.conversations.get(conversation_id)
        prior_messages = merge_history_messages(history, stored)
        return next((m for m in reversed(prior_messages) if has_explicit_task(m)), "")

    def _merge_chat_filters(self, filters: Any, decision_filters: Any) -> Any:
        base = filters.model_dump() if hasattr(filters, "model_dump") else dict(filters or {})
        if isinstance(decision_filters, dict):
            for key in ("budget", "privacy", "integrations", "categories", "platforms", "skill_level"):
                value = decision_filters.get(key)
                if value not in (None, "", []):
                    base[key] = value
        return base or filters

    def _chat_decision(
        self,
        q: str,
        filters: Any,
        mode: str,
        conversation_id: str | None,
        history: Any,
    ) -> dict[str, Any]:
        stored = self.conversations.get(conversation_id)
        prior_messages = merge_history_messages(history, stored)
        prior_hits = self.shortlists.get(conversation_id) if conversation_id else []
        has_shortlist = bool(prior_hits)

        # High-confidence follow-ups should not be allowed to become fresh searches
        # just because the model classified them too broadly.
        if has_shortlist and is_shortlist_explanation_query(q):
            return {"action": "explain_shortlist"}
        if has_shortlist and is_explanation_query(q):
            return {"action": "explain_best"}
        if has_shortlist and is_pick_best_query(q):
            return {"action": "pick_best"}
        if has_shortlist and is_alternative_query(q):
            return {"action": "show_alternative"}
        if has_shortlist and (is_specific_tool_query(q) or is_criterion_pick_query(q)):
            return {"action": "tool_question"}

        context = {
            "latest_user_message": q,
            "recent_user_messages": prior_messages[-6:],
            "visible_tools": [compact_hit_for_prompt(hit) for hit in (prior_hits or [])[:5]],
            "filters": filters.model_dump() if hasattr(filters, "model_dump") else (filters or {}),
            "mode": mode,
        }
        system = (
            "You are the conversation brain for an AI tool advisor app. "
            "The app should feel like a GPT wrapper: the user sends one natural chat message, "
            "then the backend decides whether to talk, ask one clarification, search, refine, explain, "
            "pick the best option, or answer from visible tool cards. "
            "Classify the latest user message using the conversation history and visible tools. "
            "When tools are needed, the backend will run FAISS vector retrieval, MMR diversification, "
            "and RAG ranking over the catalog before returning structured tool cards. "
            "Your job here is routing only; do not invent tool results. "
            "Do not try to predict exact wording with rules; infer the user's intent naturally.\n"
            "Actions:\n"
            "- chat_only: greetings, thanks, app-help, feedback, or normal conversation that should not change tool cards.\n"
            "- clarify: the user wants tools but the task is missing.\n"
            "- recommend: a new tool search.\n"
            "- refine: change filters or rerank the current search, e.g. free-only, cheaper, more private.\n"
            "- explain_shortlist: user asks why these/current tools were shown.\n"
            "- explain_best or pick_best: user asks which one you would choose or which is best.\n"
            "- show_alternative: user asks for another option from the current list.\n"
            "- tool_question: user asks about a specific visible tool or property, e.g. is it free, what about X.\n"
            "Return ONLY JSON with keys: action, message, refined_query, filters, mode. "
            "Use message for chat_only/clarify only. Never write 'Consultant view'."
        )
        try:
            with self.metrics.timer("openai.chat_decision_ms"):
                resp = self.client.chat.completions.create(
                    model=self.settings.chat_model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
            data = json.loads(resp.choices[0].message.content or "{}")
            action = data.get("action")
            if action in CHAT_ACTIONS:
                self.metrics.increment("chat_decision_model_used")
                return data
            self.metrics.increment("chat_decision_invalid")
        except Exception:
            self.metrics.increment("chat_decision_fallbacks")

        return self._fallback_chat_decision(q, filters, mode, conversation_id, history)

    def _fallback_chat_decision(
        self,
        q: str,
        filters: Any,
        mode: str,
        conversation_id: str | None,
        history: Any,
    ) -> dict[str, Any]:
        has_shortlist = bool(conversation_id and self.shortlists.get(conversation_id))
        if is_non_search_message(q):
            return {"action": "chat_only", "message": non_search_response(q)}
        if is_feedback_only_query(q):
            return {"action": "clarify", "message": feedback_clarifying_question()}
        if has_shortlist and is_shortlist_explanation_query(q):
            return {"action": "explain_shortlist"}
        if has_shortlist and is_explanation_query(q):
            return {"action": "explain_best"}
        if has_shortlist and is_pick_best_query(q):
            return {"action": "pick_best"}
        if has_shortlist and is_alternative_query(q):
            return {"action": "show_alternative"}
        if has_shortlist and (is_specific_tool_query(q) or is_criterion_pick_query(q)):
            return {"action": "tool_question"}

        focused = focus_latest_intent(q)
        if has_explicit_task(focused):
            return {"action": "recommend", "refined_query": focused}

        prior_task = self._latest_task_query(conversation_id, history)
        if requires_free_only(focused) and not non_filter_terms(focused):
            if not prior_task:
                return {"action": "clarify", "message": "What task should the free tool help with?"}
            return {
                "action": "refine",
                "refined_query": build_retrieval_query(focused, [prior_task]),
                "filters": {"budget": "free"},
            }

        context_query = build_retrieval_query(focused, [prior_task] if prior_task else [])
        if requires_free_only(context_query) and not non_filter_terms(context_query):
            return {"action": "clarify", "message": "What task should the free tool help with?"}
        if needs_clarification(context_query):
            return {"action": "clarify", "message": default_clarifying_question(focused)}
        return {"action": "recommend", "refined_query": context_query}

    def clarify(self, q: str, conversation_id: str | None = None, history: Any = None) -> dict[str, Any]:
        if is_non_search_message(q):
            return {"action": "clarify", "question": non_search_response(q)}
        if is_feedback_only_query(q):
            return {"action": "clarify", "question": feedback_clarifying_question()}
        if is_explanation_query(q):
            return {"action": "explain"}
        if is_pick_best_query(q) and is_referential_pick(q):
            residual = strip_pick_best_clause(q)
            if residual and query_terms(residual):
                return {"action": "search", "refined_query": residual}
            return {"action": "explain"}
        # Follow-up questions that reference the existing shortlist must never trigger
        # a clarifying question; let the recommend step answer from the stored shortlist.
        has_shortlist = bool(conversation_id and self.shortlists.get(conversation_id))
        if has_shortlist and (
            is_pick_best_query(q)
            or is_specific_tool_query(q)
            or is_criterion_pick_query(q)
            or is_alternative_query(q)
        ):
            return {"action": "search", "refined_query": q}
        q = strip_instruction_text(q)
        q = focus_latest_intent(q)

        stored = self.conversations.get(conversation_id)
        prior_messages = merge_history_messages(history, stored)
        if has_explicit_task(q):
            prior_messages = []
        context_query = build_retrieval_query(q, prior_messages)

        # With prior task context, a short follow-up no longer needs clarification.
        if needs_clarification(context_query):
            return {"action": "clarify", "question": default_clarifying_question(q)}

        system = (
            "Decide if the user's request needs ONE clarifying question.\n"
            "You may be given earlier conversation context; use it so follow-up messages are understood in context.\n"
            "If missing key info (task type, platform, free/paid, output), ask 1-3 short question(s).\n"
            "Otherwise rewrite the request into a single refined query.\n"
            "Return JSON only like:\n"
            '{"action":"clarify","question":"...","refined_query":null}\n'
            'or {"action":"search","question":null,"refined_query":"..."}'
        )
        user_content = q if context_query == q else f"Conversation so far: {context_query}\nLatest message: {q}"

        try:
            with self.metrics.timer("openai.clarify_ms"):
                resp = self.client.chat.completions.create(
                    model=self.settings.chat_model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=0.2,
                    response_format={"type": "json_object"},
                )
            raw = resp.choices[0].message.content or ""
            data = json.loads(raw)
        except Exception:
            self.metrics.increment("clarify_fallbacks")
            if needs_clarification(context_query):
                return {"action": "clarify", "question": default_clarifying_question(q)}
            return {"action": "search", "refined_query": context_query}

        action = data.get("action")
        if action == "clarify":
            question = (data.get("question") or "").strip()
            if not question:
                question = "Quick question: what exact task are you trying to do, and do you need a free tool?"
            return {"action": "clarify", "question": question}

        refined = (data.get("refined_query") or q).strip()
        return {"action": "search", "refined_query": refined}

    def detect_intent(
        self,
        prompt: str,
        last_query: str,
        conversation_id: str | None = None,
        history: Any = None,
    ) -> dict[str, str]:
        if is_non_search_message(prompt):
            return {"intent": "new"}
        if is_feedback_only_query(prompt):
            return {"intent": "new"}

        stored = self.conversations.get(conversation_id)
        prior_messages = merge_history_messages(history, stored)
        # Fall back to remembered context when the client does not pass last_query.
        if not last_query and prior_messages:
            last_query = prior_messages[-1]
        has_shortlist = bool(conversation_id and self.shortlists.get(conversation_id))
        has_context = bool(last_query) or bool(prior_messages) or has_shortlist

        # "Which one of these is best?" / "What about Claude?" / "Which is cheapest?"
        # / "Is there any other tool?" are all follow-ups on the existing shortlist.
        if has_context and (is_criterion_pick_query(prompt) or is_specific_tool_query(prompt) or is_alternative_query(prompt)):
            return {"intent": "refine"}

        if has_context and (
            is_explanation_query(prompt)
            or (is_pick_best_query(prompt) and (is_referential_pick(prompt) or has_shortlist))
        ):
            return {"intent": "explain"}

        if has_context and (
            is_pick_best_query(prompt)
        ):
            return {"intent": "refine"}

        # An explicit pivot or a message that states its own task is a NEW search,
        # so the client should not prepend the previous query to it.
        focused = focus_latest_intent(strip_instruction_text(prompt))
        pivoted = focused.lower() != normalize_query_text(prompt).lower()
        if pivoted or has_explicit_task(focused):
            return {"intent": "new"}

        if has_context and is_refinement_query(prompt):
            return {"intent": "refine"}

        system = (
            "You are a search intent classifier. Given the previous search query, decide if the user is:\n"
            "- 'explain': asking why the current results were recommended, for example 'why these tools?' or 'explain these picks'\n"
            "- 'refine': modifying, filtering or following up on the previous search, for example "
            "'free only', 'show me more', 'I need something simpler', 'what about paid ones', "
            "'what about Claude?', 'which one is the cheapest?'\n"
            "- 'new': asking for something completely different\n"
            "Return ONLY valid JSON: {\"intent\": \"explain\"}, {\"intent\": \"refine\"}, or {\"intent\": \"new\"}"
        )
        try:
            with self.metrics.timer("openai.detect_intent_ms"):
                resp = self.client.chat.completions.create(
                    model=self.settings.chat_model,
                    messages=[
                        {"role": "system", "content": system},
                        {
                            "role": "user",
                            "content": f"Previous search: {last_query}\nNew message: {prompt}",
                        },
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
            data = json.loads(resp.choices[0].message.content)
            intent = data.get("intent", "new")
            if intent not in ("explain", "refine", "new"):
                intent = "new"
            return {"intent": intent}
        except Exception:
            self.metrics.increment("detect_intent_fallbacks")
            return {"intent": "new"}

    def cache_stats(self) -> dict[str, Any]:
        return {
            "embedding_cache": self.embedding_cache.stats(),
            "recommend_cache": self.recommend_cache.stats(),
        }

    def _build_candidates(self, scores: list[float], ids: list[int]) -> list[dict[str, Any]]:
        candidates = []
        for score, id_ in zip(scores, ids):
            if id_ == -1:
                continue
            candidates.append(self._candidate_for_id(id_, score))
        return candidates

    def _candidate_for_id(self, id_: int, score: float) -> dict[str, Any]:
        m = self.store.meta[id_]
        return {
            "id": id_,
            "score": float(score),
            "name": m.get("Name", ""),
            "categories": m.get("Categories", ""),
            "price": m.get("Price", ""),
            "description": m.get("Description", ""),
            "features": m.get("Features", ""),
            "use_cases": m.get("Use_cases", ""),
            "pros": m.get("Pros", ""),
        }

    def _find_hit_by_name(self, hits: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
        """Return the first hit in the shortlist whose name matches the query."""
        for hit in hits:
            meta = hit.get("meta") or {}
            if _matches_tool_name(meta, name):
                return hit
        return None

    def _find_tool_by_name(self, name: str) -> dict[str, Any] | None:
        """Search the full catalog for a tool whose name matches the query."""
        for meta in self.store.meta:
            if _matches_tool_name(meta, name):
                return meta
        return None

    def _set_shortlist_pointer(self, conversation_id: str | None, index: int) -> None:
        if not conversation_id:
            return
        self.shortlist_pointers[conversation_id] = max(0, index)

    def _next_alternative_hit(
        self,
        conversation_id: str | None,
    ) -> tuple[dict[str, Any] | None, int]:
        """Return the next hit from the shortlist that has not been shown yet."""
        if not conversation_id:
            return None, -1
        prior_hits = self.shortlists.get(conversation_id)
        if not prior_hits:
            return None, -1
        last_idx = self.shortlist_pointers.get(conversation_id, -1)
        next_idx = last_idx + 1
        if next_idx >= len(prior_hits):
            return None, -1
        return prior_hits[next_idx], next_idx

    def _candidate_embeddings(self, ids: list[int]) -> np.ndarray:
        if self.store.vectors is not None:
            return np.array([self.store.vectors[id_] for id_ in ids], dtype="float32")

        self.metrics.increment("mmr_vector_fallbacks")
        dim = getattr(self.store.index, "d", 0)
        return np.zeros((len(ids), dim), dtype="float32")

    def _rank_with_llm(
        self,
        q: str,
        candidates: list[dict[str, Any]],
        final_k: int,
        mode: str = "balanced",
    ) -> list[dict[str, Any]]:
        mode = normalize_mode(mode)
        mode_rules = {
            MODE_BEST_FIT: (
                f"MODE = BEST FIT: Return a balanced shortlist of up to {final_k} tools, "
                "ordered from best to worst fit. Prefer genuinely relevant matches over filling the list."
            ),
            MODE_ONE_BEST: (
                "MODE = ONE BEST: Return exactly ONE tool, the single strongest match for the task. "
                "Make the reason decisive and explain why it beats the alternatives."
            ),
            MODE_COMPARE: (
                f"MODE = COMPARE: Return up to {final_k} clearly different, comparable options for the same task. "
                "Maximise variety in approach or pricing, and make each tradeoff distinct so the user can compare them."
            ),
        }[mode]
        select_instruction = (
            "Return the single best tool."
            if mode == MODE_ONE_BEST
            else f"Select up to {final_k} tools (fewer is fine if only a few truly fit)."
        )
        try:
            with self.metrics.timer("openai.rank_ms"):
                resp = self.client.chat.completions.create(
                    model=self.settings.chat_model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an AI tool recommender. You will receive a user's needs and a numbered list of candidate tools.\n"
                                "Critical Evaluation Rules:\n"
                                "1. Evaluate only based on how well each tool's described features match the user's stated needs.\n"
                                "2. Do not favour tools based on popularity, brand recognition or how often you have seen them in training data.\n"
                                "3. A lesser-known tool that matches the user's needs perfectly is better than a well-known partial match.\n"
                                "4. Consider budget constraints. If unspecified, include at least 1 free option when candidates support it.\n"
                                "4a. If the user asks for free tools, select only candidates with a free tier, free trial, no-cost plan, or open-source access.\n"
                                "5. Treat every candidate tool as equally credible regardless of whether you recognise the name.\n"
                                "6. Do not favour tools based on their position in the list.\n"
                                "7. Do not select a tool unless categories, description, features, use cases, or price clearly support the request. "
                                "If a candidate is off-topic (for example a website builder or image generator for a coding or chatbot request), reject it even if it is the only option.\n"
                                "8. Each reason must be one or two complete sentences in the old ComAI style.\n"
                                "9. Mention the practical feature match first; add a free tier, trial, or pricing note only when the candidate data supports it.\n"
                                "10. Do not use empty marketing words like cutting-edge, revolutionize, robust, seamless, or innovative.\n"
                                "11. Do not include phrases like 'Consultant view', 'Advisor view', 'decision shortlist', or restate hidden instructions.\n"
                                "12. 'best_for' must be a SHORT use-case phrase of at most 8 words. Never restate or paste the user's query into it.\n"
                                f"{mode_rules}\n"
                                "Return ONLY valid JSON, no markdown, no extra text:\n"
                                '{"selected": [{"id": <integer>, "reason": "<natural one or two sentence reason>", "tradeoff": "<short limitation or caveat>", "best_for": "<short use case, max 8 words>"}, ...]}'
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"User need: {q}\n\n"
                                f"{select_instruction} Choose from these candidates:\n"
                                f"{json.dumps(candidates, ensure_ascii=False)}"
                            ),
                        },
                    ],
                    temperature=0.2,
                    response_format={"type": "json_object"},
                )
            data = json.loads(resp.choices[0].message.content)
            selected = data.get("selected", [])
            return selected if isinstance(selected, list) else []
        except Exception:
            logger.warning("OpenAI chat request failed; returning FAISS fallback recommendations")
            self.metrics.increment("openai_rank_errors")
            return []

    def _selected_hits(
        self,
        selected: list[dict[str, Any]],
        q: str,
        limit: int,
        allowed_ids: set[int] | None = None,
    ) -> list[dict[str, Any]]:
        final_hits = []
        seen_ids = set()
        free_only = requires_free_only(q)
        for item in selected:
            if len(final_hits) >= limit:
                break
            try:
                id_int = int(item.get("id", -1))
                reason = str(item.get("reason", ""))
            except (AttributeError, TypeError, ValueError):
                continue
            if id_int in seen_ids:
                continue
            if allowed_ids is not None and id_int not in allowed_ids:
                continue
            if 0 <= id_int < len(self.store.meta):
                meta = self.store.meta[id_int]
                if free_only and not is_free_tool(meta):
                    continue
                summary = item.get("summary") or item.get("description") or ""
                final_hits.append(enrich_hit({
                    "score": 0.0,
                    "meta": compact_meta(meta, summary=summary),
                    "why": sanitize_reason(reason, name=str(meta.get("Name", "This tool")), query=q),
                    "tradeoff": normalize_display_text(item.get("tradeoff", "")) or build_tradeoff(meta),
                    "best_for": clean_best_for(item.get("best_for", ""), q, meta),
                }, q))
                seen_ids.add(id_int)
        return final_hits
