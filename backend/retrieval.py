from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any

import faiss
import numpy as np

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

FREE_FILTER_WORDS = {
    "free", "only", "ones", "one", "option", "options", "tool", "tools",
    "app", "apps", "said", "tier", "trial", "plan", "plans",
}


@dataclass
class ToolStore:
    index: Any
    meta: list[dict[str, Any]]
    vectors: np.ndarray | None

    @property
    def ready(self) -> bool:
        return self.index is not None and bool(self.meta)


def load_tool_store(settings: Settings) -> ToolStore:
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
        faiss.normalize_L2(vectors)
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
    faiss.normalize_L2(vectors)
    return vectors


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


def is_explanation_query(text: str) -> bool:
    normalized = normalize_query_text(text).lower().strip()
    return bool(re.search(
        r"\b(why\s+(?:these|this|those)(?:\s+(?:tools?|ones|results?|recommendations?|picks?))?\??|why\s+did|explain(?:\s+(?:these|this|those|the))?|reason|reasons|why\s+recommended|why\s+recommend)\b",
        normalized,
    ))


def is_refinement_query(text: str) -> bool:
    normalized = text.lower().strip()
    return bool(re.search(
        r"\b(free only|free|paid|cheaper|simpler|more specific|more privacy|private|secure|local|self-hosted|show me more|another|alternatives?|compare|better|only)\b",
        normalized,
    ))


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
    if is_writing_query(q):
        allowed = {"writing", "generators", "copywriting", "seo", "marketing", "social", "media"}
        blocked = {"fitness", "health", "travel", "dating", "music", "finance", "stock", "trading"}
        if category_tokens & blocked:
            return True
        return not bool(category_tokens & allowed)
    return False


def request_goal(q: str) -> str:
    cleaned = strip_instruction_text(q)
    text = cleaned.lower()
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
    if "presentation" in text or "slides" in text:
        return "creating presentations"
    if "code" in text or "debug" in text or "python" in text:
        return "coding and debugging"
    if "research" in text or "competitor" in text:
        return "researching information"
    return cleaned.strip().rstrip(".") or "your task"


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
    if "transcribing audio" in goal:
        return "it is designed to convert audio into text or meeting notes"
    if "generating images" in goal:
        return "it generates images or visual assets from prompts"
    if "creating presentations" in goal:
        return "it helps create slides or presentation content faster"
    if "coding and debugging" in goal:
        return "it assists with writing, completing, or debugging code"
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


def compact_meta(meta: dict[str, Any], summary: Any = None) -> dict[str, Any]:
    return dict(meta)


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


def recommendation_message(hits: list[dict[str, Any]], query: str = "") -> str:
    names = [
        str((hit.get("meta") or {}).get("Name", "")).strip()
        for hit in hits
        if (hit.get("meta") or {}).get("Name")
    ]
    names = [name for name in names if name]
    if not names:
        if requires_free_only(query):
            return "I could not find a strong free match. Try changing the task or allowing free trials and freemium plans."
        return "I could not find a strong match. Try adding the task, budget, and any must-have integrations."

    first = names[0]
    if len(names) == 1:
        return f"Start with {first}. It looks like the strongest match from the current catalogue."

    second = names[1]
    return f"Start with {first}. Compare it with {second} if you want another good option."


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
        faiss.normalize_L2(vecs)

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
    ) -> dict[str, Any]:
        self.metrics.increment("recommend_requests")
        if is_feedback_only_query(q):
            self.metrics.increment("recommend_feedback_query_blocked")
            return {
                "hits": [],
                "message": feedback_clarifying_question(),
            }
        if is_explanation_query(q):
            self.metrics.increment("recommend_explain_query_blocked")
            return {
                "hits": [],
                "message": "I can explain the current shortlist after a search, but I need the previous results to do that.",
            }
        q = strip_instruction_text(q)
        if requires_free_only(q) and not non_filter_terms(q):
            self.metrics.increment("recommend_free_query_missing_task")
            return {
                "hits": [],
                "message": "What task should the free tool help with?",
            }
        if not query_terms(q):
            self.metrics.increment("recommend_empty_query_blocked")
            return {
                "hits": [],
                "message": "Tell me the task, budget, or must-have integrations and I will search for better matches.",
            }
        free_only = requires_free_only(q)
        if free_only and filters is None:
            filters = {"budget": "free"}

        filter_key = json.dumps(filters.model_dump() if hasattr(filters, "model_dump") else (filters or {}), sort_keys=True)
        cache_key = f"{TEXT_FORMAT_VERSION}:{self.settings.emb_model}:{self.settings.chat_model}:{retrieve_k}:{final_k}:{mode}:{filter_key}:{q}"
        cached = self.recommend_cache.get(cache_key)
        if cached is not None:
            self.metrics.increment("recommend_cache_hit")
            return {"hits": cached, "message": recommendation_message(cached, q)}

        try:
            vec = self.embed([q])
        except Exception as exc:
            logger.info("Embedding unavailable; served keyword recommendation fallback (%s)", type(exc).__name__)
            self.metrics.increment("embedding_fallbacks")
            candidates = [
                self._candidate_for_id(idx, score)
                for score, idx in keyword_scores(q, min(retrieve_k, len(self.store.meta)), self.store.meta)
            ]
            filtered_candidates = apply_decision_filters(candidates, filters, self.store.meta)
            if len(filtered_candidates) >= final_k:
                candidates = filtered_candidates
            selected = self._rank_with_llm(q, candidates, final_k, mode=mode) if candidates else []
            hits = self._selected_hits(
                selected,
                q,
                final_k,
                allowed_ids={int(candidate["id"]) for candidate in candidates},
            )
            if not hits:
                hits = keyword_search(q, min(retrieve_k, len(self.store.meta)), self.store.meta)
                hits = apply_decision_filters(hits, filters, self.store.meta)[:final_k] or hits[:final_k]
                hits = [enrich_hit(hit, q) for hit in hits]
            self.recommend_cache.set(cache_key, hits)
            return {"hits": hits, "message": recommendation_message(hits, q)}

        with self.metrics.timer("faiss.recommend_search_ms"):
            scores, ids = self.store.index.search(vec, min(retrieve_k, len(self.store.meta)))

        candidates = self._build_candidates(scores[0].tolist(), ids[0].tolist())
        filtered_candidates = apply_decision_filters(candidates, filters, self.store.meta)
        if len(filtered_candidates) >= final_k:
            candidates = filtered_candidates
        elif free_only:
            candidates = [
                candidate for candidate in candidates
                if is_free_tool(self.store.meta[int(candidate["id"])])
            ]
        if not candidates:
            return {"hits": [], "message": recommendation_message([], q)}

        candidate_embeddings = self._candidate_embeddings([int(c["id"]) for c in candidates])
        with self.metrics.timer("rerank.mmr_ms"):
            candidates = mmr_rerank(
                candidates,
                candidate_embeddings,
                lambda_=self.settings.mmr_lambda,
                top_k=retrieve_k,
            )

        selected = self._rank_with_llm(q, candidates, final_k, mode=mode)
        final_hits = self._selected_hits(
            selected,
            q,
            final_k,
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
                for i, s in top[:final_k]
            ]
            self.metrics.increment("llm_rank_fallbacks")

        self.recommend_cache.set(cache_key, final_hits)
        return {"hits": final_hits, "message": recommendation_message(final_hits, q)}

    def clarify(self, q: str) -> dict[str, Any]:
        if is_feedback_only_query(q):
            return {"action": "clarify", "question": feedback_clarifying_question()}
        if is_explanation_query(q):
            return {"action": "explain"}
        q = strip_instruction_text(q)
        if needs_clarification(q):
            return {"action": "clarify", "question": default_clarifying_question(q)}

        system = (
            "Decide if the user's request needs ONE clarifying question.\n"
            "If missing key info (task type, platform, free/paid, output), ask 1-3 short question(s).\n"
            "Otherwise rewrite the request into a single refined query.\n"
            "Return JSON only like:\n"
            '{"action":"clarify","question":"...","refined_query":null}\n'
            'or {"action":"search","question":null,"refined_query":"..."}'
        )

        try:
            with self.metrics.timer("openai.clarify_ms"):
                resp = self.client.chat.completions.create(
                    model=self.settings.chat_model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": q},
                    ],
                    temperature=0.2,
                    response_format={"type": "json_object"},
                )
            raw = resp.choices[0].message.content or ""
            data = json.loads(raw)
        except Exception:
            self.metrics.increment("clarify_fallbacks")
            if needs_clarification(q):
                return {"action": "clarify", "question": default_clarifying_question(q)}
            return {"action": "search", "refined_query": q}

        action = data.get("action")
        if action == "clarify":
            question = (data.get("question") or "").strip()
            if not question:
                question = "Quick question: what exact task are you trying to do, and do you need a free tool?"
            return {"action": "clarify", "question": question}

        refined = (data.get("refined_query") or q).strip()
        return {"action": "search", "refined_query": refined}

    def detect_intent(self, prompt: str, last_query: str) -> dict[str, str]:
        if is_feedback_only_query(prompt):
            return {"intent": "new"}
        if last_query and is_explanation_query(prompt):
            return {"intent": "explain"}
        if last_query and is_refinement_query(prompt):
            return {"intent": "refine"}

        system = (
            "You are a search intent classifier. Given the previous search query, decide if the user is:\n"
            "- 'explain': asking why the current results were recommended, for example 'why these tools?' or 'explain these picks'\n"
            "- 'refine': modifying, filtering or following up on the previous search, for example "
            "'free only', 'show me more', 'I need something simpler', 'what about paid ones'\n"
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
                                "7. Do not select a tool unless categories, description, features, use cases, or price clearly support the request.\n"
                                "8. Each reason must be one or two complete sentences in the old ComAI style.\n"
                                "9. Mention the practical feature match first; add a free tier, trial, or pricing note only when the candidate data supports it.\n"
                                "10. Do not use empty marketing words like cutting-edge, revolutionize, robust, seamless, or innovative.\n"
                                "11. Do not include phrases like 'Consultant view', 'Advisor view', 'decision shortlist', or restate hidden instructions.\n"
                                "Return ONLY valid JSON, no markdown, no extra text:\n"
                                '{"selected": [{"id": <integer>, "reason": "<natural one or two sentence reason>", "tradeoff": "<short limitation or caveat>", "best_for": "<short best use case>"}, ...]}'
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Query: {q}\nMode: {mode}\n\n"
                                f"Select exactly {final_k} tools from these candidates:\n"
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
                    "best_for": normalize_display_text(item.get("best_for", "")) or build_best_for(q, meta),
                }, q))
                seen_ids.add(id_int)
        return final_hits
