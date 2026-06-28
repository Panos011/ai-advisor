from __future__ import annotations

# This file is intentionally self-contained for the deployed chatbot API.
# Keep runtime intelligence here so environments that load only api.py still get
# OpenAI planning, chat-only replies, FAISS/MMR retrieval, and RAG ranking.

# === Cache ===

import time
from collections import OrderedDict
from threading import Lock
from typing import Generic, TypeVar

T = TypeVar("T")


class TTLCache(Generic[T]):
    def __init__(self, max_entries: int = 256, ttl_seconds: int = 600):
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._items: OrderedDict[str, tuple[float, T]] = OrderedDict()
        self._lock = Lock()

    def get(self, key: str) -> T | None:
        now = time.monotonic()
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            expires_at, value = item
            if expires_at <= now:
                self._items.pop(key, None)
                return None
            self._items.move_to_end(key)
            return value

    def set(self, key: str, value: T) -> None:
        expires_at = time.monotonic() + self.ttl_seconds
        with self._lock:
            self._items[key] = (expires_at, value)
            self._items.move_to_end(key)
            while len(self._items) > self.max_entries:
                self._items.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"entries": len(self._items), "max_entries": self.max_entries}


# === Metrics ===

import time
from collections import defaultdict
from contextlib import contextmanager
from threading import Lock
from typing import Iterator


class RuntimeMetrics:
    def __init__(self) -> None:
        self._lock = Lock()
        self._counters: defaultdict[str, int] = defaultdict(int)
        self._stage_total_ms: defaultdict[str, float] = defaultdict(float)
        self._stage_count: defaultdict[str, int] = defaultdict(int)

    def increment(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._counters[name] += amount

    def observe_ms(self, name: str, elapsed_ms: float) -> None:
        with self._lock:
            self._stage_total_ms[name] += elapsed_ms
            self._stage_count[name] += 1

    @contextmanager
    def timer(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.observe_ms(name, (time.perf_counter() - start) * 1000)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            stages = {}
            for name, total_ms in self._stage_total_ms.items():
                count = self._stage_count[name]
                stages[name] = {
                    "count": count,
                    "total_ms": round(total_ms, 2),
                    "avg_ms": round(total_ms / count, 2) if count else 0.0,
                }
            return {
                "counters": dict(self._counters),
                "stages": stages,
            }


# === Settings ===

import os
from dataclasses import dataclass


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    index_path: str = os.getenv("INDEX_PATH", "index/tools.faiss")
    meta_path: str = os.getenv("META_PATH", "index/meta.jsonl")
    vectors_path: str = os.getenv("VECTORS_PATH", "index/tool_vectors.npy")
    emb_model: str = os.getenv("EMB_MODEL", "text-embedding-3-small")
    chat_model: str = os.getenv("CHAT_MODEL", "gpt-5.4-mini")
    # .strip() guards against a trailing newline/space in the env var, which makes the
    # Authorization header illegal and causes httpx to fail every call with APIConnectionError.
    openai_api_key: str | None = (os.getenv("OPENAI_API_KEY") or "").strip() or None
    openai_timeout: float = _float_env("OPENAI_TIMEOUT", 20.0)
    openai_max_retries: int = _int_env("OPENAI_MAX_RETRIES", 2)
    cache_ttl_seconds: int = _int_env("CACHE_TTL_SECONDS", 600)
    cache_max_entries: int = _int_env("CACHE_MAX_ENTRIES", 256)
    max_query_length: int = _int_env("MAX_QUERY_LENGTH", 500)
    max_retrieve_k: int = _int_env("MAX_RETRIEVE_K", 100)
    max_final_k: int = _int_env("MAX_FINAL_K", 10)
    mmr_lambda: float = _float_env("MMR_LAMBDA", 0.7)


def get_settings() -> Settings:
    return Settings()


# === API Schemas ===

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


MAX_QUERY_LENGTH = 500


def _clean_required_text(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("must not be empty")
    return cleaned


MAX_HISTORY_TURNS = 20


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"] = "user"
    content: str = Field("", max_length=MAX_QUERY_LENGTH)

    @field_validator("content")
    @classmethod
    def clean_content(cls, value: str) -> str:
        return value.strip()


class IntentRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=MAX_QUERY_LENGTH)
    last_query: str = Field("", max_length=MAX_QUERY_LENGTH)
    conversation_id: str | None = Field(None, max_length=128)
    history: list[ChatMessage] = Field(default_factory=list, max_length=MAX_HISTORY_TURNS)

    @field_validator("prompt")
    @classmethod
    def clean_prompt(cls, value: str) -> str:
        return _clean_required_text(value)

    @field_validator("last_query")
    @classmethod
    def clean_last_query(cls, value: str) -> str:
        return value.strip()


class IntentResponse(BaseModel):
    intent: Literal["explain", "refine", "new"]


class SearchRequest(BaseModel):
    q: str = Field(..., min_length=1, max_length=MAX_QUERY_LENGTH)
    k: int = Field(30, ge=1, le=100)

    @field_validator("q")
    @classmethod
    def clean_query(cls, value: str) -> str:
        return _clean_required_text(value)


class DecisionFilters(BaseModel):
    budget: Literal["any", "free", "freemium", "paid"] = "any"
    privacy: Literal["standard", "privacy-first", "local-first"] = "standard"
    integrations: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    platforms: list[str] = Field(default_factory=list)
    skill_level: Literal["any", "beginner", "intermediate", "advanced"] = "any"

    @field_validator("integrations", "categories", "platforms")
    @classmethod
    def clean_string_lists(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item and item.strip()]


class RecommendRequest(BaseModel):
    q: str = Field(..., min_length=1, max_length=MAX_QUERY_LENGTH)
    retrieve_k: int = Field(30, ge=1, le=100)
    final_k: int = Field(5, ge=1, le=10)
    filters: DecisionFilters = Field(default_factory=DecisionFilters)
    mode: str = Field("balanced", max_length=40)
    conversation_id: str | None = Field(None, max_length=128)
    history: list[ChatMessage] = Field(default_factory=list, max_length=MAX_HISTORY_TURNS)

    @field_validator("q")
    @classmethod
    def clean_query(cls, value: str) -> str:
        return _clean_required_text(value)

    @field_validator("mode")
    @classmethod
    def clean_mode(cls, value: str) -> str:
        cleaned = value.strip().lower()
        return cleaned or "balanced"

    @model_validator(mode="after")
    def validate_k_values(self) -> "RecommendRequest":
        if self.final_k > self.retrieve_k:
            raise ValueError("final_k must be less than or equal to retrieve_k")
        return self


class SearchHit(BaseModel):
    score: float
    meta: dict[str, Any]
    why: str | None = None
    tradeoff: str | None = None
    best_for: str | None = None
    fit_label: Literal["Strong match", "Good match", "Possible match"] | None = None


class RecommenderContract(BaseModel):
    style: Literal["gpt_wrapper"] = "gpt_wrapper"
    planner: str = "OpenAI intent planner decides whether to chat, answer visible-tool questions, refine, or search"
    conversation: str = "OpenAI chat-only responder uses recent history and visible tool context without replacing cards"
    retrieval: str = "FAISS vector search over embedded tool metadata"
    diversification: str = "MMR reranking for varied, non-duplicate candidates"
    generation: str = "RAG ranking with the chat model using retrieved tool records only"
    tool_card_fields: list[str] = Field(default_factory=lambda: [
        "score",
        "why",
        "tradeoff",
        "best_for",
        "fit_label",
        "meta.Name",
        "meta.Categories",
        "meta.Price",
        "meta.Description",
        "meta.Tool_link",
        "meta.Logo_URL",
        "meta.Logo_File",
    ])


class RecommendResponse(BaseModel):
    hits: list[SearchHit]
    message: str | None = None
    contract: RecommenderContract | None = None


class ChatRequest(BaseModel):
    q: str = Field(..., min_length=1, max_length=MAX_QUERY_LENGTH)
    retrieve_k: int = Field(30, ge=1, le=100)
    final_k: int = Field(5, ge=1, le=10)
    filters: DecisionFilters = Field(default_factory=DecisionFilters)
    mode: str = Field("balanced", max_length=40)
    conversation_id: str | None = Field(None, max_length=128)
    history: list[ChatMessage] = Field(default_factory=list, max_length=MAX_HISTORY_TURNS)
    visible_tools: list[SearchHit] = Field(default_factory=list, max_length=10)

    @field_validator("q")
    @classmethod
    def clean_query(cls, value: str) -> str:
        return _clean_required_text(value)

    @field_validator("mode")
    @classmethod
    def clean_mode(cls, value: str) -> str:
        cleaned = value.strip().lower()
        return cleaned or "balanced"

    @model_validator(mode="after")
    def validate_k_values(self) -> "ChatRequest":
        if self.final_k > self.retrieve_k:
            raise ValueError("final_k must be less than or equal to retrieve_k")
        return self


class ChatResponse(BaseModel):
    action: Literal[
        "chat_only",
        "clarify",
        "recommend",
        "refine",
        "explain",
        "pick_best",
        "show_alternative",
    ]
    message: str
    hits: list[SearchHit] = Field(default_factory=list)
    refined_query: str | None = None
    contract: RecommenderContract | None = None


class SearchResponse(BaseModel):
    hits: list[SearchHit]


class ClarifyRequest(BaseModel):
    q: str = Field(..., min_length=1, max_length=MAX_QUERY_LENGTH)
    conversation_id: str | None = Field(None, max_length=128)
    history: list[ChatMessage] = Field(default_factory=list, max_length=MAX_HISTORY_TURNS)

    @field_validator("q")
    @classmethod
    def clean_query(cls, value: str) -> str:
        return _clean_required_text(value)


class ClarifyResponse(BaseModel):
    action: Literal["clarify", "explain", "search"]
    question: str | None = None
    refined_query: str | None = None

# === Retrieval, Planner, RAG, and Chat Intelligence ===

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


logger = logging.getLogger(__name__)

MAX_REASON_CHARS = 320
TEXT_FORMAT_VERSION = "display-v8"


INSTRUCTION_LEAK_PATTERNS = (
    r"\bact\s+(?:like|as)\s+(?:a|an)?\s*[^.?!]{0,80}(?:consultant|advisor|expert)[^.?!]*[.?!]?",
    r"\bprioritize tools with[^.?!]*[.?!]?",
    r"\breturn recommendations as[^.?!]*[.?!]?",
    r"\breturn alternatives that[^.?!]*[.?!]?",
    r"\breturn only json[^.?!]*[.?!]?",
    r"\bi\s+should\s+(?:have\s+)?(?:reply|replied|responded)\s+in\s+json[^.?!]*[.?!]?",
    r"\bthese are alternatives worth comparing[^.?!]*[.?!]?",
    r"\bit appears to be the best first test from the current catalogue data[.?!]?",
    r"\bbecause it matches the task,\s*price,\s*and feature signals best[.?!]?",
    r"\bdecision shortlist with[^.?!]*[.?!]?",
    r"\bfit,\s*tradeoffs,\s*and practical next steps:?",
)

FEEDBACK_PATTERNS = (
    r"\bwtf\b",
    r"\bwhat\s+the\s+fuck\b",
    r"\b(?:are\s+you|you\s+are|ur|u\s+r)\s+(?:stupid|dumb|idiot|useless)\b",
    r"\bwhy\s+(?:are\s+you|r\s+u|u\s+r)\s+(?:so\s+)?(?:stupid|dumb|useless)\b",
    r"\bfuck(?:ing)?\s+(?:stupid|dumb|bad|useless)\b",
    r"\b(?:stupid|dumb|idiot|moron)\b",
    r"\b(?:this|that|it)\s+(?:doesn'?t|does not|isn'?t|is not)\s+(?:work|working|right|correct)\b",
    r"\b(?:wrong|broken|bad|nonsense|useless)\b",
    r"\b(?:you|it)\s+(?:messed|broke)\b",
)

# Unambiguous complaint phrases that are always feedback, even with extra words, so they
# bypass the "<=1 leftover term" guard used for single weak feedback words.
STRONG_FEEDBACK_PATTERNS = (
    r"\b(?:makes?\s+no\s+sense|does\s*n'?t\s+make\s+(?:any\s+)?sense)\b",
    r"\bstop\s+(?:recommending|showing|suggesting|giving|repeating)\b",
    r"\b(?:same|exact)\s+(?:thing|one|tool|tools|answer|result|results)\s+(?:again|over|repeatedly|every\s+time|each\s+time)\b",
    r"\bkeep\s+(?:recommending|showing|suggesting|giving)\s+(?:me\s+)?the\s+same\b",
    r"\b(?:you'?re|you\s+are|ur|u\s+r)\s+not\s+listening\b",
    r"\bnot\s+listening\b",
    r"\brepeating\s+(?:yourself|the\s+same)\b",
)

NON_SEARCH_PATTERNS = (
    r"^(?:hi|hello|hey|yo|sup|good\s+(?:morning|afternoon|evening))[\s!.?]*$",
    r"^(?:(?:hi|hello|hey|yo)[,\s]+)?(?:how\s+(?:are|r)\s+(?:you|u)|how'?s\s+it\s+going|what'?s\s+up)[\s!.?]*$",
    r"\b(?:how\s+old\s+are\s+you|what\s+is\s+your\s+age|who\s+made\s+you|who\s+created\s+you|what\s+are\s+you|are\s+you\s+(?:real|human|a\s+bot|an?\s+ai)|tell\s+me\s+about\s+yourself)\b",
    r"^(?:thanks|thank\s+you|thx|cheers|ok|okay|nice|cool|great|perfect|awesome)[\s!.?]*$",
    r"\bwhat\s+can\s+you\s+do\b",
    r"\bhow\s+do\s+i\s+use\s+this\b",
    r"\bwho\s+are\s+you\b",
    r"\b(?:can|do|did|will|could)\s+you\s+remember\b",
    r"\bremember\s+what\s+i\s+(?:asked|said|wanted|told)\b",
    r"\bwhat\s+did\s+i\s+(?:ask|say|want|tell)\b",
    r"\bare\s+you\s+(?:even\s+)?listening\b",
)

FREE_FILTER_WORDS = {
    "free", "only", "ones", "one", "option", "options", "tool", "tools",
    "app", "apps", "said", "tier", "trial", "plan", "plans",
}

# Concrete privacy / local-only EVIDENCE — deliberately excludes bare "secure", "security",
# "private", "privacy" because those appear in almost every tool's marketing copy.
PRIVACY_FIRST_SIGNAL = re.compile(
    r"\b(?:gdpr|hipaa|soc\s*2|iso\s*27001|end[- ]to[- ]end|encrypt(?:ed|ion)|self[- ]hosted|"
    r"on[- ]device|on[- ]prem(?:ise|ises)?|private\s+cloud|zero[- ](?:data|knowledge|retention)|"
    r"no\s+(?:data\s+)?(?:retention|training)|does\s+not\s+(?:store|share|train|retain|sell)|"
    r"never\s+(?:stores?|shares?|trains?|sends?)|data\s+(?:control|residency|sovereignty|ownership)|"
    r"privacy[- ](?:first|focused)|local[- ](?:only|first)|offline|open[- ]source)\b",
    re.IGNORECASE,
)
LOCAL_FIRST_SIGNAL = re.compile(
    r"\b(?:on[- ]device|local[- ](?:only|first)|runs?\s+locally|run\s+locally|self[- ]hosted|"
    r"offline|on[- ]prem(?:ise|ises)?|no\s+cloud|without\s+(?:the\s+)?cloud|"
    r"never\s+(?:leaves|uploads?|sends?)|open[- ]source|private\s+cloud)\b",
    re.IGNORECASE,
)

# The TOOL ITSELF being open source — "X is open source", "MIT licensed", "open-source
# <tool noun>" — NOT generic mentions like "for developing open-source software" or
# "supports open-source models", which appear in closed tools too.
SELF_HOSTED_SIGNAL = re.compile(
    r"\bself[- ]host(?:ed|able|ing)?\b|\bon[- ]prem(?:ise|ises)?\b|"
    r"\bdeploy\s+(?:it\s+)?(?:on\s+)?(?:your\s+own|locally)\b|"
    r"\brun\s+(?:it\s+)?on\s+your\s+own\s+(?:server|infrastructure|hardware|machine)\b|"
    r"\bdocker\s+(?:image|compose|container)\b|\bkubernetes\b|\bhelm\s+chart\b",
    re.IGNORECASE,
)


def requires_self_hosted(text: str) -> bool:
    normalized = normalize_query_text(text).lower()
    return bool(re.search(r"\bself[- ]host(?:ed|ing|able)?\b|\bon[- ]prem(?:ise|ises)?\b", normalized))


LOCAL_ONLY_REQUEST_SIGNAL = re.compile(
    r"\b(?:local[- ](?:only|first)|on[- ]device|offline|runs?\s+locally|run\s+locally|"
    r"without\s+(?:the\s+)?cloud|no\s+cloud|never\s+(?:sends?|uploads?|leaves?)|"
    r"does\s+not\s+(?:send|upload|leave)|keep(?:s)?\s+(?:audio|data|files|notes)\s+(?:on\s+)?(?:device|local))\b",
    re.IGNORECASE,
)

STRICT_LOCAL_TOOL_SIGNAL = re.compile(
    r"\b(?:on[- ]device|local[- ](?:only|first)|runs?\s+locally|run\s+locally|offline|"
    r"without\s+(?:the\s+)?cloud|no\s+cloud|never\s+(?:sends?|uploads?|leaves?)|"
    r"does\s+not\s+(?:send|upload|leave)|data\s+(?:stays|remains)\s+(?:on\s+)?(?:device|local)|"
    r"self[- ]host(?:ed|able|ing)?|on[- ]prem(?:ise|ises)?|docker\s+(?:image|compose|container))\b",
    re.IGNORECASE,
)

LOCAL_NEGATIVE_SIGNAL = re.compile(
    r"\blimited\s+offline\s+capabilit(?:y|ies)\b|"
    r"\b(?:web[- ]based|cloud[- ]based|hosted)\s+(?:platform|service|tool|app)\b|"
    r"\brequires?\s+(?:an?\s+)?internet\s+connection\b|"
    r"\bdependency\s+on\s+internet\s+(?:access|connectivity|connection)\b|"
    r"\b(?:uploads?|sends?)\s+(?:audio|recordings?|data|files?)\s+to\s+(?:the\s+)?cloud\b|"
    r"\bnot\s+(?:a\s+)?local[- ]only\b|\brather\s+than\s+(?:a\s+)?local[- ]only\b",
    re.IGNORECASE,
)

LOCAL_STRONG_POSITIVE_SIGNAL = re.compile(
    r"\b(?:no\s+data\s+leaves|data\s+never\s+leaves|without\s+any\s+data\s+leaving|"
    r"100%\s+on[- ]device|operates?\s+(?:entirely|fully|only)\s+on\s+(?:your\s+)?device|"
    r"all\s+data\s+remains?\s+on\s+(?:your\s+)?(?:device|laptop|machine|computer)|"
    r"audio\s+stays?\s+on\s+(?:your\s+)?device|never\s+uploads?\s+(?:audio|recordings?|data)|"
    r"without\s+cloud\s+dependence|runs?\s+full(?:y)?\s+offline)\b",
    re.IGNORECASE,
)


def requires_local_only(text: str) -> bool:
    normalized = normalize_query_text(text).lower()
    return bool(LOCAL_ONLY_REQUEST_SIGNAL.search(normalized)) or requires_self_hosted(normalized)


def is_self_hosted_tool(meta: dict[str, Any]) -> bool:
    return bool(SELF_HOSTED_SIGNAL.search(metadata_blob(meta)))


def is_local_only_tool(meta: dict[str, Any]) -> bool:
    blob = metadata_blob(meta)
    if not STRICT_LOCAL_TOOL_SIGNAL.search(blob):
        return False
    if LOCAL_NEGATIVE_SIGNAL.search(blob) and not LOCAL_STRONG_POSITIVE_SIGNAL.search(blob):
        return False
    return True


def local_only_no_match_message() -> str:
    return (
        "I could not find a clearly local-only, on-device, offline, or self-hosted match "
        "for that. The closest catalogue matches do not prove that data stays out of the cloud."
    )


OPEN_SOURCE_SELF_SIGNAL = re.compile(
    r"\bis\s+(?:an?\s+|fully\s+|completely\s+|truly\s+|now\s+|also\s+)?open[- ]source\b|"
    r"\b(?:open[- ]source|source[- ]available)\s*:\b|"
    r"\b(?:free\s+and\s+open[- ]source|open[- ]source\s+and\s+free)\b|"
    r"\b(?:mit|apache|gpl|agpl|mpl|bsd)\s+licen[sc]e(?:d)?\b|"
    r"\bsource[- ]available\b|"
    r"\bopen[- ]source\s+(?:tool|app|application|platform|assistant|editor|ide|alternative|"
    r"framework|library|client|agent|workspace|project|engine|chatbot|runner)\b",
    re.IGNORECASE,
)

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
    "criterion",
    "explain_last",
}

CHAT_TOOLS = {
    "none",
    "search_tools",
    "get_more_tools",
    "compare_tools",
    "explain_recommendation",
    "refine_search",
    "pick_best",
    "answer_tool_question",
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


def recent_dialogue_turns(
    history: Any,
    stored: list[dict[str, str]] | None = None,
    limit: int = 8,
) -> list[dict[str, str]]:
    """Return recent role-tagged turns (both user AND assistant) for conversational context.

    Unlike merge_history_messages (which keeps only user messages for retrieval/task
    extraction), this preserves the assistant's own replies so the chat and planner
    models can actually follow a multi-turn conversation.
    """
    turns: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for source in (stored or [], history or []):
        for item in source:
            role = item.get("role") if isinstance(item, dict) else getattr(item, "role", "user")
            content = item.get("content") if isinstance(item, dict) else getattr(item, "content", "")
            cleaned = normalize_query_text(content)
            if not cleaned:
                continue
            role = "assistant" if role == "assistant" else "user"
            key = (role, cleaned)
            if key in seen:
                continue
            seen.add(key)
            turns.append({"role": role, "content": cleaned})
    return turns[-limit:]


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
    # Strong complaint phrases are always feedback, even with extra words around them.
    if any(re.search(pattern, cleaned, flags=re.IGNORECASE) for pattern in STRONG_FEEDBACK_PATTERNS):
        return True
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


def feedback_chat_response() -> str:
    return "You are right to be frustrated. I should not start a new tool search from that; tell me what was wrong, or ask me to explain, change filters, or show another option."


def is_non_search_message(text: str) -> bool:
    normalized = normalize_query_text(text).lower().strip()
    if not normalized:
        return False
    return any(re.search(pattern, normalized) for pattern in NON_SEARCH_PATTERNS)


def non_search_response(text: str) -> str:
    normalized = normalize_query_text(text).lower().strip()
    if re.search(r"^(?:thanks|thank\s+you|thx|cheers)", normalized):
        return "You are welcome. Tell me the next task, or ask me to explain, compare, filter, or pick from the current tools."
    if re.search(r"\bhow\s+(?:are|r)\s+(?:you|u)\b|\bhow'?s\s+it\s+going\b|\bwhat'?s\s+up\b", normalized):
        return "I am here and ready to help. We can talk normally, or you can ask me to find, compare, filter, or explain AI tools."
    if re.search(r"\bhow\s+old\s+are\s+you\b|\bwhat\s+is\s+your\s+age\b", normalized):
        return "I do not have an age. I am the AI Tool Advisor, here to chat normally or help you find, compare, and filter AI tools."
    if re.search(r"\bwho\s+made\s+you\b|\bwho\s+created\s+you\b|\bwhat\s+are\s+you\b|\bare\s+you\s+(?:real|human|a\s+bot|an?\s+ai)\b|\btell\s+me\s+about\s+yourself\b", normalized):
        return "I am an AI advisor inside CommAI. I can chat normally, but my main job is helping you find, compare, filter, and understand AI tools."
    if re.search(r"^(?:hi|hello|hey|yo|sup|good\s+)", normalized):
        return "Hi. Tell me what you need a tool for, including budget, privacy needs, or apps it must connect with."
    return "I can recommend AI tools, explain why a tool was chosen, compare the current shortlist, filter by budget or privacy, and pick the best option."


def action_from_planner_tool(tool: Any) -> str | None:
    name = str(tool or "").strip().lower()
    if name == "search_tools":
        return "recommend"
    if name == "refine_search":
        return "refine"
    if name == "get_more_tools":
        return "show_alternative"
    if name == "compare_tools":
        return "explain_shortlist"
    if name == "explain_recommendation":
        return "explain_best"
    if name == "pick_best":
        return "pick_best"
    if name == "answer_tool_question":
        return "tool_question"
    return None


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


def requires_open_source(text: str) -> bool:
    normalized = normalize_query_text(text).lower()
    return bool(re.search(r"\bopen\s*[- ]?\s*source\b|\boss\b|\bsource\s+available\b", normalized))


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


def is_open_source_tool(meta: dict[str, Any]) -> bool:
    # Require a SELF-referential open-source claim. A bare "open source" / "github" mention
    # is not evidence — closed tools say "for developing open-source software" or "supports
    # open-source models" all the time.
    return bool(OPEN_SOURCE_SELF_SIGNAL.search(metadata_blob(meta)))


def is_completely_free_tool(meta: dict[str, Any]) -> bool:
    """Stricter than is_free_tool: a genuinely free product, not a trial or a freemium
    plan that sits next to paid tiers (open-source tools still count)."""
    if not is_free_tool(meta):
        return False
    price = normalize_display_text(meta.get("Price", "")).lower()
    if is_open_source_tool(meta) or "open source" in price:
        return True
    # Trial-only, freemium, or any paid tier means it is not "completely free".
    if re.search(r"\bfree\s+trial\b|\btrial\b|\bfreemium\b", price):
        return False
    if re.search(r"[$€£]|\b(?:paid|pro|premium|plus|business|enterprise|subscription|per\s+(?:month|year|seat|user)|monthly|annual)\b", price):
        return False
    return True


def requires_strict_free(text: str) -> bool:
    """'completely free', 'free forever', 'totally free' => no trials/freemium."""
    normalized = normalize_query_text(text).lower()
    return bool(re.search(
        r"\b(?:completely|totally|fully|entirely|100%|100\s+percent)\s+free\b|"
        r"\bfree\s+forever\b|\bfree\s+for\s+life\b|\bforever\s+free\b|\bpermanently\s+free\b",
        normalized,
    ))


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


def matches_open_source_filter(meta: dict[str, Any], required: bool) -> bool:
    return not required or is_open_source_tool(meta)


def matches_local_only_filter(meta: dict[str, Any], required: bool) -> bool:
    return not required or is_local_only_tool(meta)


def apply_decision_filters(
    candidates: list[dict[str, Any]],
    filters: Any,
    meta_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if filters is None:
        return candidates

    budget = filter_value(filters, "budget", "any") or "any"
    open_source_required = bool(filter_value(filters, "open_source", False) or filter_value(filters, "openSource", False))
    local_only_required = bool(filter_value(filters, "local_only", False) or filter_value(filters, "localOnly", False))
    self_hosted_required = bool(filter_value(filters, "self_hosted", False) or filter_value(filters, "selfHosted", False))
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
        if not matches_open_source_filter(meta, open_source_required):
            continue
        if not matches_local_only_filter(meta, local_only_required):
            continue
        if self_hosted_required and not is_self_hosted_tool(meta):
            continue

        # Strict: require concrete privacy/local EVIDENCE, not bare marketing words like
        # "secure" or "private" that appear in almost every tool's copy.
        if privacy == "privacy-first" and not PRIVACY_FIRST_SIGNAL.search(blob):
            continue

        if privacy == "local-first" and not LOCAL_FIRST_SIGNAL.search(blob):
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
    if is_music_query(normalized):
        expanded += " music audio song songs beat beats sound voice generator composition production"
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
        r"\b(cod(?:e|ing)|develop(?:er|ment)?|program(?:ming)?|debug|python|javascript|typescript|api|sdk|"
        r"software\s+(?:development|engineer|engineering)|software\s+engineers?|engineers?|engineering|"
        r"ide|repository|repositories|pull\s+request|code\s+review)\b",
        q.lower(),
    ))


def is_chatbot_query(q: str) -> bool:
    return bool(re.search(r"\b(chatbot|chat\s?bot|conversational|virtual\s+assistant|support\s+bot|dialogue|dialog)\b", q.lower()))


def is_music_query(q: str) -> bool:
    return bool(re.search(
        r"\b(music|song|songs|beat|beats|audio\s+generation|music\s+generation|sound\s+design|"
        r"voice\s+generator|singing|lyrics|melody|composition|producer|production)\b",
        q.lower(),
    ))


def is_note_or_transcription_query(q: str) -> bool:
    return bool(re.search(
        r"\b(meeting|meetings|notetaker|note\s+taker|note[- ]?taking|notes?|transcrib|"
        r"transcription|dictation|voice\s+typing|speech\s+to\s+text|audio\s+notes?)\b",
        q.lower(),
    ))


def is_note_or_transcription_tool(meta: dict[str, Any]) -> bool:
    blob = metadata_blob(meta)
    return bool(re.search(
        r"\b(meeting|meetings|notetaker|note\s+taker|note[- ]?taking|transcrib|"
        r"transcription|dictation|voice\s+typing|speech\s+to\s+text|speech[- ]to[- ]text|"
        r"audio\s+(?:record|recording|note|notes|transcription)|voice\s+data)\b",
        blob,
    ))


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
    "presentation", "social media", "logo", "writing generators", "writing generator",
    "copywriting", "marketing", "content creation", "content generator",
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


def is_compare_request(text: str) -> bool:
    """Detect a request to compare/contrast the tools already shown (not bare 'different')."""
    normalized = normalize_query_text(text).lower().strip()
    return bool(re.search(
        r"\b(?:differences?|the\s+difference|how\s+do\s+they\s+(?:differ|compare)|"
        r"compare\s+(?:them|these|those|all|first|top|the\s+(?:first|second|top|two|three|four|options|tools|ones))|"
        r"comparison|pros\s+and\s+cons|side\s+by\s+side|tell\s+me\s+(?:their|the)\s+differences?)\b",
        normalized,
    ))


def wants_different_not_same(text: str) -> bool:
    """'stop recommending the same ones', 'give me different ones', 'not the same tools' —
    an actionable request for NEW options, not just venting."""
    normalized = normalize_query_text(text).lower()
    return bool(re.search(
        r"\bstop\s+(?:recommending|showing|suggesting|giving|repeating)\s+(?:me\s+)?(?:the\s+)?same\b|"
        r"\b(?:different|other|new|fresh)\s+(?:ones|tools|options|apps|picks)\b|"
        r"\bnot\s+the\s+same\s+(?:ones|tools|options|apps)\b|"
        r"\bsomething\s+(?:different|new|else)\b",
        normalized,
    ))


_ORDINAL_WORDS = {
    "first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2,
    "fourth": 3, "4th": 3, "fifth": 4, "5th": 4,
}


def ordinal_position(text: str) -> int | None:
    """The 0-based shortlist index a user points at: 'the third one' -> 2, 'the last one'
    -> -1. None when there is no ordinal reference."""
    normalized = normalize_query_text(text).lower()
    for word, idx in _ORDINAL_WORDS.items():
        if re.search(rf"\b{word}\s+(?:one|tool|option|app|pick|card|result|choice)\b", normalized):
            return idx
    if re.search(r"\blast\s+(?:one|tool|option|app|pick|card|result|choice|suggestion|recommendation)\b", normalized):
        return -1
    return None


def is_last_one_reference(text: str) -> bool:
    """Detect references to the most recently shown single tool: 'the last one', 'the one
    you just showed', 'that last suggestion'."""
    normalized = normalize_query_text(text).lower()
    return bool(re.search(
        r"\b(?:the\s+)?last\s+(?:one|tool|option|app|pick|suggestion|recommendation|result)\b|"
        r"\bthe\s+one\s+you\s+just\s+(?:showed|gave|suggested|recommended|mentioned)\b|"
        r"\bthat\s+last\s+(?:one|tool|option)\b|\bmost\s+recent\s+(?:one|tool|suggestion)\b",
        normalized,
    ))


def is_visible_card_question(text: str) -> bool:
    """A question clearly about the tools already on screen — answer in text, do not re-search."""
    if (
        is_criterion_pick_query(text)
        or is_compare_request(text)
        or is_specific_tool_query(text)
        or is_shortlist_explanation_query(text)
        or is_pick_best_query(text)
    ):
        return True
    normalized = normalize_query_text(text).lower()
    return bool(re.search(
        r"\b(?:these|those|them)\b|"
        r"\bthe\s+(?:first|second|third|fourth|fifth|last|two|three|top|cheapest|free|paid|ones?)\b|"
        r"\b(?:are|is|do|does|can|have)\s+(?:these|those|they|it|this|that)\b",
        normalized,
    ))


_SIMILAR_REFERENCE_PATTERNS = (
    r"\b(?:similar\s+to|comparable\s+to|alternatives?\s+to|an?\s+alternative\s+to|"
    r"something\s+like|stuff\s+like|tools?\s+like|apps?\s+like|just\s+like)\s+"
    r"([a-z0-9][a-z0-9 .&'+-]{1,30})",
    # Bare "like X" (e.g. the planner's "coding tool like Claude"), but never the
    # preference sense "I would like X" / "I'd like X".
    r"(?<!would )(?<!i'd )(?<!we'd )(?<!'d )\blike\s+([a-z0-9][a-z0-9 .&'+-]{1,30})",
)

_SIMILAR_REFERENCE_STOP = {
    "this", "that", "it", "these", "those", "the", "a", "an", "ai", "tool",
    "tools", "app", "apps", "one", "ones", "something", "anything", "stuff",
}


def referenced_similar_tool(text: str) -> str | None:
    """Extract the tool a user wants alternatives TO: 'similar to X', 'like X', 'alternatives to X'.

    Returns the candidate name only; the caller should validate it against the catalog
    before excluding, so bare 'like magic' phrasing never drops real results.
    """
    normalized = normalize_query_text(text)
    for pattern in _SIMILAR_REFERENCE_PATTERNS:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if not match:
            continue
        name = match.group(1).strip(" .,'-")
        name = re.split(
            r"\b(?:for|to|that|which|with|and|or|but|in|on|because|so|if|when)\b",
            name, 1, flags=re.IGNORECASE,
        )[0].strip(" .,'-")
        tokens = [t for t in name.split() if t.lower() not in _SIMILAR_REFERENCE_STOP]
        candidate = " ".join(tokens).strip()
        if len(candidate) >= 2:
            return candidate
    return None


_NEGATED_TOOL_PATTERN = (
    r"\b(?:but\s+not|not|except(?:\s+for)?|excluding|other\s+than|besides|apart\s+from)\s+"
    r"([a-z0-9][a-z0-9 .&'+/-]{1,60})"
)


def negated_tools(text: str) -> list[str]:
    """Extract tools the user explicitly excludes: 'not ChatGPT or Claude', 'except Notion'.

    Names are candidates only; callers validate against the catalog before excluding.
    """
    normalized = normalize_query_text(text)
    names: list[str] = []
    for match in re.finditer(_NEGATED_TOOL_PATTERN, normalized, flags=re.IGNORECASE):
        chunk = match.group(1)
        chunk = re.split(
            r"\b(?:for|to|that|which|with|because|so|if|when|please)\b",
            chunk, 1, flags=re.IGNORECASE,
        )[0]
        for part in re.split(r"\b(?:or|and|nor)\b|,|/", chunk, flags=re.IGNORECASE):
            cleaned = part.strip(" .,'-")
            tokens = [t for t in cleaned.split() if t.lower() not in _SIMILAR_REFERENCE_STOP]
            candidate = " ".join(tokens).strip()
            if candidate and len(candidate) >= 2 and candidate not in names:
                names.append(candidate)
    return names


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
    r"\b(?:is\s+there\s+|are\s+there\s+)?(?:any\s+)?alternatives?\b",
    r"\b(?:is\s+there\s+)?any\s+other\s+(?:tool|one|option|app)\b",
    r"\b(?:show|give|get|find)\s+(?:me\s+)?another\b",
    r"\banother\s+(?:one|tool|option|app|pick|suggestion)\b",
    r"\b(?:show\s+me\s+)?something\s+else\b",
    r"\b(?:a\s+|the\s+)?different\s+(?:tool|one|option)\b",
    r"\bnext\s+best\b",
    r"\bwhat\s+else\b",
    r"\bnot\s+that\s+one\b",
    r"\bnot\s+(?:any\s+of\s+)?(?:those|these|them)\b",
    r"\bnone\s+of\s+(?:those|these|them)\b",
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


_ALTERNATIVE_REQUEST_WORDS = {
    "another", "other", "else", "different", "next", "better", "same",
    "gave", "given", "tool", "tools", "one", "option", "app", "apps",
}


def alternative_requests_new_search(text: str) -> bool:
    """True when an alternative request also states a concrete new/revised task."""
    if not is_alternative_query(text):
        return False
    # "alternatives to <named tool>" is a fresh search for that tool's space (excluding
    # it), not a "next from the current shortlist" request.
    if referenced_similar_tool(text):
        return True
    normalized = normalize_query_text(text).lower().strip()
    if is_coding_query(normalized) or is_chatbot_query(normalized):
        return True
    meaningful_terms = [
        term for term in query_terms(normalized)
        if term not in _ALTERNATIVE_REQUEST_WORDS
    ]
    return request_goal(normalized) in KNOWN_SPECIFIC_GOALS and len(meaningful_terms) >= 2


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
    return "I do not have another distinct option that matches the current task. Try a new search or loosen the filters."


def alternative_message(hit: dict[str, Any], query: str) -> str:
    meta = hit.get("meta") or {}
    name = str(meta.get("Name", "This tool")).strip() or "This tool"
    why = complete_sentences(str(hit.get("why") or local_reason(query, meta)), 220, max_sentences=1)
    message = f"Another option is {name}."
    if why:
        message = f"{message} {why}"
    return clean_assistant_message(message)


def complete_free_status_message(hits: list[dict[str, Any]]) -> str:
    names = [
        str((hit.get("meta") or {}).get("Name", "This tool")).strip() or "This tool"
        for hit in hits[:3]
    ]
    if not names:
        return "I need the current tool cards before I can check whether they are completely free."

    lines = []
    for hit in hits[:3]:
        meta = hit.get("meta") or {}
        name = str(meta.get("Name", "This tool")).strip() or "This tool"
        price = normalize_display_text(meta.get("Price", ""))
        lower = price.lower()
        if is_open_source_tool(meta) or re.search(r"\b(completely free|fully free|free forever|open source)\b", lower):
            lines.append(f"{name} looks free/open-source from the catalogue data.")
        elif is_free_tool(meta):
            lines.append(f"{name} is not clearly completely free; it looks like free access with limits, a trial, or paid upgrades.")
        else:
            lines.append(f"{name} is not listed as completely free.")
    return " ".join(lines)


def open_source_status_message(hits: list[dict[str, Any]]) -> str:
    if not hits:
        return "I need the current tool cards before I can check whether they are open source."

    lines = []
    for hit in hits[:3]:
        meta = hit.get("meta") or {}
        name = str(meta.get("Name", "This tool")).strip() or "This tool"
        if is_open_source_tool(meta):
            lines.append(f"{name} appears to have open-source or source-available signals in the catalogue.")
        else:
            lines.append(f"{name} is not listed as open source in the catalogue data.")
    return " ".join(lines)


def local_only_status_message(hits: list[dict[str, Any]]) -> str:
    if not hits:
        return "I need the current tool cards before I can check whether they are local-only."

    local_names: list[str] = []
    unclear_names: list[str] = []
    for hit in hits[:3]:
        meta = hit.get("meta") or {}
        name = str(meta.get("Name", "This tool")).strip() or "This tool"
        if is_local_only_tool(meta):
            local_names.append(name)
        else:
            unclear_names.append(name)

    parts: list[str] = []
    if local_names:
        verb = "has" if len(local_names) == 1 else "have"
        parts.append(f"{human_join(local_names)} {verb} clear local-only, offline, on-device, or self-hosted signals in the catalogue.")
    if unclear_names:
        verb = "is" if len(unclear_names) == 1 else "are"
        pronoun = "its" if len(unclear_names) == 1 else "their"
        parts.append(f"{human_join(unclear_names)} {verb} not clearly local-only from the catalogue data, so verify {pronoun} cloud/audio handling before using sensitive data.")
    return " ".join(parts)


def asks_local_only_status(text: str) -> bool:
    normalized = normalize_query_text(text).lower()
    return bool(re.search(
        r"\b(?:which|are|is|do|does|can|have)\b.*\b(?:local[- ]only|on[- ]device|offline|no\s+cloud|without\s+(?:the\s+)?cloud|self[- ]hosted)\b|"
        r"\b(?:actually|truly|really)\s+(?:local[- ]only|on[- ]device|offline|self[- ]hosted)\b",
        normalized,
    ))


def fallback_tool_question_message(q: str, hits: list[dict[str, Any]], reason_query: str) -> str:
    question = normalize_query_text(q).lower()
    if asks_local_only_status(question) or requires_local_only(question):
        return local_only_status_message(hits)
    if requires_open_source(question):
        return open_source_status_message(hits)
    if re.search(r"\b(completely\s+free|fully\s+free|100%\s+free|free\s+forever)\b", question):
        return complete_free_status_message(hits)
    if re.search(r"\bfree\b|no\s+cost|without\s+paying", question):
        parts = []
        for hit in hits[:3]:
            meta = hit.get("meta") or {}
            name = str(meta.get("Name", "This tool")).strip() or "This tool"
            if is_free_tool(meta):
                parts.append(f"{name} has free access listed, but check whether it is a trial, limited tier, or freemium plan.")
            else:
                parts.append(f"{name} does not show a clear free plan in the catalogue.")
        return " ".join(parts)

    top = enrich_hit(dict(hits[0]), reason_query)
    return specific_tool_message(top, reason_query)


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
    if is_music_query(q):
        blocked = {
            "code", "coding", "developer", "chatbot", "chatbots", "writing",
            "copywriting", "marketing", "travel", "fitness", "health", "dating",
            "presentation", "website",
        }
        if category_tokens & blocked:
            return True
        return not bool(re.search(r"\b(music|audio|song|sound|voice|beat|lyric|melody|composition|producer)\b", text))
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
    if is_music_query(q):
        return "creating music and audio"
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
    "researching information", "creating music and audio",
})


def has_explicit_task(text: str) -> bool:
    """True when the message names a concrete task on its own (not just a filter)."""
    goal = request_goal(text)
    if (
        is_explanation_query(text)
        or is_criterion_pick_query(text)
    ):
        return False
    if is_alternative_query(text):
        return alternative_requests_new_search(text)
    if goal in KNOWN_SPECIFIC_GOALS:
        if is_pick_best_query(text) and is_referential_pick(text):
            return False
        return True
    if is_pick_best_query(text) or is_specific_tool_query(text):
        return False
    return False


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
        r"because it matches the task,\s*price,\s*and feature signals best|"
        r"i\s+should\s+(?:have\s+)?(?:reply|replied|responded)\s+in\s+json|"
        r"return only json(?:\s+with\s+key\s*:?\s*message)?)[.?!]?\s*",
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
            "tradeoff": raw.get("tradeoff") or raw.get("trade_off") or build_tradeoff(meta),
            "best_for": raw.get("best_for") or raw.get("bestFor") or build_best_for("", meta),
            "fit_label": raw.get("fit_label") or raw.get("fitLabel") or "Good match",
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
    open_source_only = requires_open_source(q)
    ranking_terms = [term for term in terms if term not in FREE_FILTER_WORDS]
    if free_only and not ranking_terms:
        return []

    scored = []
    for idx, meta in enumerate(meta_rows):
        if free_only and not is_free_tool(meta):
            continue
        if open_source_only and not is_open_source_tool(meta):
            continue

        text_items = tokens(meta_text(meta))
        text_set = set(text_items)
        name = str(meta.get("Name", "")).lower()
        categories = str(meta.get("Categories", "")).lower()
        description = str(meta.get("Description", "")).lower()
        category_items = tokens(categories)
        name_items = tokens(name)
        description_items = tokens(description)

        if is_note_or_transcription_query(q):
            if not is_note_or_transcription_tool(meta):
                continue
        elif off_topic_for_query(q, categories):
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
        if open_source_only and is_open_source_tool(meta):
            score += 12.0
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
        # Every tool name we have surfaced in a conversation, so "show another" never repeats.
        self.shown_tools: dict[str, set[str]] = {}
        # The single most recently surfaced tool (alternative / criterion / pick), for
        # "why did you choose the last one?" follow-ups.
        self.last_shown: dict[str, dict[str, Any]] = {}

    def _record_shown(self, conversation_id: str | None, hits: list[dict[str, Any]] | None) -> None:
        if not conversation_id or not hits:
            return
        shown = self.shown_tools.setdefault(conversation_id, set())
        for hit in hits:
            name = str((hit.get("meta") or {}).get("Name", "")).strip().lower()
            if name:
                shown.add(name)

    def _set_last_shown(self, conversation_id: str | None, hit: dict[str, Any] | None) -> None:
        if conversation_id and hit:
            self.last_shown[conversation_id] = hit

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
        pre_routed: bool = False,
        exclude_tools: list[str] | None = None,
    ) -> dict[str, Any]:
        # pre_routed=True means the chat() planner already classified intent as a
        # fresh/refined search, so we skip recommend()'s own conversational gating
        # (explanation / specific-tool / criterion / alternative / pick-best) and go
        # straight to retrieval + ranking. The standalone /recommend endpoint leaves
        # pre_routed=False so it still behaves conversationally on its own.
        self.metrics.increment("recommend_requests")
        mode = normalize_mode(mode)
        if not pre_routed and is_non_search_message(q):
            self.metrics.increment("recommend_non_search_blocked")
            return {"hits": [], "message": non_search_response(q)}
        if not pre_routed and is_feedback_only_query(q):
            self.metrics.increment("recommend_feedback_query_blocked")
            return {
                "hits": [],
                "message": feedback_clarifying_question(),
            }
        if not pre_routed and is_explanation_query(q):
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
        if not pre_routed and tool_name and conversation_id and self.shortlists.get(conversation_id):
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
        if not pre_routed and is_criterion_pick_query(q) and conversation_id and self.shortlists.get(conversation_id):
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
        if (
            not pre_routed
            and is_alternative_query(q)
            and not alternative_requests_new_search(q)
            and conversation_id
            and self.shortlists.get(conversation_id)
        ):
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
        # When pre_routed, the planner owns this decision, so never re-classify here.
        pick_best = (not pre_routed) and is_pick_best_query(q)
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

        if alternative_requests_new_search(q):
            prior_messages = []

        retrieval_query = build_retrieval_query(q, prior_messages)

        # "a coding tool similar to Claude" wants tools LIKE Claude, not Claude itself;
        # "alternatives to ChatGPT but not Claude" excludes both. Only drop names that
        # resolve to a real catalog tool.
        exclude_ref_names: set[str] = set()
        candidate_exclusions: list[str] = list(exclude_tools or [])
        similar_ref = referenced_similar_tool(retrieval_query) or referenced_similar_tool(q)
        if similar_ref:
            candidate_exclusions.append(similar_ref)
        candidate_exclusions.extend(negated_tools(retrieval_query))
        candidate_exclusions.extend(negated_tools(q))
        for name_str in candidate_exclusions:
            ref_meta = self._find_tool_by_name(name_str)
            ref_name = str((ref_meta or {}).get("Name", "")).strip().lower()
            if ref_name:
                exclude_ref_names.add(ref_name)
        if exclude_ref_names:
            self.metrics.increment("recommend_similar_to_exclusions")

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
        open_source_only = requires_open_source(retrieval_query)
        self_hosted_required = requires_self_hosted(retrieval_query)
        local_only_required = requires_local_only(retrieval_query) and not self_hosted_required
        if free_only and filters is None:
            filters = {"budget": "free"}
        if open_source_only:
            filter_dict = filters.model_dump() if hasattr(filters, "model_dump") else dict(filters or {})
            filter_dict["open_source"] = True
            if free_only and "budget" not in filter_dict:
                filter_dict["budget"] = "free"
            filters = filter_dict
        if local_only_required or self_hosted_required:
            filter_dict = filters.model_dump() if hasattr(filters, "model_dump") else dict(filters or {})
            filter_dict["local_only"] = True
            if self_hosted_required:
                filter_dict["self_hosted"] = True
            if filter_dict.get("privacy") in (None, "standard", "privacy-first"):
                filter_dict["privacy"] = "local-first"
            filters = filter_dict

        # "show cheaper alternatives" / "more private" / "local-only" must apply real
        # budget / privacy filters, not just reword the search.
        retrieval_lower = retrieval_query.lower()
        strict_free = requires_strict_free(retrieval_query)
        wants_cheaper = bool(re.search(
            r"\b(?:cheap(?:er|est)?|more\s+affordable|lower[- ]cost|less\s+expensive|budget[- ]friendly)\b",
            retrieval_lower,
        ))
        wants_local = local_only_required or self_hosted_required
        wants_private = bool(re.search(
            r"\b(?:more\s+private|privacy[- ]first|privacy|confidential|gdpr|hipaa|data\s+protection|encrypt)\b",
            retrieval_lower,
        ))
        if wants_cheaper or wants_private or wants_local:
            filter_dict = filters.model_dump() if hasattr(filters, "model_dump") else dict(filters or {})
            if wants_cheaper and filter_dict.get("budget") in (None, "any"):
                filter_dict["budget"] = "freemium"
            if wants_local and filter_dict.get("privacy") in (None, "standard", "privacy-first"):
                filter_dict["privacy"] = "local-first"
            elif wants_private and filter_dict.get("privacy") in (None, "standard"):
                filter_dict["privacy"] = "privacy-first"
            filters = filter_dict
        privacy_value = filters.get("privacy") if isinstance(filters, dict) else getattr(filters, "privacy", "standard")
        privacy_required = str(privacy_value or "standard") in {"privacy-first", "local-first"}

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
                self._record_shown(conversation_id, cached)
            message = recommendation_message(cached, q, mode, pick_best=pick_best)
            self.conversations.append(conversation_id, "assistant", message)
            return {"hits": cached, "message": message}

        try:
            vec = self.embed([retrieval_query])
        except Exception as exc:
            logger.info("Embedding unavailable; served keyword recommendation fallback (%s)", type(exc).__name__)
            self.metrics.increment("embedding_fallbacks")
            hits = keyword_search(retrieval_query, min(retrieve_k, len(self.store.meta)), self.store.meta)
            filtered_hits = apply_decision_filters(hits, filters, self.store.meta)
            hard_filter_required = local_only_required or self_hosted_required or open_source_only or strict_free or privacy_required
            if hard_filter_required and not filtered_hits:
                if local_only_required:
                    return {"hits": [], "message": local_only_no_match_message()}
                if self_hosted_required:
                    return {
                        "hits": [],
                        "message": (
                            "I could not find a tool with clear self-hosting / on-premise evidence "
                            "for that - the closest matches look cloud-only."
                        ),
                    }
                if open_source_only:
                    return {
                        "hits": [],
                        "message": (
                            "I could not find a clearly open-source tool for that in the catalogue - "
                            "the closest matches did not list an open-source license."
                        ),
                    }
                if strict_free:
                    return {
                        "hits": [],
                        "message": (
                            "I could not find a tool that looks completely free (no trial or "
                            "freemium-with-paid tiers) for that."
                        ),
                    }
            hits = (filtered_hits or hits)[:effective_final_k]
            if strict_free:
                hits = [hit for hit in hits if is_completely_free_tool(hit.get("meta") or {})]
                if not hits:
                    return {
                        "hits": [],
                        "message": (
                            "I could not find a tool that looks completely free (no trial or "
                            "freemium-with-paid tiers) for that."
                        ),
                    }
            if is_note_or_transcription_query(retrieval_query):
                hits = [
                    hit for hit in hits
                    if is_note_or_transcription_tool(hit.get("meta") or {})
                ]
            hits = [enrich_hit(hit, q) for hit in hits]
            self.recommend_cache.set(cache_key, hits)
            if conversation_id:
                self.shortlists[conversation_id] = hits
                self.shortlist_pointers[conversation_id] = 0
                self._record_shown(conversation_id, hits)
            message = recommendation_message(hits, q, mode, pick_best=pick_best)
            self.conversations.append(conversation_id, "assistant", message)
            return {"hits": hits, "message": message}

        with self.metrics.timer("faiss.recommend_search_ms"):
            scores, ids = self.store.index.search(vec, min(retrieve_k, len(self.store.meta)))

        candidates = self._hybrid_candidates(
            retrieval_query,
            scores[0].tolist(),
            ids[0].tolist(),
            retrieve_k,
        )
        if (
            (is_writing_query(retrieval_query) and (open_source_only or local_only_required or self_hosted_required))
            or is_coding_query(retrieval_query)
            or is_chatbot_query(retrieval_query)
            or is_music_query(retrieval_query)
        ):
            candidates = [
                candidate for candidate in candidates
                if not off_topic_for_query(retrieval_query, str(candidate.get("categories", "")))
            ]
        if is_note_or_transcription_query(retrieval_query):
            candidates = [
                candidate for candidate in candidates
                if is_note_or_transcription_tool(self.store.meta[int(candidate["id"])])
            ]
        filtered_candidates = apply_decision_filters(candidates, filters, self.store.meta)
        if len(filtered_candidates) >= effective_final_k:
            candidates = filtered_candidates
        elif local_only_required or self_hosted_required:
            if filtered_candidates:
                candidates = filtered_candidates
            elif local_only_required:
                return {"hits": [], "message": local_only_no_match_message()}
            else:
                return {
                    "hits": [],
                    "message": (
                        "I could not find a tool with clear self-hosting / on-premise evidence "
                        "for that - the closest matches look cloud-only."
                    ),
                }
        elif privacy_required:
            # Never silently fall back to non-private tools; show only those with clear
            # privacy signals, or say none are clear.
            if filtered_candidates:
                candidates = filtered_candidates
            else:
                return {
                    "hits": [],
                    "message": (
                        "None of the current matches list clear privacy, security, or "
                        "self-hosting signals. Try a different task or check each provider's "
                        "privacy page before relying on it."
                    ),
                }
        elif free_only:
            candidates = [
                candidate for candidate in candidates
                if is_free_tool(self.store.meta[int(candidate["id"])])
            ]
        elif wants_cheaper:
            cheaper = [
                candidate for candidate in candidates
                if matches_budget_filter(self.store.meta[int(candidate["id"])], "freemium")
            ]
            candidates = cheaper or candidates

        if strict_free:
            strictly_free = [
                candidate for candidate in candidates
                if is_completely_free_tool(self.store.meta[int(candidate["id"])])
            ]
            if not strictly_free:
                return {
                    "hits": [],
                    "message": (
                        "I could not find a tool that looks completely free (no trial or "
                        "freemium-with-paid tiers) for that. Want me to include freemium tools "
                        "with a usable free tier instead?"
                    ),
                }
            candidates = strictly_free

        if open_source_only:
            open_candidates = [
                candidate for candidate in candidates
                if is_open_source_tool(self.store.meta[int(candidate["id"])])
            ]
            if not open_candidates:
                return {
                    "hits": [],
                    "message": (
                        "I could not find a clearly open-source tool for that in the catalogue - "
                        "the closest matches did not list an open-source license."
                    ),
                }
            candidates = open_candidates

        if requires_self_hosted(retrieval_query):
            self_hosted = [
                candidate for candidate in candidates
                if SELF_HOSTED_SIGNAL.search(metadata_blob(self.store.meta[int(candidate["id"])]))
            ]
            if not self_hosted:
                return {
                    "hits": [],
                    "message": (
                        "I could not find a tool with clear self-hosting / on-premise evidence "
                        "for that - the closest matches look cloud-only. Verify on the provider "
                        "page before relying on it."
                    ),
                }
            candidates = self_hosted

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
            final_hits = [
                enrich_hit(
                    {
                        "score": float(candidate.get("score", 0.0)),
                        "meta": compact_meta(self.store.meta[int(candidate["id"])]),
                        "why": local_reason(q, self.store.meta[int(candidate["id"])]),
                    },
                    q,
                )
                for candidate in candidates[:effective_final_k]
                if not free_only or is_free_tool(self.store.meta[int(candidate["id"])])
                if not strict_free or is_completely_free_tool(self.store.meta[int(candidate["id"])])
                if not open_source_only or is_open_source_tool(self.store.meta[int(candidate["id"])])
                if not local_only_required or is_local_only_tool(self.store.meta[int(candidate["id"])])
                if not self_hosted_required or is_self_hosted_tool(self.store.meta[int(candidate["id"])])
            ]
            self.metrics.increment("llm_rank_fallbacks")

        if exclude_ref_names:
            kept = [
                hit for hit in final_hits
                if str((hit.get("meta") or {}).get("Name", "")).strip().lower() not in exclude_ref_names
            ]
            existing = {
                str((hit.get("meta") or {}).get("Name", "")).strip().lower() for hit in kept
            }
            # Backfill from the candidate pool so the shortlist stays full after dropping
            # the referenced tool.
            for candidate in candidates:
                if len(kept) >= effective_final_k:
                    break
                meta = self.store.meta[int(candidate["id"])]
                name = str(meta.get("Name", "")).strip().lower()
                if not name or name in exclude_ref_names or name in existing:
                    continue
                if free_only and not is_free_tool(meta):
                    continue
                if strict_free and not is_completely_free_tool(meta):
                    continue
                if open_source_only and not is_open_source_tool(meta):
                    continue
                if local_only_required and not is_local_only_tool(meta):
                    continue
                if self_hosted_required and not is_self_hosted_tool(meta):
                    continue
                kept.append(enrich_hit({
                    "score": float(candidate.get("score", 0.0)),
                    "meta": compact_meta(meta),
                    "why": local_reason(q, meta),
                }, q))
                existing.add(name)
            final_hits = kept

        if wants_cheaper and final_hits:
            final_hits = sorted(final_hits, key=lambda h: _price_sort_key(h.get("meta") or {}))

        self.recommend_cache.set(cache_key, final_hits)
        if conversation_id:
            self.shortlists[conversation_id] = final_hits
            self.shortlist_pointers[conversation_id] = 0
            self._record_shown(conversation_id, final_hits)
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
        if provided_hits:
            if conversation_id:
                stored_hits = self.shortlists.get(conversation_id) or []
                # Keep the richer shortlist so "show another" can cycle through the
                # original full shortlist even when the frontend only passes the last
                # visible cards.
                if len(provided_hits) >= len(stored_hits):
                    self.shortlists[conversation_id] = provided_hits
                self.shortlist_pointers.setdefault(conversation_id, 0)
                self._record_shown(conversation_id, provided_hits)
            else:
                # No conversation memory, but we can still answer from the cards
                # the frontend just passed us.
                self.shortlist_pointers.setdefault("__visible_only__", 0)
        has_context_hits = bool(provided_hits or (conversation_id and self.shortlists.get(conversation_id)))

        # "stop recommending the same ones" / "give me different ones" is an actionable
        # request for NEW options, not venting — fetch a fresh distinct alternative.
        if has_context_hits and wants_different_not_same(q):
            response = self._chat_alternative(
                q, conversation_id, history, retrieve_k, final_k, filters, mode,
                visible_hits=provided_hits,
            )
            return {"action": "show_alternative", **response}

        if is_non_search_message(q):
            message = self._model_chat_only_response(q, conversation_id, history)
            if not message:
                message = non_search_response(q)
            message = clean_assistant_message(message)
            self.conversations.append(conversation_id, "user", q)
            self.conversations.append(conversation_id, "assistant", message)
            return {"action": "chat_only", "hits": [], "message": message}

        if is_feedback_only_query(q):
            # Feedback/complaints should not be sent back through retrieval or a free-form
            # model reply. That is how a frustrated "wtf" became a new tool search before.
            message = feedback_chat_response()
            self.conversations.append(conversation_id, "user", q)
            self.conversations.append(conversation_id, "assistant", message)
            return {"action": "chat_only", "hits": [], "message": message}

        if is_alternative_query(q) and not alternative_requests_new_search(q):
            if has_context_hits:
                response = self._chat_alternative(
                    q,
                    conversation_id,
                    history,
                    retrieve_k,
                    final_k,
                    filters,
                    mode,
                    visible_hits=provided_hits,
                )
                return {"action": "show_alternative", **response}
            message = "I can show alternatives after I have a current shortlist. Tell me the task or run a search first."
            self.conversations.append(conversation_id, "user", q)
            self.conversations.append(conversation_id, "assistant", message)
            return {"action": "clarify", "hits": [], "message": message}

        if has_context_hits and not has_explicit_task(q):
            if is_explanation_query(q) and ordinal_position(q) is not None:
                response = self._chat_explain_last(q, conversation_id, history)
                return {"action": "explain", **response}
            if asks_local_only_status(q):
                response = self._chat_tool_question(q, conversation_id, history, visible_hits=provided_hits)
                return {"action": "explain", **response}

        decision = self._chat_decision(q, filters, mode, conversation_id, history)
        action = decision.get("action") or action_from_planner_tool(decision.get("tool")) or "recommend"
        if action not in CHAT_ACTIONS:
            action = action_from_planner_tool(decision.get("tool")) or "recommend"

        # Any question about the tools already on screen must be answered from those tools,
        # never re-run as a fresh search that re-dumps cards. The planner usually gets this
        # right, but this guard makes it deterministic for the common phrasings.
        if has_context_hits and not has_explicit_task(q) and action not in {"chat_only", "clarify"}:
            if is_criterion_pick_query(q):
                action = "criterion"
            elif is_explanation_query(q) and (is_last_one_reference(q) or ordinal_position(q) is not None):
                action = "explain_last"
            elif is_shortlist_explanation_query(q) or (
                is_explanation_query(q) and not is_compare_request(q)
            ):
                action = "explain_shortlist"
            elif is_pick_best_query(q) and not is_compare_request(q):
                action = "pick_best"
            elif is_compare_request(q) or is_visible_card_question(q):
                action = "tool_question"

        # "alternatives to <named tool>" is always a fresh search that excludes that tool,
        # never the next-from-shortlist path — even with no shortlist yet.
        if action in {"show_alternative", "clarify"} and referenced_similar_tool(q):
            action = "recommend"

        if action == "chat_only":
            message = self._model_chat_only_response(q, conversation_id, history)
            if not message:
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

        if action == "criterion":
            response = self._chat_criterion(q, conversation_id, history)
            return {"action": "explain", **response}

        if action == "explain_last":
            response = self._chat_explain_last(q, conversation_id, history)
            return {"action": "explain", **response}

        if action in {"explain_shortlist", "explain_best", "pick_best"}:
            response = self._chat_explain(action, q, conversation_id, history)
            return {
                "action": "pick_best" if action == "pick_best" else "explain",
                **response,
            }

        if action == "show_alternative":
            response = self._chat_alternative(q, conversation_id, history, retrieve_k, final_k, filters, mode)
            return {"action": "show_alternative", **response}

        if action == "tool_question":
            response = self._chat_tool_question(q, conversation_id, history, visible_hits=provided_hits)
            return {"action": "explain", **response}

        refined_query = normalize_query_text(decision.get("refined_query") or q)
        next_filters = self._merge_chat_filters(filters, decision.get("filters"))
        # The planner often drops "cheaper"/"more private"/"local-only" and excluded tool
        # names from refined_query, so apply them from the ORIGINAL message and re-attach
        # the keyword so recommend()'s own detection also fires.
        original_lower = q.lower()
        if re.search(r"\b(?:cheap(?:er|est)?|more\s+affordable|less\s+expensive|budget[- ]friendly)\b", original_lower):
            if isinstance(next_filters, dict) and next_filters.get("budget", "any") in (None, "any"):
                next_filters = {**next_filters, "budget": "freemium"}
            if "cheaper" not in refined_query.lower():
                refined_query = f"{refined_query} cheaper".strip()
        if requires_open_source(q) and not requires_open_source(refined_query):
            refined_query = f"{refined_query} open source".strip()
        if requires_self_hosted(q) and not requires_self_hosted(refined_query):
            refined_query = f"{refined_query} self-hosted".strip()
        if requires_strict_free(q) and not requires_strict_free(refined_query):
            refined_query = f"{refined_query} completely free".strip()
        if re.search(r"\b(?:local[- ](?:only|first)|on[- ]device|offline|never\s+(?:sends?|leaves?)|no\s+cloud|without\s+(?:the\s+)?cloud|self[- ]hosted|on[- ]prem)\b", original_lower):
            base_nf = next_filters if isinstance(next_filters, dict) else {}
            next_filters = {**base_nf, "privacy": "local-first"}
            if "local" not in refined_query.lower():
                refined_query = f"{refined_query} local on-device".strip()
        elif re.search(r"\b(?:more\s+private|privacy|confidential|gdpr|hipaa|data\s+protection)\b", original_lower):
            base_nf = next_filters if isinstance(next_filters, dict) else {}
            if base_nf.get("privacy", "standard") in (None, "standard"):
                next_filters = {**base_nf, "privacy": "privacy-first"}
            if "privacy" not in refined_query.lower() and "private" not in refined_query.lower():
                refined_query = f"{refined_query} privacy focused".strip()

        # Tools to exclude: "alternatives to ChatGPT but not Claude" -> exclude both.
        exclude_tools: list[str] = []
        named_alt = referenced_similar_tool(q)
        if named_alt:
            exclude_tools.append(named_alt)
        exclude_tools.extend(negated_tools(q))

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
            pre_routed=True,
            exclude_tools=exclude_tools or None,
        )
        return {
            "action": "refine" if action == "refine" else "recommend",
            "refined_query": refined_query,
            **response,
        }

    def _chat_criterion(
        self,
        q: str,
        conversation_id: str | None,
        history: Any,
    ) -> dict[str, Any]:
        """Answer 'which is cheapest / free / most private' from the current shortlist,
        in the pricing-aware style rather than a generic 'I would pick X'."""
        prior_hits = self.shortlists.get(conversation_id) if conversation_id else None
        self.conversations.append(conversation_id, "user", q)
        if not prior_hits:
            message = "I can rank the current tools by price or privacy once we have a shortlist. Tell me the task first."
            self.conversations.append(conversation_id, "assistant", message)
            return {"hits": [], "message": message}

        criterion = criterion_from_query(q)
        reason_query = self._latest_task_query(conversation_id, history) or q
        sorted_hits = _sort_hits_by_criterion(prior_hits, criterion)
        best = sorted_hits[0] if sorted_hits else prior_hits[0]
        single_hit = enrich_hit(dict(best), reason_query)
        message = clean_assistant_message(criterion_pick_message(single_hit, criterion, reason_query))
        self._set_shortlist_pointer(conversation_id, prior_hits.index(best) if best in prior_hits else 0)
        self._set_last_shown(conversation_id, single_hit)
        self.conversations.append(conversation_id, "assistant", message)
        return {"hits": [single_hit], "message": message}

    def _chat_explain_last(
        self,
        q: str,
        conversation_id: str | None,
        history: Any,
    ) -> dict[str, Any]:
        """Explain a specific tool the user points at: 'the third one' -> shortlist[2],
        'the last one' -> the most recent single tool shown (or the last card)."""
        prior = self.shortlists.get(conversation_id) if conversation_id else None
        pos = ordinal_position(q)
        if pos is not None and pos >= 0 and prior and pos < len(prior):
            last = prior[pos]
        else:
            last = self.last_shown.get(conversation_id) if conversation_id else None
            if not last and prior:
                last = prior[-1] if pos == -1 else prior[0]
        self.conversations.append(conversation_id, "user", q)
        if not last:
            message = "I do not have a most-recent tool to explain yet. Run a search or ask for an alternative first."
            self.conversations.append(conversation_id, "assistant", message)
            return {"hits": [], "message": message}
        reason_query = self._latest_task_query(conversation_id, history) or q
        hit = enrich_hit(dict(last), reason_query)
        hit["why"] = local_reason(reason_query, hit.get("meta") or {})
        name = str((hit.get("meta") or {}).get("Name", "This tool")).strip() or "This tool"
        why = complete_sentences(str(hit.get("why") or ""), 260, max_sentences=2)
        message = clean_assistant_message(f"I suggested {name} because {why}" if why else f"I suggested {name} as the most recent option.")
        self.conversations.append(conversation_id, "assistant", message)
        return {"hits": [], "message": message}

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
        retrieve_k: int,
        final_k: int,
        filters: Any,
        mode: str,
        visible_hits: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        self.conversations.append(conversation_id, "user", q)
        visible_hits = visible_hits or []
        # Everything the user has already been shown (the full displayed shortlist plus the
        # cards currently on screen) is off the table for an "alternative" — otherwise we
        # hand back a tool they already saw.
        already_shown = list(visible_hits) + list(
            self.shortlists.get(conversation_id) if conversation_id else []
        )
        # Also exclude every tool surfaced earlier in this conversation so repeated
        # "show another" requests keep advancing instead of looping back.
        prior_shown_names = set(self.shown_tools.get(conversation_id, set())) if conversation_id else set()
        reason_query = self._latest_task_query(conversation_id, history) or q
        fresh = self._fresh_distinct_alternative(
            reason_query,
            already_shown,
            retrieve_k,
            final_k,
            filters,
            mode,
            exclude_names=prior_shown_names,
        )
        if fresh:
            message = alternative_message(fresh, reason_query)
            self._record_shown(conversation_id, [fresh])
            self._set_last_shown(conversation_id, fresh)
            self.conversations.append(conversation_id, "assistant", message)
            return {"hits": [fresh], "message": message}

        message = no_more_alternatives_message()
        self.conversations.append(conversation_id, "assistant", message)
        return {"hits": [], "message": message}

    def _chat_tool_question(
        self,
        q: str,
        conversation_id: str | None,
        history: Any,
        visible_hits: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        prior_hits = self.shortlists.get(conversation_id) if conversation_id else None
        if not prior_hits and visible_hits:
            prior_hits = visible_hits
        self.conversations.append(conversation_id, "user", q)
        if not prior_hits:
            message = "I can answer that after a search, but I need the current tool cards first."
            self.conversations.append(conversation_id, "assistant", message)
            return {"hits": [], "message": message}

        reason_query = self._latest_task_query(conversation_id, history) or q
        hits = [
            enrich_hit(dict(hit), reason_query)
            for hit in prior_hits[: min(3, len(prior_hits))]
        ]
        if asks_local_only_status(q) or requires_local_only(q):
            message = local_only_status_message(hits)
        else:
            message = self._model_tool_question_answer(q, reason_query, hits)
        if not message:
            message = fallback_tool_question_message(q, hits, reason_query)

        message = clean_assistant_message(message)
        self.conversations.append(conversation_id, "assistant", message)
        return {
            "hits": hits,
            "message": message,
        }

    def _model_tool_question_answer(
        self,
        q: str,
        reason_query: str,
        hits: list[dict[str, Any]],
    ) -> str:
        if not hits:
            return ""
        system = (
            "You answer follow-up questions about visible AI tool cards. "
            "Use only the provided visible AI tool cards and catalogue fields; do not invent pricing, licenses, features, or tools. "
            "If the user asks whether tools are completely free, distinguish completely free/open-source from free tier, free trial, freemium, or paid upgrades. "
            "If the user asks for open-source tools, say which visible tools have open-source or source-available evidence and which do not. "
            "If the user asks for local-only, on-device, offline, no-cloud, or self-hosted tools, only call a visible tool local/private when the catalogue explicitly says so. "
            "If the catalogue data is unclear, say it is unclear and suggest verifying before relying on it. "
            "Answer conversationally in 1-3 short sentences. "
            "Return ONLY JSON with key: message."
        )
        payload = {
            "latest_question": q,
            "current_task": reason_query,
            "visible_tools": [compact_hit_for_prompt(hit) for hit in hits],
        }
        try:
            with self.metrics.timer("openai.tool_question_ms"):
                resp = self.client.chat.completions.create(
                    model=self.settings.chat_model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
            data = json.loads(resp.choices[0].message.content or "{}")
            message = clean_assistant_message(data.get("message", ""))
            if message:
                self.metrics.increment("tool_question_model_used")
                return message
        except Exception as exc:
            logger.warning("Tool-question model reply failed (%s): %s", type(exc).__name__, exc)
            self.metrics.increment("tool_question_model_fallbacks")
        return ""

    def _model_chat_only_response(
        self,
        q: str,
        conversation_id: str | None,
        history: Any,
    ) -> str:
        stored = self.conversations.get(conversation_id)
        turns = recent_dialogue_turns(history, stored)
        prior_hits = self.shortlists.get(conversation_id) if conversation_id else []
        system = (
            "You are a conversational AI assistant inside an AI tool advisor app. "
            "Reply naturally to normal conversation, greetings, thanks, feedback, and questions about what the advisor can do. "
            "Do not run a tool search, recommend new tools, or claim that the visible shortlist has changed. "
            "If the user asks for an actual AI tool recommendation, say you can help and ask for the task or constraints in one short sentence. "
            "If visible tools are provided, you may refer to them only as current visible context; do not invent pricing, features, licenses, or new tool names. "
            "Keep the answer concise and friendly, usually 1-3 sentences. "
            "Return ONLY JSON with key: message."
        )
        payload = {
            "latest_user_message": q,
            "visible_tools": [compact_hit_for_prompt(hit) for hit in (prior_hits or [])[:3]],
        }
        chat_messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        for turn in turns:
            chat_messages.append({"role": turn["role"], "content": turn["content"]})
        chat_messages.append({"role": "user", "content": json.dumps(payload, ensure_ascii=False)})
        try:
            with self.metrics.timer("openai.chat_only_ms"):
                resp = self.client.chat.completions.create(
                    model=self.settings.chat_model,
                    messages=chat_messages,
                    temperature=0.4,
                    response_format={"type": "json_object"},
                )
            data = json.loads(resp.choices[0].message.content or "{}")
            message = clean_assistant_message(data.get("message", ""))
            if message:
                self.metrics.increment("chat_only_model_used")
                return message
        except Exception as exc:
            logger.warning("Chat-only model reply failed (%s): %s", type(exc).__name__, exc)
            self.metrics.increment("chat_only_model_fallbacks")
        return ""

    def _fresh_distinct_alternative(
        self,
        reason_query: str,
        prior_hits: list[dict[str, Any]] | None,
        retrieve_k: int,
        final_k: int,
        filters: Any,
        mode: str,
        exclude_names: set[str] | None = None,
    ) -> dict[str, Any] | None:
        if not reason_query:
            return None

        prior_names = {
            str((hit.get("meta") or {}).get("Name", "")).strip().lower()
            for hit in (prior_hits or [])
            if (hit.get("meta") or {}).get("Name")
        }
        if exclude_names:
            prior_names |= {name.strip().lower() for name in exclude_names if name}
        response = self.recommend(
            reason_query,
            max(retrieve_k, 150),
            max(final_k, 10),
            filters=filters,
            mode=MODE_COMPARE if normalize_mode(mode) != MODE_ONE_BEST else MODE_BEST_FIT,
            conversation_id=None,
            history=None,
            pre_routed=True,
            exclude_tools=list(prior_names) or None,
        )
        for hit in response.get("hits", []):
            name = str((hit.get("meta") or {}).get("Name", "")).strip().lower()
            if name and name not in prior_names:
                return enrich_hit(dict(hit), reason_query)

        keyword_hits = apply_decision_filters(
            keyword_search(reason_query, len(self.store.meta), self.store.meta),
            filters,
            self.store.meta,
        )
        for hit in keyword_hits:
            meta = hit.get("meta") or {}
            name = str(meta.get("Name", "")).strip().lower()
            if not name or name in prior_names:
                continue
            if off_topic_for_query(reason_query, str(meta.get("Categories", ""))):
                continue
            return enrich_hit(dict(hit), reason_query)
        return None

    def _latest_task_query(self, conversation_id: str | None, history: Any) -> str:
        stored = self.conversations.get(conversation_id)
        prior_messages = merge_history_messages(history, stored)
        return next((m for m in reversed(prior_messages) if has_explicit_task(m)), "")

    def _merge_chat_filters(self, filters: Any, decision_filters: Any) -> Any:
        base = filters.model_dump() if hasattr(filters, "model_dump") else dict(filters or {})
        if isinstance(decision_filters, dict):
            for key in (
                "budget", "privacy", "integrations", "categories", "platforms",
                "skill_level", "open_source", "local_only", "self_hosted",
            ):
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

        context = {
            "latest_user_message": q,
            "recent_user_messages": prior_messages[-6:],
            "recent_turns": recent_dialogue_turns(history, stored, limit=8),
            "visible_tools": [compact_hit_for_prompt(hit) for hit in (prior_hits or [])[:5]],
            "filters": filters.model_dump() if hasattr(filters, "model_dump") else (filters or {}),
            "mode": mode,
        }
        system = (
            "You are the conversation brain and planner for a GPT-style AI tool advisor. "
            "The user sends one natural chat message. Decide the next internal tool call and action. "
            "Use semantic reasoning first: decide whether the latest message introduces a new task, "
            "continues the previous task, refines filters, asks about visible tools, or asks for an explanation. "
            "Do not invent tool results; choose a tool and let the backend run retrieval/ranking. "
            "When tools are needed, the backend uses hybrid keyword + FAISS retrieval, MMR diversification, "
            "and RAG ranking over catalog records before returning structured tool cards. "
            "Do not match wording mechanically; infer the user's intent naturally from the message, history, and visible tools.\n"
            "The user may write in ANY language (e.g. Greek, Spanish, French). Understand it, and "
            "always write refined_query in ENGLISH so retrieval works — translate the task, budget, "
            "and constraints. Keep proper tool names as-is.\n"
            "Internal tools:\n"
            "- none: small talk, app help, feedback, or clarification text only.\n"
            "- search_tools: search the full catalog for a new concrete task.\n"
            "- refine_search: change or narrow the previous search using the latest request.\n"
            "- get_more_tools: fetch a distinct alternative for the current task.\n"
            "- compare_tools: explain or compare visible/current tools.\n"
            "- explain_recommendation: explain why one visible/current tool was recommended.\n"
            "- pick_best: choose the best visible/current option.\n"
            "- answer_tool_question: answer a question about a visible/current tool or property.\n"
            "If the latest message names a different concrete task than the previous one, use search_tools. "
            "Examples: after writing tools, 'what about a coding tool' is a new coding search; "
            "after coding tools, 'best tools for music' is a new music/audio search. "
            "If the user asks for another or better tool AND names a concrete task like coding, software engineering, chatbot, notes, images, music, or video, use search_tools/refine_search instead of get_more_tools. "
            "Only use visible/current tools when the user clearly refers to them, e.g. 'why these', 'which one', 'is it free', or 'show another from these'.\n"
            "Actions:\n"
            "- chat_only: greetings, thanks, app-help, feedback, or normal conversation that should not change tool cards.\n"
            "- clarify: the user wants tools but the task is missing.\n"
            "- recommend: a new tool search.\n"
            "- refine: change filters or rerank the current search, e.g. free-only, cheaper, more private.\n"
            "- explain_shortlist: user asks why these/current tools were shown.\n"
            "- explain_best or pick_best: user asks which one you would choose or which is best.\n"
            "- show_alternative: user asks for another option from the current list.\n"
            "- tool_question: user asks about a specific visible tool or property, e.g. is it free, what about X.\n"
            "Return ONLY JSON with keys: action, tool, message, refined_query, filters, mode. "
            "Use refined_query for the query the backend should retrieve. Use message only for chat_only/clarify. "
            "Never write 'Consultant view'."
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
            tool = data.get("tool")
            if action in CHAT_ACTIONS or tool in CHAT_TOOLS:
                self.metrics.increment("chat_decision_model_used")
                return data
            self.metrics.increment("chat_decision_invalid")
            logger.warning("Chat planner returned an unusable decision; using rule-based fallback")
        except Exception as exc:
            logger.warning("Chat planner model failed (%s): %s", type(exc).__name__, exc)
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
        focused = focus_latest_intent(q)
        if has_shortlist and has_explicit_task(focused):
            return {"action": "recommend", "tool": "search_tools", "refined_query": focused}
        if has_shortlist and is_shortlist_explanation_query(q):
            return {"action": "explain_shortlist"}
        if has_shortlist and is_explanation_query(q):
            return {"action": "explain_best"}
        if has_shortlist and is_pick_best_query(q):
            return {"action": "pick_best"}
        if has_shortlist and alternative_requests_new_search(q):
            return {"action": "recommend", "refined_query": q}
        if has_shortlist and is_alternative_query(q):
            return {"action": "show_alternative"}
        if has_shortlist and (is_specific_tool_query(q) or is_criterion_pick_query(q)):
            return {"action": "tool_question"}

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

    def _hybrid_candidates(
        self,
        q: str,
        faiss_scores: list[float],
        faiss_ids: list[int],
        retrieve_k: int,
    ) -> list[dict[str, Any]]:
        """Merge semantic FAISS hits with exact keyword/category hits before reranking."""
        by_id: dict[int, dict[str, Any]] = {}
        for candidate in self._build_candidates(faiss_scores, faiss_ids):
            by_id[int(candidate["id"])] = candidate

        keyword_limit = min(len(self.store.meta), max(retrieve_k, 50))
        for keyword_score, id_ in keyword_scores(q, keyword_limit, self.store.meta):
            candidate = by_id.get(id_)
            if candidate is None:
                by_id[id_] = self._candidate_for_id(id_, keyword_score)
                by_id[id_]["retrieval_source"] = "keyword"
            else:
                candidate["score"] = max(float(candidate.get("score", 0.0)), float(keyword_score))
                candidate["retrieval_source"] = "hybrid"

        merged = list(by_id.values())
        merged.sort(key=lambda candidate: float(candidate.get("score", 0.0)), reverse=True)
        return merged[: min(len(merged), max(retrieve_k * 2, retrieve_k + 10))]

    def _candidate_for_id(self, id_: int, score: float) -> dict[str, Any]:
        m = self.store.meta[id_]
        return {
            "id": id_,
            "score": float(score),
            "retrieval_source": "faiss",
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
        except Exception as exc:
            logger.warning(
                "OpenAI ranking failed (%s): %s; returning FAISS fallback recommendations",
                type(exc).__name__,
                exc,
            )
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
        open_source_only = requires_open_source(q)
        self_hosted_required = requires_self_hosted(q)
        local_only_required = requires_local_only(q) and not self_hosted_required
        strict_free = requires_strict_free(q)
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
                if strict_free and not is_completely_free_tool(meta):
                    continue
                if open_source_only and not is_open_source_tool(meta):
                    continue
                if local_only_required and not is_local_only_tool(meta):
                    continue
                if self_hosted_required and not is_self_hosted_tool(meta):
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

# === FastAPI App ===
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    metrics = RuntimeMetrics()
    try:
        store = load_tool_store(settings)
    except Exception:
        logger.exception("Failed to load search index or metadata")
        raise

    if not settings.openai_api_key:
        logger.warning(
            "OPENAI_API_KEY is not set. The advisor will fall back to keyword search and "
            "template replies for every request (no LLM planning, conversation, or ranking). "
            "Set OPENAI_API_KEY in the environment to enable the intelligent chatbot."
        )

    client = OpenAI(
        api_key=settings.openai_api_key or "missing",
        timeout=settings.openai_timeout,
        max_retries=settings.openai_max_retries,
    )
    app.state.settings = settings
    app.state.metrics = metrics
    app.state.recommender = RecommendationService(store, client, settings, metrics)
    yield


app = FastAPI(title="AI Tools Search API", lifespan=lifespan)


RECOMMENDER_CONTRACT = {
    "style": "gpt_wrapper",
    "planner": "OpenAI intent planner decides whether to chat, answer visible-tool questions, refine, or search",
    "conversation": "OpenAI chat-only responder uses recent history and visible tool context without replacing cards",
    "retrieval": "FAISS vector search over embedded tool metadata",
    "diversification": "MMR reranking for varied, non-duplicate candidates",
    "generation": "RAG ranking with the chat model using retrieved tool records only",
    "tool_card_fields": [
        "score",
        "why",
        "tradeoff",
        "best_for",
        "fit_label",
        "meta.Name",
        "meta.Categories",
        "meta.Price",
        "meta.Description",
        "meta.Tool_link",
        "meta.Logo_URL",
        "meta.Logo_File",
    ],
}


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=400, content={"detail": exc.errors()})


def service(request: Request) -> RecommendationService:
    recommender: Any = getattr(request.app.state, "recommender", None)
    if recommender is None:
        raise HTTPException(status_code=503, detail="Search service is not ready")
    return recommender


def clean_payload(payload: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(payload)
    for key in ("message", "question"):
        if key in cleaned and cleaned[key]:
            cleaned[key] = clean_assistant_message(cleaned[key])
    hits = cleaned.get("hits")
    if isinstance(hits, list):
        cleaned_hits = []
        for hit in hits:
            if not isinstance(hit, dict):
                cleaned_hits.append(hit)
                continue
            next_hit = dict(hit)
            for key in ("why", "tradeoff", "best_for"):
                if key in next_hit and next_hit[key]:
                    next_hit[key] = clean_assistant_message(next_hit[key])
            cleaned_hits.append(next_hit)
        cleaned["hits"] = cleaned_hits
    return cleaned


def chat_wrapper_payload(payload: dict[str, Any]) -> dict[str, Any]:
    cleaned = clean_payload(payload)
    cleaned.setdefault("contract", RECOMMENDER_CONTRACT)
    return cleaned


@app.get("/health")
def health(request: Request):
    return service(request).health()


@app.get("/")
def root(request: Request):
    health_payload = service(request).health()
    return {
        "name": "AI Tools Search API",
        "ok": health_payload["ok"],
        "health": "/health",
        "metrics": "/metrics",
        "chat": "/chat",
        "style": "gpt_wrapper",
    }


@app.get("/metrics")
def metrics(request: Request):
    recommender = service(request)
    return {
        **request.app.state.metrics.snapshot(),
        "cache": recommender.cache_stats(),
    }


@app.post("/search", response_model=SearchResponse)
def search(body: SearchRequest, request: Request):
    return service(request).search(body.q, body.k)


@app.post("/recommend", response_model=RecommendResponse)
def recommend(body: RecommendRequest, request: Request):
    return chat_wrapper_payload(service(request).recommend(
        body.q,
        body.retrieve_k,
        body.final_k,
        filters=getattr(body, "filters", None),
        mode=getattr(body, "mode", "balanced"),
        conversation_id=getattr(body, "conversation_id", None),
        history=getattr(body, "history", None),
    ))


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest, request: Request):
    return chat_wrapper_payload(service(request).chat(
        body.q,
        body.retrieve_k,
        body.final_k,
        filters=getattr(body, "filters", None),
        mode=getattr(body, "mode", "balanced"),
        conversation_id=getattr(body, "conversation_id", None),
        history=getattr(body, "history", None),
        visible_tools=getattr(body, "visible_tools", None),
    ))


@app.post("/clarify", response_model=ClarifyResponse)
def clarify(body: ClarifyRequest, request: Request):
    return clean_payload(service(request).clarify(
        body.q,
        conversation_id=getattr(body, "conversation_id", None),
        history=getattr(body, "history", None),
    ))


@app.post("/detect_intent", response_model=IntentResponse)
def detect_intent(body: IntentRequest, request: Request):
    return clean_payload(service(request).detect_intent(
        body.prompt,
        body.last_query,
        conversation_id=getattr(body, "conversation_id", None),
        history=getattr(body, "history", None),
    ))
