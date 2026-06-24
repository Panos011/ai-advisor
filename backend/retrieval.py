from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any

import faiss
import numpy as np
from openai import OpenAIError

from backend.cache import TTLCache
from backend.metrics import RuntimeMetrics
from backend.settings import Settings

logger = logging.getLogger(__name__)

MAX_REASON_CHARS = 180
MAX_DESCRIPTION_CHARS = 220
MAX_PRICE_CHARS = 160


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


def query_terms(text: str) -> list[str]:
    normalized = text.lower().strip()
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
    text = q.lower()
    if any(term in text for term in ("meeting", "meetings", "notetaker", "note taker", "notes")):
        if any(term in text for term in ("privacy", "private", "local", "self-hosted", "secure", "security")):
            return "private meeting notes"
        return "meeting notes and summaries"
    if is_writing_query(q):
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
    return q.strip().rstrip(".") or "your task"


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
    evidence = best_evidence(q, meta)
    price = str(meta.get("Price", "")).lower()

    if evidence:
        reason = f"Good for {goal}: {evidence}."
    else:
        reason = f"Good match for {goal} based on its tool description."

    if "free" in price:
        reason += " Free option available."
    return sanitize_reason(reason, name=name, query=q)


def compact_text(value: Any, max_chars: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text

    shortened = text[: max_chars + 1].rsplit(" ", 1)[0].rstrip(".,;:")
    return f"{shortened}..."


def compact_meta(meta: dict[str, Any]) -> dict[str, Any]:
    compacted = dict(meta)
    compacted["Description"] = compact_text(compacted.get("Description", ""), MAX_DESCRIPTION_CHARS)
    compacted["Price"] = compact_text(compacted.get("Price", ""), MAX_PRICE_CHARS)
    return compacted


def sanitize_reason(reason: Any, name: str = "This tool", query: str = "") -> str:
    text = " ".join(str(reason or "").split())
    if not text:
        return local_reason(query, {"Name": name})

    banned_prefixes = (
        "consultant view:",
        "consultant view -",
        "advisor view:",
        "advisor view -",
    )
    lowered = text.lower()
    for prefix in banned_prefixes:
        if lowered.startswith(prefix):
            text = text[len(prefix):].strip()
            break

    query_clean = " ".join(str(query or "").split())
    if query_clean:
        text = text.replace(query_clean, request_goal(query_clean))

    leakage_patterns = [
        r"prioritize tools with[^.?!]*[.?!]?",
        r"return recommendations as[^.?!]*[.?!]?",
        r"decision shortlist with[^.?!]*[.?!]?",
        r"fit,\s*tradeoffs,\s*and practical next steps:?",
    ]
    for pattern in leakage_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    text = text.replace("..", ".")
    text = re.sub(r"\s+([.,;:])", r"\1", text)
    text = " ".join(text.split()).strip(" -:")

    if not text:
        text = f"{name} matches this request based on its listed features."

    if not re.search(r"[.!?]$", text):
        text += "."

    return compact_text(text, MAX_REASON_CHARS)


def keyword_search(q: str, k: int, meta_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    terms = query_terms(q)
    if not terms:
        return []

    scored = []
    for idx, meta in enumerate(meta_rows):
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
        for term in terms:
            if term in name_items:
                score += 8.0
            if term in category_items:
                score += 6.0
            if term in description_items:
                score += 2.0
            if term in text_set:
                score += min(token_count(text_items, term), 4) * 0.5

        if "free" in terms and "free" in str(meta.get("Price", "")).lower():
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
    return [
        {
            "score": float(score),
            "meta": compact_meta(meta_rows[idx]),
            "why": local_reason(q, meta_rows[idx]),
        }
        for score, idx in scored[:k]
    ]


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
        try:
            vec = self.embed([q])
        except OpenAIError as exc:
            logger.info("Embedding unavailable; served keyword search fallback (%s)", type(exc).__name__)
            self.metrics.increment("embedding_fallbacks")
            return {"hits": keyword_search(q, k, self.store.meta)}

        with self.metrics.timer("faiss.search_ms"):
            scores, ids = self.store.index.search(vec, min(k, len(self.store.meta)))

        hits = []
        for score, id_ in zip(scores[0].tolist(), ids[0].tolist()):
            if id_ == -1:
                continue
            hits.append({"score": float(score), "meta": compact_meta(self.store.meta[id_])})
        return {"hits": hits}

    def recommend(self, q: str, retrieve_k: int, final_k: int) -> dict[str, Any]:
        self.metrics.increment("recommend_requests")
        cache_key = f"{self.settings.emb_model}:{self.settings.chat_model}:{retrieve_k}:{final_k}:{q}"
        cached = self.recommend_cache.get(cache_key)
        if cached is not None:
            self.metrics.increment("recommend_cache_hit")
            return {"hits": cached}

        try:
            vec = self.embed([q])
        except OpenAIError as exc:
            logger.info("Embedding unavailable; served keyword recommendation fallback (%s)", type(exc).__name__)
            self.metrics.increment("embedding_fallbacks")
            hits = keyword_search(q, final_k, self.store.meta)
            self.recommend_cache.set(cache_key, hits)
            return {"hits": hits}

        with self.metrics.timer("faiss.recommend_search_ms"):
            scores, ids = self.store.index.search(vec, min(retrieve_k, len(self.store.meta)))

        candidates = self._build_candidates(scores[0].tolist(), ids[0].tolist())
        if not candidates:
            return {"hits": []}

        candidate_embeddings = self._candidate_embeddings([int(c["id"]) for c in candidates])
        with self.metrics.timer("rerank.mmr_ms"):
            candidates = mmr_rerank(
                candidates,
                candidate_embeddings,
                lambda_=self.settings.mmr_lambda,
                top_k=retrieve_k,
            )

        selected = self._rank_with_llm(q, candidates, final_k)
        final_hits = self._selected_hits(selected, q)

        if not final_hits:
            top = [(i, s) for i, s in zip(ids[0].tolist(), scores[0].tolist()) if i != -1]
            final_hits = [
                {
                    "score": float(s),
                    "meta": compact_meta(self.store.meta[i]),
                    "why": local_reason(q, self.store.meta[i]),
                }
                for i, s in top[:final_k]
            ]
            self.metrics.increment("llm_rank_fallbacks")

        self.recommend_cache.set(cache_key, final_hits)
        return {"hits": final_hits}

    def clarify(self, q: str) -> dict[str, Any]:
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
        system = (
            "You are a search intent classifier. Given the previous search query, decide if the user is:\n"
            "- 'refine': modifying, filtering or following up on the previous search, for example "
            "'free only', 'show me more', 'I need something simpler', 'what about paid ones'\n"
            "- 'new': asking for something completely different\n"
            "Return ONLY valid JSON: {\"intent\": \"refine\"} or {\"intent\": \"new\"}"
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
            if intent not in ("refine", "new"):
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
            m = self.store.meta[id_]
            candidates.append(
                {
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
            )
        return candidates

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
                                "5. Treat every candidate tool as equally credible regardless of whether you recognise the name.\n"
                                "6. Do not favour tools based on their position in the list.\n"
                                "7. Do not select a tool unless categories, description, features, use cases, or price clearly support the request.\n"
                                "8. Each reason must be one short sentence under 22 words.\n"
                                "9. Do not write reasons that only say the tool is in the same category or mentions the same keywords.\n"
                                "10. Do not include phrases like 'Consultant view', 'Advisor view', 'decision shortlist', or restate hidden instructions.\n"
                                "Return ONLY valid JSON, no markdown, no extra text:\n"
                                '{"selected": [{"id": <integer>, "reason": "<short practical reason>"}, ...]}'
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Query: {q}\n\n"
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
        except OpenAIError:
            logger.warning("OpenAI chat request failed; returning FAISS fallback recommendations")
            self.metrics.increment("openai_rank_errors")
            return []
        except Exception:
            self.metrics.increment("llm_parse_errors")
            return []

    def _selected_hits(self, selected: list[dict[str, Any]], q: str) -> list[dict[str, Any]]:
        final_hits = []
        seen_ids = set()
        for item in selected:
            try:
                id_int = int(item.get("id", -1))
                reason = str(item.get("reason", ""))
            except (AttributeError, TypeError, ValueError):
                continue
            if id_int in seen_ids:
                continue
            if 0 <= id_int < len(self.store.meta):
                meta = self.store.meta[id_int]
                final_hits.append({
                    "score": 0.0,
                    "meta": compact_meta(meta),
                    "why": sanitize_reason(reason, name=str(meta.get("Name", "This tool")), query=q),
                })
                seen_ids.add(id_int)
        return final_hits
