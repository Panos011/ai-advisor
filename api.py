import os
import json
import logging
import re
import faiss
import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from openai import OpenAI, OpenAIError
from typing import Literal, Optional, List, Dict, Any

# INDEX_PATH contains the veector for each tool, META_PATH contains the metadata for each tool
INDEX_PATH = "index/tools.faiss"
META_PATH = "index/meta.jsonl"
EMB_MODEL = os.getenv("EMB_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-5.4-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
logger = logging.getLogger(__name__)

# Tiny app with two doors /health and /search
app = FastAPI(title="AI Tools Search API")

# Opening of the list with the vectors and reading their metadata
index = faiss.read_index(INDEX_PATH)
with open(META_PATH, "r", encoding="utf-8") as f:
    META = [json.loads(line) for line in f]

client = OpenAI(api_key=OPENAI_API_KEY or "missing")


class IntentRequest(BaseModel):
    prompt: str
    last_query: str


class IntentResponse(BaseModel):
    intent: str


class SearchRequest(BaseModel):
    q: str  # The questions
    k: int = 30  # How many results we need


class RecommendRequest(BaseModel):
    q: str
    retrieve_k: int = 30
    final_k: int = 5


class SearchHit(BaseModel):
    score: float  # How close the match is
    meta: dict  # The metadata of the tool
    why: Optional[str] = None


class RecommendResponse(BaseModel):
    hits: list[SearchHit]


class SearchResponse(BaseModel):
    hits: list[SearchHit]


class ClarifyRequest(BaseModel):
    q: str


class ClarifyResponse(BaseModel):
    action: Literal["clarify", "search"]
    question: Optional[str] = None
    refined_query: Optional[str] = None
# It checks for health and returns how many tools were loaded


@app.get("/health")
def health():
    return {"ok": True, "items": len(META), "openai_configured": bool(OPENAI_API_KEY)}
#  It turns texts into numbers so computer can measure closeness


def embed(texts: list[str]) -> np.ndarray:
    resp = client.embeddings.create(model=EMB_MODEL, input=texts)
    vecs = np.array([d.embedding for d in resp.data], dtype="float32")
    faiss.normalize_L2(vecs)
    return vecs


def query_terms(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    stopwords = {
        "a", "an", "and", "are", "as", "at", "be", "for", "from", "i", "in",
        "is", "it", "me", "my", "of", "on", "or", "that", "the", "to",
        "tool", "tools", "with", "you"
    }
    return [w for w in words if len(w) > 2 and w not in stopwords]


def meta_text(meta: dict) -> str:
    fields = (
        "Name", "Categories", "Description", "Features", "Pros", "Cons",
        "Use_cases", "Price"
    )
    return " ".join(str(meta.get(field, "")) for field in fields).lower()


def keyword_search(q: str, k: int) -> list[SearchHit]:
    terms = query_terms(q)
    if not terms:
        return []

    scored = []
    for idx, meta in enumerate(META):
        text = meta_text(meta)
        name = str(meta.get("Name", "")).lower()
        categories = str(meta.get("Categories", "")).lower()
        score = 0.0
        for term in terms:
            if term in name:
                score += 5.0
            if term in categories:
                score += 3.0
            score += min(text.count(term), 5)
        if score > 0:
            scored.append((score, idx))

    scored.sort(reverse=True, key=lambda item: item[0])
    return [
        {
            "score": float(score),
            "meta": META[idx],
            "why": "Matched locally because the AI ranking service is temporarily unavailable.",
        }
        for score, idx in scored[:k]
    ]

# Decision logic to reduce bias and promote fairness #


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


def mmr_rerank(
        candidates: List[Dict[str, Any]],
        embeddings: np.ndarray,
        lambda_: float = 0.7,
        top_k: int = 30,
) -> List[Dict[str, Any]]:
    """ Maximal Marginal Relevance reranking"""
    n = len(candidates)
    if n == 0:
        return []
    selected_indices: List[int] = []
    remaining_indices: List[int] = list(range(n))

    # Normalise relevance scores to [0, 1] for fair weighting
    max_score = max(c["score"] for c in candidates) or 1.0
    relevance = np.array([c["score"] / max_score for c in candidates])

    for _ in range(min(top_k, n)):
        best_idx = None
        best_mmr = -float("inf")

        for idx in remaining_indices:
            rel = relevance[idx]

            # Max similarity to any already-selected item
            if not selected_indices:
                max_sim = 0.0
            else:
                sims = [
                    cosine_similarity(embeddings[idx], embeddings[s])
                    for s in selected_indices
                ]
                max_sim = max(sims)
            mmr_score = lambda_ * rel - (1 - lambda_) * max_sim

            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = idx
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)
    return [candidates[i] for i in selected_indices]


# It searches for the best matches, and it returns the best 5 matches and their metadata


@app.post("/search", response_model=SearchResponse)
def search(body: SearchRequest):
    q = body.q.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Query 'q' is empty")

    try:
        vec = embed([q])
    except OpenAIError:
        logger.exception("OpenAI embedding request failed; using keyword search fallback")
        return {"hits": keyword_search(q, body.k)}

    scores, ids = index.search(vec, min(body.k, len(META)))
    hits = []
    for score, id_ in zip(scores[0].tolist(), ids[0].tolist()):
        if id_ == -1:
            continue
        hits.append({"score": float(score), "meta": META[id_]})

    return {"hits": hits}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(body: RecommendRequest):
    q = body.q.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Query 'q' is empty")

    try:
        vec = embed([q])
    except OpenAIError:
        logger.exception("OpenAI embedding request failed; using keyword recommendation fallback")
        return {"hits": keyword_search(q, body.final_k)}

    scores, ids = index.search(vec, min(body.retrieve_k, len(META)))
    candidates = []
    for score, id_ in zip(scores[0].tolist(), ids[0].tolist()):
        if id_ == -1:
            continue
        m = META[id_]
        candidates.append({
            "id": id_,
            "score": float(score),
            "name": m.get("Name", ""),
            "categories": m.get("Categories", ""),
            "price": m.get("Price", ""),
            "description": m.get("Description", "")
        })
    if not candidates:
        return {"hits": []}
    candidate_embeddings = []
    for c in candidates:
        try:
            emb = index.reconstruct(int(c["id"]))
            candidate_embeddings.append(emb)
        except Exception:
            candidate_embeddings.append(np.zeros(index.d, dtype="float32"))
    candidate_embeddings = np.array(candidate_embeddings, dtype="float32")

    candidates = mmr_rerank(candidates, candidate_embeddings, lambda_=0.7, top_k=30)

    resp = None
    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an AI tool recommender. You will receive a user's needs and a numbered list of candidate tools."
                        "Critical Evaluation Rules:"
                        "1. Evaluate only based on how well each tool's described features match the user's stated needs"
                        "2.Do not favour tools based on popularity, brand recognition or how often you seen them in your training data"
                        "3. A lesser-known tool that matching perfectly a user's needs its always a better choice to a well-known tool that partially matches"
                        "4. Consider the user's budget constraints. If they have not specified consider having at least 1 free option"
                        "5. Treat every candidate tool equally credible regardless of whether you recognise the name"
                        "6. Do not favour tools based on their position in the list. A tool at at the last place is as valid as the tool in the first place\n"
                        "Return ONLY valid JSON, no markdown, no extra text:\n"
                        '{"selected": [{"id": <integer>, "reason": "<two sentences why this fits>"}, ...]}'
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Query: {q}\n\n"
                        f"Select exactly {body.final_k} tools from these candidates:\n"
                        f"{json.dumps(candidates, ensure_ascii=False)}"
                    )
                }
            ],
            temperature=0.2,
            response_format={"type": "json_object"}  # forces valid JSON every time
        )
    except OpenAIError:
        logger.exception("OpenAI chat request failed; returning FAISS fallback recommendations")

    try:
        data = json.loads(resp.choices[0].message.content) if resp else {}
        selected = data.get("selected", [])
    except Exception:
        selected = []

    # build final hits in the chosen order
    final_hits = []
    for item in selected:
        try:
            id_int = int(item.get("id", -1))
            reason = str(item.get("reason", ""))
        except (TypeError, ValueError):
            continue
        if 0 <= id_int < len(META):
            final_hits.append({"score": 0.0, "meta": META[id_int], "why": reason})

    # fallback if LLM fails: just take top k
    if not final_hits:
        top = [(i, s) for i, s in zip(ids[0].tolist(), scores[0].tolist()) if i != -1]
        final_hits = [{"score": float(s), "meta": META[i], "why": None}
                      for i, s in top[:body.final_k]]

    return {"hits": final_hits}


@app.post("/clarify", response_model=ClarifyResponse)
def clarify(body: ClarifyRequest):
    q = body.q.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Query 'q' is empty")

    system = (
        "Decide if the user's request needs ONE clarifying question.\n"
        "If missing key info (task type, platform, free/paid, output), ask 1-3 short question(s).\n"
        "Otherwise rewrite the request into a single refined query.\n"
        "Return JSON only like:\n"
        '{"action":"clarify","question":"...","refined_query":null}\n'
        'or {"action":"search","question":null,"refined_query":"..."}'
    )

    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": q},
            ],
            temperature=0.2,
        )
        raw = resp.choices[0].message.content or ""
        data = json.loads(raw)
    except Exception:
        # fallback: just search original q
        return {"action": "search", "refined_query": q}

    action = data.get("action")
    if action == "clarify":
        question = (data.get("question") or "").strip()
        if not question:
            question = "Quick question: what exact task are you trying to do, and do you need a free tool?"
        return {"action": "clarify", "question": question}

    refined = (data.get("refined_query") or q).strip()
    return {"action": "search", "refined_query": refined}


@app.post("/detect_intent", response_model=IntentResponse)
def detect_intent(body: IntentRequest):
    system = (
        "You are a search intent classifier and give the previous search query decide if the user is:\n"
        "-'refine':modifying, filtering or following up on the previous search"
        "for example 'free only', 'show me more', 'I need something simpler', 'I need something more specific', 'what about paid ones'\n"
        "-'new': asking for something completely different\n"
        "Return ONLY valid JSON: {\"intent\": \"refine\"} or {\"intent\" : \"new\"}"
    )
    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": (
                    f"Previous search: {body.last_query}\n"
                    f"New message: {body.prompt}"
                )}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        data = json.loads(resp.choices[0].message.content)
        intent = data.get("intent", "new")
        if intent not in ("refine", "new"):
            intent = "new"
        return {"intent": intent}
    except Exception:
        return {"intent": "new"}
    
