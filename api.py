import os
import json
import faiss
import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from typing import Literal, Optional

# INDEX_PATH contains the veector for each tool, META_PATH contains the metadata for each tool
INDEX_PATH = "index/tools.faiss"
META_PATH = "index/meta.jsonl"
EMB_MODEL = os.getenv("EMB_MODEL", "text-embedding-3-small")

# Tiny app with two doors /health and /search
app = FastAPI(title="AI Tools Search API")

# Opening of the list with the vectors and reading their metadata
index = faiss.read_index(INDEX_PATH)
with open(META_PATH, "r", encoding="utf-8") as f:
    META = [json.loads(line) for line in f]

client = OpenAI()


class SearchRequest(BaseModel):
    q: str  # The questions
    k: int = 5  # How many results we need


class SearchHit(BaseModel):
    score: float  # How close the match is
    meta: dict  # The metadata of the tool


class SearchResponse(BaseModel):
    hits: list[SearchHit]

class ClarifyRequest(BaseModel):
    q: str

class ClarifyResponse(BaseModel):
    action: Literal["clarify", "search"]
    question: Optional[str] = None
    refined_queryt: Optional[str] = None
# It checks for health and returns how many tools were loaded


@app.get("/health")
def health():
    return {"ok": True, "items": len(META)}
#  It turns texts into numbers so computer can measure closeness


def embed(texts: list[str]) -> np.ndarray:
    resp = client.embeddings.create(model=EMB_MODEL, input=texts)
    vecs = np.array([d.embedding for d in resp.data], dtype="float32")
    return vecs

# It searches for the best matches, and it returns the best 5 matches and their metadata


@app.post("/search", response_model=SearchResponse)
def search(body: SearchRequest):
    q = body.q.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Query 'q' is empty")
    vec = embed([q])
    scores, ids = index.search(vec, min(body.k, len(META)))
    hits = []
    for score, id_ in zip(scores[0].tolist(), ids[0].tolist()):
        if id_ == -1:
            continue
        hits.append({"score": float(score), "meta": META[id_]})

    return {"hits": hits}

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4.1-mini")

@app.post("/clarify", response_model=ClarifyResponse)
def clarify(body: ClarifyRequest):
    q = body.q.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Query 'q' is empty")

    system = (
        "Decide if the user's request needs ONE clarifying question.\n"
        "If missing key info (task type, platform, free/paid, output), ask 1 short question.\n"
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