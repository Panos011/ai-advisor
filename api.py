import os
import json
import faiss
import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from openai import OpenAI

# INDEX_PATH contains the veector for each tool, META_PATH contains the metadata for each tool
INDEX_PATH = "index/tools.faiss"
META_PATH = "index/meta.jsonl"
EMB_MODEL = os.getenv("EMB_MODEL", "text-embedding-3-large")

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
