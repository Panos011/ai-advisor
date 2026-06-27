import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from openai import OpenAI

from backend.metrics import RuntimeMetrics
from backend.retrieval import RecommendationService, clean_assistant_message, load_tool_store
from backend.schemas import (
    ChatRequest,
    ChatResponse,
    ClarifyRequest,
    ClarifyResponse,
    IntentRequest,
    IntentResponse,
    RecommendRequest,
    RecommendResponse,
    SearchRequest,
    SearchResponse,
)
from backend.settings import get_settings

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
