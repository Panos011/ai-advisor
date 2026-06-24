import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from openai import OpenAI

from backend.metrics import RuntimeMetrics
from backend.retrieval import RecommendationService, load_tool_store
from backend.schemas import (
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


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=400, content={"detail": exc.errors()})


def service(request: Request) -> RecommendationService:
    recommender: Any = getattr(request.app.state, "recommender", None)
    if recommender is None:
        raise HTTPException(status_code=503, detail="Search service is not ready")
    return recommender


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
    return service(request).recommend(
        body.q,
        body.retrieve_k,
        body.final_k,
        filters=getattr(body, "filters", None),
        mode=getattr(body, "mode", "balanced"),
    )


@app.post("/clarify", response_model=ClarifyResponse)
def clarify(body: ClarifyRequest, request: Request):
    return service(request).clarify(body.q)


@app.post("/detect_intent", response_model=IntentResponse)
def detect_intent(body: IntentRequest, request: Request):
    return service(request).detect_intent(body.prompt, body.last_query)
