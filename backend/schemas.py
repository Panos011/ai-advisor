from __future__ import annotations

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


class RecommendResponse(BaseModel):
    hits: list[SearchHit]
    message: str | None = None


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
