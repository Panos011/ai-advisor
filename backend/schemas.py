from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


MAX_QUERY_LENGTH = 500


def _clean_required_text(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("must not be empty")
    return cleaned


class IntentRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=MAX_QUERY_LENGTH)
    last_query: str = Field("", max_length=MAX_QUERY_LENGTH)

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


class RecommendRequest(BaseModel):
    q: str = Field(..., min_length=1, max_length=MAX_QUERY_LENGTH)
    retrieve_k: int = Field(30, ge=1, le=100)
    final_k: int = Field(5, ge=1, le=10)

    @field_validator("q")
    @classmethod
    def clean_query(cls, value: str) -> str:
        return _clean_required_text(value)

    @model_validator(mode="after")
    def validate_k_values(self) -> "RecommendRequest":
        if self.final_k > self.retrieve_k:
            raise ValueError("final_k must be less than or equal to retrieve_k")
        return self


class SearchHit(BaseModel):
    score: float
    meta: dict[str, Any]
    why: str | None = None


class RecommendResponse(BaseModel):
    hits: list[SearchHit]
    message: str | None = None


class SearchResponse(BaseModel):
    hits: list[SearchHit]


class ClarifyRequest(BaseModel):
    q: str = Field(..., min_length=1, max_length=MAX_QUERY_LENGTH)

    @field_validator("q")
    @classmethod
    def clean_query(cls, value: str) -> str:
        return _clean_required_text(value)


class ClarifyResponse(BaseModel):
    action: Literal["clarify", "explain", "search"]
    question: str | None = None
    refined_query: str | None = None
