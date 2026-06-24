from __future__ import annotations

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
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
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

