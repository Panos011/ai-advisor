"""Open-vocabulary, source-backed capability claims for CommAI tools.

The recommender must reason over what a tool's record actually says, without
rounding either the user's need or the product into a fixed capability taxonomy.
This module turns existing catalogue fields into small, atomic evidence claims.
It does not decide what the claims mean; embeddings retrieve them and the
existing ranker performs requirement-to-claim entailment in one batched call.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any


CLAIM_SOURCE_FIELDS: tuple[tuple[str, str], ...] = (
    ("Description", "description"),
    ("Features", "features"),
    ("Use_cases", "use_cases"),
    ("Pros", "pros"),
    ("Unique_Value", "unique_value"),
)

_CLAIM_SPLIT = re.compile(r"\s*(?:\||[\r\n]+|•|(?<=[.!?])\s+)\s*")
_LEADING_LABEL = re.compile(r"^[A-Za-z][A-Za-z0-9 /&+\-]{1,48}:\s+")
_SPACE = re.compile(r"\s+")


def normalize_claim_text(value: Any) -> str:
    """Whitespace-normalize source text without paraphrasing it."""
    return _SPACE.sub(" ", str(value or "")).strip()


def _bounded_quote(value: str, max_chars: int = 280) -> str:
    if len(value) <= max_chars:
        return value
    shortened = value[:max_chars].rsplit(" ", 1)[0].rstrip(" ,;:")
    # Keep this an exact substring of the source. Presentation layers may add
    # an ellipsis, but provenance data itself must never contain invented text.
    return shortened if shortened else value[:max_chars]


def _claim_id(tool_id: int | str, source_field: str, quote: str) -> str:
    digest = hashlib.sha1(quote.casefold().encode("utf-8")).hexdigest()[:12]
    return f"tool:{tool_id}:{source_field}:{digest}"


def _claim_parts(value: Any) -> list[str]:
    source = normalize_claim_text(value)
    if not source:
        return []
    parts = []
    seen: set[str] = set()
    for raw in _CLAIM_SPLIT.split(source):
        quote = normalize_claim_text(raw)
        if not quote:
            continue
        # Feature fields commonly use "Feature name: explanation". The label is
        # navigation, not evidence; keep the full exact quote for display but use
        # the meaningful body when deciding whether the fragment is substantial.
        meaningful = _LEADING_LABEL.sub("", quote)
        words = re.findall(r"[A-Za-z0-9][A-Za-z0-9+\-]*", meaningful)
        if len(words) < 3 or len(meaningful) < 16:
            continue
        key = quote.casefold()
        if key in seen:
            continue
        seen.add(key)
        parts.append(_bounded_quote(quote))
    return parts


def extract_evidence_claims(
    meta: dict[str, Any],
    tool_id: int | str,
    max_claims: int = 6,
) -> list[dict[str, Any]]:
    """Extract bounded atomic claims with exact field-level provenance.

    Claims remain in the catalogue's own words. `catalogue_record` deliberately
    does not imply independent verification; stronger Tool Passport/graph
    evidence can replace it later without changing the matching architecture.
    """
    if max_claims <= 0:
        return []

    name = normalize_claim_text(meta.get("Name"))
    source_url = normalize_claim_text(meta.get("Source_URL") or meta.get("Tool_link"))
    claims: list[dict[str, Any]] = []
    seen_quotes: set[str] = set()

    # Preserve breadth: take one useful claim from each populated field before
    # filling remaining slots. Otherwise a long description can crowd out the
    # structured feature and use-case evidence.
    per_field: list[tuple[str, list[str]]] = []
    for meta_field, source_field in CLAIM_SOURCE_FIELDS:
        quotes = _claim_parts(meta.get(meta_field))
        if quotes:
            per_field.append((source_field, quotes))

    for source_field, quotes in per_field:
        quote = quotes[0]
        key = quote.casefold()
        if key in seen_quotes:
            continue
        seen_quotes.add(key)
        claims.append({
            "id": _claim_id(tool_id, source_field, quote),
            "tool_id": int(tool_id) if str(tool_id).isdigit() else str(tool_id),
            "tool_name": name,
            "statement": quote,
            "source_field": source_field,
            "source_quote": quote,
            "source_url": source_url or None,
            "evidence_status": "catalogue_record",
        })
        if len(claims) >= max_claims:
            return claims

    for source_field, quotes in per_field:
        for quote in quotes[1:]:
            key = quote.casefold()
            if key in seen_quotes:
                continue
            seen_quotes.add(key)
            claims.append({
                "id": _claim_id(tool_id, source_field, quote),
                "tool_id": int(tool_id) if str(tool_id).isdigit() else str(tool_id),
                "tool_name": name,
                "statement": quote,
                "source_field": source_field,
                "source_quote": quote,
                "source_url": source_url or None,
                "evidence_status": "catalogue_record",
            })
            if len(claims) >= max_claims:
                return claims
    return claims


def claim_embedding_text(claim: dict[str, Any], categories: str = "") -> str:
    """Contextual text embedded for requirement-to-claim retrieval."""
    return normalize_claim_text(
        f"Tool: {claim.get('tool_name', '')}. "
        f"Categories: {categories}. "
        f"Capability evidence: {claim.get('statement', '')}"
    )
