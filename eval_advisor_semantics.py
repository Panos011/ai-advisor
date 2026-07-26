"""Score live Advisor responses on meaning coverage and evidence grounding.

This is deliberately topic-agnostic: cases describe expected meanings rather
than expected product names. New catalogue tools may therefore improve the
answer without forcing a benchmark rewrite.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_URL = "https://ai-advisor-d0op.onrender.com/chat"


def normalized(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def score_case(case: dict[str, Any], response: dict[str, Any]) -> tuple[int, list[str]]:
    score = 0
    failures: list[str] = []
    action = response.get("action")
    hits = response.get("hits") if isinstance(response.get("hits"), list) else []
    intent = response.get("intent") if isinstance(response.get("intent"), dict) else {}
    requirements = (
        intent.get("requirements")
        if isinstance(intent.get("requirements"), list)
        else []
    )

    if action in {"recommend", "refine"} and hits:
        score += 20
    else:
        failures.append("no recommendation result")

    if intent.get("goal") and requirements:
        score += 20
    else:
        failures.append("missing structured functional intent")

    intent_text = normalized(
        " ".join(
            [
                str(intent.get("goal") or ""),
                *[str(value) for value in intent.get("inputs") or []],
                *[str(value) for value in intent.get("outputs") or []],
                *[str(item.get("statement") or "") for item in requirements if isinstance(item, dict)],
            ]
        )
    )
    groups = case.get("meaning_groups") or []
    matched_groups = sum(
        1
        for alternatives in groups
        if any(normalized(term) in intent_text for term in alternatives)
    )
    if groups:
        score += round(30 * matched_groups / len(groups))
    if matched_groups != len(groups):
        failures.append(f"meaning coverage {matched_groups}/{len(groups)}")

    requirement_ids = {
        str(item.get("id"))
        for item in requirements
        if isinstance(item, dict) and item.get("id")
    }
    top_assessments = (
        hits[0].get("requirement_assessments")
        if hits and isinstance(hits[0], dict)
        else []
    )
    assessment_ids = {
        str(item.get("requirement_id"))
        for item in top_assessments or []
        if isinstance(item, dict)
    }
    complete = bool(requirement_ids) and requirement_ids == assessment_ids
    evidence_valid = True
    supported = 0
    for assessment in top_assessments or []:
        if not isinstance(assessment, dict):
            evidence_valid = False
            continue
        status = assessment.get("status")
        evidence = assessment.get("evidence")
        if status in {"supported", "partial", "unsupported"}:
            supported += status in {"supported", "partial"}
            if not isinstance(evidence, list) or not evidence:
                evidence_valid = False
            elif any(
                not isinstance(claim, dict)
                or not claim.get("id")
                or not claim.get("source_quote")
                for claim in evidence
            ):
                evidence_valid = False
    if complete:
        score += 15
    else:
        failures.append("top result does not assess every requirement")
    if evidence_valid and supported:
        score += 15
    else:
        failures.append("top result lacks source-backed support")
    return score, failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--cases", default="advisor_semantic_cases.jsonl")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--minimum", type=float, default=90.0)
    parser.add_argument("--delay", type=float, default=0.25)
    args = parser.parse_args()

    cases = [
        json.loads(line)
        for line in Path(args.cases).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    scores = []
    for index, case in enumerate(cases):
        try:
            response = post_json(
                args.url,
                {
                    "q": case["query"],
                    "retrieve_k": 80,
                    "final_k": 5,
                    "mode": "best_fit",
                    "conversation_id": f"semantic-eval-{case['id']}-{int(time.time())}",
                },
                args.timeout,
            )
            score, failures = score_case(case, response)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            score, failures = 0, [f"request failed: {type(exc).__name__}"]
        scores.append(score)
        suffix = f" — {', '.join(failures)}" if failures else ""
        print(f"{case['id']}: {score}/100{suffix}")
        if index + 1 < len(cases):
            time.sleep(args.delay)

    overall = sum(scores) / len(scores) if scores else 0.0
    print(f"\nAdvisor semantic quality: {overall:.1f}/100")
    return 0 if overall >= args.minimum else 1


if __name__ == "__main__":
    raise SystemExit(main())
