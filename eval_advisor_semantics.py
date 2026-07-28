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


def hit_blob(hit: Any) -> str:
    """Everything the catalogue records about a returned tool, lowercased."""
    if not isinstance(hit, dict):
        return ""
    meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
    return normalized(" ".join(str(value or "") for value in meta.values()))


def score_retrieval(case: dict[str, Any], hits: list[Any]) -> tuple[float | None, list[str]]:
    """Did the shortlist actually contain the right KIND of tool?

    The rest of the scoring reads intent, not results, so a shortlist of the
    wrong products scored identically to the right ones. Both of the failures
    found by hand on 2026-07-27 were invisible here: an invoice query returning
    nothing scored the same as any other empty answer, and a transcription query
    answered with music editors — Otter.ai and Fireflies.ai filtered out — lost
    no points at all.

    Deliberately expressed as evidence regexes over the returned records rather
    than expected product names, per this file's topic-agnostic rule: a new
    catalogue tool should be able to improve the answer without a rewrite here.
    """
    checks: list[tuple[str, bool]] = []

    any_pattern = case.get("hit_evidence")
    if any_pattern:
        rx = re.compile(any_pattern, re.I)
        checks.append((
            f"no result matches /{any_pattern}/",
            any(rx.search(hit_blob(hit)) for hit in hits),
        ))

    top_pattern = case.get("top_hit_evidence")
    if top_pattern:
        rx = re.compile(top_pattern, re.I)
        checks.append((
            f"top result does not match /{top_pattern}/",
            bool(hits) and bool(rx.search(hit_blob(hits[0]))),
        ))

    reject_pattern = case.get("reject_evidence")
    if reject_pattern:
        rx = re.compile(reject_pattern, re.I)
        offenders = [h for h in hits if rx.search(hit_blob(h))]
        checks.append((
            f"{len(offenders)} result(s) match rejected /{reject_pattern}/",
            not offenders,
        ))

    if not checks:
        return None, []
    passed = sum(1 for _, ok in checks if ok)
    return passed / len(checks), [msg for msg, ok in checks if not ok]


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

    # Retrieval fitness is optional and renormalised, so cases that do not
    # declare it score exactly as they did before this dimension existed and
    # historical numbers stay comparable.
    retrieval_fraction, retrieval_failures = score_retrieval(case, hits)
    if retrieval_fraction is not None:
        score = round((score + 25 * retrieval_fraction) / 125 * 100)
        failures.extend(retrieval_failures)

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
    retrieval_scores: list[float] = []
    retrieval_failures: list[str] = []
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
            hits = response.get("hits") if isinstance(response.get("hits"), list) else []
            fraction, _ = score_retrieval(case, hits)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            score, failures = 0, [f"request failed: {type(exc).__name__}"]
            fraction = 0.0 if (case.get("hit_evidence") or case.get("top_hit_evidence")
                               or case.get("reject_evidence")) else None
        scores.append(score)
        if fraction is not None:
            retrieval_scores.append(fraction)
            if fraction < 1.0:
                retrieval_failures.append(case["id"])
        suffix = f" — {', '.join(failures)}" if failures else ""
        print(f"{case['id']}: {score}/100{suffix}")
        if index + 1 < len(cases):
            time.sleep(args.delay)

    overall = sum(scores) / len(scores) if scores else 0.0
    print(f"\nAdvisor semantic quality: {overall:.1f}/100")

    # Reported separately on purpose. Averaged into the headline it would move
    # it by about a point per broken case, which is under the run-to-run noise
    # on this benchmark — the reason a shortlist of the wrong products went
    # unnoticed for as long as it did. Here a regression is unmissable.
    if retrieval_scores:
        clean = sum(1 for f in retrieval_scores if f == 1.0)
        print(
            f"Retrieval fitness: {clean}/{len(retrieval_scores)} cases returned "
            f"the right kind of tool"
        )
        if retrieval_failures:
            print(f"  failing: {', '.join(retrieval_failures)}")
    return 0 if overall >= args.minimum else 1


if __name__ == "__main__":
    raise SystemExit(main())
