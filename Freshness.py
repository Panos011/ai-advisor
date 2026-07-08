"""Catalog freshness pipeline: detect dead / unreachable tools by their links.

An advisor is only as good as its catalog. Tools shut down, rebrand, or let
their domains lapse; a stale entry makes the advisor recommend something that
no longer exists — the fastest way to lose trust. This script checks every
tool's vendor link and tracks liveness across runs so the dataset stays fresher
than a static scrape.

Safety first — a single failed request NEVER prunes a tool. A tool must fail on
`--fail-threshold` consecutive runs (default 2) before it is eligible for
removal, so a momentary outage or a bot block does not delete a good listing.
`freshness_state.json` carries the consecutive-failure counters between runs.

Integration — this does not edit the catalog CSVs itself. With `--emit-deletions`
it appends prune-eligible tools to the existing `deleted_tools.json` blocklist,
which `Filter_Blocklist.py` / `Apply_Deletions.py` already consume to strip tools
from the CSVs and rebuild the index. That reuses the audited deletion path
instead of inventing a second one.

Usage:
    python Freshness.py                         # audit Clean_AI_tools.csv, write report
    python Freshness.py --limit 100 --workers 16
    python Freshness.py --emit-deletions        # also queue >=threshold-dead tools for removal
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Callable, Iterable

from Blocklist import (
    BLOCKLIST_FILE,
    clean,
    entries_match,
    load_blocklist,
    normalize_name,
    save_blocklist,
    utc_now_iso,
    website_domain,
)

DEFAULT_INPUT = "Clean_AI_tools.csv"
REPORT_FILE = "freshness_report.json"
STATE_FILE = "freshness_state.json"
DEFAULT_TIMEOUT = 12.0
DEFAULT_WORKERS = 12
DEFAULT_FAIL_THRESHOLD = 2

USER_AGENT = (
    "Mozilla/5.0 (compatible; CommAI-FreshnessBot/1.0; "
    "+https://comm-ai.net/freshness)"
)

# HTTP outcomes we treat as "the tool still exists". 401/403/405/429 mean the
# server answered but blocked our probe (bot protection, method not allowed,
# rate limit) — the domain is live, so we must NOT prune on these.
ALIVE_STATUS = {200, 201, 202, 203, 204, 206, 301, 302, 303, 307, 308, 401, 403, 405, 429}
# Definitive "this page is gone".
DEAD_STATUS = {404, 410}


def classify_status(status_code: int | None, error: str | None) -> str:
    """Map an HTTP result to alive / dead / unreachable.

    - alive: the domain answered (even if it blocked us).
    - dead: a definitive 404/410.
    - unreachable: timeouts, 5xx, DNS/connection failures — ambiguous, could be
      transient, so never pruned on a single occurrence (the consecutive-failure
      threshold decides).
    """
    if error:
        return "unreachable"
    if status_code is None:
        return "unreachable"
    if status_code in DEAD_STATUS:
        return "dead"
    if status_code in ALIVE_STATUS:
        return "alive"
    if 200 <= status_code < 400:
        return "alive"
    return "unreachable"


def check_url(url: str, session: Any = None, timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
    """Probe a single URL. HEAD first, fall back to GET when HEAD is refused.

    Never raises — always returns {url, status_code, error}. Network access
    required; injected/stubbed in tests.
    """
    import requests  # imported lazily so the module (and its tests) load without network deps

    close = False
    if session is None:
        session = requests.Session()
        close = True
    headers = {"User-Agent": USER_AGENT, "Accept": "*/*"}
    try:
        resp = session.head(url, allow_redirects=True, timeout=timeout, headers=headers)
        # Many sites don't implement HEAD correctly; confirm with a light GET.
        if resp.status_code in (403, 405, 501) or resp.status_code >= 500:
            resp = session.get(url, allow_redirects=True, timeout=timeout, headers=headers, stream=True)
            resp.close()
        return {"url": url, "status_code": int(resp.status_code), "error": None}
    except Exception as exc:  # noqa: BLE001 - any failure is a liveness signal, not a crash
        return {"url": url, "status_code": None, "error": type(exc).__name__}
    finally:
        if close:
            try:
                session.close()
            except Exception:
                pass


def load_catalog(path: str) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def audit(
    rows: Iterable[dict[str, str]],
    checker: Callable[[str], dict[str, Any]],
    workers: int = DEFAULT_WORKERS,
) -> list[dict[str, Any]]:
    """Check every row's Tool_link and return per-tool liveness results.

    `checker` takes a URL and returns {url, status_code, error}; inject a stub in
    tests. Rows without a link are reported as 'unknown' (nothing to check) and
    are never pruned.
    """
    rows = list(rows)
    indexed = []
    urls = []
    for row in rows:
        url = clean(row.get("Tool_link"))
        indexed.append((row, url))
        if url:
            urls.append(url)

    checked: dict[str, dict[str, Any]] = {}
    if urls:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for result in pool.map(checker, urls):
                checked[result["url"]] = result

    results = []
    for row, url in indexed:
        name = clean(row.get("Name"))
        if not url:
            results.append({
                "name": name,
                "url": "",
                "domain": "",
                "status": "unknown",
                "status_code": None,
                "error": "no_link",
            })
            continue
        probe = checked.get(url, {"status_code": None, "error": "not_checked"})
        results.append({
            "name": name,
            "url": url,
            "domain": website_domain(url),
            "status": classify_status(probe.get("status_code"), probe.get("error")),
            "status_code": probe.get("status_code"),
            "error": probe.get("error"),
        })
    return results


def load_state(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def state_key(result: dict[str, Any]) -> str:
    """Stable per-tool key for the failure counter (domain, else name)."""
    return result.get("domain") or normalize_name(result.get("name")) or result.get("url", "")


def update_state(state: dict[str, Any], results: list[dict[str, Any]], now: str | None = None) -> dict[str, Any]:
    """Advance the consecutive-failure counters.

    'alive' resets a tool to 0. 'dead'/'unreachable' increments it. 'unknown'
    (no link to check) leaves the counter untouched. Returns a fresh dict; the
    input is not mutated.
    """
    now = now or utc_now_iso()
    next_state: dict[str, Any] = {}
    for result in results:
        key = state_key(result)
        if not key:
            continue
        prior = state.get(key, {})
        prior_failures = int(prior.get("consecutive_failures", 0) or 0)
        status = result["status"]
        if status == "alive":
            failures = 0
        elif status in ("dead", "unreachable"):
            failures = prior_failures + 1
        else:  # unknown: nothing to check, carry the prior count forward
            failures = prior_failures

        if failures == 0:
            first_failed = None
        elif prior_failures == 0:
            first_failed = now  # transitioned from healthy to failing this run
        else:
            first_failed = prior.get("first_failed") or now

        next_state[key] = {
            "name": result.get("name", ""),
            "last_status": status,
            "consecutive_failures": failures,
            "last_checked": now,
            "first_failed": first_failed,
        }
    return next_state


def eligible_for_pruning(state: dict[str, Any], threshold: int = DEFAULT_FAIL_THRESHOLD) -> list[str]:
    """Keys that have failed on >= threshold consecutive runs."""
    return [
        key for key, entry in state.items()
        if int(entry.get("consecutive_failures", 0) or 0) >= threshold
    ]


def summarize(results: list[dict[str, Any]]) -> dict[str, int]:
    summary = {"total": len(results), "alive": 0, "dead": 0, "unreachable": 0, "unknown": 0}
    for result in results:
        summary[result["status"]] = summary.get(result["status"], 0) + 1
    return summary


def build_deletion_entries(
    results: list[dict[str, Any]],
    eligible_keys: Iterable[str],
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    """Turn prune-eligible tools into deleted_tools.json entries."""
    eligible = set(eligible_keys)
    by_key: dict[str, dict[str, Any]] = {}
    for result in results:
        key = state_key(result)
        if key in eligible and key not in by_key:
            by_key[key] = {
                "toolId": normalize_name(result.get("name")),
                "name": result.get("name", ""),
                "nameKey": normalize_name(result.get("name")),
                "domainKey": result.get("domain", ""),
                "sourceUrlKey": "",
                "action": "soft_delete",
                "requestedBy": "freshness-bot",
                "deletedAt": utc_now_iso(),
                "reason": f"link_{state.get(key, {}).get('last_status', 'dead')}",
            }
    return list(by_key.values())


def merge_into_blocklist(new_entries: list[dict[str, Any]], path: str = BLOCKLIST_FILE) -> int:
    """Append newly dead tools to the blocklist, skipping ones already present.
    Returns the number actually added."""
    existing = load_blocklist(path)
    added = 0
    for entry in new_entries:
        if not any(entries_match(entry, e) for e in existing):
            existing.append(entry)
            added += 1
    if added:
        save_blocklist(existing, path)
    return added


def write_report(results: list[dict[str, Any]], summary: dict[str, int], path: str) -> None:
    payload = {
        "generated_at": utc_now_iso(),
        "summary": summary,
        "dead": [r for r in results if r["status"] == "dead"],
        "unreachable": [r for r in results if r["status"] == "unreachable"],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit catalog tool links for freshness.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Catalog CSV to audit.")
    parser.add_argument("--report", default=REPORT_FILE)
    parser.add_argument("--state", default=STATE_FILE)
    parser.add_argument("--blocklist", default=BLOCKLIST_FILE)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--limit", type=int, default=0, help="Check only the first N tools (0 = all).")
    parser.add_argument("--fail-threshold", type=int, default=DEFAULT_FAIL_THRESHOLD,
                        help="Consecutive failed runs before a tool is prune-eligible.")
    parser.add_argument("--emit-deletions", action="store_true",
                        help="Append prune-eligible tools to the blocklist for the deletion pipeline.")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(f"Catalog not found: {args.input}")

    rows = load_catalog(args.input)
    if args.limit > 0:
        rows = rows[: args.limit]

    def checker(url: str) -> dict[str, Any]:
        return check_url(url, timeout=args.timeout)

    results = audit(rows, checker, workers=args.workers)
    summary = summarize(results)
    write_report(results, summary, args.report)

    state = update_state(load_state(args.state), results)
    with open(args.state, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
        f.write("\n")

    eligible = eligible_for_pruning(state, args.fail_threshold)
    print(
        f"Freshness: {summary['alive']} alive, {summary['dead']} dead, "
        f"{summary['unreachable']} unreachable, {summary['unknown']} unknown "
        f"(of {summary['total']}). {len(eligible)} tool(s) dead >= {args.fail_threshold} runs."
    )

    if args.emit_deletions and eligible:
        entries = build_deletion_entries(results, eligible, state)
        added = merge_into_blocklist(entries, args.blocklist)
        print(f"Queued {added} tool(s) for removal in {args.blocklist} (run Apply_Deletions.py to apply).")
    elif eligible:
        print("Re-run with --emit-deletions to queue these for removal.")


if __name__ == "__main__":
    main()
