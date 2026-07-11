"""Vendor intelligence report built from SHORTLIST telemetry lines.

The backend logs one ``SHORTLIST {json}`` line per served recommendation (see
``log_shortlist`` in api.py). Tool order is rank order: index 0 won the
recommendation, everything after it was outranked. Export the service logs
from Render (or any file containing them — extra log prefixes are fine) and
run:

    python vendor_report.py render_log.txt                 # all-tools overview
    python vendor_report.py render_log.txt --tool Descript # one vendor's view
    python vendor_report.py render_log.txt --json          # machine-readable

The per-vendor view is the sellable artifact: how often the tool was
shortlisted, how often it won, which rivals outranked it (and how often), and
which user goals it surfaces for.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from typing import Any, Iterable

MARKER = "SHORTLIST "


def parse_events(lines: Iterable[str]) -> list[dict[str, Any]]:
    """Extract SHORTLIST payloads from raw log lines.

    Tolerant by design: lines without the marker, or with malformed JSON after
    it, are skipped — a Render log export is full of unrelated lines.
    """
    events: list[dict[str, Any]] = []
    for line in lines:
        idx = line.find(MARKER)
        if idx == -1:
            continue
        try:
            payload = json.loads(line[idx + len(MARKER):].strip())
        except json.JSONDecodeError:
            continue
        tools = [
            str(name).strip()
            for name in (payload.get("tools") or [])
            if isinstance(name, str) and str(name).strip()
        ]
        if not tools:
            continue
        events.append(
            {
                "tools": tools,
                "goal": str(payload.get("goal") or "").strip(),
                "action": str(payload.get("action") or "").strip(),
                "q": str(payload.get("q") or "").strip(),
            }
        )
    return events


def aggregate(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per-tool stats across every shortlist the tool appeared in."""
    stats: dict[str, dict[str, Any]] = {}
    for event in events:
        tools = event["tools"]
        for rank, name in enumerate(tools, start=1):
            entry = stats.setdefault(
                name,
                {
                    "appearances": 0,
                    "wins": 0,
                    "rank_sum": 0,
                    "lost_to": Counter(),
                    "goals": Counter(),
                    "queries": Counter(),
                },
            )
            entry["appearances"] += 1
            entry["rank_sum"] += rank
            if rank == 1:
                entry["wins"] += 1
            for rival in tools[: rank - 1]:
                entry["lost_to"][rival] += 1
            if event["goal"]:
                entry["goals"][event["goal"]] += 1
            if event["q"]:
                entry["queries"][event["q"]] += 1
    return stats


def _win_rate(entry: dict[str, Any]) -> float:
    return entry["wins"] / entry["appearances"] if entry["appearances"] else 0.0


def _avg_rank(entry: dict[str, Any]) -> float:
    return entry["rank_sum"] / entry["appearances"] if entry["appearances"] else 0.0


def render_overview(stats: dict[str, dict[str, Any]]) -> str:
    if not stats:
        return "No SHORTLIST events found in the input."
    lines = [
        f"{'Tool':<32} {'Shortlisted':>11} {'Won':>5} {'Win %':>7} {'Avg rank':>9}",
        "-" * 68,
    ]
    ranked = sorted(
        stats.items(), key=lambda item: (-item[1]["appearances"], item[0].lower())
    )
    for name, entry in ranked:
        lines.append(
            f"{name:<32} {entry['appearances']:>11} {entry['wins']:>5} "
            f"{_win_rate(entry) * 100:>6.0f}% {_avg_rank(entry):>9.1f}"
        )
    return "\n".join(lines)


def render_tool(stats: dict[str, dict[str, Any]], tool: str) -> str:
    match = next(
        (name for name in stats if name.lower() == tool.lower().strip()), None
    )
    if match is None:
        return f"No SHORTLIST events mention {tool!r}."
    entry = stats[match]
    lines = [
        f"Vendor report: {match}",
        "-" * 40,
        f"Shortlisted: {entry['appearances']} times",
        f"Won (ranked #1): {entry['wins']} times ({_win_rate(entry) * 100:.0f}%)",
        f"Average rank: {_avg_rank(entry):.1f}",
    ]
    if entry["lost_to"]:
        lines.append("Outranked by:")
        for rival, count in entry["lost_to"].most_common(5):
            lines.append(f"  {rival}: {count} times")
    if entry["goals"]:
        lines.append("Surfaces for goals:")
        for goal, count in entry["goals"].most_common(5):
            lines.append(f"  {goal}: {count} shortlists")
    if entry["queries"]:
        lines.append("Sample queries:")
        for query, _count in entry["queries"].most_common(3):
            lines.append(f"  {query!r}")
    return "\n".join(lines)


def to_json(stats: dict[str, dict[str, Any]]) -> str:
    serializable = {
        name: {
            "appearances": entry["appearances"],
            "wins": entry["wins"],
            "win_rate": round(_win_rate(entry), 4),
            "avg_rank": round(_avg_rank(entry), 2),
            "lost_to": dict(entry["lost_to"].most_common()),
            "goals": dict(entry["goals"].most_common()),
            "queries": dict(entry["queries"].most_common(10)),
        }
        for name, entry in stats.items()
    }
    return json.dumps(serializable, indent=2, ensure_ascii=False, sort_keys=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Per-vendor recommendation intelligence from SHORTLIST logs."
    )
    parser.add_argument("logfile", help="Log export containing SHORTLIST lines")
    parser.add_argument("--tool", help="Show the detailed view for one tool")
    parser.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON"
    )
    args = parser.parse_args(argv)

    with open(args.logfile, "r", encoding="utf-8", errors="replace") as handle:
        stats = aggregate(parse_events(handle))

    if args.json:
        print(to_json(stats))
    elif args.tool:
        print(render_tool(stats, args.tool))
    else:
        print(render_overview(stats))
    return 0


if __name__ == "__main__":
    sys.exit(main())
