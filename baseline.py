"""Experiment A baseline tracer: "how good is my agent right now" per game
family, straight from logs/games.jsonl -- no code change, just measurement.

Success definition differs by family (all three families always log
outcome, but persuasion's outcome is "completed" for every finished game --
win/loss lives in payoff, not outcome):
  - bargaining / negotiation: success = outcome == "agreement"
  - persuasion: success = payoff > 0 (a completed pitch that earned nothing
    is a failure for baselining purposes, same as the prompt's own example)

Usage:
    python baseline.py                 # whole history, all 3 families
    python baseline.py --last 50       # last 50 games per family (your
                                        # "Experiment A, 50 persuasion games"
                                        # unit -- run this after a batch)
    python baseline.py --family persuasion --last 50
    python baseline.py --log-path logs/sh_agent1_games.jsonl --last 100
"""

import argparse
import json
import os
from collections import defaultdict

_LOG_PATH = os.path.join(os.path.dirname(__file__), "logs", "games.jsonl")
FAMILIES = ("bargaining", "negotiation", "persuasion")


def _is_success(row: dict) -> bool:
    if row["game_family"] == "persuasion":
        return (row.get("payoff") or 0) > 0
    return row.get("outcome") == "agreement"


def load_rows(path: str = _LOG_PATH) -> dict:
    rows = defaultdict(list)
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                fam = row.get("game_family")
                if fam in FAMILIES:
                    rows[fam].append(row)
    except FileNotFoundError:
        pass
    return rows


def stats(rows: list, last: int | None) -> dict:
    if last:
        rows = rows[-last:]
    total = len(rows)
    successful = sum(1 for r in rows if _is_success(r))
    failed = total - successful
    payoffs = [r.get("payoff") or 0 for r in rows]
    zero = sum(1 for p in payoffs if p == 0)
    total_payoff = sum(payoffs)
    avg_payoff = total_payoff / total if total else 0.0
    return {
        "total": total,
        "successful": successful,
        "failed": failed,
        "zero_payoff": zero,
        "avg_payoff": avg_payoff,
        "total_payoff": total_payoff,
    }


def _fmt(n: float) -> str:
    return f"${n:,.0f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--last", type=int, default=None, help="only the last N games per family")
    parser.add_argument("--family", choices=FAMILIES, default=None)
    parser.add_argument("--log-path", default=_LOG_PATH, help="JSONL log file to analyze")
    args = parser.parse_args()

    rows_by_family = load_rows(args.log_path)
    families = [args.family] if args.family else FAMILIES

    for fam in families:
        rows = rows_by_family.get(fam, [])
        s = stats(rows, args.last)
        print(f"=== {fam} ===")
        print(f"Total games:     {s['total']}")
        print(f"Successful:      {s['successful']}")
        print(f"Failed:          {s['failed']}")
        print(f"$0 games:        {s['zero_payoff']}")
        print(f"Average payoff:  {_fmt(s['avg_payoff'])}")
        print(f"Total payoff:    {_fmt(s['total_payoff'])}")
        print()


if __name__ == "__main__":
    main()
