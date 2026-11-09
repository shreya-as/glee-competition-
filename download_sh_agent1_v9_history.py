#!/usr/bin/env python3
"""Download sh_agent1 V9 GLEE history pages and attach replay payloads.

The output matches the existing final_logs/*_history_with_replays.json format:

[
  {"history": {...}, "replay": {...}},
  ...
]

Authentication is read from .env/environment. The public agent API key is tried
first, but the history endpoint may require dashboard auth; in that case set
GLEE_HISTORY_BEARER_TOKEN or GLEE_COOKIE before running this script.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional convenience dependency
    load_dotenv = None


BASE_URL = "https://glee-competition.com/api"
PLAYER_ID = "e86a468d-a890-4f57-a7b1-0afbab49dcf4"
DEFAULT_OUTPUT = "final_logs/final_v9_sh_agent1_history_with_replays.json"
DEFAULT_START = "2026-08-29T12:12:00+05:45"
DEFAULT_END = "2026-08-30T00:34:00+05:45"


def parse_dt(value: str) -> datetime:
    """Parse an ISO datetime and normalize it to UTC."""
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone(timedelta(hours=5, minutes=45)))
    return parsed.astimezone(timezone.utc)


def request_json(url: str, timeout: float, retries: int, sleep: float) -> Any:
    headers = {"Accept": "application/json", "User-Agent": "glee-history-downloader/1.0"}
    api_key = os.environ.get("GLEE_API_KEY")
    bearer = os.environ.get("GLEE_HISTORY_BEARER_TOKEN") or os.environ.get("GLEE_BEARER_TOKEN")
    cookie = os.environ.get("GLEE_COOKIE")
    if api_key:
        headers["X-API-Key"] = api_key
        headers.setdefault("Authorization", f"Bearer {api_key}")
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    if cookie:
        headers["Cookie"] = cookie

    request = urllib.request.Request(url, headers=headers)
    last_error: Exception | None = None
    for attempt in range(1, retries + 2):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                charset = response.headers.get_content_charset() or "utf-8"
                return json.loads(response.read().decode(charset))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            last_error = exc
            if attempt > retries:
                break
            time.sleep(sleep * attempt)
    raise RuntimeError(f"failed to fetch {url}: {last_error}") from last_error


def history_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        raise TypeError(f"unexpected history payload type: {type(payload).__name__}")
    for key in ("games", "history", "results", "items", "data"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    raise KeyError(f"could not find a history list in keys: {sorted(payload)}")


def game_id_from(history: dict[str, Any]) -> str:
    for key in ("game_id", "id"):
        value = history.get(key)
        if value:
            return str(value)
    raise KeyError(f"history row has no game_id/id: {history}")


def row_time(history: dict[str, Any]) -> datetime | None:
    for key in ("started_at", "completed_at", "created_at"):
        value = history.get(key)
        if value:
            return parse_dt(str(value))
    return None


def build_history_url(player_id: str, limit: int, offset: int) -> str:
    query = urllib.parse.urlencode(
        {
            "player_type": "agent",
            "limit": limit,
            "offset": offset,
            "player_id": player_id,
        }
    )
    return f"{BASE_URL}/games/history?{query}"


def build_replay_url(game_id: str) -> str:
    quoted = urllib.parse.quote(game_id, safe="")
    return f"{BASE_URL}/games/{quoted}/replay"


def main() -> int:
    if load_dotenv is not None:
        load_dotenv(Path(__file__).with_name(".env"))

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--player-id", default=PLAYER_ID)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--start", default=DEFAULT_START, help="inclusive ISO datetime")
    parser.add_argument("--end", default=DEFAULT_END, help="inclusive ISO datetime")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-offset", type=int, default=380)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=0.25)
    parser.add_argument(
        "--no-date-filter",
        action="store_true",
        help="save every history row from offsets 0..max-offset",
    )
    args = parser.parse_args()

    start = parse_dt(args.start)
    end = parse_dt(args.end)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    combined: list[dict[str, Any]] = []
    seen_game_ids: set[str] = set()
    fetched_history_rows = 0
    skipped_by_date = 0

    for offset in range(0, args.max_offset + 1, args.limit):
        history_url = build_history_url(args.player_id, args.limit, offset)
        payload = request_json(history_url, args.timeout, args.retries, args.sleep)
        rows = history_items(payload)
        fetched_history_rows += len(rows)
        print(f"offset={offset}: {len(rows)} history rows", file=sys.stderr)
        if not rows:
            break

        for history in rows:
            seen_at = row_time(history)
            if not args.no_date_filter:
                if seen_at is None or seen_at < start or seen_at > end:
                    skipped_by_date += 1
                    continue

            game_id = game_id_from(history)
            if game_id in seen_game_ids:
                continue
            seen_game_ids.add(game_id)

            replay = request_json(
                build_replay_url(game_id),
                args.timeout,
                args.retries,
                args.sleep,
            )
            combined.append({"history": history, "replay": replay})
            time.sleep(args.sleep)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(combined, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(
        (
            f"saved {len(combined)} games to {output_path} "
            f"({fetched_history_rows} history rows fetched, {skipped_by_date} skipped by date)"
        ),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
