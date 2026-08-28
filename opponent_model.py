"""Opponent-type classification (GLEE strategy notes, "Strategy 1").

Buckets each opponent into aggressive / cooperative / random from their
observed concession pattern, instead of playing bargaining and negotiation
the same way against everyone. Two data sources feed a classification:

- Live, in-game concession gap: available on every decision turn from
  state["last_offer"], regardless of whether the opponent's identity is
  disclosed (works in both game halves).
- Cross-game history, keyed by opponent name, in the disclosed half only:
  accept rate and rounds-to-agreement, folded in via `record_outcome` from
  the move() result of games WE end (by accepting, or by a forced close on
  the final attempt). A game the opponent ends by accepting OUR offer is
  invisible here -- the SDK's pending-games loop just stops returning it,
  with no event to hook (the same gap paper_draft.md section 6 notes for
  persuasion honesty tracking) -- so accept_rate/avg_rounds_to_agreement
  undercount opponents who mostly accept quickly. The live counter-gap read
  has no such blind spot, which is why it takes priority when cross-game
  history is thin.

Persisted to a JSON file so profiles survive process restarts and are shared
across fleet.py's multiple agent processes. Concurrent writers use a plain
load-modify-atomic-replace cycle with no cross-process lock: a rare lost
update (last writer wins) is an acceptable trade for a heuristic classifier,
not something worth a file-locking dependency over.
"""

import json
import logging
import os
import threading

logger = logging.getLogger("opponent_model")

_STORE_PATH = os.path.join(os.path.dirname(__file__), "logs", "opponent_profiles.json")
_MAX_SAMPLES = 50  # bound memory/file size; recent behavior matters most anyway

_lock = threading.Lock()


def _load() -> dict:
    try:
        with open(_STORE_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save() -> None:
    os.makedirs(os.path.dirname(_STORE_PATH), exist_ok=True)
    tmp = f"{_STORE_PATH}.{os.getpid()}.{threading.get_ident()}.tmp"
    with open(tmp, "w") as f:
        json.dump(_profiles, f, indent=2)
    os.replace(tmp, _STORE_PATH)


_profiles = _load()

# Per-game, in-memory only: the opponent's successive offer values this game.
# Keyed by game_id so the live counter-gap read works even against a
# same-game-only (hidden-identity) opponent we'll never see again.
_game_offers: dict[str, list[float]] = {}


def opponent_key(game: dict) -> str | None:
    """Cross-game identity, only available in the disclosed half of games."""
    return (game.get("opponent") or {}).get("name")


def _profile(key: str) -> dict:
    return _profiles.setdefault(
        key, {"counter_gaps": [], "accepts": 0, "closes": 0, "rounds_to_agreement": []}
    )


def extract_opponent_offer(game: dict) -> float | None:
    """Pull the number that represents "how much the opponent is asking for
    themselves" from the current decision turn, for bargaining/negotiation
    only. None on an offer turn (nothing to react to yet) or for persuasion
    (no comparable single-number offer)."""
    if (game.get("valid_actions") or {}).get("type") != "decision":
        return None
    family = game["game_family"]
    state = game["game_state"]
    offer = state.get("last_offer") or {}
    if family == "bargaining":
        me = state.get("current_player")
        other = "player_2" if me == "player_1" else "player_1"
        return offer.get(f"{other}_gain")
    if family == "negotiation":
        return offer.get("price")
    return None


def record_offer(game: dict, opponent_value: float) -> None:
    """Call on every decision turn with the opponent's just-seen offer value.
    Tracks this game's concession sequence and, when the opponent is
    disclosed, folds the latest gap into their persisted cross-game profile
    (flushed to disk lazily -- see record_outcome -- not on every call, since
    a fleet plays thousands of rounds and a per-round fsync would dominate)."""
    game_id = game["game_id"]
    offers = _game_offers.setdefault(game_id, [])
    if not offers or offers[-1] != opponent_value:
        offers.append(opponent_value)

    key = opponent_key(game)
    if key and len(offers) >= 2:
        gap = abs(offers[-1] - offers[-2])
        with _lock:
            gaps = _profile(key)["counter_gaps"]
            gaps.append(gap)
            del gaps[:-_MAX_SAMPLES]


def record_outcome(game: dict, result: dict) -> None:
    """Call with the move() result whenever OUR move ends the game (accept,
    or a forced close on the final attempt). See module docstring: a game the
    opponent ends by accepting OUR offer never reaches this function."""
    key = opponent_key(game)
    _game_offers.pop(game.get("game_id"), None)
    if not key or not result:
        return
    outcome = result.get("outcome")
    with _lock:
        profile = _profile(key)
        if outcome == "agreement":
            profile["accepts"] += 1
            round_num = result.get("agreed_round") or game.get("game_state", {}).get("round", 1)
            profile["rounds_to_agreement"].append(round_num)
            del profile["rounds_to_agreement"][:-_MAX_SAMPLES]
        else:
            profile["closes"] += 1
        _save()


def _live_gap(game: dict) -> float | None:
    offers = _game_offers.get(game["game_id"], [])
    if len(offers) < 2:
        return None
    return abs(offers[-1] - offers[-2])


def _scale(game: dict) -> float:
    """Normalizer so a counter-gap means the same thing across wildly
    different stake sizes (money_to_divide can be 10 or 1,000,000)."""
    state = game["game_state"]
    me = state.get("current_player")
    return (
        state.get("money_to_divide")
        or state.get(f"{me}_value")
        or 1
    ) or 1


def classify(game: dict) -> tuple[str, dict]:
    """Returns (label, evidence). label is one of "aggressive",
    "cooperative", "random", or "unknown" (not enough signal yet).

    Thresholds below are starting-point heuristics, not calibrated against
    labeled data -- tune once real accept/reject outcomes accumulate in
    logs/games.jsonl (see game_logger.py)."""
    key = opponent_key(game)
    profile = _profiles.get(key) if key else None

    gap = _live_gap(game)
    norm_gap = (gap / _scale(game)) if gap is not None else None
    evidence = {"norm_counter_gap": norm_gap}

    if profile and (profile["counter_gaps"] or profile["accepts"] or profile["closes"]):
        total = profile["accepts"] + profile["closes"]
        evidence["accept_rate"] = profile["accepts"] / total if total else None
        evidence["avg_rounds_to_agreement"] = (
            sum(profile["rounds_to_agreement"]) / len(profile["rounds_to_agreement"])
            if profile["rounds_to_agreement"] else None
        )
        if profile["counter_gaps"]:
            hist_gaps = profile["counter_gaps"]
            avg_gap = sum(hist_gaps) / len(hist_gaps)
            evidence["avg_counter_gap"] = avg_gap
            if len(hist_gaps) >= 2:
                variance = sum((g - avg_gap) ** 2 for g in hist_gaps) / len(hist_gaps)
                evidence["counter_gap_stdev"] = variance ** 0.5

    if norm_gap is None and evidence.get("accept_rate") is None:
        return "unknown", evidence

    # Inconsistent behavior: swings in concession size relative to its own
    # average. A consistent conceder or a consistent stonewaller both have
    # low relative spread; an opponent whose offers jump around doesn't.
    stdev, avg_gap = evidence.get("counter_gap_stdev"), evidence.get("avg_counter_gap")
    if stdev is not None and avg_gap and stdev / (avg_gap + 1e-9) > 1.2:
        return "random", evidence

    accept_rate = evidence.get("accept_rate")
    if accept_rate is not None and accept_rate < 0.3:
        return "aggressive", evidence
    if accept_rate is not None and accept_rate > 0.7:
        return "cooperative", evidence

    if norm_gap is not None:
        if norm_gap < 0.03:
            return "aggressive", evidence
        if norm_gap > 0.10:
            return "cooperative", evidence

    return "unknown", evidence
