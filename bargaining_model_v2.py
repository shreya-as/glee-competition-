"""Bargaining v2: reservation-point estimation + candidate-search payoff
optimizer, replacing simple_agent.py's fixed favor=0.55 / accept_threshold=0.45.

Current bargaining_strategy() in simple_agent.py is the fleet's strongest
family (95% success, ~$156k/game average over the last 200 games in
logs/games.jsonl) precisely because it's simple and doesn't chase a fixed
split -- so this file is a SEPARATE, NOT-YET-WIRED-IN module (same rollout
pattern as persuasion_model_v2.py: build it standalone, compare via
baseline.py, swap simple_agent.py's import only once it's actually ahead).
Do not point fleet.py at this until that comparison has run -- see this
file's own docstring note near the bottom on how to wire it in.

What changes vs the v1 fixed-favor approach:

1. Estimate the opponent's reservation point live, from their own offer
   sequence THIS game (state["last_offer"] on every decision turn -- the same
   read opponent_model.extract_opponent_offer already exposes). If their
   asks are converging (65% -> 62% -> 58% -> 57%), Aitken's delta-squared
   extrapolation projects where that geometric-looking sequence is heading,
   instead of assuming they'll keep conceding all the way to 50/50 or not at
   all. This has no blind spot -- unlike cross-game outcome logging (see
   point 3), every offer the opponent makes while a game is live is directly
   observed, regardless of how the game eventually ends.

2. Score candidate offers (a small grid of our-share fractions, 0.50-0.95)
   by expected payoff = share * money * P(accept), not by hitting a fixed
   number. P(accept) is a logistic centered on the estimated reservation
   point, lightly blended with learned_model.py's trained model (see caveat
   below) and gated by round-phase (explore / exploit / close).

3. On the deliberately NOT built: a persisted "P(opponent accepts share X)"
   table like persuasion_model_v2.py's payoff table. game_logger.py's own
   docstring (and opponent_model.py's) already flags the real constraint:
   we only ever see the outcome of games WE end (we accept, or we're
   force-closed) -- a game the opponent ends by accepting OUR offer never
   calls move() again on our side, so it's invisible to any post-hoc log.
   Both logs/games.jsonl and the model trained off it in learned_model.py
   inherit that same gap -- learned_model.predict_prob("bargaining", ...) is
   therefore mostly calibrated on "shares we personally chose to accept",
   not "shares this opponent would accept from us". It's used below as a
   small, capped nudge for exactly that reason, not as the primary signal.
   Building a NEW empirical table on the same biased data source would just
   reproduce persuasion v2's original mistake (scoring off data that can't
   actually answer the question) instead of avoiding it.

4. Accept/reject uses an actual continuation-value comparison (accept iff
   the offer on the table beats what our own best candidate-search offer is
   worth next round, discounted by our own delta) instead of a fixed,
   round-eroding accept_threshold.
"""

import json
import logging
import math
import os
import threading

import learned_model
import opponent_model

logger = logging.getLogger("bargaining_model_v2")

_STORE_PATH = os.path.join(os.path.dirname(__file__), "logs", "bargaining_reservation_v2.json")
_MAX_SAMPLES = 30

# Our-share candidate grid (fraction of money_to_divide). 0.50 floor -- no
# reason to ever propose taking less than half as the proposer.
_CANDIDATES = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

# Logistic steepness for P(accept) around the estimated reservation point.
# Higher = sharper cutoff right at the reservation estimate; lower = more
# forgiving of estimation error. Not tuned against labeled data (no clean
# signal to tune it against, see point 3 above) -- a starting point.
_STEEPNESS = 10.0

# Weight given to learned_model.py's trained-but-caveated forward P(accept)
# nudge, matching simple_agent.py's existing 0.2 blend for the same model's
# inverse direction (implied_accept_threshold).
_LEARNED_WEIGHT = 0.15

_EXPLORE_ROUNDS = 3       # anchor + observe only, no reservation estimate yet
_CLOSE_ROUND_START = 15   # start requiring a rising P(accept) floor
_CLOSE_ROUND_FULL = 30    # floor reaches its max by this round

_lock = threading.Lock()

# Per-game, in-memory: opponent's own successive self-ask fractions
# (their_gain / money), for the live reservation-point extrapolation.
_game_opp_fracs: dict[str, list[float]] = {}


def _load() -> dict:
    try:
        with open(_STORE_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save() -> None:
    os.makedirs(os.path.dirname(_STORE_PATH), exist_ok=True)
    tmp = f"{_STORE_PATH}.tmp"
    with open(tmp, "w") as f:
        json.dump(_profiles, f, indent=2)
    os.replace(tmp, _STORE_PATH)


_profiles = _load()


def _record_opponent_frac(game: dict, state: dict) -> None:
    """Fold the opponent's just-seen self-ask into this game's live sequence.
    Reuses opponent_model.extract_opponent_offer rather than re-reading
    state["last_offer"] directly, so both modules agree on what "the
    opponent's offer" means."""
    opp_amount = opponent_model.extract_opponent_offer(game)
    if opp_amount is None:
        return
    money = state.get("money_to_divide")
    if not money:
        return
    frac = opp_amount / money
    seq = _game_opp_fracs.setdefault(game["game_id"], [])
    if not seq or seq[-1] != frac:
        seq.append(frac)


def _estimate_reservation(seq: list[float]) -> float | None:
    """Aitken's delta-squared extrapolation on the last 3 points: treats a
    converging sequence (o[n-2], o[n-1], o[n]) as geometric decay toward an
    asymptote R and solves for R directly, instead of assuming the opponent
    keeps conceding at the same absolute rate forever. None if there isn't
    enough history yet, or the last two gaps aren't actually converging
    (opponent stonewalling or oscillating -- extrapolating a non-convergent
    sequence would just be noise)."""
    if len(seq) < 3:
        return None
    o0, o1, o2 = seq[-3], seq[-2], seq[-1]
    d1, d2 = o1 - o0, o2 - o1
    if d1 == 0 or d2 == 0:
        return o2  # flat -- best guess is "they've already stopped moving"
    ratio = d2 / d1
    if not (0 < ratio < 1):
        return None  # not a converging geometric pattern -- don't guess
    reservation = o2 - d2 * ratio / (1 - ratio)
    return max(0.0, min(1.0, reservation))


def _historical_reservation(key: str | None) -> float | None:
    if not key or key not in _profiles:
        return None
    ests = _profiles[key].get("reservation_estimates") or []
    return sum(ests) / len(ests) if ests else None


def _record_reservation_estimate(key: str | None, reservation: float) -> None:
    if not key:
        return  # undisclosed opponent -- nothing to persist across games
    with _lock:
        profile = _profiles.setdefault(key, {"reservation_estimates": []})
        ests = profile["reservation_estimates"]
        ests.append(reservation)
        del ests[:-_MAX_SAMPLES]
        _save()


def _blended_reservation(game: dict, seq: list[float]) -> float | None:
    live = _estimate_reservation(seq)
    key = opponent_model.opponent_key(game)
    if live is not None:
        _record_reservation_estimate(key, live)
        historical = _historical_reservation(key)
        # Live, in-game behavior dominates -- it's this exact opponent, this
        # exact game, no cross-game averaging error. Historical average is
        # just a tiebreaker toward "how this opponent has behaved before".
        return 0.7 * live + 0.3 * historical if historical is not None else live
    return _historical_reservation(key)


def _accept_probability(opp_offered_frac: float, reservation: float | None, our_frac: float, round_num: int) -> float:
    if reservation is not None:
        base = 1 / (1 + math.exp(-_STEEPNESS * (opp_offered_frac - reservation)))
    else:
        # No reservation read yet (round 1-2, or a non-converging opponent):
        # gentle generic prior, softer than a hard 50/50 cutoff since we
        # genuinely don't know yet.
        base = 1 / (1 + math.exp(-6 * (opp_offered_frac - 0.45)))

    learned = learned_model.predict_prob("bargaining", our_frac, round_num)
    if learned is not None:
        base = (1 - _LEARNED_WEIGHT) * base + _LEARNED_WEIGHT * learned
    return min(1.0, max(0.0, base))


def _close_floor(round_num: int) -> float:
    if round_num <= _CLOSE_ROUND_START:
        return 0.0
    span = _CLOSE_ROUND_FULL - _CLOSE_ROUND_START
    ramp = min(1.0, (round_num - _CLOSE_ROUND_START) / span)
    return 0.80 + 0.17 * ramp  # -> 0.97 by _CLOSE_ROUND_FULL


def _delay_cost(money: float, my_delta: float | None) -> float:
    """Rough opportunity cost of dragging out one more round: half of what
    our own discount factor already says we lose by waiting. A full
    recursive solve of the alternating-offers game is out of scope here --
    this is a bounded heuristic nudge, same spirit as the rest of this
    file's scoring."""
    if my_delta is None:
        return 0.0
    return money * 0.5 * (1 - my_delta)


def _best_candidate(game: dict, state: dict, money: float, reservation: float | None, round_num: int, my_delta: float | None) -> tuple[float, float]:
    """(our_frac, score) maximizing expected payoff over the candidate grid,
    honoring the close-phase P(accept) floor once round_num crosses it."""
    floor = _close_floor(round_num)
    scored = []
    for c in _CANDIDATES:
        opp_offered = 1 - c
        p = _accept_probability(opp_offered, reservation, c, round_num)
        score = c * money * p - _delay_cost(money, my_delta)
        scored.append((c, score, p))

    eligible = [(c, s, p) for c, s, p in scored if p >= floor]
    if eligible:
        c, s, _ = max(eligible, key=lambda x: x[1])
        return c, s
    # Close phase and nothing clears the floor -- force toward whatever
    # candidate has the single highest P(accept) rather than stalling.
    c, s, _ = max(scored, key=lambda x: x[2])
    return c, s


def bargaining_strategy_v2(game: dict, actions: dict, state: dict, favor: float = 0.55, accept_threshold: float = 0.45, concede: bool = True) -> dict:
    """Drop-in replacement for simple_agent.py's bargaining_strategy -- same
    call shape (favor/accept_threshold/concede kept as the round-1/no-data
    fallback anchor, so this degrades to something close to the v1 baseline
    before any opponent offer has been observed). See module docstring for
    what actually changed."""
    money = state["money_to_divide"]
    round_num = state.get("round", 1)
    game_id = game["game_id"]
    me = state.get("current_player")
    my_delta = state.get("delta_1" if me == "player_1" else "delta_2")
    label, _ = opponent_model.classify(game)

    if actions["type"] == "decision":
        opponent_offer = opponent_model.extract_opponent_offer(game)
        if opponent_offer is not None:
            opponent_model.record_offer(game, opponent_offer)
        _record_opponent_frac(game, state)

    seq = _game_opp_fracs.get(game_id, [])
    reservation = _blended_reservation(game, seq)

    if actions["type"] == "offer":
        if round_num <= _EXPLORE_ROUNDS and reservation is None:
            # Same anchor-and-observe opening as v1 -- nothing to estimate
            # from yet, label-based nudge is the only signal available.
            our_frac = min(0.95, max(0.05, favor + {"aggressive": -0.03, "cooperative": 0.03}.get(label, 0.0)))
        else:
            our_frac, _ = _best_candidate(game, state, money, reservation, round_num, my_delta)

        my_share = money * our_frac
        other_share = money - my_share
        my_key = "alice_gain" if me == "player_1" else "bob_gain"
        other_key = "bob_gain" if me == "player_1" else "alice_gain"
        message = (
            "Fair split?"
            if round_num <= 1
            else "Here's where the numbers land for both of us — let's close this out."
        )
        return {my_key: my_share, other_key: other_share, "message": message}

    if actions["type"] == "decision":
        offer = state["last_offer"]
        my_gain = offer[f"{state['current_player']}_gain"]

        _, continue_score = _best_candidate(game, state, money, reservation, round_num, my_delta)
        # Discount the continuation value by our own patience -- another
        # round only costs us what my_delta says it costs; if we don't
        # discount at all (my_delta unknown), fall back to accept_threshold
        # as a floor so this never accepts worse than the v1 baseline would.
        continue_value = continue_score * (my_delta if my_delta is not None else 1.0)
        floor_value = money * accept_threshold
        if my_gain >= max(continue_value * 0.98, floor_value):
            return {"decision": "accept"}
        return {"decision": "reject"}

    return {}


# ---------------------------------------------------------------------------
# To compare against the live baseline: run this alongside, don't replace it.
# e.g. in simple_agent.py, add a `strategy_bargaining_v2` variant that calls
# bargaining_model_v2.bargaining_strategy_v2 in place of bargaining_strategy,
# point one fleet.py agent slot at it, and after ~100+ games compare:
#   python3 baseline.py --family bargaining --last 100
# against the other slots still running v1, before touching the other four.
# ---------------------------------------------------------------------------
