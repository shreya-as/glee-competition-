"""Persuasion v2: payoff-aware, risk-adjusted version of persuasion_model.py.

persuasion_model.py picks an argument type by pure P(bought) -- but the logs
(logs/learning.log) show that doesn't track payoff at all: driftwood/economic
sits at 100% over 40 tries, then the very next driftwood/economic round pays
$0; Lior/economic is 100% over 8 tries and also throws $0 rounds. Acceptance
rate and payoff are just different signals here (price varies per round), so
this file scores by expected payoff instead of raw hit rate, adds a real
probing phase instead of committing early, tracks an escalating message chain
per type instead of repeating the same line, and adds a circuit breaker for a
type on a losing streak against one opponent.

One thing in the request this file deliberately does NOT implement as
described: opponent "signals" like *asks about benefits* or *mentions
reciprocity*. Confirmed in persuasion_model.py's own docstring -- buyers
never send text back, only a yes/no `bought` per round -- so there is no
buyer text to read signals out of. The only real signal available is which
argument type a given opponent has actually bought under, which is what
`classify_opponent` below is built from instead.

Also: the request's risk detector ("if estimated_loss > LOSS_THRESHOLD")
targets a fixed dollar figure, but seller payoff here is floor-0 (price if
bought, 0 if not -- see persuasion_model.py's _BLUFF_RATE comment, confirmed
from a real logged round) and observed payoffs in this environment span from
single dollars to $56,000,000 -- no one fixed dollar threshold means the same
thing at both ends. The catastrophic loss case in the request (-$8,000,000)
can only be a BUYER-side outcome (bought at a price above the item's real
value), so the risk control that matters lives in buyer_decision_action_v2
below, scaled relative to price/value rather than a fixed dollar figure.

Persisted separately from persuasion_model.py (logs/persuasion_profiles_v2.json)
-- different schema (adds payoff tracking, recent-outcome window), and this
file is not wired into simple_agent.py / fleet.py yet. To switch a strategy
over: swap its persuasion_model imports/calls for this module's
seller_message_action_v2 / seller_recommendation_action_v2 /
buyer_decision_action_v2, same call shape as persuasion_model.py's originals.
"""

import json
import logging
import math
import os
import random
import threading

logger = logging.getLogger("persuasion_model_v2")

_STORE_PATH = os.path.join(os.path.dirname(__file__), "logs", "persuasion_profiles_v2.json")
_MAX_SAMPLES = 200
_RECENT_WINDOW = 10  # how far back "resistance streak" / recency looks

ARGUMENT_TYPES = ["economic", "fairness", "safety", "social", "convenience", "long_term"]

# Continuation exploration floor once every type has been probed at least
# once -- same purpose as persuasion_model.py's _EPSILON, kept smaller
# because probing (below) already guarantees full coverage up front, so this
# only needs to keep re-checking types whose early sample was a fluke.
_EPSILON_FLOOR = 0.05

# Stdev-of-payoff penalty weight in the scoring function -- see score().
_RISK_WEIGHT = 0.15

# Consecutive endorsed-and-declined rounds on the CURRENT top-scoring type
# before forcing a pivot to something else, regardless of its score. This is
# the "opponent_resistance >= 3" rule from the request, applied to whichever
# type is actually in the lead rather than a global counter.
_RESISTANCE_LIMIT = 3

_BLUFF_RATE = 0.0  # see persuasion_model.py -- kept at 0 here too

# Per-type escalating phrasing: repeated pitches of the same type against a
# stalled opponent shouldn't resend identical text. Stage advances with how
# many times we've already tried this type on this opponent (mod chain
# length), independent of the resistance-streak circuit breaker above.
_ESCALATION = {
    "economic": [
        "This unit is worth about ${value:,.0f} to you at a price of ${price:,.0f} -- a clear profit if you buy.",
        "Run the numbers again: ${value:,.0f} in value against ${price:,.0f} is real, concrete margin, not a maybe.",
        "This offer won't sit at ${price:,.0f} indefinitely -- the math favors acting on it now, not next round.",
    ],
    "fairness": [
        "At ${price:,.0f} against roughly ${value:,.0f} in value, this is a fair, equitable split of the gains for both of us.",
        "I'm not asking you to carry more risk than I am here -- this price reflects what's fair on both sides.",
        "You've seen how I've priced things so far -- this one keeps to the same fair split, nothing shifted in my favor.",
    ],
    "safety": [
        "I've checked this one myself -- it's solid, verified quality, so there's minimal risk in buying.",
        "Nothing about this unit gives me pause -- I'd say the same if I were on your side of this trade.",
        "Low-risk call, same as the others I've vouched for -- no reason to hesitate on this one.",
    ],
    "social": [
        "We've got rounds ahead of us together -- backing this deal keeps things working well for both of us.",
        "This only works long-run if we both keep making calls that hold up -- this one's a good one to back.",
        "Trust between us is worth more than any single round -- this deal is one worth backing on that basis.",
    ],
    "convenience": [
        "Easy call this round: the numbers clearly favor buying, no need to overthink it.",
        "Simplest read all game: take this one and move on, it's not a close decision.",
        "No further analysis needed here -- straightforward buy, least effort, clear outcome.",
    ],
    "long_term": [
        "Judge this by how I've called it so far -- trust that track record and take this one.",
        "Every call I've made has been building toward being someone you can keep trading with -- this is another one.",
        "Long after this round, what'll matter is that my calls held up -- they have, and this one does too.",
    ],
}
_DECLINE_TEMPLATES = {
    "economic": "Skip this one -- the numbers don't work in your favor this round, better to preserve your capital.",
    "fairness": "Being straight with you: this wouldn't be a fair trade at this price, so I'd pass.",
    "safety": "I can't vouch for this one's quality -- too much risk here, sit it out.",
    "social": "Looking out for you here: this isn't the round to buy, let's wait for a better one.",
    "convenience": "Easy pass this round -- not worth spending more time on.",
    "long_term": "Same honesty that's guided my calls all game: this one's not worth it, hold off.",
}

# Personality label shown alongside classify_opponent()'s pick -- purely
# descriptive (see module docstring on why these aren't behavioral signals,
# just a name for "the type that's scored best against them so far").
_PERSONALITY_LABELS = {
    "economic": "economic-oriented",
    "fairness": "fairness-oriented",
    "safety": "safety-oriented",
    "social": "social-oriented",
    "convenience": "convenience-oriented",
    "long_term": "long-term-oriented",
}

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

# Per-game, in-memory: which argument type + escalation stage we used on
# each round we endorsed, and how far into state["history"] we've synced --
# same shape as persuasion_model.py, needed for the same reason (the API
# gives back message text on replay, not our internal type label).
_game_argument_log: dict[str, dict[int, str]] = {}
_game_synced_through: dict[str, int] = {}


def _opponent_key(game: dict) -> str:
    name = (game.get("opponent") or {}).get("name")
    return name if name else f"game:{game.get('game_id')}"


def _empty_type_stats() -> dict:
    return {"attempts": 0, "successes": 0, "total_payoff": 0.0, "payoff_history": [], "recent": []}


def _profile(key: str) -> dict:
    return _profiles.setdefault(key, {t: _empty_type_stats() for t in ARGUMENT_TYPES})


_DEBUG_CAPTURE_PATH = os.path.join(os.path.dirname(__file__), "logs", "persuasion_debug_capture.json")
_debug_captured = False


def _capture_unresolved_entry(entry: dict) -> None:
    # One-shot dump of a bought=True round whose price we couldn't resolve --
    # 305/305 successes in the live profile currently score payoff=0.0, which
    # means "price"/"product_price" are the wrong keys (or nested elsewhere).
    # Written once so the next fleet restart's first hit tells us the real
    # schema instead of guessing again.
    global _debug_captured
    if _debug_captured:
        return
    _debug_captured = True
    try:
        with open(_DEBUG_CAPTURE_PATH, "w") as f:
            json.dump(entry, f, indent=2, default=str)
        logger.warning(f"persuasion_model_v2: captured unresolved bought=True entry -> {_DEBUG_CAPTURE_PATH}: {entry}")
    except Exception:
        logger.exception("persuasion_model_v2: failed writing debug capture")


def _round_price(entry: dict) -> float:
    # Confirmed live via _capture_unresolved_entry's debug dump
    # (logs/persuasion_debug_capture.json): the real key is "seller_payoff",
    # not "price"/"product_price" -- every one of 305 prior successes in
    # logs/persuasion_profiles_v2.json scored payoff=0.0 because of this.
    # Old keys kept as fallback in case a future entry shape omits it.
    price = entry.get("seller_payoff")
    if price is None:
        price = entry.get("price")
    if price is None:
        price = entry.get("product_price")
    if price is None:
        _capture_unresolved_entry(entry)
    return float(price) if price is not None else 0.0


def _sync_outcomes(game: dict) -> None:
    """Fold newly-visible history rounds into the persisted profile,
    including realized payoff (price if bought, else 0 -- matches this
    game's flat seller payoff, see module docstring) and a recency window
    for the resistance-streak check in select_argument_v2."""
    game_id = game.get("game_id")
    history = (game.get("game_state") or {}).get("history") or []
    arg_log = _game_argument_log.get(game_id, {})
    synced_through = _game_synced_through.get(game_id, 0)
    key = _opponent_key(game)

    new_max = synced_through
    with _lock:
        profile = _profile(key)
        for entry in history:
            round_num = entry.get("round")
            if round_num is None or round_num <= synced_through:
                continue
            new_max = max(new_max, round_num)
            arg_type = arg_log.get(round_num)
            if arg_type is None or entry.get("bought") is None:
                continue  # a decline round, or from before we tracked this game
            stats = profile[arg_type]
            bought = bool(entry["bought"])
            payoff = _round_price(entry) if bought else 0.0
            stats["attempts"] += 1
            if bought:
                stats["successes"] += 1
                stats["total_payoff"] += payoff
            stats["payoff_history"].append(payoff)
            del stats["payoff_history"][:-_MAX_SAMPLES]
            stats["recent"].append(bought)
            del stats["recent"][:-_RECENT_WINDOW]
        if new_max > synced_through:
            _game_synced_through[game_id] = new_max
            _save()


def _laplace_rate(stats: dict) -> float:
    return (stats["successes"] + 1) / (stats["attempts"] + 2)


def _avg_success_payoff(stats: dict) -> float:
    return stats["total_payoff"] / stats["successes"] if stats["successes"] else 0.0


def _risk_penalty(stats: dict) -> float:
    """Stdev of realized payoff for this type against this opponent -- a
    type that occasionally pays huge and mostly pays $0 is riskier than one
    that pays a smaller amount consistently, even at the same mean."""
    hist = stats["payoff_history"]
    if len(hist) < 2:
        return 0.0
    mean = sum(hist) / len(hist)
    variance = sum((x - mean) ** 2 for x in hist) / len(hist)
    return _RISK_WEIGHT * math.sqrt(variance)


def score(stats: dict) -> float:
    """score(pitch) = P(success) x expected_payoff_if_successful - risk_penalty
    (the formula from the request, section 2) -- expected payoff, not raw
    acceptance rate, is what actually gets optimized here."""
    return _laplace_rate(stats) * _avg_success_payoff(stats) - _risk_penalty(stats)


def _is_resisting(stats: dict) -> bool:
    recent = stats["recent"]
    if len(recent) < _RESISTANCE_LIMIT:
        return False
    return not any(recent[-_RESISTANCE_LIMIT:])


def select_argument(game: dict) -> str:
    """Full coverage first (probe every type at least once -- section 3's
    "don't commit to one pitch immediately", generalized past just the first
    3 rounds since a 20-round game has room to actually confirm all 6, not
    just sample half of them), then payoff-weighted proportional selection
    (soft, not argmax -- section 3's 70/20/10 example) over whatever's left
    after excluding any type on an active resistance streak (section 6)."""
    key = _opponent_key(game)
    profile = _profile(key)

    unprobed = [t for t in ARGUMENT_TYPES if profile[t]["attempts"] == 0]
    if unprobed:
        return unprobed[0]  # deterministic order -- fine, order doesn't matter, coverage does

    candidates = [t for t in ARGUMENT_TYPES if not _is_resisting(profile[t])]
    if not candidates:
        candidates = list(ARGUMENT_TYPES)  # every type resisting -- nothing to exclude, use them all

    weights = {t: max(score(profile[t]), 0.0) + _EPSILON_FLOOR for t in candidates}
    total = sum(weights.values())
    r = random.random() * total
    upto = 0.0
    for t in candidates:
        upto += weights[t]
        if r <= upto:
            return t
    return candidates[-1]


def classify_opponent(game: dict) -> tuple[str, str, float] | None:
    """(argument_type, personality_label, score) for whichever type scores
    best against this opponent so far -- section 4's personality read, built
    from actual purchase-by-type data since buyer text isn't available (see
    module docstring). None if every type is still unprobed."""
    profile = _profiles.get(_opponent_key(game))
    if not profile:
        return None
    best = None
    for t in ARGUMENT_TYPES:
        stats = profile.get(t)
        if not stats or stats["attempts"] == 0:
            continue
        s = score(stats)
        if best is None or s > best[1]:
            best = (t, s)
    if best is None:
        return None
    arg_type, s = best
    return arg_type, _PERSONALITY_LABELS[arg_type], s


def best_argument(game: dict) -> tuple[str, float, int] | None:
    """(type, raw success rate, attempts) -- for reporting, mirrors
    persuasion_model.py's function of the same name so learning.log-style
    call sites can drop this in unchanged."""
    profile = _profiles.get(_opponent_key(game))
    if not profile:
        return None
    best = None
    for t in ARGUMENT_TYPES:
        stats = profile.get(t, {})
        attempts = stats.get("attempts", 0)
        if attempts == 0:
            continue
        rate = stats["successes"] / attempts
        if best is None or rate > best[1]:
            best = (t, rate, attempts)
    return best


def _should_recommend(state: dict) -> bool:
    quality = state.get("current_quality")
    if quality != "low":
        return True
    return random.random() < _BLUFF_RATE


def build_message(game: dict, state: dict, argument_type: str, endorse: bool) -> str:
    if not endorse:
        return _DECLINE_TEMPLATES[argument_type].format(
            price=state.get("product_price") or 0, value=state.get("v") or 0
        )
    stage = _profiles.get(_opponent_key(game), {}).get(argument_type, {}).get("attempts", 0)
    chain = _ESCALATION[argument_type]
    template = chain[stage % len(chain)]
    return template.format(price=state.get("product_price") or 0, value=state.get("v") or 0)


def seller_message_action_v2(game: dict, state: dict) -> dict:
    _sync_outcomes(game)
    endorse = _should_recommend(state)
    argument_type = select_argument(game)
    message = build_message(game, state, argument_type, endorse)
    if endorse:
        game_id = game.get("game_id")
        round_num = state.get("round", 1)
        _game_argument_log.setdefault(game_id, {})[round_num] = argument_type
    return {"message": message}


def seller_recommendation_action_v2(game: dict, state: dict) -> dict:
    """Binary-message variant: same honesty policy, no framing choice
    available -- identical constraint to persuasion_model.py's version."""
    return {"decision": "yes" if _should_recommend(state) else "no"}


# ---------------------------------------------------------------------------
# Buyer-side risk control (section 6's actual applicable case -- see module
# docstring on why seller payoff can't be the -$8,000,000 loss).
# ---------------------------------------------------------------------------

# How much extra weight the downside (buying into low quality) gets beyond
# plain expected value. 0 reduces to the plain EV rule simple_agent.py
# already uses; > 0 trades away some expected value to cut tail risk.
_RISK_AVERSION = 0.5

# Per additional detected lie from this seller (see _seller_lie_count),
# effective P(high quality) is scaled down by this much, floored below.
_DISTRUST_PER_LIE = 0.25
_DISTRUST_FLOOR = 0.3


def _true_quality(entry: dict) -> str | None:
    # Unconfirmed field name (unlike "bought", nothing in this codebase's
    # notes pins down what reveals actual quality post-round to the buyer).
    # Hedge every plausible key; if none are present this just returns None
    # and the caller no-ops back to the plain prior-based read.
    for key in ("quality", "true_quality", "actual_quality", "revealed_quality"):
        if key in entry:
            return entry[key]
    return None


def _seller_lie_count(state: dict) -> int:
    """How many times, in THIS game so far, the seller endorsed (a "yes"
    recommendation or a message, as opposed to a decline) a round that later
    revealed itself as low quality. Only counts rounds where quality was
    actually revealed (see _true_quality) -- 0 whenever that field isn't
    present, which just means this guard is a no-op, not wrong."""
    history = state.get("history") or []
    lies = 0
    for entry in history:
        quality = _true_quality(entry)
        if quality != "low":
            continue
        recommended = entry.get("recommendation") or entry.get("decision")
        endorsed = recommended in ("yes", True) or bool(entry.get("message"))
        if endorsed:
            lies += 1
    return lies


def buyer_decision_action_v2(game: dict, state: dict) -> dict:
    """Risk-adjusted version of simple_agent.py's buyer_decision: instead of
    buying whenever plain expected value clears price, downweight the
    probability of high quality by this seller's in-game lie count (if
    detectable -- see _true_quality) and penalize the downside (price paid
    above the low-quality value) beyond what plain EV already charges for it.
    A single seller who's already been caught overselling this game gets
    trusted less on every later round, not just flagged after the fact."""
    price = state["product_price"]
    v = state["v"]
    u = state["u"]
    p = state["p"]

    lies = _seller_lie_count(state)
    effective_p = p * max(_DISTRUST_FLOOR, 1 - _DISTRUST_PER_LIE * lies)

    upside = max(0.0, v - price)
    downside = max(0.0, price - u)
    risk_adjusted_value = effective_p * upside - (1 - effective_p) * downside
    risk_adjusted_value -= _RISK_AVERSION * (1 - effective_p) * downside

    return {"decision": "yes" if risk_adjusted_value > 0 else "no"}
