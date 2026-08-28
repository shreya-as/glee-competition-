"""Adaptive Argument Selection for GLEE's persuasion family (GLEE strategy
notes: "Adaptive Argument Selection" / preference inference).

Only meaningful for the free-text seller-message variant
(state["seller_message_type"] == "text") -- confirmed live via the GLEE API:
the other variant (seller_message_type == "binary") only ever asks for a
yes/no "seller_recommendation", with no text field to frame at all.

Loop, per opponent: pick a pitch framing ("argument type") weighted by how
often it has actually landed a sale with THIS opponent so far -> send it ->
next time we're asked for a seller move, the newly-visible round(s) in
state["history"] reveal whether the buyer bought -> fold that into the
type's success rate -> repeat. There's no direct signal for *why* a buyer
bought (buyers never send text back, only a yes/no `bought`), so "argument
type X works on this opponent" here specifically means "they buy when
pitched with X" -- an implicit preference read via message success, not a
verified causal one, and price/quality luck rides along in the same signal.
That's also the PDF's own model (a plain per-type success rate table), just
without a separately-tracked "preference vector" duplicating the same
information under a different name.

Persisted like opponent_model.py: keyed by opponent name when disclosed
(logs/persuasion_profiles.json, shared across fleet.py's processes,
last-writer-wins on concurrent saves -- same acceptable tradeoff, see that
module's docstring), per-game-only key otherwise. Declines (honestly
advising against a low-quality product) aren't a persuasion attempt and
never get scored -- there's no "argument" being made, just an honest pass.
"""

import json
import logging
import os
import random
import threading

logger = logging.getLogger("persuasion_model")

_STORE_PATH = os.path.join(os.path.dirname(__file__), "logs", "persuasion_profiles.json")
_MAX_SAMPLES = 200

# Floor exploration probability: without this, one or two unlucky early
# rounds could zero out a type's score and it would never get tried again,
# even though a handful of samples says little about a 6-way split. PDF's
# own account of the mechanism -- "if an argument repeatedly fails, lower
# its priority, try another" -- implies exactly this kind of persistent, not
# permanent, demotion.
_EPSILON = 0.15

# Chance to bluff (endorse a known-low-quality product). Set to 0 as an
# experiment: seller payoff here is flat (price if bought, 0 if not --
# confirmed from a real logged round with quality=low, bought=true,
# seller_payoff=1_000_000) so honesty earns nothing directly, only through
# whatever it does to the buyer's willingness to buy in later rounds of the
# same 20-round game. Worth checking logs/games.jsonl payoffs against the
# prior 0.15 rate once enough games accumulate rather than assuming this wins.
_BLUFF_RATE = 0.0

ARGUMENT_TYPES = ["economic", "fairness", "safety", "social", "convenience", "long_term"]

_ENDORSE_TEMPLATES = {
    "economic": "This unit is worth about ${value:,.0f} to you at a price of ${price:,.0f} -- a clear profit if you buy.",
    "fairness": "At ${price:,.0f} against roughly ${value:,.0f} in value, this is a fair, equitable split of the gains for both of us.",
    "safety": "I've checked this one myself -- it's solid, verified quality, so there's minimal risk in buying.",
    "social": "We've got rounds ahead of us together -- backing this deal keeps things working well for both of us.",
    "convenience": "Easy call this round: the numbers clearly favor buying, no need to overthink it.",
    "long_term": "Judge this by how I've called it so far -- trust that track record and take this one.",
}
_DECLINE_TEMPLATES = {
    "economic": "Skip this one -- the numbers don't work in your favor this round, better to preserve your capital.",
    "fairness": "Being straight with you: this wouldn't be a fair trade at this price, so I'd pass.",
    "safety": "I can't vouch for this one's quality -- too much risk here, sit it out.",
    "social": "Looking out for you here: this isn't the round to buy, let's wait for a better one.",
    "convenience": "Easy pass this round -- not worth spending more time on.",
    "long_term": "Same honesty that's guided my calls all game: this one's not worth it, hold off.",
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
    tmp = f"{_STORE_PATH}.tmp"
    with open(tmp, "w") as f:
        json.dump(_profiles, f, indent=2)
    os.replace(tmp, _STORE_PATH)


_profiles = _load()

# Per-game, in-memory only: which argument type we used on each round we
# endorsed, and how far into state["history"] we've already folded into the
# persisted profile. Needed because the API gives back message *text* on
# replay, not our internal type label for it.
_game_argument_log: dict[str, dict[int, str]] = {}
_game_synced_through: dict[str, int] = {}


def _opponent_key(game: dict) -> str:
    name = (game.get("opponent") or {}).get("name")
    return name if name else f"game:{game.get('game_id')}"


def _profile(key: str) -> dict:
    return _profiles.setdefault(
        key, {t: {"attempts": 0, "successes": 0} for t in ARGUMENT_TYPES}
    )


def _sync_outcomes(game: dict) -> None:
    """Fold any newly-visible history rounds into the persisted profile."""
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
                continue  # a decline round, or a round from before we tracked this game
            stats = profile[arg_type]
            stats["attempts"] += 1
            if entry["bought"]:
                stats["successes"] += 1
        if new_max > synced_through:
            _game_synced_through[game_id] = new_max
            _save()


def select_argument(game: dict) -> str:
    """Epsilon-greedy over Laplace-smoothed success rate: mostly exploit
    whatever's worked on this opponent, occasionally still try the rest so a
    type that failed early isn't shut out forever."""
    key = _opponent_key(game)
    profile = _profiles.get(key)
    if not profile or random.random() < _EPSILON:
        return random.choice(ARGUMENT_TYPES)

    def score(t: str) -> float:
        s = profile.get(t, {"attempts": 0, "successes": 0})
        return (s["successes"] + 1) / (s["attempts"] + 2)

    return max(ARGUMENT_TYPES, key=score)


def success_rates(game: dict) -> dict:
    """Read-only view for the LLM agents' prompts (informational only --
    the LLM writes its own free text, this just tells it what's landed with
    this opponent so far). Empty when there's no history yet."""
    profile = _profiles.get(_opponent_key(game))
    if not profile:
        return {}
    return {
        t: round(s["successes"] / s["attempts"], 2)
        for t, s in profile.items()
        if s["attempts"] > 0
    }


def best_argument(game: dict) -> tuple[str, float, int] | None:
    """(type, raw success rate, attempts) for whichever argument type has
    actually scored best against this opponent so far -- for reporting (see
    game_logger.py's learning.log), not selection (select_argument's
    Laplace-smoothed, epsilon-greedy version is what's actually used to pick
    a message). None if there's no data yet."""
    profile = _profiles.get(_opponent_key(game))
    if not profile:
        return None
    best = None
    for t in ARGUMENT_TYPES:
        s = profile.get(t, {"attempts": 0, "successes": 0})
        if s["attempts"] == 0:
            continue
        rate = s["successes"] / s["attempts"]
        if best is None or rate > best[1]:
            best = (t, rate, s["attempts"])
    return best


def _should_recommend(state: dict) -> bool:
    quality = state.get("current_quality")
    if quality != "low":
        return True  # high quality, or unknown -- no basis to decline
    return random.random() < _BLUFF_RATE


def build_message(state: dict, argument_type: str, endorse: bool) -> str:
    template = (_ENDORSE_TEMPLATES if endorse else _DECLINE_TEMPLATES)[argument_type]
    return template.format(price=state.get("product_price") or 0, value=state.get("v") or 0)


def seller_message_action(game: dict, state: dict) -> dict:
    """Full turn for the free-text variant: sync last round's outcome in,
    decide honesty, pick the best-scoring argument type for this opponent,
    remember the choice so next turn's history sync can score it, return the
    move."""
    _sync_outcomes(game)
    endorse = _should_recommend(state)
    argument_type = select_argument(game)
    if endorse:
        game_id = game.get("game_id")
        round_num = state.get("round", 1)
        _game_argument_log.setdefault(game_id, {})[round_num] = argument_type
    return {"message": build_message(state, argument_type, endorse)}


def seller_recommendation_action(game: dict, state: dict) -> dict:
    """Binary-message-variant turn: same honesty policy, no framing choice
    available (see module docstring)."""
    return {"decision": "yes" if _should_recommend(state) else "no"}
