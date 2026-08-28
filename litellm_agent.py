"""Example agent that lets an LLM choose every move.

This is the pattern most competitive agents start from: hand the model the
game's own prompt plus the machine-readable state, ask for a JSON action, and
never let a bad model response — or a hung provider call — cost you the game:
parse defensively, cap each LLM call with a timeout, and fall back to a safe
valid move.

Works with any provider through litellm (https://docs.litellm.ai): set
LLM_MODEL to any litellm model string and export that provider's API key.

Usage:
    pip install glee-sdk litellm

    export GLEE_API_KEY=glee_...                 # from your dashboard
    export LLM_MODEL=gemini/gemini-2.5-flash     # or gpt-5-mini, claude-sonnet-5, ...
    export GEMINI_API_KEY=...                    # whichever key LLM_MODEL needs

    python litellm_agent.py
"""

import json
import logging
import os
import re
import threading
import time

from dotenv import load_dotenv
from litellm import completion

import opponent_model
import persuasion_model
import simple_agent
from game_logger import LoggingGleeClient

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger("litellm_agent")

# The GLEE SDK polls for ALL pending games every poll_interval and can hand
# several to the strategy function back-to-back, regardless of `concurrency`
# — so a free-tier provider quota (e.g. Gemini's 5 requests/minute) blows
# past instantly unless calls are paced here, in the strategy itself. Calls
# arriving faster than MIN_CALL_INTERVAL skip the LLM and go straight to the
# safe fallback rather than wasting a call on a guaranteed rate-limit error.
MIN_CALL_INTERVAL = float(os.environ.get("LLM_MIN_CALL_INTERVAL", "13"))
_rate_lock = threading.Lock()
_last_call_at = 0.0


def _rate_limit_ok() -> bool:
    global _last_call_at
    with _rate_lock:
        now = time.monotonic()
        if now - _last_call_at < MIN_CALL_INTERVAL:
            return False
        _last_call_at = now
        return True

MODEL = os.environ.get("LLM_MODEL", "gemini/gemini-3.6-flash")

_COMMON_TAIL = """\
Play to maximize YOUR OWN payoff over the whole game.

Reply with ONLY a JSON object for your chosen action — no explanation, no
markdown fences, no text before or after the JSON.
"""

BARGAINING_SYSTEM = """\
You are playing a two-player bargaining game (Rubinstein-style alternating offers) over a fixed amount of money.
1. Estimate the opponent's patience/discount factor from how their early offers concede (or don't). A slow-conceding opponent is patient — move toward a faster, less lopsided deal with them rather than stalling; a fast-conceding opponent can be pushed for a better split.
2. Converge toward the stationary Rubinstein split implied by the discount factors, not a rigid 50/50 — but don't let the negotiation drag needlessly: every extra round risks the discounted pie shrinking for both sides, so a fair deal now beats a marginally better deal many rounds later.
3. Accept any offer at or above your fair-split threshold given the round and apparent patience; reject and counter otherwise, moving your own offer toward the midpoint over successive rounds rather than holding a fixed anchor.
4. If the opponent is flagged below as read "aggressive" (rejects often, concedes little), lean toward accepting earlier and conceding faster yourself — don't let the game deadlock against a stonewaller. If flagged "cooperative" (concedes readily), hold out longer and extract more of the surplus, since they're unlikely to walk. Ignore the flag if it says "random" or is absent — not enough signal yet.
5. Keep any message short (1-2 sentences). Delay has a real cost here — money now beats a marginally better split several rounds later — so don't manufacture extra back-and-forth once a clearly acceptable offer is on the table.
""" + _COMMON_TAIL

NEGOTIATION_SYSTEM = """\
You are playing a buyer/seller price negotiation game.
1. Anchor your opening offer near a fair-price midpoint between your own valuation and a reasonable estimate of the counterparty's — not at an extreme — and justify it briefly in your message (framing matters to the opponent's willingness to concede).
2. Never reject a mutually beneficial deal: if the price on the table is already profitable for you relative to your own valuation, accept it rather than holding out for a marginally better price. Blocking an efficient deal only lowers your own score.
3. When countering, move your price toward the midpoint of the last two offers rather than re-anchoring at your original extreme — genuine concessions signal you'll actually close, which gets you to a deal faster.
4. If the opponent is flagged below as read "aggressive" (rejects often, concedes little), lean toward accepting earlier and conceding faster yourself — don't let the game deadlock against a stonewaller. If flagged "cooperative" (concedes readily), hold out longer and extract more of the surplus, since they're unlikely to walk. Ignore the flag if it says "random" or is absent — not enough signal yet.
5. Keep any message short (1-2 sentences). Delay has a real cost here — money now beats a marginally better price several rounds later — so don't manufacture extra back-and-forth once a clearly acceptable offer is on the table.
""" + _COMMON_TAIL

PERSUASION_SYSTEM = """\
You are playing a repeated seller/buyer persuasion game about a product of uncertain quality.
1. If you are the seller facing a long-lived/repeated buyer (multiple rounds against the same counterparty, reputation persists), be mostly honest in your recommendations/messages — an occasional calculated misleading recommendation is fine when the immediate gain clearly outweighs the reputational cost of being caught, but don't lie so often that the buyer's trust collapses and they stop buying.
2. If the buyer is myopic (one-shot, no memory of your past behavior carried into their decision), you can push harder — recommend/oversell more aggressively since there is no future reputational cost with THIS buyer, but stay plausible enough to be believed given the prior probability of high quality.
3. If you are the buyer, weigh the seller's message and recommendation against the prior probability of high quality and the seller's known track record if visible; accept when your expected value exceeds the price, reject otherwise.
4. If you are the seller and the user message below shows argument-type success rates for this opponent, that's this opponent's actual buy rate per pitch framing (economic/fairness/safety/social/convenience/long_term) observed so far — lean toward whichever framing has worked, don't just repeat the same style every round.
""" + _COMMON_TAIL

_SYSTEM_PROMPTS = {
    "bargaining": BARGAINING_SYSTEM,
    "negotiation": NEGOTIATION_SYSTEM,
    "persuasion": PERSUASION_SYSTEM,
}

# In-process reputation ledger for persuasion, keyed by opponent name — only
# available in the disclosed half of games. Cross-game, unlike game_state's
# own within-game history, so a seller's honesty record with a given
# opponent can carry across separate matches against them.
_reputation: dict[str, dict] = {}


def _opponent_key(game: dict) -> str | None:
    return (game.get("opponent") or {}).get("name")


def build_user_prompt(game: dict) -> str:
    """Everything the model needs, in one message: the situation as prose,
    the raw state, and the exact shape of a legal reply."""
    return (
        f"{game['prompt']}\n\n"
        f"Full game state visible to you (includes `history` of past rounds):\n"
        f"{json.dumps(game['game_state'], indent=2)}\n\n"
        f"Your action must follow this format:\n"
        f"{json.dumps(game['valid_actions'], indent=2)}\n\n"
        f"{_opponent_note(game)}"
        f"Reply with the JSON action only."
    )


def _opponent_note(game: dict) -> str:
    """Persuasion gets its own cross-game reputation ledger (see
    _reputation, below). Bargaining/negotiation get opponent_model.py's
    aggressive/cooperative/random read instead -- see that module for what
    it's built from and its coverage gap."""
    if game["game_family"] == "persuasion":
        key = _opponent_key(game)
        reputation = _reputation.get(key) if key else None
        note = f"Known track record with this opponent: {json.dumps(reputation)}\n\n" if reputation else ""
        if game["valid_actions"].get("type") == "seller_message":
            rates = persuasion_model.success_rates(game)
            if rates:
                note += f"Argument-type success rates with this opponent so far: {json.dumps(rates)}\n\n"
        return note

    if game["valid_actions"].get("type") == "decision":
        opponent_offer = opponent_model.extract_opponent_offer(game)
        if opponent_offer is not None:
            opponent_model.record_offer(game, opponent_offer)
    label, evidence = opponent_model.classify(game)
    if label == "unknown":
        return ""
    return (
        f"Opponent read as: {label} (from observed offer pattern; "
        f"evidence: {json.dumps(evidence, default=str)}). Weigh this in your "
        f"strategy per your system prompt's guidance for that opponent type.\n\n"
    )


def parse_action(text: str) -> dict:
    """Extract the JSON object from a model reply.

    Models sometimes wrap JSON in ```fences``` or add a stray sentence; strip
    fences first, then fall back to the outermost {...} span.
    """
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def safe_action(game: dict) -> dict:
    """A guaranteed-valid move, used when the LLM call or parsing fails.
    Burning one move on this is far cheaper than burning all five attempts on
    malformed JSON, or the 120s turn clock on retries. Delegates to
    simple_agent's evaluative rule-based strategy rather than a naive
    always-accept — a fallback this is hit on often (e.g. under a tight
    provider rate limit) must still weigh whether a deal is actually good,
    not just take whatever's on the table.
    """
    return simple_agent.strategy(game)


def _record_recommendation(game: dict, action: dict) -> None:
    """Lightweight cross-game ledger: what this seller told this opponent,
    round by round, so future games against the same opponent can factor in
    their own track record. No ground-truth outcome is available here (the
    SDK's run loop doesn't surface it back to the strategy function), so this
    is a record of promises made, not a verified honesty score."""
    if game["game_family"] != "persuasion":
        return
    key = _opponent_key(game)
    if not key or "decision" not in action:
        return
    log = _reputation.setdefault(key, {"recommendations": []})["recommendations"]
    log.append(action["decision"])
    del log[:-20]  # keep it bounded


def strategy(game: dict) -> dict:
    if not _rate_limit_ok():
        fallback = safe_action(game)
        logger.info(f"Rate budget spent — playing safe action without calling the LLM: {fallback}")
        return fallback

    system_prompt = _SYSTEM_PROMPTS.get(game["game_family"], BARGAINING_SYSTEM)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": build_user_prompt(game)},
    ]

    # Two tries: if the first reply doesn't parse, show the model its own
    # reply and the error so it can correct itself. After that, play safe.
    for attempt in range(2):
        text = ""
        try:
            # timeout=60 keeps a hanging provider call well inside the 120s
            # turn clock, so the fallback below can still play a safe move.
            response = completion(model=MODEL, messages=messages, timeout=60)
            text = response.choices[0].message.content or ""
            action = parse_action(text)
            logger.info("[%s] %s -> %s", game["game_family"], game["valid_actions"]["type"], action)
            if game["valid_actions"]["type"] == "seller_recommendation":
                _record_recommendation(game, action)
            return action
        except Exception as e:  # provider error, timeout, or unparseable reply
            logger.warning("LLM attempt %d failed: %s", attempt + 1, e)
            if attempt == 0:
                messages.append({"role": "assistant", "content": text})
                messages.append({
                    "role": "user",
                    "content": f"Your reply was not a valid JSON action ({e}). "
                               f"Reply with ONLY the corrected JSON object.",
                })

    fallback = safe_action(game)
    logger.warning("Falling back to safe action: %s", fallback)
    return fallback


def run(api_key: str, base_url: str | None = None, concurrency: int = 8, poll_interval: float = 2.0) -> None:
    client = (
        LoggingGleeClient(api_key=api_key, base_url=base_url)
        if base_url
        else LoggingGleeClient(api_key=api_key)
    )
    logger.info(f"Agent stats: {client.stats()}")
    # An LLM call takes seconds, so play several games in parallel — while one
    # game waits on the model, the others keep moving. Tune concurrency and
    # poll_interval together against your provider's requests/minute quota:
    # each poll cycle can issue up to `concurrency` calls.
    client.run(strategy, concurrency=concurrency, poll_interval=poll_interval)


if __name__ == "__main__":
    api_key = os.environ.get("GLEE_API_KEY", "")
    if not api_key:
        print("Set GLEE_API_KEY environment variable")
        exit(1)

    # Defaults to the production server (https://glee-competition.com). Set
    # GLEE_API_URL only when pointing at a local backend during development.
    run(api_key, base_url=os.environ.get("GLEE_API_URL"))
