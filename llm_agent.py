"""LLM-driven agent for the GLEE competition.

Same three game families as simple_agent.py, but each move is decided by
Claude with a strategy baked into the system prompt (Rubinstein-style
bargaining, anchored negotiation, reputation-aware persuasion), backed by
light in-memory opponent modeling. Falls back to simple_agent.py's
deterministic rules if the LLM call fails or returns something invalid,
so a flaky API call never burns one of the game's limited move attempts.

Usage:
    pip install glee-sdk anthropic
    export GLEE_API_KEY=glee_...        # from your dashboard at glee-competition.com
    export ANTHROPIC_API_KEY=sk-ant-...
    python llm_agent.py
"""

import json
import logging
import os

import anthropic
from dotenv import load_dotenv

import opponent_model
import persuasion_model
import simple_agent
from game_logger import LoggingGleeClient

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger("llm_agent")

MODEL = "claude-sonnet-5"

_client = None


def _anthropic_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


# ---------------------------------------------------------------------------
# Opponent modeling — in-process, keyed by game_id (always available) and by
# opponent name (only in the disclosed half of games; used for persuasion
# reputation tracking across games with the same buyer/seller).
# ---------------------------------------------------------------------------

_game_history: dict[str, list[dict]] = {}
_reputation: dict[str, dict] = {}  # opponent name -> {"sold": int, "honest": int}


def _record_round(game: dict) -> list[dict]:
    history = _game_history.setdefault(game["game_id"], [])
    last_offer = game["game_state"].get("last_offer")
    if last_offer and (not history or history[-1] != last_offer):
        history.append(last_offer)
    return history


def _opponent_key(game: dict) -> str | None:
    opponent = game.get("opponent") or {}
    return opponent.get("name")


# ---------------------------------------------------------------------------
# System prompts — one per family, baking in the plan's strategy.
# ---------------------------------------------------------------------------

BARGAINING_SYSTEM = """You are playing a two-player bargaining game (Rubinstein-style alternating offers) over a fixed amount of money. Your goals, in priority order:
1. Estimate the opponent's patience/discount factor from how their early offers concede (or don't). A slow-conceding opponent is patient — move toward a faster, less lopsided deal with them rather than stalling; a fast-conceding opponent can be pushed for a better split.
2. Converge toward the stationary Rubinstein split implied by the discount factors, not a rigid 50/50 — but don't let the negotiation drag needlessly: every extra round risks the discounted pie shrinking for both sides, so a fair deal now beats a marginally better deal many rounds later.
3. When deciding, accept any offer that is at or above your fair-split threshold given the round and apparent patience; reject and counter otherwise, moving your own offer toward the midpoint over successive rounds rather than holding a fixed anchor.
4. If the user message flags the opponent as read "aggressive" (rejects often, concedes little), accept earlier and concede faster yourself to avoid a deadlock. If flagged "cooperative" (concedes readily), hold out longer and extract more surplus. Ignore a "random" flag or no flag at all — not enough signal yet.
5. Keep any message short (1-2 sentences). Delay has a real cost here — money now beats a marginally better split several rounds later — so don't manufacture extra back-and-forth once a clearly acceptable offer is on the table.
Respond only by calling the provided tool with your move."""

NEGOTIATION_SYSTEM = """You are playing a buyer/seller price negotiation game. Your goals, in priority order:
1. Anchor your opening offer near a fair-price midpoint between your own valuation and a reasonable estimate of the counterparty's — not at an extreme — and justify it briefly in your message (framing matters to the opponent's willingness to concede).
2. Never reject a mutually beneficial deal: if the price on the table is already profitable for you relative to your own valuation, accept it rather than holding out for a marginally better price. Blocking an efficient deal only lowers your own score.
3. When countering, move your price toward the midpoint of the last two offers rather than re-anchoring at your original extreme — genuine concessions signal you'll actually close, which gets you to a deal faster.
4. If the user message flags the opponent as read "aggressive" (rejects often, concedes little), accept earlier and concede faster yourself to avoid a deadlock. If flagged "cooperative" (concedes readily), hold out longer and extract more surplus. Ignore a "random" flag or no flag at all — not enough signal yet.
5. Keep any message short (1-2 sentences). Delay has a real cost here — money now beats a marginally better price several rounds later — so don't manufacture extra back-and-forth once a clearly acceptable offer is on the table.
Respond only by calling the provided tool with your move."""

PERSUASION_SYSTEM = """You are playing a repeated seller/buyer persuasion game about a product of uncertain quality. Your goals, in priority order:
1. If you are the seller facing a long-lived/repeated buyer (multiple rounds against the same counterparty, reputation persists), be mostly honest in your recommendations/messages — an occasional calculated misleading recommendation is fine when the immediate gain clearly outweighs the reputational cost of being caught, but don't lie so often that the buyer's trust collapses and they stop buying.
2. If the buyer is myopic (one-shot, no memory of your past behavior carried into their decision), you can push harder — recommend/oversell more aggressively since there is no future reputational cost with THIS buyer, but stay plausible enough to be believed given the prior probability of high quality.
3. If you are the buyer, weigh the seller's message and recommendation against the prior probability of high quality and the seller's known track record if visible; accept when your expected value exceeds the price, reject otherwise.
4. If you are the seller and the user message shows argument-type success rates for this opponent, that's this opponent's actual buy rate per pitch framing (economic/fairness/safety/social/convenience/long_term) observed so far — lean toward whichever framing has worked, don't just repeat the same style every round.
Respond only by calling the provided tool with your move."""


# ---------------------------------------------------------------------------
# Tool schemas — force the exact key/value shapes the GLEE server validates.
# ---------------------------------------------------------------------------

def _bargaining_tool(actions: dict, state: dict) -> dict:
    if actions["type"] == "offer":
        return {
            "name": "make_offer",
            "description": "Propose a split of the money.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "alice_gain": {"type": "number"},
                    "bob_gain": {"type": "number"},
                    "message": {"type": "string"},
                },
                "required": ["alice_gain", "bob_gain"],
            },
        }
    return {
        "name": "decide",
        "description": "Accept or reject the last offer.",
        "input_schema": {
            "type": "object",
            "properties": {"decision": {"type": "string", "enum": ["accept", "reject"]}},
            "required": ["decision"],
        },
    }


def _negotiation_tool(actions: dict, state: dict) -> dict:
    if actions["type"] == "offer":
        return {
            "name": "make_offer",
            "description": "Propose a product price.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "product_price": {"type": "number"},
                    "message": {"type": "string"},
                },
                "required": ["product_price"],
            },
        }
    return {
        "name": "decide",
        "description": "Respond to the last price offer.",
        "input_schema": {
            "type": "object",
            "properties": {
                "decision": {"type": "string", "enum": ["AcceptOffer", "RejectOffer"]},
                "product_price": {"type": "number", "description": "Counter-offer price, required when rejecting."},
                "message": {"type": "string"},
            },
            "required": ["decision"],
        },
    }


def _persuasion_tool(actions: dict, state: dict) -> dict:
    if actions["type"] == "seller_message":
        return {
            "name": "send_message",
            "description": "Send a free-text recommendation message to the buyer.",
            "input_schema": {
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
        }
    if actions["type"] == "seller_recommendation":
        return {
            "name": "recommend",
            "description": "Recommend the buyer buy or not.",
            "input_schema": {
                "type": "object",
                "properties": {"decision": {"type": "string", "enum": ["yes", "no"]}},
                "required": ["decision"],
            },
        }
    return {
        "name": "buy_decision",
        "description": "Decide whether to buy.",
        "input_schema": {
            "type": "object",
            "properties": {"decision": {"type": "string", "enum": ["yes", "no"]}},
            "required": ["decision"],
        },
    }


_TOOL_BUILDERS = {
    "bargaining": _bargaining_tool,
    "negotiation": _negotiation_tool,
    "persuasion": _persuasion_tool,
}

_SYSTEM_PROMPTS = {
    "bargaining": BARGAINING_SYSTEM,
    "negotiation": NEGOTIATION_SYSTEM,
    "persuasion": PERSUASION_SYSTEM,
}


def _build_user_message(
    game: dict,
    history: list[dict],
    reputation: dict | None,
    opponent_read: tuple[str, dict] | None,
    argument_rates: dict | None = None,
) -> str:
    parts = [
        f"Situation: {game['prompt']}",
        f"Game state: {json.dumps(game['game_state'], default=str)}",
        f"Actions expected: {json.dumps(game['valid_actions'])}",
    ]
    if history:
        parts.append(f"Offer history so far this game: {json.dumps(history, default=str)}")
    if reputation is not None:
        parts.append(f"Known track record with this opponent: {json.dumps(reputation)}")
    if argument_rates:
        parts.append(f"Argument-type success rates with this opponent so far: {json.dumps(argument_rates)}")
    if opponent_read is not None:
        label, evidence = opponent_read
        parts.append(
            f"Opponent read as: {label} (evidence: {json.dumps(evidence, default=str)})"
        )
    return "\n".join(parts)


def _fallback(game: dict) -> dict:
    logger.warning(f"Game {game['game_id']}: falling back to rule-based strategy")
    return simple_agent.strategy(game)


def decide(game: dict) -> dict:
    family = game["game_family"]
    state = game["game_state"]
    actions = game["valid_actions"]

    history = _record_round(game)
    reputation = None
    argument_rates = None
    if family == "persuasion":
        key = _opponent_key(game)
        if key:
            reputation = _reputation.setdefault(key, {"sold": 0, "honest": 0})
        if actions.get("type") == "seller_message":
            argument_rates = persuasion_model.success_rates(game) or None

    opponent_read = None
    if family in ("bargaining", "negotiation"):
        if actions.get("type") == "decision":
            opponent_offer = opponent_model.extract_opponent_offer(game)
            if opponent_offer is not None:
                opponent_model.record_offer(game, opponent_offer)
        label, evidence = opponent_model.classify(game)
        if label != "unknown":
            opponent_read = (label, evidence)

    try:
        tool = _TOOL_BUILDERS[family](actions, state)
        response = _anthropic_client().messages.create(
            model=MODEL,
            max_tokens=500,
            system=_SYSTEM_PROMPTS[family],
            tools=[tool],
            tool_choice={"type": "tool", "name": tool["name"]},
            messages=[{"role": "user", "content": _build_user_message(game, history, reputation, opponent_read, argument_rates)}],
        )
        tool_use = next(b for b in response.content if b.type == "tool_use")
        action = dict(tool_use.input)
    except Exception:
        logger.exception(f"Game {game['game_id']}: LLM call failed")
        return _fallback(game)

    if not _looks_valid(family, actions, state, action):
        logger.warning(f"Game {game['game_id']}: LLM action failed sanity check: {action}")
        return _fallback(game)

    return action


def _looks_valid(family: str, actions: dict, state: dict, action: dict) -> bool:
    """Cheap local sanity check so an obviously-broken LLM reply never gets
    submitted and burns one of the game's limited move attempts."""
    if family == "bargaining" and actions["type"] == "offer":
        money = state["money_to_divide"]
        gains = action.get("alice_gain"), action.get("bob_gain")
        if any(g is None for g in gains):
            return False
        return abs(sum(gains) - money) < 1e-6
    if family == "negotiation" and actions["type"] == "decision":
        if action.get("decision") == "RejectOffer" and action.get("product_price") is None:
            return False
    return "decision" in action or "product_price" in action or "message" in action or "alice_gain" in action


def run(api_key: str, base_url: str | None = None, concurrency: int = 4) -> None:
    client = (
        LoggingGleeClient(api_key=api_key, base_url=base_url)
        if base_url
        else LoggingGleeClient(api_key=api_key)
    )
    logger.info(f"Agent stats: {client.stats()}")
    client.run(decide, concurrency=concurrency)


if __name__ == "__main__":
    api_key = os.environ.get("GLEE_API_KEY", "")
    if not api_key:
        print("Set GLEE_API_KEY environment variable")
        exit(1)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.warning("ANTHROPIC_API_KEY not set — every move falls back to simple_agent's rule-based logic")

    run(api_key, base_url=os.environ.get("GLEE_API_URL"))
