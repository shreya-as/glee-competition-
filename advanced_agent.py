"""Advanced frozen strategies for the final GLEE push.

These versions keep bargaining and negotiation fixed across the three agents:

- Bargaining: adaptive opponent-aware concession.
- Negotiation: adaptive opponent-aware threshold.

They differ only in persuasion:

- V5: payoff-aware persuasion via persuasion_model_v2.
- V6: opponent-specific persuasion, exploiting the best known framing.
- V7: contextual-bandit persuasion with price/prior-aware exploration.
"""

import math
import random

import opponent_model
import persuasion_model_v2 as persuasion_model
import vanilla_agent


def strategy_v5(game: dict) -> dict:
    return _strategy(game, persuasion_mode="payoff")


def strategy_v6(game: dict) -> dict:
    return _strategy(game, persuasion_mode="opponent")


def strategy_v7(game: dict) -> dict:
    return _strategy(game, persuasion_mode="contextual")


def _strategy(game: dict, persuasion_mode: str) -> dict:
    family = game["game_family"]
    actions = game["valid_actions"]
    state = game["game_state"]

    if family == "bargaining":
        return bargaining_adaptive(game, actions, state)
    if family == "negotiation":
        return negotiation_adaptive_threshold(game, actions, state)
    if family == "persuasion":
        return persuasion_adaptive(game, actions, state, persuasion_mode)
    return vanilla_agent.strategy(game)


def bargaining_adaptive(game: dict, actions: dict, state: dict) -> dict:
    money = state["money_to_divide"]
    round_num = state.get("round", 1)
    me = state.get("current_player")

    favor = 0.52
    accept_threshold = 0.4
    ramp_rounds = 18

    if actions["type"] == "decision":
        opponent_offer = opponent_model.extract_opponent_offer(game)
        if opponent_offer is not None:
            opponent_model.record_offer(game, opponent_offer)

    label, _ = opponent_model.classify(game)
    if label == "aggressive":
        favor -= 0.04
        accept_threshold -= 0.06
        ramp_rounds = 8
    elif label == "cooperative":
        favor += 0.04
        accept_threshold += 0.04
        ramp_rounds = 28

    my_delta = state.get("delta_1" if me == "player_1" else "delta_2")
    other_delta = state.get("delta_2" if me == "player_1" else "delta_1")
    if my_delta is not None and other_delta is not None:
        patience_edge = max(-0.04, min(0.04, (my_delta - other_delta) * 0.12))
        favor += patience_edge
        accept_threshold += patience_edge

    ramp = min(1.0, max(0.0, (round_num - 1) / ramp_rounds))
    favor = 0.5 + (favor - 0.5) * (1 - ramp)
    accept_threshold = max(0.28, accept_threshold - 0.12 * ramp)

    favor = min(0.9, max(0.1, favor))
    accept_threshold = min(0.85, max(0.05, accept_threshold))

    if actions["type"] == "offer":
        my_share = money * favor
        other_share = money - my_share
        my_key = "alice_gain" if me == "player_1" else "bob_gain"
        other_key = "bob_gain" if me == "player_1" else "alice_gain"
        return {
            my_key: my_share,
            other_key: other_share,
            "message": "This split keeps us moving toward agreement.",
        }

    if actions["type"] == "decision":
        offer = state["last_offer"]
        my_gain = offer[f"{state['current_player']}_gain"]
        return {"decision": "accept" if my_gain >= money * accept_threshold else "reject"}

    return {}


def negotiation_adaptive_threshold(game: dict, actions: dict, state: dict) -> dict:
    me = state["current_player"]
    role = state[f"{me}_role"]
    my_value = state[f"{me}_value"]
    other = "player_2" if me == "player_1" else "player_1"
    other_value = state.get(f"{other}_value")
    round_num = state.get("round", 1)

    if actions["type"] == "offer":
        if role == "seller":
            return {"product_price": my_value * 1.35, "message": "A strong but workable opening price."}
        return {"product_price": my_value * 0.8, "message": "A serious opening offer."}

    if actions["type"] == "decision":
        opponent_offer = opponent_model.extract_opponent_offer(game)
        if opponent_offer is not None:
            opponent_model.record_offer(game, opponent_offer)
        label, _ = opponent_model.classify(game)

        offer_price = state["last_offer"]["price"]
        margin = (
            (my_value - offer_price) / my_value
            if role == "buyer"
            else (offer_price - my_value) / my_value
        )

        min_margin = 0.02
        if label == "aggressive":
            min_margin = -0.01
        elif label == "cooperative":
            min_margin = 0.08

        if other_value is not None:
            total_surplus = abs(other_value - my_value)
            if total_surplus > 1e-9:
                my_surplus = (my_value - offer_price) if role == "buyer" else (offer_price - my_value)
                min_share = max(0.25 - 0.015 * round_num, 0.04)
                if label == "aggressive":
                    min_share *= 0.7
                elif label == "cooperative":
                    min_share *= 1.25
                if my_surplus >= 0 and my_surplus / total_surplus >= min_share:
                    return {"decision": "AcceptOffer"}
        elif margin >= min_margin:
            return {"decision": "AcceptOffer"}

        ramp = _negotiation_ramp(round_num, label)
        if role == "buyer":
            counter = my_value * (0.78 + 0.20 * ramp)
            return {"decision": "RejectOffer", "product_price": min(counter, my_value)}
        counter = my_value * (1.32 - 0.22 * ramp)
        return {"decision": "RejectOffer", "product_price": max(counter, my_value)}

    return {}


def _negotiation_ramp(round_num: int, label: str) -> float:
    horizon = {"aggressive": 35, "cooperative": 75}.get(label, 55)
    return min(1.0, max(0.0, (round_num - 1) / horizon))


def persuasion_adaptive(game: dict, actions: dict, state: dict, mode: str) -> dict:
    if actions["type"] == "seller_recommendation":
        return persuasion_model.seller_recommendation_action_v2(game, state)
    if actions["type"] == "buyer_decision":
        return persuasion_model.buyer_decision_action_v2(game, state)
    if actions["type"] != "seller_message":
        return vanilla_agent.persuasion_strategy(actions, state)

    if mode == "opponent":
        return seller_message_opponent_specific(game, state)
    if mode == "contextual":
        return seller_message_contextual_bandit(game, state)
    return persuasion_model.seller_message_action_v2(game, state)


def seller_message_opponent_specific(game: dict, state: dict) -> dict:
    persuasion_model._sync_outcomes(game)
    best = persuasion_model.classify_opponent(game)
    if best is None or random.random() < 0.10:
        return persuasion_model.seller_message_action_v2(game, state)

    argument_type, _, _ = best
    endorse = persuasion_model._should_recommend(state)
    message = persuasion_model.build_message(game, state, argument_type, endorse)
    if endorse:
        game_id = game.get("game_id")
        round_num = state.get("round", 1)
        persuasion_model._game_argument_log.setdefault(game_id, {})[round_num] = argument_type
    return {"message": message}


def seller_message_contextual_bandit(game: dict, state: dict) -> dict:
    persuasion_model._sync_outcomes(game)
    argument_type = _select_contextual_argument(game, state)
    endorse = persuasion_model._should_recommend(state)
    message = persuasion_model.build_message(game, state, argument_type, endorse)
    if endorse:
        game_id = game.get("game_id")
        round_num = state.get("round", 1)
        persuasion_model._game_argument_log.setdefault(game_id, {})[round_num] = argument_type
    return {"message": message}


def _select_contextual_argument(game: dict, state: dict) -> str:
    profile = persuasion_model._profile(persuasion_model._opponent_key(game))
    untried = [t for t in persuasion_model.ARGUMENT_TYPES if profile[t]["attempts"] == 0]
    if untried:
        return random.choice(untried)

    price = float(state.get("product_price") or 0)
    high_value = float(state.get("v") or 0)
    low_value = float(state.get("u") or 0)
    prior = float(state.get("p") or 0.5)
    expected_value = prior * high_value + (1 - prior) * low_value
    value_gap = (expected_value - price) / max(abs(price), 1.0)

    total_attempts = sum(profile[t]["attempts"] for t in persuasion_model.ARGUMENT_TYPES)
    scores = {}
    for arg_type in persuasion_model.ARGUMENT_TYPES:
        stats = profile[arg_type]
        base = persuasion_model.score(stats)
        exploration = math.sqrt(math.log(total_attempts + 1) / (stats["attempts"] + 1))
        context_bonus = _context_bonus(arg_type, value_gap, prior)
        scores[arg_type] = base + 0.10 * abs(base or 1.0) * exploration + context_bonus
    return max(scores, key=scores.get)


def _context_bonus(argument_type: str, value_gap: float, prior: float) -> float:
    if argument_type == "economic" and value_gap > 0:
        return 0.05 * abs(value_gap)
    if argument_type == "safety" and prior < 0.45:
        return 0.03
    if argument_type == "convenience" and value_gap > 0.25:
        return 0.03
    if argument_type == "fairness" and -0.1 <= value_gap <= 0.15:
        return 0.02
    if argument_type == "long_term" and prior >= 0.55:
        return 0.02
    if argument_type == "social" and prior >= 0.5:
        return 0.01
    return 0.0


STRATEGIES = {
    "v5": strategy_v5,
    "v6": strategy_v6,
    "v7": strategy_v7,
}
