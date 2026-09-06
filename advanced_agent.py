"""Advanced frozen strategies for the final GLEE push.

These versions keep bargaining and negotiation fixed across V5-V7:

- Bargaining: adaptive opponent-aware concession.
- Negotiation: adaptive opponent-aware threshold.

They differ only in persuasion:

- V5: payoff-aware persuasion via persuasion_model_v2.
- V6: opponent-specific persuasion, exploiting the best known framing.
- V7: contextual-bandit persuasion with price/prior-aware exploration.
- V8: final picked hybrid:
  Bargaining V5 + Negotiation V3 + Persuasion V7.
- V9: leaderboard push with deadline-aware bargaining, value-protecting
  negotiation, and exploit-heavy contextual persuasion.
"""

import math
import random

import learned_model
import opponent_model
import persuasion_model_v2 as persuasion_model
import experiment_agent
import vanilla_agent


def strategy_v5(game: dict) -> dict:
    return _strategy(game, persuasion_mode="payoff")


def strategy_v6(game: dict) -> dict:
    return _strategy(game, persuasion_mode="opponent")


def strategy_v7(game: dict) -> dict:
    return _strategy(game, persuasion_mode="contextual")


def strategy_v8(game: dict) -> dict:
    family = game["game_family"]
    actions = game["valid_actions"]
    state = game["game_state"]

    if family == "bargaining":
        return bargaining_adaptive(game, actions, state)
    if family == "negotiation":
        return experiment_agent.negotiation_with_learned_model(game, actions, state)
    if family == "persuasion":
        return persuasion_adaptive(game, actions, state, "contextual")
    return vanilla_agent.strategy(game)


def strategy_v9(game: dict) -> dict:
    family = game["game_family"]
    actions = game["valid_actions"]
    state = game["game_state"]

    if family == "bargaining":
        return bargaining_leaderboard_push(game, actions, state)
    if family == "negotiation":
        return negotiation_value_protecting(game, actions, state)
    if family == "persuasion":
        return persuasion_leaderboard_push(game, actions, state)
    return vanilla_agent.strategy(game)


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


def bargaining_leaderboard_push(game: dict, actions: dict, state: dict) -> dict:
    money = state["money_to_divide"]
    round_num = state.get("round", 1)
    me = state.get("current_player")

    if actions["type"] == "decision":
        opponent_offer = opponent_model.extract_opponent_offer(game)
        if opponent_offer is not None:
            opponent_model.record_offer(game, opponent_offer)

    label, _ = opponent_model.classify(game)
    progress = _deadline_progress(state, soft_horizon=14)

    favor = 0.535
    accept_threshold = 0.425
    if label == "aggressive":
        favor -= 0.025
        accept_threshold -= 0.045
    elif label == "cooperative":
        favor += 0.035
        accept_threshold += 0.035

    my_delta = state.get("delta_1" if me == "player_1" else "delta_2")
    other_delta = state.get("delta_2" if me == "player_1" else "delta_1")
    if my_delta is not None and other_delta is not None:
        patience_edge = max(-0.045, min(0.045, (my_delta - other_delta) * 0.14))
        favor += patience_edge
        accept_threshold += patience_edge

    learned_accept = learned_model.implied_accept_threshold("bargaining", round_num)
    if learned_accept is not None:
        accept_threshold = 0.9 * accept_threshold + 0.1 * learned_accept

    effective_favor = 0.5 + (favor - 0.5) * (1 - progress)
    effective_accept = accept_threshold - 0.08 * progress
    if _is_final_round(state):
        effective_accept = min(effective_accept, 0.34)

    effective_favor = min(0.62, max(0.5, effective_favor))
    effective_accept = min(0.55, max(0.32, effective_accept))

    if actions["type"] == "offer":
        my_share = money * effective_favor
        other_share = money - my_share
        my_key = "alice_gain" if me == "player_1" else "bob_gain"
        other_key = "bob_gain" if me == "player_1" else "alice_gain"
        return {
            my_key: my_share,
            other_key: other_share,
            "message": "This is a clean split with enough value on both sides to close.",
        }

    if actions["type"] == "decision":
        offer = state["last_offer"]
        my_gain = offer[f"{state['current_player']}_gain"]
        return {"decision": "accept" if my_gain >= money * effective_accept else "reject"}

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


def negotiation_value_protecting(game: dict, actions: dict, state: dict) -> dict:
    me = state["current_player"]
    role = state[f"{me}_role"]
    my_value = float(state[f"{me}_value"])
    other = "player_2" if me == "player_1" else "player_1"
    other_value = state.get(f"{other}_value")
    other_value = float(other_value) if other_value is not None else None
    round_num = state.get("round", 1)
    progress = _deadline_progress(state, soft_horizon=10)

    if actions["type"] == "decision":
        opponent_offer = opponent_model.extract_opponent_offer(game)
        if opponent_offer is not None:
            opponent_model.record_offer(game, opponent_offer)
    label, _ = opponent_model.classify(game)

    if actions["type"] == "offer":
        price = _negotiation_target_price(role, my_value, other_value, progress, label, opening=True)
        return {"product_price": price, "message": "This leaves a real deal on the table now."}

    if actions["type"] != "decision":
        return {}

    offer_price = float(state["last_offer"]["price"])
    margin = (
        (my_value - offer_price) / max(abs(my_value), 1.0)
        if role == "buyer"
        else (offer_price - my_value) / max(abs(my_value), 1.0)
    )
    profitable = margin > 0

    if other_value is not None:
        seller_value = my_value if role == "seller" else other_value
        buyer_value = other_value if role == "seller" else my_value
        total_surplus = buyer_value - seller_value
        if total_surplus <= 0:
            if profitable and (margin >= 0.02 or _is_final_round(state)):
                return {"decision": "AcceptOffer"}
        else:
            my_surplus = (my_value - offer_price) if role == "buyer" else (offer_price - my_value)
            my_share = my_surplus / total_surplus
            min_share = 0.52 - 0.24 * progress
            if label == "aggressive":
                min_share -= 0.08
            elif label == "cooperative":
                min_share += 0.05
            if _is_final_round(state):
                min_share = min(min_share, 0.12)
            if my_share >= max(0.10, min_share):
                return {"decision": "AcceptOffer"}

        counter = _negotiation_target_price(role, my_value, other_value, progress, label, opening=False)
        return {
            "decision": "RejectOffer",
            "product_price": counter,
            "message": "Close, but the surplus split needs to be less one-sided.",
        }

    learned_margin = learned_model.implied_accept_threshold("negotiation", round_num)
    min_margin = 0.075 - 0.055 * progress
    if label == "aggressive":
        min_margin -= 0.025
    elif label == "cooperative":
        min_margin += 0.025
    if learned_margin is not None:
        min_margin = 0.75 * min_margin + 0.25 * max(0.0, min(0.12, learned_margin))
    if _is_final_round(state):
        min_margin = min(min_margin, 0.01)

    if profitable and margin >= max(0.008, min_margin):
        return {"decision": "AcceptOffer"}

    if role == "buyer":
        multiplier = 0.84 + 0.14 * progress
        if label == "aggressive":
            multiplier += 0.035
        counter = min(my_value * 0.985, my_value * multiplier)
    else:
        multiplier = 1.22 - 0.17 * progress
        if label == "aggressive":
            multiplier -= 0.035
        counter = max(my_value * 1.015, my_value * multiplier)

    return {
        "decision": "RejectOffer",
        "product_price": counter,
        "message": "I can move, but I still need positive surplus to close.",
    }


def _negotiation_ramp(round_num: int, label: str) -> float:
    horizon = {"aggressive": 35, "cooperative": 75}.get(label, 55)
    return min(1.0, max(0.0, (round_num - 1) / horizon))


def _deadline_progress(state: dict, soft_horizon: int) -> float:
    round_num = state.get("round", 1)
    max_rounds = state.get("max_rounds")
    if state.get("horizon_known") and max_rounds:
        return min(1.0, max(0.0, (round_num - 1) / max(1, max_rounds - 1)))
    return min(1.0, max(0.0, (round_num - 1) / soft_horizon))


def _is_final_round(state: dict) -> bool:
    max_rounds = state.get("max_rounds")
    return bool(state.get("horizon_known") and max_rounds and state.get("round", 1) >= max_rounds)


def _negotiation_target_price(
    role: str,
    my_value: float,
    other_value: float | None,
    progress: float,
    label: str,
    opening: bool,
) -> float:
    if other_value is not None:
        seller_value = my_value if role == "seller" else other_value
        buyer_value = other_value if role == "seller" else my_value
        surplus = buyer_value - seller_value
        if surplus <= 0:
            return my_value * (1.02 if role == "seller" else 0.98)
        target_share = (0.60 if opening else 0.54) - 0.18 * progress
        if label == "aggressive":
            target_share -= 0.06
        elif label == "cooperative":
            target_share += 0.04
        target_share = min(0.62, max(0.34, target_share))
        if role == "seller":
            return seller_value + surplus * target_share
        return buyer_value - surplus * target_share

    if role == "seller":
        return my_value * (1.24 - 0.16 * progress)
    return my_value * (0.82 + 0.14 * progress)


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


def persuasion_leaderboard_push(game: dict, actions: dict, state: dict) -> dict:
    if actions["type"] == "buyer_decision":
        return persuasion_model.buyer_decision_action_v2(game, state)
    if actions["type"] == "seller_recommendation":
        return {"decision": "yes" if _v9_should_recommend(state) else "no"}
    if actions["type"] != "seller_message":
        return vanilla_agent.persuasion_strategy(actions, state)

    persuasion_model._sync_outcomes(game)
    argument_type = _select_v9_argument(game, state)
    endorse = _v9_should_recommend(state)
    message = persuasion_model.build_message(game, state, argument_type, endorse)
    if endorse:
        game_id = game.get("game_id")
        round_num = state.get("round", 1)
        persuasion_model._game_argument_log.setdefault(game_id, {})[round_num] = argument_type
    return {"message": message}


def _v9_should_recommend(state: dict) -> bool:
    quality = state.get("current_quality")
    if quality != "low":
        return True
    round_num = state.get("round", 1)
    total_rounds = state.get("total_rounds") or 20
    remaining = max(0, total_rounds - round_num)
    price = float(state.get("product_price") or 0)
    high_value = float(state.get("v") or 0)
    prior = float(state.get("p") or 0.5)
    tempting_price = price >= max(1.0, 0.75 * high_value)
    if remaining <= 2 and tempting_price:
        return random.random() < 0.25
    if remaining <= 1 and prior >= 0.8:
        return random.random() < 0.15
    return False


def _select_v9_argument(game: dict, state: dict) -> str:
    profile = persuasion_model._profile(persuasion_model._opponent_key(game))
    attempted = sum(profile[t]["attempts"] for t in persuasion_model.ARGUMENT_TYPES)
    if attempted >= 4 and random.random() >= 0.03:
        return _best_contextual_argument(profile, state)

    global_profile = _global_persuasion_profile()
    if global_profile and attempted < 2:
        return _best_contextual_argument(global_profile, state)

    return _select_contextual_argument(game, state)


def _global_persuasion_profile() -> dict | None:
    profiles = getattr(persuasion_model, "_profiles", {})
    totals = {t: persuasion_model._empty_type_stats() for t in persuasion_model.ARGUMENT_TYPES}
    seen = False
    for profile in profiles.values():
        for arg_type in persuasion_model.ARGUMENT_TYPES:
            stats = profile.get(arg_type)
            if not stats:
                continue
            seen = seen or stats.get("attempts", 0) > 0
            totals[arg_type]["attempts"] += stats.get("attempts", 0)
            totals[arg_type]["successes"] += stats.get("successes", 0)
            totals[arg_type]["total_payoff"] += stats.get("total_payoff", 0.0)
            totals[arg_type]["payoff_history"].extend(stats.get("payoff_history", [])[-20:])
            totals[arg_type]["recent"].extend(stats.get("recent", [])[-10:])
            del totals[arg_type]["payoff_history"][:-200]
            del totals[arg_type]["recent"][:-10]
    return totals if seen else None


def _best_contextual_argument(profile: dict, state: dict) -> str:
    price = float(state.get("product_price") or 0)
    high_value = float(state.get("v") or 0)
    low_value = float(state.get("u") or 0)
    prior = float(state.get("p") or 0.5)
    expected_value = prior * high_value + (1 - prior) * low_value
    value_gap = (expected_value - price) / max(abs(price), 1.0)

    scores = {}
    for arg_type in persuasion_model.ARGUMENT_TYPES:
        stats = profile[arg_type]
        scores[arg_type] = persuasion_model.score(stats) + _context_bonus(arg_type, value_gap, prior)
    return max(scores, key=scores.get)


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
    "v8": strategy_v8,
    "v9": strategy_v9,
}
