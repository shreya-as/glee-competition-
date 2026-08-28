"""Clean ablation strategies for the final GLEE experiment.

Each version starts from vanilla_agent and adds one isolated component:

V0: Vanilla baseline.
V1: Vanilla + opponent_model in bargaining/negotiation only.
V2: Vanilla + negotiation concession only.
V3: Vanilla + learned_model calibration in negotiation only.
V4: Vanilla + persuasion_model_v2 in persuasion only.
V5: Best current combination, delegated to simple_agent.strategy.
"""

import learned_model
import opponent_model
import persuasion_model_v2 as persuasion_model
import simple_agent
import vanilla_agent


def strategy_v0(game: dict) -> dict:
    return vanilla_agent.strategy(game)


def strategy_v1(game: dict) -> dict:
    family = game["game_family"]
    actions = game["valid_actions"]
    state = game["game_state"]

    if family == "bargaining":
        return bargaining_with_opponent_model(game, actions, state)
    if family == "negotiation":
        return negotiation_with_opponent_model(game, actions, state)
    return vanilla_agent.strategy(game)


def strategy_v2(game: dict) -> dict:
    if game["game_family"] == "negotiation":
        return negotiation_with_concession(game, game["valid_actions"], game["game_state"])
    return vanilla_agent.strategy(game)


def strategy_v3(game: dict) -> dict:
    if game["game_family"] == "negotiation":
        return negotiation_with_learned_model(game, game["valid_actions"], game["game_state"])
    return vanilla_agent.strategy(game)


def strategy_v4(game: dict) -> dict:
    family = game["game_family"]
    actions = game["valid_actions"]
    state = game["game_state"]

    if family == "persuasion":
        if actions["type"] == "seller_message":
            return persuasion_model.seller_message_action_v2(game, state)
        if actions["type"] == "seller_recommendation":
            return persuasion_model.seller_recommendation_action_v2(game, state)
        if actions["type"] == "buyer_decision":
            return persuasion_model.buyer_decision_action_v2(game, state)
    return vanilla_agent.strategy(game)


def strategy_v5(game: dict) -> dict:
    return simple_agent.strategy(game)


def bargaining_with_opponent_model(game: dict, actions: dict, state: dict) -> dict:
    money = state["money_to_divide"]
    favor = 0.5
    accept_threshold = 0.4

    if actions["type"] == "decision":
        opponent_offer = opponent_model.extract_opponent_offer(game)
        if opponent_offer is not None:
            opponent_model.record_offer(game, opponent_offer)

    label, _ = opponent_model.classify(game)
    if label == "aggressive":
        favor -= 0.03
        accept_threshold -= 0.05
    elif label == "cooperative":
        favor += 0.03
        accept_threshold += 0.05

    favor = min(0.95, max(0.05, favor))
    accept_threshold = min(0.9, max(0.05, accept_threshold))

    if actions["type"] == "offer":
        me = state.get("current_player")
        my_share = money * favor
        other_share = money - my_share
        my_key = "alice_gain" if me == "player_1" else "bob_gain"
        other_key = "bob_gain" if me == "player_1" else "alice_gain"
        return {my_key: my_share, other_key: other_share, "message": "Fair split?"}

    if actions["type"] == "decision":
        offer = state["last_offer"]
        my_gain = offer[f"{state['current_player']}_gain"]
        return {"decision": "accept" if my_gain >= money * accept_threshold else "reject"}

    return {}


def negotiation_with_opponent_model(game: dict, actions: dict, state: dict) -> dict:
    me = state["current_player"]
    role = state[f"{me}_role"]
    my_value = state[f"{me}_value"]

    if actions["type"] == "offer":
        return vanilla_agent.negotiation_strategy(actions, state)

    if actions["type"] == "decision":
        opponent_offer = opponent_model.extract_opponent_offer(game)
        if opponent_offer is not None:
            opponent_model.record_offer(game, opponent_offer)
        label, _ = opponent_model.classify(game)

        offer_price = state["last_offer"]["price"]
        profitable = offer_price <= my_value if role == "buyer" else offer_price >= my_value
        if profitable:
            margin = (
                (my_value - offer_price) / my_value
                if role == "buyer"
                else (offer_price - my_value) / my_value
            )
            min_margin = {"aggressive": 0.02, "cooperative": 0.10}.get(label, 0.0)
            if margin >= min_margin:
                return {"decision": "AcceptOffer"}

        if role == "buyer":
            multiplier = {"aggressive": 0.9, "cooperative": 0.75}.get(label, 0.8)
            return {"decision": "RejectOffer", "product_price": my_value * multiplier}
        multiplier = {"aggressive": 1.15, "cooperative": 1.4}.get(label, 1.3)
        return {"decision": "RejectOffer", "product_price": my_value * multiplier}

    return {}


def _negotiation_concession_ramp(round_num: int) -> float:
    if round_num <= 20:
        return 0.15 * (round_num / 20)
    if round_num <= 50:
        return 0.15 + 0.30 * (round_num - 20) / 30
    if round_num <= 75:
        return 0.45 + 0.35 * (round_num - 50) / 25
    if round_num <= 90:
        return 0.80 + 0.20 * (round_num - 75) / 15
    return 1.0


def negotiation_with_concession(game: dict, actions: dict, state: dict) -> dict:
    me = state["current_player"]
    role = state[f"{me}_role"]
    my_value = state[f"{me}_value"]

    if actions["type"] == "offer":
        return vanilla_agent.negotiation_strategy(actions, state)

    if actions["type"] == "decision":
        offer_price = state["last_offer"]["price"]
        if role == "buyer" and offer_price <= my_value:
            return {"decision": "AcceptOffer"}
        if role == "seller" and offer_price >= my_value:
            return {"decision": "AcceptOffer"}

        ramp = _negotiation_concession_ramp(state.get("round", 1))
        if role == "buyer":
            return {"decision": "RejectOffer", "product_price": my_value * (0.8 + 0.15 * ramp)}
        return {"decision": "RejectOffer", "product_price": my_value * (1.3 - 0.15 * ramp)}

    return {}


def negotiation_with_learned_model(game: dict, actions: dict, state: dict) -> dict:
    me = state["current_player"]
    role = state[f"{me}_role"]
    my_value = state[f"{me}_value"]

    if actions["type"] == "offer":
        return vanilla_agent.negotiation_strategy(actions, state)

    if actions["type"] == "decision":
        offer_price = state["last_offer"]["price"]
        margin = (
            (my_value - offer_price) / my_value
            if role == "buyer"
            else (offer_price - my_value) / my_value
        )
        learned_margin = learned_model.implied_accept_threshold("negotiation", state.get("round", 1))
        min_margin = 0.0 if learned_margin is None else max(-0.1, min(0.2, learned_margin))

        if margin >= min_margin:
            return {"decision": "AcceptOffer"}

        if role == "buyer":
            return {"decision": "RejectOffer", "product_price": my_value * 0.8}
        return {"decision": "RejectOffer", "product_price": my_value * 1.3}

    return {}


STRATEGIES = {
    "v0": strategy_v0,
    "v1": strategy_v1,
    "v2": strategy_v2,
    "v3": strategy_v3,
    "v4": strategy_v4,
    "v5": strategy_v5,
}
