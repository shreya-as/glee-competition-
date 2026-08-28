"""Simple example agent that plays all three GLEE game families.

Usage:
    pip install glee-sdk
    export GLEE_API_KEY=glee_...   # from your dashboard at glee-competition.com
    python simple_agent.py
"""

import logging
import os

import learned_model
import opponent_model
import persuasion_model_v2 as persuasion_model
from game_logger import LoggingGleeClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")


def strategy(game: dict) -> dict:
    """Variant strategy under A/B test: negotiation hold-out-once (measured
    null last round, kept for now), a bargaining anchor test (55/45 split,
    0.45 accept threshold, arxiv 2512.13063), and a NEW fix in both games:
    concede=True. Our own live logs showed a real bug -- a bargaining game
    stuck at round 59 and a negotiation game at round 97, because our offers
    never move across rounds regardless of history, so two static agents can
    deadlock indefinitely. concede softens our ask toward fair as the round
    count climbs (self-value-relative only, no opponent data needed), so a
    deadlocked game actually closes and banks a profit instead of running out
    the clock. This targets realized payoff directly, not just accept rate."""
    return _strategy(game, negotiation_holdout=True, bargaining_favor=0.55, bargaining_accept=0.45, concede=True)


def strategy_control(game: dict) -> dict:
    """Control strategy for the A/B test: the proven baseline (50/50 offer,
    0.40 accept threshold, no negotiation hold-out, no round-based
    concession -- offers stay static every round like before). Run on half
    the fleet so every variant change is judged against a same-time,
    same-opponent-pool baseline instead of a noisy before/after comparison."""
    return _strategy(game, negotiation_holdout=False, bargaining_favor=0.5, bargaining_accept=0.4, concede=False)


def _strategy(
    game: dict,
    negotiation_holdout: bool,
    bargaining_favor: float,
    bargaining_accept: float,
    concede: bool,
) -> dict:
    family = game["game_family"]
    actions = game["valid_actions"]
    state = game["game_state"]

    if family == "bargaining":
        return bargaining_strategy(game, actions, state, favor=bargaining_favor, accept_threshold=bargaining_accept, concede=concede)
    elif family == "negotiation":
        return negotiation_strategy(game, actions, state, holdout=negotiation_holdout, concede=concede)
    elif family == "persuasion":
        return persuasion_strategy(game, actions, state)
    else:
        raise ValueError(f"Unknown game family: {family}")


def bargaining_strategy(game: dict, actions: dict, state: dict, favor: float = 0.5, accept_threshold: float = 0.4, concede: bool = False) -> dict:
    """Proposer offers itself `favor` share (0.5 = 50/50 baseline); accepts
    any offer giving >= accept_threshold. Only the message text is dynamic
    beyond that, since framing can only influence the opponent's behavior,
    never make my own accept/offer logic worse.

    alice/bob are fixed identities equal to player_1/player_2 respectively,
    not "me/opponent" -- confirmed live: a real player_2 game's own prompt
    reads "You are Bob", and its round history logs Bob's own past offers
    under the name Bob, not "you". The offer action must therefore address
    the current player by their real identity, not always as alice.

    concede=True erodes both numbers toward fair as the round count climbs
    (linear ramp, fully faded by round 15) -- a deadlocked game converges to
    a real deal instead of running out the clock at round 50+.

    Opponent-type classification (opponent_model.py) adjusts both knobs
    further: an opponent read as aggressive (stonewalls, concedes little)
    gets a faster concession ramp and a lower accept bar, closing the deal
    before it deadlocks; one read as cooperative (concedes readily) gets a
    slower ramp and a higher bar, so we hold out for a better split.
    `random` or `unknown` opponents (inconsistent, or not enough signal yet)
    fall back to the untouched baseline numbers.

    Discount-factor edge: bargaining's game_state exposes each side's own
    per-round discount factor as delta_1/delta_2 (confirmed via a live
    game_state pull -- not documented in the SDK). Whoever discounts more
    loses more value by dragging the game out, so a real, game-given
    patience edge -- not just one inferred from behavior -- should make us
    firmer when it favors us and softer when it doesn't. Small and clamped;
    stacks additively with the opponent-type adjustment above rather than
    overriding it.

    Learned accept threshold (learned_model.py / train_model.py): a logistic
    regression trained offline on every logged game's final offer and round,
    predicting P(agreement). Its implied accept-share at this round is
    blended in at 20% weight -- not more, since it's fit on pooled outcomes
    across every opponent and agent variant that's played so far, not this
    specific game, and the underlying data is heavily agreement-skewed (see
    train_model.py's docstring on what it can and can't tell you). None
    (untrained, or too little data) is a no-op."""
    money = state["money_to_divide"]
    round_num = state.get("round", 1)

    if actions["type"] == "decision":
        opponent_offer = opponent_model.extract_opponent_offer(game)
        if opponent_offer is not None:
            opponent_model.record_offer(game, opponent_offer)
    label, _ = opponent_model.classify(game)
    ramp_rounds = {"aggressive": 6, "cooperative": 25}.get(label, 15)
    favor = min(0.95, max(0.05, favor + {"aggressive": -0.03, "cooperative": 0.03}.get(label, 0.0)))
    accept_threshold = max(0.05, min(0.9, accept_threshold + {"aggressive": -0.05, "cooperative": 0.05}.get(label, 0.0)))

    me = state.get("current_player")
    my_delta = state.get("delta_1" if me == "player_1" else "delta_2")
    other_delta = state.get("delta_2" if me == "player_1" else "delta_1")
    if my_delta is not None and other_delta is not None:
        patience_edge = max(-0.05, min(0.05, (my_delta - other_delta) * 0.15))
        favor = min(0.95, max(0.05, favor + patience_edge))
        accept_threshold = max(0.05, min(0.9, accept_threshold + patience_edge))

    learned_accept = learned_model.implied_accept_threshold("bargaining", round_num)
    if learned_accept is not None:
        accept_threshold = 0.8 * accept_threshold + 0.2 * learned_accept

    ramp = min(1.0, max(0.0, (round_num - 1) / ramp_rounds)) if concede else 0.0
    effective_favor = favor - (favor - 0.5) * ramp
    effective_accept = accept_threshold - 0.15 * ramp

    if actions["type"] == "offer":
        my_share = money * effective_favor
        other_share = money - my_share
        message = (
            "Fair split?"
            if round_num <= 1
            else "Still a fair split — closing now beats both of us losing value to more delay."
        )
        my_key = "alice_gain" if me == "player_1" else "bob_gain"
        other_key = "bob_gain" if me == "player_1" else "alice_gain"
        return {my_key: my_share, other_key: other_share, "message": message}

    if actions["type"] == "decision":
        offer = state["last_offer"]
        # In the decision phase, current_player is always the offer's receiver.
        my_gain = offer[f"{state['current_player']}_gain"]

        if my_gain >= money * effective_accept:
            return {"decision": "accept"}
        return {"decision": "reject"}

    return {}


def _negotiation_concession_ramp(round_num: int, ramp_rounds: int) -> float:
    """Tiered concession schedule for the no-fair_price counter (see
    negotiation_strategy below), replacing a single linear ramp that
    saturated at 1.0 by round `ramp_rounds` (<=25) and then held a STATIC
    offer forever. Live logs showed negotiation deadlocking at round
    97/100/108 -- two agents that don't already overlap by round 25 never
    closed, since neither side's offer moved again after that point. This
    keeps conceding in four progressively steeper bands (protect value
    early, concede faster as rounds mount, controlled final push late)
    instead of parking indefinitely. Early bands compress for an aggressive
    opponent (ramp_rounds=6, from negotiation_strategy's `label`) so we
    concede sooner against one that stonewalls -- but the scale is capped at
    1.0 rather than also stretching out for a cooperative opponent
    (ramp_rounds=25): the whole point is guaranteeing full concession by
    ~round 90 no matter the label, since the observed deadlocks (round
    97/100/108) are exactly what an uncapped, slower-for-cooperative
    schedule would reproduce. Exact percentages are starting points, not
    tuned constants -- rerun baseline.py --last N after enough games and
    adjust the four (fraction, band-end) points below if needed."""
    scale = min(ramp_rounds / 15.0, 1.0)
    band1, band2, band3, band4 = 20 * scale, 50 * scale, 75 * scale, 90 * scale

    if round_num <= band1:
        return 0.15 * (round_num / band1)
    if round_num <= band2:
        return 0.15 + 0.30 * (round_num - band1) / (band2 - band1)
    if round_num <= band3:
        return 0.45 + 0.35 * (round_num - band2) / (band3 - band2)
    if round_num <= band4:
        return 0.80 + 0.20 * (round_num - band3) / (band4 - band3)
    return 1.0


def negotiation_strategy(game: dict, actions: dict, state: dict, holdout: bool = True, concede: bool = False) -> dict:
    """Offer midpoint; accept if profitable. Decision numbers otherwise match
    the proven baseline — only the message text is dynamic, for the same
    reason as bargaining above.

    Exception: when complete_information reveals the opponent's true
    valuation too, the fair (50/50-surplus) price is known exactly rather
    than guessed at. The GLEE paper (arxiv 2410.05254, Table 5) finds
    complete_information genuinely improves both efficiency and fairness in
    negotiation specifically (unlike bargaining, where it doesn't) — so here,
    don't accept a merely-profitable offer that hands the opponent nearly
    all of the surplus; counter toward the known fair price instead, up to
    the point of diminishing returns.

    concede=True softens the no-complete_information counter (0.8x/1.3x
    my_value, fixed forever in the baseline) toward my_value as the round
    count climbs, via _negotiation_concession_ramp's tiered schedule: small
    moves through round ~20, gradually larger through ~50 and ~75, a
    controlled final push out to ~90, then held (never past my_value — still
    profitable, never full capitulation). Two static counters that never
    move can deadlock indefinitely (observed live: real games stuck at round
    97/100/108, past the old ramp's round-25 cap) — this keeps closing the
    gap all the way out instead of parking once the ramp saturates.

    Opponent-type classification (opponent_model.py) tightens or loosens
    three of the knobs above: against an aggressive opponent (stonewalls,
    concedes little) we require a smaller share of the known surplus before
    accepting, hold out less on an unproven first offer, and concede faster
    round-over-round -- all aimed at closing before a deadlock. Against a
    cooperative opponent (concedes readily) we do the opposite on all three,
    extracting more surplus since they're not going to walk. `random` or
    `unknown` opponents get the untouched baseline."""
    me = state["current_player"]            # on your turn, that's you
    role = state[f"{me}_role"]              # "seller" or "buyer"
    my_value = state[f"{me}_value"]        # your own valuation — always visible to you
    other = "player_2" if me == "player_1" else "player_1"
    other_value = state.get(f"{other}_value")  # only visible when complete_information is true
    fair_price = (my_value + other_value) / 2 if other_value is not None else None
    total_surplus = abs(other_value - my_value) if other_value is not None else None

    if actions["type"] == "decision":
        opponent_offer = opponent_model.extract_opponent_offer(game)
        if opponent_offer is not None:
            opponent_model.record_offer(game, opponent_offer)
    label, _ = opponent_model.classify(game)
    ramp_rounds = {"aggressive": 6, "cooperative": 25}.get(label, 15)
    min_share_mult = {"aggressive": 0.7, "cooperative": 1.3}.get(label, 1.0)
    holdout_margin = {"aggressive": 0.08, "cooperative": 0.20}.get(label, 0.15)

    # Same learned-model blend as bargaining_strategy (see its docstring):
    # a small, dampened nudge toward what train_model.py's fit says tends to
    # close a deal at this round, not a replacement for the holdout logic.
    # Starts as a no-op -- negotiation's model needs my_value/role_type,
    # only logged from this feature onward, so it trains in once the fleet
    # has played enough fresh games (see train_model.py's docstring).
    learned_margin = learned_model.implied_accept_threshold("negotiation", state.get("round", 1))
    if learned_margin is not None:
        holdout_margin = max(0.02, 0.8 * holdout_margin + 0.2 * learned_margin)

    if actions["type"] == "offer":
        if role == "seller":
            return {"product_price": my_value * 1.5, "message": "Good price — priced to close quickly for both of us."}
        else:
            return {"product_price": my_value * 0.7, "message": "Fair offer — let's find a price that works for both sides."}

    if actions["type"] == "decision":
        offer_price = state["last_offer"]["price"]
        profitable = offer_price <= my_value if role == "buyer" else offer_price >= my_value

        if profitable and total_surplus and total_surplus > 1e-9:
            my_surplus = (my_value - offer_price) if role == "buyer" else (offer_price - my_value)
            round_num = state.get("round", 1)
            min_share = max((0.30 - 0.02 * round_num) * min_share_mult, 0.05)
            if my_surplus / total_surplus < min_share:
                profitable = False  # reject and counter toward the known fair price below

        # Same paper (2512.13063): even without complete_information, LLM
        # opponents rarely punish a single counter — they tend to improve
        # their next offer rather than stall or walk. So don't instant-accept
        # a barely-profitable FIRST offer when we have no fair_price to check
        # it against; hold out once, then fall through to accept on later
        # rounds via the existing round>=2 logic below.
        if holdout and profitable and fair_price is None:
            round_num = state.get("round", 1)
            margin = (
                (my_value - offer_price) / my_value
                if role == "buyer"
                else (offer_price - my_value) / my_value
            )
            if round_num <= 1 and margin < holdout_margin:
                profitable = False

        if profitable:
            return {"decision": "AcceptOffer"}

        if fair_price is not None:
            counter = min(fair_price, my_value) if role == "buyer" else max(fair_price, my_value)
            message = "That's a bit lopsided given what I know of both our numbers — let's split it fairly."
            return {"decision": "RejectOffer", "product_price": counter, "message": message}

        round_num = state.get("round", 1)
        ramp = _negotiation_concession_ramp(round_num, ramp_rounds) if concede else 0.0
        if role == "buyer":
            multiplier = 0.8 + 0.15 * ramp  # -> 0.95 by round 15, never overpays above my_value
            return {"decision": "RejectOffer", "product_price": my_value * multiplier, "message": "Closer, but still not quite there for me — here's a number that works."}
        else:  # seller
            multiplier = 1.3 - 0.15 * ramp  # -> 1.15 by round 15, never undersells below my_value
            return {"decision": "RejectOffer", "product_price": my_value * multiplier, "message": "Appreciate the offer, but I need a bit more to make this work."}

    return {}


def persuasion_strategy(game: dict, actions: dict, state: dict) -> dict:
    """Seller: persuasion_model_v2's payoff-aware Adaptive Argument Selection
    -- picks a pitch framing (economic/fairness/safety/social/convenience/
    long_term) scored by expected payoff against this opponent, not just
    acceptance rate (see that module's docstring for why: a type sitting at
    100% acceptance can still throw $0 rounds since price varies round to
    round). Probes every framing at least once before exploiting, escalates
    phrasing on repeat pitches of the same type, and pivots off a type stuck
    on a 3-round decline streak. Only meaningful for the free-text message
    variant; the binary variant (seller_recommendation) shares the same
    honesty policy with no framing choice to make.

    Buyer: risk-adjusted expected value, not plain expected value >= price --
    downweights P(high quality) by how many times this seller has already
    been caught endorsing a low-quality round earlier in THIS game, and
    penalizes the downside (price paid above the low-quality value) beyond
    what plain EV already prices in. Reduces to the old plain-EV rule when
    nothing's been detected yet (see persuasion_model_v2.buyer_decision_action_v2)."""
    if actions["type"] == "seller_message":
        return persuasion_model.seller_message_action_v2(game, state)

    if actions["type"] == "seller_recommendation":
        return persuasion_model.seller_recommendation_action_v2(game, state)

    if actions["type"] == "buyer_decision":
        return persuasion_model.buyer_decision_action_v2(game, state)

    return {}


if __name__ == "__main__":
    api_key = os.environ.get("GLEE_API_KEY", "")
    if not api_key:
        print("Set GLEE_API_KEY environment variable")
        exit(1)

    # Defaults to the production server (https://glee-competition.com). Set
    # GLEE_API_URL only when pointing at a local backend during development.
    base_url = os.environ.get("GLEE_API_URL")
    client = (
        LoggingGleeClient(api_key=api_key, base_url=base_url)
        if base_url
        else LoggingGleeClient(api_key=api_key)
    )

    print(f"Agent stats: {client.stats()}")
    client.run(strategy)
