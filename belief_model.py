"""Bayesian belief-state tracking over the opponent's hidden true valuation
(GLEE strategy notes, "Strategy 3": Dynamic Reservation Value & Belief State
Tracking).

Only meaningful for negotiation with complete_information off. Bargaining's
discount factors (delta_1/delta_2) and negotiation's other_value under
complete_information are already given directly in game_state (confirmed
live -- see opponent_model.py's docstring for the bargaining case), so
there's nothing hidden left to infer there. This activates specifically for
the one case GLEE genuinely hides a continuous parameter worth estimating
from behavior: negotiation without complete_information.

Model: a discretized grid over a plausible range for the opponent's true
valuation, updated via a soft-threshold likelihood after every observed
counter-offer -- a seller's asking price is evidence their true cost is at
or below it (downweighted, not ruled out, above it, since a seller can ask
more than their floor); a buyer's offer is evidence their true ceiling is at
or above it, the same asymmetric softness the other way. A plain grid
filter, not a particle filter or anything heavier -- the state space here is
one number, and pure Python handles a 60-point grid over a handful of
updates per game instantly.
"""

import math

_GRID_SIZE = 60
_TAU_FRACTION = 0.15  # likelihood transition width, as a fraction of my_value

# Per-game posterior, in-memory only -- this is about ONE negotiation's
# opponent-in-this-game, not a cross-game trait like opponent_model.py's
# aggressive/cooperative read. Cleared via forget_game when the game ends
# (see game_logger.py).
_posteriors: dict[str, dict] = {}


def _sigmoid(z: float) -> float:
    if z < -30:
        return 0.0
    if z > 30:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


def _init_grid(my_value: float) -> tuple[list[float], list[float]]:
    low, high = my_value * 0.2, my_value * 3.0
    step = (high - low) / (_GRID_SIZE - 1)
    grid = [low + i * step for i in range(_GRID_SIZE)]
    weights = [1.0 / _GRID_SIZE] * _GRID_SIZE
    return grid, weights


def _update(grid: list[float], weights: list[float], price: float, opponent_role: str, tau: float) -> list[float]:
    new_weights = []
    for v, w in zip(grid, weights):
        z = (v - price) / tau
        likelihood = _sigmoid(-z) if opponent_role == "seller" else _sigmoid(z)
        new_weights.append(w * (likelihood + 1e-6))  # floor: a bluff shouldn't zero out a value forever
    total = sum(new_weights)
    return [w / total for w in new_weights]


def update_and_estimate(game: dict) -> dict | None:
    """Call on a negotiation turn. Folds the opponent's latest offer (if
    any) into this game's posterior over their true valuation and returns a
    summary -- mean, and a crude 80% central interval -- or None when
    complete_information already gives the answer directly, or there's not
    enough context yet to start (no my_value/role visible)."""
    if game.get("game_family") != "negotiation":
        return None
    state = game.get("game_state") or {}
    if state.get("complete_information"):
        return None

    me = state.get("current_player")
    role = state.get(f"{me}_role") if me else None
    my_value = state.get(f"{me}_value") if me else None
    if not role or not my_value:
        return None
    opponent_role = "buyer" if role == "seller" else "seller"

    game_id = game.get("game_id")
    entry = _posteriors.get(game_id)
    if entry is None:
        grid, weights = _init_grid(my_value)
        entry = {"grid": grid, "weights": weights}
        _posteriors[game_id] = entry

    offer = state.get("last_offer") or {}
    price = offer.get("price")
    if price is not None:
        tau = max(my_value * _TAU_FRACTION, 1e-6)
        entry["weights"] = _update(entry["grid"], entry["weights"], price, opponent_role, tau)

    grid, weights = entry["grid"], entry["weights"]
    mean = sum(v * w for v, w in zip(grid, weights))

    cum, low = 0.0, grid[0]
    for v, w in zip(grid, weights):
        cum += w
        if cum >= 0.1:
            low = v
            break
    cum, high = 0.0, grid[-1]
    for v, w in zip(reversed(grid), reversed(weights)):
        cum += w
        if cum >= 0.1:
            high = v
            break

    return {"mean": mean, "low": low, "high": high}


def forget_game(game_id: str | None) -> None:
    if game_id is not None:
        _posteriors.pop(game_id, None)
