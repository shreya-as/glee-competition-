"""Inference side of train_model.py's offline logistic regression. Loads
logs/learned_{family}.json (if it exists -- absent until train_model.py has
run and had enough rows, see its _MIN_ROWS guard) and answers: "at this
round, what share/margin does the data say tends to close a deal?"

Deliberately narrow: one call, `implied_accept_threshold`, returns a single
number or None. Callers (simple_agent.py) blend it into their existing
threshold rather than replacing it outright -- this model is fit on ~2-3
features from final-decision-point data only (see train_model.py's
docstring for what that does and doesn't support), so it's a data-calibrated
nudge, not a policy replacement.
"""

import json
import logging
import math
import os

logger = logging.getLogger("learned_model")

_MODELS: dict[str, dict | None] = {}


def _model_path(family: str) -> str:
    return os.path.join(os.path.dirname(__file__), "logs", f"learned_{family}.json")


def _load(family: str) -> dict | None:
    if family in _MODELS:
        return _MODELS[family]
    try:
        with open(_model_path(family)) as f:
            model = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        model = None
    _MODELS[family] = model
    return model


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def implied_accept_threshold(family: str, round_num: int, target_prob: float = 0.5) -> float | None:
    """Solve the fitted line for the first feature (share_to_us for
    bargaining, margin for negotiation) at which P(agreement) == target_prob,
    holding round fixed at round_num. None if no model is trained yet, or if
    the fit's share/margin coefficient doesn't point the sane direction
    (higher share/margin -> more likely to close) -- a degenerate fit is
    worse than no adjustment at all."""
    model = _load(family)
    if model is None:
        return None

    weights, bias, means, stds = model["weights"], model["bias"], model["means"], model["stds"]
    w_main, w_round = weights
    if w_main <= 0:
        logger.warning(f"{family}: learned model's main coefficient isn't positive ({w_main}) -- ignoring it")
        return None

    round_frac_cap = 15 if family == "bargaining" else 30
    z_round = (min(round_num, round_frac_cap) / round_frac_cap - means[1]) / stds[1]
    z_main = (_logit(target_prob) - bias - w_round * z_round) / w_main
    value = means[0] + stds[0] * z_main

    if family == "bargaining":
        return min(0.95, max(0.05, value))
    return max(-0.9, min(0.9, value))  # margin can legitimately go negative


def predict_prob(family: str, value: float, round_num: int) -> float | None:
    """Forward direction of implied_accept_threshold: P(agreement) at a given
    share_to_us (bargaining) / margin (negotiation) and round, straight from
    the fitted line. None if untrained. Note the fit's own metrics
    (logs/learned_{family}.json) are agreement-skewed -- accuracy barely
    beats always-predict-agreement -- so treat this as one signal to blend,
    not a calibrated probability on its own."""
    model = _load(family)
    if model is None:
        return None

    weights, bias, means, stds = model["weights"], model["bias"], model["means"], model["stds"]
    w_main, w_round = weights

    round_frac_cap = 15 if family == "bargaining" else 30
    z_main = (value - means[0]) / stds[0]
    z_round = (min(round_num, round_frac_cap) / round_frac_cap - means[1]) / stds[1]
    logit = bias + w_main * z_main + w_round * z_round
    return 1 / (1 + math.exp(-logit))


def metrics(family: str) -> dict | None:
    model = _load(family)
    return model["metrics"] if model else None
