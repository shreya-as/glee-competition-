"""Offline trainer for learned_model.py: fits a small logistic regression
per family (bargaining, negotiation) on the accumulated logs/games.jsonl,
predicting P(agreement) from the terms of the offer on the table when the
game ended and the round it ended in. This is the actual data-trained model
the fleet runs -- zero LLM calls, online or offline, distinct from
opponent_model.py's per-opponent online bandit (which stays as-is: it
answers "how does this specific opponent behave", a different question from
"empirically, what offer/round combination tends to produce a deal at all").

What it can and can't tell you: games.jsonl only has the FINAL decision
point of each game (the SDK never surfaces intermediate per-round outcomes
back to the strategy function -- see game_logger.py's docstring), so this
fits the empirical relationship between "how good was the offer on the
table" and "did a deal happen", pooled across every agent variant and
opponent that's played so far. It is NOT an off-policy estimate of "would
holding out longer have paid off more" -- that would need per-round outcome
logging this project doesn't have. Treat its output as a data-calibrated
accept threshold, not a full renegotiation policy.

Negotiation needs my_value/role_type, added to game_logger.py's logged row
after this trainer was written -- older rows lack them and are skipped, so
the negotiation model starts thin and fills in as the fleet keeps playing.
Bargaining's pool size is derivable from any row (sum of both gains in
opponent_last_offer), so it trains on the full history immediately.

Usage:
    python train_model.py
Writes logs/learned_bargaining.json and logs/learned_negotiation.json.
Safe to re-run anytime -- refits from whatever is in games.jsonl right now.
"""

import json
import logging
import math
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger("train_model")

_LOG_PATH = os.path.join(os.path.dirname(__file__), "logs", "games.jsonl")
_MIN_ROWS = 40  # below this, a fit is more noise than signal -- skip rather than ship garbage
_L2 = 0.02
_LR = 0.3
_EPOCHS = 800


def _load_rows():
    rows = []
    with open(_LOG_PATH) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _bargaining_examples(rows):
    examples = []
    for r in rows:
        if r.get("game_family") != "bargaining" or r.get("outcome") not in ("agreement", "no_deal"):
            continue
        offer = r.get("opponent_last_offer") or {}
        role = r.get("role")
        if not role or offer.get("player_1_gain") is None or offer.get("player_2_gain") is None:
            continue
        my_gain = offer.get(f"{role}_gain")
        pool = r.get("pool_size") or (offer["player_1_gain"] + offer["player_2_gain"])
        if my_gain is None or not pool:
            continue
        share = my_gain / pool
        round_num = r.get("round") or 1
        label = 1.0 if r["outcome"] == "agreement" else 0.0
        examples.append(([share, min(round_num, 15) / 15], label))
    return examples


def _negotiation_examples(rows):
    examples = []
    for r in rows:
        if r.get("game_family") != "negotiation" or r.get("outcome") not in ("agreement", "no_deal"):
            continue
        offer = r.get("opponent_last_offer") or {}
        my_value = r.get("my_value")
        role_type = r.get("role_type")
        price = offer.get("price")
        if my_value in (None, 0) or role_type not in ("buyer", "seller") or price is None:
            continue
        margin = (my_value - price) / my_value if role_type == "buyer" else (price - my_value) / my_value
        round_num = r.get("round") or 1
        label = 1.0 if r["outcome"] == "agreement" else 0.0
        examples.append(([margin, min(round_num, 30) / 30], label))
    return examples


FEATURE_NAMES = {
    "bargaining": ["share_to_us", "round_frac"],
    "negotiation": ["margin", "round_frac"],
}


def _standardize(examples):
    n = len(examples)
    dims = len(examples[0][0])
    means = [sum(x[i] for x, _ in examples) / n for i in range(dims)]
    stds = []
    for i in range(dims):
        var = sum((x[i] - means[i]) ** 2 for x, _ in examples) / n
        stds.append(math.sqrt(var) or 1.0)
    standardized = [([(x[i] - means[i]) / stds[i] for i in range(dims)], y) for x, y in examples]
    return standardized, means, stds


def _sigmoid(z):
    if z < -30:
        return 0.0
    if z > 30:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


def _fit_logistic(examples):
    """Batch gradient descent, L2-regularized. examples: [(features, label)]."""
    standardized, means, stds = _standardize(examples)
    dims = len(standardized[0][0])
    weights = [0.0] * dims
    bias = 0.0
    n = len(standardized)

    for _ in range(_EPOCHS):
        grad_w = [0.0] * dims
        grad_b = 0.0
        for x, y in standardized:
            pred = _sigmoid(bias + sum(w * xi for w, xi in zip(weights, x)))
            err = pred - y
            for i in range(dims):
                grad_w[i] += err * x[i]
            grad_b += err
        for i in range(dims):
            weights[i] -= _LR * (grad_w[i] / n + _L2 * weights[i])
        bias -= _LR * (grad_b / n)

    return weights, bias, means, stds


def _evaluate(examples, weights, bias, means, stds):
    correct = 0
    total_loss = 0.0
    positives = sum(1 for _, y in examples if y == 1.0)
    for x_raw, y in examples:
        z = [(x_raw[i] - means[i]) / stds[i] for i in range(len(x_raw))]
        pred = _sigmoid(bias + sum(w * zi for w, zi in zip(weights, z)))
        correct += (pred >= 0.5) == (y == 1.0)
        pred_clamped = min(max(pred, 1e-9), 1 - 1e-9)
        total_loss += -(y * math.log(pred_clamped) + (1 - y) * math.log(1 - pred_clamped))
    n = len(examples)
    baseline = max(positives, n - positives) / n
    return {
        "accuracy": round(correct / n, 3),
        "baseline_accuracy": round(baseline, 3),
        "log_loss": round(total_loss / n, 4),
        "n": n,
        "positive_rate": round(positives / n, 3),
    }


def train_family(family, examples):
    if len(examples) < _MIN_ROWS:
        logger.warning(f"{family}: only {len(examples)} usable rows (need {_MIN_ROWS}+) -- skipping, no model written")
        return None

    weights, bias, means, stds = _fit_logistic(examples)
    metrics = _evaluate(examples, weights, bias, means, stds)

    if metrics["accuracy"] < metrics["baseline_accuracy"] + 0.02:
        logger.warning(
            f"{family}: fit barely beats the majority-class baseline "
            f"({metrics['accuracy']} vs {metrics['baseline_accuracy']}) -- "
            f"writing it anyway but learned_model.py's blend weight should stay small"
        )

    model = {
        "family": family,
        "feature_names": FEATURE_NAMES[family],
        "weights": weights,
        "bias": bias,
        "means": means,
        "stds": stds,
        "metrics": metrics,
    }
    path = os.path.join(os.path.dirname(__file__), "logs", f"learned_{family}.json")
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(model, f, indent=2)
    os.replace(tmp, path)
    logger.info(f"{family}: trained on {metrics['n']} rows, accuracy={metrics['accuracy']} (baseline {metrics['baseline_accuracy']}), log_loss={metrics['log_loss']} -> {path}")
    return model


if __name__ == "__main__":
    rows = _load_rows()
    logger.info(f"Loaded {len(rows)} rows from {_LOG_PATH}")
    train_family("bargaining", _bargaining_examples(rows))
    train_family("negotiation", _negotiation_examples(rows))
