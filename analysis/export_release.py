"""Build the public paper_data/ folder that analysis/check_paper.py reads.

Run from the repo root after the earlier steps of analysis/run_api_audit.sh. Inputs are
the private raw files (final_logs/ via analysis/out/, and logs/); outputs contain no
message text from any agent. Files are written deterministically (sorted where order
does not matter, gzip with a fixed timestamp) so re-running gives identical bytes.

    python3 analysis/export_release.py
"""
import collections
import csv
import gzip
import io
import json
import os
import re
import shutil
import sys

sys.path.insert(0, ".")
import train_model as tm  # noqa: E402

OUT = "paper_data"
os.makedirs(OUT, exist_ok=True)


def gz_writer(path):
    raw = open(path, "wb")
    gz = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)
    return raw, gz, io.TextIOWrapper(gz, encoding="utf-8", newline="")


def cell(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    return repr(v) if isinstance(v, float) else str(v)


# ---------------------------------------------------------------- platform_games.csv.gz
GAME_COLS = [
    ("game_id", "gid"), ("account", "acct"), ("version", "ver"), ("assigned_by", "how"), ("family", "fam"),
    ("started_at", "start"), ("completed_at", "end"), ("our_player", "me"), ("outcome", "outcome"),
    ("our_payoff", "pay"), ("opponent_payoff", "opay"), ("rating_delta", "rd"),
    ("final_decision_by", "tactor"), ("invalid_moves", "inval"),
    ("invalid_decision_when_offer_due", "inv_stale"), ("invalid_resent_within_1s", "inv_resent"),
    ("our_accepts", "dec_accept"), ("our_rejects", "dec_reject"), ("our_other_decisions", "dec_other"),
    ("in_local_log", "local"), ("opponent_type", "opp_type"),
]
CFG_ORDER = ["money_to_divide", "delta_1", "delta_2", "seller_value", "buyer_value", "product_price_order",
             "product_price", "total_rounds", "p", "v", "c", "max_rounds", "complete_information",
             "messages_allowed", "is_myopic", "is_seller_know_cv", "seller_message_type"]
rows = [json.loads(line) for line in open("analysis/out/api_assigned.jsonl")]
cfg_keys = sorted({k for r in rows for k in (r.get("cfg") or {})}, key=lambda k: (CFG_ORDER.index(k) if k in CFG_ORDER else 99, k))
raw, gz, w = gz_writer(f"{OUT}/platform_games.csv.gz")
cw = csv.writer(w)
cw.writerow([c for c, _ in GAME_COLS] + cfg_keys)
for r in rows:
    cfg = r.get("cfg") or {}
    cw.writerow([cell(r.get(k)) for _, k in GAME_COLS] + [cell(cfg.get(k)) for k in cfg_keys])
w.close(); raw.close()
print(f"platform_games.csv.gz: {len(rows)} games, {len(GAME_COLS) + len(cfg_keys)} columns")

# ---------------------------------------------------------------- local_logger_games.csv.gz
raw, gz, w = gz_writer(f"{OUT}/local_logger_games.csv.gz")
cw = csv.writer(w)
cw.writerow(["version", "agent_name", "timestamp", "game_id", "family", "outcome"])
n_local = 0
for v in range(10):
    for line in open(f"logs/v{v}_games.jsonl"):
        d = json.loads(line)
        cw.writerow([f"V{v}", d.get("agent_name"), d.get("timestamp"), d.get("game_id"), d.get("game_family"), d.get("outcome")])
        n_local += 1
w.close(); raw.close()
print(f"local_logger_games.csv.gz: {n_local} rows from logs/v0..v9_games.jsonl")

# ---------------------------------------------------------------- run_log_rejections.csv
pat = re.compile(r"Game ([0-9a-f-]{36}): move rejected as invalid — ([\"'])(.*?)\2")
counts, unparsed = collections.Counter(), 0
for log in ("v0_run.log", "v9_run.log"):
    for line in open(f"logs/{log}", errors="ignore"):
        if "move rejected as invalid" not in line:
            continue
        m = pat.search(line)
        if m:
            counts[(log, m.group(1), m.group(3))] += 1
        else:
            unparsed += 1
with open(f"{OUT}/run_log_rejections.csv", "w", newline="", encoding="utf-8") as f:
    cw = csv.writer(f)
    cw.writerow(["run_log", "game_id", "server_message", "times"])
    for (log, gid, msg), n in sorted(counts.items()):
        cw.writerow([log, gid, msg, n])
print(f"run_log_rejections.csv: {len(counts)} (log, game, message) rows; unparsed lines: {unparsed}")

# ---------------------------------------------------------------- learned model
examples = tm._negotiation_examples(tm._load_rows())
with open(f"{OUT}/learned_model_training.csv", "w", newline="", encoding="utf-8") as f:
    cw = csv.writer(f)
    cw.writerow(["margin", "round_frac", "agreement"])
    for (margin, round_frac), label in examples:
        cw.writerow([repr(margin), repr(round_frac), int(label)])
shutil.copyfile("logs/learned_negotiation.json", f"{OUT}/learned_negotiation.json")
print(f"learned_model_training.csv: {len(examples)} rows (logging order); learned_negotiation.json copied")
