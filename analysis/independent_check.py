"""Recompute the paper's headline numbers straight from the platform downloads.

This script reads only final_logs/*.json. It does not use the local logs or the
version-assignment step (analysis/assign.py): inside W1 and W3 each account ran a
single version, so filtering by account and start time is enough. Counts in W1 can
differ by a few games from Figure 4 because three accounts switched versions in the
last seconds of W1, and those games are dropped by the full pipeline.

    python3 analysis/independent_check.py
"""
import datetime as dt
import json
import statistics as st

T = dt.datetime.fromisoformat
FILES = {
    "sh_agent1": "final_logs/sh_agent1_all_games_with_replays.json",
    "shreyaAgent2": "final_logs/shreyaAgent2_all_games_with_replays (3).json",
    "Agent_3": "final_logs/Agent_3_all_games_with_replays.json",
    "shreya_agent_4": "final_logs/shreya_agent_4_all_games_with_replays.json",
    "agent_5": "final_logs/agent_5_all_games_with_replays.json",
}
W = {
    "W1": (T("2026-08-28T10:58:00+00:00"), T("2026-08-28T15:30:00+00:00")),
    "W3": (T("2026-08-29T12:28:00+00:00"), T("2026-08-29T18:48:00+00:00")),
}
VERSION = {
    "W1": {"sh_agent1": "V0", "shreyaAgent2": "V1", "Agent_3": "V2", "shreya_agent_4": "V3", "agent_5": "V4"},
    "W3": {"shreya_agent_4": "V0", "shreyaAgent2": "V1", "Agent_3": "V3", "agent_5": "V7", "sh_agent1": "V9"},
}
# (window, version, family) -> (N, success %, normalized payoff) as printed in Figures 3 and 4
PAPER = {
    ("W3", "V0", "bargaining"): (1069, 92.5, 0.441), ("W3", "V7", "bargaining"): (1042, 95.1, 0.442),
    ("W3", "V9", "bargaining"): (1287, 96.3, 0.449), ("W3", "V0", "negotiation"): (572, 81.1, 0.350),
    ("W3", "V7", "negotiation"): (580, 72.2, 0.366), ("W3", "V9", "negotiation"): (736, 90.1, 0.406),
    ("W3", "V0", "persuasion"): (1107, 53.7, 0.430), ("W3", "V7", "persuasion"): (1021, 45.7, 0.400),
    ("W3", "V9", "persuasion"): (1245, 52.9, 0.438),
    ("W1", "V0", "negotiation"): (406, 86.0, 0.349), ("W1", "V1", "negotiation"): (409, 81.7, 0.363),
    ("W1", "V2", "negotiation"): (398, 84.9, 0.354), ("W1", "V3", "negotiation"): (403, 85.4, 0.348),
    ("W1", "V4", "negotiation"): (410, 82.9, 0.338),
}


def outcome(h, rp):
    """Return (success, normalized payoff) for one game, or None for infeasible negotiation."""
    fam, me, res, cfg = h["game_family"], h["your_player"], h["result"], rp["config"]
    pay = res.get(f"{me}_payoff") or 0
    if fam == "bargaining":
        return res.get("outcome") == "agreement", pay / cfg["money_to_divide"]
    if fam == "negotiation":
        surplus = (cfg["buyer_value"] - cfg["seller_value"]) * cfg["product_price_order"]
        if surplus <= 0:
            return None
        return res.get("outcome") == "agreement", pay / surplus
    return pay > 0, pay / (cfg["product_price"] * cfg["total_rounds"])


total = 0
found = {}
spot = {"V3 money-losing deal": [], "V9 deal closed by the opponent's acceptance": []}
for acct, path in FILES.items():
    games = json.load(open(path))
    total += len(games)
    print(f"{acct}: {len(games)} games in {path}")
    for g in games:
        h, rp = g["history"], g.get("replay") or {}
        t = T(h["started_at"])
        for wn, (a, b) in W.items():
            if not (a <= t <= b):
                continue
            v = VERSION[wn][acct]
            o = outcome(h, rp)
            if o is not None:
                found.setdefault((wn, v, h["game_family"]), []).append(o)
            me = h["your_player"]
            if h["game_family"] == "negotiation" and v == "V3" and (h["result"].get(f"{me}_payoff") or 0) < 0:
                spot["V3 money-losing deal"].append((h["game_id"], wn, me, h["result"]))
            last = (rp.get("state") or {}).get("history") or []
            if (h["game_family"] == "negotiation" and v == "V9" and h["result"].get("outcome") == "agreement"
                    and last and last[-1].get("decided_by") not in (None, me)):
                spot["V9 deal closed by the opponent's acceptance"].append((h["game_id"], wn, me, h["result"]))
    del games

print(f"\nTotal games: {total} (paper: 75,258)\n")
print(f"{'window version family':32s} {'N':>11s} {'success %':>15s} {'norm. payoff':>17s}")
for key, (pn, ps, pp) in PAPER.items():
    rows = found.get(key, [])
    n = len(rows)
    s = 100 * sum(x for x, _ in rows) / n
    p = st.mean(y for _, y in rows)
    print(f"{' '.join(key):32s} {n:5d} ({pn:4d}) {s:7.1f} ({ps:5.1f}) {p:8.3f} ({pp:.3f})")
print("(paper values in brackets)\n")

print("Game IDs you can open on the GLEE platform to spot-check by hand:")
for what, items in spot.items():
    print(f"  {what} ({len(items)} found):")
    for gid, wn, me, res in items[:3]:
        print(f"    {gid}  {wn}  we were {me}, our payoff {res.get(me + '_payoff')}, price {res.get('agreed_price')}")
