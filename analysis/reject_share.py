"""Share of our negotiation decisions that were RejectOffer (feasible games, W3)."""
import collections
import datetime as dt
import json
import sys

T = dt.datetime.fromisoformat
a, b = T("2026-08-29T12:28:00+00:00"), T("2026-08-29T18:48:00+00:00")
d = json.load(open(sys.argv[1]))
c = collections.Counter()
for r in d:
    h = r["history"]
    if h["game_family"] != "negotiation" or not (a <= T(h["started_at"]) <= b):
        continue
    cfg = r["replay"]["config"]
    if cfg["buyer_value"] <= cfg["seller_value"]:
        continue
    for x in r["replay"]["state"].get("history", []):
        if x.get("decided_by") == h["your_player"]:
            c[x.get("decision")] += 1
print(h["your_player_name"], dict(c), f"reject share {100 * c['RejectOffer'] / sum(c.values()):.1f}%")
