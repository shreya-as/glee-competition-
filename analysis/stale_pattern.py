"""For W3 bargaining games: how often is an invalid move a decision payload that follows
a valid decision in the same game within one second (a re-sent decision)?"""
import datetime as dt
import json
import sys

T = dt.datetime.fromisoformat
a, b = T("2026-08-29T12:28:00+00:00"), T("2026-08-29T18:48:00+00:00")
d = json.load(open(sys.argv[1]))
n = resent = 0
for r in d:
    h = r["history"]
    if h["game_family"] != "bargaining" or not (a <= T(h["started_at"]) <= b):
        continue
    mv = r["replay"]["moves"]
    for i, m in enumerate(mv):
        if m["is_valid"] is not False:
            continue
        n += 1
        prev = mv[i - 1] if i else None
        if (prev and prev["is_valid"] and prev["move_type"] == "decision" and "decision" in m["move_data"]
                and (T(m["created_at"]) - T(prev["created_at"])).total_seconds() < 1.0):
            resent += 1
print(d[0]["history"]["your_player_name"], "invalid moves", n, "re-sent decision <1s after a valid decision", resent, f"{100 * resent / n:.1f}%")
