import json,sys,datetime as dt,collections
T=dt.datetime.fromisoformat
a,b=T("2026-08-29T12:28:00+00:00"),T("2026-08-29T18:48:00+00:00")
d=json.load(open(sys.argv[1]))
c=collections.Counter()
for r in d:
    h=r["history"]
    if not (a<=T(h["started_at"])<=b): continue
    mv=r["replay"]["moves"]; inv=[m for m in mv if m.get("is_valid") is False]
    f=h["game_family"]; c[(f,"games")]+=1
    if inv:
        c[(f,"games_inv")]+=1
        c[(f,"inv_moves")]+=len(inv)
        c[(f,"inv_fast")]+=sum(m["response_time_ms"]<1000 for m in inv)
        c[(f,"inv_stale")]+=sum((m["move_type"]=="offer" and "decision" in m["move_data"]) or (m["move_type"]=="decision" and "decision" not in m["move_data"]) for m in inv)
        c[(f,"inv_agree")]+=h["result"]["outcome"]=="agreement"
    else: c[(f,"noinv_agree")]+=h["result"]["outcome"]=="agreement"
print(h["your_player_name"],{k:v for k,v in sorted(c.items()) if k[0]!="persuasion"})
