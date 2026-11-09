import json,sys,collections
from datetime import datetime
src=sys.argv[1]; out=sys.argv[2]
d=json.load(open(src))
with open(out,"w") as w:
    for r in d:
        h=r["history"]; rp=r.get("replay") or {}
        me=h["your_player"]; res=h.get("result") or {}
        moves=rp.get("moves") or []
        st=(rp.get("state") or {})
        sh=st.get("history") or []
        # who made final decision/ move
        last_mover=None
        if moves: last_mover=moves[-1].get("player") or moves[-1].get("player_id") or moves[-1].get("by")
        last_decider=sh[-1].get("decided_by") if sh and isinstance(sh[-1],dict) else None
        other={"player_1":"player_2","player_2":"player_1"}[me]
        term=None
        if sh and isinstance(sh[-1],dict):
            x=sh[-1]
            if h["game_family"]=="negotiation": term=x.get("decided_by")
            elif h["game_family"]=="bargaining":
                p=x.get("proposer") or (x.get("offer") or {}).get("proposer")
                term=other.get(p) if False else ({"player_1":"player_2","player_2":"player_1"}.get(p) if x.get("decision") else p)
            elif h["game_family"]=="persuasion": term="player_2"
        if res.get("walked_away_by"): term=res["walked_away_by"]
        if res.get("outcome") in ("timeout","invalid_moves_exceeded"): term="system"
        tactor=None if term is None else ("me" if term==me else ("system" if term=="system" else "opp"))
        inval=sum(1 for m in moves if m.get("is_valid") is False)
        # Invalid moves that are a decision payload sent when an offer was due (or the reverse),
        # and the subset sent less than 1 s after a valid decision in the same game (a re-send).
        inv_stale=inv_resent=0
        for i,m in enumerate(moves):
            if m.get("is_valid") is not False: continue
            md=m.get("move_data") or {}
            if (m.get("move_type")=="offer" and "decision" in md) or (m.get("move_type")=="decision" and "decision" not in md):
                inv_stale+=1
            prev=moves[i-1] if i else None
            if (prev and prev.get("is_valid") and prev.get("move_type")=="decision" and "decision" in md
                    and (datetime.fromisoformat(m["created_at"])-datetime.fromisoformat(prev["created_at"])).total_seconds()<1.0):
                inv_resent+=1
        # Our own accept/reject decisions in negotiation (from the replay's decision history).
        dec=collections.Counter(x.get("decision") for x in sh if isinstance(x,dict) and x.get("decided_by")==me) if h["game_family"]=="negotiation" else collections.Counter()
        rec=dict(acct=h.get("your_player_name"),gid=h["game_id"],fam=h["game_family"],status=h.get("status"),
          me=me,outcome=res.get("outcome"),pay=res.get(f"{me}_payoff"),opay=res.get("player_1_payoff" if me=="player_2" else "player_2_payoff"),
          start=h.get("started_at"),end=h.get("completed_at"),rd=h.get("rating_delta"),
          nmoves=len(moves),inval=inval,last_decider=last_decider,tactor=tactor,ninfo=(rp.get('state') or {}).get('complete_information'),
          move_keys=list(moves[-1].keys()) if moves else None,
          walked_by=res.get("walked_away_by"),opp_type=h.get("opponent_type"),cfg=rp.get("config"),opp=h.get("opponent_name") or h.get("opponent_id"),
          hkeys=list(h.keys()),inv_stale=inv_stale,inv_resent=inv_resent,
          dec_accept=dec["AcceptOffer"],dec_reject=dec["RejectOffer"],dec_other=sum(dec.values())-dec["AcceptOffer"]-dec["RejectOffer"])
        w.write(json.dumps(rec)+"\n")
print(src,len(d))
