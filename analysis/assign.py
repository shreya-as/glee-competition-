import json,collections,datetime as dt,sys
L="logs/"
def ts(s): return dt.datetime.fromisoformat(s.replace("+0545","+05:45")).astimezone(dt.timezone.utc)
# A version's run on an account is split into separate segments wherever its local log
# has a gap longer than GAP (e.g. V1 on shreyaAgent2 ran on 28 Aug and again on 29 Aug,
# with V5 in between). Unlogged platform games are assigned to the version whose segment
# contains their start time.
GAP=dt.timedelta(minutes=60)
gid2v={}; stamps=collections.defaultdict(list); local={}
for v in range(10):
    for line in open(f"{L}v{v}_games.jsonl"):
        d=json.loads(line); a=d["agent_name"]; t=ts(d["timestamp"]); V=f"V{v}"
        stamps[(a,V)].append(t)
        g=d.get("game_id")
        if g: gid2v.setdefault(g,set()).add((a,V)); local[(a,g)]=d
win=collections.defaultdict(lambda: collections.defaultdict(list))
for (a,V),ts_list in stamps.items():
    ts_list.sort(); s=e=ts_list[0]
    for t in ts_list[1:]:
        if t-e>GAP: win[a][V].append((s,e)); s=t
        e=t
    win[a][V].append((s,e))
for a in win:
    for V,segs in win[a].items():
        for s,e in segs: print(a,V,s.isoformat()[:19],e.isoformat()[:19])
out=open("analysis/out/api_assigned.jsonl","w")
cnt=collections.Counter()
for k in "12345":
    for line in open(f"analysis/out/a{k}.jsonl"):
        r=json.loads(line); a=r["acct"]; t=ts(r["start"])
        hits=[V for (aa,V) in gid2v.get(r["gid"],()) if aa==a]
        if len(hits)==1: r["ver"]=hits[0]; r["how"]="logged"
        else:
            cands=sorted({V for V,segs in win[a].items() for (s,e) in segs if s-dt.timedelta(minutes=2)<=t<=e+dt.timedelta(minutes=1)})
            if len(cands)==1: r["ver"]=cands[0]; r["how"]="time"
            elif cands: r["ver"]="/".join(sorted(cands)); r["how"]="ambiguous"
            else: r["ver"]="gap"; r["how"]="gap"
        r["local"]=(a,r["gid"]) in local
        if r["local"]:
            d=local[(a,r["gid"])]; r["l_out"]=d.get("outcome"); r["l_pay"]=d.get("payoff")
        cnt[(a,r["ver"],r["how"])]+=1
        out.write(json.dumps(r)+"\n")
for k,v in sorted(cnt.items()): print(k,v)
# local rows not in API
api={(json.loads(l)["acct"],json.loads(l)["gid"]) for k in "12345" for l in open(f"analysis/out/a{k}.jsonl")}
lo=collections.Counter((a,list(v for aa,v in gid2v[g] if aa==a)[0]) for (a,g) in local if (a,g) not in api)
print("local-only",lo)
print("local rows w/o gid", sum(1 for v in range(10) for l in open(f"{L}v{v}_games.jsonl") if not json.loads(l).get("game_id")))
