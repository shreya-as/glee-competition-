import json,collections,math,random,datetime as dt
R=[json.loads(l) for l in open("analysis/out/api_assigned.jsonl")]
def T(s): return dt.datetime.fromisoformat(s)
W={"W1":(T("2026-08-28T10:58:00+00:00"),T("2026-08-28T15:30:00+00:00")),
   "W3":(T("2026-08-29T12:28:00+00:00"),T("2026-08-29T18:48:00+00:00"))}
def norm(r):
    c=r["cfg"] or {}; p=r["pay"] or 0
    if r["fam"]=="bargaining": return p/c["money_to_divide"]
    if r["fam"]=="negotiation":
        s=(c["buyer_value"]-c["seller_value"])*c["product_price_order"]; return p/s if s>0 else None
    if r["fam"]=="persuasion": return p/(c["product_price"]*c["total_rounds"])
def succ(r):
    if r["fam"]=="persuasion": return (r["pay"] or 0)>0
    return r["outcome"]=="agreement"
def wil(k,n,z=1.96):
    if n==0: return (0,0)
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return (c-h,c+h)
def boot(xs,B=2000):
    random.seed(0); m=[]
    for _ in range(B):
        s=[random.choice(xs) for _ in xs]; m.append(sum(s)/len(s))
    m.sort(); return m[int(.025*B)],m[int(.975*B)]
out={}
for wn,(a,b) in W.items():
    print("=====",wn)
    rows=[r for r in R if a<=T(r["start"])<=b and r["how"]!="ambiguous" and r["ver"]!="gap"]
    vers=sorted({r["ver"] for r in rows})
    for fam in ["bargaining","negotiation","persuasion"]:
        print("--",fam)
        print(f"{'ver':4}{'acct':15}{'N':>6}{'succ%':>7}{'CI':>14}{'normpay':>9}{'normpayCI':>16}{'condnorm':>9}{'rd_sum':>8}{'rd/g':>7}{'opp_end%':>9}{'sys%':>6}{'cov%':>6}{'loc_succ%':>10}{'locN':>6}{'apiSucc|opp':>12}{'inv/g':>7}")
        for v in vers:
            g=[r for r in rows if r["ver"]==v and r["fam"]==fam]
            if fam=="negotiation":
                allg=g; g=[r for r in g if r["cfg"]["buyer_value"]>r["cfg"]["seller_value"]]
                inf=[r for r in allg if r not in g]
                neg=sum((r["pay"] or 0)<0 for r in allg); infagree=sum(r["outcome"]=="agreement" for r in inf)
                if len(allg)>20: print(f"   [{v}] all={len(allg)} feasible={len(g)} allAgree%={100*sum(succ(r) for r in allg)/len(allg):.1f} infeasibleAgree={infagree}/{len(inf)} negPay={neg}")
            if len(g)<20: continue
            n=len(g); k=sum(succ(r) for r in g); ci=wil(k,n)
            nz=[norm(r) for r in g if norm(r) is not None]; mp=sum(nz)/len(nz); bci=boot(nz)
            cz=[norm(r) for r in g if succ(r) and norm(r) is not None]; cm=sum(cz)/len(cz) if cz else 0
            rd=sum(r["rd"] or 0 for r in g)
            oe=sum(r["tactor"]=="opp" for r in g)/n; se=sum(r["tactor"]=="system" for r in g)/n
            loc=[r for r in g if r["local"]]; cov=len(loc)/n
            ls=sum(succ(r) for r in loc)/len(loc) if loc else float('nan')
            go=[r for r in g if r["tactor"]=="opp"]; so=sum(succ(r) for r in go)/len(go) if go else float('nan')
            inv=sum(r["inval"] for r in g)/n
            print(f"{v:4}{g[0]['acct']:15}{n:6}{100*k/n:7.1f} [{100*ci[0]:4.1f},{100*ci[1]:4.1f}]{mp:9.3f} [{bci[0]:.3f},{bci[1]:.3f}]{cm:9.3f}{rd:8.0f}{rd/n:7.2f}{100*oe:9.1f}{100*se:6.1f}{100*cov:6.1f}{100*ls:10.1f}{len(loc):6}{100*so:12.1f}{inv:7.2f}")
            out[f"{wn}|{v}|{fam}"]=dict(n=n,succ=k/n,ci=ci,normpay=mp,normci=bci,cond=cm,rd=rd,oppend=oe,sysend=se,cov=cov,locsucc=ls,locn=len(loc),acct=g[0]["acct"])
    # overall rating per version
    print("-- rating total & outcomes")
    for v in vers:
        g=[r for r in rows if r["ver"]==v]
        oc=collections.Counter(r["outcome"] for r in g)
        print(v,g[0]["acct"],len(g),"rd_sum=%.0f"%sum(r["rd"] or 0 for r in g),"rd/g=%.3f"%(sum(r["rd"] or 0 for r in g)/len(g)),"timeouts",oc["timeout"],"invalid_exceeded",oc["invalid_moves_exceeded"],"inval_moves",sum(r["inval"] for r in g),"games_w_inval",sum(r["inval"]>0 for r in g))
json.dump(out,open("analysis/out/metrics.json","w"),indent=1)
