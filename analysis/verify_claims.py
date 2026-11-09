"""Recompute every number quoted in competition_paper_neurips2026_revised_v3.tex.

Run from the repo root after analysis/run_api_audit.sh. Prints one line per claim.
"""
import collections
import datetime as dt
import json
import math
import random
import re
import statistics as st

T = dt.datetime.fromisoformat
R = [json.loads(l) for l in open("analysis/out/api_assigned.jsonl")]
W = {
    "W1": (T("2026-08-28T10:58:00+00:00"), T("2026-08-28T15:30:00+00:00")),
    "W3": (T("2026-08-29T12:28:00+00:00"), T("2026-08-29T18:48:00+00:00")),
}
FAMS = ["bargaining", "negotiation", "persuasion"]
VERS = {"W1": ["V0", "V1", "V2", "V3", "V4"], "W3": ["V0", "V1", "V3", "V7", "V9"]}


def win(name):
    a, b = W[name]
    return [r for r in R if a <= T(r["start"]) <= b and r["ver"] in VERS[name]]


def feasible(r):
    c = r["cfg"]
    return c["buyer_value"] > c["seller_value"]


def norm(r):
    c = r["cfg"]; p = r["pay"] or 0
    if r["fam"] == "bargaining":
        return p / c["money_to_divide"]
    if r["fam"] == "negotiation":
        s = (c["buyer_value"] - c["seller_value"]) * c["product_price_order"]
        return p / s if s > 0 else None
    return p / (c["product_price"] * c["total_rounds"])


def succ(r):
    return (r["pay"] or 0) > 0 if r["fam"] == "persuasion" else r["outcome"] == "agreement"


def rows(w, v, fam, feas=True):
    g = [r for r in w if r["ver"] == v and r["fam"] == fam]
    if fam == "negotiation" and feas:
        g = [r for r in g if feasible(r)]
    return g


def ztest(x, y):
    k1, n1, k2, n2 = sum(map(succ, x)), len(x), sum(map(succ, y)), len(y)
    p = (k1 + k2) / (n1 + n2)
    z = (k1 / n1 - k2 / n2) / math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return z, math.erfc(abs(z) / math.sqrt(2))


def welch(x, y):
    a = [norm(r) for r in x]; b = [norm(r) for r in y]
    t = (st.mean(a) - st.mean(b)) / math.sqrt(st.variance(a) / len(a) + st.variance(b) / len(b))
    return st.mean(a) - st.mean(b), math.erfc(abs(t) / math.sqrt(2))


def pct(x):
    return f"{100 * x:.1f}%"


print("== Data totals")
c = collections.Counter(r["acct"] for r in R)
print("records per account", dict(c), "total", sum(c.values()))
hw = collections.Counter(r["how"] for r in R)
amb = collections.Counter(r["ver"] for r in R if r["how"] == "ambiguous")
print("assignment", dict(hw), "ambiguous by pair", dict(amb))
neg = [r for r in R if r["fam"] == "negotiation"]
print("negotiation infeasible share", pct(sum(not feasible(r) for r in neg) / len(neg)))
for fam, key in [("bargaining", "money_to_divide"), ("negotiation", "product_price_order"), ("persuasion", "product_price")]:
    vals = sorted({r["cfg"][key] for r in R if r["fam"] == fam})
    print("scale", fam, key, vals[0], "..", vals[-1])
first_v9 = min(T(r["start"]) for r in R if r["acct"] == "sh_agent1" and r["ver"] == "V9")
print("V8-logged games on sh_agent1 at/after first V9 game:",
      sum(1 for r in R if r["acct"] == "sh_agent1" and r["ver"] == "V8" and T(r["start"]) >= T("2026-08-29T06:27:21+00:00")))

print("\n== Window composition")
for wn in W:
    w = win(wn)
    print(wn, {v: (w2[0]["acct"], len(w2)) for v in VERS[wn] for w2 in [[r for r in w if r["ver"] == v]]})

print("\n== Logger coverage")
for wn in W:
    w = win(wn)
    for v in VERS[wn]:
        g = [r for r in w if r["ver"] == v]
        me = [r for r in g if r["tactor"] == "me"]; op = [r for r in g if r["tactor"] == "opp"]
        ng = [r for r in g if r["fam"] == "negotiation"]; nl = [r for r in ng if r["local"]]
        nf = [r for r in ng if feasible(r)]
        print(wn, v, "n", len(g), "opp-ended", pct(len(op) / len(g)), "opp-ended logged", pct(sum(r["local"] for r in op) / len(op)),
              "own-ended logged", pct(sum(r["local"] for r in me) / len(me)), "logged overall", pct(sum(r["local"] for r in g) / len(g)),
              "| neg agree all: platform", pct(sum(map(succ, ng)) / len(ng)), "local", pct(sum(map(succ, nl)) / len(nl)),
              "| feasible neg opp-ended", pct(sum(r["tactor"] == "opp" for r in nf) / len(nf)))
for v in ["V0", "V9"]:
    g = [r for r in R if r["ver"] == v]
    print("whole run", v, "n", len(g), "logged overall", pct(sum(r["local"] for r in g) / len(g)))

print("\n== Outcomes and tests (vs V0, same window)")
pvals = []
for wn in W:
    w = win(wn)
    for fam in FAMS:
        base = rows(w, "V0", fam)
        for v in VERS[wn]:
            g = rows(w, v, fam)
            cond = [norm(r) for r in g if succ(r) and norm(r) is not None]
            line = f"{wn} {fam:11s} {v} n={len(g)} succ={pct(sum(map(succ, g)) / len(g))} pay={st.mean(norm(r) for r in g):.3f} cond={st.mean(cond):.3f}"
            if v != "V0":
                z, pz = ztest(g, base); d, pt = welch(g, base)
                line += f" | dSucc={100 * (sum(map(succ, g)) / len(g) - sum(map(succ, base)) / len(base)):+.1f}pp z={z:.2f} p={pz:.2g} | dPay={d:+.3f} p={pt:.2g}"
                if wn == "W3":
                    pvals += [pz, pt]
            print(line)
print("W3 tests:", len(pvals), "Bonferroni alpha", 0.05 / len(pvals))

print("\n== Rating delta per game")
for wn in W:
    w = win(wn)
    print(wn, {v: round(sum(r["rd"] or 0 for r in w if r["ver"] == v) / sum(1 for r in w if r["ver"] == v), 3) for v in VERS[wn]})

print("\n== Outcome counts per version (timeouts / invalid_moves_exceeded / games with any invalid move)")
for wn in W:
    w = win(wn)
    for v in VERS[wn]:
        g = [r for r in w if r["ver"] == v]
        oc = collections.Counter(r["outcome"] for r in g)
        print(wn, v, oc["timeout"], oc["invalid_moves_exceeded"], sum(r["inval"] > 0 for r in g))

print("\n== V3 money-losing agreements")
for v in sorted({r["ver"] for r in R}):
    lose = [r for r in R if r["ver"] == v and r["fam"] == "negotiation" and (r["pay"] or 0) < 0]
    if not lose:
        continue
    def own_value(r):
        c = r["cfg"]; seller = (r["me"] == "player_1")  # player_1 is seller in negotiation replays
        return (c["seller_value"] if seller else c["buyer_value"]) * c["product_price_order"]
    by_w = {wn: sum(1 for r in lose if W[wn][0] <= T(r["start"]) <= W[wn][1]) for wn in W}
    print(v, "total", len(lose), "by window", by_w, "decided by", dict(collections.Counter(r["tactor"] for r in lose)),
          "mean loss / own value", f"{st.mean(-r['pay'] / own_value(r) for r in lose):.3f}")

print("\n== Run-log rejections for W3 bargaining games that contain an invalid move")
for v, log in [("V9", "logs/v9_run.log"), ("V0", "logs/v0_run.log")]:
    msgs = collections.defaultdict(collections.Counter)
    for line in open(log, errors="ignore"):
        m = re.search(r"Game ([0-9a-f-]{36}): move rejected as invalid — (.{0,40})", line)
        if m:
            msgs[m.group(1)][m.group(2)] += 1
    g = [r for r in rows(win("W3"), v, "bargaining") if r["inval"] > 0]
    hit = [r for r in g if r["gid"] in msgs]
    kinds = collections.Counter(k for r in hit for k in msgs[r["gid"]])
    print(v, "games", len(g), "with a logged rejection", len(hit), pct(len(hit) / len(g)), "| message types (games):", kinds.most_common(3))

print("\n== Learned negotiation threshold")
import sys
sys.path.insert(0, ".")
import learned_model
import train_model as tm
print("implied 50% margin by round:",
      {r: round(learned_model.implied_accept_threshold("negotiation", r), 3) for r in (1, 4, 8, 9, 10, 12)})
ex = tm._negotiation_examples(tm._load_rows())
cut = int(0.8 * len(ex))
w_, b_, m_, s_ = tm._fit_logistic(ex[:cut])
print("rows", len(ex), "| last 20% in logging order:", tm._evaluate(ex[cut:], w_, b_, m_, s_))
random.seed(0); sh = ex[:]; random.shuffle(sh)
w_, b_, m_, s_ = tm._fit_logistic(sh[:cut])
print("random 20% split:", tm._evaluate(sh[cut:], w_, b_, m_, s_))
