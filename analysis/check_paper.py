"""Check every data claim in the revised paper (default: competition_paper_neurips2026_revised_v3.tex).

Reads only the paper, the agent source files, and the public paper_data/ folder, so it runs
on a fresh clone of the repository. From the repo root:

    python3 analysis/check_paper.py                      # checks the v3 paper
    python3 analysis/check_paper.py competition_paper_neurips2026_revised_v1.tex

(analysis/run_api_audit.sh rebuilds paper_data/ from the raw platform downloads and then runs
this script; only the account owner can export those downloads.) Each line prints PASS
or FAIL, the value as written in the paper, and the value recomputed from the data. A
number passes when the recomputed value rounds to the paper's printed precision. The
script exits with status 1 if any claim fails.
"""
import collections
import csv
import datetime as dt
import gzip
import math
import random
import re
import statistics as st
import sys

sys.path.insert(0, ".")

T = dt.datetime.fromisoformat
PAPER = sys.argv[1] if len(sys.argv) > 1 else "competition_paper_neurips2026_revised_v3.tex"


def parse(v):
    """CSV cell -> Python value (empty -> None, true/false -> bool, numbers -> int/float)."""
    if v == "":
        return None
    if v in ("true", "false"):
        return v == "true"
    for kind in (int, float):
        try:
            return kind(v)
        except ValueError:
            pass
    return v


COLS = {"game_id": "gid", "account": "acct", "version": "ver", "assigned_by": "how", "family": "fam",
        "started_at": "start", "completed_at": "end", "our_player": "me", "outcome": "outcome",
        "our_payoff": "pay", "opponent_payoff": "opay", "rating_delta": "rd", "final_decision_by": "tactor",
        "invalid_moves": "inval", "invalid_decision_when_offer_due": "inv_stale",
        "invalid_resent_within_1s": "inv_resent", "our_accepts": "dec_accept", "our_rejects": "dec_reject",
        "our_other_decisions": "dec_other", "in_local_log": "local",
        "opponent_type": "opp_type"}


def read_gz_csv(path):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


R = []
for row in read_gz_csv("paper_data/platform_games.csv.gz"):
    r = {COLS[k]: parse(v) for k, v in row.items() if k in COLS}
    r["cfg"] = {k: parse(v) for k, v in row.items() if k not in COLS and v != ""}
    R.append(r)
LOCAL = read_gz_csv("paper_data/local_logger_games.csv.gz")
W = {
    "W1": (T("2026-08-28T10:58:00+00:00"), T("2026-08-28T15:30:00+00:00")),
    "W3": (T("2026-08-29T12:28:00+00:00"), T("2026-08-29T18:48:00+00:00")),
}
VERS = {"W1": ["V0", "V1", "V2", "V3", "V4"], "W3": ["V0", "V1", "V3", "V7", "V9"]}
ACCOUNT_NO = {"sh_agent1": 1, "shreyaAgent2": 2, "Agent_3": 3, "shreya_agent_4": 4, "agent_5": 5}

failures = 0


def report(section, what, paper, data, ok):
    global failures
    failures += not ok
    print(f"{'PASS' if ok else 'FAIL'}  [{section}] {what}: paper {paper} | data {data}")


def rounds_to(value, paper):
    """True if value, rounded to the decimals printed in `paper`, equals it."""
    s = str(paper).replace(",", "").replace("+", "").replace("−", "-")
    decimals = len(s.split(".")[1]) if "." in s else 0
    return abs(round(value, decimals) - float(s)) < 1e-9


# The paper text with LaTeX number formatting removed, used to confirm that each value
# this script expects is really printed in the paper (catches edits to the paper that
# were not mirrored here). Values shorter than three digits are too common to search for.
TEX = re.sub(r"\{,\}|\$|\\%|\\approx", "", open(PAPER).read()).replace("--", "-")


def in_paper(text):
    s = str(text)
    return len(re.sub(r"[^0-9]", "", s)) < 3 or s in TEX or s.lstrip("+-") in TEX


def num(section, what, paper, value, scale=1.0):
    v = value * scale
    shown = in_paper(paper)
    report(section, what + ("" if shown else " (value not found in paper text)"), paper, f"{v:.6g}",
           rounds_to(v, paper) and shown)


def opt_num(section, what, paper, value, scale=1.0):
    """For a sentence that only some paper versions contain: SKIP when its value is absent."""
    if in_paper(paper):
        num(section, what, paper, value, scale)
    else:
        print(f"SKIP  [{section}] {what}: not in this version of the paper")


def span(section, what, paper, values, scale=100.0):
    """Paper writes a range 'lo-hi'; check min and max round to it."""
    lo, hi = paper.split("-")
    vs = [v * scale for v in values]
    shown = in_paper(paper)
    report(section, what + ("" if shown else " (value not found in paper text)"), paper,
           f"{min(vs):.2f}-{max(vs):.2f}", rounds_to(min(vs), lo) and rounds_to(max(vs), hi) and shown)


def cond(section, what, paper, data, ok):
    report(section, what, paper, data, bool(ok))


# ---------------------------------------------------------------- helpers
def win(name):
    a, b = W[name]
    return [r for r in R if a <= T(r["start"]) <= b and r["ver"] in VERS[name]]


def feasible(r):
    return r["cfg"]["buyer_value"] > r["cfg"]["seller_value"]


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


def sel(w, v, fam, feas=True):
    g = [r for r in w if r["ver"] == v and r["fam"] == fam]
    return [r for r in g if feasible(r)] if fam == "negotiation" and feas else g


def rate(g):
    return sum(map(succ, g)) / len(g)


def mean_pay(g):
    return st.mean(norm(r) for r in g)


def cond_pay(g):
    return st.mean(norm(r) for r in g if succ(r))


def ztest(x, y):
    k1, n1, k2, n2 = sum(map(succ, x)), len(x), sum(map(succ, y)), len(y)
    p = (k1 + k2) / (n1 + n2)
    z = (k1 / n1 - k2 / n2) / math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return z, math.erfc(abs(z) / math.sqrt(2))


def welch(x, y):
    a = [norm(r) for r in x]; b = [norm(r) for r in y]
    t = (st.mean(a) - st.mean(b)) / math.sqrt(st.variance(a) / len(a) + st.variance(b) / len(b))
    return st.mean(a) - st.mean(b), math.erfc(abs(t) / math.sqrt(2))


def boot(xs, B=2000):
    """Same resampling as analysis/metrics.py (seed 0 per call)."""
    random.seed(0); m = []
    for _ in range(B):
        s = [random.choice(xs) for _ in xs]; m.append(sum(s) / len(s))
    m.sort()
    return m[int(.025 * B)], m[int(.975 * B)]


W1, W3 = win("W1"), win("W3")

# ---------------------------------------------------------------- data and method
S = "Data"
num(S, "total platform records", "75258", len(R))
counts = collections.Counter(r["acct"] for r in R)
for acct, paper in [("sh_agent1", "16693"), ("shreyaAgent2", "14767"), ("Agent_3", "14695"),
                    ("shreya_agent_4", "14780"), ("agent_5", "14323")]:
    num(S, f"records for account {ACCOUNT_NO[acct]} ({acct})", paper, counts[acct])
num(S, "strategy versions compared (W1 + W3)", "7", len(set(VERS["W1"]) | set(VERS["W3"])))
num(S, "accounts", "5", len(counts))
for wn in W:
    for v in VERS[wn]:
        accts = {r["acct"] for r in (W1 if wn == "W1" else W3) if r["ver"] == v}
        cond("Table 1", f"{v} runs on one account in {wn}", "one account", sorted(accts), len(accts) == 1)
for v, paper in [("V0", "1 (W1), 4 (W3)"), ("V1", "2 (W1, W3)"), ("V2", "3 (W1)"), ("V3", "4 (W1), 3 (W3)"),
                 ("V4", "5 (W1)"), ("V7", "5 (W3)"), ("V9", "1 (W3)")]:
    got = []
    for wn, rows in (("W1", W1), ("W3", W3)):
        accts = {ACCOUNT_NO[r["acct"]] for r in rows if r["ver"] == v}
        got += [f"{a} ({wn})" for a in accts]
    norm_expect = {f"{acct} ({w_.strip()})" for acct, ws in re.findall(r"(\d) \(([^)]*)\)", paper) for w_ in ws.split(",")}
    cond("Table 1", f"{v} account (window)", paper, ", ".join(sorted(got)), set(got) == norm_expect)
num(S, "W3 length in hours", "6.3", (W["W3"][1] - W["W3"][0]).total_seconds() / 3600)


def local_times(version, agent):
    return [T(d["timestamp"].replace("+0545", "+05:45")) for d in LOCAL
            if d["version"] == version and d["agent_name"] == agent]


v8, v9 = local_times("V8", "sh_agent1"), local_times("V9", "sh_agent1")
num(S, "V8/V9 overlap on account 1 (hours)", "5.4", (max(v8) - min(v9)).total_seconds() / 3600)
overlap = [r for r in R if r["ver"] == "V8/V9"]
num(S, "unassigned games from the V8/V9 overlap", "1892", len(overlap))
num(S, "  of which unlogged", "1778", sum(not r["local"] for r in overlap))
both_logs = collections.defaultdict(set)
for d in LOCAL:
    if d["agent_name"] == "sh_agent1":
        both_logs[d["game_id"]].add(d["version"])
num(S, "  of which logged under both V8 and V9", "114", sum({"V8", "V9"} <= both_logs[r["gid"]] for r in overlap))
num(S, "other unassigned games (restarts, version boundaries)", "1809",
    sum(1 for r in R if r["how"] in ("ambiguous", "gap") and r["ver"] != "V8/V9"))
inside = [r for r in R if r["how"] in ("ambiguous", "gap") and any(a <= T(r["start"]) <= b for a, b in W.values())]
cond(S, "unassigned games inside W1/W3, all in the last 30 s of W1", "8",
     f"{len(inside)}, starting {min(r['start'][11:19] for r in inside)}-{max(r['start'][11:19] for r in inside)}",
     len(inside) == 8 and all(W["W1"][1] - dt.timedelta(seconds=30) <= T(r["start"]) <= W["W1"][1] for r in inside))
cond(S, "no V8/V9 overlap games inside W1/W3", "none", sum(r["ver"] == "V8/V9" for r in inside),
     not any(r["ver"] == "V8/V9" for r in inside))
neg_all = [r for r in R if r["fam"] == "negotiation"]
num(S, "negotiation games with seller value >= buyer value (%)", "45",
    sum(not feasible(r) for r in neg_all) / len(neg_all), 100)
for fam, key in [("bargaining", "money_to_divide"), ("negotiation", "product_price_order"), ("persuasion", "product_price")]:
    vals = [r["cfg"][key] for r in R if r["fam"] == fam]
    cond("Overview", f"{fam} payoff scale", "10^2 to 10^6", f"{min(vals)} to {max(vals)}", min(vals) == 100 and max(vals) == 10 ** 6)

# W3 tests vs V0 (used for Bonferroni and the result paragraphs)
tests = {}
for fam in ["bargaining", "negotiation", "persuasion"]:
    base = sel(W3, "V0", fam)
    for v in VERS["W3"][1:]:
        g = sel(W3, v, fam)
        tests[(fam, v)] = (ztest(g, base), welch(g, base))
pvals = [p for (z, pz), (d, pt) in tests.values() for p in (pz, pt)]
num(S, "number of W3 version-vs-V0 tests", "24", len(pvals))
alpha = 0.05 / len(pvals)
num(S, "Bonferroni alpha", "0.0021", alpha)
sig_v9 = [p for fam in ["bargaining", "negotiation", "persuasion"] for p in (tests[(fam, "V9")][0][1], tests[(fam, "V9")][1][1]) if p < 0.05]
cond(S, "every significant V9 result survives Bonferroni", f"p < {alpha:.4f}", [f"{p:.2g}" for p in sig_v9], all(p < alpha for p in sig_v9))

# ---------------------------------------------------------------- logger bias / Figure 2
S = "Logger"
opp_shares, own_logged = [], {}
for wn, rows in (("W1", W1), ("W3", W3)):
    for v in VERS[wn]:
        g = [r for r in rows if r["ver"] == v]
        opp = [r for r in g if r["tactor"] == "opp"]; me = [r for r in g if r["tactor"] == "me"]
        opp_shares.append(len(opp) / len(g))
        own_logged[(wn, v)] = sum(r["local"] for r in me) / len(me)
        cond(S, f"{wn} {v}: opponent-ended games logged", "none", sum(r["local"] for r in opp), sum(r["local"] for r in opp) == 0)
span(S, "games ending on the opponent's move, both windows (%)", "45-51", opp_shares)
cond(S, "W1: logger captured every own-move ending", "100%", [f"{100 * own_logged[('W1', v)]:.1f}" for v in VERS["W1"]],
     all(own_logged[("W1", v)] == 1 for v in VERS["W1"]))
span(S, "W3: own-move endings missed (%)", "19-38", [1 - own_logged[("W3", v)] for v in VERS["W3"]])
fig2 = {"V0": ("3235", "46.8", "33.7", "63.4", "46.6"), "V1": ("3149", "46.5", "34.9", "59.6", "44.6"),
        "V3": ("3152", "47.4", "33.7", "67.9", "53.4"), "V7": ("3072", "46.6", "32.7", "49.2", "42.6"),
        "V9": ("3867", "51.4", "38.9", "33.3", "51.3")}
for v, (n, opp, logged, loc, plat) in fig2.items():
    g = [r for r in W3 if r["ver"] == v]; ng = [r for r in g if r["fam"] == "negotiation"]
    num("Fig 2", f"{v} games", n, len(g))
    num("Fig 2", f"{v} opponent-ended %", opp, sum(r["tactor"] == "opp" for r in g) / len(g), 100)
    num("Fig 2", f"{v} logged overall %", logged, sum(r["local"] for r in g) / len(g), 100)
    num("Fig 2", f"{v} negotiation agreement, platform %", plat, rate(ng), 100)
    if loc:
        num("Fig 2", f"{v} negotiation agreement, local logs %", loc, rate([r for r in ng if r["local"]]), 100)
v0_loc, v9_loc = rate([r for r in W3 if r["ver"] == "V0" and r["fam"] == "negotiation" and r["local"]]), \
    rate([r for r in W3 if r["ver"] == "V9" and r["fam"] == "negotiation" and r["local"]])
v0_pl, v9_pl = rate([r for r in W3 if r["ver"] == "V0" and r["fam"] == "negotiation"]), \
    rate([r for r in W3 if r["ver"] == "V9" and r["fam"] == "negotiation"])
loc_vs_plat = {v: (rate([r for r in W3 if r["ver"] == v and r["fam"] == "negotiation" and r["local"]]),
                   rate([r for r in W3 if r["ver"] == v and r["fam"] == "negotiation"])) for v in VERS["W3"]}
cond("Fig 2", "local logs overstate agreement for every version except V9", "all but V9 too high",
     {v: f"{100 * a:.1f} vs {100 * b:.1f}" for v, (a, b) in loc_vs_plat.items()},
     all((a > b) == (v != "V9") for v, (a, b) in loc_vs_plat.items()))
cond("Fig 2", "local logs reverse the V0-V9 direction", "V0 > V9 locally, V9 > V0 on platform",
     f"local {100 * v0_loc:.1f} vs {100 * v9_loc:.1f}; platform {100 * v0_pl:.1f} vs {100 * v9_pl:.1f}", v0_loc > v9_loc and v9_pl > v0_pl)
v0_deals = [r for r in W3 if r["ver"] == "V0" and r["fam"] == "negotiation" and succ(r)]
own_share = sum(r["tactor"] == "me" for r in v0_deals) / len(v0_deals)
cond(S, "most of V0's W3 negotiation deals end on its own move", "most", f"{100 * own_share:.1f}%", own_share > 0.5)
for v, paper in [("V9", "66"), ("V0", "27")]:
    g = sel(W3, v, "negotiation")
    num(S, f"{v} feasible negotiation games ending on opponent's move (%)", paper, sum(r["tactor"] == "opp" for r in g) / len(g), 100)
for v, paper in [("V9", "87"), ("V0", "75")]:
    g = sel(W3, v, "negotiation")
    acc, rej, oth = (sum(r[k] for r in g) for k in ("dec_accept", "dec_reject", "dec_other"))
    num(S, f"{v} share of its feasible negotiation decisions that reject (%)", paper, rej / (acc + rej + oth), 100)
w1_v0 = [r for r in W1 if r["ver"] == "V0" and r["fam"] == "negotiation"]
num(S, "W1 V0 negotiation agreement, local logs (%)", "65.3", rate([r for r in w1_v0 if r["local"]]), 100)
num(S, "W1 V0 negotiation agreement, platform (%)", "49.3", rate(w1_v0), 100)
for v, paper in [("V0", "64.6"), ("V9", "30.6")]:
    neg = [d for d in LOCAL if d["version"] == v and d["family"] == "negotiation"]
    num("Abstract", f"full local {v} log negotiation agreement (%)", paper, sum(d["outcome"] == "agreement" for d in neg) / len(neg), 100)
missed = 1 - sum(r["how"] == "logged" for r in R) / len(R)
cond("Lessons", "local logs missed half or more of all games", ">= 50%", f"{100 * missed:.1f}%", missed >= 0.5)

# ---------------------------------------------------------------- Figure 3 (W3 outcomes)
fig3 = {
    ("bargaining", "V0"): ("1069", "92.5", "0.441", "0.429", "0.452", None, None),
    ("bargaining", "V7"): ("1042", "95.1", "0.442", "0.432", "0.451", "0.001", "0.87"),
    ("bargaining", "V9"): ("1287", "96.3", "0.449", "0.440", "0.457", "0.008", "0.28"),
    ("negotiation", "V0"): ("572", "81.1", "0.350", "0.322", "0.381", None, None),
    ("negotiation", "V7"): ("580", "72.2", "0.366", "0.341", "0.390", "0.015", "0.43"),
    ("negotiation", "V9"): ("736", "90.1", "0.406", "0.387", "0.424", "0.055", "0.002"),
    ("persuasion", "V0"): ("1107", "53.7", "0.430", "0.398", "0.465", None, None),
    ("persuasion", "V7"): ("1021", "45.7", "0.400", "0.366", "0.434", "-0.030", "0.23"),
    ("persuasion", "V9"): ("1245", "52.9", "0.438", "0.404", "0.473", "0.008", "0.75"),
}
for (fam, v), (n, s, pay, lo, hi, delta, p) in fig3.items():
    g = sel(W3, v, fam)
    tag = f"{fam} {v}"
    num("Fig 3", f"{tag} N", n, len(g))
    num("Fig 3", f"{tag} success %", s, rate(g), 100)
    num("Fig 3", f"{tag} normalized payoff", pay, mean_pay(g))
    b_lo, b_hi = boot([norm(r) for r in g])
    num("Fig 3", f"{tag} 95% CI low", lo, b_lo)
    num("Fig 3", f"{tag} 95% CI high", hi, b_hi)
    if delta:
        d, pt = tests[(fam, v)][1]
        num("Fig 3", f"{tag} payoff difference vs V0", delta, d)
        num("Fig 3", f"{tag} payoff p-value", p, pt)

# ---------------------------------------------------------------- W3 result paragraphs
S = "Results"
(z, pz), (d, pt) = tests[("bargaining", "V9")]
num(S, "bargaining V9 vs V0 z", "4.10", z)
cond(S, "bargaining V9 vs V0 agreement p", "< 1e-4", f"{pz:.2g}", pz < 1e-4)
(z, pz), (d, pt) = tests[("negotiation", "V9")]
num(S, "negotiation V9 - V0 agreement (points)", "9.0", rate(sel(W3, "V9", "negotiation")) - rate(sel(W3, "V0", "negotiation")), 100)
num(S, "negotiation V9 vs V0 z", "4.66", z)
cond(S, "negotiation V9 vs V0 agreement p", "< 1e-5", f"{pz:.2g}", pz < 1e-5)
num(S, "negotiation payoff per agreement V9", "0.451", cond_pay(sel(W3, "V9", "negotiation")))
num(S, "negotiation payoff per agreement V0", "0.432", cond_pay(sel(W3, "V0", "negotiation")))
(z, pz), (d, pt) = tests[("negotiation", "V7")]
num(S, "negotiation V7 - V0 agreement (points)", "-8.9", rate(sel(W3, "V7", "negotiation")) - rate(sel(W3, "V0", "negotiation")), 100)
cond(S, "negotiation V7 agreement p", "< 0.001", f"{pz:.2g}", pz < 0.001)
num(S, "negotiation payoff per agreement V7", "0.507", cond_pay(sel(W3, "V7", "negotiation")))
for v, paper in [("V1", "77.7"), ("V3", "81.8")]:
    (z, pz), (d, pt) = tests[("negotiation", v)]
    num(S, f"negotiation {v} agreement %", paper, rate(sel(W3, v, "negotiation")), 100)
    cond(S, f"negotiation {v} does not differ from V0", "p > 0.05", f"{pz:.2g}, {pt:.2g}", pz > 0.05 and pt > 0.05)
beats = [v for v in VERS["W3"][1:] for (z, pz), (d, pt) in [tests[("persuasion", v)]] if (z > 0 and pz < 0.05) or (d > 0 and pt < 0.05)]
cond(S, "no persuasion variant beats V0", "none", beats, not beats)
(z, pz), (d, pt) = tests[("persuasion", "V9")]
num(S, "persuasion V9 - V0 success (points)", "-0.9", rate(sel(W3, "V9", "persuasion")) - rate(sel(W3, "V0", "persuasion")), 100)
num(S, "persuasion V9 success p", "0.66", pz)
(z, pz), (d, pt) = tests[("persuasion", "V7")]
cond(S, "persuasion V7 success p", "< 0.001", f"{pz:.2g}", pz < 0.001)
w3_pers = [rate(sel(W3, v, "persuasion")) for v in ("V0", "V1", "V3")]
num(S, "W3 vanilla persuasion arms span (points)", "0.8", max(w3_pers) - min(w3_pers), 100)
w1_pers = {v: sel(W1, v, "persuasion") for v in ("V0", "V1", "V2", "V3")}
rates = [rate(g) for g in w1_pers.values()]
num(S, "W1 vanilla persuasion arms span (points)", "7.1", max(rates) - min(rates), 100)
span(S, "W1 vanilla persuasion success range (%)", "44.0-51.1", rates)
cond(S, "W1 vanilla persuasion arms have about 730 games each", "about 730", [len(g) for g in w1_pers.values()],
     all(abs(len(g) - 730) <= 10 for g in w1_pers.values()))
z, pz = ztest(w1_pers["V2"], w1_pers["V0"])
num(S, "W1 persuasion V2 - V0 (points)", "6.3", rate(w1_pers["V2"]) - rate(w1_pers["V0"]), 100)
num(S, "W1 persuasion V2 vs V0 p", "0.015", pz)
w3_barg_gap = abs(rate(sel(W3, "V3", "bargaining")) - rate(sel(W3, "V0", "bargaining")))
num(S, "W3 largest gap between identical vanilla arms (points)", "1.0", max(w3_barg_gap, max(w3_pers) - min(w3_pers)), 100)
opt_num(S, "V1 bargaining agreement W1 (%)", "97.3", rate(sel(W1, "V1", "bargaining")), 100)
num(S, "V1 bargaining agreement W3 (%)", "92.5", rate(sel(W3, "V1", "bargaining")), 100)

# ---------------------------------------------------------------- Figure 4 (W1 negotiation ladder)
fig4 = {"V0": ("406", "86.0", None, "0.349", None), "V1": ("409", "81.7", "0.10", "0.363", "0.54"),
        "V2": ("398", "84.9", "0.68", "0.354", "0.81"), "V3": ("403", "85.4", "0.81", "0.348", "0.97"),
        "V4": ("410", "82.9", "0.23", "0.338", "0.63")}
base = sel(W1, "V0", "negotiation")
for v, (n, s, ps, pay, pp) in fig4.items():
    g = sel(W1, v, "negotiation")
    num("Fig 4", f"{v} N", n, len(g))
    num("Fig 4", f"{v} agreement %", s, rate(g), 100)
    num("Fig 4", f"{v} normalized payoff", pay, mean_pay(g))
    if ps:
        (z, pz), (d, pt) = ztest(g, base), welch(g, base)
        num("Fig 4", f"{v} agreement p vs V0", ps, pz)
        num("Fig 4", f"{v} payoff p vs V0", pp, pt)
        cond("Fig 4", f"{v} does not differ from V0", "p > 0.05", f"{pz:.2g}, {pt:.2g}", pz > 0.05 and pt > 0.05)
if "sat 3.0 points below V0" in TEX:
    num("Fig 4", "V4 placebo gap below V0 (points)", "3.0", rate(base) - rate(sel(W1, "V4", "negotiation")), 100)

# ---------------------------------------------------------------- reliability
S = "Reliability"
w1_inv = {v: sum(r["inval"] > 0 for r in W1 if r["ver"] == v) for v in VERS["W1"]}
num(S, "W1: most games with an invalid move in one version", "7", max(w1_inv.values()))
barg = {v: sel(W3, v, "bargaining") for v in VERS["W3"]}
span(S, "W3 bargaining games with an invalid move, per account (%)", "14-23",
     [sum(r["inval"] > 0 for r in g) / len(g) for g in barg.values()])
with_inv = {v: [r for r in g if r["inval"] > 0] for v, g in barg.items()}
without = {v: [r for r in g if r["inval"] == 0] for v, g in barg.items()}
span(S, "agreement in games with such a move (%)", "73-85", [rate(g) for g in with_inv.values()])
span(S, "agreement in games without (%)", "96-100", [rate(g) for g in without.values()])
span(S, "failed bargaining games that contain one (%)", "58-94",
     [sum(1 for r in with_inv[v] if not succ(r)) / sum(1 for r in g if not succ(r)) for v, g in barg.items()])
num(S, "V9 agreement in games with such a move (%)", "84.8", rate(with_inv["V9"]), 100)
num(S, "V0 agreement in games with such a move (%)", "73.3", rate(with_inv["V0"]), 100)
stale = {v: sum(r["inv_stale"] for r in g) / sum(r["inval"] for r in g) for v, g in barg.items()}
cond(S, "invalid moves that are a decision sent when an offer was due", "over 98%",
     {v: f"{100 * s:.1f}" for v, s in stale.items()}, all(s > 0.98 for s in stale.values()))
resent = {v: sum(r["inv_resent"] for r in g) / sum(r["inval"] for r in g) for v, g in barg.items()}
cond(S, "invalid moves within 1 s of a valid decision", "most (> 50%)",
     {v: f"{100 * s:.1f}" for v, s in resent.items()}, all(s > 0.5 for s in resent.values()))
REJ = collections.defaultdict(set)
with open("paper_data/run_log_rejections.csv", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        REJ[(row["run_log"], row["game_id"])].add(row["server_message"])
for v, log, paper in [("V9", "v9_run.log", "94.5"), ("V0", "v0_run.log", "64")]:
    hit = [r for r in with_inv[v] if (log, r["gid"]) in REJ]
    num(S, f"{v} affected games with a server rejection in our run log (%)", paper, len(hit) / len(with_inv[v]), 100)
    schema = sum(any("Offer must include" in k for k in REJ[(log, r["gid"])]) for r in hit) / len(hit)
    cond(S, f"{v} rejections are almost always the offer-schema error", ">= 90%", f"{100 * schema:.1f}%", schema >= 0.9)
all_rej = collections.defaultdict(set)
with open("paper_data/run_log_rejections.csv", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        all_rej[row["server_message"]].add((row["run_log"], row["game_id"]))
opt_num(S, "games with any run-log rejection (V0 + V9 logs)", "4126", len(set().union(*all_rej.values())))
for msg, paper in [("It is not your turn", "3394"), ("Offer must include", "651"), ("Game is not active", "393")]:
    opt_num(S, f"games with rejection '{msg}'", paper, len(set().union(*[g for m, g in all_rej.items() if m.startswith(msg)])))
humans = sum(r["opp_type"] == "human" for r in R)
opt_num("Limitations", "games against human opponents", "122", humans)
num("Limitations", "share of games against humans (%)", "0.2", humans / len(R), 100)
oc = {v: collections.Counter(r["outcome"] for r in W3 if r["ver"] == v) for v in VERS["W3"]}
num(S, "W3 most timeouts in one version", "20", max(c["timeout"] for c in oc.values()))
num(S, "W3 most invalid_moves_exceeded in one version", "2", max(c["invalid_moves_exceeded"] for c in oc.values()))

# ---------------------------------------------------------------- learned threshold
S = "Learned"
import learned_model
import train_model as tm

learned_model._model_path = lambda family: f"paper_data/learned_{family}.json"   # use the released model file
learned_model._MODELS.clear()
ex = [([float(row["margin"]), float(row["round_frac"])], float(row["agreement"]))
      for row in csv.DictReader(open("paper_data/learned_model_training.csv", newline="", encoding="utf-8"))]
num(S, "training decision points", "5969", len(ex))
cut = int(0.8 * len(ex))
w_, b_, m_, s_ = tm._fit_logistic(ex[:cut])
res = tm._evaluate(ex[cut:], w_, b_, m_, s_)
num(S, "accuracy on last 20% (logging order)", "0.907", res["accuracy"])
num(S, "majority baseline on last 20%", "0.534", res["baseline_accuracy"])
random.seed(0); sh = ex[:]; random.shuffle(sh)
w_, b_, m_, s_ = tm._fit_logistic(sh[:cut])
res = tm._evaluate(sh[cut:], w_, b_, m_, s_)
num(S, "accuracy on random 20%", "0.796", res["accuracy"])
num(S, "majority baseline on random 20%", "0.503", res["baseline_accuracy"])
th = {r: learned_model.implied_accept_threshold("negotiation", r) for r in range(1, 10)}
cond(S, "50% point negative in rounds 1-8, not in round 9", "rounds 1-8",
     {r: round(v, 3) for r, v in th.items()}, all(th[r] < 0 for r in range(1, 9)) and th[9] >= 0)


def losing(v):
    return [r for r in R if r["ver"] == v and r["fam"] == "negotiation" and (r["pay"] or 0) < 0]


def in_w(r, wn):
    return W[wn][0] <= T(r["start"]) <= W[wn][1]


v3 = losing("V3")
num(S, "V3 money-losing deals", "160", len(v3))
num(S, "V3 money-losing deals in W1", "60", sum(in_w(r, "W1") for r in v3))
num(S, "V3 money-losing deals in W3", "99", sum(in_w(r, "W3") for r in v3))


def own_value(r):
    c = r["cfg"]
    return (c["seller_value"] if r["me"] == "player_1" else c["buyer_value"]) * c["product_price_order"]


num(S, "V3 mean loss per losing deal (% of own valuation)", "5.2", st.mean(-r["pay"] / own_value(r) for r in v3), 100)
num(S, "V8 money-losing deals", "153", len(losing("V8")))
for v in ("V0", "V2", "V4", "V9"):
    num(S, f"{v} money-losing deals", "0", len(losing(v)))

# ---------------------------------------------------------------- official rating
S = "Rating"
for wn, rows, papers in [("W3", W3, {"V9": "-0.016", "V0": "-0.037", "V1": "-0.041", "V7": "-0.045", "V3": "-0.124"}),
                         ("W1", W1, {"V2": "0.054", "V3": "-0.103", "V4": "-0.112", "V0": "-0.031"})]:
    per = {v: st.mean(r["rd"] or 0 for r in rows if r["ver"] == v) for v in VERS[wn]}
    for v, paper in papers.items():
        num(S, f"{wn} {v} mean rating change per game", paper, per[v])
    if wn == "W3":
        cond(S, "W3: V9 loses the least rating per game, V3 the most", "V9 / V3",
             f"{max(per, key=per.get)} / {min(per, key=per.get)}", max(per, key=per.get) == "V9" and min(per, key=per.get) == "V3")

# ---------------------------------------------------------------- strategy descriptions vs. source code
S = "Code"
src = {f: open(f, encoding="utf-8").read() for f in ("vanilla_agent.py", "experiment_agent.py", "advanced_agent.py")}
cond(S, "V0 bargaining proposes an even split", "even split", "half = money / 2", "half = money / 2" in src["vanilla_agent.py"])
cond(S, "V0 bargaining accepts at least 40% of the pool", "40%", "my_gain >= money * 0.4", "my_gain >= money * 0.4" in src["vanilla_agent.py"])
cond(S, "V3 learned threshold clipped to [-0.10, 0.20]", "[-0.10, 0.20]", "max(-0.1, min(0.2, learned_margin))",
     "max(-0.1, min(0.2, learned_margin))" in src["experiment_agent.py"])
favor = float(re.search(r"favor = ([\d.]+)", src["advanced_agent.py"]).group(1))
cond(S, "V9 bargaining opens slightly in its own favor", "> 0.5", favor, 0.5 < favor < 0.6)
adv = src["advanced_agent.py"]
for what, paper, snippet in [
    ("V9 bargaining opens at 53.5% of the pool", "53.5%", "favor = 0.535"),
    ("V9 bargaining accepts at 42.5% of the pool", "42.5%", "accept_threshold = 0.425"),
    ("V9 bargaining takes 10% of its threshold from the learned model", "10%", "0.9 * accept_threshold + 0.1 * learned_accept"),
    ("V9 bargaining threshold falls by up to 8 points", "8 points", "accept_threshold - 0.08 * progress"),
    ("V9 bargaining accepts at most 34% in the final round", "34%", "min(effective_accept, 0.34)"),
    ("V9 negotiation surplus-share threshold starts at 52%", "52%", "min_share = 0.52 - 0.24 * progress"),
    ("V9 negotiation blends 0.75 hand-set, 0.25 learned, clipped to [0, 0.12]", "0.75 / 0.25 / [0, 0.12]",
     "0.75 * min_margin + 0.25 * max(0.0, min(0.12, learned_margin))"),
    ("V9 negotiation accepts only profitable offers", "profitable", "if profitable and margin >= max(0.008, min_margin)"),
    ("V9 persuasion exploits best frame 97% of the time after four attempts", "97% / four",
     "if attempted >= 4 and random.random() >= 0.03"),
]:
    cond(S, what, paper, snippet, snippet in adv)
cond(S, "V9 negotiation counter-offers stay profitable", "profitable", "my_value * 0.985 / my_value * 1.015",
     "min(my_value * 0.985, my_value * multiplier)" in adv and "max(my_value * 1.015, my_value * multiplier)" in adv
     and "min(0.62, max(0.34, target_share))" in adv)
import persuasion_model_v2
cond(S, "six persuasion argument frames", "economic, fairness, safety, social, convenience, long-term",
     persuasion_model_v2.ARGUMENT_TYPES,
     persuasion_model_v2.ARGUMENT_TYPES == ["economic", "fairness", "safety", "social", "convenience", "long_term"])
# Platform history entries carry no recommendation/decision/message key, so the lie count is always 0 and the
# buyer rule reduces to p*(v-price) - 1.5*(1-p)*(price-u) > 0. v=10, u=2, p=0.6: plain EV (6.8) buys at 6.05, the 1.5 weight does not.
buy = lambda price: persuasion_model_v2.buyer_decision_action_v2({}, {"product_price": price, "v": 10, "u": 2, "p": 0.6, "history": []})
cond(S, "V9 buyer weights the possible loss 1.5 times", "1.5", f"{persuasion_model_v2._RISK_AVERSION} extra",
     persuasion_model_v2._RISK_AVERSION == 0.5 and buy(6.05)["decision"] == "no" and buy(5.9)["decision"] == "yes")

# ---------------------------------------------------------------- bar widths in Figures 3 and 4
tex = open(PAPER).read()
for width, pct in re.findall(r"\\barcell\{\w+\}\{([\d.]+)\} ([\d.]+)\\%", tex):
    report("Bars", f"bar for {pct}%", width, f"{float(pct) / 100 * 0.88:.3f}", rounds_to(float(pct) / 100 * 0.88, width))

print(f"\n{'ALL CLAIMS PASS' if failures == 0 else f'{failures} CLAIM(S) FAILED'}")
sys.exit(1 if failures else 0)
