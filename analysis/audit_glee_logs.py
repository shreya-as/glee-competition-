#!/usr/bin/env python3
"""Reproduce the manuscript audit from saved exports; no SDK/network needed.

Run from any directory: python analysis/audit_glee_logs.py
Outputs audit_results.json, audit_report.md, and audit_tables.tex beside this file.
Raw logs and model files are never modified. Monetary values are game payoff
units, not verified dollars. Export completeness and strategy identity for
unlogged games cannot be established from the saved files alone.
"""

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
FAMILIES = ("bargaining", "negotiation", "persuasion")
SOURCES = {
    "v0": ("final_logs/final_v0_shreya_agent_4_history_with_replays.json", "shreya_agent_4"),
    "v9": ("final_logs/final_v9_sh_agent1_all_games_with_replays.json", "sh_agent1"),
}


def dt(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def success(row):
    h = row["history"]
    return payoff(row) > 0 if h["game_family"] == "persuasion" else h["result"]["outcome"] == "agreement"


def payoff(row):
    h = row["history"]
    return h["result"][h["your_player"] + "_payoff"]


def metrics(rows):
    if not rows:
        return {"n": 0}
    won = [r for r in rows if success(r)]
    return {"n": len(rows), "successes": len(won), "success_pct": 100 * len(won) / len(rows),
            "mean_payoff": mean(payoff(r) for r in rows),
            "mean_payoff_success": mean(payoff(r) for r in won) if won else None,
            "outcomes": dict(sorted(Counter(r["history"]["result"]["outcome"] for r in rows).items()))}


def acceptance_actor(row):
    h, state = row["history"], row["replay"]["state"]
    if h["game_family"] == "persuasion" or h["result"]["outcome"] != "agreement":
        return None
    last = (state.get("history") or [{}])[-1]
    actor = last.get("decided_by")
    if h["game_family"] == "bargaining" and last.get("decision") == "accept":
        proposer = last.get("proposer")
        actor = {"player_1": "player_2", "player_2": "player_1"}.get(proposer)
    return "unknown" if actor is None else "self" if actor == h["your_player"] else "opponent"


def coverage(rows, local):
    api_ids = {r["history"]["game_id"] for r in rows}
    local_ids = {r["game_id"] for r in local}
    grouped = defaultdict(list)
    for r in local:
        grouped[r["game_id"]].append(r)
    disagreements, missing_results = [], []
    for r in rows:
        gid = r["history"]["game_id"]
        matched = grouped.get(gid, [])
        terminal = [x for x in matched if x.get("outcome") is not None and x.get("payoff") is not None]
        if matched and not terminal:
            missing_results.append(gid)
        if terminal and any(x["role"] != r["history"]["your_player"] or x["payoff"] != payoff(r)
                            or x["outcome"] != r["history"]["result"]["outcome"] for x in terminal):
            disagreements.append(gid)
    groups = {}
    for fam in FAMILIES:
        sub = [r for r in rows if r["history"]["game_family"] == fam]
        both = [r for r in sub if r["history"]["game_id"] in local_ids]
        missing = [r for r in sub if r["history"]["game_id"] not in local_ids]
        acceptances = Counter(("matched_" if r["history"]["game_id"] in local_ids else "missing_") + acceptance_actor(r)
                              for r in sub if acceptance_actor(r))
        groups[fam] = {"all": metrics(sub), "matched": metrics(both), "api_only": metrics(missing),
                       "acceptance_actors": dict(sorted(acceptances.items()))}
    return {"api_games": len(api_ids), "local_rows": len(local), "local_unique": len(local_ids),
            "duplicate_local_rows": len(local) - len(local_ids), "in_both": len(api_ids & local_ids),
            "api_only": len(api_ids - local_ids), "local_only": len(local_ids - api_ids),
            "missing_local_terminal_result": len(missing_results),
            "terminal_disagreement_ids": disagreements, "families": groups}


def compact(value):
    return f"{value / 1e6:.2f}M" if abs(value) >= 1e6 else f"{value / 1e3:.1f}K"


def main():
    local = {f"v{i}": [json.loads(line) for line in (ROOT / f"logs/v{i}_games.jsonl").read_text().splitlines() if line.strip()]
             for i in range(10)}
    api = {v: json.loads((ROOT / path).read_text()) for v, (path, _) in SOURCES.items()}
    manifest = {}
    for path in [p for p, _ in SOURCES.values()] + [f"logs/v{i}_games.jsonl" for i in range(10)] + ["logs/learned_bargaining.json", "logs/learned_negotiation.json"]:
        raw = (ROOT / path).read_bytes()
        manifest[path] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    checks = {}
    for v, rows in api.items():
        assert len(rows) == len({r["history"]["game_id"] for r in rows}), f"Duplicate official game IDs: {v}"
        for r in rows:
            h, replay = r["history"], r["replay"]
            assert h["game_id"] == replay["game_id"]
            assert h["game_family"] == replay["game_family"]
            assert h["your_player"] == replay["your_player"]
            assert h["result"] == replay["result"]
            assert h["your_player_name"] == SOURCES[v][1]
            assert isinstance(payoff(r), (int, float))
        checks[v] = {"rows": len(rows), "player_ids": sorted({r["history"]["your_player_id"] for r in rows}),
                     "start_min_utc": min(dt(r["history"]["started_at"]) for r in rows).isoformat(),
                     "start_max_utc": max(dt(r["history"]["started_at"]) for r in rows).isoformat()}
    start = max(min(dt(r["history"]["started_at"]) for r in rows) for rows in api.values())
    end = min(max(dt(r["history"]["started_at"]) for r in rows) for rows in api.values())
    full, aligned = {}, {}
    aligned_rows = {}
    for v, rows in api.items():
        account = SOURCES[v][1]
        account_local = [r for r in local[v] if r["agent_name"] == account]
        full[v] = coverage(rows, account_local)
        selected = [r for r in rows if start <= dt(r["history"]["started_at"]) <= end]
        selected_ids = {r["history"]["game_id"] for r in selected}
        # For aligned coverage, use membership, not local completion timestamps:
        # local timestamps describe a different event from server started_at.
        aligned_rows[v] = selected
        aligned[v] = coverage(selected, [r for r in account_local if r["game_id"] in selected_ids])
    v8_ids = {r["game_id"] for r in local["v8"] if r["agent_name"] == SOURCES["v9"][1]}
    v9_ids = {r["game_id"] for r in local["v9"]}
    official_v9_ids = {r["history"]["game_id"] for r in api["v9"]}
    aligned_v9_ids = {r["history"]["game_id"] for r in aligned_rows["v9"]}
    identity = {"official_v9_ids_with_v8_label": len(official_v9_ids & v8_ids),
                "official_v9_ids_with_both_v8_v9_labels": len(official_v9_ids & v8_ids & v9_ids),
                "aligned_v9_ids_with_v8_label": len(aligned_v9_ids & v8_ids),
                "aligned_v9_ids_with_v9_label": len(aligned_v9_ids & v9_ids),
                "aligned_v9_ids_without_local_version_evidence": len(aligned_v9_ids - v9_ids - v8_ids),
                "last_v8_local_timestamp": max(r["timestamp"] for r in local["v8"])}
    strata = {}
    for v, rows in aligned_rows.items():
        grouped = defaultdict(list)
        for r in rows:
            h, replay = r["history"], r["replay"]
            config, state = replay["config"], replay["state"]
            # Explicitly descriptive: these groups do not identify opponents.
            role = state.get(h["your_player"] + "_role", h["your_player"])
            scale = config.get("money_to_divide", config.get("product_price_order", config.get("product_price", "unknown")))
            key = f"{h['game_family']}|role={role}|complete_information={config.get('complete_information', 'NA')}|scale={scale}"
            grouped[key].append(r)
        strata[v] = {key: metrics(group) for key, group in sorted(grouped.items())}
    models = {f: json.loads((ROOT / f"logs/learned_{f}.json").read_text())["metrics"] for f in ("bargaining", "negotiation")}
    result = {"sources": manifest, "integrity": checks, "shared_start_window_utc": [start.isoformat(), end.isoformat()],
              "full_exports": full, "shared_window": aligned, "version_identity": identity,
              "shared_window_strata": strata, "saved_training_metrics_not_held_out": models}
    (OUT / "audit_results.json").write_text(json.dumps(result, indent=2) + "\n")
    report = ["# Saved GLEE log audit", "", "Reproduce: `python analysis/audit_glee_logs.py` (Python standard library only).",
              "", "The exported filenames identify accounts/windows, not authenticated strategy versions. All official IDs, roles, families and results agree between history and replay. Source hashes are in `audit_results.json`.",
              "", f"Shared window: games **started** from {start.isoformat()} to {end.isoformat()}, inclusive. Local timestamps mark completion; membership is therefore joined by game ID, not timestamp equality.",
              "", "## Full-export coverage (same account and version-labelled local file)", "",
              "| Label | API games | Local rows | Local unique | Shared | API only | Local only |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for v, c in full.items():
        report.append(f"| {v} | {c['api_games']} | {c['local_rows']} | {c['local_unique']} | {c['in_both']} | {c['api_only']} | {c['local_only']} |")
    report += ["", "The V0 local file also contains 1,669 rows from sh_agent1. Those rows belong to another account/window and are excluded from this coverage denominator.",
               "", "## Version identity", "", *[f"- `{k}`: {value}" for k, value in identity.items()],
               "", "The shared window avoids all observed V8-labelled games. Unmatched official games still have no independent strategy label. Absence of observed V8 overlap does not prove exclusive V9 execution.",
               "", "## Shared-window outcomes", "", "Payoffs are game units. Success is agreement for bargaining/negotiation and strictly positive own payoff for persuasion; negative and zero persuasion payoffs remain in the unconditional mean.", "",
               "| Family | Label | N | Success % | Mean payoff | Mean payoff given success | Matched N | Matched success % | API-only N | API-only success % |",
               "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for fam in FAMILIES:
        for v in SOURCES:
            g = aligned[v]["families"][fam]
            a, m, u = (g[k] for k in ("all", "matched", "api_only"))
            report.append(f"| {fam} | {v} | {a['n']} | {a['success_pct']:.2f} | {a['mean_payoff']:.3f} | {a['mean_payoff_success']:.3f} | {m['n']} | {m['success_pct']:.2f} | {u['n']} | {u['success_pct']:.2f} |")
    report += ["", "## Replay acceptance attribution in shared window", "", "| Family | Label | Self, matched | Opponent, matched | Self, API only | Opponent, API only |", "|---|---|---:|---:|---:|---:|"]
    for fam in FAMILIES[:2]:
        for v in SOURCES:
            c = aligned[v]["families"][fam]["acceptance_actors"]
            report.append(f"| {fam} | {v} | {c.get('matched_self', 0)} | {c.get('matched_opponent', 0)} | {c.get('missing_self', 0)} | {c.get('missing_opponent', 0)} |")
    report += ["", "Bargaining actor is the other player from the final proposer; negotiation uses the final decided_by field. Missing self acceptances show that opponent termination is only one coverage mechanism.",
               "", "## Local result integrity", ""]
    for v, c in full.items():
        report.append(f"- {v}: {c['missing_local_terminal_result']} shared IDs have no local terminal result; {len(c['terminal_disagreement_ids'])} IDs have conflicting terminal outcome/payoff/role. Duplicate local rows are collapsed by ID only for coverage; all performance metrics use server results.")
    report += ["", "## Reliability outcomes in shared window", "", "| Label | Family | Outcomes |", "|---|---|---|"]
    for v in SOURCES:
        for fam in FAMILIES:
            report.append(f"| {v} | {fam} | {json.dumps(aligned[v]['families'][fam]['all']['outcomes'], sort_keys=True)} |")
    report += ["", "Timeout/invalid-move outcomes are game-level observations; this audit does not assign fault to our agent, infer schema root causes, or estimate concurrency failure rates.",
               "", "## Saved model metrics", "", "These are training-set metrics: train_model.py fits and evaluates on the same examples. They provide no held-out validation.", "", "```json", json.dumps(models, indent=2), "```",
               "", "## Remaining evidence gaps", "", "- Download pagination/completeness is unverified; counts describe saved exports.",
               "- Shared clock time does not control opponent matching, account ratings, configurations or adaptive state. Config/role/scale summaries are in audit_results.json; they are descriptive, not causal adjustments.",
               "- Historical random seeds and exact runtime revisions were not preserved in these game rows. Later runner seed support cannot reconstruct them.",
               "- No official leaderboard snapshot/rank is present in the audited artifacts. Per-game rating_delta is not a leaderboard snapshot."]
    (OUT / "audit_report.md").write_text("\n".join(report) + "\n")
    tex = ["% Generated by analysis/audit_glee_logs.py; do not edit numbers by hand.",
           r"\begin{table}[H]\centering\small", r"\caption{Full-export coverage after matching the local account. Counts are unique game IDs except local rows.}\label{tab:coverage}",
           r"\begin{tabular}{lrrrrrr}\toprule", r"Label & API & Local rows & Local IDs & Shared & API-only & Local-only \\\midrule"]
    for v, c in full.items():
        tex.append(f"{v.upper()} & {c['api_games']} & {c['local_rows']} & {c['local_unique']} & {c['in_both']} & {c['api_only']} & {c['local_only']} " + r"\\")
    tex += [r"\bottomrule\end{tabular}\end{table}", r"\begin{table}[H]\centering\small",
            r"\caption{Server outcomes in the shared start-time window. Payoffs are game units; K and M denote thousands and millions. V9 denotes the account window, with strategy-identity limits described in the text.}\label{tab:outcomes}",
            r"\begin{tabular}{llrrrr}\toprule", r"Family & Label & N & Success & Mean payoff & Given success \\\midrule"]
    for fam in FAMILIES:
        for v in SOURCES:
            a = aligned[v]["families"][fam]["all"]
            tex.append(f"{fam.title()} & {v.upper()} & {a['n']} & {a['success_pct']:.1f}\\% & {compact(a['mean_payoff'])} & {compact(a['mean_payoff_success'])} " + r"\\")
    tex += [r"\bottomrule\end{tabular}\end{table}", r"\begin{table}[H]\centering\small",
            r"\caption{Selection effects in the shared window. Matched and API-only success rates both use server outcomes. Opponent accepts are replay-confirmed agreements absent from the version-labelled local file.}\label{tab:selection}",
            r"\begin{tabular}{llrrrrr}\toprule", r"Family & Label & Matched N & Success & API-only N & Success & Opp. accepts \\\midrule"]
    for fam in FAMILIES:
        for v in SOURCES:
            g = aligned[v]["families"][fam]
            m, u = g["matched"], g["api_only"]
            actor = str(g["acceptance_actors"].get("missing_opponent", 0)) if fam != "persuasion" else "--"
            tex.append(f"{fam.title()} & {v.upper()} & {m['n']} & {m['success_pct']:.1f}\\% & {u['n']} & {u['success_pct']:.1f}\\% & {actor} " + r"\\")
    tex += [r"\bottomrule\end{tabular}\end{table}", r"\begin{table}[H]\centering\small",
            r"\caption{Game-level non-agreement outcomes in the shared window. Counts include either player's failures; they do not establish our agent's fault or a concurrency/schema root cause.}\label{tab:reliability}",
            r"\begin{tabular}{llrrrr}\toprule", r"Family & Label & Timeout & Invalid-limit & No deal & Walkaway \\\midrule"]
    for fam in FAMILIES:
        for v in SOURCES:
            c = aligned[v]["families"][fam]["all"]["outcomes"]
            tex.append(f"{fam.title()} & {v.upper()} & {c.get('timeout',0)} & {c.get('invalid_moves_exceeded',0)} & {c.get('no_deal',0)} & {c.get('walked_away',0)} " + r"\\")
    tex += [r"\bottomrule\end{tabular}\end{table}"]
    (OUT / "audit_tables.tex").write_text("\n".join(tex) + "\n")
    print(f"Validated {sum(len(rows) for rows in api.values()):,} server records; wrote audit_results.json, audit_report.md, audit_tables.tex")


if __name__ == "__main__":
    main()
