# GLEE Competition Paper — Scope Comparison & Data Audit

Compares `competition_paper_draft.html` against the official GLEE competition-paper call, and cross-checks its factual claims against the actual codebase (`fleet.py`, `advanced_agent.py`, `experiment_agent.py`, `run_sh_agent1_v9.py`) and logs (`logs/games.jsonl`) as of 2026-09-04.

---

## 1. Section-by-Section Coverage

| Required Section (per call) | Status | Coverage | Notes |
|---|---|---|---|
| Agent overview & motivation | ✅ Complete | ~90% | Abstract + Section 1 clearly state the reliability-first motivation and contributions. |
| Technical approach (models, prompts, planning, memory, learning, opponent modeling) | ⚠️ Partial | ~60% | Opponent modeling, concession ramps, learned calibration described narratively. **No prompts** (agent is rule-based, not LLM — fine, but should say so explicitly since the call asks about prompts). No hyperparameters, no formulas for concession schedule or learned-model architecture/features beyond one sentence. `bargaining_model_v2.py`, `belief_model.py` exist in repo but are never mentioned. |
| Strategic design choices | ✅ Complete | ~85% | Section 3 covers incentive/opponent/uncertainty reasoning per family reasonably well. |
| Development process (tried/failed/evolved) | ⚠️ Partial | ~50% | Mentions reverted "calibrated lying" persuasion variant and negotiation deadlock fix. **Omits** `llm_agent.py`, `litellm_agent.py`, `reflective_agent.py` — LLM-based approaches that exist in the repo and were apparently abandoned in favor of the rule-based fleet. This is exactly the kind of negative result the call asks for, and it's currently missing. |
| Evaluation and analysis | ⚠️ Partial | ~40% | Game-level table (Section 5) present but numbers don't reproduce (see §3 below). Section 5.1 controlled-comparison table — the paper's headline "contribution" — is **100% placeholder** (`TBD` in every cell). |
| **"Agent Behavior Analysis" section** | ❌ Missing | 0% | The call explicitly **requires** a section by this name addressing whether behavior matches design intent and how alignment was verified. Draft has no section with this name or explicit scope. Section 6 (Discussion) gestures at this but doesn't satisfy the requirement as written. |
| Limitations and reproducibility | ⚠️ Partial | ~55% | Limitations (confounded components, selection bias) are honestly discussed. Reproducibility is weak: no seeds, no date range for logs, no environment/version pinning, no link to a specific commit hash (ironically, the paper itself recommends logging commit hash but doesn't cite its own). |
| References | ⚠️ Partial | — | 4 references present; one has a **fabricated title** for a real arXiv ID (see §3). |

---

## 2. Mandatory Submission Requirements — Outstanding Items

| Requirement | Status | Action Needed |
|---|---|---|
| **"Agent Behavior Analysis" section** (explicitly mandatory) | ❌ Not present | Add a dedicated section: state the intended behavior per design, show log-based evidence it happened (or didn't), and describe the verification method. |
| Format: NeurIPS 2026 paper template | ❌ Non-compliant | Draft is custom single-column HTML/CSS (Liberation Serif, 10.5pt), not the NeurIPS two-column LaTeX style. Needs to be rebuilt in the actual NeurIPS template before submission. |
| Length: ≤4 pages + unlimited references | ⚠️ Likely over | Current PDF renders **7 pages total** in this custom layout. Even accounting for the density difference of the real two-column NeurIPS template, the body (Sections 1–7, four tables, two ASCII figures) is long and should be re-checked once reflowed into the actual template. |
| Corresponding GLEE Agent ID | ❌ Not in draft | Not a paper-body requirement per se, but must be supplied in the OpenReview form — confirm which of the 5 fleet agents (`sh_agent1`, `shreyaAgent2`, `Agent_3`, `shreya_agent_4`, `agent_5`) is the one being submitted, since the paper currently misidentifies `sh_agent1`'s role (see §3, finding 1). |
| Active OpenReview profile (≥2 weeks pre-deadline) | ❓ Unknown | Outside this repo's scope — confirm separately with authors. |
| Author name / affiliation | ❌ Placeholder | Title page still has `[Author Name], [Affiliation]`. |

---

## 3. False / Inaccurate / Unverifiable Data

This table is the direct answer to "if there is any false data or unnatural data... provide that data too" and "assumption-based vs accurate result." Each row is something checked against the actual code or logs, not just re-read from the draft.

| # | Claim in Paper | What Was Checked | Actual Finding | Verdict |
|---|---|---|---|---|
| 1 | "we configured `sh_agent1` as a separate **vanilla-control** agent" (Abstract, Intro, Section 2, Section 5) — this is one of the paper's 4 headline contributions | `fleet.py` `AGENT_VERSIONS` map + `run_sh_agent1_v9.py` | `sh_agent1` runs **`v9` = "leaderboard-push"**, the most adaptive strategy (imports `learned_model`, `opponent_model`, `persuasion_model_v2`, `experiment_agent`, `vanilla_agent`). The actual vanilla control is `shreya_agent_4` (`v0`). | **FALSE.** This inverts the paper's central controlled-experiment claim. Needs correction before submission — as written it misrepresents the experimental design to reviewers. |
| 2 | Architecture (Section 2) describes exactly two strategy files: `vanilla_agent.py` (baseline) and `simple_agent.py` (adaptive), each family dispatched through one of the two | `fleet.py`, `experiment_agent.py` (`STRATEGIES` = v0–v5), `advanced_agent.py` (`STRATEGIES` = v5–v9) | Fleet actually runs **10 named strategy versions (v0–v9)** across 5 separate agent accounts, backed by `experiment_agent.py` + `advanced_agent.py`, plus supporting modules `learned_model.py`, `opponent_model.py`, `persuasion_model_v2.py`, `bargaining_model_v2.py`, `belief_model.py` — none of which are named in the paper. | **Oversimplified to the point of being misleading.** Not necessarily "fabricated," but the two-file picture doesn't match what's running. |
| 3 | Reference: Shapira, Tennenholtz, Reichart, "**Sequential LLM Release Facilitates Manipulation in Regulated Markets**," arXiv:2601.11496 | Web search on the arXiv ID | arXiv:2601.11496 is a real paper by the same three authors, but its actual title is **"The Poisoned Apple Effect: Strategic Manipulation of Mediated Markets via Technology Expansion of AI Agents."** The cited title does not exist. | **FALSE citation title** (right ID, wrong/invented title — classic hallucination pattern). The other two 2026 citations (2605.12411, 2603.17218) checked out with correct titles and authors. |
| 4 | Section 5 table — "Last 100 / Last 500 Games" success counts and avg payoff per family, e.g. Negotiation last-100 avg payoff **$23,581**; Persuasion last-100 avg payoff **$1,938,249** | Recomputed directly from `logs/games.jsonl` (last 100 / last 500 rows per `game_family`, same success definitions the paper states) | Bargaining is roughly in range (98 vs 97 successful; ~$151.8k vs ~$115.8k avg — off but same order). **Negotiation** recomputes to **~$45,948** avg (≈2x paper's number). **Persuasion** recomputes to **~$11.3M** avg on last-100 (≈5.8x paper's number) and **~$8.57M** on last-500 (≈3.8x). | **Does not reproduce from the log file that matches the paper's own description.** Either the numbers were computed from a different/stale snapshot, a filtered subset (e.g., one specific agent instead of the fleet), or they are estimates rather than a direct log query. Needs the exact script + log file + timestamp cited, or the numbers corrected. |
| 5 | "Leaderboard snapshots... negotiation ratings near 1880-1900... persuasion had declined from earlier values near 1800 to roughly the 1400-1500 range" | Searched repo for any leaderboard export/snapshot file | No leaderboard snapshot file exists anywhere in the repo (`logs/` contains only game/learning logs, no rating history). | **Unverifiable from repo contents as-is.** May be accurate (pulled from the live GLEE dashboard at the time) but there's no artifact backing it. Recommend saving a dated screenshot/export next time a number like this is quoted, and citing it explicitly. |
| 6 | Section 5.1 controlled vanilla-vs-adaptive table and Section 5.1 ablation matrix (V0–V5) | Compared against `fleet.py` `VERSION_LABELS` (v0–v9) and existing log files (`v0_games.jsonl` ... `v9_games.jsonl` all already populated with data) | The vanilla-vs-adaptive table is **all `TBD`** — but the underlying data to fill at least a first pass already exists in `logs/v0_games.jsonl` through `logs/v9_games.jsonl`. The ablation matrix also only lists V0–V5, silently dropping V6 (opponent-specific), V7 (contextual-bandit), V8 (final combined), V9 (leaderboard-push) — all of which have logs already. | **Not false, but avoidably incomplete** — this is presented as future work when the raw data to compute it already sits in the repo. |

---

## 4. Assumption-Based vs. Data-Backed Claims

| Claim Type | Example | Basis |
|---|---|---|
| **Data-backed (verified against logs/code)** | Fallback architecture (Fig. 1), family dispatch logic, existence of a JSON temp-file race fixed for `persuasion_profiles_v2.json`, queue-pause retry logic | Confirmed present in `fleet.py` / `game_logger.py` behavior described matches code structure. |
| **Data-backed but numerically wrong** | Section 5 last-100/last-500 stats (Finding #4 above) | Real log file exists, but recomputation doesn't match the quoted numbers. |
| **Assumption / narrative interpretation presented as fact** | "bargaining was already near a local ceiling," "this suggested opponent-pool shift" for persuasion decline | These are reasonable *interpretations* but are stated as conclusions rather than flagged as hypotheses. Should be softened to "we hypothesize" or backed with a specific log query. |
| **Unsupported by any artifact in repo** | Leaderboard rating figures (Finding #5) | No snapshot file exists; likely true but currently just asserted. |
| **Factually incorrect relative to current code** | `sh_agent1` = vanilla control (Finding #1) | Contradicts `fleet.py` and `run_sh_agent1_v9.py` directly. |

---

## 5. Recommendations

1. **Fix Finding #1 immediately** — either relabel which agent is the actual vanilla control throughout the paper (it's `shreya_agent_4` / v0, not `sh_agent1`), or reconfigure the fleet to match what the paper claims before the evaluation window closes. This affects the abstract, intro contributions list, Section 2, and Section 5 — all reference the same wrong mapping.
2. **Add the mandatory "Agent Behavior Analysis" section** — pull concrete examples from `logs/v9_games.jsonl` (or whichever is the real submitted agent) showing designed-for behavior (e.g., concession ramps actually changing offers round-over-round, argument-type selection shifting toward higher-EV frames) vs. designed-against behavior that still slipped through.
3. **Re-derive Section 5's table directly from `logs/games.jsonl` (or the correct current source) with the script saved alongside the paper**, so the numbers are reproducible and match what a reviewer would get running the same query.
4. **Fill Section 5.1 using existing `v0`–`v9` logs** — the data already exists; there's no need to leave the headline comparison table as `TBD`.
5. **Correct or drop the arXiv:2601.11496 citation title** to "The Poisoned Apple Effect: Strategic Manipulation of Mediated Markets via Technology Expansion of AI Agents."
6. **Migrate to the actual NeurIPS 2026 LaTeX template** and re-check the 4-page limit once reflowed — current 7-page custom layout is not directly comparable.
7. **Mention the abandoned LLM-based agent attempts** (`llm_agent.py`, `litellm_agent.py`, `reflective_agent.py`) in the Development Process section — the call explicitly welcomes "what failed" narratives, and this is free content already sitting in the repo.
8. **Fill in author name/affiliation** and confirm the Agent ID for the OpenReview form before submission.
9. Clean up duplicate dead files (`persuasion_model_v2.py` vs `persuasion_model_v2_final.py`, `simple_agent.py` vs `simple_agent_full_v2.py` — byte-identical sizes) so a reviewer trying to reproduce results isn't left guessing which file is canonical.
