> **Update, 15 October 2026:** The 393-game sample notes below describe the earlier artifact state. A new audit uses 3,265 baseline-account and 7,787 final-account server games. It identifies mixed-account V0 coverage, duplicate V9 rows, and V8/V9 label overlap. See [the reproducible audit](analysis/audit_report.md) and [reviewer response](analysis/reviewer_response.md) for current evidence; the revised manuscript uses a shared late-run window. The older claim that only a 393-game replay file is available is superseded.

# OpenReview Reviewer Concerns Reference

Audit date: 2026-10-07

Purpose: keep the reviewer concerns in one Markdown file so future edits can point back here. The V9 data is mentioned only where it helps answer a concern.

## Decision

| Item | Text |
|---|---|
| Decision | Reject, encouraged to revise and resubmit |
| Main reason | The failure analysis is useful, but the empirical comparisons are too confounded to support stronger conclusions. |

## Reviewer Concern Table

| Concern | Why It Matters | Evidence Currently Available | Is Current Evidence Enough? | Revision Action |
|---|---|---|---|---|
| Evaluation is heavily confounded by changing opponent pool. | V0 and V9 may have faced different opponents/configurations, so differences may not be caused by strategy alone. | Local logs: `logs/v0_games.jsonl`, `logs/v9_games.jsonl`. Replay-backed V9 sample: `final_logs/final_v9_sh_agent1_history_with_replays.json`. | No. | Reword claims as live-run observational evidence, not clean causal proof. Add opponent/config stratification if possible. |
| V9 bundles many changes, so attribution is unclear. | The paper cannot say which V9 component caused the outcome shift. | Version ladder logs exist for V0-V9. | Partly. | Say V9 is a bundled policy. Use V1-V4/V5-V8 only as diagnostic evidence, not definitive attribution. |
| Logger records only games ending on our move. | Missing opponent acceptances may bias success/payoff estimates. | `game_logger.py` limitation is real. API history/replay data can help because it includes server-side game history. | Not yet. Current V9 replay file is only 393 late-window games. | Download full-window API history/replays, compare API IDs vs local logger IDs, and quantify missing local coverage. |
| Missing opponent acceptances could directly bias success/payoff estimates. | If opponent-accepted games are omitted, agreement rate and payoff may be undercounted or distorted. | Current V9 downloaded history has 393 games, 254 not in local V9 log. | Partly, but sample is too small. | Compute full-window "API history vs local logger" table: in both, API-only, local-only, by family and outcome. |
| V1-V4 do not give clean attribution. | Reviewer notes V4 only changes persuasion, yet negotiation payoff shifts, showing runs are noisy/confounded. | Local version logs. | No, if framed as clean ablation. | Rename as "diagnostic version ladder" or "live ablation-style runs," not controlled ablations. |
| Learned component is under-evaluated. | A learned threshold model needs validation beyond final outcomes. | `learned_model.py`, `train_model.py`, learned logs/models. | Probably not. | Add small validation summary: training rows, features, held-out/calibration metric if available; otherwise state limitation. |
| Reliability claims are under-evaluated. | Paper says schema/concurrency/reliability mattered, but reviewers want counts. | Logs may contain crashes, timeouts, no-deals, invalid actions if recorded. | Not enough yet. | Add quantified failure table: timeout count, invalid/schema errors, queue pauses, crashes, fixes. |
| Persuasion remains inconclusive. | Persuasion results are noisy and stochastic, so strong claims are unsupported. | V0/V9 persuasion logs and replay sample. | Enough only for cautious negative result. | Say persuasion was inconclusive; avoid claiming improvement. Report stochasticity and variance. |
| Stochastic persuasion seeds were not logged. | Results cannot be exactly reproduced. | Current logs do not include seeds for V7-V9 persuasion choices. | No. | State non-reproducibility clearly; add seed logging for future runs if revising code. |
| Need official leaderboard snapshots. | Local payoff metrics may not match competition objective. | No leaderboard snapshot file found in repo. | No. | Add screenshot/export if available; otherwise explicitly say leaderboard snapshots were not preserved. |
| Acknowledgments may be wrong. | Reviewer said acknowledgments thanked a different competition. | Need check final manuscript text. | Unknown from this file. | Verify acknowledgments and correct to GLEE/IAB only. |
| Writing feels formulaic. | Reviewer noticed repeated sentence patterns like "X rather than Y." | Manuscript text. | Fixable. | Do a prose pass to vary repeated constructions and shorten claims. |

## What The Current V9 Replay File Can Support

`final_logs/final_v9_sh_agent1_history_with_replays.json` can support:

| Use | Okay? | Note |
|---|---|---|
| Qualitative examples of V9 behavior | Yes | It has complete `history` and `replay` payloads. |
| Checking replay structure and exact trajectories | Yes | `history.game_id` matches `replay.game_id` for all 393 games. |
| Showing API history can recover games missing from local logger | Partly | 254 of 393 history IDs were not in `logs/v9_games.jsonl`. |
| Replacing full V9 paper metrics | No | It covers only the final late-window sample. |
| Fully answering logger-selection-bias concern | No | Need full-window API history/replay download. |

## Current Data Facts To Remember

| Fact | Value |
|---|---:|
| Current V9 replay-backed file size | 393 games |
| Current V9 replay-backed local-time range | 2026-08-29 23:52:49 to 2026-08-30 00:33:53 |
| Full local V9 run local-time range | 2026-08-29 12:12:44 to 2026-08-30 00:34:36 |
| Local V9 log rows in requested window | 3264 |
| Shared IDs between current V9 replay file and local V9 log | 139 |
| Current conclusion | useful sample, not full evidence base |

## Best Revision Framing

Use cautious language:

| Strong Claim To Avoid | Safer Claim |
|---|---|
| V9 caused negotiation agreement collapse. | In the live logs, the bundled V9 policy coincided with lower negotiation agreement and higher payoff conditional on agreement. |
| V1-V4 prove which component caused the result. | V1-V4 are diagnostic live runs that suggest, but do not prove, where the failure came from. |
| Local logs fully measure performance. | Local logs are incomplete because they miss games ending on the opponent's move; API history/replay is needed to bound that bias. |
| Persuasion improved. | Persuasion remained noisy and inconclusive. |

## Next Evidence Table To Build

After downloading full-window API history/replays, build this table:

| Family | API History Games | Local Logger Games | IDs In Both | API-Only | Local-Only | API Success Rate | Local Success Rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Bargaining | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Negotiation | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Persuasion | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
