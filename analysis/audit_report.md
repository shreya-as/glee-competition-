# Saved GLEE log audit

Reproduce: `python analysis/audit_glee_logs.py` (Python standard library only).

The exported filenames identify accounts/windows, not authenticated strategy versions. All official IDs, roles, families and results agree between history and replay. Source hashes are in `audit_results.json`.

Shared window: games **started** from 2026-08-29T12:26:58.465813+00:00 to 2026-08-29T18:48:53.297024+00:00, inclusive. Local timestamps mark completion; membership is therefore joined by game ID, not timestamp equality.

## Full-export coverage (same account and version-labelled local file)

| Label | API games | Local rows | Local unique | Shared | API only | Local only |
|---|---:|---:|---:|---:|---:|---:|
| v0 | 3265 | 1103 | 1103 | 1103 | 2162 | 0 |
| v9 | 7787 | 3266 | 3003 | 3000 | 4787 | 3 |

The V0 local file also contains 1,669 rows from sh_agent1. Those rows belong to another account/window and are excluded from this coverage denominator.

## Version identity

- `official_v9_ids_with_v8_label`: 586
- `official_v9_ids_with_both_v8_v9_labels`: 114
- `aligned_v9_ids_with_v8_label`: 0
- `aligned_v9_ids_with_v9_label`: 1513
- `aligned_v9_ids_without_local_version_evidence`: 2372
- `last_v8_local_timestamp`: 2026-08-29T17:39:36+0545

The shared window avoids all observed V8-labelled games. Unmatched official games still have no independent strategy label. Absence of observed V8 overlap does not prove exclusive V9 execution.

## Shared-window outcomes

Payoffs are game units. Success is agreement for bargaining/negotiation and strictly positive own payoff for persuasion; negative and zero persuasion payoffs remain in the unconditional mean.

| Family | Label | N | Success % | Mean payoff | Mean payoff given success | Matched N | Matched success % | API-only N | API-only success % |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bargaining | v0 | 1077 | 92.29 | 144185.870 | 156225.535 | 297 | 84.51 | 780 | 95.26 |
| bargaining | v9 | 1293 | 96.37 | 159616.714 | 165637.569 | 619 | 94.67 | 674 | 97.92 |
| negotiation | v0 | 1067 | 46.58 | 24633.760 | 52885.758 | 427 | 63.23 | 640 | 35.47 |
| negotiation | v9 | 1341 | 51.30 | 26870.619 | 52374.273 | 383 | 33.16 | 958 | 58.56 |
| persuasion | v0 | 1115 | 53.90 | 2927387.534 | 5492893.344 | 378 | 41.27 | 737 | 60.38 |
| persuasion | v9 | 1251 | 52.92 | 3111493.765 | 5904243.656 | 511 | 36.79 | 740 | 64.05 |

## Replay acceptance attribution in shared window

| Family | Label | Self, matched | Opponent, matched | Self, API only | Opponent, API only |
|---|---|---:|---:|---:|---:|
| bargaining | v0 | 251 | 0 | 154 | 589 |
| bargaining | v9 | 586 | 0 | 128 | 532 |
| negotiation | v0 | 270 | 0 | 137 | 90 |
| negotiation | v9 | 127 | 0 | 71 | 490 |

Bargaining actor is the other player from the final proposer; negotiation uses the final decided_by field. Missing self acceptances show that opponent termination is only one coverage mechanism.

## Local result integrity

- v0: 43 shared IDs have no local terminal result; 0 IDs have conflicting terminal outcome/payoff/role. Duplicate local rows are collapsed by ID only for coverage; all performance metrics use server results.
- v9: 101 shared IDs have no local terminal result; 0 IDs have conflicting terminal outcome/payoff/role. Duplicate local rows are collapsed by ID only for coverage; all performance metrics use server results.

## Reliability outcomes in shared window

| Label | Family | Outcomes |
|---|---|---|
| v0 | bargaining | {"agreement": 994, "invalid_moves_exceeded": 1, "no_deal": 73, "timeout": 4, "walked_away": 5} |
| v0 | negotiation | {"agreement": 497, "no_deal": 510, "timeout": 9, "walked_away": 51} |
| v0 | persuasion | {"completed": 1106, "timeout": 9} |
| v9 | bargaining | {"agreement": 1246, "no_deal": 42, "walked_away": 5} |
| v9 | negotiation | {"agreement": 688, "invalid_moves_exceeded": 2, "no_deal": 569, "timeout": 3, "walked_away": 79} |
| v9 | persuasion | {"completed": 1240, "timeout": 11} |

Timeout/invalid-move outcomes are game-level observations; this audit does not assign fault to our agent, infer schema root causes, or estimate concurrency failure rates.

## Saved model metrics

These are training-set metrics: train_model.py fits and evaluates on the same examples. They provide no held-out validation.

```json
{
  "bargaining": {
    "accuracy": 0.989,
    "baseline_accuracy": 0.956,
    "log_loss": 0.0315,
    "n": 7992,
    "positive_rate": 0.956
  },
  "negotiation": {
    "accuracy": 0.839,
    "baseline_accuracy": 0.519,
    "log_loss": 0.466,
    "n": 5969,
    "positive_rate": 0.519
  }
}
```

## Remaining evidence gaps

- Download pagination/completeness is unverified; counts describe saved exports.
- Shared clock time does not control opponent matching, account ratings, configurations or adaptive state. Config/role/scale summaries are in audit_results.json; they are descriptive, not causal adjustments.
- Historical random seeds and exact runtime revisions were not preserved in these game rows. Later runner seed support cannot reconstruct them.
- No official leaderboard snapshot/rank is present in the audited artifacts. Per-game rating_delta is not a leaderboard snapshot.
