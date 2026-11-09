# Revision notes for Submission 32

These notes describe local manuscript changes; nothing has been submitted to OpenReview.

Revised source: `competition_paper_neurips2026_revised.tex`. Reproduce all new tables with `python analysis/audit_glee_logs.py`. Exact values, input hashes, and strata are in `analysis/audit_results.json`; methods and coverage are in `analysis/audit_report.md`.

| Reviewer concern | Revision and evidence | Remaining limitation |
|---|---|---|
| Local logger omits opponent acceptances | Join server and local IDs by account. In the shared window, replays recover 90 baseline and 490 final-account negotiation agreements accepted by the opponent but absent locally. Matched versus missing success rates are reported separately. | The saved exports' download completeness is unverified; some self acceptances are also missing. We do not attribute every gap to opponent termination. |
| Evaluation confounded by opponent pools and run windows | Restrict primary results to the same server start-time interval. Describe results as account-window observations. Withdraw causal strategy and component claims. | Different opponents, account ratings, adaptive state and game configurations remain uncontrolled. |
| Strategy attribution / bundled V9 | The full V9-labelled export overlaps 586 V8-labelled games; 114 have both V8 and V9 local labels. The shared window contains no observed V8-labelled IDs. Explicitly report 2,372 final-account games lacking local strategy evidence. | Absence of an observed V8 label does not authenticate every move as V9. |
| Account and duplicate handling | Exclude 1,669 local V0 rows belonging to another account from the baseline-account coverage denominator. Identify 263 repeated V9 rows and use one server outcome per game. | Local-only IDs and absent terminal results cannot be evaluated through the move logger alone. |
| Strong agreement/payoff tradeoff claims | Replace headline results: negotiation agreement 46.6% versus 51.3%, conditional payoff 52.9K versus 52.4K in the shared window. Withdraw the earlier collapse and increased-conditional-payoff interpretation. | Descriptive changes do not establish improvement or a policy effect. |
| V1–V4 not clean ablations; V4 negotiation changes | Remove the numerical ladder from the revised outcome evidence. Explain that one-component code differences do not create controlled live comparisons, using V4's unchanged negotiation code as the example. | No randomized or matched component evaluation is available. |
| Learned component validation | Report saved training counts, accuracies, majority baselines and log losses. Verify that train_model.py fits and evaluates on the same examples. Explain terminal-selection and counterfactual limitations. | No held-out validation or calibration study is supplied; none is claimed. |
| Reliability claims lack counts | Add per-family timeout, invalid-limit, no-deal and walkaway counts. Distinguish strategic outcomes from software errors. | Counts are game-level and may reflect either participant. Schema and concurrency root causes are not identified. The claim that runtime reliability outweighed strategy is withdrawn. |
| Persuasion remains inconclusive | Report game-level positive-payoff rates and payoff means; explicitly distinguish this proxy from argument-level persuasion success. | No frame-specific causal evidence, historical seeds or controlled opponent pool. |
| Official leaderboard snapshots | State explicitly that none were preserved in the audited artifacts and that rating_delta does not recover rank. | This reviewer request remains unsupported by the supplied files. No rank or objective improvement is invented. |
| Reproducibility / random seeds | Provide an offline standard-library audit, input SHA-256 manifest, deterministic generated tables, and exact window boundaries. State that later seed support cannot reproduce historical play. | Live tournament reruns require historical seeds, runtime revisions, profile state, scheduling and opponents. |
| Incorrect acknowledgments | Revised manuscript thanks the GLEE Competition and IAB Workshop organizers. | Existing revised acknowledgments were already corrected; retained. |
| Formulaic prose | Rewrite abstract, empirical interpretation, limitations and lessons; remove repeated contrast constructions and unsupported superiority language. | — |

Suggested replacement OpenReview TL;DR:

> Auditing a hybrid GLEE agent's server replays overturns an apparent negotiation agreement collapse and exposes logging selection bias and strategy-provenance gaps.

The revised LaTeX source includes `analysis/audit_tables.tex`; keep that file with the manuscript when compiling or packaging it. A PDF build additionally requires a LaTeX engine and the existing `neurips_2026` style, neither of which is available in this workspace.
