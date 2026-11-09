# Response to Reviewers

Paper: *Selective Adaptation and Reliability Lessons from a Hybrid GLEE Agent* (Submission 32)
Revised manuscript: `competition_paper_neurips2026_revised_v3.tex` (5 pages)
Verification: clone the repository and run `python3 analysis/check_paper.py` (Python 3.10+, standard library only). It reads the per-game data in `paper_data/` (all 75,258 platform games, without message text) and the agent source, and prints PASS/FAIL for every data claim and strategy description in the paper (276 checks, all passing). The raw platform downloads are not public because their replays contain other teams' messages; the account owner rebuilds `paper_data/` from them with `bash analysis/run_api_audit.sh`, and `analysis/independent_check.py` recomputes Figures 3 and 4 from the raw downloads alone.

## Summary of the main change

We used the official platform game histories with replays for all five accounts (75,258 games) and rebuilt the evaluation from that data. The local logger was not used for any headline number. The new evidence reverses the main negotiation claim of the first submission:

| | First submission (local logs, all negotiation games) | Revised (platform records, W3, feasible games) |
|---|---|---|
| V0 negotiation agreement | 64.6% | 81.1% |
| V9 negotiation agreement | 30.6% | 90.1% |
| V9 vs V0 normalized negotiation payoff | not measured | +0.055, p = 0.002 |

The two columns use different game sets. For a like-for-like comparison over the same hours and all negotiation games, local logs give V0 63.4% vs V9 33.3%, while platform records give 46.6% vs 51.3%. Because V9 counter-offers more often, more of its games ended on the opponent's move, and the logger never recorded those games.

## Reviewer bVML

| Concern | Response | Where |
|---|---|---|
| Changing opponent pool confounds V0 vs V9 | The main comparison is now window W3 (29 Aug 12:28–18:48 UTC), where V0, V1, V3, V7 and V9 ran at the same time on five accounts; the W1 ladder (28 Aug 10:58–15:30 UTC) likewise runs V0 alongside V1–V4. We no longer compare across windows. Accounts running the identical vanilla rule differ by at most 1.0 point in W3 but by up to 7.1 points in W1 (persuasion), so account effects are reported, not assumed away. Limitations states that the records give each opponent's type but not its identity or rating, so we could not observe how the platform paired agents; 122 of the 75,258 games (0.2%) were against humans. | §1, §4, §5, Figure 3 |
| V9 bundles changes | Stated explicitly: V9's gains are attributed only to the full policy, and attributing them to single components would need one-at-a-time tests. The bargaining result notes that the gain cannot be credited to the deadline schedule alone. | §4, §5 |
| Logger records only games ending on our move | Quantified: in both windows 45–51% of games ended on the opponent's move, and the logger recorded none of them. In W3 it also lost 19–38% of own-move endings. All results now use platform records. | §4 "Logger bias", Figure 2 |
| V1–V4 attribution; V4 negotiation shift | The ladder is recomputed from platform records with V0 running in the same window. No single negotiation change differs from V0 (all p ≥ 0.10). V4 runs V0's negotiation code (`experiment_agent.py` falls through to `vanilla_agent.strategy`), so its 3.0-point gap (p = 0.23) is a placebo difference that measures drift between accounts. | §4, Figure 4 |
| Learned component under-evaluated | Added held-out evaluation: 0.907 accuracy on the chronologically last 20% (majority baseline 0.534) and 0.796 on a random 20% split (baseline 0.503), with the caveat that the rows come from the biased logger. The model predicts agreement well, but V3 accepted 160 money-losing deals (60 in W1, 99 in W3; V8, which reused the rule, 153 more) because its 50%-agreement point is negative in rounds 1–8. The paper gives the fix: choose the threshold that maximizes expected profit (agreement probability × margin), floored at zero margin. The method section now states that V9 clips the learned margin to [0, 0.12] and accepts and counters only at a profit, which is why V9 accepted no losing deals. | §2, §4 "Learned threshold" |
| Reliability failures not quantified | In W1, at most 7 games per version contained an invalid move; in W3, 14–23% of bargaining games did. Over 98% were an accept/reject payload sent when the server expected an offer, mostly within a second of a valid decision, which fits a decision re-sent from stale game state. Run logs record a server rejection for 94.5% of affected V9 games and 64% of V0 games, and rejections in 4,126 games overall (3,394 "It is not your turn", 651 offer-format, 393 "Game is not active"). Games with such a move reached agreement 73–85% of the time against 96–100% without, and they account for 58–94% of failed bargaining games; the paper states this is an association, since longer games give more chances for such a move. Timeout and invalid_moves_exceeded counts are also given. | §4 "Reliability behavior" |
| Persuasion inconclusive | Reported as a null result. V9 differs from V0 by −0.9 points (p = 0.66); V7 had lower success (p < 0.001) with no significant payoff change. Identical vanilla persuasion arms span 0.8 points in W3 but 7.1 points in W1, where V2 and V0 (same rule) differ by 6.3 points at p = 0.015, so differences of this size can come from account effects alone. | §4 "Persuasion behavior" |
| Formulaic writing ("X rather than Y") | Rewrote the paper; that construction no longer appears. | throughout |

## Reviewer EDRX

| Request | Response | Where |
|---|---|---|
| Bound the selection bias (share of games ending on opponent acceptance) | Per-version share of opponent-ended games, logger coverage, and local-vs-platform agreement are shown side by side; local logs overstate agreement for every version except V9, reversing the V0–V9 comparison. | Figure 2 |
| Official leaderboard snapshots | Snapshots were not saved during the competition. The paper gives account 1's public leaderboard standing in October 2026 (rank 91, rating 1715.7; that account ran V0, V8 and V9) and the official per-game `rating_delta` by version in W1 and W3. In W3, V9 lost the least rating per game (−0.016) and V3 the most (−0.124). Rating is treated as supporting evidence only. | §4 "Official rating" |
| Wrong acknowledgments (MediaEval) | Acknowledgments now refer only to the GLEE Competition and the IAB Workshop at NeurIPS 2026. | Acknowledgments |
| Seeds for stochastic persuasion | Stated that V7–V9 persuasion decisions cannot be replayed exactly; aggregate results come from platform records and do not depend on replays. A `GLEE_RANDOM_SEED` option has since been added to the fleet and is written to every log row. | §5 |

## Other changes in this version

- The method section (§2) now gives V9's actual parameters, each checked against `advanced_agent.py`: bargaining opens at 53.5% and accepts at 42.5% of the pool, with a 10% weight on the learned threshold and up to 8 points of deadline concession (34% in the final round); negotiation uses a 52% surplus-share threshold when the opponent's value is visible and otherwise a 0.75/0.25 blend of a hand-set and a clipped learned margin; persuasion plays the best frame 97% of the time after four attempts against an opponent, and the buyer weights possible losses 1.5 times.
- V9's buyer code also contains a rule meant to distrust sellers caught lying. Platform histories never contain the fields it reads, so it never triggered; the paper does not claim it.
- The significant V9 results survive a Bonferroni correction for the 24 W3 tests against V0 (α ≈ 0.0021).

## Known remaining limitations

- 1,892 games from the V8/V9 overlap on account 1 (1,778 unlogged, 114 logged under both versions) and 1,809 games started during restarts or at version boundaries are left unassigned. W1 and W3 contain no overlaps and only 8 unassigned games.
- Matchmaking is platform-controlled, so W1 and W3 are contemporaneous but not randomized.
- V9 bundles changes in all three families; separating them would need new one-at-a-time runs.
- Leaderboard snapshots were not saved; `rating_delta` is used as a proxy.
