# A Reliability-First Adaptive Agent for the GLEE Competition

*[Author Name], [Affiliation]*

## Abstract

We describe an autonomous agent for the GLEE Competition, which evaluates agents in three language-based economic game families: bargaining, negotiation, and persuasion. Our design follows a reliability-first principle: because invalid actions, timeouts, and abandoned games can create severe scoring penalties, every strategic component is wrapped in a deterministic, schema-compatible fallback. The submitted system combines simple economic heuristics, opponent-behavior profiling, round-based concession schedules, lightweight learned calibration from logged games, and persuasion-specific payoff tracking. We also maintained a vanilla control agent during the final evaluation window to distinguish true strategic improvement from leaderboard noise. Our central finding is that the value of adaptation is strongly game-dependent: bargaining was already difficult to improve beyond a simple baseline, negotiation was the most promising target for explicit economic reasoning and concession, and persuasion remained high-variance despite payoff-aware argument selection. Robust competition performance depended less on adding complexity and more on validity, fast response, controlled comparison, and careful failure analysis.

## 1. Introduction

GLEE (Games in Language-based Economic Environments) evaluates agents in sequential two-player economic interactions where natural-language communication and structured economic actions are both important. The competition contains three families: bargaining over a fixed surplus, buyer-seller negotiation over an indivisible good, and persuasion under asymmetric information. Agents interact through the official API and are ranked by family-level scores that reflect payoff relative to other players in comparable roles and configurations.

This setting creates a practical challenge that is easy to underestimate. A strategy can be theoretically appealing but competitively harmful if it occasionally fails to answer, emits an invalid action, or deadlocks against another static agent. Our agent was therefore built around a conservative engineering claim: first never lose a game mechanically, then improve payoff through adaptation. The system evolved through several live experiments during the competition window, including reverted persuasion variants, negotiation hold-out tests, and a final split where `sh_agent1` was kept as a vanilla control while the remaining agents ran the current adaptive strategy.

The resulting paper is not a clean laboratory ablation study. It is a competition-system report based on live online play, noisy opponent pools, and multiple agent slots. This is still enough for a GLEE competition paper because the track asks participants to describe their agent, approach, and findings, including negative results. The strongest scientific value is in the engineering and empirical lessons: which strategies were stable, which were noisy, which failure modes mattered most, and why raw game-level payoff can diverge from leaderboard rating.

## 2. Agent Architecture

Our implementation uses the official Python GLEE SDK. Each incoming game is dispatched by family to a deterministic strategy function. The current fleet has two relevant variants:

- `vanilla_agent.py`: a simple baseline agent. In bargaining it proposes an even split and accepts offers worth at least 40% of the pot. In negotiation it accepts profitable offers and otherwise counters from its own valuation. In persuasion it recommends the product as seller and buys as buyer when expected value exceeds price.
- `simple_agent.py`: the adaptive strategy. This extends the vanilla logic with opponent modeling, concession schedules, discount-factor adjustments in bargaining, fair-price logic under complete information in negotiation, learned threshold calibration, and payoff-aware persuasion argument selection.

During the final data-collection phase, we configured `sh_agent1` as a separate vanilla-control agent and routed its outcomes into separate logs:

```text
logs/sh_agent1_games.jsonl
logs/sh_agent1_learning.log
```

The remaining agents run the adaptive strategy and write to the shared fleet log. This split is important because leaderboard changes alone were not reliable enough to guide decisions. Several earlier changes appeared beneficial in one batch and harmful in the next, especially in persuasion.

## 3. Strategy Design

### 3.1 Bargaining

The bargaining strategy began from a robust baseline: propose a 50/50 split and accept any offer giving at least 40% of the pot. Later variants tested a more self-favoring anchor and learned threshold adjustments, but bargaining scores did not consistently improve on the leaderboard. This suggested that bargaining was already near a local ceiling for our account and opponent pool.

The adaptive strategy still includes two mechanisms that are defensible. First, it uses opponent behavior to classify counterparts as aggressive, cooperative, random, or unknown based on concession patterns and cross-game outcomes. Against aggressive opponents, it concedes faster and lowers the acceptance threshold; against cooperative opponents, it can ask for slightly more. Second, it uses a round-based concession ramp to avoid static-offer deadlocks. This was motivated by observed games where fixed offers failed to converge.

Given the final 30-hour window, our operational decision was to stop making major bargaining changes. Bargaining had high success in recent logs, while further tuning risked lowering a stable score.

### 3.2 Negotiation

Negotiation was our main improvement target. The vanilla strategy accepts any profitable offer and otherwise proposes a price based on the agent's own valuation. The adaptive strategy adds three ideas.

First, when complete information reveals both players' valuations, the agent computes the fair surplus-splitting price and avoids accepting a merely profitable offer that gives almost all surplus to the opponent. This is the most defensible strategy change because it directly uses information exposed by the environment.

Second, the agent uses a concession schedule. Earlier versions could repeat static counteroffers for many rounds, creating long games and no-deal outcomes. The updated strategy gradually moves counteroffers toward a still-profitable closing price over time. This is not just parameter tuning; it fixes an observed failure mode.

Third, the agent blends in a lightweight learned model trained from logged outcomes. The model predicts agreement probability from final offer terms and round number. Because the log does not contain full counterfactual trajectories, this model is used only as a small calibration signal, not as a replacement policy.

### 3.3 Persuasion

Persuasion was the highest-variance family. The seller can earn very large payoffs when buyers accept, but many games produce zero payoff. We tested multiple persuasion ideas, including calibrated lying formulas and payoff-aware argument selection. The calibrated lying variants were reverted because their effects changed sign across successive live batches.

The current adaptive persuasion model tracks six argument framings: economic, fairness, safety, social, convenience, and long-term. Rather than selecting the message type by raw acceptance rate, it scores each framing by estimated expected payoff with a penalty for payoff variance. This matters because in persuasion, high acceptance rate does not always imply high payoff.

However, persuasion remained unstable. Recent leaderboard drops suggest either an opponent-pool shift, insufficient adaptation speed, or damage from earlier operational failures. We therefore treat persuasion results cautiously and avoid overclaiming that the adaptive persuasion strategy improved performance.

## 4. Reliability and Failure Handling

The most important engineering lesson was that mechanical reliability is a competitive strategy. During live runs, we found a multiprocessing race in profile persistence: multiple agent processes wrote to the same temporary JSON filename before replacing `logs/persuasion_profiles_v2.json`. This caused intermittent `FileNotFoundError` crashes during persuasion decisions. The fix was to make temporary filenames process- and thread-specific before atomic replacement.

We also observed server-side queue pauses after agents timed out in several consecutive games. To recover from this safely, the fleet runner now parses the server retry time, sleeps until the pause expires, and then rejoins the queue automatically. This prevents a paused agent from silently dying during the final evaluation window.

These details are worth reporting because they affected real competition performance. In GLEE, a strong strategy that occasionally crashes can be worse than a simple strategy that always replies.

## 5. Evaluation

We evaluate using local logs collected from live competition play. For bargaining and negotiation, success is `outcome == "agreement"`. For persuasion, success is defined as positive payoff because completed persuasion games can still yield zero payoff.

We separate evaluation into two layers. The first is game-level behavior: success rate, payoff, median payoff, rounds per game, failure rate, timeout rate, and family-specific diagnostics such as final price or argument type. The second is the competition metric: leaderboard rating. This distinction matters because an intervention can improve a local metric while reducing leaderboard score after GLEE's role, configuration, opponent, and reference-pool adjustments.

Recent shared-log performance before the final control split was:

| Family | Last 100 Games | Last 500 Games | Interpretation |
|---|---:|---:|---|
| Bargaining | 97 successful, avg payoff $115,848 | 491 successful, avg payoff $142,620 | Strong and stable; avoid further risky tuning. |
| Negotiation | 53 successful, avg payoff $23,581 | 280 successful, avg payoff $26,818 | Moderate; best target for improvement. |
| Persuasion | 31 positive-payoff, avg payoff $1,938,249 | 157 positive-payoff, avg payoff $2,260,787 | High variance; recent decline requires caution. |

Leaderboard snapshots around the final run showed negotiation ratings near 1880-1900 for several agents, while persuasion had declined from earlier values near 1800 to roughly the 1400-1500 range. This created an important evaluation warning: local metrics such as average payoff, total payoff, and agreement rate are not identical to the GLEE leaderboard objective. A strategy can increase raw payoff in the games we log while still losing rating if it performs worse after role, configuration, opponent, or reference-pool adjustment. For example, a lower-agreement strategy with larger occasional wins can look attractive in local average-payoff summaries but score worse than a stable baseline in the competition rating.

This mismatch motivated the final experimental setup: keep `sh_agent1` as a vanilla control and use the remaining agents to test whether the adaptive strategy still outperforms baseline under the current opponent pool. The control is especially important because the original/simple GLEE-style baseline had previously reached ratings near 1900 in some families. If a more complex variant falls to 1400-1500, the right interpretation is not that the local logs are useless, but that the variant must be evaluated against the leaderboard objective rather than against local payoff alone.

### 5.1 Controlled Ablation Plan

The main experiment for the final window is vanilla versus adaptive under the same live opponent pool. The core table we aim to fill is:

| Family | Strategy | Games | Success | Avg payoff | Median payoff | Avg rounds | Rating change |
|---|---|---:|---:|---:|---:|---:|---:|
| Bargaining | Vanilla | TBD | TBD | TBD | TBD | TBD | TBD |
| Bargaining | Adaptive | TBD | TBD | TBD | TBD | TBD | TBD |
| Negotiation | Vanilla | TBD | TBD | TBD | TBD | TBD | TBD |
| Negotiation | Adaptive | TBD | TBD | TBD | TBD | TBD | TBD |
| Persuasion | Vanilla | TBD | TBD | TBD | TBD | TBD | TBD |
| Persuasion | Adaptive | TBD | TBD | TBD | TBD | TBD | TBD |

We also define the component-level ablation matrix for cleaner follow-up analysis:

| Variant | Opponent model | Concession | Learned model | Payoff-aware persuasion | Purpose |
|---|---|---|---|---|---|
| V0 Vanilla | No | No | No | No | Reference point. |
| V1 Opponent | Yes | No | No | No | Test adaptation to opponent behavior. |
| V2 Concession | No | Yes | No | No | Test deadlock prevention. |
| V3 Learning | No | No | Yes | No | Test learned threshold calibration. |
| V4 Persuasion | No | No | No | Yes | Test payoff-aware argument selection. |
| V5 Full | Yes | Yes | Yes | Yes | Integrated production strategy. |

Two family-specific analyses are especially important. For negotiation, we compare static counters, linear concession, and tiered concession using agreement rate, average rounds, payoff, and no-deal frequency. For persuasion, we compare argument selection by buy probability, expected payoff, and expected payoff minus risk penalty, reporting attempts, buy rate, average payoff, median payoff, and payoff variance per argument type. This directly tests our observation that high acceptance rate does not necessarily imply high payoff.

## 6. Discussion

Our experiments are sufficient for a competition paper, but not for a strong causal claim about every component. The strongest supported claims are:

1. A vanilla baseline is difficult to beat in bargaining, where simple fair splits and conservative acceptance already perform well.
2. Negotiation benefits most from environment-aware rules, especially complete-information fair-price reasoning and concessions that prevent deadlock.
3. Persuasion is noisy and brittle; variants that looked promising in one batch did not consistently survive later batches.
4. Reliability failures are not merely implementation details. They directly affect score because timeouts and crashes change the agent's ability to remain in the queue and complete games.

The main limitation is that several components changed together. The adaptive agent combines opponent modeling, concession schedules, discount-factor adjustment, anchoring, learned thresholds, payoff-aware persuasion, buyer risk adjustment, and exploration. If the score changes after deploying the full system, we cannot tell from the final score alone which component helped and which hurt. This is the standard motivation for ablation experiments.

Our final data-collection setup therefore reframes the remaining evaluation as an ablation-style comparison rather than a single full-system test. Separating vanilla-control logs from adaptive-agent logs fixes part of the earlier limitation. A cleaner future version would also log the exact strategy version, agent name, timestamp, and commit hash for every completed game. For game-family analysis, the most useful additional fields are initial offer, final offer, round, accept/reject decision, payoff, and opponent identity for bargaining; seller/buyer role, valuation, price path, final price, and failure reason for negotiation; and argument type, price, quality, bought/not bought, and realized payoff for persuasion.

The persuasion experiment has an additional selection-bias limitation. If one argument type has more attempts and higher payoff, that does not prove the argument type is intrinsically better; it may have been selected more often against easier opponents or more favorable prices. To reduce this confounding, the agent should force early balanced exploration across economic, fairness, safety, social, convenience, and long-term arguments before exploiting the best-scoring frame.

Finally, we summarize the main observed failure modes:

| Failure mode | Cause | Fix or mitigation | Paper value |
|---|---|---|---|
| Static-offer deadlock | Repeated bargaining or negotiation counters did not move over rounds. | Round-based concession schedule. | Shows why time-awareness matters. |
| Queue pause | Consecutive timed-out turns made GLEE pause queue joins. | Parse retry time and automatically rejoin after pause. | Shows reliability affects score. |
| Persuasion crash | Shared JSON temp filename across processes caused profile-save races. | Process/thread-specific temp files before atomic replace. | Concrete engineering lesson. |
| Zero-payoff persuasion | Buyer rejection or unfavorable accepted trades. | Payoff-aware argument tracking and buyer risk adjustment. | Motivates payoff rather than acceptance-only modeling. |

## 7. Conclusion

We built a reliability-first adaptive agent for GLEE and tested it through live competition play. The final system combines deterministic valid actions, opponent profiling, concession schedules, learned calibration, and persuasion payoff tracking. Our experience suggests that in language-based economic-agent competitions, robustness and disciplined experimentation matter as much as sophisticated reasoning. The best use of the remaining competition window is not broad rewrites, but controlled comparison: preserve a vanilla agent, run adaptive agents separately, and make strategy decisions only when fresh batches show stable improvement.

## References

Shapira, E., Madmon, O., Reinman, I., Amouyal, S. J., Reichart, R., and Tennenholtz, M. GLEE: A Unified Framework and Benchmark for Language-based Economic Environments. arXiv:2410.05254.

Shapira, E., Tennenholtz, M., and Reichart, R. Predicting Decisions of AI Agents from Limited Interaction through Text-Tabular Modeling. arXiv:2605.12411.

Shapira, E., Tennenholtz, M., and Reichart, R. Alignment Makes Language Models Normative, Not Descriptive. arXiv:2603.17218.

Shapira, E., Tennenholtz, M., and Reichart, R. Sequential LLM Release Facilitates Manipulation in Regulated Markets. arXiv:2601.11496.
