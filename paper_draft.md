# A Hybrid Rule-Based / LLM Agent for the GLEE Competition

*[Author name], [Affiliation]*

## Abstract

We describe an agent submitted to the GLEE Competition (Games in Language-based
Economic Environments), covering all three game families — bargaining,
negotiation, and persuasion. The agent has two interchangeable strategy
backends behind a common GLEE SDK integration: a fast, deterministic
heuristic strategy layered with opponent modeling, Bayesian belief tracking,
and offline-learned calibration, and an LLM-driven strategy that reasons over
each game's natural-language prompt with the same opponent signals injected
into its prompt. Both are unified by a shared safety layer — every proposed
move is validated against the game's exact action schema before submission,
and any invalid, malformed, rate-limited, or timed-out LLM response falls
back to the heuristic strategy — so the agent never risks the platform's
harshest penalty, an abandoned game scored at the 5th percentile, regardless
of provider reliability. In practice the heuristic backend, not the LLM
backend, is what we currently run across all five registered agent slots: an
internal A/B test between a heuristic variant and a heuristic control settled
that comparison, and rate-limit-driven fallback meant the LLM backend was
exercised far less than intended during evaluation anyway. We report the
strategy logic per game family in the detail it's actually implemented, the
engineering history behind two live deadlock bugs and their fixes, and
self-play results — 3,200+ bargaining games, 3,100+ negotiation games, and
2,900+ persuasion games — collected during the evaluation window, including a
recency-bucketed trend analysis per family.

## 1. Introduction

The GLEE Competition matches autonomous agents (and humans) against each
other in three canonical economic games — bargaining, negotiation, and
persuasion — via a REST API. Agents are ranked by self-gain, computed as a
configuration-and-opponent-adjusted percentile against a large reference
pool. A defining feature of the competition's scoring is asymmetric downside
risk: failing to respond, or submitting five consecutive invalid moves,
force-closes the game and scores it at the 5th percentile — a far worse
outcome than any real negotiated deal. This shapes agent design as much as
the choice of strategy itself: an agent that plays a mediocre but
always-valid move outperforms one that occasionally plays a brilliant move
and occasionally crashes.

Our submission is built around this constraint. We treat "never abandon a
game" as a hard requirement enforced in code, and layer strategic
sophistication — opponent modeling, belief tracking, offline-learned
calibration, and (optionally) an LLM call — on top of it rather than relying
on any one of those to succeed.

## 2. Agent Architecture

The agent has two interchangeable strategy backends behind the same GLEE SDK
integration:

**Heuristic strategy (`simple_agent.py`).** A deterministic function per game
family (Section 3) that requires no external calls, cannot time out, and
always returns a well-formed action. It isn't a static rulebook: each
decision reads opponent-behavior signals from `opponent_model.py`
(aggressive/cooperative/random classification), a per-round concession
schedule, a discount-factor edge, and a small blended nudge from an
offline-trained logistic regression (`learned_model.py`). This is both the
fallback for the LLM-driven agent and, in the current deployment, the
strategy actually run on all five registered agent slots — see "which
backend is live," below.

**LLM-driven strategy (`litellm_agent.py`, `reflective_agent.py`).** Each
pending move is turned into a prompt containing the game's own
natural-language description (`prompt`), the full machine-readable
`game_state` (including per-round history where available), the exact
expected action schema (`valid_actions`), and the same opponent-model signal
the heuristic backend uses (for persuasion, a cross-game reputation ledger of
past recommendations plus per-argument-type success rates instead). The
model is instructed to reply with only a JSON action object. `reflective_agent.py`
adds two further pieces from our internal GLEE strategy notes: a Bayesian
posterior over the opponent's hidden true valuation
(`belief_model.py`, negotiation only, and only when `complete_information` is
off — the one case something is actually hidden) injected into the prompt,
and a four-question chain-of-thought "Reasoning:" preamble the model must
work through — target outcome, what the opponent will infer, their likely
response, and a re-check against our own reservation value — before giving
its action, run inline in the same completion rather than as a second
chained call. We use `litellm` as a provider abstraction layer so the
backend model is a single configuration string; this decouples strategy
design from any one vendor's SDK or pricing/availability.

**Which backend is live.** `fleet.py`, which drives all five of our
registered agent keys, currently points every key at the heuristic backend.
Earlier in the project it split the fleet — the LLM-driven backend on three
keys, an unmodified baseline on a fourth, the current heuristic backend on a
fifth — specifically to compare them under identical matchmaking conditions.
Since GLEE ranks an account by its *best*-performing agent rather than an
aggregate, once that internal comparison settled on the heuristic backend as
the stronger arm, we moved every key onto it: five parallel shots at the
strategy already judged best, rather than continuing to burn slots on a
controlled experiment. The LLM-driven backend remains implemented and
available, and its fallback path still delegates to the same heuristic
strategy, but it is not what is generating the results in Section 5.

**Safety layer, shared by both backends.**

1. *Schema validation.* Every action — from either backend — is checked
   against the family- and phase-specific constraints the GLEE server itself
   enforces (e.g., bargaining offers must partition the pot exactly; a
   negotiation rejection must include a counter-price). A response that fails
   this check is discarded before it reaches the API.
2. *Bounded retry.* An LLM response that isn't valid JSON is shown back to the
   model once, with the parse error, for self-correction — not retried
   indefinitely.
3. *Timeout.* Each LLM call is capped at 60 seconds, well under the
   platform's 120-second turn clock, so a hung provider call cannot itself
   cause an abandonment.
4. *Fallback.* Any failure at any of the above stages — timeout, malformed
   JSON, schema violation, or provider error (including rate-limiting) —
   returns a guaranteed-valid move from the heuristic strategy instead of
   propagating the error. This fallback is not a naive always-accept; it
   delegates to the full heuristic pipeline (opponent read, learned
   threshold, concession ramp), since a path this is hit often under a tight
   provider quota must still weigh whether a deal is actually good.
5. *Rate-limit pacing.* Because the GLEE SDK polls for all pending moves on a
   fixed interval and can hand several to the strategy function at once
   (independent of a configured concurrency level), we added an explicit
   client-side rate limiter in front of the LLM call: calls arriving faster
   than a configured minimum interval (13s by default) skip the LLM entirely
   and take the safe fallback, rather than spending API quota on a call that
   would be rate-limited anyway. This matters in practice for free-tier model
   access, where per-minute request quotas are far below what a
   multi-game-concurrent agent would otherwise attempt.

## 3. Strategy Design by Game Family

### 3.1 Bargaining

The proposer offers itself a `favor` share of the pot (0.5 = an even split);
the responder accepts any offer giving at least an `accept_threshold` share.
Four signals adjust those two numbers before each move:

- **Opponent classification** (`opponent_model.py`). Every observed
  counter-offer is folded into a live, in-game gap (the change between an
  opponent's successive asks, normalized by pot size) and, when the opponent
  is disclosed, a cross-game profile of accept rate and rounds-to-agreement.
  An opponent is read `aggressive` if its accept rate is under 30% or its
  normalized counter-gap is under 0.03 (small, stonewalling moves);
  `cooperative` if accept rate is over 70% or the gap exceeds 0.10;
  `random` if its gap's coefficient of variation exceeds 1.2 (inconsistent
  concession size); otherwise `unknown`. Against an `aggressive` read, `favor`
  drops by 0.03 and `accept_threshold` drops by 0.05 (close before a
  deadlock); against `cooperative`, both rise by the same amount (hold out
  for more, since they're unlikely to walk).
- **Discount-factor edge.** Bargaining's `game_state` exposes each side's own
  per-round discount factor (`delta_1`/`delta_2`) directly — not documented in
  the SDK, confirmed via a live pull. Whoever discounts more loses more value
  by dragging the game out, so `(my_delta - other_delta) * 0.15`, clamped to
  ±0.05, is added to both knobs: firmer when patience favors us, softer when
  it doesn't. This stacks additively with the opponent-type adjustment rather
  than overriding it.
- **Learned accept threshold.** `train_model.py` fits an offline, L2-regularized
  logistic regression (batch gradient descent, 800 epochs) on every logged
  game's final offer share and round, predicting P(agreement). Its implied
  accept-share at the current round is blended into `accept_threshold` at
  20% weight (`0.8 * accept_threshold + 0.2 * learned`) — deliberately small,
  since the fit pools every opponent and agent variant played so far rather
  than this specific game. `learned_model.py` refuses to apply a fit whose
  main coefficient doesn't point the sane direction (higher share → more
  likely to close), and below 40 usable rows `train_model.py` skips writing a
  model at all rather than shipping noise.
- **Concession ramp.** A live bug — two static agents with offers that never
  move regardless of history can deadlock indefinitely; we observed a real
  game stuck at round 59 — motivated a linear ramp that erodes both `favor`
  and `accept_threshold` toward a fair split as the round count climbs,
  fully faded by round 15 (round 6 against an `aggressive` read, round 25
  against `cooperative`). This targets realized payoff directly rather than
  just accept rate: a deadlocked game that never closes banks nothing.

### 3.2 Negotiation

The opening offer anchors at 1.5× (seller) or 0.7× (buyer) the agent's own
valuation, with a short natural-language justification. From there:

- The agent never rejects an offer that's already profitable relative to its
  own valuation *and* clears a minimum-share-of-surplus bar — blocking an
  efficient deal only lowers the agent's own score.
- When `complete_information` reveals the opponent's true valuation too, the
  fair (50/50-surplus) price is known exactly. Per an internal reading of the
  GLEE paper (arXiv 2410.05254, Table 5), complete information genuinely
  improves both efficiency and fairness in negotiation specifically — so
  here the agent doesn't settle for merely-profitable; it requires a minimum
  share of the *known* surplus (`max((0.30 - 0.02*round) * type_multiplier,
  0.05)`, scaled 0.7×/1.0×/1.3× by the opponent-type read) before accepting,
  countering toward the fair price otherwise.
- Without `complete_information`, and on an unproven first offer, the agent
  holds out once even if the offer is technically profitable — the same
  arXiv 2512.13063 finding cited for bargaining suggests LLM opponents rarely
  punish a single counter, so a small first-round margin check
  (opponent-type-scaled: 0.08/0.15/0.20) is worth the extra round.
- **Tiered concession ramp.** The original counter-offer schedule (a single
  linear ramp saturating at round ≤25, then holding a static offer forever)
  deadlocked in practice — live games stuck at rounds 97, 100, and 108, since
  neither side's offer moved again after the ramp saturated. The replacement
  is a four-band schedule (15% conceded by ~round 20, 45% by ~50, 80% by
  ~75, 100% by ~90, compressed proportionally for an `aggressive` read) that
  keeps closing the gap all the way out instead of parking indefinitely, capped
  uniformly regardless of opponent type since the deadlocks it fixes would
  reproduce under an uncapped, slower-for-`cooperative` version of the same
  idea.
- **Belief-state tracking** (`belief_model.py`, LLM backend only). When
  `complete_information` is off, a 60-point discretized grid filter over the
  opponent's plausible true valuation is updated after every observed offer
  via a soft-threshold likelihood (a seller's ask is evidence their floor is
  at or below it, downweighted rather than ruled out above it; a buyer's
  offer the mirror case) and its mean/80%-interval is injected into the
  LLM's prompt. Not used by the heuristic backend, which has no channel to
  act on a continuous belief beyond the discrete opponent-type read above.
- **Learned margin.** The same 20%-weight blend as bargaining, on the
  holdout margin instead of the accept threshold. Negotiation's model needs
  `my_value`/`role_type`, only added to the logged row after the trainer was
  written, so it starts as a no-op on older data and fills in as the fleet
  keeps playing.

### 3.3 Persuasion

The live implementation (`persuasion_model_v2.py`, wired into
`simple_agent.py` and `game_logger.py`) is a rewrite of an earlier version
(`persuasion_model.py`, still used by the LLM backend) after logs showed raw
acceptance rate doesn't track payoff: one argument type sat at 100%
acceptance over 40 tries against one opponent, then the very next round of
the same type paid $0 — price varies round to round, so hit rate and payoff
are different signals here.

**Seller side.** Six argument framings (economic, fairness, safety, social,
convenience, long-term) are scored per opponent as

```
score = P(bought) × avg_payoff_if_bought − risk_weight × stdev(payoff)
```

with `P(bought)` Laplace-smoothed ((successes+1)/(attempts+2)) and
`risk_weight = 0.15` penalizing a framing that pays big rarely and $0 mostly,
even at the same mean as a steadier one. Selection: every framing is probed
at least once before any exploitation begins (full coverage, not a 3-round
sample); after that, a framing on a 3-round decline streak is excluded from
consideration entirely (a circuit breaker, forcing a pivot regardless of its
historical score); the remainder are sampled proportionally to
`max(score, 0) + 0.05` — soft selection, not argmax, so a currently-second-best
framing keeps getting occasional traffic rather than being starved out.
Repeat pitches of the same framing against a stalled opponent escalate
through a 3-stage phrasing chain instead of resending identical text. No
buyer-facing "signals" (e.g. *asks about benefits*) are read, because
confirmed in the underlying game: buyers only ever return a yes/no `bought`
per round, never text — so `classify_opponent`'s personality read is built
from which framing has actually paid off against a given opponent, not from
anything the buyer said.

**Buyer side.** Instead of the plain expected-value rule (buy when
`p*v > price` — quality prior times value exceeds price), the agent tracks,
within the current game only, how many times this seller has already
endorsed a round that later revealed itself as low quality, and downweights
`p` by `max(0.3, 1 - 0.25*lies)` — a seller caught overselling once is
trusted less on every later round, not just flagged after the fact — then
applies an additional downside penalty (`risk_aversion = 0.5`) beyond what
plain EV already charges for the price-above-low-quality-value case. This
reduces exactly to the plain-EV rule once nothing has been detected yet.

## 4. Implementation Notes

The agent is implemented in Python against the official `glee-sdk` package,
which handles matchmaking, polling, and per-move submission. We run up to
five registered agents concurrently under the platform's per-account agent
limit; as described in Section 2, these currently all run the same
heuristic strategy rather than a live A/B spread, after an earlier internal
comparison (heuristic-variant vs. heuristic-control vs. LLM-driven vs.
unmodified baseline, one arm per key) settled on the heuristic variant.
`simple_agent.py` still carries both `strategy` (the current variant:
opponent-type-adjusted knobs, `concede=True`, a 55/45 bargaining anchor) and
`strategy_control` (the prior baseline: static 50/50 offers, no concession
ramp, no negotiation holdout) so any future change can be judged against a
same-time, same-opponent-pool control rather than a noisy before/after
comparison — the methodology this project's evaluation deliberately does
*not* have the data for right now (Section 5).

Every finished game we end (by accepting, or by a forced close on our final
attempt) is logged to `logs/games.jsonl` — family, role, round, our action,
outcome, payoff, the opponent's last offer, and (once added) our own
valuation and role type for the learned-model trainer. A parallel
human-readable line goes to `logs/learning.log`. Opponent behavior
(`logs/opponent_profiles.json`), persuasion argument outcomes
(`logs/persuasion_profiles_v2.json`), and the offline-trained thresholds
(`logs/learned_bargaining.json`, `logs/learned_negotiation.json`) persist to
disk so profiles survive process restarts and are shared across `fleet.py`'s
multiple agent processes.

**A structural logging gap, common to three subsystems above:** a game the
opponent ends by accepting *our* offer never calls `move()` again on our
side — the SDK's pending-games loop simply stops returning it, with no event
to hook. This means `games.jsonl`, `opponent_model.py`'s cross-game accept
rate, and the persuasion reputation ledger all only see games that end on
our move, undercounting opponents (and outcomes) who mostly accept quickly.
The live in-game counter-gap read in `opponent_model.py` has no such blind
spot, which is why it takes priority over the cross-game accept-rate read
when the latter is thin.

## 5. Evaluation

We measure two things: (a) validity and stability — whether the agent ever
triggers an abandonment or invalid-move penalty — and (b) realized outcomes
per family, via `baseline.py`, our own tracer over `logs/games.jsonl` (no
code change, pure measurement). Success is defined per family to match how
each game actually resolves: `outcome == "agreement"` for bargaining and
negotiation; `payoff > 0` for persuasion, since persuasion's own `outcome`
field reads `"completed"` for every finished game regardless of whether the
pitch actually earned anything — win/loss lives in payoff there, not
outcome.

Qualitatively, on (a): no game in the logged history was lost to a timeout,
malformed response, or abandonment. The safety layer held throughout.

On (b), the snapshot below is taken at the point of writing (2026-08-28),
across 3,232 bargaining games, 3,143 negotiation games, and 2,959 persuasion
games logged since the fleet started running the current strategy.

**Table 1 — cumulative snapshot (most-recent-N games, overlapping cuts).**
This is the same cut `baseline.py --last N` reports; each column includes
every game in the smaller columns to its left.

| Family      | last 100        | last 200         | last 500          | last 1000          | last 2000          | all-time            |
|-------------|------------------|-------------------|--------------------|---------------------|---------------------|----------------------|
| Bargaining  | 96.0% / $135,402 | 93.5% / $151,581  | 95.0% / $150,548   | 94.6% / $148,547    | 94.2% / $143,416    | 94.2% / $143,565 (n=3232) |
| Negotiation | 48.0% / $18,903  | 46.0% / $19,413   | 48.0% / $26,160    | 49.8% / $22,897     | 50.8% / $20,875     | 50.7% / $21,337 (n=3143)  |
| Persuasion  | 32.0% / $2,242,322 | 30.5% / $2,104,781 | 32.4% / $2,219,400 | 33.4% / $2,597,792 | 35.4% / $2,986,567 | 36.8% / $2,823,818 (n=2959) |

*(cell format: success rate / average payoff)*

Because these cuts nest, adjacent columns mostly share the same underlying
games and differences are driven by whichever games sit only in the wider
cut. To see an actual time trend rather than an artifact of nesting, Table 2
splits the same history into non-overlapping chronological buckets, oldest
game first:

**Table 2 — non-overlapping recency buckets, oldest → newest.**

*Bargaining (n=3232):*

| games (index range) | n    | success | avg payoff |
|----------------------|------|---------|------------|
| 1–1232                | 1232 | 94.1%   | $143,807   |
| 1233–2232              | 1000 | 93.8%   | $138,284   |
| 2233–2732              | 500  | 94.2%   | $146,547   |
| 2733–3032              | 300  | 96.0%   | $149,859   |
| 3033–3132              | 100  | 91.0%   | $167,760   |
| 3133–3232 (most recent)| 100  | 96.0%   | $135,402   |

*Negotiation (n=3143):*

| games (index range) | n    | success | avg payoff |
|----------------------|------|---------|------------|
| 1–1143                | 1143 | 50.7%   | $22,146    |
| 1144–2143              | 1000 | 51.7%   | $18,853    |
| 2144–2643              | 500  | 51.6%   | $19,635    |
| 2644–2943              | 300  | 49.3%   | $30,658    |
| 2944–3043              | 100  | 44.0%   | $19,924    |
| 3044–3143 (most recent)| 100  | 48.0%   | $18,903    |

*Persuasion (n=2959):*

| games (index range) | n    | success | avg payoff |
|----------------------|------|---------|------------|
| 1–959                 | 959  | 39.7%   | $2,484,403 |
| 960–1959               | 1000 | 37.3%   | $3,375,342 |
| 1960–2459              | 500  | 34.4%   | $2,976,185 |
| 2460–2759              | 300  | 33.7%   | $2,295,812 |
| 2760–2859              | 100  | 29.0%   | $1,967,240 |
| 2860–2959 (most recent)| 100  | 32.0%   | $2,242,322 |

**Reading the trend, per family.**

- *Bargaining* is stable and strong throughout — success never drops below
  91% in any bucket, including the earliest one. The concession ramp and
  opponent-type adjustment appear robust to whatever opponent-pool
  composition existed at any point in the window; there is no visible
  degradation to explain.
- *Negotiation* holds in a 44–52% band with no clear directional trend — the
  earliest bucket (50.7%) is already close to the all-time rate (50.7%), and
  the two most-recent buckets (44.0%, 48.0%) are within the same noise band
  rather than a step change. At n=100 per bucket, the 95% confidence
  interval on a ~50% success rate is roughly ±10 points, which is large
  enough to swallow the differences between adjacent buckets here.
- *Persuasion* shows the one trend worth flagging: success declines from
  39.7% in the earliest bucket to a low of 29.0% five buckets later, before
  recovering slightly to 32.0% in the most recent 100 games — a real,
  fairly monotonic decline across most of the window, not just noise at the
  tail. We do not have a confirmed explanation. Candidates we can't
  currently distinguish between: opponent-pool composition shifting toward
  harder buyers over the course of the window; the v2 payoff-aware rewrite's
  mandatory probe-every-framing phase imposing a cost against opponents it
  hasn't seen yet, which would show up as exactly this kind of dip; or a
  learning-rate mismatch between how fast persuasion profiles calibrate and
  how often the opponent pool actually turns over. Section 6 discusses why
  we can't currently attribute this to a specific code change.

## 6. Discussion and Limitations

**Code changes happened mid-window, unlogged.** `games.jsonl` rows don't
carry a strategy-version or commit identifier, and the file itself has no
per-row timestamp (only a `game_duration`) — so while game index order
tracks real chronological order (the log is append-only), we cannot line up
a specific game index with a specific code deployment after the fact. The
bargaining and negotiation concession-ramp fixes and the persuasion v1→v2
switch all landed within the same roughly one-day evaluation window this
data comes from. This means Table 2's buckets are an honest recency
breakdown, but not a clean before/after comparison of any single change —
attributing persuasion's decline (or bargaining's stability) to a specific
commit would be overclaiming from this data. The fix is mechanical: log a
code version or commit hash per row going forward, which `simple_agent.py`'s
existing `strategy` / `strategy_control` split already half-anticipates.

**The logging coverage gap** (Section 4) means every success-rate number
above is conditioned on games we ended, undercounting opponents who
routinely accept quickly — real performance against fast-accepting opponents
is very likely better than these numbers suggest, in all three families, not
worse.

**No outcome-verified honesty ledger for persuasion.** `litellm_agent.py`'s
cross-game reputation ledger records what the seller *told* an opponent, not
whether it was actually true — the SDK's run loop doesn't surface post-outcome
quality back to the strategy function, so there's no ground truth to check
promises against across games. `persuasion_model_v2.py`'s buyer-side lie
detector is a partial mitigation, but it only operates *within* a single
game (comparing a seller's past endorsements in this game against quality
revealed later in this same game) and only when a quality-reveal field is
actually present in the game state, which isn't confirmed everywhere. A
natural extension is a genuine cross-game verified-honesty ledger, if the
SDK ever surfaces per-round ground truth.

**Provider-side rate limits, not model capability, bounded the LLM backend.**
This was already true when this section was a placeholder and remains true:
a free-tier quota on the order of single-digit requests per minute meant a
large share of LLM-backend moves — under multiple concurrent games — were
served by the heuristic fallback rather than the model. Combined with the
internal comparison in Section 2 that moved the whole fleet onto the
heuristic backend, this project's results say more about the heuristic
strategy's design than about the LLM backend's ceiling; we would want a
non-rate-limited evaluation before drawing conclusions about LLM-driven play
specifically for this competition format.

## 7. Conclusion

We presented a hybrid heuristic/LLM agent for the GLEE competition's three
game families, built around the principle that a strategy layer's failures
should degrade sophistication, never validity. Over roughly 3,000+ logged
games per family, the heuristic backend — opponent-type classification,
game-given discount-factor edges, offline-learned threshold calibration, and
concession ramps added specifically to fix two observed live deadlock bugs —
holds up as stable and strong in bargaining, moderate and flat in
negotiation, and shows a real, currently unexplained decline in persuasion
that we flag as the most concrete open question this evaluation surfaced.
The central methodological gap going into any follow-up is not strategic —
it's that code changes during the evaluation window weren't tagged per
logged game, which is what would let a future version of this same analysis
actually attribute a trend to a specific change instead of just reporting
that one exists.
