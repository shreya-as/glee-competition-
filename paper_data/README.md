# Data for "Selective Adaptation and Reliability Lessons from a Hybrid GLEE Agent"

This folder holds everything needed to recompute the numbers in
`competition_paper_neurips2026_revised_v3.tex`. It is derived from the official GLEE
game histories with replays for our five agents (75,258 games, 28–29 Aug 2026) and
from our own local logs. **No message text from any agent is included.**

## Verify the paper

From the repository root, with Python 3.10 or newer (standard library only):

```
python3 analysis/check_paper.py
```

It prints one line per claim in the paper (the value as printed, the value
recomputed from these files, PASS or FAIL) and ends with `ALL CLAIMS PASS` or the
number of failures. It also fails if a number in the `.tex` file no longer matches.

## Files

| File | Rows | Contents |
|---|---:|---|
| `platform_games.csv.gz` | 75,258 | One row per platform game (columns below). |
| `local_logger_games.csv.gz` | 35,717 | One row per game our local logger recorded. |
| `run_log_rejections.csv` | 4,446 | Server rejections of our moves found in our V0 and V9 run logs. |
| `learned_model_training.csv` | 5,969 | Training rows of the learned negotiation threshold (V3), in logging order. |
| `learned_negotiation.json` | – | The deployed learned model (weights, feature scaling, in-sample metrics). |

### `platform_games.csv.gz`

| Column | Meaning |
|---|---|
| `game_id` | Platform game ID; can be opened on the GLEE replay page. |
| `account` | Our agent: `sh_agent1` (1), `shreyaAgent2` (2), `Agent_3` (3), `shreya_agent_4` (4), `agent_5` (5). |
| `version` | Strategy version (`V0`–`V9`). `V8/V9`: game in the V8/V9 overlap on account 1 that cannot be tied to one version. `V1/V5`, `V3/V6`, `V4/V7`: game at a version switch. `gap`: game started when no version's run window was active on that account (around restarts). |
| `assigned_by` | `logged`: the game ID is in the local log of exactly one version. `time`: unlogged, but only one version was running on that account. `ambiguous` or `gap`: not assigned. |
| `family` | `bargaining`, `negotiation`, or `persuasion`. |
| `started_at`, `completed_at` | UTC timestamps from the platform. |
| `our_player` | `player_1` or `player_2`. In negotiation, `player_1` is the seller. |
| `outcome` | Platform outcome: `agreement`, `no_deal`, `walked_away`, `timeout`, `invalid_moves_exceeded`, or `completed` (persuasion). |
| `our_payoff`, `opponent_payoff` | Final payoffs. |
| `rating_delta` | The platform's rating change for our agent in this game. |
| `final_decision_by` | Who made the game-ending decision: `me`, `opp`, or `system` (timeouts and invalid-move limits). |
| `invalid_moves` | Moves the replay marks invalid (either player; replays do not label the sender). |
| `invalid_decision_when_offer_due` | Of those, an accept/reject payload sent when an offer was due, or an offer when a decision was due. |
| `invalid_resent_within_1s` | Of the invalid moves, decision payloads sent less than 1 second after a valid decision in the same game (a re-sent decision). |
| `our_accepts`, `our_rejects`, `our_other_decisions` | Our decisions in negotiation, from the replay's decision history. |
| `in_local_log` | `true` if our local logger recorded the game. |
| `opponent_type` | `agent` or `human`, as reported by the platform (no opponent identity or rating is given). |
| Game settings | Bargaining: `money_to_divide`, `delta_1`, `delta_2`. Negotiation: `seller_value` and `buyer_value` (multiples of `product_price_order`). Persuasion: `product_price`, `total_rounds`, `p`, `v`, `c`, `is_myopic`, `is_seller_know_cv`, `seller_message_type`. All families: `max_rounds`, `complete_information`, `messages_allowed`. Empty when not used by that family. |

### Other files

- `local_logger_games.csv.gz`: `version` (from the log file name), `agent_name`,
  `timestamp` (as written by the logger, Nepal time, +05:45), `game_id`, `family`, `outcome`.
- `run_log_rejections.csv`: `run_log` (`v0_run.log` or `v9_run.log`), `game_id`,
  `server_message` (the server's rejection text), `times` (how often it was logged for that game).
- `learned_model_training.csv`: `margin` (offer margin relative to our value),
  `round_frac` (round / 30, capped at 1), `agreement` (1 if the game ended in agreement).

## What is not included

The raw history and replay downloads (about 1.1 GB) are not included because the
replays contain other teams' agents' messages. Only the owner of an account can
export them from the GLEE platform. The owner rebuilds this folder from them with
`bash analysis/run_api_audit.sh`, which writes it through `analysis/export_release.py`.
