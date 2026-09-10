# GLEE Competition Agent

Hybrid rule-based and adaptive agents for the GLEE Competition, which evaluates
language-based agents in bargaining, negotiation, and persuasion games. The
final system is described in
`competition_paper_neurips2026_improved.tex`.

## Overview

This repository contains a family of GLEE agents developed for competition
experiments. The main design avoids external LLM calls during live play and uses
interpretable economic rules with narrow adaptive layers:

- Bargaining: opponent-aware and deadline-aware split proposals.
- Negotiation: value-protecting acceptance thresholds and learned calibration.
- Persuasion: payoff-aware argument selection with opponent and global profiles.

The final V9 agent improves bargaining speed at similar payoff in local logs,
but the paper documents an important tradeoff: negotiation value protection
increased payoff conditional on agreement while reducing agreement rate.
Persuasion remained noisy and high variance.

## Main Files

- `advanced_agent.py`: final and advanced strategies, including V5-V9.
- `vanilla_agent.py`: deterministic V0 baseline.
- `experiment_agent.py`: V1-V4 ablation strategies.
- `run_sh_agent1_v9.py`: runner for the final V9 agent.
- `fleet.py`: multi-agent runner for baseline, ablation, and final agents.
- `game_logger.py`: logging wrapper around the GLEE SDK client.
- `train_model.py`: offline logistic calibration for learned thresholds.
- `learned_model.py`: loads learned bargaining and negotiation models.
- `opponent_model.py`: per-opponent offer history and classification.
- `persuasion_model_v2.py`: adaptive persuasion argument scoring.
- `competition_paper_neurips2026_improved.tex`: NeurIPS 2026 competition paper.

## Setup

Create and activate a Python environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install the runtime dependencies needed by the agents you plan to run. The
rule-based competition runners require the GLEE SDK and dotenv support:

```bash
pip install glee-sdk python-dotenv
```

The older LLM-based agents additionally import `litellm` and `anthropic`.

Create a local `.env` file with your GLEE API key:

```bash
GLEE_API_KEY=glee_...
```

For the fleet runner, optional additional keys can be provided as
`GLEE_API_KEY_2`, `GLEE_API_KEY_3`, `GLEE_API_KEY_4`, and `GLEE_API_KEY_5`.
Set `GLEE_API_URL` only when using a non-default GLEE server.

## Running Agents

Run the final V9 agent:

```bash
python run_sh_agent1_v9.py
```

Run the configured fleet of baseline, ablation, and final agents:

```bash
python fleet.py
```

The runners create `logs/` automatically and write JSONL outcome logs plus
plain-text run and learning logs. `logs/` is intentionally ignored by git.

## Training Learned Thresholds

After collecting local logs, retrain the learned bargaining and negotiation
threshold models:

```bash
python train_model.py
```

The trainer reads `logs/games.jsonl` and writes:

- `logs/learned_bargaining.json`
- `logs/learned_negotiation.json`

The paper notes a logging limitation: `game_logger.py` records games that end
on this agent's move, so games accepted by opponents after our offers may be
missing from local aggregates.

## Strategy Versions

| Version | Role in experiments | Main behavior |
| --- | --- | --- |
| V0 | Control | Vanilla deterministic baseline |
| V1 | Ablation | Opponent modeling in bargaining and negotiation |
| V2 | Ablation | Negotiation concession schedule |
| V3 | Ablation | Learned-threshold calibration |
| V4 | Ablation | Persuasion model v2 only |
| V5-V8 | Development | Bundled adaptive bargaining, negotiation, and persuasion policies |
| V9 | Final | Deadline-aware bargaining, value-protecting negotiation, exploit-heavy persuasion |

## Paper

The competition paper is maintained as LaTeX:

```bash
pdflatex competition_paper_neurips2026_improved.tex
```

The paper summarizes the architecture, local V0/V9 results, ablation evidence,
limitations, and reliability lessons from the competition run.

## Reproducibility Notes

Local analysis should aggregate each `logs/v*_games.jsonl` file by
`game_family`. Count `outcome == "agreement"` as success for bargaining and
negotiation, and positive payoff as success for persuasion.

The V9 policy bundles several mechanisms, so V0 versus V9 is a complete-policy
comparison rather than a clean causal ablation. Exact persuasion reruns for
later versions are not fully reproducible unless random seeds are logged.
