# GLEE Competition Agent

Hybrid rule-based and adaptive agents for the GLEE Competition, which evaluates
language-based agents in bargaining, negotiation, and persuasion games. The
agents and their evaluation are described in
`competition_paper_neurips2026_revised_v3.tex`, the revised paper
(`competition_paper_neurips2026_improved.tex` is the version first submitted).

## Overview

This repository contains a family of GLEE agents developed for competition
experiments. The main design avoids external LLM calls during live play and uses
interpretable economic rules with narrow adaptive layers:

- Bargaining: opponent-aware and deadline-aware split proposals.
- Negotiation: value-protecting acceptance thresholds and learned calibration.
- Persuasion: payoff-aware argument selection with opponent and global profiles.

The revised paper evaluates the agents from the official GLEE game records. In
the last 6.3 hours of our runs, when five versions played at the same time, V9
reached 90.1% agreement on feasible negotiation games against 81.1% for the
vanilla baseline, with higher normalized payoff, and bargaining agreement rose
from 92.5% to 96.3%. Persuasion showed no detectable difference. An earlier
conclusion that V9's negotiation agreement collapsed came from the local logger,
which never recorded games that ended on the opponent's move.

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
- `competition_paper_neurips2026_revised_v3.tex`: revised NeurIPS 2026 competition paper.
- `paper_data/`: per-game data behind the paper (see `paper_data/README.md`).
- `analysis/`: scripts that build `paper_data/` and check every number in the paper.

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

`game_logger.py` records only games that end on this agent's move. The revised
paper measures the effect: 45-51% of games ended on the opponent's move and were
never logged, so local aggregates are biased. The paper uses the official
platform records instead.

## Strategy Versions

| Version | Role in experiments | Main behavior |
| --- | --- | --- |
| V0 | Control | Vanilla deterministic baseline |
| V1 | Ablation | Opponent modeling in bargaining and negotiation |
| V2 | Ablation | Negotiation concession schedule |
| V3 | Ablation | Learned-threshold calibration |
| V4 | Ablation | Persuasion model v2 only |
| V5-V8 | Development | Bundled adaptive bargaining, negotiation, and persuasion policies |
| V9 | Final | Deadline-aware bargaining, surplus-share negotiation |

## Paper

The revised paper is LaTeX in the NeurIPS 2026 style. Put `neurips_2026.sty`
next to it (without it, the file falls back to plain margins):

```bash
pdflatex competition_paper_neurips2026_revised_v3.tex
```

It evaluates the agents from the official GLEE game records, measures the local
logger's bias, and reports the reliability failures found in the replays.

## Reproducing the Paper

Every number in the paper can be checked from a fresh clone (Python 3.10 or
newer, standard library only):

```bash
python3 analysis/check_paper.py
```

It reads `paper_data/` and prints PASS or FAIL for each claim;
`paper_data/README.md` describes the files. The raw GLEE downloads are not in
the repository because the replays contain other teams' messages. The account
owner rebuilds `paper_data/` from them with `bash analysis/run_api_audit.sh`.

V9 bundles several mechanisms, so its results describe the full policy. V7-V9
persuasion used unseeded randomness during the competition; set
`GLEE_RANDOM_SEED` so that future runs record their seed.
