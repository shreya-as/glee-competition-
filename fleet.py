"""Run clean GLEE ablation agents.

Agent/version assignment:
    1 sh_agent1      -> V9 leaderboard push
    2 shreyaAgent2   -> V1 opponent-model
    3 Agent_3        -> V3 learned-model
    4 shreya_agent_4 -> V0 vanilla
    5 agent_5        -> V7 advanced + contextual-bandit persuasion

V1/V3/V0 ablation strategies live in experiment_agent.py.

GLEE allows up to 5 agents per account. Create extras in your Dashboard
(each gets its own API key), then list them all in .env:

    GLEE_API_KEY=glee_...
    GLEE_API_KEY_2=glee_...
    GLEE_API_KEY_3=glee_...
    GLEE_API_KEY_4=glee_...
    GLEE_API_KEY_5=glee_...

Runs with just GLEE_API_KEY fine too (agent 1 only).

Usage:
    python fleet.py
"""

import logging
import multiprocessing
import os
import re
import time
from datetime import datetime, timezone

from dotenv import load_dotenv

import experiment_agent
import advanced_agent
from game_logger import LoggingGleeClient

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger("fleet")

# Rule-based moves are instant and free, so these can run wide open.
RULE_AGENT_CONCURRENCY = 8

AGENT_NAMES = {1: "sh_agent1", 2: "shreyaAgent2", 3: "Agent_3", 4: "shreya_agent_4", 5: "agent_5"}
AGENT_VERSIONS = {1: "v9", 2: "v1", 3: "v3", 4: "v0", 5: "v7"}
VERSION_LABELS = {
    "v0": "vanilla",
    "v1": "opponent-model",
    "v2": "concession",
    "v3": "learned-model",
    "v4": "persuasion-adaptation",
    "v5": "payoff-aware",
    "v6": "opponent-specific",
    "v7": "contextual-bandit",
    "v8": "final-bargaining-v5-negotiation-v3-persuasion-v7",
    "v9": "leaderboard-push",
}
_QUEUE_PAUSE_RE = re.compile(r"try again after ([0-9T:\-.]+Z)")


def _discover_keys() -> list[str]:
    keys = []
    if os.environ.get("GLEE_API_KEY"):
        keys.append(os.environ["GLEE_API_KEY"])
    for name, value in sorted(os.environ.items()):
        if re.fullmatch(r"GLEE_API_KEY_\d+", name) and value:
            keys.append(value)
    return keys


def _client_for(index: int, api_key: str) -> LoggingGleeClient:
    base_url = os.environ.get("GLEE_API_URL")
    name = AGENT_NAMES.get(index, f"agent{index}")
    version = AGENT_VERSIONS.get(index, "v5")
    kwargs = {
        "api_key": api_key,
        "agent_name": name,
        "agent_version": version,
        "strategy_version": VERSION_LABELS.get(version, version),
        "log_path": os.path.join(os.path.dirname(__file__), "logs", f"{version}_games.jsonl"),
        "learning_log_path": os.path.join(os.path.dirname(__file__), "logs", f"{version}_learning.log"),
        "update_opponent_profile": version in {"v1", "v5", "v8", "v9"},
    }
    if base_url:
        kwargs["base_url"] = base_url
    return LoggingGleeClient(**kwargs)


def _strategy_for(index: int):
    version = AGENT_VERSIONS.get(index, "v5")
    if version in advanced_agent.STRATEGIES:
        return advanced_agent.STRATEGIES[version]
    return experiment_agent.STRATEGIES[version]


def _queue_pause_seconds(error: Exception) -> float | None:
    match = _QUEUE_PAUSE_RE.search(str(error))
    if not match:
        return None
    retry_at = datetime.fromisoformat(match.group(1).replace("Z", "+00:00"))
    return max(5.0, (retry_at - datetime.now(timezone.utc)).total_seconds() + 5.0)


def _run_rule_agent(index: int, api_key: str) -> None:
    # force=True: the child inherits the parent's already-configured root
    # logger via fork(), so a plain basicConfig() call here is a silent
    # no-op (basicConfig only takes effect when no handlers exist yet) --
    # every child would log under the same generic "fleet" name with no way
    # to tell them apart. force=True actually replaces the inherited config
    # with this process's own.
    agent_name = AGENT_NAMES.get(index, f"agent{index}")
    version = AGENT_VERSIONS.get(index, "v5")
    strategy_name = f"{version}-{VERSION_LABELS.get(version, version)}"
    run_log_path = os.path.join(os.path.dirname(__file__), "logs", f"{version}_run.log")
    os.makedirs(os.path.dirname(run_log_path), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s {agent_name}-{strategy_name} %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(run_log_path)],
        force=True,
    )
    strategy = _strategy_for(index)
    while True:
        try:
            client = _client_for(index, api_key)
            logging.getLogger("fleet").info(
                "Starting %s as %s with %s.%s; outcome log=%s; run log=%s",
                agent_name,
                version,
                strategy.__module__,
                strategy.__name__,
                client.log_path,
                run_log_path,
            )
            logging.getLogger("fleet").info(f"Agent {index} ({agent_name}, {strategy_name}) stats: {client.stats()}")
            client.run(strategy, concurrency=RULE_AGENT_CONCURRENCY)
            logging.getLogger("fleet").info("Agent %s run returned; restarting in 30s", agent_name)
            time.sleep(30)
        except Exception as exc:
            pause_seconds = _queue_pause_seconds(exc)
            if pause_seconds is not None:
                logging.getLogger("fleet").warning(
                    "Agent %s queue-paused by server; retrying in %.0fs", agent_name, pause_seconds
                )
                time.sleep(pause_seconds)
                continue
            logging.getLogger("fleet").exception(f"Agent {index} crashed; retrying in 60s")
            time.sleep(60)


def _process_for(index: int, key: str) -> multiprocessing.Process:
    return multiprocessing.Process(target=_run_rule_agent, args=(index, key), daemon=True)


if __name__ == "__main__":
    keys = _discover_keys()
    if not keys:
        print("Set GLEE_API_KEY (and optionally GLEE_API_KEY_2..5) in .env")
        raise SystemExit(1)

    logger.info(
        "Launching %s agent(s): %s",
        len(keys),
        [
            f"{AGENT_NAMES.get(i + 1, f'agent{i+1}')}:{AGENT_VERSIONS.get(i + 1, 'v5')}"
            for i in range(len(keys))
        ],
    )

    processes = [_process_for(i + 1, key) for i, key in enumerate(keys)]
    for p in processes:
        p.start()

    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        logger.info("Interrupted — stopping all agents...")
        for p in processes:
            p.terminate()
        for p in processes:
            p.join()
