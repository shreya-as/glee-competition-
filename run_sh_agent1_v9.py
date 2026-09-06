"""Run only sh_agent1 on V9 across all three GLEE game families."""

import logging
import os
import time
from datetime import datetime, timezone

from dotenv import load_dotenv

import advanced_agent
from fleet import _queue_pause_seconds
from game_logger import LoggingGleeClient

load_dotenv()

AGENT_NAME = "sh_agent1"
AGENT_VERSION = "v9"
STRATEGY_LABEL = "leaderboard-push"
CONCURRENCY = 4


def main() -> None:
    api_key = os.environ.get("GLEE_API_KEY")
    if not api_key:
        print("Set GLEE_API_KEY in .env")
        raise SystemExit(1)

    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    run_log_path = os.path.join(log_dir, "v9_run.log")
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s {AGENT_NAME}-{AGENT_VERSION}-{STRATEGY_LABEL} %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(run_log_path)],
        force=True,
    )
    logger = logging.getLogger("sh_agent1_v9")

    kwargs = {
        "api_key": api_key,
        "agent_name": AGENT_NAME,
        "agent_version": AGENT_VERSION,
        "strategy_version": STRATEGY_LABEL,
        "log_path": os.path.join(log_dir, "v9_games.jsonl"),
        "learning_log_path": os.path.join(log_dir, "v9_learning.log"),
        "update_opponent_profile": True,
    }
    base_url = os.environ.get("GLEE_API_URL")
    if base_url:
        kwargs["base_url"] = base_url

    while True:
        try:
            client = LoggingGleeClient(**kwargs)
            logger.info(
                "Starting %s as %s with %s.%s; outcome log=%s; run log=%s",
                AGENT_NAME,
                AGENT_VERSION,
                advanced_agent.strategy_v9.__module__,
                advanced_agent.strategy_v9.__name__,
                client.log_path,
                run_log_path,
            )
            logger.info("Stats: %s", client.stats())
            client.run(advanced_agent.strategy_v9, concurrency=CONCURRENCY)
            logger.info("Run returned; exiting")
            return
        except Exception as exc:
            pause_seconds = _queue_pause_seconds(exc)
            if pause_seconds is not None:
                retry_at = datetime.now(timezone.utc).timestamp() + pause_seconds
                logger.warning("Queue-paused by server until %.0f; retrying in %.0fs", retry_at, pause_seconds)
                time.sleep(pause_seconds)
                continue
            logger.exception("sh_agent1 V9 crashed; retrying in 60s")
            time.sleep(60)


if __name__ == "__main__":
    main()
