"""Per-game data logging (see GLEE_strategy_data_history.pdf, "Data
Requirements I Would Gather"): appends one JSONL row per finished game to
logs/games.jsonl with the fields the strategy layer can actually observe,
and feeds the same outcome into opponent_model's cross-game profile.

Also writes logs/learning.log: one plain-text, human-readable line per
finished game (tail -f it) -- games.jsonl and the *_profiles.json files are
the machine-readable version of the same events, meant for opponent_model.py
and persuasion_model.py to read back, not for a person to skim. This is the
distillation. Note the two "learn from" paths are already separate and both
live: the *_profiles.json files are what's actually injected into the LLM
agents' prompts (see litellm_agent.py/llm_agent.py's _opponent_note /
success_rates calls) -- this log doesn't feed back into that, it's for you.

Wraps GleeClient.move() -- not GleeClient.run()'s private _handle_game --
so this works from whichever script drives the client (simple_agent,
litellm_agent, llm_agent, fleet) without depending on run()'s internals.

Coverage gap (same one paper_draft.md section 6 notes for the persuasion
honesty ledger): this only sees games that end on OUR move -- we accept, or
the server force-closes on our final attempt. A game the opponent ends by
accepting OUR offer never calls move() again on our side, so it never
reaches here. final_rating_delta is likewise not in the per-move result (see
logs/fleet.log samples) -- only cumulative per-family rating is available,
via stats() -- so it's left out of the row rather than faked; a periodic
stats() snapshot is a separate, coarser signal if that's ever needed.
"""

import json
import logging
import os
import subprocess
import threading
import time

from glee_sdk import GleeClient

import belief_model
import opponent_model
import persuasion_model_v2 as persuasion_model

logger = logging.getLogger("game_logger")

_LOG_PATH = os.path.join(os.path.dirname(__file__), "logs", "games.jsonl")
_TRAJECTORY_LOG_PATH = os.path.join(os.path.dirname(__file__), "logs", "trajectory.jsonl")
_LEARNING_LOG_PATH = os.path.join(os.path.dirname(__file__), "logs", "learning.log")
_write_lock = threading.Lock()

_GAME_STARTED_AT: dict[str, float] = {}  # game_id -> monotonic time first seen, for game_duration


def _write_row(row: dict, path: str = _LOG_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with _write_lock, open(path, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")


def _commit_hash() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(__file__),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _info_setting(game: dict) -> str:
    opponent = game.get("opponent") or {}
    bits = [f"opponent_disclosed={opponent.get('type') != 'hidden'}"]
    complete_info = (game.get("game_state") or {}).get("complete_information")
    if complete_info is not None:
        bits.append(f"complete_information={complete_info}")
    return ",".join(bits)


def _terminal_actor(game: dict, action: dict, result: dict) -> str:
    """Best-effort terminal attribution from the event visible to this client."""
    if action.get("decision") in {"accept", "AcceptOffer", "yes", "no"}:
        return "self"
    if result.get("outcome") in {"no_deal", "completed"}:
        return "environment"
    return "self"


def log_outcome(
    game: dict,
    action: dict,
    result: dict,
    path: str = _LOG_PATH,
    agent_name: str | None = None,
    agent_version: str | None = None,
    strategy_version: str | None = None,
    commit_hash: str | None = None,
    random_seed: int | str | None = None,
) -> None:
    """Call right after move() returns game_over=True for a move we made."""
    game_id = game.get("game_id")
    started = _GAME_STARTED_AT.pop(game_id, None)
    role = game.get("your_player")
    state = game.get("game_state") or {}
    row = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "game_family": game.get("game_family"),
        "role": role,
        "round": state.get("round"),
        "offer": action,
        "accepted": result.get("outcome") == "agreement",
        "payoff": result.get(f"{role}_payoff") if role else None,
        "opponent_id": opponent_model.opponent_key(game),
        "info_setting": _info_setting(game),
        "message": action.get("message"),
        "opponent_last_offer": state.get("last_offer"),
        "concession_size": opponent_model.extract_opponent_offer(game),
        "game_duration": (time.monotonic() - started) if started is not None else None,
        # Added for learned_model.py's offline training -- negotiation's
        # profitability is meaningless without knowing our own valuation and
        # buyer/seller side, and bargaining's pool size is a more direct
        # source than re-deriving it from opponent_last_offer's two gains.
        # Rows logged before this field existed will simply lack it.
        "my_value": state.get(f"{role}_value") if role else None,
        "role_type": state.get(f"{role}_role") if role else None,
        "pool_size": state.get("money_to_divide"),
        "game_id": game_id,
        "outcome": result.get("outcome"),
        "ended_by": _terminal_actor(game, action, result),
        "commit_hash": commit_hash,
        "random_seed": random_seed,
    }
    if agent_name:
        row["agent_name"] = agent_name
    if agent_version:
        row["agent_version"] = agent_version
    if strategy_version:
        row["strategy_version"] = strategy_version
    _write_row(row, path)


def _base_event(
    event_type: str,
    game: dict | None = None,
    *,
    game_id: str | None = None,
    agent_name: str | None = None,
    agent_version: str | None = None,
    strategy_version: str | None = None,
    commit_hash: str | None = None,
    random_seed: int | str | None = None,
) -> dict:
    game = game or {}
    state = game.get("game_state") or {}
    gid = game_id or game.get("game_id")
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "event_type": event_type,
        "game_id": gid,
        "game_family": game.get("game_family"),
        "phase": game.get("phase"),
        "role": game.get("your_player"),
        "round": state.get("round"),
        "opponent_id": opponent_model.opponent_key(game) if game else None,
        "info_setting": _info_setting(game) if game else None,
        "agent_name": agent_name,
        "agent_version": agent_version,
        "strategy_version": strategy_version,
        "commit_hash": commit_hash,
        "random_seed": random_seed,
    }


def log_trajectory_event(
    event_type: str,
    game: dict | None = None,
    *,
    path: str = _TRAJECTORY_LOG_PATH,
    game_id: str | None = None,
    action: dict | None = None,
    response: dict | None = None,
    error: str | None = None,
    stats: dict | None = None,
    agent_name: str | None = None,
    agent_version: str | None = None,
    strategy_version: str | None = None,
    commit_hash: str | None = None,
    random_seed: int | str | None = None,
) -> None:
    """Append one reviewer-facing event row per observed state/action/result."""
    row = _base_event(
        event_type,
        game,
        game_id=game_id,
        agent_name=agent_name,
        agent_version=agent_version,
        strategy_version=strategy_version,
        commit_hash=commit_hash,
        random_seed=random_seed,
    )
    if game:
        state = game.get("game_state") or {}
        role = game.get("your_player")
        row.update(
            {
                "state": state,
                "opponent_last_offer": state.get("last_offer"),
                "my_value": state.get(f"{role}_value") if role else None,
                "role_type": state.get(f"{role}_role") if role else None,
                "pool_size": state.get("money_to_divide"),
            }
        )
    if action is not None:
        row["action"] = action
    if response is not None:
        row["response"] = response
        row["valid"] = response.get("valid")
        row["game_over"] = response.get("game_over")
        row["error"] = response.get("error")
        if response.get("game_over"):
            result = response.get("result") or {}
            row["outcome"] = result.get("outcome")
            row["ended_by"] = _terminal_actor(game or {}, action or {}, result)
    if error is not None:
        row["error"] = error
    if stats is not None:
        row["stats"] = stats
    _write_row(row, path)


def _write_learning_line(
    game: dict,
    result: dict,
    path: str = _LEARNING_LOG_PATH,
    agent_name: str | None = None,
    agent_version: str | None = None,
) -> None:
    """One readable line per finished game: what family/opponent, how it
    ended, and -- for bargaining/negotiation -- our current read on that
    opponent, or -- for persuasion -- which pitch has worked best on them.
    Called after opponent_model/persuasion_model's profiles are updated with
    this game's outcome, so the read reflects what we now know, including
    this game."""
    family = game.get("game_family")
    role = game.get("your_player")
    opponent_id = opponent_model.opponent_key(game)
    opponent_label = opponent_id or "hidden opponent"
    outcome = result.get("outcome")
    round_num = (game.get("game_state") or {}).get("round")
    payoff = result.get(f"{role}_payoff") if role else None
    payoff_str = f"${payoff:,.0f}" if isinstance(payoff, (int, float)) else "?"
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    extra = ""
    if family in ("bargaining", "negotiation"):
        label, _ = opponent_model.classify(game)
        if label != "unknown":
            extra = f" (read: {label})"
    elif family == "persuasion":
        best = persuasion_model.best_argument(game)
        if best:
            arg_type, rate, attempts = best
            extra = f", best pitch so far: {arg_type} ({rate * 100:.0f}% over {attempts} tries)"

    line = (
        f"{timestamp} [{family}]"
        f"{f' agent={agent_name}' if agent_name else ''} "
        f"{f'version={agent_version} ' if agent_version else ''}"
        f"role={role} vs {opponent_label}{extra} "
        f"-> {outcome} after {round_num} rounds, payoff={payoff_str}"
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with _write_lock, open(path, "a") as f:
        f.write(line + "\n")


class LoggingGleeClient(GleeClient):
    """Drop-in GleeClient: on top of the normal API, logs every game that
    ends via our own move to logs/games.jsonl and updates opponent_model's
    persisted profiles (see module docstring for the coverage gap)."""

    def __init__(
        self,
        *args,
        log_path: str = _LOG_PATH,
        learning_log_path: str = _LEARNING_LOG_PATH,
        trajectory_log_path: str = _TRAJECTORY_LOG_PATH,
        agent_name: str | None = None,
        agent_version: str | None = None,
        strategy_version: str | None = None,
        commit_hash: str | None = None,
        random_seed: int | str | None = None,
        update_opponent_profile: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.log_path = log_path
        self.learning_log_path = learning_log_path
        self.trajectory_log_path = trajectory_log_path
        self.agent_name = agent_name
        self.agent_version = agent_version
        self.strategy_version = strategy_version or agent_version
        self.commit_hash = commit_hash or _commit_hash()
        self.random_seed = random_seed
        self.update_opponent_profile = update_opponent_profile

    def _event_kwargs(self) -> dict:
        return {
            "path": self.trajectory_log_path,
            "agent_name": self.agent_name,
            "agent_version": self.agent_version,
            "strategy_version": self.strategy_version,
            "commit_hash": self.commit_hash,
            "random_seed": self.random_seed,
        }

    def log_stats_snapshot(self, label: str) -> None:
        try:
            stats = self.stats()
        except Exception as exc:
            log_trajectory_event("stats_error", error=f"{label}: {exc}", **self._event_kwargs())
            return
        log_trajectory_event(f"stats_{label}", stats=stats, **self._event_kwargs())

    def _handle_game(self, strategy, game):
        game_id = game.get("game_id")
        if game_id is not None:
            _GAME_STARTED_AT.setdefault(game_id, time.monotonic())
        log_trajectory_event("observed", game, **self._event_kwargs())
        try:
            action = strategy(game)
        except Exception as exc:
            log_trajectory_event("strategy_exception", game, error=repr(exc), **self._event_kwargs())
            logger.exception(f"Strategy error for game {game_id}")
            return False

        log_trajectory_event("action", game, action=action, **self._event_kwargs())
        try:
            result = self.move(game_id, action)
        except Exception as exc:
            log_trajectory_event("move_exception", game, action=action, error=repr(exc), **self._event_kwargs())
            raise
        log_trajectory_event("move_result", game, action=action, response=result, **self._event_kwargs())

        if result.get("valid") is False:
            attempts = result.get("attempts_left")
            suffix = f" ({attempts} attempts left)" if attempts is not None else ""
            logger.warning(
                f"Game {game_id}: move rejected as invalid — "
                f"{result.get('error')!r}{suffix}"
            )
        else:
            logger.info(f"Move result: {result}")
        if result.get("game_over"):
            logger.info(f"Game {game_id} finished! Result: {result.get('result')}")
            return True
        return False

    def move(self, game_id: str, action: dict) -> dict:
        response = super().move(game_id, action)
        if response.get("game_over"):
            try:
                game = self.game_state(game_id)
            except Exception:
                logger.warning(f"Game {game_id}: couldn't refetch final state for logging")
                game = {"game_id": game_id}
            result = response.get("result") or {}
            try:
                log_outcome(
                    game,
                    action,
                    result,
                    path=self.log_path,
                    agent_name=self.agent_name,
                    agent_version=self.agent_version,
                    strategy_version=self.strategy_version,
                    commit_hash=self.commit_hash,
                    random_seed=self.random_seed,
                )
                if self.update_opponent_profile:
                    opponent_model.record_outcome(game, result)
                belief_model.forget_game(game_id)
                _write_learning_line(
                    game,
                    result,
                    path=self.learning_log_path,
                    agent_name=self.agent_name,
                    agent_version=self.agent_version,
                )
            except Exception:
                logger.exception(f"Game {game_id}: outcome logging failed")
        return response
