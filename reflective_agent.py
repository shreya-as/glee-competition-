"""LLM-driven agent for sh_agent1 / Agent_3 / shreya_agent_4 (GLEE_API_KEY,
GLEE_API_KEY_3, GLEE_API_KEY_4) -- litellm_agent.py's pipeline, plus two
additions from the GLEE strategy notes:

- Strategy 3 (Bayesian belief-state tracking): belief_model.py's posterior
  over the opponent's true valuation, injected into the prompt whenever
  negotiation's complete_information is off (the one case something is
  actually hidden -- see that module's docstring).
- Strategy 5 (self-play / chain-of-thought reflection): the model must work
  through four reflection questions -- target outcome, what the opponent
  will infer, their likely best response, and a re-check against our own
  reservation value -- before giving its action. This runs as a "Reasoning:"
  preamble in the SAME completion, not a second chained call: a real
  self-play simulator would cost a full extra LLM call per move, and
  litellm_agent.parse_action already strips any text before the JSON, so
  asking for the reasoning inline is free.

Everything else is inherited from litellm_agent.py unchanged: rate limiting,
JSON parsing/retry, the safe_action fallback (simple_agent.strategy, which
itself carries opponent_model.py's read, persuasion_model.py's argument
tracking, learned_model.py's threshold, and the delta-factor edge),
opponent_model's aggressive/cooperative/random note, and persuasion_model's
argument-success-rate note. This file only overrides the prompts.

Usage: identical to litellm_agent.py -- same env vars, same entry point.
    python reflective_agent.py
"""

import logging

import belief_model
import litellm_agent
from game_logger import LoggingGleeClient
from litellm import completion

logger = logging.getLogger("reflective_agent")

_REFLECTION_INSTRUCTION = """

Before answering, work through this in a short "Reasoning:" section (plain \
text, 2-4 sentences -- stay concise, this isn't the deliverable):
1. What is my target outcome this move?
2. What will the opponent infer from my message or offer?
3. What is the opponent's most likely response to what I'm about to do?
4. Re-checked against that response, does this still clear my own reservation value?
Then, on a new line, give ONLY the JSON action -- no markdown fences, no text after it.
"""


def _system_prompt(family: str) -> str:
    base = litellm_agent._SYSTEM_PROMPTS.get(family, litellm_agent.BARGAINING_SYSTEM)
    return base + _REFLECTION_INSTRUCTION


def _belief_note(game: dict) -> str:
    posterior = belief_model.update_and_estimate(game)
    if posterior is None:
        return ""
    return (
        f"Belief over the opponent's true valuation (Bayesian filter over their "
        f"observed offers -- complete_information is off, so this is inferred, "
        f"not given): mean≈{posterior['mean']:.2f}, 80% range≈[{posterior['low']:.2f}, {posterior['high']:.2f}]\n\n"
    )


def build_user_prompt(game: dict) -> str:
    return _belief_note(game) + litellm_agent.build_user_prompt(game)


def strategy(game: dict) -> dict:
    if not litellm_agent._rate_limit_ok():
        fallback = litellm_agent.safe_action(game)
        logger.info(f"Rate budget spent — playing safe action without calling the LLM: {fallback}")
        return fallback

    family = game["game_family"]
    system_prompt = _system_prompt(family)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": build_user_prompt(game)},
    ]

    for attempt in range(2):
        text = ""
        try:
            response = completion(model=litellm_agent.MODEL, messages=messages, timeout=60)
            text = response.choices[0].message.content or ""
            action = litellm_agent.parse_action(text)
            logger.info("[%s] %s -> %s", family, game["valid_actions"]["type"], action)
            if game["valid_actions"]["type"] == "seller_recommendation":
                litellm_agent._record_recommendation(game, action)
            return action
        except Exception as e:
            logger.warning("LLM attempt %d failed: %s", attempt + 1, e)
            if attempt == 0:
                messages.append({"role": "assistant", "content": text})
                messages.append({
                    "role": "user",
                    "content": f"Your reply was not a valid JSON action ({e}). "
                               f"Reply with ONLY the corrected JSON object (reasoning is optional on this retry).",
                })

    fallback = litellm_agent.safe_action(game)
    logger.warning("Falling back to safe action: %s", fallback)
    return fallback


def run(api_key: str, base_url: str | None = None, concurrency: int = 8, poll_interval: float = 2.0) -> None:
    client = (
        LoggingGleeClient(api_key=api_key, base_url=base_url)
        if base_url
        else LoggingGleeClient(api_key=api_key)
    )
    logger.info(f"Agent stats: {client.stats()}")
    client.run(strategy, concurrency=concurrency, poll_interval=poll_interval)


if __name__ == "__main__":
    import os

    api_key = os.environ.get("GLEE_API_KEY", "")
    if not api_key:
        print("Set GLEE_API_KEY environment variable")
        exit(1)
    run(api_key, base_url=os.environ.get("GLEE_API_URL"))
