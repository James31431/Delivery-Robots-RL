"""Evaluation loop. Environment and agent are injected by the caller."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from rl.interfaces import AgentProtocol, EnvironmentProtocol


def evaluate_agent(
    env: EnvironmentProtocol,
    agent: AgentProtocol,
    episodes: int = 100,
    max_steps: int = 50,
    render: bool = False,
) -> Dict[str, float]:
    """Run ``agent`` on ``env`` and report metrics.

    By default actions are greedy (exploration disabled): if the agent has an
    ``epsilon`` attribute it is temporarily set to 0 and restored afterwards.
    If ``temperature`` is given, the agent is queried with Boltzmann/softmax
    sampling instead (agents that don't support it fall back to greedy).
    Baseline policies without epsilon work fine too.
    """
    total_rewards = 0.0
    successes = 0
    success_steps_sum = 0

    has_epsilon = hasattr(agent, "epsilon")
    saved_epsilon = getattr(agent, "epsilon", None)
    if has_epsilon:
        setattr(agent, "epsilon", 0.0)

    episode_path = Path(record_episode_path) if record_episode_path else None

    try:
        for ep in range(1, episodes + 1):
            state = env.reset()
            ep_reward = 0.0
            ep_steps = 0
            done = False
            success = False
            episode_trace: Optional[List[Dict[str, Any]]] = (
                [] if episode_path is not None and ep == 1 else None
            )

            if episode_trace is not None:
                episode_trace.append(
                    {
                        "step": 0,
                        "state": state,
                        "action": None,
                        "reward": 0.0,
                        "next_state": state,
                        "done": False,
                        "info": {"phase": "reset"},
                    }
                )

            while not done and ep_steps < max_steps:
                # Prefer greedy=True if the agent supports it.
                try:
                    action = agent.choose_action(state, greedy=True)  # type: ignore[call-arg]
                except TypeError:
                    action = agent.choose_action(state)
                state, reward, done, info = env.step(action)
                ep_reward += reward
                ep_steps += 1
                if isinstance(info, dict) and info.get("success"):
                    success = True

                if episode_trace is not None:
                    episode_trace.append(
                        {
                            "step": ep_steps,
                            "state": state,
                            "action": action,
                            "reward": reward,
                            "next_state": next_state,
                            "done": done,
                            "info": info,
                        }
                    )

                state = next_state

            if not success:
                success = bool(getattr(env, "delivered", False))

            if episode_trace is not None:
                record = {
                    "episode": ep,
                    "episodes_total": episodes,
                    "max_steps": max_steps,
                    "reward": ep_reward,
                    "steps": ep_steps,
                    "success": success,
                    "metadata": _env_metadata(env),
                    "trajectory": episode_trace,
                }
                _save_episode_record(episode_path, record)

            total_rewards += ep_reward
            if success:
                successes += 1
                success_steps_sum += ep_steps

            if render:
                print(f"\nEpisode {ep} (reward={ep_reward:.1f}, success={success}):")
                env.render()
    finally:
        if has_epsilon:
            setattr(agent, "epsilon", saved_epsilon)

    avg_reward = total_rewards / episodes if episodes > 0 else 0.0
    success_rate = successes / episodes if episodes > 0 else 0.0
    avg_steps_success = (
        success_steps_sum / successes if successes > 0 else float("nan")
    )

    print("\n=== Evaluation Summary ===")
    print(f"Episodes:                {episodes}")
    print(f"Average reward:          {avg_reward:.2f}")
    print(f"Success rate:            {success_rate * 100:.1f}%")
    if successes > 0:
        print(f"Avg steps (successes):   {avg_steps_success:.2f}")
    else:
        print("Avg steps (successes):   n/a (no successes)")
    print("==========================")

    return {
        "average_reward": avg_reward,
        "success_rate": success_rate,
        "average_steps_successful": avg_steps_success,
        "episodes": episodes,
    }
