"""Evaluation loop. Environment and agent are injected by the caller."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from rl.interfaces import AgentProtocol, EnvironmentProtocol


def _greedy_action(agent: AgentProtocol, state):
    """Query an agent greedily, tolerating agents without a ``greedy`` kwarg."""
    try:
        return agent.choose_action(state, greedy=True)  # type: ignore[call-arg]
    except TypeError:
        return agent.choose_action(state)


def _json_safe(value: Any) -> Any:
    """Convert common Python objects into JSON-serializable structures."""
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _env_metadata(env: EnvironmentProtocol) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "env_class": env.__class__.__name__,
        "action_space_size": getattr(env, "action_space_size", None),
    }
    cfg = getattr(env, "cfg", None)
    if cfg is not None:
        metadata["config"] = _json_safe(cfg)
    return metadata


def _save_episode_record(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(record), handle, indent=2, sort_keys=True)
        handle.write("\n")


def evaluate_agent(
    env: EnvironmentProtocol,
    agent: AgentProtocol,
    episodes: int = 100,
    max_steps: int = 50,
    render: bool = False,
    temperature: float = None,
    record_episode_path: Optional[str] = None,
    record_successful_episode_only: bool = False,
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
    recorded_successful_episode = False

    try:
        for ep in range(1, episodes + 1):
            state = env.reset()
            ep_reward = 0.0
            ep_steps = 0
            done = False
            success = False
            episode_trace: Optional[List[Dict[str, Any]]] = (
                []
                if episode_path is not None and (not record_successful_episode_only or not recorded_successful_episode)
                else None
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
                if temperature is not None:
                    # Boltzmann/softmax query; fall back to greedy, then plain.
                    try:
                        action = agent.choose_action(state, temperature=temperature)  # type: ignore[call-arg]
                    except TypeError:
                        action = _greedy_action(agent, state)
                else:
                    action = _greedy_action(agent, state)
                next_state, reward, done, info = env.step(action)
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
                if not record_successful_episode_only or success:
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
                    recorded_successful_episode = True

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
