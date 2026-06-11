"""Evaluation loop. Environment and agent are injected by the caller."""

from __future__ import annotations

from typing import Any, Dict

from rl.interfaces import AgentProtocol, EnvironmentProtocol


def _greedy_action(agent: AgentProtocol, state):
    """Query an agent greedily, tolerating agents without a ``greedy`` kwarg."""
    try:
        return agent.choose_action(state, greedy=True)  # type: ignore[call-arg]
    except TypeError:
        return agent.choose_action(state)


def evaluate_agent(
    env: EnvironmentProtocol,
    agent: AgentProtocol,
    episodes: int = 100,
    max_steps: int = 50,
    render: bool = False,
    temperature: float = None,
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

    try:
        for ep in range(1, episodes + 1):
            state = env.reset()
            ep_reward = 0.0
            ep_steps = 0
            done = False
            success = False

            while not done and ep_steps < max_steps:
                if temperature is not None:
                    # Boltzmann/softmax query; fall back to greedy, then plain.
                    try:
                        action = agent.choose_action(state, temperature=temperature)  # type: ignore[call-arg]
                    except TypeError:
                        action = _greedy_action(agent, state)
                else:
                    action = _greedy_action(agent, state)
                state, reward, done, info = env.step(action)
                ep_reward += reward
                ep_steps += 1
                if isinstance(info, dict) and info.get("success"):
                    success = True

            if not success:
                success = bool(getattr(env, "delivered", False))

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
