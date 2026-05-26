"""Training loop. Environment and agent are injected by the caller."""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional

from rl.interfaces import AgentProtocol, EnvironmentProtocol


def train_agent(
    env: EnvironmentProtocol,
    agent: AgentProtocol,
    episodes: int = 2000,
    max_steps: int = 50,
    render_every: Optional[int] = None,
) -> "tuple[AgentProtocol, Dict[str, Any]]":
    """Train ``agent`` on ``env`` for a number of episodes.

    The env and agent are provided by the caller (dependency injection), so
    this function works with any compatible Environment/Agent pair.

    ``max_steps`` is kept as an argument for API compatibility, but the
    environment is the authority on episode termination via its ``done`` flag.

    Returns
    -------
    (agent, stats)
        ``stats`` contains per-episode lists and summary averages.
    """
    episode_rewards: List[float] = []
    episode_steps: List[int] = []
    episode_successes: List[int] = []
    epsilon_history: List[float] = []

    recent_rewards: deque = deque(maxlen=100)
    recent_success: deque = deque(maxlen=100)

    has_epsilon = hasattr(agent, "epsilon")

    for ep in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0.0
        steps = 0
        done = False
        success = False

        while not done and steps < max_steps:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps += 1
            if isinstance(info, dict) and info.get("success"):
                success = True

        # Some envs report success only through info; fall back to the
        # ``delivered`` attribute if available.
        if not success:
            success = bool(getattr(env, "delivered", False))

        if hasattr(agent, "decay_epsilon"):
            agent.decay_epsilon()

        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        episode_successes.append(1 if success else 0)
        if has_epsilon:
            epsilon_history.append(float(getattr(agent, "epsilon")))

        recent_rewards.append(total_reward)
        recent_success.append(1.0 if success else 0.0)

        if render_every and ep % render_every == 0:
            print(f"\n--- Render after episode {ep} ---")
            env.render()

        if ep % 100 == 0:
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            success_rate = sum(recent_success) / len(recent_success)
            eps_str = (
                f" | epsilon={getattr(agent, 'epsilon'):.3f}" if has_epsilon else ""
            )
            print(
                f"Episode {ep:5d} | "
                f"avg_reward(last100)={avg_reward:7.2f} | "
                f"success_rate(last100)={success_rate*100:5.1f}%"
                f"{eps_str}"
            )

    stats: Dict[str, Any] = {
        "episode_rewards": episode_rewards,
        "episode_steps": episode_steps,
        "episode_successes": episode_successes,
        "epsilon_history": epsilon_history,
        "average_reward_last_100": (
            sum(episode_rewards[-100:]) / min(100, len(episode_rewards))
            if episode_rewards
            else 0.0
        ),
        "success_rate_last_100": (
            sum(episode_successes[-100:]) / min(100, len(episode_successes))
            if episode_successes
            else 0.0
        ),
    }
    return agent, stats
