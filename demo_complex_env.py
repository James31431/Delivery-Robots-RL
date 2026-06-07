"""Demo: train and evaluate the existing tabular Q-learning agent on the new
ComplexBuildingEnv, alongside a random baseline.

The point isn't to show Q-learning succeeding — the flat action space is 576
and the state space is combinatorially huge, so tabular methods are expected
to barely beat random. This script makes that visible and motivates moving
to function approximation (DQN) in a follow-up issue.

Usage:
    python demo_complex_env.py
"""

from __future__ import annotations

from environment.complex_env import ComplexBuildingEnv, ComplexEnvConfig
from rl.baselines import RandomAgent
from rl.evaluate import evaluate_agent
from rl.q_learning_agent import QLearningAgent
from rl.train import train_agent


def main() -> None:
    cfg = ComplexEnvConfig(max_steps=200)
    print(f"ComplexBuildingEnv config: {cfg}")

    probe = ComplexBuildingEnv(config=cfg, seed=42)
    print(f"\nFlat action space size: {probe.flat_action_space_size}")
    print(f"Per-robot action size:  {probe.per_robot_action_size}")
    print(f"State vector size:      {probe.state_vector_size}")

    print("\n--- Initial state ---")
    probe.render()

    print("\n--- Training Q-learning on ComplexBuildingEnv (300 episodes) ---")
    train_env = ComplexBuildingEnv(config=cfg, seed=42)
    q_agent = QLearningAgent(action_space_size=train_env.action_space_size, seed=42)
    _, stats = train_agent(train_env, q_agent, episodes=300, max_steps=cfg.max_steps)
    print(f"Q-table size after training: {len(q_agent.q_table)} states")

    n_eval = 30
    print(f"\n--- Evaluating both agents on {n_eval} episodes (eval seed=123) ---")

    q_eval = ComplexBuildingEnv(config=cfg, seed=123)
    q_results = evaluate_agent(q_eval, q_agent, episodes=n_eval, max_steps=cfg.max_steps)

    rand_eval = ComplexBuildingEnv(config=cfg, seed=123)
    rand = RandomAgent(action_space_size=rand_eval.action_space_size, seed=42)
    rand_results = evaluate_agent(rand_eval, rand, episodes=n_eval, max_steps=cfg.max_steps)

    print("\n=== Comparison ===")
    print(f"{'Agent':<12} | {'AvgReward':>10} | {'Success%':>8}")
    print("-" * 36)
    print(
        f"{'Q-Learning':<12} | "
        f"{q_results['average_reward']:>10.2f} | "
        f"{q_results['success_rate'] * 100:>7.1f}%"
    )
    print(
        f"{'Random':<12} | "
        f"{rand_results['average_reward']:>10.2f} | "
        f"{rand_results['success_rate'] * 100:>7.1f}%"
    )
    print("=" * 36)
    print(
        "\nTabular Q-learning is expected to struggle here: the flat action\n"
        "space is large and the state space is combinatorial. This motivates\n"
        "function approximation (DQN) in a follow-up issue."
    )


if __name__ == "__main__":
    main()
