"""Entry point: build env+agent, train, evaluate, and save artifacts."""

from __future__ import annotations

import os
import sys

# Ensure the project root is importable when run directly.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from environment.simple_env import SimpleBuildingEnv
from rl.baselines import OptimalAgent, RandomAgent
from rl.q_learning_agent import QLearningAgent
from rl.train import train_agent
from rl.evaluate import evaluate_agent
from utils.logger import save_training_stats
from utils.plot import plot_training_curves


def _print_comparison(results: dict) -> None:
    """Pretty-print a side-by-side comparison of evaluation results."""
    print("\n=== Baseline Comparison ===")
    print(f"{'Agent':<12} | {'AvgReward':>10} | {'Success%':>8} | {'AvgSteps':>9}")
    print("-" * 50)
    for name, r in results.items():
        steps = r["average_steps_successful"]
        steps_str = f"{steps:.2f}" if steps == steps else "n/a"  # NaN check
        print(
            f"{name:<12} | {r['average_reward']:>10.2f} | "
            f"{r['success_rate'] * 100:>7.1f}% | {steps_str:>9}"
        )
    print("=" * 50)


def main() -> None:
    # Build environment and agent here (dependency injection).
    env = SimpleBuildingEnv(max_steps=config.MAX_STEPS, seed=config.TRAIN_SEED)
    agent = QLearningAgent(
        action_space_size=env.action_space_size,
        learning_rate=config.LEARNING_RATE,
        discount_factor=config.DISCOUNT_FACTOR,
        epsilon=config.EPSILON,
        epsilon_decay=config.EPSILON_DECAY,
        min_epsilon=config.MIN_EPSILON,
        seed=config.TRAIN_SEED,
    )

    print("Training Q-learning agent on SimpleBuildingEnv...")
    agent, stats = train_agent(
        env,
        agent,
        episodes=config.NUM_TRAINING_EPISODES,
        max_steps=config.MAX_STEPS,
    )

    q_path = os.path.join(PROJECT_ROOT, config.Q_TABLE_PATH)
    agent.save(q_path)
    print(f"\nSaved Q-table to {q_path}")

    stats_path = os.path.join(PROJECT_ROOT, config.TRAINING_STATS_PATH)
    save_training_stats(stats, stats_path)
    print(f"Saved training stats to {stats_path}")

    # Evaluate Q-agent and baselines on identically-seeded eval envs so every
    # agent sees the same target-floor sequence.
    def fresh_env() -> SimpleBuildingEnv:
        return SimpleBuildingEnv(max_steps=config.MAX_STEPS, seed=config.EVAL_SEED)

    agents = {
        "Q-Learning": agent,
        "Random": RandomAgent(env.action_space_size, seed=config.EVAL_SEED),
        "Optimal": OptimalAgent(env.action_space_size),
    }

    results: dict = {}
    for name, a in agents.items():
        print(f"\nEvaluating {name}...")
        results[name] = evaluate_agent(
            fresh_env(),
            a,
            episodes=config.NUM_EVALUATION_EPISODES,
            max_steps=config.MAX_STEPS,
        )

    _print_comparison(results)

    plot_path = os.path.join(PROJECT_ROOT, "training_curves.png")
    plot_training_curves(
        stats_path=stats_path,
        output_path=plot_path,
        optimal_reward=results["Optimal"]["average_reward"],
    )
    print(f"Saved training curves to {plot_path}")


if __name__ == "__main__":
    main()
