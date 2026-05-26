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
from rl.q_learning_agent import QLearningAgent
from rl.train import train_agent
from rl.evaluate import evaluate_agent
from utils.logger import save_training_stats


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

    print("\nEvaluating trained agent...")
    eval_env = SimpleBuildingEnv(max_steps=config.MAX_STEPS, seed=config.EVAL_SEED)
    evaluate_agent(
        eval_env,
        agent,
        episodes=config.NUM_EVALUATION_EPISODES,
        max_steps=config.MAX_STEPS,
    )

    q_path = os.path.join(PROJECT_ROOT, config.Q_TABLE_PATH)
    agent.save(q_path)
    print(f"\nSaved Q-table to {q_path}")

    stats_path = os.path.join(PROJECT_ROOT, config.TRAINING_STATS_PATH)
    save_training_stats(stats, stats_path)
    print(f"Saved training stats to {stats_path}")


if __name__ == "__main__":
    main()
