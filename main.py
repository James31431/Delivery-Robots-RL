"""Entry point: train, evaluate, and save artifacts for one of the two envs.

Two pipelines are kept here side-by-side so you can switch with one flag.

Usage:
    python main.py                  # default — SimpleBuildingEnv
    python main.py --env simple     # explicit
    python main.py --env complex    # the harder, stochastic, multi-robot env

Or change ``DEFAULT_ENV`` below to flip the default permanently.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Callable, Dict, Optional

# Ensure the project root is importable when run directly.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from environment.complex_env import ComplexBuildingEnv, ComplexEnvConfig
from environment.simple_env import SimpleBuildingEnv
from rl.baselines import OptimalAgent, RandomAgent
from rl.multi_agent_tabular import MultiAgentTabularQ
from rl.q_learning_agent import QLearningAgent
from rl.train import train_agent
from rl.evaluate import evaluate_agent
from utils.logger import save_training_stats
from utils.plot import plot_training_curves


# Change to "simple" to flip the default without editing the CLI invocation.
DEFAULT_ENV = "complex"


# --- Shared helpers --------------------------------------------------------

def _build_q_agent(action_space_size: int) -> QLearningAgent:
    return QLearningAgent(
        action_space_size=action_space_size,
        learning_rate=config.LEARNING_RATE,
        discount_factor=config.DISCOUNT_FACTOR,
        epsilon=config.EPSILON,
        epsilon_decay=config.EPSILON_DECAY,
        min_epsilon=config.MIN_EPSILON,
        seed=config.TRAIN_SEED,
    )


def _save_artifacts(agent: Any, stats: Dict[str, Any], env_name: str) -> str:
    q_path = os.path.join(PROJECT_ROOT, config.q_table_path(env_name))
    agent.save(q_path)
    print(f"\nSaved Q-table to {q_path}")
    stats_path = os.path.join(PROJECT_ROOT, config.training_stats_path(env_name))
    save_training_stats(stats, stats_path)
    print(f"Saved training stats to {stats_path}")
    return stats_path


def _evaluate_all(
    agents: Dict[str, Any],
    fresh_env_fn: Callable[[], Any],
    max_steps: int,
) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    for name, agent in agents.items():
        print(f"\nEvaluating {name}...")
        results[name] = evaluate_agent(
            fresh_env_fn(),
            agent,
            episodes=config.NUM_EVALUATION_EPISODES,
            max_steps=max_steps,
        )
    return results


def _print_comparison(results: Dict[str, Dict[str, float]]) -> None:
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


def _plot(stats_path: str, optimal_reward: Optional[float], env_name: str) -> None:
    plot_path = os.path.join(PROJECT_ROOT, config.training_curves_path(env_name))
    plot_training_curves(
        stats_path=stats_path,
        output_path=plot_path,
        optimal_reward=optimal_reward,
    )
    print(f"Saved training curves to {plot_path}")


# --- Pipelines -------------------------------------------------------------

def run_simple() -> None:
    """Pipeline against SimpleBuildingEnv: Q-Learning vs Random vs Optimal."""
    env = SimpleBuildingEnv(max_steps=config.MAX_STEPS, seed=config.TRAIN_SEED)
    agent = _build_q_agent(env.action_space_size)

    print("Training Q-learning agent on SimpleBuildingEnv...")
    agent, stats = train_agent(
        env,
        agent,
        episodes=config.NUM_TRAINING_EPISODES,
        max_steps=config.MAX_STEPS,
    )

    stats_path = _save_artifacts(agent, stats, "simple")

    def fresh_env() -> SimpleBuildingEnv:
        return SimpleBuildingEnv(max_steps=config.MAX_STEPS, seed=config.EVAL_SEED)

    agents = {
        "Q-Learning": agent,
        "Random": RandomAgent(env.action_space_size, seed=config.EVAL_SEED),
        "Optimal": OptimalAgent(env.action_space_size),
    }

    results = _evaluate_all(agents, fresh_env, config.MAX_STEPS)
    _print_comparison(results)
    _plot(stats_path, results["Optimal"]["average_reward"], "simple")


def run_complex() -> None:
    """Pipeline against ComplexBuildingEnv with per-robot partial observability.

    Uses ``obs_mode='per_robot'`` so each robot's observation is a small
    local tuple, and a multi-agent tabular Q-learner (shared Q-table) acts
    independently for each robot. The Random baseline keeps the same env
    config but emits a flat action int — the env decodes it into per-robot
    actions internally. OptimalAgent is intentionally omitted (hand-coded
    for SimpleBuildingEnv's state shape).
    """
    env_cfg = ComplexEnvConfig(obs_mode="per_robot", max_steps=500)
    env = ComplexBuildingEnv(config=env_cfg, seed=config.TRAIN_SEED)
    agent = MultiAgentTabularQ(
        per_robot_action_size=env.per_robot_action_size,
        n_robots=env_cfg.n_robots,
        learning_rate=config.LEARNING_RATE,
        discount_factor=config.DISCOUNT_FACTOR,
        epsilon=config.EPSILON,
        epsilon_decay=config.EPSILON_DECAY,
        min_epsilon=config.MIN_EPSILON,
        seed=config.TRAIN_SEED,
    )

    print("Training MultiAgentTabularQ on ComplexBuildingEnv (per-robot obs)...")
    print(
        f"  per_robot_action_size={env.per_robot_action_size}, "
        f"flat_action_space_size={env.flat_action_space_size}, "
        f"max_steps={env_cfg.max_steps}"
    )
    agent, stats = train_agent(
        env,
        agent,
        episodes=config.NUM_TRAINING_EPISODES,
        max_steps=env_cfg.max_steps,
    )

    stats_path = _save_artifacts(agent, stats, "complex")

    def fresh_env() -> ComplexBuildingEnv:
        return ComplexBuildingEnv(config=env_cfg, seed=config.EVAL_SEED)

    agents = {
        "MultiAgent-Q": agent,
        "Random": RandomAgent(env.action_space_size, seed=config.EVAL_SEED),
    }

    results = _evaluate_all(agents, fresh_env, env_cfg.max_steps)
    _print_comparison(results)
    _plot(stats_path, None, "complex")


# --- Entry point -----------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate Q-learning on a delivery-robot env."
    )
    parser.add_argument(
        "--env",
        choices=["simple", "complex"],
        default=DEFAULT_ENV,
        help=f"Which environment to use (default: {DEFAULT_ENV}).",
    )
    args = parser.parse_args()

    if args.env == "simple":
        run_simple()
    else:
        run_complex()


if __name__ == "__main__":
    main()
