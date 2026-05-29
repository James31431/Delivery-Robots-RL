"""Multi-seed experiment runner.

Trains a fresh Q-learning agent on each of N seeds and evaluates it (plus the
Random and Optimal baselines) on a fixed eval seed. Reports mean ± std across
seeds, saves a per-(seed, agent) CSV, and produces a multi-seed plot.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import os
import sys
from typing import Dict, List

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from environment.simple_env import SimpleBuildingEnv
from rl.baselines import OptimalAgent, RandomAgent
from rl.evaluate import evaluate_agent
from rl.q_learning_agent import QLearningAgent
from rl.train import train_agent
from utils.logger import save_training_stats
from utils.plot import plot_multi_seed_curves


METRICS = ("average_reward", "success_rate", "average_steps_successful")


def _evaluate_silent(env, agent, **kwargs) -> dict:
    """Run evaluate_agent without polluting stdout — we print our own table."""
    with contextlib.redirect_stdout(io.StringIO()):
        return evaluate_agent(env, agent, **kwargs)


def run_one_seed(
    seed: int,
    episodes: int,
    max_steps: int,
    results_dir: str,
) -> Dict[str, dict]:
    """Train + evaluate all three agents for a single seed."""
    env = SimpleBuildingEnv(max_steps=max_steps, seed=seed)
    agent = QLearningAgent(
        action_space_size=env.action_space_size,
        learning_rate=config.LEARNING_RATE,
        discount_factor=config.DISCOUNT_FACTOR,
        epsilon=config.EPSILON,
        epsilon_decay=config.EPSILON_DECAY,
        min_epsilon=config.MIN_EPSILON,
        seed=seed,
    )

    print(f"\n--- Seed {seed} ---")
    agent, stats = train_agent(env, agent, episodes=episodes, max_steps=max_steps)

    stats_path = os.path.join(results_dir, f"training_stats_seed_{seed}.csv")
    save_training_stats(stats, stats_path)

    pairs = [
        ("Q-Learning", agent),
        ("Random", RandomAgent(env.action_space_size, seed=seed)),
        ("Optimal", OptimalAgent(env.action_space_size)),
    ]

    out: Dict[str, dict] = {}
    for name, a in pairs:
        eval_env = SimpleBuildingEnv(max_steps=max_steps, seed=config.EVAL_SEED)
        out[name] = _evaluate_silent(
            eval_env,
            a,
            episodes=config.NUM_EVALUATION_EPISODES,
            max_steps=max_steps,
        )
    return out


def aggregate(all_results: List[Dict[str, dict]]) -> Dict[str, Dict[str, tuple]]:
    """Compute (mean, std) per metric per agent across seeds."""
    agents = list(all_results[0].keys())
    out: Dict[str, Dict[str, tuple]] = {}
    for agent in agents:
        out[agent] = {}
        for metric in METRICS:
            vals = [r[agent][metric] for r in all_results]
            vals = [v for v in vals if v == v]  # drop NaN
            if not vals:
                out[agent][metric] = (float("nan"), float("nan"))
            else:
                out[agent][metric] = (float(np.mean(vals)), float(np.std(vals)))
    return out


def print_comparison_table(agg: Dict[str, Dict[str, tuple]], num_seeds: int) -> None:
    print(f"\n=== Multi-Seed Comparison (mean +/- std over {num_seeds} seeds) ===")
    header = f"{'Agent':<12} | {'AvgReward':>18} | {'Success %':>16} | {'AvgSteps':>14}"
    print(header)
    print("-" * len(header))
    for name, metrics in agg.items():
        rew_m, rew_s = metrics["average_reward"]
        suc_m, suc_s = metrics["success_rate"]
        stp_m, stp_s = metrics["average_steps_successful"]
        rew = f"{rew_m:7.2f} +/- {rew_s:5.2f}"
        suc = f"{suc_m * 100:5.1f}% +/- {suc_s * 100:4.1f}%"
        stp = f"{stp_m:5.2f} +/- {stp_s:4.2f}" if stp_m == stp_m else "n/a"
        print(f"{name:<12} | {rew:>18} | {suc:>16} | {stp:>14}")
    print("=" * len(header))


def save_per_seed_csv(all_results: List[Dict[str, dict]], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "agent", "avg_reward", "success_rate", "avg_steps_successful"])
        for seed, r in enumerate(all_results):
            for agent_name, metrics in r.items():
                writer.writerow([
                    seed,
                    agent_name,
                    metrics["average_reward"],
                    metrics["success_rate"],
                    metrics["average_steps_successful"],
                ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5, help="Number of training seeds (0..N-1).")
    parser.add_argument("--episodes", type=int, default=config.NUM_TRAINING_EPISODES)
    parser.add_argument("--max-steps", type=int, default=config.MAX_STEPS)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--plot", default="training_curves_multi_seed.png")
    args = parser.parse_args()

    results_dir = os.path.join(PROJECT_ROOT, args.results_dir)
    os.makedirs(results_dir, exist_ok=True)

    all_results = [
        run_one_seed(seed, args.episodes, args.max_steps, results_dir)
        for seed in range(args.seeds)
    ]

    agg = aggregate(all_results)
    print_comparison_table(agg, args.seeds)

    csv_path = os.path.join(results_dir, "experiment_results.csv")
    save_per_seed_csv(all_results, csv_path)
    print(f"\nSaved per-seed results to {csv_path}")

    plot_path = os.path.join(PROJECT_ROOT, args.plot)
    stat_paths = [
        os.path.join(results_dir, f"training_stats_seed_{s}.csv") for s in range(args.seeds)
    ]
    plot_multi_seed_curves(
        stat_paths,
        plot_path,
        optimal_reward=agg["Optimal"]["average_reward"][0],
    )
    print(f"Saved multi-seed plot to {plot_path}")


if __name__ == "__main__":
    main()
