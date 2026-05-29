"""Plot training curves from a training_stats.csv produced by utils.logger."""

from __future__ import annotations

import csv
import os
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")  # headless backend — works without a display
import matplotlib.pyplot as plt
import numpy as np


def _rolling_mean(values: List[float], window: int) -> np.ndarray:
    """Simple trailing rolling mean. Output length matches input."""
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return arr
    window = max(1, min(window, len(arr)))
    kernel = np.ones(window) / window
    # 'same' would center the window; we want a trailing average so use a
    # cumulative approach for the leading edge.
    out = np.convolve(arr, kernel, mode="full")[: len(arr)]
    # Correct the early elements where the window isn't yet full.
    for i in range(min(window, len(arr))):
        out[i] = arr[: i + 1].mean()
    return out


def _read_stats(path: str) -> dict:
    episodes: List[int] = []
    rewards: List[float] = []
    steps: List[float] = []
    successes: List[float] = []
    epsilons: List[float] = []

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            rewards.append(float(row["reward"]))
            steps.append(float(row["steps"]) if row["steps"] else float("nan"))
            successes.append(float(row["success"]) if row["success"] else 0.0)
            if row.get("epsilon"):
                epsilons.append(float(row["epsilon"]))

    return {
        "episodes": episodes,
        "rewards": rewards,
        "steps": steps,
        "successes": successes,
        "epsilons": epsilons,
    }


def plot_training_curves(
    stats_path: str,
    output_path: str = "training_curves.png",
    window: int = 100,
    optimal_reward: Optional[float] = None,
) -> str:
    """Read training_stats.csv and save a 2x2 figure with the standard curves.

    Parameters
    ----------
    stats_path : path to the CSV written by save_training_stats.
    output_path : where to save the PNG.
    window : rolling-mean window in episodes.
    optimal_reward : if provided, drawn as a dashed reference line on the
        reward plot (useful when you've evaluated a hand-coded optimal agent).
    """
    data = _read_stats(stats_path)
    episodes = data["episodes"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Training curves (rolling window = {window} episodes)")

    # Reward per episode.
    ax = axes[0, 0]
    ax.plot(episodes, data["rewards"], color="C0", alpha=0.25, label="per episode")
    ax.plot(episodes, _rolling_mean(data["rewards"], window), color="C0", label=f"rolling mean")
    if optimal_reward is not None:
        ax.axhline(optimal_reward, color="green", linestyle="--", label=f"optimal ({optimal_reward:.1f})")
    ax.set_title("Reward")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    # Success rate (rolling).
    ax = axes[0, 1]
    ax.plot(episodes, _rolling_mean(data["successes"], window), color="C2")
    ax.set_title("Success rate")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Rolling success rate")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # Steps per episode.
    ax = axes[1, 0]
    ax.plot(episodes, data["steps"], color="C1", alpha=0.25, label="per episode")
    ax.plot(episodes, _rolling_mean(data["steps"], window), color="C1", label="rolling mean")
    ax.set_title("Steps per episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Epsilon decay.
    ax = axes[1, 1]
    if data["epsilons"]:
        ax.plot(episodes[: len(data["epsilons"])], data["epsilons"], color="C3")
        ax.set_title("Exploration (epsilon)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Epsilon")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
    else:
        ax.set_title("Exploration (epsilon)")
        ax.text(0.5, 0.5, "no epsilon data", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


def plot_multi_seed_curves(
    stats_paths: List[str],
    output_path: str = "training_curves_multi_seed.png",
    window: int = 100,
    optimal_reward: Optional[float] = None,
) -> str:
    """Plot mean curves with ±1 std bands across multiple training runs.

    One CSV per seed (paths from run_experiments.py). Runs are aligned by
    episode index and trimmed to the shortest one if lengths differ.
    """
    runs = [_read_stats(p) for p in stats_paths]
    if not runs:
        raise ValueError("No stats files provided.")

    n = min(len(r["episodes"]) for r in runs)
    episodes = runs[0]["episodes"][:n]

    def _stack(key: str) -> np.ndarray:
        return np.array([_rolling_mean(r[key][:n], window) for r in runs])

    rewards = _stack("rewards")
    successes = _stack("successes")
    steps = _stack("steps")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Multi-seed training curves ({len(runs)} seeds, window={window})")

    def _band(ax, arr: np.ndarray, color: str, label: str) -> None:
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        ax.plot(episodes, mean, color=color, label=f"{label} (mean)")
        ax.fill_between(episodes, mean - std, mean + std, color=color, alpha=0.25, label="±1 std")

    ax = axes[0, 0]
    _band(ax, rewards, "C0", "Reward")
    if optimal_reward is not None:
        ax.axhline(optimal_reward, color="green", linestyle="--", label=f"optimal ({optimal_reward:.1f})")
    ax.set_title("Reward")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    _band(ax, successes, "C2", "Success rate")
    ax.set_title("Success rate")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Rolling success rate")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    _band(ax, steps, "C1", "Steps")
    ax.set_title("Steps per episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Epsilon decay is deterministic given the schedule, so plot one run.
    ax = axes[1, 1]
    if runs[0]["epsilons"]:
        eps = runs[0]["epsilons"][:n]
        ax.plot(episodes[: len(eps)], eps, color="C3")
        ax.set_title("Exploration (epsilon)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Epsilon")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
    else:
        ax.set_title("Exploration (epsilon)")
        ax.text(0.5, 0.5, "no epsilon data", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot training curves from a stats CSV.")
    parser.add_argument("--stats", default="training_stats.csv", help="Path to stats CSV.")
    parser.add_argument("--out", default="training_curves.png", help="Output PNG path.")
    parser.add_argument("--window", type=int, default=100, help="Rolling-mean window.")
    parser.add_argument("--optimal", type=float, default=None, help="Optional optimal-reward reference line.")
    args = parser.parse_args()

    if not os.path.exists(args.stats):
        raise SystemExit(f"Stats file not found: {args.stats}")

    out = plot_training_curves(args.stats, args.out, args.window, args.optimal)
    print(f"Saved {out}")
