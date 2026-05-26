"""CSV logging helpers for training experiments."""

from __future__ import annotations

import csv
from typing import Any, Dict


def save_training_stats(stats: Dict[str, Any], path: str = "training_stats.csv") -> None:
    """Write per-episode training stats to a CSV file.

    Columns: episode, reward, steps, success, epsilon.
    The epsilon column is left blank for agents that do not use epsilon.
    """
    rewards = stats.get("episode_rewards", [])
    steps_list = stats.get("episode_steps", [])
    successes = stats.get("episode_successes", [])
    epsilons = stats.get("epsilon_history", [])

    n = len(rewards)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "steps", "success", "epsilon"])
        for i in range(n):
            writer.writerow([
                i + 1,
                rewards[i],
                steps_list[i] if i < len(steps_list) else "",
                successes[i] if i < len(successes) else "",
                epsilons[i] if i < len(epsilons) else "",
            ])
