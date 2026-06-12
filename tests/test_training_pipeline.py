"""Tests proving the training/evaluation pipeline accepts injected dependencies."""

from __future__ import annotations

import json
import os
import random
import sys

import pytest

# Make the project root importable regardless of pytest's cwd.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from environment.simple_env import SimpleBuildingEnv
from rl.evaluate import evaluate_agent
from rl.q_learning_agent import QLearningAgent
from rl.train import train_agent
from utils.logger import save_training_stats


def test_train_agent_accepts_injected_env_and_agent(tmp_path):
    """train_agent must use the env and agent passed in, not build its own."""
    env = SimpleBuildingEnv(max_steps=50, seed=0)
    agent = QLearningAgent(action_space_size=env.action_space_size, seed=0)

    returned_agent, stats = train_agent(env, agent, episodes=20, max_steps=50)

    # Same object identity proves dependency injection rather than re-creation.
    assert returned_agent is agent

    assert "episode_rewards" in stats
    assert "episode_steps" in stats
    assert "episode_successes" in stats
    assert "epsilon_history" in stats
    assert "average_reward_last_100" in stats
    assert "success_rate_last_100" in stats
    assert len(stats["episode_rewards"]) == 20
    assert len(stats["epsilon_history"]) == 20

    # Q-table should have been populated by training.
    assert len(agent.q_table) > 0


def test_evaluate_agent_with_baseline_random_policy():
    """A minimal baseline policy (no epsilon) should still work with evaluate_agent."""

    class RandomPolicy:
        def __init__(self, n_actions, seed=0):
            self.n_actions = n_actions
            self._rng = random.Random(seed)

        def choose_action(self, state):
            return self._rng.randrange(self.n_actions)

        def update(self, state, action, reward, next_state, done):
            pass  # baseline: doesn't learn

    env = SimpleBuildingEnv(max_steps=50, seed=7)
    policy = RandomPolicy(env.action_space_size, seed=7)

    results = evaluate_agent(env, policy, episodes=5, max_steps=50)
    assert set(results) == {
        "average_reward",
        "success_rate",
        "average_steps_successful",
        "episodes",
    }
    assert results["episodes"] == 5


def test_evaluate_agent_can_record_single_episode(tmp_path):
    class GreedyPolicy:
        def __init__(self, action_space_size):
            self.action_space_size = action_space_size

        def choose_action(self, state, greedy=False):
            return 0

        def update(self, state, action, reward, next_state, done):
            pass

    env = SimpleBuildingEnv(max_steps=5, seed=0)
    policy = GreedyPolicy(env.action_space_size)
    out = tmp_path / "episode.json"

    results = evaluate_agent(
        env,
        policy,
        episodes=1,
        max_steps=5,
        record_episode_path=str(out),
    )

    assert results["episodes"] == 1
    assert out.exists()

    payload = json.loads(out.read_text())
    assert payload["episode"] == 1
    assert payload["metadata"]["env_class"] == "SimpleBuildingEnv"
    assert payload["trajectory"][0]["info"]["phase"] == "reset"
    assert payload["trajectory"][0]["step"] == 0
    assert payload["trajectory"][-1]["step"] >= 1
    assert "reward" in payload


def test_save_training_stats_writes_csv(tmp_path):
    stats = {
        "episode_rewards": [1.0, 2.0, 3.0],
        "episode_steps": [4, 5, 6],
        "episode_successes": [0, 1, 1],
        "epsilon_history": [0.9, 0.8, 0.7],
    }
    out = tmp_path / "stats.csv"
    save_training_stats(stats, str(out))

    content = out.read_text().strip().splitlines()
    assert content[0] == "episode,reward,steps,success,epsilon"
    assert len(content) == 4  # header + 3 rows
    assert content[1].startswith("1,1.0,4,0,")
