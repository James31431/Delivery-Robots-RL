"""Tests for the RandomAgent and OptimalAgent baselines."""

from __future__ import annotations

import os
import sys

import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from environment.simple_env import SimpleBuildingEnv
from rl.baselines import OptimalAgent, RandomAgent
from rl.evaluate import evaluate_agent
from rl.interfaces import AgentProtocol


# --- OptimalAgent ---------------------------------------------------------

@pytest.mark.parametrize("target_floor", [1, 2, 3, 4])
def test_optimal_agent_solves_every_target(target_floor):
    """For each possible target, the optimal policy must deliver in the
    theoretical minimum of (target_floor + 3) steps with reward 100 - (steps - 1)."""
    env = SimpleBuildingEnv(max_steps=50)
    env.reset()
    env.target_floor = target_floor  # pin the target to make the test deterministic

    agent = OptimalAgent(env.action_space_size)
    state = env.get_state()
    total_reward = 0.0
    steps = 0
    done = False

    while not done and steps < 50:
        action = agent.choose_action(state, greedy=True)
        state, reward, done, _ = env.step(action)
        total_reward += reward
        steps += 1

    assert env.delivered == 1
    expected_steps = target_floor + 3  # enter, up*t, exit, deliver
    assert steps == expected_steps
    # Reward: (steps - 1) regular steps at -1, plus +100 on the final delivery.
    assert total_reward == 100.0 - (expected_steps - 1)


def test_optimal_agent_100_percent_over_eval():
    """OptimalAgent must hit 100% success across a full evaluation run."""
    env = SimpleBuildingEnv(max_steps=50, seed=123)
    agent = OptimalAgent(env.action_space_size)
    results = evaluate_agent(env, agent, episodes=100, max_steps=50)

    assert results["success_rate"] == 1.0
    # Avg successful-episode length lies in [4, 7] since targets are floors 1..4
    # → optimal steps are 4..7.
    assert 4.0 <= results["average_steps_successful"] <= 7.0


# --- RandomAgent ----------------------------------------------------------

def test_random_agent_is_seed_reproducible():
    """Two RandomAgents with the same seed must emit the same action sequence."""
    a1 = RandomAgent(action_space_size=6, seed=42)
    a2 = RandomAgent(action_space_size=6, seed=42)
    seq1 = [a1.choose_action(state=None) for _ in range(50)]
    seq2 = [a2.choose_action(state=None) for _ in range(50)]
    assert seq1 == seq2


def test_random_agent_respects_action_space_bounds():
    """Sampled actions must all fall inside [0, action_space_size)."""
    n_actions = 6
    agent = RandomAgent(action_space_size=n_actions, seed=0)
    for _ in range(500):
        a = agent.choose_action(state=None)
        assert 0 <= a < n_actions


# --- Protocol conformance -------------------------------------------------

@pytest.mark.parametrize(
    "agent",
    [RandomAgent(action_space_size=6, seed=0), OptimalAgent(action_space_size=6)],
)
def test_baselines_satisfy_agent_protocol(agent):
    """Both baselines must duck-type as AgentProtocol (choose_action + update)."""
    assert isinstance(agent, AgentProtocol)
