"""Tests for partial-observation mode in ComplexBuildingEnv and the
multi-agent tabular Q-learning agent that consumes it."""

from __future__ import annotations

import os
import sys

import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from environment.complex_env import (
    ComplexBuildingEnv,
    ComplexEnvConfig,
    NO_TASK,
    NOT_INSIDE,
    TASK_PENDING,
    TASK_PICKED_UP,
)
from rl.evaluate import evaluate_agent
from rl.multi_agent_tabular import MultiAgentTabularQ
from rl.train import train_agent


# --- obs_mode validation ---------------------------------------------------

def test_invalid_obs_mode_raises():
    with pytest.raises(ValueError, match="obs_mode"):
        ComplexBuildingEnv(config=ComplexEnvConfig(obs_mode="bogus"))


# --- Full-mode preserves prior behavior ------------------------------------

def test_full_mode_state_shape_unchanged():
    env = ComplexBuildingEnv(config=ComplexEnvConfig(obs_mode="full"), seed=0)
    state = env.reset()
    expected = env.cfg.n_robots + env.cfg.n_elevators + env.cfg.n_tasks
    assert len(state) == expected


# --- Per-robot observation shape -------------------------------------------

def test_per_robot_mode_state_is_tuple_of_per_robot_observations():
    cfg = ComplexEnvConfig(obs_mode="per_robot")
    env = ComplexBuildingEnv(config=cfg, seed=0)
    state = env.reset()
    assert isinstance(state, tuple)
    assert len(state) == cfg.n_robots
    # Each per-robot obs has fixed length: 3 robot fields + 2 per elevator + 2 task targets
    expected_obs_len = 3 + 2 * cfg.n_elevators + 2
    for obs in state:
        assert isinstance(obs, tuple)
        assert len(obs) == expected_obs_len
        assert all(isinstance(v, int) for v in obs)


def test_per_robot_obs_reflects_carrying_task():
    cfg = ComplexEnvConfig(obs_mode="per_robot", p_elevator_delay=0.0,
                           p_breakdown=0.0, p_new_task=0.0)
    env = ComplexBuildingEnv(config=cfg, seed=0)
    env.reset()
    # Force robot 0 to be carrying task 0 with delivery_floor=5.
    env.tasks[0]["pickup_floor"] = 1
    env.tasks[0]["delivery_floor"] = 5
    env.tasks[0]["status"] = TASK_PICKED_UP
    env.robots[0]["carrying_task"] = 0
    env.robots[0]["inside_elevator"] = NOT_INSIDE
    env.robots[0]["floor"] = 1

    obs = env._per_robot_obs(0)
    # Last two fields: pickup_target == -1, delivery_target == 5.
    assert obs[-2] == -1
    assert obs[-1] == 5


def test_per_robot_obs_targets_nearest_pending_pickup():
    cfg = ComplexEnvConfig(obs_mode="per_robot", n_tasks=3,
                           p_elevator_delay=0.0, p_breakdown=0.0, p_new_task=0.0)
    env = ComplexBuildingEnv(config=cfg, seed=0)
    env.reset()
    env.robots[0]["floor"] = 4
    env.robots[0]["carrying_task"] = NO_TASK
    env.robots[0]["inside_elevator"] = NOT_INSIDE
    # Set up two pending tasks at floors 0 and 5; nearest to robot @ 4 is 5.
    env.tasks[0]["status"] = TASK_PENDING
    env.tasks[0]["pickup_floor"] = 0
    env.tasks[1]["status"] = TASK_PENDING
    env.tasks[1]["pickup_floor"] = 5
    env.tasks[2]["status"] = TASK_PENDING
    env.tasks[2]["pickup_floor"] = 7

    obs = env._per_robot_obs(0)
    assert obs[-2] == 5  # closest pickup floor
    assert obs[-1] == -1  # not carrying


# --- MultiAgentTabularQ basic behavior -------------------------------------

def test_multi_agent_choose_action_returns_tuple_of_ints():
    cfg = ComplexEnvConfig(obs_mode="per_robot")
    env = ComplexBuildingEnv(config=cfg, seed=0)
    state = env.reset()
    agent = MultiAgentTabularQ(
        per_robot_action_size=env.per_robot_action_size,
        n_robots=cfg.n_robots,
        seed=0,
    )
    action = agent.choose_action(state)
    assert isinstance(action, tuple)
    assert len(action) == cfg.n_robots
    for a in action:
        assert 0 <= a < env.per_robot_action_size


def test_multi_agent_update_populates_shared_table():
    cfg = ComplexEnvConfig(obs_mode="per_robot")
    env = ComplexBuildingEnv(config=cfg, seed=0)
    state = env.reset()
    agent = MultiAgentTabularQ(
        per_robot_action_size=env.per_robot_action_size,
        n_robots=cfg.n_robots,
        seed=0,
    )
    action = agent.choose_action(state)
    next_state, reward, _, _ = env.step(action)
    agent.update(state, action, reward, next_state, done=False)
    # Both robots' obs (and possibly their next_obs) populated entries.
    assert len(agent.q_table) >= 1
    # Entries are numpy arrays of per_robot_action_size length.
    for q in agent.q_table.values():
        assert q.shape == (env.per_robot_action_size,)


# --- Integration: full train+eval cycle ------------------------------------

def test_multi_agent_pipeline_runs_end_to_end():
    cfg = ComplexEnvConfig(obs_mode="per_robot", max_steps=80,
                           total_task_budget=2)
    env = ComplexBuildingEnv(config=cfg, seed=0)
    agent = MultiAgentTabularQ(
        per_robot_action_size=env.per_robot_action_size,
        n_robots=cfg.n_robots,
        learning_rate=0.3,
        epsilon_decay=0.99,
        seed=0,
    )
    _, stats = train_agent(env, agent, episodes=20, max_steps=cfg.max_steps)
    assert len(stats["episode_rewards"]) == 20

    eval_env = ComplexBuildingEnv(config=cfg, seed=1)
    results = evaluate_agent(eval_env, agent, episodes=3, max_steps=cfg.max_steps)
    assert "average_reward" in results


# --- Memory sanity: per-robot mode keeps the Q-table small ----------------

def test_per_robot_q_table_stays_bounded_relative_to_steps():
    """A 1000-step rollout should produce <<1000 unique per-robot obs."""
    cfg = ComplexEnvConfig(obs_mode="per_robot", max_steps=1000,
                           total_task_budget=6)
    env = ComplexBuildingEnv(config=cfg, seed=0)
    agent = MultiAgentTabularQ(
        per_robot_action_size=env.per_robot_action_size,
        n_robots=cfg.n_robots,
        seed=0,
    )
    state = env.reset()
    n_steps = 0
    done = False
    while not done and n_steps < 1000:
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.update(state, action, reward, next_state, done)
        state = next_state
        n_steps += 1
    # Memory sanity: a single 1000-step episode must not produce ~1 obs/step.
    assert len(agent.q_table) < n_steps  # well below 1:1 — typically much less
