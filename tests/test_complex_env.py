"""Tests for ComplexBuildingEnv."""

from __future__ import annotations

import os
import random as rng_mod
import sys

import numpy as np
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from environment.complex_env import (
    ComplexBuildingEnv,
    ComplexEnvConfig,
    NOT_INSIDE,
    NO_TASK,
    TASK_INACTIVE,
    TASK_PICKED_UP,
)


def _deterministic_cfg(**overrides) -> ComplexEnvConfig:
    """Config with stochasticity disabled — for clean property tests."""
    base = dict(p_elevator_delay=0.0, p_breakdown=0.0, p_new_task=0.0)
    base.update(overrides)
    return ComplexEnvConfig(**base)


def _deliver_action_id(cfg: ComplexEnvConfig) -> int:
    """Per-robot action id for the 'deliver' sub-action."""
    return 1 + cfg.n_elevators + 1 + cfg.n_elevators * cfg.n_floors + cfg.n_tasks


# --- Shape & vector encoding ------------------------------------------------

def test_reset_returns_state_with_expected_shape():
    env = ComplexBuildingEnv(seed=0)
    state = env.reset()
    assert isinstance(state, tuple)
    # Per-entity tuples only; the step counter is intentionally excluded so
    # tabular Q-tables can revisit states (see ComplexBuildingEnv.get_state).
    expected_parts = env.cfg.n_robots + env.cfg.n_elevators + env.cfg.n_tasks
    assert len(state) == expected_parts


def test_to_vector_returns_ndarray_with_expected_length():
    env = ComplexBuildingEnv(seed=0)
    env.reset()
    v = env.to_vector()
    assert isinstance(v, np.ndarray)
    assert v.shape == (env.state_vector_size,)
    assert v.dtype == np.float32


# --- Validity / dynamics ---------------------------------------------------

def test_enter_on_different_floor_is_invalid():
    cfg = _deterministic_cfg()
    env = ComplexBuildingEnv(config=cfg, seed=0)
    env.reset()
    env.robots[0]["floor"] = 0
    env.robots[0]["inside_elevator"] = NOT_INSIDE
    env.elevators[0]["floor"] = 5
    env.elevators[0]["target_floor"] = 5

    # per-robot: robot 0 → enter elevator 0 (action id 1), robot 1 → wait
    action = [1] + [0] * (cfg.n_robots - 1)
    _, _, _, info = env.step(action)
    assert env.robots[0]["inside_elevator"] == NOT_INSIDE
    assert info["invalid_count"] >= 1


def test_robot_inside_elevator_moves_with_elevator():
    cfg = _deterministic_cfg()
    env = ComplexBuildingEnv(config=cfg, seed=0)
    env.reset()
    env.robots[0]["floor"] = 2
    env.robots[0]["inside_elevator"] = 0
    env.elevators[0]["floor"] = 2
    env.elevators[0]["target_floor"] = 5
    env.elevators[0]["broken_remaining"] = 0

    env.step([0] * cfg.n_robots)
    assert env.elevators[0]["floor"] == 3
    assert env.robots[0]["floor"] == 3


def test_elevator_delay_one_means_no_movement():
    cfg = ComplexEnvConfig(p_elevator_delay=1.0, p_breakdown=0.0, p_new_task=0.0)
    env = ComplexBuildingEnv(config=cfg, seed=0)
    env.reset()
    env.elevators[0]["floor"] = 0
    env.elevators[0]["target_floor"] = 7

    for _ in range(10):
        env.step([0] * cfg.n_robots)

    assert env.elevators[0]["floor"] == 0


def test_breakdown_ejects_carrying_robot():
    cfg = ComplexEnvConfig(
        p_elevator_delay=0.0,
        p_breakdown=1.0,
        p_new_task=0.0,
        breakdown_duration=3,
    )
    env = ComplexBuildingEnv(config=cfg, seed=0)
    env.reset()
    env.elevators[0]["floor"] = 4
    env.elevators[0]["broken_remaining"] = 0
    env.robots[0]["floor"] = 4
    env.robots[0]["inside_elevator"] = 0
    env.robots[0]["carrying_task"] = 0
    env.tasks[0]["status"] = TASK_PICKED_UP

    _, reward, _, info = env.step([0] * cfg.n_robots)
    assert env.robots[0]["inside_elevator"] == NOT_INSIDE
    assert env.robots[0]["floor"] == 4
    assert info["breakdowns"] >= 1
    # Reward must include the carrier ejection penalty.
    assert reward <= cfg.reward_breakdown_eject + 0.0


# --- Reward / termination --------------------------------------------------

def test_delivery_increments_total_and_gives_reward():
    cfg = _deterministic_cfg()
    env = ComplexBuildingEnv(config=cfg, seed=0)
    env.reset()
    env.tasks[0]["pickup_floor"] = 1
    env.tasks[0]["delivery_floor"] = 3
    env.tasks[0]["status"] = TASK_PICKED_UP
    env.robots[0]["floor"] = 3
    env.robots[0]["inside_elevator"] = NOT_INSIDE
    env.robots[0]["carrying_task"] = 0

    action = [_deliver_action_id(cfg)] + [0] * (cfg.n_robots - 1)
    _, reward, _, _ = env.step(action)
    assert env._total_delivered == 1
    assert env.tasks[0]["status"] == TASK_INACTIVE
    assert env.robots[0]["carrying_task"] == NO_TASK
    assert reward > 50  # +100 delivery dominates the small penalties


def test_all_deliveries_terminate_episode():
    cfg = _deterministic_cfg(
        n_floors=4,
        n_robots=1,
        n_elevators=1,
        n_tasks=1,
        total_task_budget=1,
    )
    env = ComplexBuildingEnv(config=cfg, seed=0)
    env.reset()
    env.tasks[0]["pickup_floor"] = 1
    env.tasks[0]["delivery_floor"] = 2
    env.tasks[0]["status"] = TASK_PICKED_UP
    env.robots[0]["floor"] = 2
    env.robots[0]["inside_elevator"] = NOT_INSIDE
    env.robots[0]["carrying_task"] = 0

    _, _, done, info = env.step([_deliver_action_id(cfg)])
    assert done is True
    assert info["success"] is True
    assert env.delivered is True


# --- Reproducibility -------------------------------------------------------

def test_seeded_reset_is_reproducible():
    cfg = ComplexEnvConfig()
    env1 = ComplexBuildingEnv(config=cfg, seed=42)
    env2 = ComplexBuildingEnv(config=cfg, seed=42)
    assert env1.get_state() == env2.get_state()
    for _ in range(20):
        env1.step(0)
        env2.step(0)
        assert env1.get_state() == env2.get_state()


# --- Smoke test ------------------------------------------------------------

def test_random_rollout_does_not_crash():
    env = ComplexBuildingEnv(seed=0, max_steps=50)
    env.reset()
    rng = rng_mod.Random(0)
    n = env.flat_action_space_size
    for _ in range(50):
        state, reward, done, info = env.step(rng.randrange(n))
        assert isinstance(state, tuple)
        for key in ("tasks_remaining", "breakdowns", "invalid_count", "success"):
            assert key in info
        if done:
            break


# --- Integration with existing pipeline ------------------------------------

def test_works_with_existing_train_evaluate_pipeline():
    """ComplexBuildingEnv satisfies EnvironmentProtocol — train/evaluate run unchanged.

    We don't assert any performance threshold here; tabular Q-learning is
    expected to do badly on this env. We're only proving the API contract.
    """
    from rl.q_learning_agent import QLearningAgent
    from rl.train import train_agent
    from rl.evaluate import evaluate_agent

    cfg = _deterministic_cfg(max_steps=30)
    env = ComplexBuildingEnv(config=cfg, seed=0)
    agent = QLearningAgent(action_space_size=env.action_space_size, seed=0)

    _, stats = train_agent(env, agent, episodes=5, max_steps=cfg.max_steps)
    assert len(stats["episode_rewards"]) == 5

    eval_env = ComplexBuildingEnv(config=cfg, seed=1)
    results = evaluate_agent(eval_env, agent, episodes=3, max_steps=cfg.max_steps)
    assert "average_reward" in results
