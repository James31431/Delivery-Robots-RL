"""Basic tests for SimpleBuildingEnv."""

import os
import sys

import pytest

# Make sure the project root is importable when running pytest from any cwd.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from environment.simple_env import (
    SimpleBuildingEnv,
    ELEVATOR_UP,
    ELEVATOR_DOWN,
    DELIVER_PACKAGE,
)


def test_reset_returns_tuple_state():
    env = SimpleBuildingEnv(seed=0)
    state = env.reset()
    assert isinstance(state, tuple)
    assert len(state) == 5
    # All entries should be ints
    for v in state:
        assert isinstance(v, int)


def test_elevator_cannot_go_below_floor_0():
    env = SimpleBuildingEnv(seed=0)
    env.reset()
    assert env.elevator_floor == 0
    state, reward, _, info = env.step(ELEVATOR_DOWN)
    assert env.elevator_floor == 0
    assert reward == -5.0
    assert info.get("invalid") is True


def test_elevator_cannot_go_above_floor_4():
    env = SimpleBuildingEnv(seed=0)
    env.reset()
    # Move up to the top floor
    for _ in range(SimpleBuildingEnv.NUM_FLOORS - 1):
        env.step(ELEVATOR_UP)
    assert env.elevator_floor == SimpleBuildingEnv.NUM_FLOORS - 1
    _, reward, _, info = env.step(ELEVATOR_UP)
    assert env.elevator_floor == SimpleBuildingEnv.NUM_FLOORS - 1
    assert reward == -5.0
    assert info.get("invalid") is True


def test_invalid_delivery_gives_negative_reward():
    env = SimpleBuildingEnv(seed=0)
    state = env.reset()
    # Robot is at floor 0 and target is 1..4, so delivery here is invalid.
    assert state[0] != state[1]
    _, reward, done, info = env.step(DELIVER_PACKAGE)
    assert reward == -5.0
    assert info.get("invalid") is True
    assert env.delivered == 0
    assert done is False


def test_valid_delivery_ends_episode():
    env = SimpleBuildingEnv(seed=0)
    env.reset()
    # Force a known target for deterministic test
    env.target_floor = 0
    # Robot starts at floor 0, outside the elevator, target = 0 -> valid delivery.
    _, reward, done, info = env.step(DELIVER_PACKAGE)
    assert reward == 100.0
    assert done is True
    assert info["success"] is True
    assert env.delivered == 1
