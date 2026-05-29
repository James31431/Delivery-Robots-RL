"""Baseline agents for comparison against the trained Q-learning agent.

Both agents satisfy ``AgentProtocol`` so they drop into ``evaluate_agent``
unchanged. They never learn, so ``update`` is a no-op.
"""

from __future__ import annotations

import random
from typing import Any, Optional, Tuple


# Action constants — must match environment/simple_env.py.
WAIT = 0
ENTER_ELEVATOR = 1
EXIT_ELEVATOR = 2
ELEVATOR_UP = 3
ELEVATOR_DOWN = 4
DELIVER_PACKAGE = 5


State = Tuple[int, int, int, int, int]


class RandomAgent:
    """Picks a uniformly random action every step. Performance floor."""

    def __init__(self, action_space_size: int, seed: Optional[int] = None) -> None:
        self.action_space_size = action_space_size
        self._rng = random.Random(seed)

    def choose_action(self, state: Any, greedy: bool = False) -> int:
        return self._rng.randrange(self.action_space_size)

    def update(self, *args: Any, **kwargs: Any) -> None:
        return None


class OptimalAgent:
    """Hand-coded optimal policy for SimpleBuildingEnv. Performance ceiling.

    For a target floor t starting from floor 0, the optimal trajectory is:
    enter, up x t, exit, deliver — i.e. t + 3 steps.
    """

    def __init__(self, action_space_size: int = 6) -> None:
        self.action_space_size = action_space_size

    def choose_action(self, state: State, greedy: bool = True) -> int:
        robot_floor, target_floor, elevator_floor, inside, delivered = state

        if delivered:
            return WAIT

        if not inside and robot_floor == target_floor:
            return DELIVER_PACKAGE

        if inside:
            if elevator_floor == target_floor:
                return EXIT_ELEVATOR
            return ELEVATOR_UP if elevator_floor < target_floor else ELEVATOR_DOWN

        if robot_floor == elevator_floor:
            return ENTER_ELEVATOR
        # Robot stranded outside elevator on a non-elevator floor — call it over.
        return ELEVATOR_UP if elevator_floor < robot_floor else ELEVATOR_DOWN

    def update(self, *args: Any, **kwargs: Any) -> None:
        return None
