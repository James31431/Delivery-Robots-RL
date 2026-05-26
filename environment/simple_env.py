"""Simple multi-floor building environment for a delivery robot using an elevator."""

from __future__ import annotations

import random
from typing import Optional, Tuple, Dict, Any


# Action constants
WAIT = 0
ENTER_ELEVATOR = 1
EXIT_ELEVATOR = 2
ELEVATOR_UP = 3
ELEVATOR_DOWN = 4
DELIVER_PACKAGE = 5

ACTION_NAMES = {
    WAIT: "wait",
    ENTER_ELEVATOR: "enter_elevator",
    EXIT_ELEVATOR: "exit_elevator",
    ELEVATOR_UP: "elevator_up",
    ELEVATOR_DOWN: "elevator_down",
    DELIVER_PACKAGE: "deliver_package",
}


State = Tuple[int, int, int, int, int]


class SimpleBuildingEnv:
    """A minimal multi-floor delivery environment.

    The world is a 5-floor building (floors 0..4) with one robot, one elevator,
    and one delivery task. The robot must reach the target floor (using the
    elevator if needed) and deliver the package.
    """

    NUM_FLOORS = 5
    ACTION_SPACE_SIZE = 6

    def __init__(self, max_steps: int = 50, seed: Optional[int] = None) -> None:
        self.max_steps = max_steps
        self._rng = random.Random(seed)

        # State variables (initialized in reset)
        self.robot_floor: int = 0
        self.target_floor: int = 1
        self.elevator_floor: int = 0
        self.robot_inside_elevator: int = 0
        self.delivered: int = 0
        self.steps: int = 0

    @property
    def action_space_size(self) -> int:
        return self.ACTION_SPACE_SIZE

    def seed(self, seed: Optional[int]) -> None:
        self._rng = random.Random(seed)

    def reset(self) -> State:
        """Reset the environment to a new episode and return the initial state."""
        self.robot_floor = 0
        self.elevator_floor = 0
        self.robot_inside_elevator = 0
        self.delivered = 0
        self.steps = 0
        # Target is any floor 1..4 (must differ from start)
        self.target_floor = self._rng.randint(1, self.NUM_FLOORS - 1)
        return self.get_state()

    def get_state(self) -> State:
        return (
            self.robot_floor,
            self.target_floor,
            self.elevator_floor,
            self.robot_inside_elevator,
            self.delivered,
        )

    def step(self, action: int) -> Tuple[State, float, bool, Dict[str, Any]]:
        """Apply an action and return (next_state, reward, done, info)."""
        if self.delivered:
            # Episode already finished; ignore further actions.
            return self.get_state(), 0.0, True, {"reason": "already_done"}

        self.steps += 1
        reward = -1.0  # default per-step cost
        info: Dict[str, Any] = {"action": ACTION_NAMES.get(action, "unknown")}
        invalid = False

        if action == WAIT:
            reward = -2.0

        elif action == ENTER_ELEVATOR:
            if (
                not self.robot_inside_elevator
                and self.robot_floor == self.elevator_floor
            ):
                self.robot_inside_elevator = 1
            else:
                invalid = True

        elif action == EXIT_ELEVATOR:
            if self.robot_inside_elevator:
                self.robot_inside_elevator = 0
                # Robot exits at the elevator's current floor
                self.robot_floor = self.elevator_floor
            else:
                invalid = True

        elif action == ELEVATOR_UP:
            if self.elevator_floor < self.NUM_FLOORS - 1:
                self.elevator_floor += 1
                if self.robot_inside_elevator:
                    self.robot_floor = self.elevator_floor
            else:
                invalid = True

        elif action == ELEVATOR_DOWN:
            if self.elevator_floor > 0:
                self.elevator_floor -= 1
                if self.robot_inside_elevator:
                    self.robot_floor = self.elevator_floor
            else:
                invalid = True

        elif action == DELIVER_PACKAGE:
            if (
                not self.robot_inside_elevator
                and self.robot_floor == self.target_floor
            ):
                self.delivered = 1
                reward = 100.0
            else:
                invalid = True

        else:
            invalid = True

        if invalid:
            reward = -5.0
            info["invalid"] = True

        done = bool(self.delivered) or self.steps >= self.max_steps
        info["steps"] = self.steps
        info["success"] = bool(self.delivered)
        return self.get_state(), reward, done, info

    def render(self) -> None:
        """Print a readable text view of the current state."""
        print("=" * 30)
        print(f"Step: {self.steps} / {self.max_steps}")
        print(f"Target floor: {self.target_floor}")
        print(f"Robot floor:    {self.robot_floor} "
              f"(inside elevator: {bool(self.robot_inside_elevator)})")
        print(f"Elevator floor: {self.elevator_floor}")
        print(f"Delivered: {bool(self.delivered)}")
        # Simple vertical building diagram (top = floor 4)
        for f in range(self.NUM_FLOORS - 1, -1, -1):
            marks = []
            if f == self.target_floor and not self.delivered:
                marks.append("T")
            if f == self.robot_floor and not self.robot_inside_elevator:
                marks.append("R")
            if f == self.elevator_floor:
                marks.append("E" + ("[R]" if self.robot_inside_elevator else ""))
            print(f"  Floor {f}: {' '.join(marks)}")
        print("=" * 30)
