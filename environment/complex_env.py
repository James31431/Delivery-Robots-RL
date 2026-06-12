"""Complex multi-floor building environment.

A harder sibling to ``SimpleBuildingEnv`` featuring:

* Multiple robots, multiple elevators, a refilling task queue
* Stochastic elevator dynamics (per-step delay, random breakdowns that
  eject any robot inside)
* Configurable shape via ``ComplexEnvConfig``
* Same ``EnvironmentProtocol`` interface so ``train_agent`` /
  ``evaluate_agent`` work unchanged.

The flat action space is per-robot-actions ** n_robots; for defaults
(2 robots × 24 per-robot actions) this is 576. The state space is
combinatorially huge — tabular Q-learning is expected to do badly here,
which is the motivation for future function-approximation work.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


# --- Constants (kept as ints so the state remains hashable) ---------------

# Task slot statuses
TASK_INACTIVE = 0   # empty slot, available for spawn
TASK_PENDING = 1    # waiting to be picked up
TASK_PICKED_UP = 2  # being carried by a robot

# Elevator directions
DIR_IDLE = 0
DIR_UP = 1
DIR_DOWN = 2

# Sentinels
NOT_INSIDE = -1
NO_TASK = -1
NO_ACTION = -1


@dataclass
class ComplexEnvConfig:
    """All tunable knobs for ComplexBuildingEnv."""

    # Shape
    n_floors: int = 8
    n_robots: int = 2
    n_elevators: int = 2
    n_tasks: int = 3            # max simultaneous active tasks (queue cap)
    total_task_budget: int = 6  # tasks across the whole episode
    max_steps: int = 300

    # Stochastic dynamics
    p_elevator_delay: float = 0.10  # elevator stalls for one step
    p_breakdown: float = 0.01       # elevator breaks down (per step, per elevator)
    breakdown_duration: int = 5     # steps the elevator stays out of service
    p_new_task: float = 0.20        # chance per step to refill an INACTIVE slot

    # Observation mode: "full" returns the global tuple-of-tuples state;
    # "per_robot" returns a tuple of per-robot local observations (a POMDP),
    # which makes the observation space small enough for tabular methods.
    obs_mode: str = "full"

    # 1-step memory: append each robot's previous action to its per-robot
    # observation, turning the memoryless reactive policy into a 1-step-history
    # policy. Only affects obs_mode="per_robot". Multiplies the observation
    # space by per_robot_action_size, but disambiguates the aliased states
    # responsible for the stall/oscillation loops under greedy evaluation.
    include_last_action: bool = False

    # Reward weights
    reward_delivery: float = 100.0
    reward_pickup: float = 10.0
    reward_step: float = -0.5
    reward_wait: float = -2.0
    reward_invalid: float = -5.0
    reward_breakdown_eject: float = -20.0


# Type aliases
State = Tuple
ActionVec = Union[int, Sequence[int]]


class ComplexBuildingEnv:
    """Stochastic multi-robot delivery environment."""

    def __init__(
        self,
        config: Optional[ComplexEnvConfig] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        if config is None:
            config = ComplexEnvConfig(**kwargs)
        elif kwargs:
            raise TypeError("Pass either a ComplexEnvConfig or kwargs, not both.")
        if not 1 <= config.n_robots <= 4:
            raise ValueError("n_robots must be in 1..4")
        if config.obs_mode not in ("full", "per_robot"):
            raise ValueError(
                f"obs_mode must be 'full' or 'per_robot', got {config.obs_mode!r}"
            )
        self.cfg = config
        self.max_steps = config.max_steps
        self._rng = random.Random(seed)

        # State containers (filled by reset)
        self.steps: int = 0
        self._total_delivered: int = 0
        self._total_spawned: int = 0
        self.robots: List[Dict[str, int]] = []
        self.elevators: List[Dict[str, int]] = []
        self.tasks: List[Dict[str, int]] = []
        self._info_breakdowns_step: int = 0
        self._info_invalid_step: int = 0

        self.reset()

    # --- Sizes / properties -------------------------------------------------

    @property
    def per_robot_action_size(self) -> int:
        """Number of distinct sub-actions for one robot."""
        return (
            1                                                  # wait
            + self.cfg.n_elevators                             # enter elevator i
            + 1                                                # exit
            + self.cfg.n_elevators * self.cfg.n_floors         # request (e, f)
            + self.cfg.n_tasks                                 # pick up task slot
            + 1                                                # deliver
        )

    @property
    def flat_action_space_size(self) -> int:
        return self.per_robot_action_size ** self.cfg.n_robots

    # Protocol-facing alias used by agents that introspect the env.
    @property
    def action_space_size(self) -> int:
        return self.flat_action_space_size

    @property
    def state_vector_size(self) -> int:
        return (
            3 * self.cfg.n_robots
            + 3 * self.cfg.n_elevators
            + 3 * self.cfg.n_tasks
            + 1
        )

    @property
    def delivered(self) -> bool:
        """Whole-episode success (parity with SimpleBuildingEnv.delivered)."""
        return self._total_delivered >= self.cfg.total_task_budget

    # --- Seeding / reset ---------------------------------------------------

    def seed(self, seed: Optional[int]) -> None:
        self._rng = random.Random(seed)

    def reset(self) -> State:
        self.steps = 0
        self._total_delivered = 0
        self._total_spawned = 0
        self._info_breakdowns_step = 0
        self._info_invalid_step = 0

        self.robots = [
            {
                "floor": self._rng.randrange(self.cfg.n_floors),
                "inside_elevator": NOT_INSIDE,
                "carrying_task": NO_TASK,
                "last_action": NO_ACTION,
            }
            for _ in range(self.cfg.n_robots)
        ]

        self.elevators = []
        for _ in range(self.cfg.n_elevators):
            f = self._rng.randrange(self.cfg.n_floors)
            self.elevators.append({
                "floor": f,
                "target_floor": f,
                "direction": DIR_IDLE,
                "broken_remaining": 0,
            })

        # Allocate task slots; spawn into as many as the budget allows.
        self.tasks = [
            {"pickup_floor": 0, "delivery_floor": 0, "status": TASK_INACTIVE}
            for _ in range(self.cfg.n_tasks)
        ]
        n_initial = min(self.cfg.n_tasks, self.cfg.total_task_budget)
        for i in range(n_initial):
            self._spawn_into_slot(i)

        return self.get_state()

    def _spawn_into_slot(self, slot: int) -> None:
        if self._total_spawned >= self.cfg.total_task_budget:
            return
        pickup = self._rng.randrange(self.cfg.n_floors)
        delivery = self._rng.randrange(self.cfg.n_floors)
        while delivery == pickup:
            delivery = self._rng.randrange(self.cfg.n_floors)
        self.tasks[slot]["pickup_floor"] = pickup
        self.tasks[slot]["delivery_floor"] = delivery
        self.tasks[slot]["status"] = TASK_PENDING
        self._total_spawned += 1

    # --- State accessors ---------------------------------------------------

    def get_state(self) -> State:
        """Return the agent-facing observation.

        In ``obs_mode='full'`` this returns the joint tuple-of-tuples — the
        same shape across the whole multi-agent team. In ``obs_mode='per_robot'``
        it returns a tuple of per-robot local observations (one per robot),
        which is the right shape for a multi-agent tabular agent with a
        shared Q-table.

        ``self.steps`` is intentionally NOT included in the hashable state.
        Including a monotonic step counter would make every state unique and
        bloat the Q-table without helping decisions. ``to_vector()`` does
        include it because function approximators can use it productively.
        """
        if self.cfg.obs_mode == "per_robot":
            return self.get_observations()

        parts: List[Tuple] = []
        for r in self.robots:
            parts.append((r["floor"], r["inside_elevator"], r["carrying_task"]))
        for e in self.elevators:
            parts.append((e["floor"], e["direction"], e["broken_remaining"]))
        for t in self.tasks:
            parts.append((t["pickup_floor"], t["delivery_floor"], t["status"]))
        return tuple(parts)

    def get_observations(self) -> Tuple[Tuple[int, ...], ...]:
        """Per-robot local observations. One small tuple per robot."""
        return tuple(self._per_robot_obs(i) for i in range(self.cfg.n_robots))

    def _per_robot_obs(self, robot_idx: int) -> Tuple[int, ...]:
        """Compact local observation for one robot. Designed to be the same
        shape regardless of which robot's perspective it is, so a single
        shared Q-table can be used across all robots.

        Fields (for n_elevators=2): own_floor, own_inside_elevator,
        own_carrying_task, elev0_floor, elev0_broken, elev1_floor,
        elev1_broken, pickup_target (-1 if carrying or none), delivery_target
        (-1 if not carrying).
        """
        r = self.robots[robot_idx]
        obs: List[int] = [
            r["floor"],
            r["inside_elevator"],
            r["carrying_task"],
        ]
        for e in self.elevators:
            obs.append(e["floor"])
            obs.append(1 if e["broken_remaining"] > 0 else 0)

        if r["carrying_task"] != NO_TASK:
            obs.append(-1)
            obs.append(self.tasks[r["carrying_task"]]["delivery_floor"])
        else:
            best_pickup = -1
            best_dist: Optional[int] = None
            for t in self.tasks:
                if t["status"] != TASK_PENDING:
                    continue
                d = abs(t["pickup_floor"] - r["floor"])
                if best_dist is None or d < best_dist:
                    best_pickup = t["pickup_floor"]
                    best_dist = d
            obs.append(best_pickup)
            obs.append(-1)

        if self.cfg.include_last_action:
            obs.append(r["last_action"])
        return tuple(obs)

    def to_vector(self) -> np.ndarray:
        """Flat float32 array suitable for function-approximation agents."""
        out: List[float] = []
        for r in self.robots:
            out.extend([r["floor"], r["inside_elevator"], r["carrying_task"]])
        for e in self.elevators:
            out.extend([e["floor"], e["direction"], e["broken_remaining"]])
        for t in self.tasks:
            out.extend([t["pickup_floor"], t["delivery_floor"], t["status"]])
        out.append(self.steps)
        return np.asarray(out, dtype=np.float32)

    # --- Action decoding ---------------------------------------------------

    def _to_per_robot_actions(self, action: ActionVec) -> List[int]:
        """Accept a flat int or a per-robot sequence; always return per-robot list."""
        if isinstance(action, (int, np.integer)):
            n = self.per_robot_action_size
            x = int(action)
            actions: List[int] = []
            for _ in range(self.cfg.n_robots):
                actions.append(x % n)
                x //= n
            return actions
        actions = list(action)
        if len(actions) != self.cfg.n_robots:
            raise ValueError(
                f"Expected {self.cfg.n_robots} per-robot actions, got {len(actions)}"
            )
        return [int(a) for a in actions]

    def _decode_robot_action(self, a: int) -> Tuple[str, Dict[str, int]]:
        cfg = self.cfg
        if a < 0 or a >= self.per_robot_action_size:
            return ("invalid", {})
        if a == 0:
            return ("wait", {})
        a -= 1
        if a < cfg.n_elevators:
            return ("enter", {"elevator_id": a})
        a -= cfg.n_elevators
        if a == 0:
            return ("exit", {})
        a -= 1
        if a < cfg.n_elevators * cfg.n_floors:
            return ("request", {"elevator_id": a // cfg.n_floors,
                                "floor": a % cfg.n_floors})
        a -= cfg.n_elevators * cfg.n_floors
        if a < cfg.n_tasks:
            return ("pickup", {"task_slot": a})
        a -= cfg.n_tasks
        if a == 0:
            return ("deliver", {})
        return ("invalid", {})

    # --- Step --------------------------------------------------------------

    def step(self, action: ActionVec) -> Tuple[State, float, bool, Dict[str, Any]]:
        if self.delivered:
            return self.get_state(), 0.0, True, {"reason": "already_done"}

        self.steps += 1
        self._info_breakdowns_step = 0
        self._info_invalid_step = 0
        deliveries_before = self._total_delivered
        reward = 0.0

        per_robot = self._to_per_robot_actions(action)
        for robot_idx, a in enumerate(per_robot):
            reward += self._apply_robot_action(robot_idx, a)
            self.robots[robot_idx]["last_action"] = a

        reward += self._tick_elevators()
        self._spawn_step()

        # Per-step cost — once per robot per step, regardless of sub-action.
        reward += self.cfg.reward_step * self.cfg.n_robots

        done = self.delivered or self.steps >= self.max_steps
        info: Dict[str, Any] = {
            "tasks_remaining": self.cfg.total_task_budget - self._total_delivered,
            "delivered_total": self._total_delivered,
            "breakdowns": self._info_breakdowns_step,
            "invalid_count": self._info_invalid_step,
            "success": done and self.delivered,
            "success_this_step": self._total_delivered > deliveries_before,
            "steps": self.steps,
        }
        return self.get_state(), reward, done, info

    def _apply_robot_action(self, robot_idx: int, a: int) -> float:
        kind, params = self._decode_robot_action(a)
        cfg = self.cfg
        r = self.robots[robot_idx]
        reward = 0.0

        if kind == "wait":
            reward += cfg.reward_wait
            return reward

        if kind == "enter":
            elev = self.elevators[params["elevator_id"]]
            if (
                elev["broken_remaining"] == 0
                and r["inside_elevator"] == NOT_INSIDE
                and r["floor"] == elev["floor"]
            ):
                r["inside_elevator"] = params["elevator_id"]
                return reward
            return self._invalid()

        if kind == "exit":
            if r["inside_elevator"] != NOT_INSIDE:
                elev = self.elevators[r["inside_elevator"]]
                r["floor"] = elev["floor"]
                r["inside_elevator"] = NOT_INSIDE
                return reward
            return self._invalid()

        if kind == "request":
            elev = self.elevators[params["elevator_id"]]
            if elev["broken_remaining"] != 0:
                return self._invalid()
            target = params["floor"]
            elev["target_floor"] = target
            if target > elev["floor"]:
                elev["direction"] = DIR_UP
            elif target < elev["floor"]:
                elev["direction"] = DIR_DOWN
            else:
                elev["direction"] = DIR_IDLE
            return reward

        if kind == "pickup":
            slot = params["task_slot"]
            t = self.tasks[slot]
            if (
                r["inside_elevator"] == NOT_INSIDE
                and r["carrying_task"] == NO_TASK
                and t["status"] == TASK_PENDING
                and r["floor"] == t["pickup_floor"]
            ):
                t["status"] = TASK_PICKED_UP
                r["carrying_task"] = slot
                reward += cfg.reward_pickup
                return reward
            return self._invalid()

        if kind == "deliver":
            slot = r["carrying_task"]
            if (
                slot != NO_TASK
                and r["inside_elevator"] == NOT_INSIDE
                and r["floor"] == self.tasks[slot]["delivery_floor"]
            ):
                self.tasks[slot]["status"] = TASK_INACTIVE
                r["carrying_task"] = NO_TASK
                self._total_delivered += 1
                reward += cfg.reward_delivery
                return reward
            return self._invalid()

        return self._invalid()

    def _invalid(self) -> float:
        self._info_invalid_step += 1
        return self.cfg.reward_invalid

    def _tick_elevators(self) -> float:
        """Move each elevator one step toward its target; handle breakdowns.
        Returns the total ejection penalty for this step (already negative)."""
        penalty = 0.0
        for e_id, elev in enumerate(self.elevators):
            if elev["broken_remaining"] > 0:
                elev["broken_remaining"] -= 1
                elev["direction"] = DIR_IDLE
                continue

            if self._rng.random() < self.cfg.p_breakdown:
                elev["broken_remaining"] = self.cfg.breakdown_duration
                elev["direction"] = DIR_IDLE
                self._info_breakdowns_step += 1
                for r in self.robots:
                    if r["inside_elevator"] == e_id:
                        r["floor"] = elev["floor"]
                        r["inside_elevator"] = NOT_INSIDE
                        if r["carrying_task"] != NO_TASK:
                            penalty += self.cfg.reward_breakdown_eject
                continue

            if elev["target_floor"] == elev["floor"]:
                elev["direction"] = DIR_IDLE
                continue

            if self._rng.random() < self.cfg.p_elevator_delay:
                continue

            if elev["target_floor"] > elev["floor"]:
                elev["floor"] += 1
                elev["direction"] = DIR_UP
            else:
                elev["floor"] -= 1
                elev["direction"] = DIR_DOWN

            for r in self.robots:
                if r["inside_elevator"] == e_id:
                    r["floor"] = elev["floor"]

            if elev["floor"] == elev["target_floor"]:
                elev["direction"] = DIR_IDLE

        return penalty

    def _spawn_step(self) -> None:
        for slot, t in enumerate(self.tasks):
            if t["status"] != TASK_INACTIVE:
                continue
            if self._total_spawned >= self.cfg.total_task_budget:
                continue
            if self._rng.random() < self.cfg.p_new_task:
                self._spawn_into_slot(slot)

    # --- Render ------------------------------------------------------------

    def render(self) -> None:
        cfg = self.cfg
        print("=" * 70)
        print(
            f"Step: {self.steps}/{self.max_steps} | "
            f"Delivered: {self._total_delivered}/{cfg.total_task_budget} | "
            f"Spawned: {self._total_spawned}/{cfg.total_task_budget} | "
            f"Breakdowns this step: {self._info_breakdowns_step}"
        )
        print("-" * 70)
        dir_glyph = {DIR_IDLE: ".", DIR_UP: "^", DIR_DOWN: "v"}
        for f in range(cfg.n_floors - 1, -1, -1):
            parts: List[str] = []
            for e_id, elev in enumerate(self.elevators):
                if elev["floor"] == f:
                    glyph = dir_glyph[elev["direction"]]
                    broken = (
                        f"BROKEN({elev['broken_remaining']})"
                        if elev["broken_remaining"] > 0
                        else ""
                    )
                    inside = [
                        f"R{idx}"
                        for idx, r in enumerate(self.robots)
                        if r["inside_elevator"] == e_id
                    ]
                    inside_str = ("|" + ",".join(inside)) if inside else ""
                    parts.append(f"[E{e_id}{glyph}{broken}{inside_str}]")
            for idx, r in enumerate(self.robots):
                if r["floor"] == f and r["inside_elevator"] == NOT_INSIDE:
                    carry = (
                        f"(T{r['carrying_task']})"
                        if r["carrying_task"] != NO_TASK
                        else ""
                    )
                    parts.append(f"R{idx}{carry}")
            for slot, t in enumerate(self.tasks):
                if t["status"] == TASK_PENDING and t["pickup_floor"] == f:
                    parts.append(f"T{slot}(->{t['delivery_floor']})")
            print(f"  Floor {f}: " + " ".join(parts))
        print("=" * 70)
