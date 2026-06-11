"""Multi-agent tabular Q-learning with a SHARED Q-table across robots.

Designed to pair with ``ComplexBuildingEnv(obs_mode='per_robot')``. Each step:
- The env returns a tuple of N per-robot observations
- The agent picks one action per robot via epsilon-greedy lookup in the shared Q-table
- The env consumes the per-robot action vector and returns a single team reward
- Each robot's (obs, action, reward, next_obs) transition triggers an
  independent Q-update against the shared table — i.e. independent learners
  sharing parameters and team reward.

Sharing the Q-table is sample-efficient: experience gathered by any robot
benefits all of them. The downside (no explicit coordination signal) is the
classic multi-agent credit-assignment issue, but works fine in practice for
this env.

The agent satisfies the ``AgentProtocol`` shape — ``choose_action(state)``
returns a tuple of ints which ``ComplexBuildingEnv.step`` accepts directly,
so ``train_agent`` / ``evaluate_agent`` work without modification.
"""

from __future__ import annotations

import pickle
import random
from typing import Any, Dict, Hashable, Optional, Sequence, Tuple

import numpy as np


Observation = Tuple[int, ...]


class MultiAgentTabularQ:
    def __init__(
        self,
        per_robot_action_size: int,
        n_robots: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.05,
        seed: Optional[int] = None,
    ) -> None:
        self.per_robot_action_size = per_robot_action_size
        self.n_robots = n_robots
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.q_table: Dict[Hashable, np.ndarray] = {}
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

    # Logical flat action-space size for any introspection. Not used internally.
    @property
    def action_space_size(self) -> int:
        return self.per_robot_action_size ** self.n_robots

    def get_q_values(self, obs: Hashable) -> np.ndarray:
        if obs not in self.q_table:
            self.q_table[obs] = np.zeros(self.per_robot_action_size, dtype=np.float64)
        return self.q_table[obs]

    def _pick_one(
        self,
        obs: Hashable,
        greedy: bool,
        temperature: Optional[float],
    ) -> int:
        q = self.get_q_values(obs)
        if greedy:
            return int(self._np_rng.choice(np.flatnonzero(q == q.max())))
        if temperature is not None:
            # Boltzmann/softmax sampling: P(a) ∝ exp(Q(a) / T). Subtract the
            # max for numerical stability. Low T → greedy, high T → uniform.
            p = np.exp((q - q.max()) / temperature)
            p /= p.sum()
            return int(self._np_rng.choice(len(q), p=p))
        if self._rng.random() < self.epsilon:
            return self._rng.randrange(self.per_robot_action_size)
        return int(self._np_rng.choice(np.flatnonzero(q == q.max())))

    def choose_action(
        self,
        observations: Sequence[Observation],
        greedy: bool = False,
        temperature: Optional[float] = None,
    ) -> Tuple[int, ...]:
        """Pick one action per robot. ``greedy`` takes precedence; otherwise a
        ``temperature`` selects Boltzmann/softmax sampling, and if neither is
        set the agent falls back to epsilon-greedy."""
        return tuple(self._pick_one(obs, greedy, temperature) for obs in observations)

    def update(
        self,
        observations: Sequence[Observation],
        actions: Sequence[int],
        reward: float,
        next_observations: Sequence[Observation],
        done: bool,
    ) -> None:
        for obs, action, next_obs in zip(observations, actions, next_observations):
            q = self.get_q_values(obs)
            next_max = 0.0 if done else float(self.get_q_values(next_obs).max())
            target = reward + self.gamma * next_max
            q[action] += self.lr * (target - q[action])

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save(self, path: str) -> None:
        payload: Dict[str, Any] = {
            "q_table": {k: v.tolist() for k, v in self.q_table.items()},
            "per_robot_action_size": self.per_robot_action_size,
            "n_robots": self.n_robots,
            "lr": self.lr,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "min_epsilon": self.min_epsilon,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self.q_table = {
            tuple(k) if isinstance(k, (list, tuple)) else k:
                np.asarray(v, dtype=np.float64)
            for k, v in payload["q_table"].items()
        }
        self.per_robot_action_size = payload["per_robot_action_size"]
        self.n_robots = payload["n_robots"]
        self.lr = payload["lr"]
        self.gamma = payload["gamma"]
        self.epsilon = payload["epsilon"]
        self.epsilon_decay = payload["epsilon_decay"]
        self.min_epsilon = payload["min_epsilon"]
