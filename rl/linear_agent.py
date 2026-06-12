"""Linear function-approximation Q-learning (NumPy only).

Drop-in alternative to MultiAgentTabularQ for ComplexBuildingEnv. Instead of a dict
Q-table keyed by exact observation tuples, it learns a weight matrix W of shape
(n_features, per_robot_action_size) and estimates
    Q(obs, a) = phi(obs) . W[:, a]
where phi(obs) one-hot encodes the per-robot observation.

This is the "next step beyond a bigger table": it generalizes across
observations and can consume a richer feature encoding (e.g. the full global
state) that a tabular method could never enumerate. Interface matches MultiAgentTabularQ
so train_agent / evaluate_agent work unchanged.
"""

from __future__ import annotations

import pickle
import random
from typing import Callable, Optional, Sequence, Tuple

import numpy as np

Observation = Tuple[int, ...]
Featurizer = Callable[[Observation], np.ndarray]


def _per_robot_action_size(cfg) -> int:
    """Mirror of ComplexBuildingEnv.per_robot_action_size, computed from cfg
    alone so the featurizer doesn't need a live env instance."""
    return (
        1                                       # wait
        + cfg.n_elevators                       # enter elevator i
        + 1                                     # exit
        + cfg.n_elevators * cfg.n_floors        # request (e, f)
        + cfg.n_tasks                           # pick up task slot
        + 1                                     # deliver
    )


def make_per_robot_featurizer(cfg) -> "tuple[Featurizer, int]":
    """One-hot featurizer for ComplexBuildingEnv per-robot observations.

    Layout (see ComplexBuildingEnv._per_robot_obs):
        own_floor, own_inside_elevator, own_carrying_task,
        [elev_floor, elev_broken] * n_elevators, pickup_target, delivery_target
    Each field is one-hot encoded over its value range and concatenated.
    Sentinel -1 values are handled by shifting each field by its minimum value.

    If cfg.include_last_action is True, a trailing one-hot field for the
    robot's previous action is appended automatically, so the featurizer stays
    aligned with the observation the env actually emits.

    Returns (featurize, n_features).
    """
    F, E, T = cfg.n_floors, cfg.n_elevators, cfg.n_tasks
    # (min_value, num_values) for every field, in observation order.
    fields = [
        (0, F),        # own_floor             0 .. F-1
        (-1, E + 1),   # own_inside_elevator  -1 .. E-1   (-1 = not inside)
        (-1, T + 1),   # own_carrying_task    -1 .. T-1   (-1 = not carrying)
    ]
    for _ in range(E):
        fields.append((0, F))   # elevator floor   0 .. F-1
        fields.append((0, 2))   # elevator broken  0 / 1
    fields.append((-1, F + 1))  # pickup_target    -1 .. F-1  (-1 = none / carrying)
    fields.append((-1, F + 1))  # delivery_target  -1 .. F-1  (-1 = not carrying)

    if getattr(cfg, "include_last_action", False):
        n_act = _per_robot_action_size(cfg)
        fields.append((-1, n_act + 1))  # last_action  -1 .. n_act-1  (-1 = episode start)

    sizes = [s for _, s in fields]
    offsets = np.cumsum([0] + sizes)[:-1]
    n_features = int(sum(sizes))

    def featurize(obs: Observation) -> np.ndarray:
        vec = np.zeros(n_features, dtype=np.float64)
        for value, (lo, size), off in zip(obs, fields, offsets):
            idx = int(value) - lo
            if 0 <= idx < size:     # guard against unexpected out-of-range values
                vec[off + idx] = 1.0
        return vec

    return featurize, n_features


class LinearMultiAgentQ:
    """Shared-weights linear Q-learner across robots. Mirrors MultiAgentTabularQ.

    Each robot looks up Q(obs, .) = phi(obs) @ W and acts epsilon-greedily;
    every robot's transition makes a semi-gradient TD update to the shared W,
    so experience is pooled exactly as in the tabular shared-table design.
    """

    def __init__(
        self,
        featurize: Featurizer,
        n_features: int,
        per_robot_action_size: int,
        n_robots: int,
        learning_rate: float = 0.05,   # smaller than tabular: FA needs gentler steps
        discount_factor: float = 0.90,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.9999,
        min_epsilon: float = 0.2,
        seed: Optional[int] = None,
    ) -> None:
        self.featurize = featurize
        self.n_features = n_features
        self.per_robot_action_size = per_robot_action_size
        self.n_robots = n_robots
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.W = np.zeros((n_features, per_robot_action_size), dtype=np.float64)
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

    # Logical flat action-space size for introspection. Not used internally.
    @property
    def action_space_size(self) -> int:
        return self.per_robot_action_size ** self.n_robots

    def q_values(self, obs: Observation) -> np.ndarray:
        """Q-values for every per-robot action: shape (per_robot_action_size,)."""
        return self.featurize(obs) @ self.W

    def _pick_one(self, obs: Observation, greedy: bool) -> int:
        if not greedy and self._rng.random() < self.epsilon:
            return self._rng.randrange(self.per_robot_action_size)
        q = self.q_values(obs)
        best = np.flatnonzero(q == q.max())
        return int(self._np_rng.choice(best))

    def choose_action(
        self,
        observations: Sequence[Observation],
        greedy: bool = False,
    ) -> Tuple[int, ...]:
        return tuple(self._pick_one(obs, greedy) for obs in observations)

    def update(
        self,
        observations: Sequence[Observation],
        actions: Sequence[int],
        reward: float,
        next_observations: Sequence[Observation],
        done: bool,
    ) -> None:
        for obs, action, next_obs in zip(observations, actions, next_observations):
            phi = self.featurize(obs)
            q_sa = float(phi @ self.W[:, action])
            next_max = 0.0 if done else float((self.featurize(next_obs) @ self.W).max())
            td_error = reward + self.gamma * next_max - q_sa
            self.W[:, action] += self.lr * td_error * phi   # semi-gradient step

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save(self, path: str) -> None:
        payload = {
            "W": self.W.tolist(),
            "n_features": self.n_features,
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
        self.W = np.asarray(payload["W"], dtype=np.float64)
        self.n_features = payload["n_features"]
        self.per_robot_action_size = payload["per_robot_action_size"]
        self.n_robots = payload["n_robots"]
        self.lr = payload["lr"]
        self.gamma = payload["gamma"]
        self.epsilon = payload["epsilon"]
        self.epsilon_decay = payload["epsilon_decay"]
        self.min_epsilon = payload["min_epsilon"]
