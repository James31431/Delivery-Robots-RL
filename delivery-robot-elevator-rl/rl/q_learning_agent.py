"""Tabular Q-learning agent using a dictionary-based Q-table."""

from __future__ import annotations

import pickle
import random
from typing import Dict, Hashable, Optional, Tuple

import numpy as np


State = Tuple[int, ...]


class QLearningAgent:
    """A simple tabular Q-learning agent.

    The Q-table maps state tuples to a numpy array of Q-values, one per action.
    Unseen states are lazily initialized to zeros.
    """

    def __init__(
        self,
        action_space_size: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.05,
        seed: Optional[int] = None,
    ) -> None:
        self.action_space_size = action_space_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.q_table: Dict[Hashable, np.ndarray] = {}
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

    def get_q_values(self, state: Hashable) -> np.ndarray:
        """Return the Q-values for a state, initializing zeros if unseen."""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space_size, dtype=np.float64)
        return self.q_table[state]

    def choose_action(self, state: Hashable, greedy: bool = False) -> int:
        """Epsilon-greedy action selection. Set greedy=True to disable exploration."""
        if not greedy and self._rng.random() < self.epsilon:
            return self._rng.randrange(self.action_space_size)

        q_values = self.get_q_values(state)
        # Break ties randomly among the best actions
        max_q = q_values.max()
        best_actions = np.flatnonzero(q_values == max_q)
        return int(self._np_rng.choice(best_actions))

    def update(
        self,
        state: Hashable,
        action: int,
        reward: float,
        next_state: Hashable,
        done: bool,
    ) -> None:
        """Standard Q-learning update rule."""
        q_values = self.get_q_values(state)
        next_max = 0.0 if done else float(self.get_q_values(next_state).max())
        target = reward + self.gamma * next_max
        q_values[action] += self.lr * (target - q_values[action])

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save(self, path: str) -> None:
        """Persist the Q-table and hyperparameters to disk."""
        payload = {
            "q_table": {k: v.tolist() for k, v in self.q_table.items()},
            "action_space_size": self.action_space_size,
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
            tuple(k) if isinstance(k, (list, tuple)) else k: np.asarray(v, dtype=np.float64)
            for k, v in payload["q_table"].items()
        }
        self.action_space_size = payload["action_space_size"]
        self.lr = payload["lr"]
        self.gamma = payload["gamma"]
        self.epsilon = payload["epsilon"]
        self.epsilon_decay = payload["epsilon_decay"]
        self.min_epsilon = payload["min_epsilon"]
