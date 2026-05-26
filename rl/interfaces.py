"""Lightweight typing protocols for environments and agents.

These are duck-typing contracts used only for type hints. They let the
training/evaluation code accept any object that "looks like" an environment
or agent, without requiring inheritance from a common base class.
"""

from __future__ import annotations

from typing import Any, Protocol, Tuple, runtime_checkable


@runtime_checkable
class EnvironmentProtocol(Protocol):
    action_space_size: int

    def reset(self) -> Any: ...
    def step(self, action: int) -> Tuple[Any, float, bool, dict]: ...
    def render(self) -> None: ...


@runtime_checkable
class AgentProtocol(Protocol):
    def choose_action(self, state: Any) -> int: ...
    def update(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        done: bool,
    ) -> None: ...
