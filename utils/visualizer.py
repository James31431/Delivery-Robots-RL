"""Pygame renderers for ComplexBuildingEnv and recorded episode replay.

The renderer is intentionally standalone: it does not change the environment
or the training loop. It can either be driven manually with the keyboard or
used from a small driver script that steps the environment and redraws after
each action.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from environment.complex_env import (
    ComplexBuildingEnv,
    ComplexEnvConfig,
    DIR_DOWN,
    DIR_IDLE,
    DIR_UP,
    NO_TASK,
    NOT_INSIDE,
    TASK_INACTIVE,
    TASK_PENDING,
    TASK_PICKED_UP,
)

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

try:
    import pygame
except ImportError:  # pragma: no cover - dependency is optional at import time.
    pygame = None  # type: ignore[assignment]


Color = tuple[int, int, int]

FLOOR_LINE: Color = (92, 104, 125)
ROBOT: Color = (98, 214, 150)
ELEVATOR: Color = (75, 141, 222)
ELEVATOR_DARK: Color = (46, 82, 136)
SUCCESS: Color = (84, 189, 86)

COMPLEX_BACKGROUND: Color = (16, 20, 29)
COMPLEX_PANEL: Color = (24, 29, 42)
COMPLEX_BUILDING: Color = (21, 27, 39)
COMPLEX_SIDEBAR: Color = (26, 32, 46)
COMPLEX_FLOOR_A: Color = (31, 38, 53)
COMPLEX_FLOOR_B: Color = (27, 34, 48)
COMPLEX_SHADOW: Color = (12, 16, 24)
COMPLEX_BROKEN: Color = (221, 88, 84)
COMPLEX_PENDING: Color = (239, 194, 74)
COMPLEX_PICKED: Color = (123, 196, 255)
COMPLEX_INACTIVE: Color = (122, 133, 149)
COMPLEX_TEXT: Color = (233, 238, 245)
COMPLEX_MUTED: Color = (171, 179, 193)
ROBOT_PALETTE: tuple[Color, ...] = (
    (98, 214, 150),
    (110, 168, 255),
    (255, 157, 92),
    (203, 124, 255),
)


@dataclass(frozen=True)
class ComplexRenderGeometry:
    width: int = 1320
    height: int = 880
    margin: int = 28
    top_panel_height: int = 128
    sidebar_width: int = 320


class ComplexBuildingPygameRenderer:
    """Draw ComplexBuildingEnv as a multi-elevator vertical building."""

    def __init__(
        self,
        env: ComplexBuildingEnv,
        geometry: ComplexRenderGeometry = ComplexRenderGeometry(),
        title: str = "ComplexBuildingEnv Renderer",
    ) -> None:
        if pygame is None:
            raise RuntimeError(
                "pygame is not installed. Install it with `pip install -r requirements.txt`."
            )

        self.env = env
        self.geometry = geometry
        self.title = title
        self._screen = None
        self._clock = None
        self._font = None
        self._small_font = None
        self._tiny_font = None

    def _ensure_window(self) -> None:
        if self._screen is not None:
            return
        pygame.init()
        self._screen = pygame.display.set_mode(
            (self.geometry.width, self.geometry.height)
        )
        pygame.display.set_caption(self.title)
        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont("arial", 22)
        self._small_font = pygame.font.SysFont("arial", 18)
        self._tiny_font = pygame.font.SysFont("arial", 15)

    def _draw_text(
        self,
        text: str,
        x: int,
        y: int,
        color: Color = COMPLEX_TEXT,
        *,
        small: bool = False,
        tiny: bool = False,
    ) -> None:
        font = self._font
        if tiny:
            font = self._tiny_font
        elif small:
            font = self._small_font
        assert font is not None
        surface = font.render(text, True, color)
        self._screen.blit(surface, (x, y))

    def _layout(self) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
        left = self.geometry.margin
        top = self.geometry.top_panel_height + self.geometry.margin // 2
        main_width = self.geometry.width - self.geometry.sidebar_width - 3 * self.geometry.margin
        main_height = self.geometry.height - top - self.geometry.margin
        sidebar_left = left + main_width + self.geometry.margin
        sidebar_height = main_height
        return (
            (left, top, main_width, main_height),
            (sidebar_left, top, self.geometry.sidebar_width, sidebar_height),
        )

    def _floor_bounds(self, floor: int) -> tuple[int, int]:
        (left, top, width, height), _ = self._layout()
        floor_height = height / self.env.cfg.n_floors
        y1 = int(top + (self.env.cfg.n_floors - floor - 1) * floor_height)
        y2 = int(y1 + floor_height)
        return y1, y2

    def _floor_center(self, floor: int) -> int:
        y1, y2 = self._floor_bounds(floor)
        return (y1 + y2) // 2

    def _shaft_geometry(self, elevator_idx: int) -> tuple[int, int]:
        (left, _, width, _), _ = self._layout()
        shaft_count = max(1, self.env.cfg.n_elevators)
        available = max(220, width - 150)
        shaft_spacing = available / shaft_count
        shaft_center = left + 120 + int(shaft_spacing * (elevator_idx + 0.5))
        shaft_width = min(84, max(56, int(shaft_spacing * 0.42)))
        return shaft_center, shaft_width

    def _draw_panel(self) -> None:
        assert self._screen is not None
        pygame.draw.rect(
            self._screen,
            COMPLEX_PANEL,
            (0, 0, self.geometry.width, self.geometry.top_panel_height),
        )
        delivered_total = getattr(self.env, "_total_delivered", 0)
        spawned_total = getattr(self.env, "_total_spawned", 0)
        self._draw_text("ComplexBuildingEnv", 24, 16)
        self._draw_text(
            f"Step {self.env.steps}/{self.env.max_steps}",
            24,
            52,
            COMPLEX_MUTED,
            small=True,
        )
        self._draw_text(
            f"Delivered {delivered_total}/{self.env.cfg.total_task_budget}",
            210,
            52,
            SUCCESS,
            small=True,
        )
        self._draw_text(
            f"Spawned {spawned_total}/{self.env.cfg.total_task_budget}",
            420,
            52,
            COMPLEX_PENDING,
            small=True,
        )
        self._draw_text(
            f"Breakdowns {self.env._info_breakdowns_step}",
            620,
            52,
            COMPLEX_BROKEN if self.env._info_breakdowns_step else COMPLEX_MUTED,
            small=True,
        )
        self._draw_text(
            f"Invalid {self.env._info_invalid_step}",
            770,
            52,
            COMPLEX_BROKEN if self.env._info_invalid_step else COMPLEX_MUTED,
            small=True,
        )
        self._draw_text(
            "Space/Enter step | A autoplay | R reset | Esc quit",
            24,
            82,
            COMPLEX_MUTED,
            small=True,
        )

    def _draw_building(self) -> None:
        assert self._screen is not None
        (left, top, width, height), _ = self._layout()
        pygame.draw.rect(
            self._screen,
            COMPLEX_SHADOW,
            (left + 12, top + 10, width, height),
            border_radius=14,
        )
        pygame.draw.rect(
            self._screen,
            COMPLEX_BUILDING,
            (left, top, width, height),
            border_radius=14,
        )
        pygame.draw.rect(
            self._screen,
            FLOOR_LINE,
            (left, top, width, height),
            width=2,
            border_radius=14,
        )

        for floor in range(self.env.cfg.n_floors):
            y1, y2 = self._floor_bounds(floor)
            band_color = COMPLEX_FLOOR_A if floor % 2 == 0 else COMPLEX_FLOOR_B
            floor_rect = pygame.Rect(left + 2, y1, width - 4, y2 - y1)
            pygame.draw.rect(self._screen, band_color, floor_rect)

            pygame.draw.line(
                self._screen,
                FLOOR_LINE,
                (left, y1),
                (left + width, y1),
                2,
            )
            self._draw_text(f"F{floor}", left + 14, y1 + 8, COMPLEX_MUTED, small=True)

        pygame.draw.line(
            self._screen,
            FLOOR_LINE,
            (left, top + height),
            (left + width, top + height),
            2,
        )

    def _draw_elevators(self) -> None:
        assert self._screen is not None
        (left, top, width, height), _ = self._layout()
        shaft_top = top + 12
        shaft_height = height - 24

        for elevator_idx, elevator in enumerate(self.env.elevators):
            shaft_center, shaft_width = self._shaft_geometry(elevator_idx)
            shaft_x = shaft_center - shaft_width // 2
            shaft_rect = pygame.Rect(shaft_x, shaft_top, shaft_width, shaft_height)
            pygame.draw.rect(
                self._screen,
                (38, 47, 66),
                shaft_rect,
                border_radius=8,
            )
            pygame.draw.rect(
                self._screen,
                FLOOR_LINE,
                shaft_rect,
                width=2,
                border_radius=8,
            )

            y1, y2 = self._floor_bounds(elevator["floor"])
            car_rect = pygame.Rect(shaft_x + 7, y1 + 6, shaft_width - 14, (y2 - y1) - 12)
            car_color = COMPLEX_BROKEN if elevator["broken_remaining"] > 0 else ELEVATOR
            pygame.draw.rect(self._screen, car_color, car_rect, border_radius=8)
            pygame.draw.rect(
                self._screen,
                (240, 245, 251),
                car_rect,
                width=2,
                border_radius=8,
            )

            if elevator["broken_remaining"] > 0:
                overlay = pygame.Surface((car_rect.width, car_rect.height), pygame.SRCALPHA)
                overlay.fill((255, 80, 80, 60))
                self._screen.blit(overlay, car_rect.topleft)

            direction = {DIR_IDLE: ".", DIR_UP: "^", DIR_DOWN: "v"}[elevator["direction"]]
            self._draw_text(
                f"E{elevator_idx}{direction}",
                car_rect.x + 6,
                car_rect.y + 8,
                (12, 16, 23),
                small=True,
            )
            if elevator["broken_remaining"] > 0:
                self._draw_text(
                    f"B{elevator['broken_remaining']}",
                    car_rect.x + 6,
                    car_rect.y + 28,
                    (12, 16, 23),
                    small=True,
                )

    def _draw_tasks(self) -> None:
        assert self._screen is not None
        (left, _, width, _), (sidebar_left, sidebar_top, sidebar_width, _) = self._layout()
        for slot, task in enumerate(self.env.tasks):
            if task["status"] == TASK_INACTIVE:
                continue
            y1, y2 = self._floor_bounds(task["pickup_floor"])
            marker_x = left + 58
            marker_y = (y1 + y2) // 2
            if task["status"] == TASK_PENDING:
                color = COMPLEX_PENDING
                label = f"T{slot}->{task['delivery_floor']}"
            else:
                color = COMPLEX_PICKED
                label = f"T{slot} carried"
            pygame.draw.rect(
                self._screen,
                color,
                (marker_x, marker_y - 12, 34, 24),
                border_radius=4,
            )
            pygame.draw.rect(
                self._screen,
                (15, 20, 28),
                (marker_x, marker_y - 12, 34, 24),
                width=2,
                border_radius=4,
            )
            self._draw_text(f"T{slot}", marker_x + 6, marker_y - 9, (15, 20, 28), small=True)
            self._draw_text(label, marker_x + 48, marker_y - 10, COMPLEX_TEXT, tiny=True)

        # Sidebar task queue.
        pygame.draw.rect(
            self._screen,
            COMPLEX_SIDEBAR,
            (sidebar_left, sidebar_top, sidebar_width, self.geometry.height - sidebar_top - self.geometry.margin),
            border_radius=14,
        )
        pygame.draw.rect(
            self._screen,
            FLOOR_LINE,
            (sidebar_left, sidebar_top, sidebar_width, self.geometry.height - sidebar_top - self.geometry.margin),
            width=2,
            border_radius=14,
        )
        self._draw_text("Task queue", sidebar_left + 16, sidebar_top + 12)

        row_y = sidebar_top + 48
        for slot, task in enumerate(self.env.tasks):
            if task["status"] == TASK_INACTIVE:
                status = "inactive"
                color = COMPLEX_INACTIVE
            elif task["status"] == TASK_PENDING:
                status = f"pickup {task['pickup_floor']}"
                color = COMPLEX_PENDING
            else:
                status = f"carried by R{next((idx for idx, robot in enumerate(self.env.robots) if robot['carrying_task'] == slot), '?')}"
                color = COMPLEX_PICKED
            row = f"T{slot}: {task['pickup_floor']} -> {task['delivery_floor']}"
            self._draw_text(row, sidebar_left + 16, row_y, color, small=True)
            self._draw_text(status, sidebar_left + 16, row_y + 20, COMPLEX_MUTED, tiny=True)
            row_y += 46

    def _draw_robots(self) -> None:
        assert self._screen is not None
        (left, _, width, _), _ = self._layout()
        floor_groups: dict[int, list[tuple[int, dict[str, int]]]] = {}
        elevator_groups: dict[int, list[tuple[int, dict[str, int]]]] = {}
        for robot_idx, robot in enumerate(self.env.robots):
            if robot["inside_elevator"] == NOT_INSIDE:
                floor_groups.setdefault(robot["floor"], []).append((robot_idx, robot))
            else:
                elevator_groups.setdefault(robot["inside_elevator"], []).append((robot_idx, robot))

        for floor, robots in floor_groups.items():
            y1, y2 = self._floor_bounds(floor)
            center_y = (y1 + y2) // 2
            base_x = left + 92
            for offset, (robot_idx, robot) in enumerate(robots):
                color = ROBOT_PALETTE[robot_idx % len(ROBOT_PALETTE)]
                center_x = base_x + offset * 44
                pygame.draw.circle(self._screen, color, (center_x, center_y), 18)
                pygame.draw.circle(self._screen, (14, 18, 25), (center_x, center_y), 18, 2)
                self._draw_text(f"R{robot_idx}", center_x - 12, center_y - 10, (14, 18, 25), small=True)
                if robot["carrying_task"] != NO_TASK:
                    self._draw_text(
                        f"T{robot['carrying_task']}",
                        center_x + 20,
                        center_y - 10,
                        COMPLEX_PICKED,
                        tiny=True,
                    )

        for elevator_idx, robots in elevator_groups.items():
            shaft_center, shaft_width = self._shaft_geometry(elevator_idx)
            elevator = self.env.elevators[elevator_idx]
            y1, y2 = self._floor_bounds(elevator["floor"])
            car_rect = pygame.Rect(
                shaft_center - shaft_width // 2 + 7,
                y1 + 6,
                shaft_width - 14,
                (y2 - y1) - 12,
            )
            for slot, (robot_idx, robot) in enumerate(robots):
                color = ROBOT_PALETTE[robot_idx % len(ROBOT_PALETTE)]
                center_x = car_rect.centerx - 18 + (slot % 2) * 32
                center_y = car_rect.centery - 10 + (slot // 2) * 22
                pygame.draw.circle(self._screen, color, (center_x, center_y), 14)
                pygame.draw.circle(self._screen, (14, 18, 25), (center_x, center_y), 14, 2)
                self._draw_text(f"R{robot_idx}", center_x - 11, center_y - 9, (14, 18, 25), tiny=True)
                if robot["carrying_task"] != NO_TASK:
                    self._draw_text(
                        f"T{robot['carrying_task']}",
                        center_x + 16,
                        center_y - 9,
                        COMPLEX_PICKED,
                        tiny=True,
                    )

    def _draw_legend(self) -> None:
        assert self._screen is not None
        _, (sidebar_left, sidebar_top, sidebar_width, _) = self._layout()
        legend_y = self.geometry.height - 118
        self._draw_text("Legend", sidebar_left + 16, legend_y, COMPLEX_TEXT)
        self._draw_text("Robot", sidebar_left + 16, legend_y + 28, COMPLEX_MUTED, tiny=True)
        self._draw_text("Task pending", sidebar_left + 120, legend_y + 28, COMPLEX_PENDING, tiny=True)
        self._draw_text("Task carried", sidebar_left + 16, legend_y + 52, COMPLEX_PICKED, tiny=True)
        self._draw_text("Broken elevator", sidebar_left + 120, legend_y + 52, COMPLEX_BROKEN, tiny=True)

    def draw(self) -> None:
        self._ensure_window()
        assert self._screen is not None
        self._screen.fill(COMPLEX_BACKGROUND)
        self._draw_panel()
        self._draw_building()
        self._draw_tasks()
        self._draw_elevators()
        self._draw_robots()
        self._draw_legend()
        if self.env.delivered:
            self._draw_text("Episode complete", 24, 108, SUCCESS)
        pygame.display.flip()

    def close(self) -> None:
        if pygame is not None and self._screen is not None:
            pygame.quit()
        self._screen = None
        self._clock = None
        self._font = None
        self._small_font = None
        self._tiny_font = None


class EpisodeReplayRenderer:
    """Replay a recorded evaluation episode from a JSON file."""

    def __init__(
        self,
        record_path: str,
        geometry: Optional[ComplexRenderGeometry] = None,
        title: str = "Episode Replay",
    ) -> None:
        if pygame is None:
            raise RuntimeError(
                "pygame is not installed. Install it with `pip install -r requirements.txt`."
            )

        self.record_path = Path(record_path)
        self.geometry = geometry or ComplexRenderGeometry()
        self.title = title
        self._screen = None
        self._clock = None
        self._font = None
        self._small_font = None
        self._tiny_font = None
        self._payload: dict[str, Any] = {}
        self._trajectory: list[dict[str, Any]] = []
        self._index = 0
        self._paused = True
        self._env_class = "ComplexBuildingEnv"
        self._load()

    def _load(self) -> None:
        with self.record_path.open("r", encoding="utf-8") as handle:
            self._payload = json.load(handle)
        self._trajectory = list(self._payload.get("trajectory", []))
        if not self._trajectory:
            raise ValueError(f"No trajectory found in {self.record_path}")
        self._env_class = str(self._payload.get("metadata", {}).get("env_class", "ComplexBuildingEnv"))
        if self._env_class != "ComplexBuildingEnv":
            raise ValueError(
                f"EpisodeReplayRenderer only supports ComplexBuildingEnv episodes, got {self._env_class!r}"
            )

    def _ensure_window(self) -> None:
        if self._screen is not None:
            return
        pygame.init()
        self._screen = pygame.display.set_mode((self.geometry.width, self.geometry.height))
        pygame.display.set_caption(self.title)
        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont("arial", 22)
        self._small_font = pygame.font.SysFont("arial", 18)
        self._tiny_font = pygame.font.SysFont("arial", 15)

    def _draw_text(
        self,
        text: str,
        x: int,
        y: int,
        color: Color = COMPLEX_TEXT,
        *,
        small: bool = False,
        tiny: bool = False,
    ) -> None:
        font = self._font
        if tiny:
            font = self._tiny_font
        elif small:
            font = self._small_font
        assert font is not None
        surface = font.render(text, True, color)
        self._screen.blit(surface, (x, y))

    def _state_from_frame(self, frame: dict[str, Any]) -> list[Any]:
        state = frame.get("state")
        if state is None:
            state = frame.get("next_state")
        if state is None:
            raise ValueError("Frame is missing state data")
        return state

    def _payload_cfg(self) -> dict[str, Any]:
        cfg = self._payload.get("metadata", {}).get("config", {})
        return cfg if isinstance(cfg, dict) else {}

    def _env_shape(self) -> tuple[str, dict[str, Any]]:
        metadata = self._payload.get("metadata", {})
        return str(metadata.get("env_class", "ComplexBuildingEnv")), self._payload_cfg()

    def _draw_replay_hud(self, frame: dict[str, Any]) -> None:
        assert self._screen is not None
        pygame.draw.rect(self._screen, COMPLEX_PANEL, (0, 0, self.geometry.width, self.geometry.top_panel_height))
        episode = self._payload.get("episode", 1)
        step_number = int(frame.get("step", self._index + 1))
        cumulative_reward = 0.0
        for item in self._trajectory[: self._index + 1]:
            cumulative_reward += float(item.get("reward", 0.0))
        self._draw_text("Episode Replay", 24, 16)
        self._draw_text(
            f"Episode {episode} | step {step_number}/{len(self._trajectory) - 1}",
            24,
            52,
            COMPLEX_MUTED,
            small=True,
        )
        self._draw_text(
            f"Reward {frame.get('reward', 0.0):.2f}",
            320,
            52,
            SUCCESS if frame.get("reward", 0.0) >= 0 else COMPLEX_BROKEN,
            small=True,
        )
        self._draw_text(
            f"Cumulative {cumulative_reward:.2f}",
            470,
            52,
            SUCCESS if cumulative_reward >= 0 else COMPLEX_BROKEN,
            small=True,
        )
        self._draw_text(
            f"Done {bool(frame.get('done', False))}",
            650,
            52,
            COMPLEX_MUTED,
            small=True,
        )
        info = frame.get("info", {})
        if isinstance(info, dict):
            self._draw_text(
                f"Success {bool(info.get('success', False))}",
                760,
                52,
                SUCCESS if info.get("success") else COMPLEX_MUTED,
                small=True,
            )
        self._draw_text("Space/Enter step | A autoplay | Left/Right seek | R restart | Esc quit", 24, 82, COMPLEX_MUTED, small=True)

    def _extract_entities(self, state: list[Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], int]:
        cfg = self._payload_cfg()
        n_robots = int(cfg.get("n_robots", 2))
        n_elevators = int(cfg.get("n_elevators", 2))
        n_tasks = int(cfg.get("n_tasks", 3))
        flat_state: list[Any] = []
        for item in state:
            if isinstance(item, (list, tuple)):
                flat_state.extend(item)
            else:
                flat_state.append(item)
        robots = []
        idx = 0
        for _ in range(n_robots):
            floor, inside, carrying = flat_state[idx: idx + 3]
            robots.append({"floor": int(floor), "inside_elevator": int(inside), "carrying_task": int(carrying)})
            idx += 3
        elevators = []
        for _ in range(n_elevators):
            floor, direction, broken_remaining = flat_state[idx: idx + 3]
            elevators.append({"floor": int(floor), "direction": int(direction), "broken_remaining": int(broken_remaining)})
            idx += 3
        tasks = []
        for _ in range(n_tasks):
            pickup_floor, delivery_floor, status = flat_state[idx: idx + 3]
            tasks.append({"pickup_floor": int(pickup_floor), "delivery_floor": int(delivery_floor), "status": int(status)})
            idx += 3
        steps = int(flat_state[idx]) if idx < len(flat_state) else 0
        return robots, elevators, tasks, steps

    def _layout(self) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
        left = self.geometry.margin
        top = self.geometry.top_panel_height + self.geometry.margin // 2
        main_width = self.geometry.width - self.geometry.sidebar_width - 3 * self.geometry.margin
        main_height = self.geometry.height - top - self.geometry.margin
        sidebar_left = left + main_width + self.geometry.margin
        return (
            (left, top, main_width, main_height),
            (sidebar_left, top, self.geometry.sidebar_width, main_height),
        )

    def _floor_bounds(self, floor: int, n_floors: int) -> tuple[int, int]:
        (left, top, width, height), _ = self._layout()
        floor_height = height / n_floors
        y1 = int(top + (n_floors - floor - 1) * floor_height)
        y2 = int(y1 + floor_height)
        return y1, y2

    def _shaft_geometry(self, elevator_idx: int, n_elevators: int) -> tuple[int, int]:
        (left, _, width, _), _ = self._layout()
        shaft_count = max(1, n_elevators)
        available = max(220, width - 150)
        shaft_spacing = available / shaft_count
        shaft_center = left + 120 + int(shaft_spacing * (elevator_idx + 0.5))
        shaft_width = min(84, max(56, int(shaft_spacing * 0.42)))
        return shaft_center, shaft_width

    def _draw_frame(self, frame: dict[str, Any]) -> None:
        assert self._screen is not None
        self._screen.fill(COMPLEX_BACKGROUND)
        env_class, cfg = self._env_shape()
        robots, elevators, tasks, steps = self._extract_entities(self._state_from_frame(frame))
        n_floors = int(cfg.get("n_floors", 8))
        n_elevators = int(cfg.get("n_elevators", max(1, len(elevators))))

        # HUD
        self._draw_replay_hud(frame)

        # Layout
        (left, top, width, height), (sidebar_left, sidebar_top, sidebar_width, _) = self._layout()

        # Main building
        pygame.draw.rect(self._screen, COMPLEX_SHADOW, (left + 12, top + 10, width, height), border_radius=14)
        pygame.draw.rect(self._screen, COMPLEX_BUILDING, (left, top, width, height), border_radius=14)
        pygame.draw.rect(self._screen, FLOOR_LINE, (left, top, width, height), width=2, border_radius=14)
        for floor in range(n_floors):
            y1, y2 = self._floor_bounds(floor, n_floors)
            band_color = COMPLEX_FLOOR_A if floor % 2 == 0 else COMPLEX_FLOOR_B
            floor_rect = pygame.Rect(left + 2, y1, width - 4, y2 - y1)
            pygame.draw.rect(self._screen, band_color, floor_rect)
            pygame.draw.line(self._screen, FLOOR_LINE, (left, y1), (left + width, y1), 2)
            self._draw_text(f"F{floor}", left + 14, y1 + 8, COMPLEX_MUTED, small=True)
        pygame.draw.line(self._screen, FLOOR_LINE, (left, top + height), (left + width, top + height), 2)

        # Tasks in building
        for slot, task in enumerate(tasks):
            if task["status"] == TASK_INACTIVE:
                continue
            y1, y2 = self._floor_bounds(task["pickup_floor"], n_floors)
            marker_x = left + 58
            marker_y = (y1 + y2) // 2
            if task["status"] == TASK_PENDING:
                color = COMPLEX_PENDING
                label = f"T{slot}->{task['delivery_floor']}"
            else:
                color = COMPLEX_PICKED
                label = f"T{slot} carried"
            pygame.draw.rect(self._screen, color, (marker_x, marker_y - 12, 34, 24), border_radius=4)
            pygame.draw.rect(self._screen, (15, 20, 28), (marker_x, marker_y - 12, 34, 24), width=2, border_radius=4)
            self._draw_text(f"T{slot}", marker_x + 6, marker_y - 9, (15, 20, 28), small=True)
            self._draw_text(label, marker_x + 48, marker_y - 10, COMPLEX_TEXT, tiny=True)

        # Elevators
        shaft_top = top + 12
        shaft_height = height - 24
        for elevator_idx, elevator in enumerate(elevators):
            shaft_center, shaft_width = self._shaft_geometry(elevator_idx, n_elevators)
            shaft_x = shaft_center - shaft_width // 2
            shaft_rect = pygame.Rect(shaft_x, shaft_top, shaft_width, shaft_height)
            pygame.draw.rect(self._screen, (38, 47, 66), shaft_rect, border_radius=8)
            pygame.draw.rect(self._screen, FLOOR_LINE, shaft_rect, width=2, border_radius=8)
            y1, y2 = self._floor_bounds(elevator["floor"], n_floors)
            car_rect = pygame.Rect(shaft_x + 7, y1 + 6, shaft_width - 14, (y2 - y1) - 12)
            car_color = COMPLEX_BROKEN if elevator["broken_remaining"] > 0 else ELEVATOR
            pygame.draw.rect(self._screen, car_color, car_rect, border_radius=8)
            pygame.draw.rect(self._screen, (240, 245, 251), car_rect, width=2, border_radius=8)
            if elevator["broken_remaining"] > 0:
                overlay = pygame.Surface((car_rect.width, car_rect.height), pygame.SRCALPHA)
                overlay.fill((255, 80, 80, 60))
                self._screen.blit(overlay, car_rect.topleft)
            direction = {DIR_IDLE: ".", DIR_UP: "^", DIR_DOWN: "v"}.get(elevator["direction"], ".")
            self._draw_text(f"E{elevator_idx}{direction}", car_rect.x + 6, car_rect.y + 8, (12, 16, 23), small=True)
            if elevator["broken_remaining"] > 0:
                self._draw_text(f"B{elevator['broken_remaining']}", car_rect.x + 6, car_rect.y + 28, (12, 16, 23), small=True)

        # Robots
        floor_groups: dict[int, list[tuple[int, dict[str, int]]]] = {}
        elevator_groups: dict[int, list[tuple[int, dict[str, int]]]] = {}
        for robot_idx, robot in enumerate(robots):
            if robot["inside_elevator"] == NOT_INSIDE:
                floor_groups.setdefault(robot["floor"], []).append((robot_idx, robot))
            else:
                elevator_groups.setdefault(robot["inside_elevator"], []).append((robot_idx, robot))
        for floor, entries in floor_groups.items():
            y1, y2 = self._floor_bounds(floor, n_floors)
            center_y = (y1 + y2) // 2
            base_x = left + 92
            for offset, (robot_idx, robot) in enumerate(entries):
                color = ROBOT_PALETTE[robot_idx % len(ROBOT_PALETTE)]
                center_x = base_x + offset * 44
                pygame.draw.circle(self._screen, color, (center_x, center_y), 18)
                pygame.draw.circle(self._screen, (14, 18, 25), (center_x, center_y), 18, 2)
                self._draw_text(f"R{robot_idx}", center_x - 12, center_y - 10, (14, 18, 25), small=True)
                if robot["carrying_task"] != NO_TASK:
                    self._draw_text(f"T{robot['carrying_task']}", center_x + 20, center_y - 10, COMPLEX_PICKED, tiny=True)
        for elevator_idx, entries in elevator_groups.items():
            shaft_center, shaft_width = self._shaft_geometry(elevator_idx, n_elevators)
            elevator = elevators[elevator_idx]
            y1, y2 = self._floor_bounds(elevator["floor"], n_floors)
            car_rect = pygame.Rect(shaft_center - shaft_width // 2 + 7, y1 + 6, shaft_width - 14, (y2 - y1) - 12)
            for slot, (robot_idx, robot) in enumerate(entries):
                color = ROBOT_PALETTE[robot_idx % len(ROBOT_PALETTE)]
                center_x = car_rect.centerx - 18 + (slot % 2) * 32
                center_y = car_rect.centery - 10 + (slot // 2) * 22
                pygame.draw.circle(self._screen, color, (center_x, center_y), 14)
                pygame.draw.circle(self._screen, (14, 18, 25), (center_x, center_y), 14, 2)
                self._draw_text(f"R{robot_idx}", center_x - 11, center_y - 9, (14, 18, 25), tiny=True)
                if robot["carrying_task"] != NO_TASK:
                    self._draw_text(f"T{robot['carrying_task']}", center_x + 16, center_y - 9, COMPLEX_PICKED, tiny=True)

        # Sidebar
        pygame.draw.rect(self._screen, COMPLEX_SIDEBAR, (sidebar_left, sidebar_top, sidebar_width, self.geometry.height - sidebar_top - self.geometry.margin), border_radius=14)
        pygame.draw.rect(self._screen, FLOOR_LINE, (sidebar_left, sidebar_top, sidebar_width, self.geometry.height - sidebar_top - self.geometry.margin), width=2, border_radius=14)
        self._draw_text("Timeline", sidebar_left + 16, sidebar_top + 12)
        self._draw_text(f"Env: {env_class}", sidebar_left + 16, sidebar_top + 40, COMPLEX_MUTED, tiny=True)
        self._draw_text(f"Steps: {int(frame.get('step', steps))}", sidebar_left + 16, sidebar_top + 60, COMPLEX_MUTED, tiny=True)
        row_y = sidebar_top + 96
        for item in self._trajectory[max(0, self._index - 7): self._index + 1]:
            step = item.get("step", 0)
            action = item.get("action", None)
            reward = item.get("reward", 0.0)
            self._draw_text(f"{step:03d}: action={action} reward={reward:.1f}", sidebar_left + 16, row_y, COMPLEX_TEXT, tiny=True)
            row_y += 22

        self._draw_text("Legend", sidebar_left + 16, self.geometry.height - 118, COMPLEX_TEXT)
        self._draw_text("Robot", sidebar_left + 16, self.geometry.height - 90, COMPLEX_MUTED, tiny=True)
        self._draw_text("Pending task", sidebar_left + 120, self.geometry.height - 90, COMPLEX_PENDING, tiny=True)
        self._draw_text("Broken elevator", sidebar_left + 16, self.geometry.height - 66, COMPLEX_BROKEN, tiny=True)
        self._draw_text("Picked task", sidebar_left + 120, self.geometry.height - 66, COMPLEX_PICKED, tiny=True)

    def _advance(self, delta: int = 1) -> None:
        self._index = max(0, min(len(self._trajectory) - 1, self._index + delta))

    def run(self, autoplay: bool = False, fps: int = 4) -> None:
        self._ensure_window()
        assert self._clock is not None

        self._index = 0
        running = True
        paused = not autoplay
        self._draw_frame(self._trajectory[self._index])
        pygame.display.flip()

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        self._index = 0
                    elif event.key == pygame.K_a:
                        paused = not paused
                    elif event.key in (pygame.K_SPACE, pygame.K_RETURN):
                        self._advance(1)
                    elif event.key == pygame.K_LEFT:
                        self._advance(-1)
                    elif event.key == pygame.K_RIGHT:
                        self._advance(1)

            if not paused:
                self._advance(1)
                self._clock.tick(fps)
            else:
                self._clock.tick(30)

            self._draw_frame(self._trajectory[self._index])
            pygame.display.flip()

        self.close()

    def close(self) -> None:
        if pygame is not None and self._screen is not None:
            pygame.quit()
        self._screen = None
        self._clock = None
        self._font = None
        self._small_font = None
        self._tiny_font = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Pygame renderer for ComplexBuildingEnv.")
    parser.add_argument("--max-steps", type=int, default=50, help="Episode length cap.")
    parser.add_argument("--seed", type=int, default=0, help="Environment seed.")
    parser.add_argument("--n-floors", type=int, default=8, help="Complex env: number of floors.")
    parser.add_argument("--n-robots", type=int, default=2, help="Complex env: number of robots.")
    parser.add_argument("--n-elevators", type=int, default=2, help="Complex env: number of elevators.")
    parser.add_argument("--n-tasks", type=int, default=3, help="Complex env: active task slots.")
    parser.add_argument("--total-task-budget", type=int, default=6, help="Complex env: total tasks per episode.")
    parser.add_argument("--obs-mode", choices=["full", "per_robot"], default="full", help="Complex env observation mode.")
    parser.add_argument("--p-elevator-delay", type=float, default=0.10, help="Complex env elevator delay probability.")
    parser.add_argument("--p-breakdown", type=float, default=0.01, help="Complex env elevator breakdown probability.")
    parser.add_argument("--breakdown-duration", type=int, default=5, help="Complex env breakdown duration.")
    parser.add_argument("--p-new-task", type=float, default=0.20, help="Complex env new task spawn probability.")
    parser.add_argument(
        "--autoplay",
        action="store_true",
        help="Advance the environment automatically by waiting each frame.",
    )
    parser.add_argument("--fps", type=int, default=4, help="Autoplay frames per second.")
    args = parser.parse_args()

    if pygame is None:
        raise SystemExit("pygame is not installed. Run `pip install -r requirements.txt` first.")

    cfg = ComplexEnvConfig(
        n_floors=args.n_floors,
        n_robots=args.n_robots,
        n_elevators=args.n_elevators,
        n_tasks=args.n_tasks,
        total_task_budget=args.total_task_budget,
        max_steps=args.max_steps,
        p_elevator_delay=args.p_elevator_delay,
        p_breakdown=args.p_breakdown,
        breakdown_duration=args.breakdown_duration,
        p_new_task=args.p_new_task,
        obs_mode=args.obs_mode,
    )
    env = ComplexBuildingEnv(config=cfg, seed=args.seed)
    renderer = ComplexBuildingPygameRenderer(env)
    renderer.run(autoplay=args.autoplay, fps=args.fps)


if __name__ == "__main__":
    main()