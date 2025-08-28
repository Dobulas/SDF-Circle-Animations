"""Movement logic for PNF 404 Radio animations."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict

import numpy as np


class MovementMode(Enum):
    """Available sprite movement modes."""

    CIRCLE = auto()
    FIGURE_EIGHT = auto()
    DRIFT = auto()


@dataclass
class MovementController:
    """Update sprite positions based on the selected movement mode.

    Parameters
    ----------
    sprite_count:
        Number of sprites to animate.
    ring_radius:
        Radius of the circular path.
    boundary_x:
        Half-width of the bounding rectangle for drift behaviour.
    boundary_y:
        Half-height of the bounding rectangle for drift behaviour.
    start_delay:
        Seconds to hold the initial arrangement before motion begins.
    angular_speed:
        Base angular speed in radians per second for circular motion.
    """

    sprite_count: int
    ring_radius: float
    boundary_x: float
    boundary_y: float
    start_delay: float = 0.0
    angular_speed: float = 0.6
    angles: np.ndarray = field(init=False)
    positions: np.ndarray = field(init=False)
    velocities: np.ndarray = field(init=False)
    t: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        """Initialise per-sprite state."""

        self.angles = np.linspace(0, 2 * np.pi, self.sprite_count, endpoint=False)
        self.positions = np.stack(
            [
                self.ring_radius * np.cos(self.angles),
                self.ring_radius * np.sin(self.angles),
            ],
            axis=1,
        )
        self.velocities = np.zeros_like(self.positions)

    def base_scale_for(self, mode: MovementMode) -> np.ndarray:
        """Return mode-specific starting scales."""

        return np.ones(self.sprite_count)

    def get_start_state(self, mode: MovementMode) -> tuple[np.ndarray, np.ndarray]:
        """Return starting positions and velocities for ``mode``.

        The controller itself is not modified. Returned arrays may be used to
        smoothly transition into ``mode``.
        """

        positions = np.zeros_like(self.positions)
        velocities = np.zeros_like(self.positions)

        if mode is MovementMode.CIRCLE:
            r = self.ring_radius
            positions[:, 0] = r * np.cos(self.angles)
            positions[:, 1] = r * np.sin(self.angles)
        elif mode is MovementMode.FIGURE_EIGHT:
            r = self.ring_radius
            positions[:, 0] = r * np.sin(self.angles)
            positions[:, 1] = r * np.sin(2 * self.angles)
        elif mode is MovementMode.DRIFT:
            positions[:] = self.positions
            speed = .1  # doubled from 0.05 to increase drift speed
            velocities = np.random.uniform(-speed, speed, size=(self.sprite_count, 2))
        else:
            positions[:] = self.positions

        return positions, velocities

    def update(self, mode: MovementMode, dt: float) -> Dict[str, np.ndarray]:
        """Advance the animation by ``dt`` seconds."""

        self.t += dt

        if mode is MovementMode.CIRCLE:
            return self._update_circle(dt)
        if mode is MovementMode.FIGURE_EIGHT:
            return self._update_figure_eight(dt)
        if mode is MovementMode.DRIFT:
            return self._update_drift(dt)
        return {}

    def _update_circle(self, dt: float) -> Dict[str, np.ndarray]:
        """Advance animation using circular motion."""

        if self.t >= self.start_delay:
            self.angles += self.angular_speed * dt
        r = self.ring_radius
        self.positions[:, 0] = r * np.cos(self.angles)
        self.positions[:, 1] = r * np.sin(self.angles)
        return {}

    def _update_figure_eight(self, dt: float) -> Dict[str, np.ndarray]:
        """Advance animation using figure-eight motion."""

        if self.t >= self.start_delay:
            self.angles += self.angular_speed * dt
        r = self.ring_radius
        self.positions[:, 0] = r * np.sin(self.angles)
        self.positions[:, 1] = r * np.sin(2 * self.angles)
        return {}

    def _update_drift(self, dt: float) -> Dict[str, np.ndarray]:
        """Advance animation using constant drift motion."""

        frame_scale = dt * 60.0
        self.positions += self.velocities * frame_scale
        self._apply_boundary()
        return {}

    def _apply_boundary(self) -> None:
        """Keep sprites inside the boundary box."""

        for i in range(self.sprite_count):
            x, y = self.positions[i]
            if abs(x) > self.boundary_x:
                self.velocities[i, 0] *= -1
                x = np.sign(x) * self.boundary_x
            if abs(y) > self.boundary_y:
                self.velocities[i, 1] *= -1
                y = np.sign(y) * self.boundary_y
            self.positions[i] = x, y
