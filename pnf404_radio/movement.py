"""Movement logic for PNF 404 Radio animations."""

from __future__ import annotations

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict

import numpy as np


class MovementMode(Enum):
    """Available sprite movement modes."""

    CIRCLE = auto()
    RANDOM = auto()
    ELLIPTICAL = auto()
    SPIRAL = auto()
    LISSAJOUS = auto()
    WOBBLE = auto()
    AVOIDANCE = auto()
    WAVE = auto()
    HELIX = auto()
    FIGURE_EIGHT = auto()
    CASCADE = auto()
    DRIFT = auto()


@dataclass
class MovementController:
    """Update sprite positions based on the selected movement mode."""

    sprite_count: int
    ring_radius: float
    boundary: float
    angles: np.ndarray = field(init=False)
    positions: np.ndarray = field(init=False)
    velocities: np.ndarray = field(init=False)
    phases: np.ndarray = field(init=False)
    ellipse_axes: np.ndarray = field(init=False)
    lissajous_freqs: np.ndarray = field(init=False)
    cascade_radii: np.ndarray = field(init=False)
    t: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self.angles = np.linspace(0, 2 * np.pi, self.sprite_count, endpoint=False)
        self.positions = np.stack(
            [
                self.ring_radius * np.cos(self.angles),
                self.ring_radius * np.sin(self.angles),
            ],
            axis=1,
        )
        self.velocities = np.zeros_like(self.positions)
        self.phases = np.random.uniform(0, 2 * np.pi, self.sprite_count)
        self.ellipse_axes = np.stack(
            [
                np.full(self.sprite_count, self.ring_radius * 1.2),
                np.full(self.sprite_count, self.ring_radius * 0.8),
            ],
            axis=1,
        )
        self.lissajous_freqs = np.stack(
            [np.full(self.sprite_count, 1.0), np.full(self.sprite_count, 2.0)], axis=1
        )
        self.cascade_radii = self.ring_radius * (
            1 + np.arange(self.sprite_count) / self.sprite_count
        )

    def base_scale_for(self, mode: MovementMode) -> np.ndarray:
        """Return mode-specific starting scales."""

        if mode is MovementMode.HELIX:
            depth = 0.5 + 0.5 * np.sin(self.angles + self.t)
            return 0.5 + depth
        return np.ones(self.sprite_count)

    def get_transition_state(self, mode: MovementMode) -> dict[str, np.ndarray | float]:
        """Return state after one step of ``mode`` without modifying the controller."""

        pos_backup = self.positions.copy()
        vel_backup = self.velocities.copy()
        ang_backup = self.angles.copy()
        t_backup = self.t

        if mode in {MovementMode.RANDOM, MovementMode.DRIFT}:
            self.velocities = np.random.uniform(-1.5, 1.5, size=(self.sprite_count, 2))
        else:
            self.velocities[:] = 0

        extras = self.update(mode)
        state: dict[str, np.ndarray | float] = {
            "positions": self.positions.copy(),
            "velocities": self.velocities.copy(),
            "angles": self.angles.copy(),
            "t": self.t,
        }
        if "scale" in extras:
            state["scale"] = extras["scale"].copy()

        self.positions = pos_backup
        self.velocities = vel_backup
        self.angles = ang_backup
        self.t = t_backup

        return state

    def update(self, mode: MovementMode) -> Dict[str, np.ndarray]:
        """Advance the animation by one frame."""

        dt = 0.01
        self.t += dt
        extras: Dict[str, np.ndarray] = {}

        if mode is MovementMode.CIRCLE:
            self.angles += 0.01
            r = self.ring_radius
            self.positions[:, 0] = r * np.cos(self.angles)
            self.positions[:, 1] = r * np.sin(self.angles)
        elif mode is MovementMode.RANDOM:
            self.velocities += np.random.uniform(-0.2, 0.2, size=(self.sprite_count, 2))
            self.velocities *= 0.98
            np.clip(self.velocities, -3, 3, out=self.velocities)
            self.positions += self.velocities
            self._apply_boundary()
        elif mode is MovementMode.ELLIPTICAL:
            self.angles += 0.01
            orientation = self.t * 0.05
            cos_o, sin_o = np.cos(orientation), np.sin(orientation)
            a = self.ellipse_axes[:, 0]
            b = self.ellipse_axes[:, 1]
            x = a * np.cos(self.angles)
            y = b * np.sin(self.angles)
            self.positions[:, 0] = cos_o * x - sin_o * y
            self.positions[:, 1] = sin_o * x + cos_o * y
        elif mode is MovementMode.SPIRAL:
            self.angles += 0.02
            amplitude = self.ring_radius * 0.3
            r = self.ring_radius + amplitude * np.sin(self.t + self.phases)
            self.positions[:, 0] = r * np.cos(self.angles)
            self.positions[:, 1] = r * np.sin(self.angles)
        elif mode is MovementMode.LISSAJOUS:
            freq = self.lissajous_freqs
            self.positions[:, 0] = self.ring_radius * np.sin(
                freq[:, 0] * self.t + self.phases
            )
            self.positions[:, 1] = self.ring_radius * np.sin(
                freq[:, 1] * self.t + 2 * self.phases
            )
        elif mode is MovementMode.WOBBLE:
            self.angles += 0.02
            amplitude = self.ring_radius * 0.1
            r = self.ring_radius + amplitude * np.sin(self.t * 2 + self.phases)
            self.positions[:, 0] = r * np.cos(self.angles)
            self.positions[:, 1] = r * np.sin(self.angles)
        elif mode is MovementMode.AVOIDANCE:
            self._avoidance_step()
        elif mode is MovementMode.WAVE:
            self.angles += 0.02
            phase = self.phases
            offset = 0.5 * np.sin(self.t + np.arange(self.sprite_count))
            self.positions[:, 0] = self.ring_radius * np.cos(
                self.angles + offset + phase
            )
            self.positions[:, 1] = self.ring_radius * np.sin(
                self.angles + offset + phase
            )
        elif mode is MovementMode.HELIX:
            self.angles += 0.02
            depth = 0.5 + 0.5 * np.sin(self.angles + self.t)
            extras["scale"] = 0.5 + depth
            self.positions[:, 0] = self.ring_radius * np.cos(self.angles)
            self.positions[:, 1] = self.ring_radius * np.sin(self.angles)
        elif mode is MovementMode.FIGURE_EIGHT:
            self.angles += 0.02
            a = self.ring_radius * 0.6
            b = self.ring_radius * 0.4
            self.positions[:, 0] = a * np.sin(self.angles)
            self.positions[:, 1] = b * np.sin(2 * self.angles)
        elif mode is MovementMode.CASCADE:
            self.angles += 0.02
            for i in range(self.sprite_count):
                r = self.cascade_radii[i]
                angle = self.angles[i] * (self.ring_radius / r)
                self.positions[i, 0] = r * np.cos(angle)
                self.positions[i, 1] = r * np.sin(angle)
        elif mode is MovementMode.DRIFT:
            self.velocities += np.random.uniform(
                -0.05, 0.05, size=(self.sprite_count, 2)
            )
            self.velocities *= 0.99
            np.clip(self.velocities, -1, 1, out=self.velocities)
            self.positions += self.velocities
            self._apply_boundary()

        return extras

    def _apply_boundary(self) -> None:
        """Keep sprites inside the boundary box."""
        for i in range(self.sprite_count):
            x, y = self.positions[i]
            if abs(x) > self.boundary:
                self.velocities[i, 0] *= -1
                x = np.sign(x) * self.boundary
            if abs(y) > self.boundary:
                self.velocities[i, 1] *= -1
                y = np.sign(y) * self.boundary
            self.positions[i] = x, y

    def _avoidance_step(self) -> None:
        """Simple avoidance with center attraction and mutual repulsion."""
        attraction = -self.positions * 0.005
        repulsion = np.zeros_like(self.positions)
        for i in range(self.sprite_count):
            diff = self.positions[i] - self.positions
            dist2 = np.sum(diff**2, axis=1) + 1e-4
            mask = np.arange(self.sprite_count) != i
            rep = np.sum(diff[mask] / dist2[mask, None], axis=0)
            repulsion[i] = rep
        self.velocities += attraction + repulsion * 0.01
        self.velocities *= 0.95
        np.clip(self.velocities, -2, 2, out=self.velocities)
        self.positions += self.velocities
        self._apply_boundary()
