"""Utility functions for PNF 404 Radio animations."""

from __future__ import annotations

from collections.abc import Sequence
from types import ModuleType
from typing import Any, Tuple

import numpy as np


def get_random_center(width: int, height: int, margin: int) -> Tuple[int, int]:
    """Return a random ``(x, y)`` point within the given margins."""
    center_x = np.random.randint(margin, width - margin)
    center_y = np.random.randint(margin, height - margin)
    return int(center_x), int(center_y)


def generate_simple_noise(
    width: int,
    height: int,
    scale: float = 0.05,
    intensity: float = 10,
    xp: ModuleType = np,
    *,
    device: Any | None = None,
) -> np.ndarray:
    """Generate sine-cosine-based noise using ``xp`` (NumPy, CuPy, or Torch).

    Parameters
    ----------
    width, height:
        Dimensions of the generated field.
    scale:
        Frequency of the sine/cosine pattern.
    intensity:
        Amplitude multiplier for the noise.
    xp:
        Numerical backend. Supports NumPy, CuPy, or Torch.
    device:
        Optional device used when ``xp`` is Torch.
    """

    if xp.__name__ == "torch":
        y = xp.arange(height, device=device)
        x = xp.arange(width, device=device)
        y, x = xp.meshgrid(y, x, indexing="ij")
    else:
        y, x = xp.meshgrid(xp.arange(height), xp.arange(width), indexing="ij")
    noise = xp.sin(x * scale) + xp.cos(y * scale)
    return noise * intensity


def create_sprite(
    center_x: int,
    center_y: int,
    radii: Sequence[int],
    *,
    width: int,
    height: int,
    noise_scale: float,
    noise_intensity: float,
    xp: ModuleType = np,
    device: Any | None = None,
) -> np.ndarray:
    """Create a layered signed distance field array with noise.

    Parameters
    ----------
    center_x, center_y:
        Central point of the sprite.
    radii:
        Sequence of radii defining the ring layers.
    width, height:
        Dimensions of the output array.
    noise_scale, noise_intensity:
        Controls for procedural noise generation.
    xp:
        Numerical backend. Supports NumPy, CuPy, or Torch.
    device:
        Optional device used when ``xp`` is Torch.
    """

    if xp.__name__ == "torch":
        y = xp.arange(height, device=device)
        x = xp.arange(width, device=device)
        y, x = xp.meshgrid(y, x, indexing="ij")
        combined_sdf = xp.zeros((height, width), device=device)
    else:
        y, x = xp.meshgrid(xp.arange(height), xp.arange(width), indexing="ij")
        combined_sdf = xp.zeros((height, width))
    for radius in radii:
        sdf = xp.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) - radius
        noise = generate_simple_noise(
            width,
            height,
            scale=noise_scale,
            intensity=noise_intensity,
            xp=xp,
            device=device,
        )
        sdf_with_noise = sdf + noise
        combined_sdf = xp.minimum(combined_sdf, sdf_with_noise)
    return combined_sdf
