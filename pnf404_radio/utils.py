"""Utility functions for PNF 404 Radio animations."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def get_random_center(width: int, height: int, margin: int) -> tuple[int, int]:
    """Return a random ``(x, y)`` point within the given margins."""
    center_x = np.random.randint(margin, width - margin)
    center_y = np.random.randint(margin, height - margin)
    return int(center_x), int(center_y)


def generate_simple_noise(
    width: int,
    height: int,
    scale: float = 0.05,
    intensity: float = 10,
) -> np.ndarray:
    """Generate sine-cosine-based noise."""
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    noise = np.sin(x * scale) + np.cos(y * scale)
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
) -> tuple[np.ndarray, tuple[int, int]]:
    """Create a layered signed distance field array with noise.

    Parameters
    ----------
    center_x, center_y:
        Position of the sprite's centre within the returned array.
    radii:
        Radii for the concentric circles making up the sprite.
    width, height:
        Dimensions of the generated sprite. These may match the sprite's own
        size instead of the full screen.
    noise_scale, noise_intensity:
        Parameters controlling the simple noise overlay.

    Returns
    -------
    tuple[np.ndarray, tuple[int, int]]
        The signed distance field and the centre point used to generate it.
    """

    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    combined_sdf = np.zeros((height, width))
    for radius in radii:
        sdf = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) - radius
        noise = generate_simple_noise(
            width, height, scale=noise_scale, intensity=noise_intensity
        )
        sdf_with_noise = sdf + noise
        combined_sdf = np.minimum(combined_sdf, sdf_with_noise)
    return combined_sdf, (int(center_x), int(center_y))
