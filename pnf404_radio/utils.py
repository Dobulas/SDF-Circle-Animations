"""Utility functions for PNF 404 Radio animations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Tuple

import numpy as np
import taichi as ti

try:
    ti.init(arch=ti.metal, default_fp=ti.f32)
except Exception:
    ti.init(arch=ti.cpu, default_fp=ti.f32)


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
) -> np.ndarray:
    """Create a layered signed distance field array with noise on the GPU."""

    radii_arr = np.array(radii, dtype=np.float32)
    result = np.zeros((height, width), dtype=np.float32)

    _sdf_kernel(
        center_x,
        center_y,
        radii_arr,
        noise_scale,
        noise_intensity,
        result,
    )
    return result


@ti.kernel
def _sdf_kernel(
    center_x: ti.i32,
    center_y: ti.i32,
    radii: ti.types.ndarray(dtype=ti.f32, ndim=1),
    scale: ti.f32,
    intensity: ti.f32,
    out: ti.types.ndarray(dtype=ti.f32, ndim=2),
) -> None:
    """Kernel to compute the signed distance field with simple noise."""

    for i, j in out:
        x = j - center_x
        y = i - center_y
        noise = ti.sin(j * scale) + ti.cos(i * scale)
        min_sdf = 1e8
        for k in range(radii.shape[0]):
            sdf = ti.sqrt(x * x + y * y) - radii[k] + noise * intensity
            min_sdf = ti.min(min_sdf, sdf)
        out[i, j] = min_sdf
