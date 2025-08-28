"""Utility functions for PNF 404 Radio animations."""

from collections.abc import Sequence
from typing import Tuple, TYPE_CHECKING, Any

import numpy as np
import taichi as ti

# Prefer running Taichi on a GPU when available, falling back to the CPU.
try:
    ti.init(arch=ti.gpu, default_fp=ti.f32)
except Exception:
    ti.init(arch=ti.cpu, default_fp=ti.f32)

# Type aliases to keep Pylance/pyright happy while giving Taichi the runtime types
if TYPE_CHECKING:
    F32 = float
    Nd1F = Any
    Nd2F = Any
else:
    F32 = ti.f32
    Nd1F = ti.types.ndarray(dtype=ti.f32, ndim=1)
    Nd2F = ti.types.ndarray(dtype=ti.f32, ndim=2)


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
    center_x: F32,
    center_y: F32,
    radii: Nd1F,
    scale: F32,
    intensity: F32,
    out: Nd2F,
):
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
