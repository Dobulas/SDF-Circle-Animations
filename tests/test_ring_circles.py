import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np  # noqa: E402
from pnf404_radio import ring_circles  # noqa: E402


def test_get_random_center_bounds() -> None:
    np.random.seed(0)
    cx, cy = ring_circles._get_random_center(100, 80, 10)
    assert 10 <= cx < 90
    assert 10 <= cy < 70


def test_generate_simple_noise_shape() -> None:
    noise = ring_circles._generate_simple_noise(20, 10)
    assert noise.shape == (20, 10)


def test_create_sprite_shape() -> None:
    np.random.seed(0)
    sprite = ring_circles._create_sprite(
        50,
        50,
        [10, 20],
        width=100,
        height=100,
        noise_scale=0.1,
        noise_intensity=1.0,
    )
    assert sprite.shape == (100, 100)
