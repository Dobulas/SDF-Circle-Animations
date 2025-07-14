"""Simple ring circle animation for PNF 404 Radio visuals."""

from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets


def _get_random_center(width: int, height: int, margin: int) -> tuple[int, int]:
    """Return a random (x, y) point within the given margins."""
    center_x = np.random.randint(margin, width - margin)
    center_y = np.random.randint(margin, height - margin)
    return center_x, center_y


def _generate_simple_noise(width: int, height: int, scale: float = 0.05, intensity: float = 10) -> np.ndarray:
    """Generate sine-cosine-based noise."""
    y, x = np.meshgrid(np.arange(height), np.arange(width))
    noise = np.sin(x * scale) + np.cos(y * scale)
    return noise * intensity


def _create_sprite(center_x: int, center_y: int, radii: list[int], *, width: int, height: int,
                   noise_scale: float, noise_intensity: float) -> np.ndarray:
    """Create a layered signed distance field array with noise."""
    y, x = np.meshgrid(np.arange(height), np.arange(width))
    combined_sdf = np.zeros((height, width))
    for radius in radii:
        sdf = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) - radius
        noise = _generate_simple_noise(width, height, scale=noise_scale, intensity=noise_intensity)
        sdf_with_noise = sdf + noise
        combined_sdf = np.minimum(combined_sdf, sdf_with_noise)
    return combined_sdf


def run():
    """Display the ring circle animation."""
    width, height = 1000, 1000
    margin = 150
    radii = [100, 80, 60]
    noise_scale = 0.02
    noise_intensity = 5
    sprite_count = 15

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = pg.GraphicsLayoutWidget(show=True, title="PNF 404 Radio - Ring Circles")
    win.resize(800, 800)
    win.setBackground("w")

    plot = win.addPlot()
    plot.setAspectLocked(True)
    plot.hideAxis("bottom")
    plot.hideAxis("left")

    sequential_colormaps = [
        plt.cm.Greys, plt.cm.Purples, plt.cm.Blues, plt.cm.Greens, plt.cm.Oranges, plt.cm.Reds,
        plt.cm.YlOrBr, plt.cm.YlOrRd, plt.cm.OrRd, plt.cm.PuRd, plt.cm.RdPu, plt.cm.BuPu,
        plt.cm.GnBu, plt.cm.PuBu, plt.cm.YlGnBu, plt.cm.PuBuGn, plt.cm.BuGn, plt.cm.YlGn,
    ]
    colormap_cycle = cycle(sequential_colormaps)

    sprites = []
    for _ in range(sprite_count):
        cx, cy = _get_random_center(width, height, margin)
        layered_sdf = _create_sprite(
            cx,
            cy,
            radii,
            width=width,
            height=height,
            noise_scale=noise_scale,
            noise_intensity=noise_intensity,
        )
        normalized_sdf = (layered_sdf - layered_sdf.min()) / (layered_sdf.max() - layered_sdf.min())
        cmap = next(colormap_cycle)
        colors_layered = cmap(normalized_sdf)
        alpha_layered = np.clip(1 - normalized_sdf ** 2, 0, 1)
        colors_layered[..., -1] = alpha_layered
        sprite_rgba_8u = (colors_layered * 255).astype(np.uint8)
        sprites.append(sprite_rgba_8u)

    sprite_items = []
    for sprite in sprites:
        item = pg.ImageItem(image=sprite)
        item.setOpts(axisOrder="row-major")
        plot.addItem(item)
        sprite_items.append(item)

    ring_radius = 300
    angles = np.linspace(0, 2 * np.pi, sprite_count, endpoint=False)
    for i, item in enumerate(sprite_items):
        theta = angles[i]
        x = ring_radius * np.cos(theta)
        y = ring_radius * np.sin(theta)
        item.setPos(x - width / 2, y - height / 2)

    def update():
        nonlocal angles
        rotation_speed = 0.01
        angles += rotation_speed
        for i, item in enumerate(sprite_items):
            theta = angles[i]
            x = ring_radius * np.cos(theta)
            y = ring_radius * np.sin(theta)
            item.setPos(x - width / 2, y - height / 2)

    timer = pg.QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(16)  # ~60 FPS

    return QtWidgets.QApplication.instance().exec_()


if __name__ == "__main__":
    run()
