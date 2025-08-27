"""Simple ring circle animation for PNF 404 Radio visuals."""

from __future__ import annotations

from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtWidgets

from .movement import MovementController, MovementMode
from .utils import create_sprite, get_random_center


def run() -> int:
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
        plt.cm.Greys,
        plt.cm.Purples,
        plt.cm.Blues,
        plt.cm.Greens,
        plt.cm.Oranges,
        plt.cm.Reds,
        plt.cm.YlOrBr,
        plt.cm.YlOrRd,
        plt.cm.OrRd,
        plt.cm.PuRd,
        plt.cm.RdPu,
        plt.cm.BuPu,
        plt.cm.GnBu,
        plt.cm.PuBu,
        plt.cm.YlGnBu,
        plt.cm.PuBuGn,
        plt.cm.BuGn,
        plt.cm.YlGn,
    ]
    colormap_cycle = cycle(sequential_colormaps)

    sprites = []
    for _ in range(sprite_count):
        cx, cy = get_random_center(width, height, margin)
        layered_sdf = create_sprite(
            cx,
            cy,
            radii,
            width=width,
            height=height,
            noise_scale=noise_scale,
            noise_intensity=noise_intensity,
        )
        normalized_sdf = (layered_sdf - layered_sdf.min()) / (
            layered_sdf.max() - layered_sdf.min()
        )
        cmap = next(colormap_cycle)
        colors_layered = cmap(normalized_sdf)
        alpha_layered = np.clip(1 - normalized_sdf**2, 0, 1)
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
    boundary = width / 2 - 50
    controller = MovementController(sprite_count, ring_radius, boundary)
    movement_mode = MovementMode.CIRCLE

    for i, item in enumerate(sprite_items):
        x, y = controller.positions[i]
        item.setPos(x - width / 2, y - height / 2)

    transition_active = False
    transition_frame = 0
    transition_frames = 0
    start_positions = np.zeros((sprite_count, 2))
    start_scales = np.ones(sprite_count)
    target_positions = np.zeros((sprite_count, 2))
    target_scales = np.ones(sprite_count)
    transition_state: dict[str, np.ndarray | float] = {}
    pending_mode = movement_mode

    def transition_to_mode(new_mode: MovementMode, duration: float) -> None:
        """Interpolate sprites toward ``new_mode`` over ``duration`` frames."""

        nonlocal transition_active, transition_frame, transition_frames
        nonlocal pending_mode, transition_state

        for i, item in enumerate(sprite_items):
            pos = item.pos()
            start_positions[i] = (
                pos.x() + width / 2,
                pos.y() + height / 2,
            )
            start_scales[i] = item.scale()

        state = controller.get_transition_state(new_mode)
        target_positions[:] = state["positions"]
        target_scales[:] = state.get("scale", controller.base_scale_for(new_mode))
        transition_state = state
        pending_mode = new_mode
        transition_frames = max(1, int(duration))
        transition_frame = 0
        transition_active = True

    transition_duration = 60

    circle_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("C"), win)
    circle_shortcut.activated.connect(
        lambda: transition_to_mode(MovementMode.CIRCLE, transition_duration)
    )

    random_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("R"), win)
    random_shortcut.activated.connect(
        lambda: transition_to_mode(MovementMode.RANDOM, transition_duration)
    )

    elliptical_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("E"), win)
    elliptical_shortcut.activated.connect(
        lambda: transition_to_mode(MovementMode.ELLIPTICAL, transition_duration)
    )

    spiral_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("S"), win)
    spiral_shortcut.activated.connect(
        lambda: transition_to_mode(MovementMode.SPIRAL, transition_duration)
    )

    lissajous_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("L"), win)
    lissajous_shortcut.activated.connect(
        lambda: transition_to_mode(MovementMode.LISSAJOUS, transition_duration)
    )

    wobble_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("W"), win)
    wobble_shortcut.activated.connect(
        lambda: transition_to_mode(MovementMode.WOBBLE, transition_duration)
    )

    avoidance_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("A"), win)
    avoidance_shortcut.activated.connect(
        lambda: transition_to_mode(MovementMode.AVOIDANCE, transition_duration)
    )

    wave_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("V"), win)
    wave_shortcut.activated.connect(
        lambda: transition_to_mode(MovementMode.WAVE, transition_duration)
    )

    helix_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("H"), win)
    helix_shortcut.activated.connect(
        lambda: transition_to_mode(MovementMode.HELIX, transition_duration)
    )

    figure_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("F"), win)
    figure_shortcut.activated.connect(
        lambda: transition_to_mode(MovementMode.FIGURE_EIGHT, transition_duration)
    )

    cascade_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("O"), win)
    cascade_shortcut.activated.connect(
        lambda: transition_to_mode(MovementMode.CASCADE, transition_duration)
    )

    drift_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("D"), win)
    drift_shortcut.activated.connect(
        lambda: transition_to_mode(MovementMode.DRIFT, transition_duration)
    )

    def update() -> None:
        """Advance the animation by one frame."""

        nonlocal transition_active, transition_frame, movement_mode

        default_scale = 1.0

        if transition_active:
            transition_frame += 1
            alpha = transition_frame / transition_frames
            interp_pos = (1 - alpha) * start_positions + alpha * target_positions
            interp_scale = (1 - alpha) * start_scales + alpha * target_scales
            for i, item in enumerate(sprite_items):
                x, y = interp_pos[i]
                item.setPos(x - width / 2, y - height / 2)
                item.setScale(interp_scale[i])
            if transition_frame >= transition_frames:
                transition_active = False
                controller.positions = transition_state["positions"]
                controller.velocities = transition_state["velocities"]
                controller.angles = transition_state["angles"]
                controller.t = float(transition_state["t"])
                movement_mode = pending_mode
                if "scale" not in transition_state:
                    for item in sprite_items:
                        item.setScale(default_scale)
            return

        extras = controller.update(movement_mode)
        has_scale = "scale" in extras
        for i, item in enumerate(sprite_items):
            x, y = controller.positions[i]
            item.setPos(x - width / 2, y - height / 2)
            if has_scale:
                item.setScale(extras["scale"][i])
            else:
                item.setScale(default_scale)

    timer = pg.QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(16)  # ~60 FPS

    return app.exec_()


if __name__ == "__main__":
    run()
