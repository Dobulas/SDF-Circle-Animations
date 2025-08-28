"""Simple ring circle animation for PNF 404 Radio visuals."""

from __future__ import annotations

import random

import numpy as np
import pyqtgraph as pg
from matplotlib.colors import to_rgba
from pyqtgraph.Qt import QtGui, QtWidgets

from .movement import MovementController, MovementMode
from .utils import create_sprite


def run() -> int:
    """Display the ring circle animation.

    Sprites hold a circular formation briefly before beginning their
    rotation, emphasising the starting shape.
    """

    width, height = 1920, 1080
    center_x, center_y = width // 2, height // 2
    radii = [100, 80, 60]
    noise_scale = 0.02
    noise_intensity = 5
    sprite_count = 13

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = pg.GraphicsLayoutWidget(show=True, title="PNF 404 Radio - Ring Circles")
    win.resize(width, height)
    win.setBackground("#D6DDBE")

    plot = win.addPlot()
    plot.setAspectLocked(True)
    plot.hideAxis("bottom")
    plot.hideAxis("left")

    palette = [
        "#adb397",
        "#bcc2a8",
        "#909771",
        "#6a7432",
        "#7b835b",
        "#747c4d",
    ]

    sprites = []
    for _ in range(sprite_count):
        layered_sdf = create_sprite(
            center_x,
            center_y,
            radii,
            width=width,
            height=height,
            noise_scale=noise_scale,
            noise_intensity=noise_intensity,
        )
        normalized_sdf = (layered_sdf - layered_sdf.min()) / (
            layered_sdf.max() - layered_sdf.min()
        )
        r, g, b, _ = to_rgba(random.choice(palette))
        colors_layered = np.zeros((*normalized_sdf.shape, 4))
        colors_layered[..., 0] = r
        colors_layered[..., 1] = g
        colors_layered[..., 2] = b
        alpha_layered = np.clip(1 - normalized_sdf**2, 0, 1)
        colors_layered[..., 3] = alpha_layered
        sprite_rgba_8u = (colors_layered * 255).astype(np.uint8)
        sprites.append(sprite_rgba_8u)

    sprite_items = []
    for sprite in sprites:
        item = pg.ImageItem(image=sprite)
        item.setOpts(axisOrder="row-major")
        plot.addItem(item)
        sprite_items.append(item)

    ring_radius = 450
    boundary_x = width / 2 - 20
    boundary_y = height / 2 - 20
    controller = MovementController(
        sprite_count, ring_radius, boundary_x, boundary_y, start_delay=1.0
    )
    movement_mode = MovementMode.CIRCLE

    for i, item in enumerate(sprite_items):
        x, y = controller.positions[i]
        item.setPos(x - width / 2, y - height / 2)

    transition_active = False
    transition_frame = 0
    transition_frames = 0
    start_positions = np.zeros((sprite_count, 2))
    start_scales = np.ones(sprite_count)
    target_scales = np.ones(sprite_count)
    pending_mode = movement_mode
    fixed_dt = 1.0 / 60.0  # assume a steady 60 FPS frame time

    def transition_to_mode(new_mode: MovementMode, duration: float) -> None:
        """Interpolate sprites toward ``new_mode`` over ``duration`` frames."""

        nonlocal transition_active, transition_frame, transition_frames
        nonlocal pending_mode

        for i, item in enumerate(sprite_items):
            pos = item.pos()
            start_positions[i] = (
                pos.x() + width / 2,
                pos.y() + height / 2,
            )
            start_scales[i] = item.scale()

        positions, velocities = controller.get_start_state(new_mode)
        controller.positions = positions
        controller.velocities = velocities
        target_scales[:] = controller.base_scale_for(new_mode)
        pending_mode = new_mode
        transition_frames = max(1, int(duration))
        transition_frame = 0
        transition_active = True

    transition_duration = 90

    circle_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("C"), win)
    circle_shortcut.activated.connect(
        lambda: transition_to_mode(MovementMode.CIRCLE, transition_duration)
    )

    figure_eight_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("F"), win)
    figure_eight_shortcut.activated.connect(
        lambda: transition_to_mode(MovementMode.FIGURE_EIGHT, transition_duration)
    )

    drift_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("R"), win)
    drift_shortcut.activated.connect(
        lambda: transition_to_mode(MovementMode.DRIFT, transition_duration)
    )

    def update() -> None:
        """Advance the animation by one frame."""

        nonlocal transition_active, transition_frame, movement_mode

        dt = fixed_dt
        default_scale = 1.0

        if transition_active:
            extras = controller.update(pending_mode, dt)
            transition_frame += 1
            alpha = transition_frame / transition_frames
            alpha = 0.5 - 0.5 * np.cos(np.pi * alpha)
            target_pos = controller.positions
            target_scale = extras.get("scale", target_scales)
            interp_pos = (1 - alpha) * start_positions + alpha * target_pos
            interp_scale = (1 - alpha) * start_scales + alpha * target_scale
            for i, item in enumerate(sprite_items):
                x, y = interp_pos[i]
                item.setPos(x - width / 2, y - height / 2)
                item.setScale(interp_scale[i])
            if transition_frame >= transition_frames:
                transition_active = False
                movement_mode = pending_mode
            return

        extras = controller.update(movement_mode, dt)
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
