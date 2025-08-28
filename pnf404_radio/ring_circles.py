"""Simple ring circle animation for PNF 404 Radio visuals."""

from __future__ import annotations

import random
from typing import Dict, List

import numpy as np
import pyqtgraph as pg
from matplotlib.colors import to_rgba
from pyqtgraph.Qt import QtGui, QtWidgets

from .movement import MovementController, MovementMode
from .utils import create_sprite

PALETTES: Dict[int, Dict[str, List[str]]] = {
    1: {
        "background": "#D6DDBE",
        "palette": [
            "#adb397",
            "#bcc2a8",
            "#909771",
            "#6a7432",
            "#7b835b",
            "#747c4d",
        ],
    },
    2: {
        "background": "#fff0da",
        "palette": [
            "#e1a232",
            "#e39f30",
            "#ea8d21",
            "#f08117",
            "#f27b12",
            "#fc6501",
        ],
    },
    3: {
        "background": "#fdfdfc",
        "palette": [
            "#90b2d5",
            "#629ccc",
            "#5a8ab7",
            "#2a619e",
            "#50606f",
            "#2d4966",
        ],
    },
    4: {
        "background": "#e7e7e7",
        "palette": [
            "#606060",
            "#989898",
            "#d7d7d7",
            "#c7c7c7",
            "#262626",
            "#b7b7b7",
        ],
    },
}


def run() -> int:
    """Display the ring circle animation.

    Sprites hold a circular formation briefly before beginning their
    rotation, emphasising the starting shape. Random drift respects the
    window's aspect ratio.
    """

    window_width, window_height = 1920, 1080
    radii = [90, 70, 50]
    noise_scale = 0.02
    noise_intensity = 5
    sprite_count = 15
    margin = 20
    max_radius = max(radii)
    sprite_size = int(2 * max_radius + margin)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = pg.GraphicsLayoutWidget(show=True, title="PNF 404 Radio - Ring Circles")
    win.resize(window_width, window_height)

    palette_cfg = PALETTES[1]
    win.setBackground(palette_cfg["background"])

    plot = win.addPlot()
    plot.setAspectLocked(True)
    plot.hideAxis("bottom")
    plot.hideAxis("left")

    def colorize(field: np.ndarray, palette: List[str]) -> np.ndarray:
        """Return an RGBA sprite for ``field`` using a random ``palette`` color."""

        r, g, b, _ = to_rgba(random.choice(palette))
        colors_layered = np.zeros((*field.shape, 4))
        colors_layered[..., 0] = r
        colors_layered[..., 1] = g
        colors_layered[..., 2] = b
        alpha_layered = np.clip(1 - field**2, 0, 1)
        colors_layered[..., 3] = alpha_layered
        return (colors_layered * 255).astype(np.uint8)

    fields: List[np.ndarray] = []
    centers = np.zeros((sprite_count, 2))
    for i in range(sprite_count):
        layered_sdf, center = create_sprite(
            sprite_size // 2,
            sprite_size // 2,
            radii,
            width=sprite_size,
            height=sprite_size,
            noise_scale=noise_scale,
            noise_intensity=noise_intensity,
        )
        normalized_sdf = (layered_sdf - layered_sdf.min()) / (
            layered_sdf.max() - layered_sdf.min()
        )
        fields.append(normalized_sdf)
        centers[i] = center

    color_cache: Dict[int, List[np.ndarray]] = {idx: [] for idx in PALETTES}
    for idx, cfg in PALETTES.items():
        for field in fields:
            color_cache[idx].append(colorize(field, cfg["palette"]))

    sprite_items = []
    transition_items = []
    for i in range(sprite_count):
        base = pg.ImageItem(image=color_cache[1][i])
        base.setOpts(axisOrder="row-major")
        base.setZValue(i * 2)
        plot.addItem(base)
        sprite_items.append(base)

        overlay = pg.ImageItem(image=color_cache[1][i])
        overlay.setOpts(axisOrder="row-major")
        overlay.setOpacity(0)
        overlay.setZValue(i * 2 + 1)
        plot.addItem(overlay)
        transition_items.append(overlay)

    def hex_to_rgb_array(color: str) -> np.ndarray:
        """Return the RGB components of ``color`` as a float array."""

        return np.array(pg.mkColor(color).getRgb()[:3], dtype=float)

    current_palette = 1
    color_transition_active = False
    color_transition_frame = 0
    color_transition_frames = 60
    color_target_palette = 1
    pending_palettes: List[int] = []
    bg_start = hex_to_rgb_array(PALETTES[current_palette]["background"])
    bg_target = bg_start.copy()

    def finish_color_transition() -> None:
        """Finalize any active color transition."""

        nonlocal color_transition_active, current_palette, bg_start
        if not color_transition_active:
            return
        for i in range(sprite_count):
            sprite_items[i], transition_items[i] = transition_items[i], sprite_items[i]
            transition_items[i].setOpacity(0)
            sprite_items[i].setOpacity(1)
        current_palette = color_target_palette
        bg_start = hex_to_rgb_array(PALETTES[current_palette]["background"])
        win.setBackground(PALETTES[current_palette]["background"])
        color_transition_active = False

    def trigger_next_palette_transition() -> None:
        """Start the next transition in the queue if available."""

        nonlocal color_transition_active, color_transition_frame
        nonlocal color_target_palette, bg_target
        if not pending_palettes:
            return
        next_index = pending_palettes.pop(0)
        color_target_palette = next_index
        for i in range(sprite_count):
            transition_items[i].setImage(color_cache[next_index][i])
            transition_items[i].setOpacity(0)
        bg_target = hex_to_rgb_array(PALETTES[next_index]["background"])
        color_transition_frame = 0
        color_transition_active = True

    def start_palette_transition(target_index: int) -> None:
        """Queue a palette change using palette ``4`` as a transition."""

        nonlocal pending_palettes
        if target_index == current_palette:
            return
        finish_color_transition()
        if target_index != 4 and current_palette != 4:
            pending_palettes = [4, target_index]
        else:
            pending_palettes = [target_index]
        trigger_next_palette_transition()

    def apply_palette(index: int) -> None:
        """Start a palette change sequence for ``index``."""

        if index not in PALETTES:
            return
        start_palette_transition(index)

    for i in PALETTES:
        shortcut = QtWidgets.QShortcut(QtGui.QKeySequence(str(i)), win)
        shortcut.activated.connect(lambda i=i: apply_palette(i))

    ring_radius = 450
    aspect_ratio = window_width / window_height
    # Keep drift boundary proportional to the window aspect ratio
    if window_width >= window_height:
        boundary_y = window_height / 2 - margin - max_radius
        boundary_x = boundary_y * aspect_ratio
    else:
        boundary_x = window_width / 2 - margin - max_radius
        boundary_y = boundary_x / aspect_ratio

    border = QtWidgets.QGraphicsRectItem(
        -boundary_x - max_radius,
        -boundary_y - max_radius,
        2 * (boundary_x + max_radius),
        2 * (boundary_y + max_radius),
    )
    border.setPen(pg.mkPen(color="black", width=1))
    border.setZValue(sprite_count * 2 + 2)
    plot.addItem(border)

    controller = MovementController(
        sprite_count,
        ring_radius,
        boundary_x,
        boundary_y,
        start_delay=1.0,
        rose_k=5,
    )
    movement_mode = MovementMode.CIRCLE

    for i, item in enumerate(sprite_items):
        x, y = controller.positions[i]
        item.setPos(x - centers[i, 0], y - centers[i, 1])

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
                pos.x() + centers[i, 0],
                pos.y() + centers[i, 1],
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

    rose_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("P"), win)
    rose_shortcut.activated.connect(
        lambda: transition_to_mode(MovementMode.ROSE, transition_duration)
    )

    def update() -> None:
        """Advance the animation by one frame."""

        nonlocal transition_active, transition_frame, movement_mode
        nonlocal color_transition_frame

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
            for i, (item_a, item_b) in enumerate(zip(sprite_items, transition_items)):
                x, y = interp_pos[i]
                item_a.setPos(x - centers[i, 0], y - centers[i, 1])
                item_b.setPos(x - centers[i, 0], y - centers[i, 1])
                item_a.setScale(interp_scale[i])
                item_b.setScale(interp_scale[i])
            if transition_frame >= transition_frames:
                transition_active = False
                movement_mode = pending_mode
        else:
            extras = controller.update(movement_mode, dt)
            has_scale = "scale" in extras
            for i, (item_a, item_b) in enumerate(zip(sprite_items, transition_items)):
                x, y = controller.positions[i]
                item_a.setPos(x - centers[i, 0], y - centers[i, 1])
                item_b.setPos(x - centers[i, 0], y - centers[i, 1])
                scale = extras["scale"][i] if has_scale else default_scale
                item_a.setScale(scale)
                item_b.setScale(scale)

        if color_transition_active:
            color_transition_frame += 1
            alpha = color_transition_frame / color_transition_frames
            alpha = 0.5 - 0.5 * np.cos(np.pi * alpha)
            for i in range(sprite_count):
                sprite_items[i].setOpacity(1 - alpha)
                transition_items[i].setOpacity(alpha)
            bg_interp = (1 - alpha) * bg_start + alpha * bg_target
            win.setBackground(pg.mkColor(tuple(bg_interp.astype(int))))
            if color_transition_frame >= color_transition_frames:
                finish_color_transition()
                trigger_next_palette_transition()

    timer = pg.QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(16)  # ~60 FPS

    return app.exec_()


if __name__ == "__main__":
    run()
