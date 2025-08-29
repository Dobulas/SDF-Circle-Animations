"""Simple ring circle animation for PNF 404 Radio visuals."""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

import numpy as np
from matplotlib.colors import to_rgba

from .movement import MovementController, MovementMode
from .utils import colorize, colorize_with_color, create_sprite

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


def parse_timeline(spec: str, fps: int) -> List[Tuple[int, str]]:
    """Parse a timeline specification into frame-scheduled commands.

    Parameters
    ----------
    spec:
        Comma-separated string where each entry is ``MM:SS COMMANDS``.
    fps:
        Frames per second of the rendered video. Timestamps are converted to
        frame indices using this value.

    Returns
    -------
    list of tuple
        A sorted list of ``(frame, commands)`` pairs.
    """

    events: List[Tuple[int, str]] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        time_part, *cmd_part = part.split()
        if not cmd_part:
            raise ValueError(f"No commands provided for '{time_part}'")
        minutes_str, seconds_str = time_part.split(":")
        total_seconds = int(minutes_str) * 60 + int(seconds_str)
        frame = int(total_seconds * fps)
        events.append((frame, cmd_part[0]))
    events.sort(key=lambda e: e[0])
    return events


def run() -> int:
    """Display the ring circle animation.

    Sprites hold a circular formation briefly before beginning their
    rotation, emphasising the starting shape. Random drift respects the
    window's aspect ratio.
    """

    import pyqtgraph as pg
    from pyqtgraph.Qt import QtGui, QtWidgets

    window_width, window_height = 1920, 1080
    radii = [70, 50, 30]
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

    crazy_colors = [color for cfg in PALETTES.values() for color in cfg["palette"]]
    crazy_cache: List[List[np.ndarray]] = []
    for field in fields:
        crazy_cache.append(
            [colorize_with_color(field, color) for color in crazy_colors]
        )

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
    crazy_mode = False
    crazy_progress = np.zeros(sprite_count)
    crazy_delay = np.zeros(sprite_count)
    crazy_active = np.zeros(sprite_count, dtype=bool)
    crazy_next_images: List[np.ndarray] = [
        crazy_cache[i][0] for i in range(sprite_count)
    ]

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

        if crazy_mode or index not in PALETTES:
            return
        start_palette_transition(index)

    for i in PALETTES:
        shortcut = QtWidgets.QShortcut(QtGui.QKeySequence(str(i)), win)
        shortcut.activated.connect(lambda i=i: apply_palette(i))

    def toggle_crazy_mode() -> None:
        """Toggle crazy mode where sprites randomly change color."""

        nonlocal crazy_mode, pending_palettes
        finish_color_transition()
        pending_palettes = []
        crazy_mode = not crazy_mode
        crazy_progress[:] = 0
        crazy_active[:] = False
        if crazy_mode:
            crazy_delay[:] = np.random.uniform(0, color_transition_frames, sprite_count)
            win.setBackground(PALETTES[4]["background"])
            for i in range(sprite_count):
                sprite_items[i].setImage(
                    crazy_cache[i][random.randrange(len(crazy_colors))]
                )
                transition_items[i].setOpacity(0)
        else:
            win.setBackground(PALETTES[current_palette]["background"])
            for i in range(sprite_count):
                sprite_items[i].setImage(color_cache[current_palette][i])
                transition_items[i].setOpacity(0)

    crazy_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("Space"), win)
    crazy_shortcut.activated.connect(toggle_crazy_mode)

    ring_radius = 450
    aspect_ratio = window_width / window_height
    # Determine border size while preserving the window's aspect ratio
    if window_width >= window_height:
        border_half_height = window_height / 2 - margin
        border_half_width = border_half_height * aspect_ratio
    else:
        border_half_width = window_width / 2 - margin
        border_half_height = border_half_width / aspect_ratio

    # Movement boundary is constrained inside the border to keep sprites visible
    boundary_x = border_half_width - max_radius
    boundary_y = border_half_height - max_radius

    border = QtWidgets.QGraphicsRectItem(
        -border_half_width,
        -border_half_height,
        2 * border_half_width,
        2 * border_half_height,
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

    current_speed = 1.0
    target_speed = 1.0
    speed_change_rate = 1.0  # speed units per second
    speed_levels = [0.5, 1.0, 2.0, 3.0]
    speed_index = 0

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

    def apply_speed(index: int) -> None:
        """Update the target speed to the multiplier at ``index``."""

        nonlocal speed_index, target_speed
        speed_index = index
        target_speed = speed_levels[speed_index]

    def change_speed(delta: int) -> None:
        """Move to another speed level by ``delta`` steps."""

        new_index = max(0, min(len(speed_levels) - 1, speed_index + delta))
        apply_speed(new_index)

    up_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("Up"), win)
    up_shortcut.activated.connect(lambda: change_speed(1))
    down_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("Down"), win)
    down_shortcut.activated.connect(lambda: change_speed(-1))

    def update() -> None:
        """Advance the animation by one frame."""

        nonlocal transition_active, transition_frame, movement_mode
        nonlocal color_transition_frame, current_speed

        if current_speed < target_speed:
            current_speed = min(
                target_speed, current_speed + speed_change_rate * fixed_dt
            )
        elif current_speed > target_speed:
            current_speed = max(
                target_speed, current_speed - speed_change_rate * fixed_dt
            )

        dt = fixed_dt * current_speed
        default_scale = 1.0

        if transition_active:
            extras = controller.update(pending_mode, dt)
            transition_frame += current_speed
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
            scales = extras["scale"] if has_scale else None
            update_transitions = color_transition_active or crazy_mode
            for i, item_a in enumerate(sprite_items):
                x, y = controller.positions[i]
                pos_x = x - centers[i, 0]
                pos_y = y - centers[i, 1]
                scale = scales[i] if has_scale else default_scale
                item_a.setPos(pos_x, pos_y)
                item_a.setScale(scale)
                if update_transitions:
                    item_b = transition_items[i]
                    item_b.setPos(pos_x, pos_y)
                    item_b.setScale(scale)

        if color_transition_active and not crazy_mode:
            color_transition_frame += current_speed
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

        if crazy_mode:
            for i in range(sprite_count):
                if crazy_active[i]:
                    crazy_progress[i] += current_speed
                    alpha = crazy_progress[i] / color_transition_frames
                    alpha = 0.5 - 0.5 * np.cos(np.pi * alpha)
                    sprite_items[i].setOpacity(1 - alpha)
                    transition_items[i].setOpacity(alpha)
                    if crazy_progress[i] >= color_transition_frames:
                        sprite_items[i].setImage(crazy_next_images[i])
                        sprite_items[i].setOpacity(1)
                        transition_items[i].setOpacity(0)
                        crazy_active[i] = False
                        crazy_progress[i] = 0
                        crazy_delay[i] = random.uniform(0, color_transition_frames)
                else:
                    crazy_progress[i] += current_speed
                    if crazy_progress[i] >= crazy_delay[i]:
                        crazy_active[i] = True
                        crazy_progress[i] = 0
                        crazy_next_images[i] = crazy_cache[i][
                            random.randrange(len(crazy_colors))
                        ]
                        transition_items[i].setImage(crazy_next_images[i])

    timer = pg.QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(16)  # ~60 FPS

    return app.exec_()


def render_headless_mp4(
    duration: float = 5.0,
    output: str = "ring_circles.mp4",
    fps: int = 60,
    timeline: str | None = None,
) -> None:
    """Render the animation to an MP4 using ``ffmpeg``.

    Parameters
    ----------
    duration:
        Length of the video in seconds.
    output:
        Path to the output MP4 file.
    fps:
        Frames per second for the generated video.
    timeline:
        Optional comma-separated timeline of timestamped commands. Each entry
        should be formatted as ``MM:SS COMMANDS`` where ``COMMANDS`` may include
        palette numbers (1-4), movement letters (R, C, F, P) and ``>``/``<`` for
        speed changes.
    """

    import subprocess

    total_frames = int(duration * fps)

    window_width, window_height = 1920, 1080
    radii = [70, 50, 30]
    noise_scale = 0.02
    noise_intensity = 5
    sprite_count = 15
    margin = 20
    max_radius = max(radii)
    sprite_size = int(2 * max_radius + margin)

    ring_radius = 450
    aspect_ratio = window_width / window_height
    if window_width >= window_height:
        border_half_height = window_height / 2 - margin
        border_half_width = border_half_height * aspect_ratio
    else:
        border_half_width = window_width / 2 - margin
        border_half_height = border_half_width / aspect_ratio

    boundary_x = border_half_width - max_radius
    boundary_y = border_half_height - max_radius

    controller = MovementController(
        sprite_count=sprite_count,
        ring_radius=ring_radius,
        boundary_x=boundary_x,
        boundary_y=boundary_y,
        start_delay=1.0,
        rose_k=5,
    )

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

    color_cache: Dict[int, List[np.ndarray]] = {}
    bg_cache: Dict[int, np.ndarray] = {}
    for idx, cfg in PALETTES.items():
        color_cache[idx] = [colorize(field, cfg["palette"]) for field in fields]
        bg_cache[idx] = (np.array(to_rgba(cfg["background"])) * 255).astype(np.uint8)

    current_palette = 1
    sprites = color_cache[current_palette]
    frame_bg = np.tile(bg_cache[current_palette], (window_height, window_width, 1))

    # Palette transition state
    palette_start = sprites
    palette_target = sprites
    bg_start = bg_cache[current_palette]
    bg_target = bg_start
    palette_transition_active = False
    palette_transition_frame = 0
    palette_transition_frames = int(1.0 * fps)

    def start_palette_transition(target: int) -> None:
        """Begin a cross-fade to ``target`` palette."""

        nonlocal current_palette, palette_start, palette_target
        nonlocal bg_start, bg_target, palette_transition_active
        nonlocal palette_transition_frame
        if target == current_palette:
            return
        palette_start = sprites
        palette_target = color_cache[target]
        bg_start = bg_cache[current_palette]
        bg_target = bg_cache[target]
        current_palette = target
        palette_transition_frame = 0
        palette_transition_active = True

    movement_mode = MovementMode.CIRCLE
    transition_active = False
    transition_frame = 0
    transition_frames = int(1.5 * fps)
    start_positions = controller.positions.copy()
    pending_mode = movement_mode

    def start_movement_transition(new_mode: MovementMode) -> None:
        """Smoothly interpolate sprites into ``new_mode``."""

        nonlocal transition_active, transition_frame, pending_mode
        nonlocal start_positions
        start_positions = controller.positions.copy()
        positions, velocities = controller.get_start_state(new_mode)
        controller.positions = positions
        controller.velocities = velocities
        pending_mode = new_mode
        transition_frame = 0
        transition_active = True

    speed_levels = [0.5, 1.0, 2.0, 3.0]
    speed_index = 1
    current_speed = speed_levels[speed_index]
    target_speed = current_speed
    speed_change_rate = 1.0

    events = parse_timeline(timeline, fps) if timeline else []
    event_idx = 0

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{window_width}x{window_height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        output,
    ]
    with subprocess.Popen(cmd, stdin=subprocess.PIPE) as proc:
        fixed_dt = 1.0 / fps
        for frame_num in range(total_frames):
            while event_idx < len(events) and frame_num >= events[event_idx][0]:
                commands = events[event_idx][1].replace(" ", "")
                for ch in commands:
                    if ch == ">":
                        speed_index = min(len(speed_levels) - 1, speed_index + 1)
                        target_speed = speed_levels[speed_index]
                    elif ch == "<":
                        speed_index = max(0, speed_index - 1)
                        target_speed = speed_levels[speed_index]
                    elif ch in "1234":
                        start_palette_transition(int(ch))
                    elif ch in "RCFP":
                        start_movement_transition(
                            {
                                "R": MovementMode.DRIFT,
                                "C": MovementMode.CIRCLE,
                                "F": MovementMode.FIGURE_EIGHT,
                                "P": MovementMode.ROSE,
                            }[ch]
                        )
                event_idx += 1

            if current_speed < target_speed:
                current_speed = min(
                    target_speed, current_speed + speed_change_rate * fixed_dt
                )
            elif current_speed > target_speed:
                current_speed = max(
                    target_speed, current_speed - speed_change_rate * fixed_dt
                )

            dt = fixed_dt * current_speed
            if transition_active:
                controller.update(pending_mode, dt)
                transition_frame += current_speed
                alpha_t = transition_frame / transition_frames
                alpha_t = 0.5 - 0.5 * np.cos(np.pi * alpha_t)
                render_positions = (1 - alpha_t) * start_positions + alpha_t * (
                    controller.positions
                )
                if transition_frame >= transition_frames:
                    transition_active = False
                    movement_mode = pending_mode
            else:
                controller.update(movement_mode, dt)
                render_positions = controller.positions

            if palette_transition_active:
                palette_transition_frame += current_speed
                alpha_p = palette_transition_frame / palette_transition_frames
                alpha_p = 0.5 - 0.5 * np.cos(np.pi * alpha_p)
                bg = (1 - alpha_p) * bg_start + alpha_p * bg_target
                frame = np.tile(bg.astype(np.uint8), (window_height, window_width, 1))
            else:
                frame = frame_bg.copy()

            for i in range(sprite_count):
                x, y = render_positions[i]
                x = int(window_width / 2 + x - centers[i, 0])
                y = int(window_height / 2 - y - centers[i, 1])
                h, w, _ = sprites[i].shape
                x1, x2 = max(0, x), min(window_width, x + w)
                y1, y2 = max(0, y), min(window_height, y + h)
                if x1 >= x2 or y1 >= y2:
                    continue
                sub_frame = frame[y1:y2, x1:x2]
                if palette_transition_active:
                    for img, fade in (
                        (palette_start[i], 1 - alpha_p),
                        (palette_target[i], alpha_p),
                    ):
                        sub_img = img[y1 - y : y2 - y, x1 - x : x2 - x]
                        alpha = (sub_img[..., 3:4] / 255.0) * fade
                        sub_frame[..., :3] = (1 - alpha) * sub_frame[
                            ..., :3
                        ] + alpha * sub_img[..., :3]
                else:
                    sub_img = sprites[i][y1 - y : y2 - y, x1 - x : x2 - x]
                    alpha = sub_img[..., 3:4] / 255.0
                    sub_frame[..., :3] = (1 - alpha) * sub_frame[
                        ..., :3
                    ] + alpha * sub_img[..., :3]

            if (
                palette_transition_active
                and palette_transition_frame >= palette_transition_frames
            ):
                palette_transition_active = False
                sprites = palette_target
                frame_bg = np.tile(bg_target, (window_height, window_width, 1))

            proc.stdin.write(frame[..., :3].tobytes())
        proc.stdin.close()
        proc.wait()


if __name__ == "__main__":
    run()
