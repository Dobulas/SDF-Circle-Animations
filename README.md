# PNF 404 Radio Animations

This project contains experimental visual animations for **PNF 404 Radio: The Walkman Has Landed**. Animations are implemented with [PyQtGraph](https://www.pyqtgraph.org/) and can be chained together to create visuals for audio mixes.

## Requirements

Install dependencies with pip:

```bash
pip install -r requirements.txt
```

## Usage

Run the ring circle animation:

```bash
python run.py
```

A window will open displaying rotating circles with layered noise. This serves as a starting point for creating additional animations.
You can switch between movement modes while the window is active:

* **C** - circular motion
* **F** - figure-eight paths
* **R** - faster drifting within the boundary
* **P** - rose-curve paths with configurable petal count
* **Up / Down arrows** - increase or decrease animation speed (1×, 2×, 3×)
* **Space** - toggle crazy mode with a gray backdrop and rapidly changing colors

To render a headless MP4, use the ``--render`` flag. You can optionally supply a
comma-separated timeline of timestamped commands that change palette, movement,
or speed while rendering. The duration accepts either seconds or ``MM:SS``
notation:

```bash
python run.py --render --duration 0:10 --output demo.mp4 --timeline "0:00 2R, 1:47 3, 1:59 F"
```

Each timeline entry uses ``MM:SS`` followed by commands. Palette numbers ``1-4``
switch colour schemes, letters ``R``, ``C``, ``F`` and ``P`` change movement
modes, and ``>``/``<`` simulate up/down arrow presses to alter speed.

