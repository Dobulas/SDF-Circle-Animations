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
* **R** - random wandering
* **E** - rotating elliptical orbits
* **S** - spiraling paths
* **L** - Lissajous curves
* **W** - radial wobble
* **A** - avoidance with mild center attraction
* **V** - phase-offset wave pattern
* **H** - helix motion with depth scaling
* **F** - figure eight path
* **O** - cascading rings
* **D** - drifting motion within the bounds
