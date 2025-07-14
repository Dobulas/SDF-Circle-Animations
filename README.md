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

* Press **C** for the default circular motion.
* Press **R** for random wandering.

## Testing

Basic tests cover helper functions used by the animation code:

```bash
pytest
```

When running the GUI as part of tests or CI, set `QT_QPA_PLATFORM=offscreen`:

```bash
QT_QPA_PLATFORM=offscreen python run.py
```
