# PNF 404 Radio Animations

Experimental visual animations for **PNF 404 Radio: The Walkman Has Landed**. Animations are implemented with [PyQtGraph](https://www.pyqtgraph.org/) and Matplotlib and can be chained to visuals for audio mixes.

## Installation

Python 3.10+ is required. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Launch the interactive ring circles animation:

```bash
python run.py
```

The window shows rotating circles with layered noise. Keyboard controls:

- **C** – circular motion
- **F** – figure-eight paths
- **R** – faster drifting within the boundary
- **P** – rose-curve paths with a configurable petal count
- **Up / Down arrows** – increase or decrease animation speed (1×, 2×, 3×)
- **Space** – toggle crazy mode with a gray backdrop and rapidly changing colors

### Rendering

To render a 10-bit ProRes MOV instead of displaying a window:

```bash
python run.py --render --duration 0:10 --output demo.mov --timeline "0:00 2R, 1:47 3, 1:59 F"
```

The optional timeline uses `MM:SS` followed by commands. Palette numbers `1-4` switch colour schemes, letters `R`, `C`, `F` and `P` change movement modes, and `>`/`<` simulate up/down arrow presses to alter speed.

## Project Structure

- `run.py` – command-line entry point for running or rendering animations.
- `pnf404_radio/` – package containing animation logic.
  - `movement.py` – movement modes and controller.
  - `ring_circles.py` – example ring-circles animation.
  - `utils.py` – helper functions for noise and sprite creation.

## Development

Formatting uses [black](https://black.readthedocs.io/en/stable/) and linting uses [flake8](https://flake8.pycqa.org/). Run:

```bash
black .
flake8
```

Verify changes manually with:

```bash
python run.py
```

New animations should live in `pnf404_radio` and be callable from `run.py`.
