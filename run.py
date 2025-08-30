"""Entry point for PNF 404 Radio animations."""

from __future__ import annotations

import argparse

from pnf404_radio import ring_circles


def parse_duration(value: str) -> float:
    """Return seconds represented by ``value``.

    ``value`` may be a plain number of seconds or an ``MM:SS`` string.
    """

    try:
        if ":" in value:
            minutes_str, seconds_str = value.split(":")
            return int(minutes_str) * 60 + float(seconds_str)
        return float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid duration '{value}'") from exc


def main() -> None:
    """Run the ring circles animation or render it to a 10-bit ProRes MOV."""

    parser = argparse.ArgumentParser(description="PNF 404 Radio animations")
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render a headless 10-bit ProRes MOV instead of displaying the window.",
    )
    parser.add_argument(
        "--duration",
        type=parse_duration,
        default=5.0,
        help=(
            "Duration of the output video in seconds or MM:SS "
            "(only used with --render)."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ring_circles.mov",
        help=("Filename for the rendered MOV (only used with --render)."),
    )
    parser.add_argument(
        "--timeline",
        type=str,
        default=None,
        help=(
            "Comma-separated list of timestamped commands for rendering, "
            "e.g. '0:00 2R, 1:47 3'."
        ),
    )
    args = parser.parse_args()
    if args.render:
        ring_circles.render_headless_prores(
            duration=args.duration,
            output=args.output,
            timeline=args.timeline,
        )
    else:
        ring_circles.run()


if __name__ == "__main__":
    main()
