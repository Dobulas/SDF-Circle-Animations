"""Entry point for PNF 404 Radio animations."""

from __future__ import annotations

import argparse

from pnf404_radio import ring_circles


def main() -> None:
    """Run the ring circles animation or render it to an MP4 file."""

    parser = argparse.ArgumentParser(description="PNF 404 Radio animations")
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render a headless MP4 instead of displaying the window.",
    )
    args = parser.parse_args()
    if args.render:
        ring_circles.render_headless_mp4()
    else:
        ring_circles.run()


if __name__ == "__main__":
    main()
