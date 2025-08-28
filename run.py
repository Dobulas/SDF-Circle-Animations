"""Entry point for PNF 404 Radio animations."""

from __future__ import annotations

import argparse

from pnf404_radio import ring_circles


def main() -> None:
    """Parse CLI options and start the animation."""

    parser = argparse.ArgumentParser(description="PNF 404 Radio animation runner.")
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU acceleration when available.",
    )
    args = parser.parse_args()
    ring_circles.run(use_gpu=args.gpu)


if __name__ == "__main__":
    main()
