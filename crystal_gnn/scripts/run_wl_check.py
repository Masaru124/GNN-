"""Run WL collision scan script wrapper."""

from __future__ import annotations

import argparse

from crystal_gnn.expressivity.wl_check import main as wl_main


def main() -> None:
    parser = argparse.ArgumentParser(description="Run WL check")
    parser.add_argument("--chemsys", default="O-Ti")
    parser.add_argument("--max_materials", default="40")
    parser.add_argument("--radii", default="4.0,6.0,8.0")
    args = parser.parse_args()

    import sys

    sys.argv = [
        "wl_check",
        "--chemsys",
        str(args.chemsys),
        "--max_materials",
        str(args.max_materials),
        "--radii",
        str(args.radii),
    ]
    wl_main()


if __name__ == "__main__":
    main()
