"""Command-line interface for the labeling pipeline."""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Label dynamic C3D using subject's labeled static C3D.",
    )
    parser.add_argument("static", help="Path to labeled static C3D file")
    parser.add_argument("dynamic", help="Path to unlabeled dynamic C3D file")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output path prefix (default: dynamic path with _labeled suffix)",
    )
    parser.add_argument(
        "--no-filled",
        action="store_true",
        help="Do not export filled C3D/CSV",
    )
    parser.add_argument(
        "--obstacle-visibility",
        type=float,
        default=0.80,
        metavar="FRAC",
        help="Min visibility for obstacle candidates (default: 0.80)",
    )
    parser.add_argument(
        "--max-interp-frames",
        type=int,
        default=10,
        metavar="N",
        help="Max gap length for spline interpolation (default: 10)",
    )
    args = parser.parse_args()
    out_prefix = args.output
    if out_prefix is None:
        out_prefix = args.dynamic.rsplit(".", 1)[0] if "." in args.dynamic else args.dynamic
        out_prefix = f"{out_prefix}_labeled"
    try:
        from .pipeline import run_pipeline
        info = run_pipeline(
            args.static,
            args.dynamic,
            out_prefix,
            obstacle_visibility_min=args.obstacle_visibility,
            export_filled=not args.no_filled,
            max_interp_frames=args.max_interp_frames,
        )
        print(f"Labeled {info['n_markers']} markers, {info['n_frames']} frames.")
        print(f"Output: {out_prefix}_labeled.c3d, {out_prefix}_labeled.csv")
        if not args.no_filled:
            print(f"Filled: {out_prefix}_labeled_filled.c3d, {out_prefix}_labeled_filled.csv")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
