#!/usr/bin/env python3
"""Generate portfolio demo GIF using the same renderer as marker-label-view."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parent
REPO = EXAMPLES.parent
sys.path.insert(0, str(EXAMPLES))
sys.path.insert(0, str(REPO / "src"))

from demo_synthetic_walking import (
    extract_demo_walking_clip,
    generate_synthetic_walking,
)
from marker_label.export import export_csv
from marker_label.qc_viewer import export_qc_viewer_gif

ASSETS = REPO / "docs" / "assets"
DEMO_CSV = EXAMPLES / "demo_data" / "demo_labeled.csv"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate QC viewer demo GIF (marker-label-view styling)"
    )
    parser.add_argument(
        "--synthetic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Procedural walk only (default: reference trial clip, centered)",
    )
    parser.add_argument(
        "--direct",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Export input CSV as-is (no centering); matches marker-label-view on that file",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Local labeled CSV for centered reference clip (--reference required unless --synthetic)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input CSV path (required with --direct)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ASSETS / "qc_viewer_demo.gif",
        help="Output GIF path",
    )
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument(
        "--speed",
        type=float,
        default=0.5,
        help="GIF playback speed multiplier (<1 slower; default 0.5)",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=3.0,
        help="Walking clip length in seconds (real-time playback)",
    )
    parser.add_argument(
        "--frame-start",
        type=int,
        default=0,
        help="Start frame index (0-based)",
    )
    parser.add_argument(
        "--hide-obstacles",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Hide obstacle markers and OBS labels (default: true)",
    )
    parser.add_argument(
        "--x-clip-max",
        type=float,
        default=None,
        metavar="MM",
        help="Hide markers with lab X >= MM (e.g. 540 drops +X obstacles)",
    )
    parser.add_argument(
        "--y-clip-min",
        type=float,
        default=-1000.0,
        metavar="MM",
        help="Hide markers with lab Y below MM (default -1000)",
    )
    parser.add_argument(
        "--y-clip-max",
        type=float,
        default=1200.0,
        metavar="MM",
        help="Hide markers with lab Y above MM (default 1200)",
    )
    args = parser.parse_args()

    reference = args.input or args.reference

    if args.direct:
        if reference is None or not reference.is_file():
            raise SystemExit("Input not found. Use --direct --input path/to/local.csv")
        export_qc_viewer_gif(
            str(reference),
            args.output,
            fps=args.fps,
            playback_speed=args.speed,
            frame_start=args.frame_start,
            duration_s=args.duration_s,
            hide_obstacles=args.hide_obstacles,
            x_clip_max_mm=args.x_clip_max,
            y_clip_min_mm=args.y_clip_min,
            y_clip_max_mm=args.y_clip_max,
        )
        print(f"Wrote {args.output}")
        print(f"Source (local only): {reference}")
        return

    if args.synthetic:
        fs = 100.0
        n_frames = max(1, int(round(args.duration_s * fs)))
        points, labels, rate = generate_synthetic_walking(n_frames=n_frames, fs=fs)
    else:
        if reference is None or not reference.is_file():
            raise SystemExit(
                "Reference trial not found. "
                "Use --synthetic, --direct --input PATH, or --reference PATH."
            )
        points, labels, rate = extract_demo_walking_clip(
            str(reference),
            frame_start=args.frame_start,
            duration_s=args.duration_s,
        )

    DEMO_CSV.parent.mkdir(parents=True, exist_ok=True)
    export_csv(str(DEMO_CSV), points, labels, rate=rate, first_frame=0)

    export_qc_viewer_gif(
        str(DEMO_CSV),
        args.output,
        fps=args.fps,
        playback_speed=args.speed,
        frame_start=0,
        duration_s=args.duration_s,
        hide_obstacles=args.hide_obstacles,
        x_clip_max_mm=args.x_clip_max,
        y_clip_min_mm=args.y_clip_min,
        y_clip_max_mm=args.y_clip_max,
    )
    print(f"Wrote {args.output}")
    print(f"Demo CSV (gitignored): {DEMO_CSV}")
    if not args.synthetic:
        print(f"Kinematics template (local only): {reference}")


if __name__ == "__main__":
    main()
