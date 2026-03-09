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
    parser.add_argument(
        "--static-facing",
        type=str,
        default=None,
        metavar="AXIS",
        choices=("x", "-x", "y", "-y"),
        help="Lab axis subject faces in static (e.g. y). Use with --dynamic-facing when orientations differ.",
    )
    parser.add_argument(
        "--dynamic-facing",
        type=str,
        default=None,
        metavar="AXIS",
        choices=("x", "-x", "y", "-y"),
        help="Lab axis subject faces in dynamic (e.g. x). Rotates dynamic to align with static for matching.",
    )
    parser.add_argument(
        "--static-unit",
        type=str,
        default="mm",
        choices=("mm", "m"),
        help="Unit of static C3D coordinates: mm or m (default mm). Use m if file is in meters; values are scaled to mm internally.",
    )
    parser.add_argument(
        "--dynamic-unit",
        type=str,
        default="mm",
        choices=("mm", "m"),
        help="Unit of dynamic C3D coordinates: mm or m (default mm). Use m if file is in meters; values are scaled to mm internally.",
    )
    parser.add_argument(
        "--max-match-distance",
        type=float,
        default=None,
        metavar="MM",
        help="Reject initial assignment if point-template distance > MM mm (e.g. 300).",
    )
    parser.add_argument(
        "--max-propagation-distance",
        type=float,
        default=None,
        metavar="MM",
        help="Do not propagate label if nearest point is > MM mm (e.g. 150) to reduce swaps after dropout.",
    )
    parser.add_argument(
        "--whole-body-39",
        action="store_true",
        help="Use 39 whole-body markers only; anchor-based best frame and tiered distance caps (alpha*s).",
    )
    parser.add_argument(
        "--z-band-by-value",
        action="store_true",
        help="Define Z-bands by equal Z span (template min–max).",
    )
    parser.add_argument(
        "--z-band-by-gap",
        action="store_true",
        help="Define Z-band boundaries at largest Z gaps between consecutive markers (by difference).",
    )
    parser.add_argument(
        "--z-band-by-rank",
        action="store_true",
        help="Assign bands by Z rank in dynamic frame (highest Z → band 11, etc.); do not use static trial Z.",
    )
    parser.add_argument(
        "--match-head-first",
        action="store_true",
        help="Match head first, then RSHO/LSHO/C7 by midline (forward walking along X: L/R from Y, A/P from X).",
    )
    parser.add_argument(
        "--walking-axis-y",
        action="store_true",
        help="Forward walking is along Y-axis (L/R from X, A/P from Y). Default is X-axis.",
    )
    parser.add_argument(
        "--left-side-positive-y",
        action="store_true",
        help="Left side of body = positive Y in dynamic trial. Use if head or shoulder L/R are flipped (e.g. LFHD on right, RFHD on left).",
    )
    parser.add_argument(
        "--head-flip-lr",
        action="store_true",
        help="Flip head L/R only (LFHD↔RFHD, LBHD↔RBHD). Use when shoulders are correct but head markers are swapped.",
    )
    parser.add_argument(
        "--head-anterior-smaller-x",
        action="store_true",
        help="Head anterior (front) = smaller X. Use when LFHD and RBHD are swapped (front of head has smaller X in trial).",
    )
    parser.add_argument(
        "--head-anterior-larger-x",
        action="store_true",
        help="Force head anterior = larger X (overrides walking direction). Use when subject walks +X but LFHD/RBHD are still swapped (computed walking direction may be wrong).",
    )
    parser.add_argument(
        "--head-swap-lfhd-rbhd",
        action="store_true",
        help="After head assignment, swap LFHD and RBHD labels only (fixes diagonal swap when other 4-head labels are correct).",
    )
    parser.add_argument(
        "--head-align-to-shoulders",
        action="store_true",
        help="Align head L/R to shoulders (flip head if LFHD is on opposite side from LSHO). Default: L/R from walking direction only, no flip.",
    )
    parser.add_argument(
        "--reference-report",
        type=str,
        default=None,
        metavar="JSON",
        help="Use reference analysis JSON (from marker-label-analyze -o) to set best-frame window and optional distance thresholds.",
    )
    args = parser.parse_args()
    out_prefix = args.output
    if out_prefix is None:
        out_prefix = args.dynamic.rsplit(".", 1)[0] if "." in args.dynamic else args.dynamic
        out_prefix = f"{out_prefix}_labeled"

    middle_start = 0.20
    middle_end = 0.80
    max_match_distance = args.max_match_distance
    max_propagation_distance = args.max_propagation_distance

    if args.reference_report:
        import json
        from .analyze_reference import suggested_pipeline_params_from_reference
        with open(args.reference_report) as f:
            ref_report = json.load(f)
        ref_params = suggested_pipeline_params_from_reference(ref_report)
        middle_start = ref_params["middle_start"]
        middle_end = ref_params["middle_end"]
        if max_match_distance is None and ref_params.get("max_match_distance") is not None:
            max_match_distance = ref_params["max_match_distance"]
        if max_propagation_distance is None and ref_params.get("max_propagation_distance") is not None:
            max_propagation_distance = ref_params["max_propagation_distance"]

    unit_scale = {"mm": 1.0, "m": 1000.0}
    static_scale = unit_scale[args.static_unit]
    dynamic_scale = unit_scale[args.dynamic_unit]

    try:
        from .pipeline import run_pipeline
        info = run_pipeline(
            args.static,
            args.dynamic,
            out_prefix,
            obstacle_visibility_min=args.obstacle_visibility,
            use_whole_body_39=args.whole_body_39,
            z_band_by_value=args.z_band_by_value,
            z_band_by_gap=args.z_band_by_gap,
            z_band_by_rank=args.z_band_by_rank,
            match_head_first=args.match_head_first,
            walking_axis_x=not args.walking_axis_y,
            left_side_positive_lr=args.left_side_positive_y,
            head_flip_lr=args.head_flip_lr,
            head_anterior_smaller_x=args.head_anterior_smaller_x,
            head_anterior_larger_x=args.head_anterior_larger_x,
            head_swap_lfhd_rbhd=args.head_swap_lfhd_rbhd,
            head_align_to_shoulders=args.head_align_to_shoulders,
            static_facing_axis=args.static_facing,
            dynamic_facing_axis=args.dynamic_facing,
            static_scale=static_scale,
            dynamic_scale=dynamic_scale,
            middle_start=middle_start,
            middle_end=middle_end,
            max_match_distance=max_match_distance,
            max_propagation_distance=max_propagation_distance,
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
