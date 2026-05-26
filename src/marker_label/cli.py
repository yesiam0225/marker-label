"""Command-line interface for the labeling pipeline."""

from __future__ import annotations

import argparse
import sys

from .constants import (
    DEFAULT_EXTRA_STATIONARY_MOTION_MAX_MM,
    DEFAULT_OBSTACLE_CANDIDATE_Y_MAX_MM,
    DEFAULT_OBSTACLE_CANDIDATE_Y_MIN_MM,
    DEFAULT_OBSTACLE_MAX_MOTION_MM,
    DEFAULT_OBSTACLE_ROD_LENGTH_MIN_MM,
    DEFAULT_OBSTACLE_ROD_PAIR_DX_MAX_MM,
    DEFAULT_OBSTACLE_ROD_PAIR_DZ_MAX_MM,
    DEFAULT_OBSTACLE_ROD_PAIR_LENGTH_TOL_FRACTION,
    DEFAULT_OBSTACLE_ROD_PAIR_LENGTH_TOL_MM,
    DEFAULT_OBSTACLE_ROD_PAIR_P90_MOTION_MAX_MM,
    DEFAULT_OBSTACLE_ROD_MAX_PAIR_CANDIDATES,
    DEFAULT_OBSTACLE_ROD_MIN_OVERLAP_FRAMES,
    DEFAULT_OBSTACLE_ROD_SCORE_WEIGHT_LENGTH,
    DEFAULT_OBSTACLE_ROD_SCORE_WEIGHT_MOTION,
    DEFAULT_OBSTACLE_ROD_SCORE_WEIGHT_X,
    DEFAULT_OBSTACLE_ROD_SCORE_WEIGHT_Z,
    DEFAULT_OBSTACLE_ROD_VISIBILITY_MIN,
    DEFAULT_OBSTACLE_VISIBILITY_FLOOR,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Label dynamic C3D or flat CSV using the subject's labeled static C3D or flat CSV. "
            "Both inputs may be .c3d or .csv (CSV = same layout as pipeline export: frame, time, marker_x/y/z)."
        ),
    )
    parser.add_argument(
        "static",
        help="Labeled static C3D or labeled flat CSV (anatomical marker columns)",
    )
    parser.add_argument(
        "dynamic",
        help="Unlabeled dynamic C3D or flat CSV (same frame/time/marker triplet layout as pipeline CSV export)",
    )
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
        default=DEFAULT_OBSTACLE_ROD_VISIBILITY_MIN,
        metavar="FRAC",
        help=(
            f"Min visibility for obstacle candidates (default: {DEFAULT_OBSTACLE_ROD_VISIBILITY_MIN}, "
            "aligned with rod_pair). Use e.g. 0.80 with --obstacle-mode legacy if needed."
        ),
    )
    parser.add_argument(
        "--obstacle-max-motion-mm",
        type=float,
        default=None,
        metavar="MM",
        help=(
            f"Obstacle candidates only if mean inter-frame displacement ≤ MM mm/frame "
            f"(same as motion_score). Omitted uses default {DEFAULT_OBSTACLE_MAX_MOTION_MM} mm/frame. "
            "Use 0 to disable the cap (legacy behavior)."
        ),
    )
    parser.add_argument(
        "--obstacle-mode",
        type=str,
        default="rod_pair",
        choices=("legacy", "rod_pair"),
        help=(
            "Obstacle detection: legacy (two lowest mean-motion columns among qualified) or "
            "rod_pair (median motion + rod geometry: lateral spread vs axial separation). "
            "For rod_pair, you may need a lower --obstacle-visibility (e.g. 0.72) if one endpoint "
            "has dropouts."
        ),
    )
    parser.add_argument(
        "--obstacle-visibility-floor",
        type=float,
        default=None,
        metavar="FRAC",
        help=(
            "Rod mode: exclude columns with visibility below this before pairing "
            f"(default {DEFAULT_OBSTACLE_VISIBILITY_FLOOR})."
        ),
    )
    parser.add_argument(
        "--obstacle-rod-axis",
        type=str,
        default="y",
        choices=("x", "y", "z"),
        help="Rod mode: lab axis along the bar (separation axis; default y).",
    )
    parser.add_argument(
        "--obstacle-rod-length-min-mm",
        type=float,
        default=None,
        metavar="MM",
        help=(
            "Rod mode: minimum axial separation (mm) for a valid pair "
            f"(default {DEFAULT_OBSTACLE_ROD_LENGTH_MIN_MM})."
        ),
    )
    parser.add_argument(
        "--obstacle-rod-length-target-mm",
        type=float,
        default=None,
        metavar="MM",
        help="Rod mode: optional soft target axial separation (mm) for the same rod across trials.",
    )
    parser.add_argument(
        "--obstacle-rod-max-candidates",
        type=int,
        default=None,
        metavar="K",
        help=(
            "Rod mode: only the K lowest-median-motion columns enter pairwise search "
            f"(default {DEFAULT_OBSTACLE_ROD_MAX_PAIR_CANDIDATES})."
        ),
    )
    parser.add_argument(
        "--obstacle-rod-min-overlap-frames",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Rod mode: minimum simultaneous finite-XYZ frames for pair geometry "
            f"(default {DEFAULT_OBSTACLE_ROD_MIN_OVERLAP_FRAMES})."
        ),
    )
    parser.add_argument(
        "--obstacle-rod-dx-max-mm",
        type=float,
        default=None,
        metavar="MM",
        help=(
            "Rod mode: pair gate for |dx| between endpoints in mm (for y-axis rod; "
            f"default {DEFAULT_OBSTACLE_ROD_PAIR_DX_MAX_MM}, <=0 disables)."
        ),
    )
    parser.add_argument(
        "--obstacle-rod-dz-max-mm",
        type=float,
        default=None,
        metavar="MM",
        help=(
            "Rod mode: pair gate for |dz| between endpoints in mm (for y-axis rod; "
            f"default {DEFAULT_OBSTACLE_ROD_PAIR_DZ_MAX_MM}, <=0 disables)."
        ),
    )
    parser.add_argument(
        "--obstacle-rod-length-tol-mm",
        type=float,
        default=None,
        metavar="MM",
        help=(
            "Rod mode: absolute tolerance for |axial-target| when --obstacle-rod-length-target-mm is set "
            f"(default {DEFAULT_OBSTACLE_ROD_PAIR_LENGTH_TOL_MM})."
        ),
    )
    parser.add_argument(
        "--obstacle-rod-length-tol-frac",
        type=float,
        default=None,
        metavar="F",
        help=(
            "Rod mode: relative tolerance fraction for |axial-target| when target is set "
            f"(default {DEFAULT_OBSTACLE_ROD_PAIR_LENGTH_TOL_FRACTION})."
        ),
    )
    parser.add_argument(
        "--obstacle-rod-p90-motion-max-mm",
        type=float,
        default=None,
        metavar="MM",
        help=(
            "Rod mode: per-column p90 inter-frame displacement cap in mm/frame "
            f"(default {DEFAULT_OBSTACLE_ROD_PAIR_P90_MOTION_MAX_MM}; <=0 disables)."
        ),
    )
    parser.add_argument(
        "--obstacle-rod-score-weight-x",
        type=float,
        default=None,
        metavar="W",
        help=f"Rod mode score weight for |dx| term (default {DEFAULT_OBSTACLE_ROD_SCORE_WEIGHT_X}).",
    )
    parser.add_argument(
        "--obstacle-rod-score-weight-length",
        type=float,
        default=None,
        metavar="W",
        help=f"Rod mode score weight for length mismatch term (default {DEFAULT_OBSTACLE_ROD_SCORE_WEIGHT_LENGTH}).",
    )
    parser.add_argument(
        "--obstacle-rod-score-weight-z",
        type=float,
        default=None,
        metavar="W",
        help=f"Rod mode score weight for |dz| term (default {DEFAULT_OBSTACLE_ROD_SCORE_WEIGHT_Z}).",
    )
    parser.add_argument(
        "--obstacle-rod-score-weight-motion",
        type=float,
        default=None,
        metavar="W",
        help=f"Rod mode score weight for motion tie term (default {DEFAULT_OBSTACLE_ROD_SCORE_WEIGHT_MOTION}).",
    )
    parser.add_argument(
        "--obstacle-candidate-y-min-mm",
        type=float,
        default=DEFAULT_OBSTACLE_CANDIDATE_Y_MIN_MM,
        metavar="MM",
        help=(
            "Obstacle columns only if median lab Y (mm) is in [min, max] (default: "
            f"{DEFAULT_OBSTACLE_CANDIDATE_Y_MIN_MM}..{DEFAULT_OBSTACLE_CANDIDATE_Y_MAX_MM})."
        ),
    )
    parser.add_argument(
        "--obstacle-candidate-y-max-mm",
        type=float,
        default=DEFAULT_OBSTACLE_CANDIDATE_Y_MAX_MM,
        metavar="MM",
        help="Upper bound of median Y (mm) for obstacle candidates; see --obstacle-candidate-y-min-mm.",
    )
    parser.add_argument(
        "--obstacle-no-candidate-y-range",
        action="store_true",
        help="Allow obstacle selection without filtering by median lab Y (disables the default band).",
    )
    parser.add_argument(
        "--obstacle-force-columns",
        type=str,
        default=None,
        metavar="I,J",
        help=(
            "Two loaded 0-based dynamic column indices (comma-separated) to use as obstacles; "
            "skips automatic detection. After --drop-loaded-columns, use indices in that file."
        ),
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
        "--reference-report",
        type=str,
        default=None,
        metavar="JSON",
        help="Use reference analysis JSON (from marker-label-analyze -o) to set best-frame window and optional distance thresholds.",
    )
    parser.add_argument(
        "--no-check-screened-count",
        action="store_true",
        help=(
            "Do not require at least 41 screened columns after extra stationary drop "
            "(use for trials with different channel count)."
        ),
    )
    parser.add_argument(
        "--y-range-screening",
        action="store_true",
        help=(
            "Enable Step 1: drop columns by fixed lab Y band ([-1000, 1000] mm by default). "
            "Default: Step 1 is skipped."
        ),
    )
    parser.add_argument(
        "--obstacle-y-aggregate",
        choices=("median", "mean"),
        default="median",
        help="Trial-wide vertical band from obstacle markers: median or mean Y per endpoint (default: median).",
    )
    parser.add_argument(
        "--skip-visibility-screening",
        action="store_true",
        help=(
            "Skip Step 2: do not drop columns by per-column visibility fraction "
            "(after optional Step 1 Y screening and optional frame trim)."
        ),
    )
    parser.add_argument(
        "--first-frame",
        type=int,
        default=None,
        metavar="N",
        help="Step 1.5: first frame to keep (1-based). Use with --last-frame to trim trial range.",
    )
    parser.add_argument(
        "--last-frame",
        type=int,
        default=None,
        metavar="N",
        help="Step 1.5: last frame to keep (1-based). Use with --first-frame to trim trial range.",
    )
    parser.add_argument(
        "--y-outside-fraction",
        type=float,
        default=None,
        metavar="F",
        help="Step 1: drop column if (finite Y outside band) / (finite Y frames) > F (default: from constants, 0.9).",
    )
    parser.add_argument(
        "--min-finite-y-frames",
        type=int,
        default=None,
        metavar="N",
        help="Step 1: drop column if count of frames with finite Y < N (default: from constants, 5).",
    )
    parser.add_argument(
        "--unlabeled-label-base",
        type=int,
        default=None,
        choices=(0, 1),
        metavar="B",
        help="Unlabeled marker names: display (loaded 0-based index) + B (default: from constants, 1 = 1-based).",
    )
    parser.add_argument(
        "--drop-loaded-columns",
        type=str,
        default=None,
        metavar="I[,J,...]",
        help="0-based column indices to remove from loaded dynamic (comma-separated), before Y/visibility screening.",
    )
    parser.add_argument(
        "--drop-loaded-label-regex",
        type=str,
        default=None,
        metavar="PATTERN",
        help=(
            "Python re pattern: drop loaded dynamic columns whose stripped marker name matches "
            "via re.search (before Y/visibility screening). Union with --drop-loaded-columns. "
            "If a trial uses only placeholder names (e.g. *0…*N), a broad pattern can match all "
            "columns; narrow the regex or use --drop-loaded-columns / --auto-drop-missing-fraction."
        ),
    )
    parser.add_argument(
        "--skip-labels",
        type=str,
        default=None,
        metavar="NAME[,NAME,...]",
        help="Static body label names to omit from labeling and from output (comma-separated), after column drop/screening.",
    )
    parser.add_argument(
        "--best-frame",
        type=int,
        default=None,
        metavar="N",
        help=(
            "0-based frame index for labeling (skip automatic best-frame selection). "
            "Does not change extra stationary column dropping (that uses full-trial motion)."
        ),
    )
    parser.add_argument(
        "--column-fixed-labels",
        action="store_true",
        help=(
            "Keep each body's label from the labeling frame on the same column for all frames "
            "(skip temporal nearest-neighbor propagation)."
        ),
    )
    parser.add_argument(
        "--auto-drop-missing-fraction",
        type=float,
        default=None,
        metavar="F",
        help=(
            "After --drop-loaded-columns, drop columns where at least this fraction of frames "
            "have no finite XYZ (e.g. 0.95 for empty C3D channels; disabled if omitted)."
        ),
    )
    parser.add_argument(
        "--drop-extra-stationary-below-mm",
        type=float,
        default=None,
        metavar="MM",
        help=(
            "After obstacle detection, drop non-obstacle screened columns when mean inter-frame "
            f"speed < MM (mm/frame) and visibility >= obstacle threshold. When omitted, uses "
            f"default {DEFAULT_EXTRA_STATIONARY_MOTION_MAX_MM} mm/frame (standard processing). "
            "Use 0 to disable only for exceptional cases (e.g. unusual channel count, debugging). "
            "Requires two obstacle markers. Uses full trial motion; not affected by --best-frame."
        ),
    )
    parser.add_argument(
        "--no-drop-extra-stationary",
        action="store_true",
        help=(
            "Disable extra stationary column dropping (overrides default threshold). "
            "Standard runs keep this enabled; use only for exceptional cases — document why."
        ),
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

    drop_loaded_columns: list[int] | None = None
    if args.drop_loaded_columns:
        parts = [p.strip() for p in args.drop_loaded_columns.split(",") if p.strip()]
        drop_loaded_columns = [int(p) for p in parts]

    skip_labels: list[str] | None = None
    if args.skip_labels:
        skip_labels = [p.strip() for p in args.skip_labels.split(",") if p.strip()]

    if args.no_drop_extra_stationary:
        drop_extra_stationary_mm: float | None = 0.0
    elif args.drop_extra_stationary_below_mm is not None:
        drop_extra_stationary_mm = float(args.drop_extra_stationary_below_mm)
    else:
        drop_extra_stationary_mm = None

    try:
        from .pipeline import run_pipeline
        _vf = (
            args.obstacle_visibility_floor
            if args.obstacle_visibility_floor is not None
            else DEFAULT_OBSTACLE_VISIBILITY_FLOOR
        )
        _rmin = (
            args.obstacle_rod_length_min_mm
            if args.obstacle_rod_length_min_mm is not None
            else DEFAULT_OBSTACLE_ROD_LENGTH_MIN_MM
        )
        _rk = (
            args.obstacle_rod_max_candidates
            if args.obstacle_rod_max_candidates is not None
            else DEFAULT_OBSTACLE_ROD_MAX_PAIR_CANDIDATES
        )
        _rov = (
            args.obstacle_rod_min_overlap_frames
            if args.obstacle_rod_min_overlap_frames is not None
            else DEFAULT_OBSTACLE_ROD_MIN_OVERLAP_FRAMES
        )
        _c_y_min: float | None
        _c_y_max: float | None
        if args.obstacle_no_candidate_y_range:
            _c_y_min, _c_y_max = None, None
        else:
            _c_y_min, _c_y_max = (
                float(args.obstacle_candidate_y_min_mm),
                float(args.obstacle_candidate_y_max_mm),
            )
        _rdx = (
            args.obstacle_rod_dx_max_mm
            if args.obstacle_rod_dx_max_mm is not None
            else DEFAULT_OBSTACLE_ROD_PAIR_DX_MAX_MM
        )
        _rdz = (
            args.obstacle_rod_dz_max_mm
            if args.obstacle_rod_dz_max_mm is not None
            else DEFAULT_OBSTACLE_ROD_PAIR_DZ_MAX_MM
        )
        _rtol_mm = (
            args.obstacle_rod_length_tol_mm
            if args.obstacle_rod_length_tol_mm is not None
            else DEFAULT_OBSTACLE_ROD_PAIR_LENGTH_TOL_MM
        )
        _rtol_frac = (
            args.obstacle_rod_length_tol_frac
            if args.obstacle_rod_length_tol_frac is not None
            else DEFAULT_OBSTACLE_ROD_PAIR_LENGTH_TOL_FRACTION
        )
        _rp90 = (
            args.obstacle_rod_p90_motion_max_mm
            if args.obstacle_rod_p90_motion_max_mm is not None
            else DEFAULT_OBSTACLE_ROD_PAIR_P90_MOTION_MAX_MM
        )
        _w_x = (
            args.obstacle_rod_score_weight_x
            if args.obstacle_rod_score_weight_x is not None
            else DEFAULT_OBSTACLE_ROD_SCORE_WEIGHT_X
        )
        _w_l = (
            args.obstacle_rod_score_weight_length
            if args.obstacle_rod_score_weight_length is not None
            else DEFAULT_OBSTACLE_ROD_SCORE_WEIGHT_LENGTH
        )
        _w_z = (
            args.obstacle_rod_score_weight_z
            if args.obstacle_rod_score_weight_z is not None
            else DEFAULT_OBSTACLE_ROD_SCORE_WEIGHT_Z
        )
        _w_m = (
            args.obstacle_rod_score_weight_motion
            if args.obstacle_rod_score_weight_motion is not None
            else DEFAULT_OBSTACLE_ROD_SCORE_WEIGHT_MOTION
        )
        _obstacle_force: tuple[int, int] | None
        if args.obstacle_force_columns:
            _parts = [p.strip() for p in str(args.obstacle_force_columns).split(",") if p.strip()]
            if len(_parts) != 2:
                print(
                    "Error: --obstacle-force-columns requires exactly two comma-separated indices (e.g. 0,25).",
                    file=sys.stderr,
                )
                raise SystemExit(2)
            _obstacle_force = (int(_parts[0]), int(_parts[1]))
        else:
            _obstacle_force = None
        info = run_pipeline(
            args.static,
            args.dynamic,
            out_prefix,
            obstacle_visibility_min=args.obstacle_visibility,
            obstacle_max_motion_mm=args.obstacle_max_motion_mm,
            obstacle_detection_mode=args.obstacle_mode,
            obstacle_visibility_floor=_vf,
            obstacle_rod_separation_axis=args.obstacle_rod_axis,
            obstacle_rod_length_min_mm=_rmin,
            obstacle_rod_length_target_mm=args.obstacle_rod_length_target_mm,
            obstacle_rod_max_pair_candidates=_rk,
            obstacle_rod_min_overlap_frames=_rov,
            obstacle_rod_pair_dx_max_mm=_rdx,
            obstacle_rod_pair_dz_max_mm=_rdz,
            obstacle_rod_pair_length_tol_mm=_rtol_mm,
            obstacle_rod_pair_length_tol_fraction=_rtol_frac,
            obstacle_rod_pair_p90_motion_max_mm=_rp90,
            obstacle_rod_score_weight_x=_w_x,
            obstacle_rod_score_weight_length=_w_l,
            obstacle_rod_score_weight_z=_w_z,
            obstacle_rod_score_weight_motion=_w_m,
            obstacle_candidate_y_min_mm=_c_y_min,
            obstacle_candidate_y_max_mm=_c_y_max,
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
            check_screened_count=not args.no_check_screened_count,
            trim_first_frame=args.first_frame,
            trim_last_frame=args.last_frame,
            skip_visibility_screening=args.skip_visibility_screening,
            skip_y_range_screening=not args.y_range_screening,
            obstacle_y_aggregate=args.obstacle_y_aggregate,
            y_outside_fraction_threshold=args.y_outside_fraction,
            min_finite_y_frames=args.min_finite_y_frames,
            unlabeled_numeric_base=args.unlabeled_label_base,
            drop_loaded_column_indices=drop_loaded_columns,
            drop_loaded_column_label_regex=args.drop_loaded_label_regex,
            skip_label_names=skip_labels,
            fixed_best_frame=args.best_frame,
            auto_drop_missing_fraction_ge=args.auto_drop_missing_fraction,
            column_fixed_labels=args.column_fixed_labels,
            drop_extra_stationary_motion_max_mm=drop_extra_stationary_mm,
            obstacle_force_loaded_column_indices=_obstacle_force,
        )
        print(f"Labeled {info['n_markers']} markers, {info['n_frames']} frames.")
        if "best_frame_1based" in info:
            print(
                f"Best frame (for review): 1-based = {info['best_frame_1based']}, "
                f"body columns = {info['n_body_marker_columns']}, "
                f"finite XYZ at best frame = {info['n_finite_xyz_at_best_frame']}."
            )
        print(f"Output: {out_prefix}_labeled.c3d, {out_prefix}_labeled.csv")
        if not args.no_filled:
            print(f"Filled: {out_prefix}_labeled_filled.c3d, {out_prefix}_labeled_filled.csv")
    except Exception as e:
        from .body_labeling import CLAVRBAKValidationError, C7ShoulderValidationError
        from .errors import LabelingPipelineError
        from .screening import ScreeningError

        if isinstance(e, LabelingPipelineError):
            print(
                f"Error [{e.error_code}] step={e.step}: {e.message}",
                file=sys.stderr,
            )
        elif isinstance(e, ScreeningError):
            print(
                f"Error [{e.error_code}] screening_step={e.step}: {e.message}",
                file=sys.stderr,
            )
        elif isinstance(e, CLAVRBAKValidationError):
            print(
                f"Error [{e.error_code}] step=labeling: {e.message}",
                file=sys.stderr,
            )
        elif isinstance(e, C7ShoulderValidationError):
            print(
                f"Error [{e.error_code}] step=labeling: {e.message}",
                file=sys.stderr,
            )
        else:
            print(f"Error: {e}", file=sys.stderr)
        if isinstance(e, CLAVRBAKValidationError) and (e.clav_idx is not None or e.rbak_idx is not None):
            print(f"Points tried as CLAV: index {e.clav_idx}, RBAK: index {e.rbak_idx}", file=sys.stderr)
        if isinstance(e, C7ShoulderValidationError) and (e.c7_idx is not None or e.lsho_idx is not None or e.rsho_idx is not None):
            print(f"C7/shoulder indices: C7={e.c7_idx}, LSHO={e.lsho_idx}, RSHO={e.rsho_idx}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
