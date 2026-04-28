"""Full labeling pipeline: static + dynamic -> labeled C3D and CSV."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .io import load_c3d
from .constants import (
    PELVIS_MARKERS,
    DEFAULT_EXTRA_STATIONARY_MOTION_MAX_MM,
    DEFAULT_OBSTACLE_MAX_MOTION_MM,
    DEFAULT_OBSTACLE_ROD_LENGTH_MIN_MM,
    DEFAULT_OBSTACLE_ROD_PAIR_DX_MAX_MM,
    DEFAULT_OBSTACLE_ROD_PAIR_DZ_MAX_MM,
    DEFAULT_OBSTACLE_ROD_PAIR_LENGTH_TOL_FRACTION,
    DEFAULT_OBSTACLE_ROD_PAIR_LENGTH_TOL_MM,
    DEFAULT_OBSTACLE_ROD_PAIR_P90_MOTION_MAX_MM,
    DEFAULT_OBSTACLE_ROD_MAX_PAIR_CANDIDATES,
    DEFAULT_OBSTACLE_CANDIDATE_Y_MAX_MM,
    DEFAULT_OBSTACLE_CANDIDATE_Y_MIN_MM,
    DEFAULT_OBSTACLE_ROD_MIN_OVERLAP_FRAMES,
    DEFAULT_OBSTACLE_ROD_SCORE_WEIGHT_LENGTH,
    DEFAULT_OBSTACLE_ROD_SCORE_WEIGHT_MOTION,
    DEFAULT_OBSTACLE_ROD_SCORE_WEIGHT_X,
    DEFAULT_OBSTACLE_ROD_SCORE_WEIGHT_Z,
    DEFAULT_OBSTACLE_ROD_VISIBILITY_MIN,
    DEFAULT_OBSTACLE_VISIBILITY_FLOOR,
    MIN_BODY_MARKER_COLUMNS,
    MIN_SCREENED_COLUMNS_AFTER_EXTRA_STATIONARY,
    OBSTACLE_LABELS,
    SCREENING_Y_MIN_FINITE_FRAMES,
    SCREENING_Y_OUTSIDE_FRACTION_THRESHOLD,
    UNLABELED_NUMERIC_LABEL_BASE,
    WHOLE_BODY_39_SET,
    Z_BAND_SIZES_NO_ARM_HAND,
)
from .errors import (
    ERR_AUTO_DROP_INVALID_THRESHOLD,
    ERR_AUTO_DROP_REMOVE_ALL,
    ERR_BODY_COLUMNS_LT_MIN,
    ERR_DROP_COLUMN_OUT_OF_RANGE,
    ERR_DROP_COLUMN_REMOVE_ALL,
    ERR_EXTRA_STATIONARY_INVALID,
    ERR_EXTRA_STATIONARY_REMOVE_ALL,
    ERR_OBSTACLE_FORCED_INVALID,
    ERR_OBSTACLE_FORCED_NOT_FOUND,
    ERR_OBSTACLE_MODE,
    ERR_SCREENED_COLUMNS_LT_MIN,
    ERR_SKIP_LABEL_EMPTY_TEMPLATE,
    ERR_SKIP_LABEL_OBSTACLE,
    ERR_SKIP_LABEL_UNKNOWN,
    LabelingPipelineError,
)
from .screening import (
    apply_initial_screening_steps_1_and_2,
    compute_walking_direction_x,
    get_lr_ap_axes_from_walking,
    loaded_point_indices_for_body,
    ScreeningError,
)
from .obstacle import (
    detect_obstacle_markers,
    detect_obstacle_markers_rod_pair,
    order_obstacle_l_r_for_screened_pair,
    remap_indices_after_screened_drops,
    screened_indices_extra_stationary_to_drop,
    trial_obstacle_y_band,
)
from .body_labeling import build_template_from_static, label_body_markers
from .static_geometry import (
    build_z_band_by_rank_from_static,
    build_z_band_no_arm_hand_from_static,
    build_segment_geometry_from_static,
    save_z_band_to_json,
)
from .export import build_full_trajectory_matrix, export_labeled

# Facing axis: subject's forward direction in lab (z = up). Used to align dynamic to static.
FACING_AXIS_OPTIONS = ("x", "-x", "y", "-y")


def apply_drop_loaded_column_indices_to_c3d_dict(
    data: dict,
    drop_indices: Sequence[int],
) -> None:
    """
    Remove marker columns from a ``load_c3d``-style dict **in place** (before screening).

    ``drop_indices`` are 0-based column indices in the loaded dynamic C3D (same convention
    as ``loaded_point_indices_for_body`` / unlabeled numeric labels without display base).
    """
    if not drop_indices:
        return
    pts = data["points"]
    n_points = int(pts.shape[1])
    drop_set = {int(i) for i in drop_indices}
    invalid = sorted(drop_set - set(range(n_points)))
    if invalid:
        raise LabelingPipelineError(
            ERR_DROP_COLUMN_OUT_OF_RANGE,
            f"drop_loaded_column_indices out of range for loaded dynamic (n_points={n_points}): {invalid}",
            step="drop_columns",
        )
    if len(drop_set) >= n_points:
        raise LabelingPipelineError(
            ERR_DROP_COLUMN_REMOVE_ALL,
            "drop_loaded_column_indices would remove all marker columns.",
            step="drop_columns",
        )
    keep = [i for i in range(n_points) if i not in drop_set]
    data["points"] = data["points"][:, keep, :].copy()
    res = data.get("residual")
    if res is not None:
        data["residual"] = res[:, keep].copy()
    labels = data.get("labels") or []
    new_labels = [labels[i] for i in keep] if labels else [f"Point_{j}" for j in range(len(keep))]
    data["labels"] = new_labels
    data["point_labels"] = new_labels
    data["n_points"] = len(keep)


def apply_drop_almost_empty_columns_to_c3d_dict(
    data: dict,
    *,
    missing_fraction_ge: float,
    loaded_column_aliases: np.ndarray,
) -> tuple[np.ndarray, list[int]]:
    """
    Remove columns where the fraction of frames with no finite XYZ is >= ``missing_fraction_ge``.

    C3D stores missing markers as (0,0,0); ``load_c3d`` converts those to NaN, which count as
    non-finite here. Runs **after** manual ``drop_loaded_column_indices``.

    Parameters
    ----------
    data : ``load_c3d``-style dict, modified in place.
    missing_fraction_ge : drop column j if ``(1 - finite_frame_fraction[j]) >= missing_fraction_ge``.
        Typical values: 0.9–0.99 for nearly-unused capture channels.
    loaded_column_aliases : length ``n_points``; ``loaded_column_aliases[j]`` = original file
        0-based column index for current column ``j``.

    Returns
    -------
    loaded_column_aliases : updated array, length n_keep
    removed_original_indices : original-file 0-based indices for dropped columns
    """
    if not (0.0 < float(missing_fraction_ge) <= 1.0):
        raise LabelingPipelineError(
            ERR_AUTO_DROP_INVALID_THRESHOLD,
            f"auto_drop_missing_fraction_ge must be in (0, 1], got {missing_fraction_ge!r}",
            step="drop_columns",
        )
    pts = data["points"]
    n_frames, n_points, _ = pts.shape
    if n_frames < 1 or n_points < 1:
        return loaded_column_aliases, []
    finite = np.isfinite(pts).all(axis=2)
    miss_frac = 1.0 - np.mean(finite, axis=0).astype(np.float64)
    drop_mask = miss_frac >= float(missing_fraction_ge) - 1e-15
    n_drop = int(np.sum(drop_mask))
    if n_drop == 0:
        return loaded_column_aliases, []
    removed_original = [int(loaded_column_aliases[j]) for j in range(n_points) if drop_mask[j]]
    if n_drop >= n_points:
        raise LabelingPipelineError(
            ERR_AUTO_DROP_REMOVE_ALL,
            f"auto-drop would remove all {n_points} marker columns (missing_fraction_ge={missing_fraction_ge}).",
            step="drop_columns",
        )
    keep = [j for j in range(n_points) if not drop_mask[j]]
    data["points"] = data["points"][:, keep, :].copy()
    res = data.get("residual")
    if res is not None:
        data["residual"] = res[:, keep].copy()
    labels = data.get("labels") or []
    new_labels = [labels[i] for i in keep] if labels else [f"Point_{j}" for j in range(len(keep))]
    data["labels"] = new_labels
    data["point_labels"] = new_labels
    data["n_points"] = len(keep)
    new_aliases = loaded_column_aliases[np.asarray(keep, dtype=np.int64)]
    return new_aliases, removed_original


def _rotation_matrix_z_rad(angle_rad: float) -> np.ndarray:
    """Rotation matrix around lab z (counterclockwise looking down). Shape (3, 3)."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def rotation_for_facing_axes(
    static_facing: str,
    dynamic_facing: str,
) -> np.ndarray | None:
    """
    Rotation matrix (3, 3) to align dynamic trial with static trial in the horizontal plane.

    Use when static was captured with subject facing one lab axis (e.g. y) and dynamic
    with subject facing another (e.g. x). Applying this rotation to dynamic points
    before matching makes template positions comparable.

    Parameters
    ----------
    static_facing : 'x', '-x', 'y', or '-y' (direction subject faces in static)
    dynamic_facing : same for dynamic trial

    Returns
    -------
    R : (3, 3) or None if no rotation needed (same axis) or invalid
    """
    static_facing = static_facing.strip().lower()
    dynamic_facing = dynamic_facing.strip().lower()
    if static_facing not in FACING_AXIS_OPTIONS or dynamic_facing not in FACING_AXIS_OPTIONS:
        return None
    if static_facing == dynamic_facing:
        return None
    # Angle of "forward" in xy plane: x=0, y=pi/2, -x=pi, -y=-pi/2
    axis_to_angle = {"x": 0.0, "y": math.pi / 2, "-x": math.pi, "-y": -math.pi / 2}
    a_static = axis_to_angle[static_facing]
    a_dynamic = axis_to_angle[dynamic_facing]
    angle_rad = a_static - a_dynamic
    return _rotation_matrix_z_rad(angle_rad)


def _apply_drop_screened_columns(
    points_d: np.ndarray,
    residual_d: np.ndarray | None,
    drop_screened: Sequence[int],
    screening_keep_for_loaded: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Remove screened columns by index; update ``screening_keep_for_loaded``."""
    drop_set = sorted(set(int(x) for x in drop_screened))
    n = int(points_d.shape[1])
    keep = [j for j in range(n) if j not in drop_set]
    if len(keep) == 0:
        raise LabelingPipelineError(
            ERR_EXTRA_STATIONARY_REMOVE_ALL,
            "extra stationary drop would remove all screened marker columns.",
            step="extra_stationary",
        )
    new_pts = points_d[:, keep, :].copy()
    new_res = residual_d[:, keep].copy() if residual_d is not None else None
    new_sk = screening_keep_for_loaded[np.asarray(keep, dtype=np.int64)]
    return new_pts, new_res, new_sk


def run_pipeline(
    static_path: str,
    dynamic_path: str,
    out_prefix: str,
    *,
    obstacle_visibility_min: float = DEFAULT_OBSTACLE_ROD_VISIBILITY_MIN,
    obstacle_max_motion_mm: float | None = None,
    obstacle_detection_mode: str = "rod_pair",
    obstacle_visibility_floor: float = DEFAULT_OBSTACLE_VISIBILITY_FLOOR,
    obstacle_rod_separation_axis: str = "y",
    obstacle_rod_length_min_mm: float = DEFAULT_OBSTACLE_ROD_LENGTH_MIN_MM,
    obstacle_rod_length_target_mm: float | None = None,
    obstacle_rod_max_pair_candidates: int = DEFAULT_OBSTACLE_ROD_MAX_PAIR_CANDIDATES,
    obstacle_rod_min_overlap_frames: int = DEFAULT_OBSTACLE_ROD_MIN_OVERLAP_FRAMES,
    obstacle_rod_pair_dx_max_mm: float = DEFAULT_OBSTACLE_ROD_PAIR_DX_MAX_MM,
    obstacle_rod_pair_dz_max_mm: float = DEFAULT_OBSTACLE_ROD_PAIR_DZ_MAX_MM,
    obstacle_rod_pair_length_tol_mm: float = DEFAULT_OBSTACLE_ROD_PAIR_LENGTH_TOL_MM,
    obstacle_rod_pair_length_tol_fraction: float = DEFAULT_OBSTACLE_ROD_PAIR_LENGTH_TOL_FRACTION,
    obstacle_rod_pair_p90_motion_max_mm: float = DEFAULT_OBSTACLE_ROD_PAIR_P90_MOTION_MAX_MM,
    obstacle_rod_score_weight_x: float = DEFAULT_OBSTACLE_ROD_SCORE_WEIGHT_X,
    obstacle_rod_score_weight_length: float = DEFAULT_OBSTACLE_ROD_SCORE_WEIGHT_LENGTH,
    obstacle_rod_score_weight_z: float = DEFAULT_OBSTACLE_ROD_SCORE_WEIGHT_Z,
    obstacle_rod_score_weight_motion: float = DEFAULT_OBSTACLE_ROD_SCORE_WEIGHT_MOTION,
    obstacle_candidate_y_min_mm: float | None = DEFAULT_OBSTACLE_CANDIDATE_Y_MIN_MM,
    obstacle_candidate_y_max_mm: float | None = DEFAULT_OBSTACLE_CANDIDATE_Y_MAX_MM,
    use_pelvis_frame: bool = False,
    static_facing_axis: str | None = None,
    dynamic_facing_axis: str | None = None,
    static_scale: float = 1.0,
    dynamic_scale: float = 1.0,
    middle_start: float = 0.20,
    middle_end: float = 0.80,
    use_hungarian: bool = True,
    max_match_distance: float | None = None,
    max_propagation_distance: float | None = None,
    export_filled: bool = True,
    max_interp_frames: int = 10,
    require_obstacles: bool = False,
    check_screened_count: bool = True,
    trim_first_frame: int | None = None,
    trim_last_frame: int | None = None,
    skip_visibility_screening: bool = False,
    skip_y_range_screening: bool = True,
    obstacle_y_aggregate: str = "median",
    y_outside_fraction_threshold: float | None = None,
    min_finite_y_frames: int | None = None,
    unlabeled_numeric_base: int | None = None,
    drop_loaded_column_indices: Sequence[int] | None = None,
    skip_label_names: Sequence[str] | None = None,
    fixed_best_frame: int | None = None,
    auto_drop_missing_fraction_ge: float | None = None,
    column_fixed_labels: bool = False,
    drop_extra_stationary_motion_max_mm: float | None = None,
    obstacle_force_loaded_column_indices: tuple[int, int] | list[int] | None = None,
) -> dict:
    """
    Run the full labeling pipeline.

    Parameters
    ----------
    static_path : path to labeled static C3D
    dynamic_path : path to unlabeled dynamic C3D
    out_prefix : output path prefix (e.g. "out/trial_01")
    obstacle_visibility_min : min visibility for obstacle candidates (default
        ``DEFAULT_OBSTACLE_ROD_VISIBILITY_MIN`` = 0.72; use a higher value with
        ``obstacle_detection_mode="legacy"`` if needed)
    obstacle_max_motion_mm : max mean inter-frame displacement (mm/frame) for obstacle candidates;
        same statistic as ``motion_score_per_marker``. ``None`` uses ``DEFAULT_OBSTACLE_MAX_MOTION_MM``
        (5.0). Use ``0`` to disable the motion upper bound (legacy behavior).
        For ``obstacle_detection_mode="rod_pair"``, the cap applies to **median** inter-frame speed.
    obstacle_detection_mode : ``"legacy"`` (lowest-mean-motion pair among visibility/motion-qualified
        columns) or ``"rod_pair"`` (rod geometry: lateral spread vs axial separation + median motion).
    obstacle_visibility_floor : in ``rod_pair`` mode, columns below this visibility are excluded
        before pairing (junk channels).
    obstacle_rod_separation_axis : lab axis along which the rod is oriented (``"x"``, ``"y"``, or
        ``"z"``). Lateral spread uses the other two axes.
    obstacle_rod_length_min_mm : minimum axial separation (mm) for a valid obstacle pair.
    obstacle_rod_length_target_mm : optional soft target for axial separation (same rod across trials).
    obstacle_rod_max_pair_candidates : only the lowest-median-motion K columns enter pairwise search.
    obstacle_rod_min_overlap_frames : minimum simultaneous finite-XYZ frames for a pair when
        computing mean positions for rod geometry.
    obstacle_rod_pair_dx_max_mm, obstacle_rod_pair_dz_max_mm : hard pairwise mismatch gates in mm
        for ``rod_pair`` mode when ``obstacle_rod_separation_axis='y'``.
    obstacle_rod_pair_length_tol_mm, obstacle_rod_pair_length_tol_fraction : when
        ``obstacle_rod_length_target_mm`` is set, require ``|axial-target|`` <=
        ``max(length_tol_mm, length_tol_fraction*target)``.
    obstacle_rod_pair_p90_motion_max_mm : ``rod_pair`` candidate filter on per-column p90 inter-frame
        displacement; <= 0 disables.
    obstacle_rod_score_weight_x, obstacle_rod_score_weight_length, obstacle_rod_score_weight_z,
    obstacle_rod_score_weight_motion : weighted-score coefficients in ``rod_pair`` pair selection.
    obstacle_candidate_y_min_mm, obstacle_candidate_y_max_mm : if either is not ``None``, only
        markers with **median lab Y** in ``[y_min, y_max]`` (inclusive) are obstacle candidates
        (default ``-500`` to ``1500`` mm). Pass both ``None`` to disable (no Y restriction).
    use_pelvis_frame : build template in pelvis frame (default False = lab frame)
    static_facing_axis : lab axis subject faces in static: 'x', '-x', 'y', '-y' (optional)
    dynamic_facing_axis : lab axis subject faces in dynamic (optional). If both set and
        different, dynamic body points are rotated in xy so orientations align for matching.
    static_scale : multiply static point coordinates by this after loading (use 1000 if file is in m, default 1 = mm).
    dynamic_scale : multiply dynamic point coordinates by this after loading (use 1000 if file is in m).
    middle_start, middle_end : frame range for best-frame selection
    use_hungarian : use optimal assignment at best frame (default True)
    max_match_distance : reject initial assignment if distance > this mm (optional)
    max_propagation_distance : do not propagate if nearest point > this mm per frame (optional)
    export_filled : also export filled C3D/CSV
    max_interp_frames : gap-fill threshold
    require_obstacles : if True, raise ScreeningError (Step 4) when fewer than 2 obstacle
        markers are detected (default False).
    check_screened_count : if True (default), after extra stationary drop require at least
        ``MIN_SCREENED_COLUMNS_AFTER_EXTRA_STATIONARY`` (41) screened columns. Set False to skip.
    skip_y_range_screening : if True (default), skip Step 1 (fixed lab Y band column drop).
        Set False to enable legacy Step 1 ``drop_columns_outside_y_range``.
    obstacle_y_aggregate : ``\"median\"`` or ``\"mean\"`` for trial-wide obstacle Y band
        (``trial_obstacle_y_band``) used for best-frame selection and labeling exclusion.
    trim_first_frame, trim_last_frame : optional 1-based first/last frame to keep
        (Step 1.5). If both set, trim dynamic to this range before visibility screening.
    skip_visibility_screening : if True, skip Step 2 (do not drop columns by visibility).
    y_outside_fraction_threshold, min_finite_y_frames : Step 1 Y screening; None uses defaults from constants.
    unlabeled_numeric_base : added to loaded 0-based index for unlabeled marker names in export (None = constant).
    drop_loaded_column_indices : optional 0-based point column indices to remove from dynamic
        immediately after ``load_c3d`` (before screening). Per-trial fix for bad channels.
    skip_label_names : optional anatomical labels to **not** assign during body labeling, applied
        after column drop and screening. Output omits these columns entirely (no NaN placeholders).
    fixed_best_frame : optional 0-based frame index for head/Z-band assignment (skips automatic
        best-frame selection). Does **not** affect extra stationary column dropping: that step runs
        earlier, uses mean inter-frame motion over **all** frames in the current dynamic trial, and
        never consults ``fixed_best_frame`` or automatic best-frame logic.
    auto_drop_missing_fraction_ge : optional; after manual column drop, remove any column whose
        fraction of frames without finite XYZ is >= this value (e.g. 0.95 for empty C3D channels).
        Disabled when ``None`` (default).
    column_fixed_labels : if True, keep each body's best-frame label on the same column for all
        frames (skip temporal nearest-neighbor propagation).
    drop_extra_stationary_motion_max_mm : mean inter-frame speed threshold (mm/frame) for dropping
        extra stationary screened columns after obstacle detection. ``None`` uses
        ``DEFAULT_EXTRA_STATIONARY_MOTION_MAX_MM`` from constants — **standard pipeline** behavior
        when two obstacles are detected. Use ``0`` only to disable for **exceptional** cases
        (e.g. debugging, trials where the drop removes too many channels); team policy should
        record the reason out of band. Requires at least two obstacle markers detected.
        Motion is computed over the **full** trajectory (same as obstacle motion scores); it does
        not depend on ``fixed_best_frame`` or ``middle_start`` / ``middle_end``.
    obstacle_force_loaded_column_indices : if set, two **loaded** 0-based dynamic column indices
        (same convention as ``drop_loaded_column_indices`` after any manual column drop) that are
        the obstacle pair. Skips automatic obstacle detection; L/R is assigned from lab Y. Raises
        if either column is not in the screened set.

    Returns
    -------
    info : dict with keys (static_labels, obstacle_indices, etc.)
    """
    static = load_c3d(static_path, scale_factor=static_scale)
    dynamic = load_c3d(dynamic_path, scale_factor=dynamic_scale)
    points_s = static["points"]
    labels_s = static["labels"]
    original_n_dynamic = int(dynamic["points"].shape[1])
    drop_set = (
        frozenset(int(i) for i in drop_loaded_column_indices)
        if drop_loaded_column_indices
        else frozenset()
    )
    if drop_loaded_column_indices:
        apply_drop_loaded_column_indices_to_c3d_dict(dynamic, drop_loaded_column_indices)
    # Map each dynamic column (after optional drop) -> original loaded file 0-based column index
    loaded_column_aliases = np.array(
        [j for j in range(original_n_dynamic) if j not in drop_set],
        dtype=np.int64,
    )

    auto_dropped_loaded_indices: list[int] = []
    if auto_drop_missing_fraction_ge is not None:
        loaded_column_aliases, auto_dropped_loaded_indices = apply_drop_almost_empty_columns_to_c3d_dict(
            dynamic,
            missing_fraction_ge=auto_drop_missing_fraction_ge,
            loaded_column_aliases=loaded_column_aliases,
        )
        if auto_dropped_loaded_indices:
            print(
                "Auto-dropped loaded columns (missing XYZ fraction >= "
                f"{float(auto_drop_missing_fraction_ge):.4f}): "
                f"{sorted(auto_dropped_loaded_indices)}"
            )

    points_d = dynamic["points"]
    residual_d = dynamic.get("residual")
    rate = dynamic.get("rate") or 0.0
    first_frame = dynamic.get("first_frame") or 1

    _y_frac = (
        SCREENING_Y_OUTSIDE_FRACTION_THRESHOLD
        if y_outside_fraction_threshold is None
        else y_outside_fraction_threshold
    )
    _y_min_fin = (
        SCREENING_Y_MIN_FINITE_FRAMES
        if min_finite_y_frames is None
        else min_finite_y_frames
    )

    # Dynamic trial initial screening (Steps 1–1.5–2): optional Y range (default off), trim, visibility
    points_d, residual_d, screening_keep, frame_trim_start = apply_initial_screening_steps_1_and_2(
        points_d, residual_d,
        trim_first_frame=trim_first_frame,
        trim_last_frame=trim_last_frame,
        skip_visibility_step=skip_visibility_screening,
        skip_y_range_screening=skip_y_range_screening,
        y_outside_fraction_threshold=_y_frac,
        min_finite_y_frames=_y_min_fin,
    )
    screening_keep_for_loaded = loaded_column_aliases[np.asarray(screening_keep, dtype=np.int64)]
    if trim_first_frame is not None and trim_last_frame is not None:
        first_frame = trim_first_frame
    else:
        first_frame = (first_frame or 1) + frame_trim_start
    n_screened = points_d.shape[1]
    # Step 3: walking direction and L/R–A/P axes (for head marker assignment)
    wdx = compute_walking_direction_x(points_d)
    d_back, d_right = get_lr_ap_axes_from_walking(wdx)

    # Step 4: Obstacle detection on screened dynamic; L/R by y-axis (walking along x)
    _mode = str(obstacle_detection_mode).strip().lower()
    _obstacle_forced: tuple[int, int] | None = None
    if obstacle_force_loaded_column_indices is not None:
        try:
            a, b = (
                int(obstacle_force_loaded_column_indices[0]),
                int(obstacle_force_loaded_column_indices[1]),
            )
        except (TypeError, ValueError, IndexError) as e:
            raise LabelingPipelineError(
                ERR_OBSTACLE_FORCED_INVALID,
                f"obstacle_force_loaded_column_indices must be a pair of ints, got {obstacle_force_loaded_column_indices!r}.",
                step="obstacle",
            ) from e
        if a == b:
            raise LabelingPipelineError(
                ERR_OBSTACLE_FORCED_INVALID,
                "obstacle_force_loaded_column_indices must be two different columns.",
                step="obstacle",
            )
        _obstacle_forced = (a, b)
        loaded_to_screen: dict[int, int] = {}
        for j in range(n_screened):
            li = int(screening_keep_for_loaded[j])
            if li == a:
                loaded_to_screen[a] = j
            if li == b:
                loaded_to_screen[b] = j
        miss = [x for x in (a, b) if x not in loaded_to_screen]
        if miss:
            raise LabelingPipelineError(
                ERR_OBSTACLE_FORCED_NOT_FOUND,
                f"forced obstacle column(s) not in screened set (dropped by screening or out of range): {miss}.",
                step="obstacle",
            )
        try:
            obstacle_indices, obstacle_labels = order_obstacle_l_r_for_screened_pair(
                points_d, loaded_to_screen[a], loaded_to_screen[b], lr_axis="y"
            )
        except ValueError as e:
            raise LabelingPipelineError(
                ERR_OBSTACLE_FORCED_INVALID,
                str(e),
                step="obstacle",
            ) from e
    elif _mode == "rod_pair":
        obstacle_indices, obstacle_labels = detect_obstacle_markers_rod_pair(
            points_d,
            visibility_min=obstacle_visibility_min,
            visibility_floor=obstacle_visibility_floor,
            motion_max_mm=obstacle_max_motion_mm,
            rod_separation_axis=obstacle_rod_separation_axis,
            rod_length_min_mm=obstacle_rod_length_min_mm,
            rod_length_target_mm=obstacle_rod_length_target_mm,
            rod_max_pair_candidates=obstacle_rod_max_pair_candidates,
            rod_min_overlap_frames=obstacle_rod_min_overlap_frames,
            rod_pair_dx_max_mm=obstacle_rod_pair_dx_max_mm,
            rod_pair_dz_max_mm=obstacle_rod_pair_dz_max_mm,
            rod_pair_length_tol_mm=obstacle_rod_pair_length_tol_mm,
            rod_pair_length_tol_fraction=obstacle_rod_pair_length_tol_fraction,
            rod_pair_p90_motion_max_mm=obstacle_rod_pair_p90_motion_max_mm,
            rod_score_weight_x=obstacle_rod_score_weight_x,
            rod_score_weight_length=obstacle_rod_score_weight_length,
            rod_score_weight_z=obstacle_rod_score_weight_z,
            rod_score_weight_motion=obstacle_rod_score_weight_motion,
            lr_axis="y",
            candidate_y_min_mm=obstacle_candidate_y_min_mm,
            candidate_y_max_mm=obstacle_candidate_y_max_mm,
        )
    elif _mode == "legacy":
        obstacle_indices, obstacle_labels = detect_obstacle_markers(
            points_d,
            visibility_min=obstacle_visibility_min,
            lr_axis="y",
            motion_max_mm=obstacle_max_motion_mm,
            candidate_y_min_mm=obstacle_candidate_y_min_mm,
            candidate_y_max_mm=obstacle_candidate_y_max_mm,
        )
    else:
        raise LabelingPipelineError(
            ERR_OBSTACLE_MODE,
            f"obstacle_detection_mode must be 'legacy' or 'rod_pair', got {obstacle_detection_mode!r}.",
            step="obstacle",
        )

    if require_obstacles and len(obstacle_indices) < 2:
        _cap = (
            DEFAULT_OBSTACLE_MAX_MOTION_MM
            if obstacle_max_motion_mm is None
            else float(obstacle_max_motion_mm)
        )
        _cap_note = (
            "no motion cap"
            if _cap <= 0
            else (
                f"median inter-frame displacement ≤ {_cap:g} mm/frame"
                if _mode == "rod_pair"
                else f"mean inter-frame displacement ≤ {_cap:g} mm/frame"
            )
        )
        if _mode == "rod_pair":
            if (obstacle_candidate_y_min_mm is not None) or (
                obstacle_candidate_y_max_mm is not None
            ):
                _lo = (
                    f"{obstacle_candidate_y_min_mm:g}"
                    if obstacle_candidate_y_min_mm is not None
                    else "-inf"
                )
                _hi = (
                    f"{obstacle_candidate_y_max_mm:g}"
                    if obstacle_candidate_y_max_mm is not None
                    else "inf"
                )
                _y_band = f" median Y in [{_lo}, {_hi}] mm,"
            else:
                _y_band = ""
            detail = (
                f"rod_pair: need 2 candidates (visibility floor {obstacle_visibility_floor:.0%}, "
                f"min visibility {obstacle_visibility_min:.0%},{_y_band} {_cap_note}) "
                f"and a pair with axial separation ≥ {obstacle_rod_length_min_mm:g} mm "
                f"on axis {obstacle_rod_separation_axis!r}."
            )
        else:
            if (obstacle_candidate_y_min_mm is not None) or (
                obstacle_candidate_y_max_mm is not None
            ):
                _lo = (
                    f"{obstacle_candidate_y_min_mm:g}"
                    if obstacle_candidate_y_min_mm is not None
                    else "-inf"
                )
                _hi = (
                    f"{obstacle_candidate_y_max_mm:g}"
                    if obstacle_candidate_y_max_mm is not None
                    else "inf"
                )
                _y_leg = f" median Y in [{_lo}, {_hi}] mm, "
            else:
                _y_leg = ""
            detail = (
                f"visibility >= {obstacle_visibility_min:.0%}, {_y_leg}{_cap_note}, then lowest motion."
            )
        raise ScreeningError(
            4,
            f"could not detect 2 obstacle markers (found {len(obstacle_indices)}; {detail})",
        )

    dropped_extra_stationary_loaded: list[int] = []
    # Extra stationary drop: full-trial mean motion only; runs before template/body labeling and is
    # independent of fixed_best_frame (manual or automatic best frame is chosen later).
    if drop_extra_stationary_motion_max_mm is None:
        _extra_mm = float(DEFAULT_EXTRA_STATIONARY_MOTION_MAX_MM)
    else:
        _extra_mm = float(drop_extra_stationary_motion_max_mm)
    if _extra_mm < 0:
        raise LabelingPipelineError(
            ERR_EXTRA_STATIONARY_INVALID,
            "drop_extra_stationary_motion_max_mm must be None, >= 0, or omitted (use default).",
            step="extra_stationary",
        )
    if _extra_mm > 0 and len(obstacle_indices) >= 2:
        extra_drop = screened_indices_extra_stationary_to_drop(
            points_d,
            obstacle_indices,
            motion_max_mm=_extra_mm,
            visibility_min=float(obstacle_visibility_min),
        )
        if extra_drop:
            dropped_extra_stationary_loaded = [
                int(screening_keep_for_loaded[j]) for j in extra_drop
            ]
            points_d, residual_d, screening_keep_for_loaded = _apply_drop_screened_columns(
                points_d,
                residual_d,
                extra_drop,
                screening_keep_for_loaded,
            )
            obstacle_indices = remap_indices_after_screened_drops(
                obstacle_indices, extra_drop
            )
            n_screened = int(points_d.shape[1])
            dynamic["points"] = points_d
            if residual_d is not None:
                dynamic["residual"] = residual_d
            print(
                f"Dropped {len(extra_drop)} extra stationary screened column(s); "
                f"loaded 0-based indices: {sorted(dropped_extra_stationary_loaded)}"
            )

    n_screened = int(points_d.shape[1])
    if check_screened_count and n_screened < MIN_SCREENED_COLUMNS_AFTER_EXTRA_STATIONARY:
        raise LabelingPipelineError(
            ERR_SCREENED_COLUMNS_LT_MIN,
            (
                f"after screening and extra stationary drop, need at least "
                f"{MIN_SCREENED_COLUMNS_AFTER_EXTRA_STATIONARY} marker columns, got {n_screened}."
            ),
            step="screening",
        )

    n_frames_d = points_d.shape[0]
    # Extract obstacle trajectories; default (n_frames, 2, 3) NaN if none
    if len(obstacle_indices) >= 2:
        points_obstacle = points_d[:, obstacle_indices[:2], :].copy()
    elif len(obstacle_indices) == 1:
        points_obstacle = np.full((n_frames_d, 2, 3), np.nan)
        points_obstacle[:, 0, :] = points_d[:, obstacle_indices[0], :]
    else:
        points_obstacle = np.full((n_frames_d, 2, 3), np.nan)
    # Remove obstacle points for body labeling
    if obstacle_indices:
        keep = [i for i in range(points_d.shape[1]) if i not in obstacle_indices]
        points_d_body = points_d[:, keep, :]
        residual_d_body = residual_d[:, keep] if residual_d is not None else None
    else:
        keep = list(range(points_d.shape[1]))
        points_d_body = points_d
        residual_d_body = residual_d

    n_body_cols = int(points_d_body.shape[1])
    if n_body_cols < MIN_BODY_MARKER_COLUMNS:
        raise LabelingPipelineError(
            ERR_BODY_COLUMNS_LT_MIN,
            (
                f"body marker columns after removing obstacles must be at least "
                f"{MIN_BODY_MARKER_COLUMNS}, got {n_body_cols}."
            ),
            step="screening",
        )

    # 2) Template from static (exclude obstacle labels if present)
    body_labels_static = [l for l in labels_s if l not in ("OBSTACLE_L", "OBSTACLE_R")]
    if not body_labels_static:
        body_labels_static = list(labels_s)
    template, _origin, _R = build_template_from_static(
        points_s, labels_s, use_pelvis_frame=use_pelvis_frame
    )
    # Filter template to body only
    template_body = {k: v for k, v in template.items() if k in body_labels_static}

    # Z-band by rank and segment geometry from static (39 markers only)
    template_39 = {
        k: v for k, v in template_body.items()
        if str(k).strip().upper() in WHOLE_BODY_39_SET
    }

    # Skip anatomical labels for this trial (after column drop): no matching, no output columns.
    body_labels_for_export = list(body_labels_static)
    if skip_label_names:
        skip_tokens = [s.strip() for s in skip_label_names if s.strip()]
        skip_upper = {s.upper() for s in skip_tokens}
        obstacle_u = {str(x).upper() for x in OBSTACLE_LABELS}
        if skip_upper & obstacle_u:
            raise LabelingPipelineError(
                ERR_SKIP_LABEL_OBSTACLE,
                "skip_label_names must not include OBSTACLE_L or OBSTACLE_R.",
                step="labeling",
            )
        body_by_upper = {str(lab).strip().upper(): lab for lab in body_labels_static}
        unknown = sorted(s for s in skip_upper if s not in body_by_upper)
        if unknown:
            raise LabelingPipelineError(
                ERR_SKIP_LABEL_UNKNOWN,
                f"unknown label(s) (not in static body labels): {unknown}",
                step="labeling",
            )

        def _skipped(lab: str) -> bool:
            return str(lab).strip().upper() in skip_upper

        template_body = {k: v for k, v in template_body.items() if not _skipped(str(k))}
        template_39 = {k: v for k, v in template_39.items() if not _skipped(str(k))}
        body_labels_for_export = [l for l in body_labels_static if not _skipped(str(l))]
        tpl_any = template_39 if template_39 else template_body
        if not tpl_any:
            raise LabelingPipelineError(
                ERR_SKIP_LABEL_EMPTY_TEMPLATE,
                "all body template markers were removed by skip_label_names.",
                step="labeling",
            )

    # Use no-arm-hand Z-band for dynamic labeling: save it and pass to label_body_markers
    label_to_band_no_arm_hand = (
        build_z_band_no_arm_hand_from_static(template_39) if template_39 else {}
    )
    z_band_no_arm_hand_path = f"{out_prefix}_z_band_no_arm_hand.json"
    if label_to_band_no_arm_hand:
        save_z_band_to_json(label_to_band_no_arm_hand, z_band_no_arm_hand_path)
    segment_geometry = build_segment_geometry_from_static(template_39) if template_39 else []

    # Align dynamic to static orientation when facing axes differ (e.g. static=y, dynamic=x)
    points_d_body_for_matching = points_d_body
    points_obstacle_for_matching = points_obstacle
    if static_facing_axis and dynamic_facing_axis:
        R = rotation_for_facing_axes(static_facing_axis, dynamic_facing_axis)
        if R is not None:
            points_d_body_for_matching = points_d_body @ R.T  # (n_frames, n_pts, 3)
            points_obstacle_for_matching = points_obstacle @ R.T

    obstacle_y_lo_hi: tuple[float, float] | None = None
    if len(obstacle_indices) >= 2 and int(points_obstacle_for_matching.shape[1]) >= 2:
        obstacle_y_lo_hi = trial_obstacle_y_band(
            points_obstacle_for_matching,
            aggregate=obstacle_y_aggregate,
        )

    # 3) Body labeling: head by top-4-Z + L/R/A/P at best frame; rest by no-arm-hand Z-band
    skip_for_axis_labeling = (
        [s.strip() for s in skip_label_names if s.strip()] if skip_label_names else None
    )
    labels_body_out, label_per_frame, best_frame = label_body_markers(
        points_d_body_for_matching,
        template_39 if template_39 else template_body,
        residual_dynamic=residual_d_body,
        middle_start=middle_start,
        middle_end=middle_end,
        use_hungarian=use_hungarian,
        max_match_distance=max_match_distance,
        max_propagation_distance=max_propagation_distance,
        label_to_band=label_to_band_no_arm_hand if label_to_band_no_arm_hand else None,
        band_sizes=Z_BAND_SIZES_NO_ARM_HAND if label_to_band_no_arm_hand else None,
        segment_geometry=segment_geometry if segment_geometry else None,
        walking_direction_x=wdx,
        d_back_xy=d_back,
        d_right_xy=d_right,
        fixed_best_frame=fixed_best_frame,
        column_fixed_labels=column_fixed_labels,
        obstacle_points_dynamic=points_obstacle_for_matching,
        obstacle_y_lo_hi=obstacle_y_lo_hi,
        skipped_label_names=skip_for_axis_labeling,
    )

    print(f"Best frame for labeling: {best_frame} (0-based index; 1-based frame = {best_frame + 1})")
    p_best = points_d_body_for_matching[best_frame]
    n_finite_xyz_best = int(np.sum(np.isfinite(p_best).all(axis=1)))
    n_body_cols = int(points_d_body.shape[1])
    print(
        f"At best frame: body marker columns (points to label) = {n_body_cols}; "
        f"finite XYZ at that frame = {n_finite_xyz_best}."
    )

    # 4) Build full output: body (static order) + obstacle
    if points_obstacle.shape[1] < 2:
        obs_fill = np.full((points_d.shape[0], 2, 3), np.nan)
        obs_fill[:, : points_obstacle.shape[1], :] = points_obstacle
        points_obstacle = obs_fill
    obs_labels = ["OBSTACLE_L", "OBSTACLE_R"]
    if obstacle_indices and len(obstacle_labels) == 2:
        obs_labels = obstacle_labels
    loaded_for_body = loaded_point_indices_for_body(screening_keep_for_loaded, keep)
    _unlab_base = (
        UNLABELED_NUMERIC_LABEL_BASE
        if unlabeled_numeric_base is None
        else unlabeled_numeric_base
    )
    points_full, labels_full = build_full_trajectory_matrix(
        body_labels_for_export,
        points_d_body,
        label_per_frame,
        points_obstacle,
        obs_labels,
        loaded_indices_for_body=loaded_for_body,
        reference_frame=best_frame,
        unlabeled_numeric_base=_unlab_base,
    )

    # 5) Export original + filled
    export_labeled(
        points_full,
        labels_full,
        out_prefix,
        rate=rate,
        first_frame=first_frame,
        export_filled=export_filled,
        max_interp_frames=max_interp_frames,
    )

    # QC viewer reads ``<csv_path>.bestframe`` (1-based frame) next to the labeled CSV
    try:
        Path(f"{out_prefix}_labeled.csv").with_suffix(
            Path(f"{out_prefix}_labeled.csv").suffix + ".bestframe"
        ).write_text(str(best_frame + 1))
    except OSError:
        pass

    return {
        "static_labels": labels_s,
        "body_labels_static": body_labels_static,
        "body_labels_for_export": body_labels_for_export,
        "skipped_label_names": sorted(
            {str(l).strip().upper() for l in body_labels_static}
            - {str(l).strip().upper() for l in body_labels_for_export}
        ),
        "n_screened": n_screened,
        "obstacle_indices": obstacle_indices,
        "obstacle_labels": obs_labels,
        "labels_out": labels_full,
        "n_frames": points_full.shape[0],
        "n_markers": points_full.shape[1],
        "z_band_no_arm_hand_path": z_band_no_arm_hand_path if label_to_band_no_arm_hand else None,
        "best_frame": best_frame,
        "best_frame_1based": best_frame + 1,
        "n_body_marker_columns": n_body_cols,
        "n_finite_xyz_at_best_frame": n_finite_xyz_best,
        "auto_dropped_loaded_column_indices": auto_dropped_loaded_indices,
        "dropped_extra_stationary_loaded_indices": dropped_extra_stationary_loaded,
        "obstacle_detection_mode": _mode,
        "obstacle_force_loaded_column_indices": _obstacle_forced,
    }
