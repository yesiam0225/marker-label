"""Full labeling pipeline: static + dynamic -> labeled C3D and CSV."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .io import load_c3d
from .constants import (
    PELVIS_MARKERS,
    DEFAULT_OBSTACLE_VISIBILITY_MIN,
    WHOLE_BODY_39_SET,
    Z_BAND_SIZES_NO_ARM_HAND,
    EXPECTED_SCREENED_MARKERS,
    SCREENING_Y_OUTSIDE_FRACTION_THRESHOLD,
    SCREENING_Y_MIN_FINITE_FRAMES,
    UNLABELED_NUMERIC_LABEL_BASE,
)
from .screening import (
    apply_initial_screening_steps_1_and_2,
    compute_walking_direction_x,
    get_lr_ap_axes_from_walking,
    loaded_point_indices_for_body,
    ScreeningError,
)
from .obstacle import detect_obstacle_markers
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
        raise ValueError(
            f"drop_loaded_column_indices out of range for loaded dynamic (n_points={n_points}): {invalid}",
        )
    if len(drop_set) >= n_points:
        raise ValueError("drop_loaded_column_indices would remove all marker columns.")
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


def run_pipeline(
    static_path: str,
    dynamic_path: str,
    out_prefix: str,
    *,
    obstacle_visibility_min: float = DEFAULT_OBSTACLE_VISIBILITY_MIN,
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
    y_outside_fraction_threshold: float | None = None,
    min_finite_y_frames: int | None = None,
    unlabeled_numeric_base: int | None = None,
    drop_loaded_column_indices: Sequence[int] | None = None,
) -> dict:
    """
    Run the full labeling pipeline.

    Parameters
    ----------
    static_path : path to labeled static C3D
    dynamic_path : path to unlabeled dynamic C3D
    out_prefix : output path prefix (e.g. "out/trial_01")
    obstacle_visibility_min : min visibility for obstacle candidates
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
    check_screened_count : if True (default), raise ScreeningError (Step 2) when screened
        column count is not EXPECTED_SCREENED_MARKERS (41). Set False to run on trials
        with different channel count (e.g. for head-only check).
    trim_first_frame, trim_last_frame : optional 1-based first/last frame to keep
        (Step 1.5). If both set, trim dynamic to this range before visibility screening.
    skip_visibility_screening : if True, skip Step 2 (do not drop columns by visibility).
    y_outside_fraction_threshold, min_finite_y_frames : Step 1 Y screening; None uses defaults from constants.
    unlabeled_numeric_base : added to loaded 0-based index for unlabeled marker names in export (None = constant).
    drop_loaded_column_indices : optional 0-based point column indices to remove from dynamic
        immediately after ``load_c3d`` (before screening). Per-trial fix for bad channels.

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

    # Dynamic trial initial screening (Steps 1–1.5–2): Y range, optional trim, visibility
    points_d, residual_d, screening_keep, frame_trim_start = apply_initial_screening_steps_1_and_2(
        points_d, residual_d,
        trim_first_frame=trim_first_frame,
        trim_last_frame=trim_last_frame,
        skip_visibility_step=skip_visibility_screening,
        y_outside_fraction_threshold=_y_frac,
        min_finite_y_frames=_y_min_fin,
    )
    screening_keep_for_loaded = loaded_column_aliases[np.asarray(screening_keep, dtype=np.int64)]
    if trim_first_frame is not None and trim_last_frame is not None:
        first_frame = trim_first_frame
    else:
        first_frame = (first_frame or 1) + frame_trim_start
    n_screened = points_d.shape[1]
    if check_screened_count and n_screened != EXPECTED_SCREENED_MARKERS:
        if n_screened < EXPECTED_SCREENED_MARKERS:
            msg = (
                f"expected {EXPECTED_SCREENED_MARKERS} marker columns (39 body + 2 obstacles) "
                f"after screening, got {n_screened} (too few; check Y range ±1.5 m and visibility ≥50%)."
            )
        else:
            msg = (
                f"expected {EXPECTED_SCREENED_MARKERS} marker columns (39 body + 2 obstacles) "
                f"after screening, got {n_screened} (too many; check dynamic trial channel count)."
            )
        raise ScreeningError(2, msg)
    # Step 3: walking direction and L/R–A/P axes (for head marker assignment)
    wdx = compute_walking_direction_x(points_d)
    d_back, d_right = get_lr_ap_axes_from_walking(wdx)

    # Step 4: Obstacle detection on screened dynamic; L/R by y-axis (walking along x)
    obstacle_indices, obstacle_labels = detect_obstacle_markers(
        points_d,
        visibility_min=obstacle_visibility_min,
        lr_axis="y",
    )
    if require_obstacles and len(obstacle_indices) < 2:
        raise ScreeningError(
            4,
            f"could not detect 2 obstacle markers (found {len(obstacle_indices)} with "
            f"visibility >= {obstacle_visibility_min:.0%} and lowest motion).",
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
    if static_facing_axis and dynamic_facing_axis:
        R = rotation_for_facing_axes(static_facing_axis, dynamic_facing_axis)
        if R is not None:
            points_d_body_for_matching = points_d_body @ R.T  # (n_frames, n_pts, 3)

    # 3) Body labeling: head by top-4-Z + L/R/A/P at best frame; rest by no-arm-hand Z-band
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
        body_labels_static,
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
    }
