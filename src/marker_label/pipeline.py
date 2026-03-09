"""Full labeling pipeline: static + dynamic -> labeled C3D and CSV."""

from __future__ import annotations

import math
import numpy as np

from .io import load_c3d
from .constants import (
    DEFAULT_DYNAMIC_VISIBILITY_MIN,
    PELVIS_MARKERS,
    DEFAULT_OBSTACLE_VISIBILITY_MIN,
    WHOLE_BODY_39,
)
from .obstacle import detect_obstacle_markers
from .body_labeling import (
    build_template_from_static,
    label_body_markers,
    compute_walking_direction_x,
    get_lr_ap_axes_from_walking,
)
from .export import build_full_trajectory_matrix, export_labeled

# Facing axis: subject's forward direction in lab (z = up). Used to align dynamic to static.
FACING_AXIS_OPTIONS = ("x", "-x", "y", "-y")


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
    use_whole_body_39: bool = False,
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
    z_band_by_value: bool = False,
    z_band_by_gap: bool = False,
    z_band_by_rank: bool = False,
    match_head_first: bool = False,
    walking_axis_x: bool = True,
    left_side_positive_lr: bool = False,
    head_flip_lr: bool = False,
    head_anterior_smaller_x: bool = False,
    head_anterior_larger_x: bool = False,
    head_swap_lfhd_rbhd: bool = False,
    head_align_to_shoulders: bool = False,
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
    use_whole_body_39 : if True, restrict to 39 whole-body markers and use anchor-based best frame + tiered caps
    z_band_by_value : if True, Z-bands by equal Z span (min–max template).
    z_band_by_gap : if True, Z-band boundaries at largest Z gaps between consecutive markers (default False).
    z_band_by_rank : if True, assign bands by Z rank in dynamic frame (highest Z → band 11, etc.); no static Z.
    match_head_first : if True, match head then RSHO/LSHO/C7 by midline, then rest by Z-band.
    walking_axis_x : if True (default), forward walking is along X-axis: L/R from Y, A/P from X.
    left_side_positive_lr : if True, left side of body = positive L/R axis (e.g. dynamic left = positive Y).
    head_flip_lr : if True, flip head L/R when assigning LFHD/RFHD/LBHD/RBHD (use when shoulders are correct but head L/R are swapped).
    head_anterior_smaller_x : if True, head anterior (front) = smaller X. Use when LFHD/RBHD are swapped because front of head has smaller X.
    head_anterior_larger_x : if True, force larger X = anterior for head. Use when subject walks +X but computed walking direction is wrong (LFHD/RBHD swapped).
    head_swap_lfhd_rbhd : if True, after head assignment swap LFHD and RBHD labels only (fixes diagonal swap).
    head_align_to_shoulders : if False (default), L/R is from walking direction only; no flip. If True, after C7+shoulders, flip head L/R so LFHD is on same side as LSHO.

    Returns
    -------
    info : dict with keys (static_labels, obstacle_indices, etc.)
    """
    static = load_c3d(static_path, scale_factor=static_scale)
    dynamic = load_c3d(dynamic_path, scale_factor=dynamic_scale)
    points_s = static["points"]
    labels_s = static["labels"]
    points_d = dynamic["points"]
    residual_d = dynamic.get("residual")
    rate = dynamic.get("rate") or 0.0
    first_frame = dynamic.get("first_frame") or 1

    # 0) Drop markers not visible more than 50% of the trial
    n_frames_d, n_pts_d, _ = points_d.shape
    visible_frac = np.sum(np.isfinite(points_d).all(axis=2), axis=0) / n_frames_d
    keep_vis = np.where(visible_frac > DEFAULT_DYNAMIC_VISIBILITY_MIN)[0]
    if len(keep_vis) < n_pts_d:
        points_d = points_d[:, keep_vis, :]
        if residual_d is not None:
            residual_d = residual_d[:, keep_vis]

    # 1) Obstacle detection on dynamic
    obstacle_indices, obstacle_labels = detect_obstacle_markers(
        points_d,
        visibility_min=obstacle_visibility_min,
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
        points_d_body = points_d
        residual_d_body = residual_d

    # 2) Template from static (exclude obstacle and markers starting with *)
    body_labels_static = [
        l for l in labels_s
        if l not in ("OBSTACLE_L", "OBSTACLE_R") and not (l.strip().startswith("*"))
    ]
    # Restrict to 39 whole-body markers for the entire labeling process (*39, *40, etc. excluded)
    whole_set = {m.upper() for m in WHOLE_BODY_39}
    body_labels_static = [l for l in body_labels_static if l.strip().upper() in whole_set]
    if use_whole_body_39:
        pass  # already filtered to WHOLE_BODY_39 above
    if not body_labels_static:
        body_labels_static = list(labels_s)
    body_labels_static = [l for l in body_labels_static if l.strip().upper() in whole_set]
    keep_static = [i for i in range(len(labels_s)) if labels_s[i] in body_labels_static]
    points_s_body = points_s[:, keep_static, :] if keep_static else points_s
    labels_s_body = [labels_s[i] for i in keep_static] if keep_static else labels_s
    template, _origin, _R = build_template_from_static(
        points_s_body, labels_s_body, use_pelvis_frame=use_pelvis_frame
    )
    template_body = {k: v for k, v in template.items() if k in body_labels_static}

    # Align dynamic to static orientation when facing axes differ (e.g. static=y, dynamic=x)
    points_d_body_for_matching = points_d_body
    R_facing = None
    if static_facing_axis and dynamic_facing_axis:
        R = rotation_for_facing_axes(static_facing_axis, dynamic_facing_axis)
        if R is not None:
            R_facing = R
            points_d_body_for_matching = points_d_body @ R.T  # (n_frames, n_pts, 3)

    # Plan 1: Set L/R and A/P from walking direction *before* any labeling (used for rest of pipeline).
    # Use full dynamic points for walking direction so the centroid X trend is stable (same as report script).
    # When facing rotation is applied, rotate d_back and d_right into the matching frame so head A/P and L/R
    # use the same coordinate system as the points (rule-based; avoids LFHD/RBHD swap from frame mismatch).
    lr_ap_from_walking = None
    if walking_axis_x:
        wdx = compute_walking_direction_x(points_d)
        d_back, d_right = get_lr_ap_axes_from_walking(
            wdx, left_side_positive_lr=left_side_positive_lr
        )
        centroid_xy = np.nanmean(
            np.nanmean(points_d_body_for_matching[:, :, :2], axis=1), axis=0
        ).astype(np.float64)
        if R_facing is not None:
            R_2d = R_facing[:2, :2].astype(np.float64)
            d_back = (R_2d @ d_back).astype(np.float64)
            d_right = (R_2d @ d_right).astype(np.float64)
        lr_ap_from_walking = (d_back, d_right, centroid_xy)

    # 3) Body labeling (use rotated points for matching; output keeps original coords)
    labels_body_out, label_per_frame = label_body_markers(
        points_d_body_for_matching,
        template_body,
        residual_dynamic=residual_d_body,
        middle_start=middle_start,
        middle_end=middle_end,
        use_hungarian=use_hungarian,
        max_match_distance=max_match_distance,
        max_propagation_distance=max_propagation_distance,
        use_whole_body_39=use_whole_body_39,
        z_band_by_value=z_band_by_value,
        z_band_by_gap=z_band_by_gap,
        z_band_by_rank=z_band_by_rank,
        match_head_first=match_head_first,
        walking_axis_x=walking_axis_x,
        left_side_positive_lr=left_side_positive_lr,
        head_flip_lr=head_flip_lr,
        head_anterior_smaller_x=head_anterior_smaller_x,
        head_anterior_larger_x=head_anterior_larger_x,
        head_swap_lfhd_rbhd=head_swap_lfhd_rbhd,
        head_align_to_shoulders=head_align_to_shoulders,
        lr_ap_from_walking=lr_ap_from_walking,
    )

    # 4) Build full output: body (static order) + obstacle
    if points_obstacle.shape[1] < 2:
        obs_fill = np.full((points_d.shape[0], 2, 3), np.nan)
        obs_fill[:, : points_obstacle.shape[1], :] = points_obstacle
        points_obstacle = obs_fill
    obs_labels = ["OBSTACLE_L", "OBSTACLE_R"]
    if obstacle_indices and len(obstacle_labels) == 2:
        obs_labels = obstacle_labels
    points_full, labels_full = build_full_trajectory_matrix(
        body_labels_static,
        points_d_body,
        label_per_frame,
        points_obstacle,
        obs_labels,
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

    return {
        "static_labels": labels_s,
        "body_labels_static": body_labels_static,
        "obstacle_indices": obstacle_indices,
        "obstacle_labels": obs_labels,
        "labels_out": labels_full,
        "n_frames": points_full.shape[0],
        "n_markers": points_full.shape[1],
    }
