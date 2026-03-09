"""Body marker labeling via subject-specific static template and temporal propagation."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .constants import (
    ALPHA_ANCHOR,
    ALPHA_DISTAL,
    ALPHA_MID,
    ALPHA_PROP,
    ANTERIOR_LABELS,
    ARM_HAND_LABELS,
    DEFAULT_MIDDLE_END,
    DEFAULT_MIDDLE_START,
    DEFAULT_VISIBILITY_MIN_FRACTION,
    FALLBACK_CAP_MM,
    LEFT_ARM_CHAIN_ORDER,
    RIGHT_ARM_CHAIN_ORDER,
    Z_WEIGHT_FOR_LABELING,
    HEAD_MARKER_SET,
    NECK_SHOULDER_CLAV_SET,
    NECK_SHOULDER_SET,
    MIN_CAP_ANCHOR_MM,
    MIN_CAP_DISTAL_MM,
    MIN_CAP_MID_MM,
    N_Z_BANDS,
    Z_BAND_LABEL_TO_BAND,
    Z_BAND_LABELS,
    POSTERIOR_LABELS,
    REFERENCE_39_MEAN_RMS_MM,
    SUGGESTED_MAX_MATCH_DISTANCE_39_MM,
    THORAX_LABELS_FOR_RBAK,
    TRUNK_AFTER_SHOULDERS_SET,
    TRUNK_LABELS,
    WHOLE_BODY_ANCHOR_MARKERS,
    WHOLE_BODY_ANCHOR_SET,
    WHOLE_BODY_DISTAL_LABELS,
    WHOLE_BODY_MID_LABELS,
)
from .pelvis import build_pelvis_frame, points_to_pelvis_frame

_TRUNK_SET = {t.upper() for t in TRUNK_LABELS}


def compute_walking_direction_x(points: np.ndarray) -> int:
    """
    Determine walking direction along the lab X-axis from dynamic trial.
    Uses centroid X position over time: if X increases with frame, return +1;
    if X decreases, return -1; if unclear, return 0.

    Parameters
    ----------
    points : (n_frames, n_points, 3)

    Returns
    -------
    sign : +1 (x increases), -1 (x decreases), or 0 (unclear)
    """
    n_frames, n_pts, _ = points.shape
    if n_frames < 2 or n_pts < 1:
        return 0
    valid = np.isfinite(points).all(axis=2)
    centroid_x = np.full(n_frames, np.nan, dtype=np.float64)
    for t in range(n_frames):
        if np.any(valid[t]):
            centroid_x[t] = np.nanmean(points[t, valid[t], 0])
    finite = np.isfinite(centroid_x)
    if np.sum(finite) < 2:
        return 0
    dx = np.diff(centroid_x[finite])
    mean_dx = np.nanmean(dx)
    if mean_dx > 5.0:  # mm per frame step
        return 1
    if mean_dx < -5.0:
        return -1
    return 0


def get_lr_ap_axes_from_walking(
    walking_direction_x: int,
    *,
    left_side_positive_lr: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (d_back, d_right) 2D unit vectors for L/R and A/P from walking direction.
    Forward walking along X: A = anterior = direction of travel, P = posterior = back.
    - If x increases with frame: A has larger x, P smaller x; R = smaller y, L = larger y.
    - If x decreases with frame: A has smaller x, P larger x; R = larger y, L = smaller y.
    left_side_positive_lr: if True, flip d_right (positive L/R axis = left).
    """
    if walking_direction_x > 0:
        d_back = np.array([-1.0, 0.0], dtype=np.float64)   # posterior = -X
        d_right = np.array([0.0, -1.0], dtype=np.float64)  # R = smaller Y
    elif walking_direction_x < 0:
        d_back = np.array([1.0, 0.0], dtype=np.float64)   # posterior = +X
        d_right = np.array([0.0, 1.0], dtype=np.float64)   # R = larger Y
    else:
        d_back = np.array([-1.0, 0.0], dtype=np.float64)
        d_right = np.array([0.0, -1.0], dtype=np.float64)
    if left_side_positive_lr:
        d_right = -d_right
    return d_back, d_right


def _trunk_labels_in_template(template: dict) -> list[str]:
    """Return template keys whose label (case-insensitive) is in TRUNK_LABELS."""
    return [k for k in template if str(k).strip().upper() in _TRUNK_SET]


def _rigid_transform_3d(src: np.ndarray, tgt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Procrustes: find R (3x3), t (3,) so that tgt ≈ R @ src + t. src, tgt (n, 3). Returns R, t."""
    n = src.shape[0]
    if n < 3:
        return np.eye(3), np.zeros(3)
    src_c = src - np.nanmean(src, axis=0)
    tgt_c = tgt - np.nanmean(tgt, axis=0)
    H = src_c.T @ tgt_c
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt = Vt.copy()
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = np.nanmean(tgt, axis=0) - R @ np.nanmean(src, axis=0)
    return R, t


def _apply_rigid_to_template(template: dict[str, np.ndarray], R: np.ndarray, t: np.ndarray) -> dict[str, np.ndarray]:
    """Return new template with each position transformed: pos -> R @ pos + t."""
    out = {}
    for k, pos in template.items():
        if np.isfinite(pos).all():
            out[k] = (R @ pos) + t
        else:
            out[k] = pos.copy()
    return out


def build_template_from_static(
    points_static: np.ndarray,
    labels_static: list[str],
    *,
    use_pelvis_frame: bool = True,
) -> tuple[dict[str, np.ndarray], np.ndarray | None, np.ndarray | None]:
    """
    Build body template from labeled static trial (lab or pelvis frame).

    Parameters
    ----------
    points_static : (n_frames, n_points, 3) in lab frame
    labels_static : list of str
    use_pelvis_frame : if True, template positions are in pelvis frame; else lab frame

    Returns
    -------
    template : dict label -> (3,) mean position
    origin : (3,) or None
    R : (3,3) or None
    """
    if use_pelvis_frame:
        result = build_pelvis_frame(points_static, labels_static, 0)
        if result is None:
            use_pelvis_frame = False
            origin, R = None, None
        else:
            origin, R, _ = result
            points_static = points_to_pelvis_frame(points_static, origin, R)
    else:
        origin, R = None, None
    # Mean position per label
    template = {}
    label_to_idx = {lab.strip().upper(): i for i, lab in enumerate(labels_static)}
    for label in labels_static:
        key = label.strip().upper()
        idx = label_to_idx[key]
        pos = points_static[:, idx, :]
        valid = np.isfinite(pos).all(axis=1)
        if valid.any():
            template[label] = np.nanmean(pos[valid], axis=0)
        else:
            template[label] = np.array([np.nan, np.nan, np.nan])
    return template, origin, R


def best_frame_for_matching(
    points: np.ndarray,
    residual: np.ndarray | None,
    *,
    middle_start: float = DEFAULT_MIDDLE_START,
    middle_end: float = DEFAULT_MIDDLE_END,
    visibility_min_fraction: float = DEFAULT_VISIBILITY_MIN_FRACTION,
) -> int:
    """
    Choose frame index in the middle portion with best quality (lowest mean residual or most valid).
    Only frames where at least visibility_min_fraction of markers are valid are considered.

    Parameters
    ----------
    points : (n_frames, n_points, 3)
    residual : (n_frames, n_points) or None
    middle_start, middle_end : fraction of frames for middle portion
    visibility_min_fraction : only consider frames with at least this fraction of valid points (default 0.9)

    Returns
    -------
    frame_idx : int
    """
    n_frames, n_points, _ = points.shape
    start = int(n_frames * middle_start)
    end = int(n_frames * middle_end)
    start = max(0, min(start, n_frames - 1))
    end = max(start + 1, min(end, n_frames))
    candidates = list(range(start, end))
    n_valid_per_frame = np.sum(np.isfinite(points).all(axis=2), axis=1)
    visibility_ok = n_valid_per_frame >= (visibility_min_fraction * n_points)
    candidates_visible = [f for f in candidates if visibility_ok[f]]
    if not candidates_visible:
        candidates_visible = candidates
    if residual is not None:
        def score(f):
            s = np.nanmean(residual[f, :])
            return s if np.isfinite(s) else np.inf
        best = min(candidates_visible, key=score)
        return best
    best = max(candidates_visible, key=lambda f: n_valid_per_frame[f])
    return best


def best_frame_heel_contact(
    points: np.ndarray,
    residual: np.ndarray | None,
    *,
    middle_start: float = DEFAULT_MIDDLE_START,
    middle_end: float = DEFAULT_MIDDLE_END,
    visibility_min_fraction: float = DEFAULT_VISIBILITY_MIN_FRACTION,
    walking_axis_x: bool = True,
) -> int:
    """
    Choose frame in the middle with best quality among frames that have both left and
    right heel contacts with the floor. Heel contact is heuristic: the two lowest-Z
    points must lie on opposite sides of the body midline (Y when walking_axis_x, else X).
    If no frame satisfies this, fall back to best_frame_for_matching.

    Parameters
    ----------
    points : (n_frames, n_points, 3)
    residual : (n_frames, n_points) or None
    middle_start, middle_end : fraction of frames for middle portion
    visibility_min_fraction : only consider frames with at least this fraction of valid points
    walking_axis_x : if True, midline for L/R is Y (else X)

    Returns
    -------
    frame_idx : int
    """
    n_frames, n_points, _ = points.shape
    start = int(n_frames * middle_start)
    end = int(n_frames * middle_end)
    start = max(0, min(start, n_frames - 1))
    end = max(start + 1, min(end, n_frames))
    candidates = list(range(start, end))
    n_valid_per_frame = np.sum(np.isfinite(points).all(axis=2), axis=1)
    visibility_ok = n_valid_per_frame >= (visibility_min_fraction * n_points)
    candidates_visible = [f for f in candidates if visibility_ok[f]]
    if not candidates_visible:
        candidates_visible = candidates

    coord_lr = 1 if walking_axis_x else 0  # Y for L/R when walking along X

    def has_both_heel_contacts(f: int) -> bool:
        z = points[f, :, 2]
        valid = np.isfinite(z)
        if valid.sum() < 2:
            return False
        idx = np.where(valid)[0]
        z_vals = z[idx]
        order = np.argsort(z_vals)
        # Two lowest-Z points in the frame (heels)
        lowest_indices = idx[order[:2]]
        lr_vals = points[f, lowest_indices, coord_lr]
        if not np.isfinite(lr_vals).all():
            return False
        median_lr = float(np.median(points[f, valid, coord_lr]))
        one_below = np.any(lr_vals < median_lr)
        one_above = np.any(lr_vals > median_lr)
        return one_below and one_above

    candidates_heel = [f for f in candidates_visible if has_both_heel_contacts(f)]
    if not candidates_heel:
        return best_frame_for_matching(
            points, residual,
            middle_start=middle_start, middle_end=middle_end,
            visibility_min_fraction=visibility_min_fraction,
        )

    if residual is not None:
        def score(f):
            s = np.nanmean(residual[f, :])
            return s if np.isfinite(s) else np.inf
        return min(candidates_heel, key=score)
    return max(candidates_heel, key=lambda f: n_valid_per_frame[f])


def match_markers_to_template(
    points_frame: np.ndarray,
    template: dict[str, np.ndarray],
    *,
    use_hungarian: bool = True,
    max_match_distance: float | None = None,
    per_label_max_distance: dict[str, float] | None = None,
) -> list[tuple[int, str]]:
    """
    Match unlabeled points to template (lab or pelvis frame).

    Parameters
    ----------
    points_frame : (n_points, 3) one frame
    template : dict label -> (3,) position
    use_hungarian : if True, use global optimal assignment (min total distance); else greedy
    max_match_distance : if set, reject assignment when distance > this (mm)
    per_label_max_distance : if set, override per label: cap for label L is per_label_max_distance.get(L, max_match_distance or inf)

    Returns
    -------
    list of (point_idx, label) for each point assigned
    """
    n_points = points_frame.shape[0]
    labels = list(template.keys())
    positions = np.array([template[l] for l in labels])
    valid_template = np.isfinite(positions).all(axis=1)
    if not valid_template.any():
        return []
    n_labels = len(labels)

    def cap_for_label(li: int) -> float:
        if per_label_max_distance is not None:
            lab = labels[li]
            key = lab.strip() if hasattr(lab, "strip") else lab
            if key in per_label_max_distance:
                return per_label_max_distance[key]
        return max_match_distance if max_match_distance is not None else np.inf

    if use_hungarian and n_points >= 1 and n_labels >= 1:
        try:
            # Cost matrix: (n_points, n_labels); inf for invalid
            cost = np.full((n_points, n_labels), np.inf)
            for pi in range(n_points):
                if not np.isfinite(points_frame[pi]).all():
                    continue
                for li in range(n_labels):
                    if not valid_template[li]:
                        continue
                    d = np.linalg.norm(points_frame[pi] - positions[li])
                    if d <= cap_for_label(li):
                        cost[pi, li] = d
            # Hungarian: assign labels to points (minimize total cost). Rows=points, cols=labels.
            row_ind, col_ind = linear_sum_assignment(cost)
            assignments = []
            for r, c in zip(row_ind, col_ind):
                d = cost[r, c]
                if not np.isfinite(d):
                    continue
                assignments.append((r, labels[c]))
            return assignments
        except Exception:
            pass  # fall back to greedy

    # Greedy fallback
    used_label = set()
    assignments = []
    for pi in range(n_points):
        p = points_frame[pi]
        if not np.isfinite(p).all():
            continue
        best_dist = np.inf
        best_label = None
        for li, label in enumerate(labels):
            if label in used_label or not valid_template[li]:
                continue
            d = np.linalg.norm(p - positions[li])
            cap = cap_for_label(li)
            if d < best_dist and d <= cap:
                best_dist = d
                best_label = label
        if best_label is not None:
            used_label.add(best_label)
            assignments.append((pi, best_label))
    return assignments


def _z_band_boundaries_from_template(
    template: dict[str, np.ndarray],
    n_bands: int = N_Z_BANDS,
    z_band_by_value: bool = False,
    z_band_by_gap: bool = False,
) -> tuple[np.ndarray, dict[str, int]]:
    """
    Compute Z-band boundaries and label -> band index from template (Z = third coordinate).
    Band 0 = lowest Z (feet), band n_bands-1 = highest Z (head).
    Returns (boundaries of length n_bands-1, label_to_band dict).

    Parameters
    ----------
    template : label -> (3,) position
    n_bands : number of vertical bands (default N_Z_BANDS)
    z_band_by_value : if True, bands by equal Z span (min–max divided into n_bands).
    z_band_by_gap : if True, boundaries at the (n_bands-1) largest Z gaps between consecutive
                    markers (sorted by Z). Puts band boundaries where vertical distance between
                    adjacent markers is largest (e.g. ankle–knee, trunk–head). Overrides z_band_by_value.
    """
    labels_with_z = [
        (lab, float(template[lab][2]))
        for lab in template
        if np.isfinite(template[lab]).all()
    ]
    if len(labels_with_z) < n_bands:
        n_bands = max(1, len(labels_with_z))

    if z_band_by_gap:
        # Use fixed band number and label assignment (Z differs per subject; only band–label is fixed)
        if n_bands == len(Z_BAND_LABELS):
            # Label -> band from constant (for labeling)
            label_to_band = {}
            for lab, z in labels_with_z:
                key = lab.strip().upper()
                if key in Z_BAND_LABEL_TO_BAND:
                    label_to_band[lab] = Z_BAND_LABEL_TO_BAND[key]
                else:
                    # Fallback: assign by Z vs boundaries we compute from template
                    label_to_band[lab] = -1  # temporary
            # Boundaries from subject template: between band b and b+1 use (max Z in band b + min Z in band b+1) / 2
            z_by_band = [[] for _ in range(n_bands)]
            for lab, z in labels_with_z:
                key = lab.strip().upper()
                if key in Z_BAND_LABEL_TO_BAND:
                    b = Z_BAND_LABEL_TO_BAND[key]
                    z_by_band[b].append(z)
            boundaries = np.empty(n_bands - 1, dtype=np.float64)
            for b in range(n_bands - 1):
                if z_by_band[b] and z_by_band[b + 1]:
                    boundaries[b] = (max(z_by_band[b]) + min(z_by_band[b + 1])) / 2.0
                else:
                    boundaries[b] = np.nan
            # Fill any missing boundaries (e.g. empty band) from neighbors or template Z range
            all_z = [z for _, z in labels_with_z]
            z_min, z_max = min(all_z), max(all_z)
            for b in range(n_bands - 1):
                if np.isnan(boundaries[b]):
                    boundaries[b] = z_min + (b + 1) * (z_max - z_min) / n_bands
            # Sort so boundaries are ascending (in case of fill)
            boundaries.sort()
            # Fallback for labels not in Z_BAND_LABEL_TO_BAND: assign by Z
            for lab, z in labels_with_z:
                if label_to_band.get(lab) == -1:
                    band = sum(1 for b in boundaries if z > b)
                    label_to_band[lab] = min(band, n_bands - 1)
            return boundaries, label_to_band
        # Otherwise compute boundaries at the (n_bands-1) largest gaps (template-based)
        labels_with_z.sort(key=lambda x: x[1])
        n_labels = len(labels_with_z)
        n_splits = min(n_bands - 1, n_labels - 1)
        if n_splits <= 0:
            boundaries = np.full(n_bands - 1, labels_with_z[0][1])
            label_to_band = {lab: 0 for lab, _ in labels_with_z}
            return boundaries, label_to_band
        gaps = np.array([
            labels_with_z[i + 1][1] - labels_with_z[i][1]
            for i in range(n_labels - 1)
        ], dtype=np.float64)
        gap_indices = np.argsort(gaps)[::-1][:n_splits]
        gap_indices.sort()
        boundaries = np.array([
            (labels_with_z[gap_indices[b]][1] + labels_with_z[gap_indices[b] + 1][1]) / 2.0
            for b in range(n_splits)
        ], dtype=np.float64)
        if n_splits < n_bands - 1:
            boundaries = np.resize(boundaries, n_bands - 1)
            boundaries[n_splits:] = boundaries[n_splits - 1]
        label_to_band = {}
        for idx, (lab, z) in enumerate(labels_with_z):
            band = sum(1 for gi in gap_indices if gi < idx)
            label_to_band[lab] = min(band, n_bands - 1)
        return boundaries, label_to_band

    if z_band_by_value:
        # Define bands by Z coordinate value: equal Z span from template min to max
        z_list = [z for _, z in labels_with_z]
        min_z = min(z_list)
        max_z = max(z_list)
        span = max_z - min_z
        if span < 1e-6:
            boundaries = np.full(n_bands - 1, min_z)
            label_to_band = {lab: 0 for lab, _ in labels_with_z}
            return boundaries, label_to_band
        boundaries = np.array([
            min_z + (b + 1) * span / n_bands
            for b in range(n_bands - 1)
        ], dtype=np.float64)
        label_to_band = {}
        for lab, z in labels_with_z:
            if z < boundaries[0]:
                label_to_band[lab] = 0
            elif z >= boundaries[-1]:
                label_to_band[lab] = n_bands - 1
            else:
                for b in range(1, n_bands - 1):
                    if boundaries[b - 1] <= z < boundaries[b]:
                        label_to_band[lab] = b
                        break
                else:
                    label_to_band[lab] = n_bands - 1
        return boundaries, label_to_band

    # Original: equal count per band, boundary = midpoint between groups
    labels_with_z.sort(key=lambda x: x[1])
    labels_ordered = [x[0] for x in labels_with_z]
    n_per = len(labels_ordered) // n_bands
    remainder = len(labels_ordered) % n_bands
    label_to_band = {}
    idx = 0
    for b in range(n_bands):
        size = n_per + (1 if b < remainder else 0)
        for _ in range(size):
            label_to_band[labels_ordered[idx]] = b
            idx += 1
    boundaries = np.full(n_bands - 1, np.nan)
    idx = 0
    for b in range(n_bands - 1):
        size_b = n_per + (1 if b < remainder else 0)
        max_z_b = labels_with_z[idx + size_b - 1][1]
        min_z_next = labels_with_z[idx + size_b][1]
        boundaries[b] = (max_z_b + min_z_next) / 2.0
        idx += size_b
    return boundaries, label_to_band


def _point_band_indices(
    points_frame: np.ndarray,
    boundaries: np.ndarray,
) -> np.ndarray:
    """Return (n_points,) int array: band index 0..n_bands-1, or -1 if Z invalid."""
    n_points = points_frame.shape[0]
    z = points_frame[:, 2] if points_frame.ndim >= 2 else np.full(n_points, np.nan)
    out = np.full(n_points, -1, dtype=np.int32)
    valid = np.isfinite(z)
    n_bands = boundaries.shape[0] + 1
    for b in range(n_bands):
        if b == 0:
            mask = valid & (z < boundaries[0])
        elif b == n_bands - 1:
            mask = valid & (z >= boundaries[-1])
        else:
            mask = valid & (z >= boundaries[b - 1]) & (z < boundaries[b])
        out[mask] = b
    return out


def _point_band_indices_by_z_rank(
    points_frame: np.ndarray,
    band_sizes: list[int] | None = None,
    n_bands: int = N_Z_BANDS,
) -> np.ndarray:
    """
    Assign each point to a band by Z rank in the dynamic frame (no static/template Z).

    Valid points are sorted by Z descending (highest first). The first band_sizes[11] points
    get band 11, the next band_sizes[10] get band 10, etc. At most N = sum(band_sizes) points
    are assigned; the rest get -1. If there are fewer valid points than N, lower bands may
    get fewer points (flexible per-band count).
    """
    if band_sizes is None:
        band_sizes = [len(Z_BAND_LABELS[b]) for b in range(n_bands)]
    n_points = points_frame.shape[0]
    out = np.full(n_points, -1, dtype=np.int32)
    valid = np.isfinite(points_frame).all(axis=1)
    valid_indices = np.where(valid)[0]
    if len(valid_indices) == 0:
        return out
    z = points_frame[valid_indices, 2]
    order = np.argsort(z)[::-1]  # descending Z
    sorted_indices = valid_indices[order]
    n_total_slots = sum(band_sizes)
    n_use = min(len(sorted_indices), n_total_slots)
    idx = 0
    for b in range(n_bands - 1, -1, -1):  # 11 down to 0
        size_b = band_sizes[b]
        take = min(size_b, n_use - idx)
        if take <= 0:
            break
        for j in range(take):
            out[sorted_indices[idx + j]] = b
        idx += take
        if idx >= n_use:
            break
    return out


def _get_thorax_band_indices(label_to_band: dict[str, int]) -> set[int]:
    """Band indices that contain any of THORAX_LABELS_FOR_RBAK (for RBAK/T10 identification)."""
    thorax_set = {s.upper() for s in THORAX_LABELS_FOR_RBAK}
    bands = set()
    for lab, b in label_to_band.items():
        if lab.strip().upper() in thorax_set:
            bands.add(b)
    return bands


def _fallback_lr_ap_axes(
    points_frame: np.ndarray,
    *,
    walking_axis_x: bool = True,
    left_side_positive_lr: bool = False,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """
    Fallback L/R and A/P axes when RBAK/T10 cannot be identified.
    Uses frame centroid and walking-axis convention: forward = X (or Y), right = Y (or X).
    Returns (t10_xy, centroid_xy, d_back, d_right) or (None, None, None, None).
    """
    xy = points_frame[:, :2].astype(np.float64)
    valid = np.isfinite(xy).all(axis=1)
    if valid.sum() < 3:
        return None, None, None, None
    centroid_xy = np.nanmean(xy[valid], axis=0)
    t10_xy = centroid_xy.copy()
    if walking_axis_x:
        # Forward = +X, so back = -X; L/R from Y
        d_back = np.array([-1.0, 0.0], dtype=np.float64)
        d_right = np.array([0.0, 1.0], dtype=np.float64) if not left_side_positive_lr else np.array([0.0, -1.0], dtype=np.float64)
    else:
        # Forward = +Y, so back = -Y; L/R from X
        d_back = np.array([0.0, -1.0], dtype=np.float64)
        d_right = np.array([1.0, 0.0], dtype=np.float64) if not left_side_positive_lr else np.array([-1.0, 0.0], dtype=np.float64)
    return t10_xy, centroid_xy, d_back, d_right


def _identify_rbak_t10_from_points(
    points_frame: np.ndarray,
    point_indices: np.ndarray,
) -> tuple[int | None, int | None, np.ndarray | None]:
    """
    From thorax-band points (XY), identify RBAK and T10 by geometry: PCA, posterior = far end,
    RBAK = more lateral of the two posterior points, T10 = more central.
    Returns (rbak_global_idx, t10_global_idx, centroid_xy) or (None, None, None) if failure.
    """
    if len(point_indices) < 3:
        return None, None, None
    xy = points_frame[point_indices, :2].astype(np.float64)
    valid = np.isfinite(xy).all(axis=1)
    if valid.sum() < 3:
        return None, None, None
    xy = xy[valid]
    idx_valid = point_indices[valid]
    centroid = np.nanmean(xy, axis=0)
    centered = xy - centroid
    cov = centered.T @ centered
    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
        pc1 = eigvecs[:, np.argmax(eigvals)]  # direction of max variance (A/P)
        pc2 = eigvecs[:, np.argmin(eigvals)]   # lateral
    except Exception:
        return None, None, None
    proj1 = centered @ pc1
    proj2 = centered @ pc2
    # Posterior = points at the "back" end of PC1 (choose the end with fewer points or both ends and take farthest)
    # We want the two points farthest in the negative PC1 direction (or positive - we don't know sign)
    # So take the two with smallest proj1 (most negative)
    order = np.argsort(proj1)
    posterior_pair = order[:2]
    if proj1[order[0]] == proj1[order[1]]:
        posterior_pair = order[:1]
    if len(posterior_pair) < 2:
        return None, None, None
    i0, i1 = posterior_pair[0], posterior_pair[1]
    # More lateral (larger |PC2|) = RBAK, more central = T10
    if abs(proj2[i0]) >= abs(proj2[i1]):
        rbak_local, t10_local = i0, i1
    else:
        rbak_local, t10_local = i1, i0
    rbak_global = int(idx_valid[rbak_local])
    t10_global = int(idx_valid[t10_local])
    return rbak_global, t10_global, centroid


def _classify_point_lr_ap(
    p_xy: np.ndarray,
    t10_xy: np.ndarray,
    centroid_xy: np.ndarray,
    d_back: np.ndarray,
    d_right: np.ndarray,
    threshold_mm: float = 50.0,
) -> tuple[str, str]:
    """Return (lr, ap): 'L'|'R'|'mid', 'ant'|'post'|'mid'."""
    vec_right = p_xy - t10_xy
    dot_right = float(np.dot(vec_right, d_right))
    dot_back = float(np.dot(p_xy - centroid_xy, d_back))
    lr = "mid" if abs(dot_right) < threshold_mm else ("R" if dot_right > 0 else "L")
    ap = "mid" if abs(dot_back) < threshold_mm else ("post" if dot_back > 0 else "ant")
    return lr, ap


def _classify_label_lr_ap(label: str) -> tuple[str, str]:
    """Return (lr, ap): 'L'|'R'|'mid', 'ant'|'post'|'mid' from label name and anatomy."""
    u = label.strip().upper()
    lr = "mid"
    if u.startswith("LB"):
        lr = "L"
    elif u.startswith("RB"):
        lr = "R"
    elif u.startswith("L"):
        lr = "L"
    elif u.startswith("R"):
        lr = "R"
    ap = "mid"
    if u in {s.upper() for s in POSTERIOR_LABELS}:
        ap = "post"
    elif u in {s.upper() for s in ANTERIOR_LABELS}:
        ap = "ant"
    return lr, ap


def _classify_point_lr_ap_head(
    p_xy: np.ndarray,
    median_x: float,
    median_y: float,
    threshold_mm: float = 50.0,
    *,
    walking_axis_x: bool = True,
    left_side_positive_lr: bool = False,
) -> tuple[str, str]:
    """
    Classify a head-band point as L/R and A/P (forward walking).

    When walking is along the X-axis (walking_axis_x=True, default):
    - Left/Right from Y: point y > median_y -> R, y < median_y -> L by default; use
      left_side_positive_lr=True (or CLI --left-side-positive-y) if head L/R are flipped.
    - Anterior/Posterior from X: point x > median_x -> ant (front), x < median_x -> post (back).

    left_side_positive_lr: when True, positive Y = left (fixes flipped head L/R if your lab has left = +Y).
    Returns (lr, ap): 'L'|'R'|'mid', 'ant'|'post'|'mid'.
    """
    dx = float(p_xy[0] - median_x)
    dy = float(p_xy[1] - median_y)
    if walking_axis_x:
        if abs(dy) < threshold_mm:
            lr = "mid"
        else:
            lr = "L" if (dy > 0) == left_side_positive_lr else "R"
        ap = "mid" if abs(dx) < threshold_mm else ("ant" if dx > 0 else "post")
    else:
        if abs(dx) < threshold_mm:
            lr = "mid"
        else:
            lr = "L" if (dx > 0) == left_side_positive_lr else "R"
        ap = "mid" if abs(dy) < threshold_mm else ("ant" if dy > 0 else "post")
    return lr, ap


def _match_head_markers_first(
    points_frame: np.ndarray,
    template: dict[str, np.ndarray],
    point_bands: np.ndarray,
    label_to_band: dict[str, int],
    *,
    use_hungarian: bool = True,
    max_match_distance: float | None = None,
    head_lr_ap_threshold_mm: float = 50.0,
    walking_axis_x: bool = True,
    left_side_positive_lr: bool = False,
    head_flip_lr: bool = False,
    head_anterior_smaller_x: bool = False,
    head_anterior_larger_x: bool = False,
    head_swap_lfhd_rbhd: bool = False,
    use_top4_z: bool = True,
    t10_xy: np.ndarray | None = None,
    centroid_xy: np.ndarray | None = None,
    d_back: np.ndarray | None = None,
    d_right: np.ndarray | None = None,
) -> list[tuple[int, str]]:
    """
    Match head markers: top 4 Z-value points are candidates. Among them, assign
    LFHD, RFHD, LBHD, RBHD by Fore/Back (A/P) and Left/Right (L/R).
    Fore/Back: when d_back and centroid_xy are provided, use projection along d_back (same frame as points;
    robust with facing rotation). Else use X with d_back[0] or head_anterior_* or default.
    Left/Right: left_side_positive_lr True => larger Y = Left, else smaller Y = Left.
    """
    head_labels_in_template = [
        lab for lab in template
        if str(lab).strip().upper() in HEAD_MARKER_SET and np.isfinite(template[lab]).all()
    ]
    if not head_labels_in_template:
        return []
    n_head = len(head_labels_in_template)
    head_key = {str(lab).strip().upper(): lab for lab in head_labels_in_template}

    if use_top4_z:
        z = points_frame[:, 2]
        valid = np.isfinite(z)
        if valid.sum() < 4:
            return []
        idx_all = np.where(valid)[0]
        z_vals = z[idx_all]
        order_z = np.argsort(z_vals)[::-1]
        n_take = min(4, n_head, len(order_z))
        point_indices = list(idx_all[order_z[:n_take]])
    else:
        head_bands = {label_to_band.get(lab, -1) for lab in head_labels_in_template}
        head_bands.discard(-1)
        if not head_bands:
            return []
        point_indices = list(np.unique(np.concatenate([np.where(point_bands == b)[0] for b in head_bands])))
        if len(point_indices) < 4:
            return []
        # Take top 4 by Z from this band
        z_band = points_frame[point_indices, 2]
        order_z = np.argsort(z_band)[::-1]
        point_indices = [point_indices[i] for i in order_z[:4]]

    if len(point_indices) < 4:
        return []

    # pts_xy: (4, 2) with columns X, Y
    pts_xy = points_frame[point_indices, :2].astype(np.float64)
    if not np.isfinite(pts_xy).all():
        return []

    # Fore/Back (A/P): rule-based. When d_back and centroid_xy are available, use projection along d_back
    # so the same frame is used as elsewhere (correct with or without facing rotation). Otherwise use X.
    order_by_x = np.argsort(pts_xy[:, 0])[::-1]  # descending: first = largest X (fallback)
    if d_back is not None and len(d_back) >= 1 and centroid_xy is not None and np.isfinite(centroid_xy).all():
        # Project each point onto d_back (relative to centroid). Larger dot = more posterior.
        cen = np.asarray(centroid_xy, dtype=np.float64).reshape(2)
        d_back_2d = np.asarray(d_back[:2], dtype=np.float64)
        dot_back = (pts_xy - cen) @ d_back_2d
        order_by_back = np.argsort(dot_back)  # ascending: first = most anterior (smallest dot_back)
        ant_idx = list(order_by_back[:2])
        post_idx = list(order_by_back[2:])
    elif d_back is not None and len(d_back) >= 1:
        # d_back but no centroid: keep X-based rule for backward compatibility
        if d_back[0] > 0:
            ant_idx = list(order_by_x[2:])
            post_idx = list(order_by_x[:2])
        else:
            ant_idx = list(order_by_x[:2])
            post_idx = list(order_by_x[2:])
    elif head_anterior_smaller_x:
        ant_idx = list(order_by_x[2:])
        post_idx = list(order_by_x[:2])
    elif head_anterior_larger_x:
        ant_idx = list(order_by_x[:2])
        post_idx = list(order_by_x[2:])
    else:
        ant_idx = list(order_by_x[:2])  # default: larger X = anterior
        post_idx = list(order_by_x[2:])

    # Left/Right by Y: left_side_positive_lr => larger Y = Left
    def assign_lr(two_local_indices: list[int]) -> tuple[int, int]:
        i0, i1 = two_local_indices[0], two_local_indices[1]
        y0, y1 = pts_xy[i0, 1], pts_xy[i1, 1]
        if left_side_positive_lr:
            left_i = i0 if y0 >= y1 else i1
            right_i = i1 if left_i == i0 else i0
        else:
            left_i = i0 if y0 <= y1 else i1
            right_i = i1 if left_i == i0 else i0
        return left_i, right_i

    lfhd_i, rfhd_i = assign_lr(ant_idx)
    lbhd_i, rbhd_i = assign_lr(post_idx)

    assignments: list[tuple[int, str]] = []
    if "LFHD" in head_key:
        assignments.append((int(point_indices[lfhd_i]), head_key["LFHD"]))
    if "RFHD" in head_key:
        assignments.append((int(point_indices[rfhd_i]), head_key["RFHD"]))
    if "LBHD" in head_key:
        assignments.append((int(point_indices[lbhd_i]), head_key["LBHD"]))
    if "RBHD" in head_key:
        assignments.append((int(point_indices[rbhd_i]), head_key["RBHD"]))

    if head_swap_lfhd_rbhd:
        lfhd_tup = next(((pi, lab) for pi, lab in assignments if str(lab).strip().upper() == "LFHD"), None)
        rbhd_tup = next(((pi, lab) for pi, lab in assignments if str(lab).strip().upper() == "RBHD"), None)
        if lfhd_tup and rbhd_tup and "LFHD" in head_key and "RBHD" in head_key:
            for i, (pi, lab) in enumerate(assignments):
                u = str(lab).strip().upper()
                if u == "LFHD":
                    assignments[i] = (pi, head_key["RBHD"])
                elif u == "RBHD":
                    assignments[i] = (pi, head_key["LFHD"])

    if max_match_distance is not None and assignments:
        kept = []
        for pi, lab in assignments:
            if lab in template and np.isfinite(template[lab]).all():
                if np.linalg.norm(points_frame[pi] - template[lab]) <= max_match_distance:
                    kept.append((pi, lab))
            else:
                kept.append((pi, lab))
        if len(kept) < 4:
            return kept
        assignments = kept
    return assignments


def _match_c7_shoulders_only(
    points_frame: np.ndarray,
    template: dict[str, np.ndarray],
    point_bands: np.ndarray,
    label_to_band: dict[str, int],
    assigned_pts: set[int],
    assigned_lab: set[str],
    *,
    centroid_xy: np.ndarray,
    t10_xy: np.ndarray,
    d_right: np.ndarray,
    left_side_positive_lr: bool = False,
    max_match_distance: float | None = None,
) -> list[tuple[int, str]]:
    """
    Match only C7, LSHO, RSHO. C7 has the next highest Z after head markers and sits near midline.
    Pool = unassigned points with the next highest Z (after head). Among these 3, C7 = the one
    in the middle in L/R (median position along d_right), so we do not rely on full-body centroid
    which can be offset at neck height; LSHO/RSHO = the other two by d_right (left/right).
    """
    neck_labels = [
        lab for lab in template
        if str(lab).strip().upper() in NECK_SHOULDER_SET
        and np.isfinite(template[lab]).all()
        and lab not in assigned_lab
    ]
    if len(neck_labels) != 3:
        return []
    lab_c7 = next(lab for lab in neck_labels if str(lab).strip().upper() == "C7")
    lab_lsho = next(lab for lab in neck_labels if str(lab).strip().upper() == "LSHO")
    lab_rsho = next(lab for lab in neck_labels if str(lab).strip().upper() == "RSHO")
    # Pool = points not yet assigned, with valid Z; take next highest Z after head (top 3 by Z descending)
    n_pts = points_frame.shape[0]
    candidates = [
        i for i in range(n_pts)
        if i not in assigned_pts and np.isfinite(points_frame[i]).all()
    ]
    if len(candidates) < 3:
        return []
    z_vals = points_frame[candidates, 2]
    order = np.argsort(z_vals)[::-1]
    top3_idx = [candidates[order[j]] for j in range(min(3, len(candidates)))]
    valid_pts = top3_idx
    # C7 = among these 3, the one in the middle in L/R (median along d_right). Anatomically C7 is
    # medial to both shoulders; full-body centroid_xy can lie closer to one shoulder at neck height.
    dot_rights_all = [
        float(np.dot(points_frame[valid_pts[j], :2] - t10_xy, d_right))
        for j in range(len(valid_pts))
    ]
    # Index (0,1,2) with median dot_right = C7; the other two = LSHO, RSHO by L/R order
    order_by_dot = sorted(range(3), key=lambda j: dot_rights_all[j])
    c7_local = order_by_dot[1]  # median = middle index
    point_c7 = valid_pts[c7_local]
    left_idx = order_by_dot[0]   # smallest dot_right
    right_idx = order_by_dot[2]  # largest dot_right
    if left_side_positive_lr:
        lsho_local = right_idx   # positive d_right = left
        rsho_local = left_idx
    else:
        lsho_local = left_idx    # smaller dot = left
        rsho_local = right_idx
    point_lsho = valid_pts[lsho_local]
    point_rsho = valid_pts[rsho_local]
    out = [
        (int(point_c7), lab_c7),
        (int(point_lsho), lab_lsho),
        (int(point_rsho), lab_rsho),
    ]
    if max_match_distance is not None:
        for pi, lab in out:
            d = np.linalg.norm(points_frame[pi] - template[lab])
            if d > max_match_distance:
                return []
    return out


def _match_trunk_markers_after_shoulders(
    points_frame: np.ndarray,
    template: dict[str, np.ndarray],
    point_bands: np.ndarray,
    label_to_band: dict[str, int],
    assigned_pts: set[int],
    assigned_lab: set[str],
    midline_xy: np.ndarray,
    d_back: np.ndarray,
    d_right: np.ndarray,
    *,
    max_match_distance: float | None = None,
    left_side_positive_lr: bool = False,
) -> list[tuple[int, str]]:
    """
    Match trunk markers CLAV, RBAK, STRN, T10 in Z-band order (band 9 → 8 → 7).
    Midline is the C7 position (midline_xy). A/P from d_back; L/R from d_right.
    Band 9: CLAV (min Z in band). Band 8: RBAK = most posterior in band 8 only (no fallback to band 7,
    since that would steal T10). Band 7: When RBAK was not assigned from band 8, assign RBAK+T10+STRN
    together: two most posterior points → T10 = more central (smaller |d_right|), RBAK = more right;
    most anterior → STRN. When RBAK already assigned: T10 = most posterior, STRN = most anterior.
    """
    assignments: list[tuple[int, str]] = []
    trunk_in_tpl = [
        lab for lab in template
        if str(lab).strip().upper() in TRUNK_AFTER_SHOULDERS_SET
        and np.isfinite(template[lab]).all()
        and lab not in assigned_lab
    ]
    if not trunk_in_tpl:
        return []
    used_pts = set(assigned_pts)
    for band_b in (9, 8, 7):  # Z-band order high to low
        labels_in_b = [lab for lab in trunk_in_tpl if label_to_band.get(lab, -1) == band_b]
        if not labels_in_b:
            continue
        pt_idx = np.where(point_bands == band_b)[0]
        pt_idx = np.array([i for i in pt_idx if i not in used_pts], dtype=int)
        pt_idx = [i for i in pt_idx if np.isfinite(points_frame[i]).all()]
        if band_b == 8:
            # RBAK: most posterior in band 8 only. Do not use band 7 fallback: the most posterior in band 7 is T10.
            rbak_lab = next((lab for lab in labels_in_b if str(lab).strip().upper() == "RBAK"), None)
            if rbak_lab and len(pt_idx) >= 1:
                pts_rbak = points_frame[pt_idx]
                dots = [float(np.dot(pts_rbak[j, :2] - midline_xy, d_back)) for j in range(len(pt_idx))]
                best = int(np.argmax(dots))
                pi = int(pt_idx[best])
                if max_match_distance is None or np.linalg.norm(points_frame[pi] - template[rbak_lab]) <= max_match_distance:
                    assignments.append((pi, rbak_lab))
                    used_pts.add(pi)
            continue
        if not pt_idx:
            continue
        pts_b = points_frame[pt_idx]
        if band_b == 9:
            clav_lab = next((lab for lab in labels_in_b if str(lab).strip().upper() == "CLAV"), None)
            if clav_lab and len(pt_idx) >= 1:
                best = min(range(len(pt_idx)), key=lambda j: float(pts_b[j, 2]))
                pi = int(pt_idx[best])
                if max_match_distance is None or np.linalg.norm(points_frame[pi] - template[clav_lab]) <= max_match_distance:
                    assignments.append((pi, clav_lab))
                    used_pts.add(pi)
        elif band_b == 7:
            t10_lab = next((lab for lab in labels_in_b if str(lab).strip().upper() == "T10"), None)
            strn_lab = next((lab for lab in labels_in_b if str(lab).strip().upper() == "STRN"), None)
            rbak_lab = next((lab for lab in trunk_in_tpl if str(lab).strip().upper() == "RBAK"), None)
            rbak_already = any(str(lab).strip().upper() == "RBAK" for _, lab in assignments)
            # When RBAK not yet assigned, pool band 7 + band 8 so we have enough candidates (RBAK may have fallen into band 7 or 8)
            if not rbak_already and rbak_lab:
                band_8_idx = np.where(point_bands == 8)[0]
                band_8_idx = [i for i in band_8_idx if i not in used_pts and np.isfinite(points_frame[i]).all()]
                pt_idx_7 = list(pt_idx)
                pt_idx = list(set(pt_idx_7) | set(band_8_idx))
                pt_idx = np.array([i for i in pt_idx if i not in used_pts and np.isfinite(points_frame[i]).all()], dtype=int)
            if len(pt_idx) == 0:
                continue
            pts_b = points_frame[pt_idx]
            dots = [float(np.dot(pts_b[j, :2] - midline_xy, d_back)) for j in range(len(pt_idx))]
            rightness = [
                float(np.dot(pts_b[j, :2] - midline_xy, -d_right if left_side_positive_lr else d_right))
                for j in range(len(pt_idx))
            ]
            order_post = sorted(range(len(pt_idx)), key=lambda j: dots[j], reverse=True)
            if not rbak_already and rbak_lab and t10_lab and strn_lab and len(pt_idx) >= 3:
                # Assign RBAK, T10, STRN from band 7: two most posterior → T10 (central), RBAK (right); most anterior → STRN
                idx_a, idx_b = order_post[0], order_post[1]
                if rightness[idx_a] >= rightness[idx_b]:
                    pi_t10, pi_rbak = int(pt_idx[idx_b]), int(pt_idx[idx_a])
                else:
                    pi_t10, pi_rbak = int(pt_idx[idx_a]), int(pt_idx[idx_b])
                pi_strn = int(pt_idx[order_post[-1]])
                if max_match_distance is None or (
                    np.linalg.norm(points_frame[pi_t10] - template[t10_lab]) <= max_match_distance
                    and np.linalg.norm(points_frame[pi_rbak] - template[rbak_lab]) <= max_match_distance
                    and np.linalg.norm(points_frame[pi_strn] - template[strn_lab]) <= max_match_distance
                ):
                    assignments.append((pi_t10, t10_lab))
                    assignments.append((pi_rbak, rbak_lab))
                    assignments.append((pi_strn, strn_lab))
                    used_pts.add(pi_t10)
                    used_pts.add(pi_rbak)
                    used_pts.add(pi_strn)
            elif len(pt_idx) >= 2 and t10_lab and strn_lab:
                pi_t10 = int(pt_idx[order_post[0]])
                pi_strn = int(pt_idx[order_post[-1]])
                if max_match_distance is None or (
                    np.linalg.norm(points_frame[pi_t10] - template[t10_lab]) <= max_match_distance
                    and np.linalg.norm(points_frame[pi_strn] - template[strn_lab]) <= max_match_distance
                ):
                    assignments.append((pi_t10, t10_lab))
                    assignments.append((pi_strn, strn_lab))
                    used_pts.add(pi_t10)
                    used_pts.add(pi_strn)
                # If RBAK still not assigned (e.g. band 8 empty, band 7 had only 2 points), try band 6: most posterior on the right
                if not rbak_already and rbak_lab:
                    band_6_idx = np.where(point_bands == 6)[0]
                    band_6_idx = [i for i in band_6_idx if i not in used_pts and np.isfinite(points_frame[i]).all()]
                    if band_6_idx:
                        pts_6 = points_frame[band_6_idx]
                        dots_6 = [float(np.dot(pts_6[j, :2] - midline_xy, d_back)) for j in range(len(band_6_idx))]
                        right_6 = [float(np.dot(pts_6[j, :2] - midline_xy, -d_right if left_side_positive_lr else d_right)) for j in range(len(band_6_idx))]
                        # Prefer right-side posterior; if none on right, take most posterior
                        best_r = max(range(len(band_6_idx)), key=lambda j: (right_6[j] if right_6[j] > 0 else -1e9, dots_6[j]))
                        pi_rbak = int(band_6_idx[best_r])
                        if max_match_distance is None or np.linalg.norm(points_frame[pi_rbak] - template[rbak_lab]) <= max_match_distance:
                            assignments.append((pi_rbak, rbak_lab))
                            used_pts.add(pi_rbak)
            elif len(pt_idx) >= 1 and t10_lab and not strn_lab:
                dots = [float(np.dot(pts_b[j, :2] - midline_xy, d_back)) for j in range(len(pt_idx))]
                best = max(range(len(pt_idx)), key=lambda j: dots[j])
                pi = int(pt_idx[best])
                if max_match_distance is None or np.linalg.norm(points_frame[pi] - template[t10_lab]) <= max_match_distance:
                    assignments.append((pi, t10_lab))
                    used_pts.add(pi)
            elif len(pt_idx) >= 1 and strn_lab and not t10_lab:
                dots = [float(np.dot(pts_b[j, :2] - midline_xy, d_back)) for j in range(len(pt_idx))]
                best = min(range(len(pt_idx)), key=lambda j: dots[j])
                pi = int(pt_idx[best])
                if max_match_distance is None or np.linalg.norm(points_frame[pi] - template[strn_lab]) <= max_match_distance:
                    assignments.append((pi, strn_lab))
                    used_pts.add(pi)
    return assignments


def _match_trunk_from_anatomy_pool(
    points_frame: np.ndarray,
    template: dict[str, np.ndarray],
    assigned_pts: set[int],
    assigned_lab: set[str],
    c7_xy: np.ndarray,
    c7_z: float,
    d_back: np.ndarray,
    d_right: np.ndarray,
    *,
    max_match_distance: float | None = None,
    left_side_positive_lr: bool = False,
    z_margin_below: float = 350.0,
    z_margin_above: float = 50.0,
    lsho_xy: np.ndarray | None = None,
    rsho_xy: np.ndarray | None = None,
    max_lateral_ratio: float = 0.5,
    max_lateral_mm: float = 95.0,
) -> list[tuple[int, str]]:
    """
    Assign CLAV, RBAK, T10, STRN from an anatomy-based Z pool (not Z-rank bands).
    Used when z_band_by_rank so trunk is not forced by strict band order.
    Pool = unassigned points with Z in [c7_z - z_margin_below, c7_z + z_margin_above],
    restricted to points near midline: |lateral| <= min(max_lateral_ratio * shoulder half-span, max_lateral_mm)
    to exclude arm markers (LUPA, RUPA).
    CLAV = most anterior among high-Z points that are also central (small |lateral|), so RBAK (right) is excluded.
    Then RBAK = most posterior + right, T10 = most posterior + central, STRN = most anterior.
    """
    trunk_labs = [
        lab for lab in template
        if str(lab).strip().upper() in TRUNK_AFTER_SHOULDERS_SET
        and np.isfinite(template[lab]).all()
        and lab not in assigned_lab
    ]
    if not trunk_labs:
        return []
    z_lo = c7_z - z_margin_below
    z_hi = c7_z + z_margin_above
    pool = [
        i for i in range(points_frame.shape[0])
        if i not in assigned_pts
        and np.isfinite(points_frame[i]).all()
        and z_lo <= points_frame[i, 2] <= z_hi
    ]
    midline_xy = c7_xy.astype(np.float64)
    # Restrict pool to points near midline (exclude LUPA/RUPA): use tighter limit
    if lsho_xy is not None and rsho_xy is not None and len(pool) >= 4:
        half_span = min(
            float(np.linalg.norm(lsho_xy[:2].astype(np.float64) - midline_xy)),
            float(np.linalg.norm(rsho_xy[:2].astype(np.float64) - midline_xy)),
        )
        if half_span > 1.0:
            lateral_limit = min(max_lateral_ratio * half_span, max_lateral_mm)
            pool = [
                i for i in pool
                if abs(float(np.dot(points_frame[i, :2] - midline_xy, d_right))) <= lateral_limit
            ]
    if len(pool) < 4:
        return []
    pts = points_frame[pool]
    dots_back = [float(np.dot(pts[j, :2] - midline_xy, d_back)) for j in range(len(pool))]
    rightness = [
        float(np.dot(pts[j, :2] - midline_xy, -d_right if left_side_positive_lr else d_right))
        for j in range(len(pool))
    ]
    assignments: list[tuple[int, str]] = []
    used = set()

    clav_lab = next((l for l in trunk_labs if str(l).strip().upper() == "CLAV"), None)
    if clav_lab:
        # CLAV = most anterior among high-Z and central (small |rightness|) so we don't pick RBAK (right-side)
        order_z = sorted(range(len(pool)), key=lambda j: float(pts[j, 2]), reverse=True)
        n_high = max(1, len(pool) // 2)
        high_z_indices = order_z[:n_high]
        abs_right = [abs(rightness[j]) for j in range(len(pool))]
        # Prefer anterior (min dots_back) then more central (min abs_right); RBAK has large rightness
        best_ant = min(high_z_indices, key=lambda j: (dots_back[j], abs_right[j]))
        pi = int(pool[best_ant])
        if max_match_distance is None or np.linalg.norm(points_frame[pi] - template[clav_lab]) <= max_match_distance:
            assignments.append((pi, clav_lab))
            used.add(best_ant)

    rbak_lab = next((l for l in trunk_labs if str(l).strip().upper() == "RBAK"), None)
    t10_lab = next((l for l in trunk_labs if str(l).strip().upper() == "T10"), None)
    strn_lab = next((l for l in trunk_labs if str(l).strip().upper() == "STRN"), None)
    remaining = [j for j in range(len(pool)) if j not in used]
    if len(remaining) < 3 or not (rbak_lab and t10_lab and strn_lab):
        return assignments
    order_post = sorted(remaining, key=lambda j: dots_back[j], reverse=True)
    # Two most posterior → T10 (central), RBAK (right); most anterior → STRN
    idx_a, idx_b = order_post[0], order_post[1]
    if rightness[idx_a] >= rightness[idx_b]:
        pi_t10, pi_rbak = int(pool[idx_b]), int(pool[idx_a])
    else:
        pi_t10, pi_rbak = int(pool[idx_a]), int(pool[idx_b])
    pi_strn = int(pool[order_post[-1]])
    if max_match_distance is None or (
        np.linalg.norm(points_frame[pi_t10] - template[t10_lab]) <= max_match_distance
        and np.linalg.norm(points_frame[pi_rbak] - template[rbak_lab]) <= max_match_distance
        and np.linalg.norm(points_frame[pi_strn] - template[strn_lab]) <= max_match_distance
    ):
        assignments.append((pi_t10, t10_lab))
        assignments.append((pi_rbak, rbak_lab))
        assignments.append((pi_strn, strn_lab))
    return assignments


def _match_shoulder_neck_from_midline(
    points_frame: np.ndarray,
    template: dict[str, np.ndarray],
    point_bands: np.ndarray,
    label_to_band: dict[str, int],
    midline_x: float,
    midline_y: float,
    assigned_pts: set[int],
    assigned_lab: set[str],
    *,
    max_match_distance: float | None = None,
    walking_axis_x: bool = True,
    left_side_positive_lr: bool = False,
    t10_xy: np.ndarray | None = None,
    centroid_xy: np.ndarray | None = None,
    d_back: np.ndarray | None = None,
    d_right: np.ndarray | None = None,
) -> list[tuple[int, str]]:
    """
    Match CLAV, C7, LSHO, RSHO. CLAV = smallest Z. When t10_xy/centroid_xy/d_right are
    provided, use same thorax axes: C7 = closest to centroid, LSHO/RSHO by d_right (L/R).
    Otherwise use midline_x/y and coord_lr (Y or X).
    """
    neck_labels_in_template = [
        lab for lab in template
        if str(lab).strip().upper() in NECK_SHOULDER_SET
        and np.isfinite(template[lab]).all()
        and lab not in assigned_lab
    ]
    if len(neck_labels_in_template) != 3:
        return []
    bands_with_neck = {
        label_to_band.get(lab, -1) for lab in neck_labels_in_template
    }
    bands_with_neck.discard(-1)
    if len(bands_with_neck) != 1:
        return []
    band_shoulder = next(iter(bands_with_neck))
    lab_clav = None
    band_clav = -1
    for lab in template:
        if str(lab).strip().upper() == "CLAV" and np.isfinite(template[lab]).all() and lab not in assigned_lab:
            lab_clav = lab
            band_clav = label_to_band.get(lab, -1)
            break
    pool_band_indices = {band_shoulder}
    if lab_clav is not None and band_clav >= 0:
        pool_band_indices.add(band_clav)
    point_indices = np.concatenate([
        np.where(point_bands == b)[0] for b in pool_band_indices
    ])
    point_indices = np.unique(np.array([i for i in point_indices if i not in assigned_pts], dtype=int))
    valid_pts = [
        i for i in point_indices
        if np.isfinite(points_frame[i]).all()
    ]
    use_thorax_axes = (
        t10_xy is not None and centroid_xy is not None and d_right is not None
    )
    coord_lr = 1 if walking_axis_x else 0
    midline_lr = midline_y if walking_axis_x else midline_x

    lab_c7 = next(lab for lab in neck_labels_in_template if str(lab).strip().upper() == "C7")
    lab_lsho = next(lab for lab in neck_labels_in_template if str(lab).strip().upper() == "LSHO")
    lab_rsho = next(lab for lab in neck_labels_in_template if str(lab).strip().upper() == "RSHO")

    if lab_clav is not None and len(valid_pts) >= 4:
        # 4-way: CLAV first (smallest Z), then C7, LSHO, RSHO
        used = set()
        clav_local = min(
            range(len(valid_pts)),
            key=lambda j: float(points_frame[valid_pts[j], 2]),
        )
        point_clav = valid_pts[clav_local]
        used.add(clav_local)
        remaining_idx = [j for j in range(len(valid_pts)) if j not in used]
        if len(remaining_idx) < 3:
            return []
        if use_thorax_axes:
            # C7 = closest to centroid (same midline as all bands)
            best = min(
                remaining_idx,
                key=lambda j: float(np.linalg.norm(points_frame[valid_pts[j], :2] - centroid_xy)),
            )
            point_c7 = valid_pts[best]
            used.add(best)
            # LSHO/RSHO by d_right (same L/R as all bands)
            remaining_idx = [j for j in range(len(valid_pts)) if j not in used]
            if len(remaining_idx) < 2:
                return []
            dot_rights = [
                float(np.dot(points_frame[valid_pts[j], :2] - t10_xy, d_right))
                for j in remaining_idx
            ]
            if left_side_positive_lr:
                lsho_local = 0 if dot_rights[0] >= dot_rights[1] else 1
            else:
                lsho_local = 0 if dot_rights[0] <= dot_rights[1] else 1
            rsho_local = 1 - lsho_local
            point_lsho = valid_pts[remaining_idx[lsho_local]]
            point_rsho = valid_pts[remaining_idx[rsho_local]]
        else:
            best = min(
                remaining_idx,
                key=lambda j: abs(float(points_frame[valid_pts[j], coord_lr]) - midline_lr),
            )
            point_c7 = valid_pts[best]
            used.add(best)
            remaining_idx = [j for j in range(len(valid_pts)) if j not in used]
            if len(remaining_idx) < 2:
                return []
            lr_vals = [float(points_frame[valid_pts[j], coord_lr]) for j in remaining_idx]
            lsho_local = max(range(len(remaining_idx)), key=lambda k: lr_vals[k])
            lsho_idx = remaining_idx[lsho_local]
            point_lsho = valid_pts[lsho_idx]
            used.add(lsho_idx)
            remaining_idx = [j for j in range(len(valid_pts)) if j not in used]
            if len(remaining_idx) < 1:
                return []
            point_rsho = valid_pts[remaining_idx[0]]
        assignments = [
            (int(point_clav), lab_clav),
            (int(point_c7), lab_c7),
            (int(point_lsho), lab_lsho),
            (int(point_rsho), lab_rsho),
        ]
    else:
        # 3-way: no CLAV or not enough points
        if len(valid_pts) < 3:
            return []
        if use_thorax_axes:
            central_idx = min(
                range(len(valid_pts)),
                key=lambda j: float(np.linalg.norm(points_frame[valid_pts[j], :2] - centroid_xy)),
            )
            point_c7 = valid_pts[central_idx]
            remaining = [j for j in range(len(valid_pts)) if j != central_idx]
            if len(remaining) != 2:
                return []
            dot_rights = [
                float(np.dot(points_frame[valid_pts[j], :2] - t10_xy, d_right))
                for j in remaining
            ]
            if left_side_positive_lr:
                lsho_local = 0 if dot_rights[0] >= dot_rights[1] else 1
            else:
                lsho_local = 0 if dot_rights[0] <= dot_rights[1] else 1
            rsho_local = 1 - lsho_local
            point_lsho = valid_pts[remaining[lsho_local]]
            point_rsho = valid_pts[remaining[rsho_local]]
        else:
            lr_vals = [float(points_frame[i, coord_lr]) for i in valid_pts]
            central_idx = min(range(len(valid_pts)), key=lambda j: abs(lr_vals[j] - midline_lr))
            point_c7 = valid_pts[central_idx]
            remaining = [j for j in range(len(valid_pts)) if j != central_idx]
            if len(remaining) != 2:
                return []
            lr_rem = [lr_vals[j] for j in remaining]
            if left_side_positive_lr:
                lsho_local = 0 if lr_rem[0] >= lr_rem[1] else 1
                rsho_local = 1 - lsho_local
            else:
                lsho_local = 0 if lr_rem[0] <= lr_rem[1] else 1
                rsho_local = 1 - lsho_local
            point_lsho = valid_pts[remaining[lsho_local]]
            point_rsho = valid_pts[remaining[rsho_local]]
        assignments = [
            (int(point_c7), lab_c7),
            (int(point_lsho), lab_lsho),
            (int(point_rsho), lab_rsho),
        ]

    if max_match_distance is not None:
        kept = []
        for pi, lab in assignments:
            d = np.linalg.norm(points_frame[pi] - template[lab])
            if d <= max_match_distance:
                kept.append((pi, lab))
        return kept
    return assignments


def match_markers_to_template_z_bands(
    points_frame: np.ndarray,
    template: dict[str, np.ndarray],
    *,
    n_bands: int = N_Z_BANDS,
    boundaries: np.ndarray | None = None,
    label_to_band: dict[str, int] | None = None,
    z_band_by_value: bool = False,
    z_band_by_gap: bool = False,
    z_band_by_rank: bool = False,
    use_hungarian: bool = True,
    max_match_distance: float | None = None,
    match_head_first: bool = False,
    head_lr_ap_threshold_mm: float = 50.0,
    walking_axis_x: bool = True,
    left_side_positive_lr: bool = False,
    head_flip_lr: bool = False,
    head_align_to_shoulders: bool = False,
    head_anterior_smaller_x: bool = False,
    head_anterior_larger_x: bool = False,
    head_swap_lfhd_rbhd: bool = False,
    lr_ap_from_walking: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> list[tuple[int, str]]:
    """
    Match points to template by Z bands. Each band uses the same two-step process:
    1) Points in that Z-band are candidates for that band's markers.
    2) Specific markers are identified using L/R, midline, and A/P (cost = Z similarity + distance;
       only (point, label) pairs that agree on L/R and A/P are allowed). Head: top 4 Z points as
       candidates then L/R and A/P from their median. Other bands: thorax or fallback axes for L/R and A/P.

    Plan 1: When lr_ap_from_walking is set, the same d_back/d_right/centroid_xy are used for all bands.
    Plan 3: When match_head_first, order is head → C7+shoulders → trunk (CLAV,RBAK,STRN,T10) → arm/hand chain → band loop.
    When not match_head_first, trunk then arm/hand chain run after band 10 (same axes; arms by chain, not T-pose Z-band).

    When boundaries and label_to_band are None, they are computed from the template.
    z_band_by_value : if True, bands are equal Z span (min to max template Z).
    z_band_by_gap : if True, boundaries at largest Z gaps between consecutive markers (default False).
    z_band_by_rank : if True, assign bands by Z rank in dynamic frame (highest Z → band 11, etc.); no template Z.
    match_head_first : if True, match head then RSHO/LSHO/C7 by midline, then rest by band.
    walking_axis_x : if True (default), forward walking is along X-axis: L/R from Y, A/P from X.
    head_flip_lr : if True, flip head L/R when assigning LFHD/RFHD/LBHD/RBHD.
    head_align_to_shoulders : if False (default), L/R is from walking direction only; no flip. If True, after C7+shoulders are assigned, flip head labels so LFHD is on same side as LSHO.
    head_anterior_smaller_x : if True, head anterior (front) = smaller X (LFHD, RFHD). Use when front of head has smaller X than back (fixes LFHD/RBHD swap).
    head_anterior_larger_x : if True, force larger X = anterior for head (overrides walking d_back). Use when subject walks +X but computed walking direction is wrong (e.g. centroid X decreases), which would otherwise swap LFHD/RBHD.
    head_swap_lfhd_rbhd : if True, after head assignment swap LFHD and RBHD labels only (fixes diagonal swap when A/P logic is correct but LFHD/RBHD are swapped).
    lr_ap_from_walking : optional (d_back, d_right, centroid_xy) from walking direction; when set,
        these define L/R and A/P for all bands instead of thorax or fallback.
    """
    if z_band_by_rank:
        # Bands by Z rank in dynamic frame: valid points sorted Z descending, assign to band 11, 10, ..., 0 by band sizes.
        if label_to_band is None:
            label_to_band = {}
            for lab in template:
                key = lab.strip().upper() if hasattr(lab, "strip") else str(lab).upper()
                if key in Z_BAND_LABEL_TO_BAND:
                    label_to_band[lab] = Z_BAND_LABEL_TO_BAND[key]
                else:
                    label_to_band[lab] = -1
        point_bands = _point_band_indices_by_z_rank(points_frame, n_bands=n_bands)
        n_bands_actual = n_bands
        boundaries = np.array([])  # unused when z_band_by_rank
    else:
        if boundaries is None or label_to_band is None:
            boundaries, label_to_band = _z_band_boundaries_from_template(
                template, n_bands, z_band_by_value=z_band_by_value, z_band_by_gap=z_band_by_gap
            )
        point_bands = _point_band_indices(points_frame, boundaries)
        n_bands_actual = boundaries.shape[0] + 1

    # Plan 1: L/R and A/P from walking direction (lr_ap_from_walking) when set; else thorax or fallback.
    # Same d_back, d_right, centroid_xy used for all bands (head, C7/shoulders, trunk, arms, band loop).
    thorax_bands = _get_thorax_band_indices(label_to_band)
    rbak_idx, t10_idx, centroid_xy = None, None, None
    t10_xy: np.ndarray | None = None
    d_back, d_right = None, None
    if lr_ap_from_walking is not None:
        d_back, d_right, centroid_xy = lr_ap_from_walking
        t10_xy = centroid_xy.copy().astype(np.float64)
    elif thorax_bands:
        thorax_point_indices = np.concatenate(
            [np.where(point_bands == b)[0] for b in thorax_bands]
        )
        thorax_point_indices = np.unique(thorax_point_indices)
        rbak_idx, t10_idx, centroid_xy = _identify_rbak_t10_from_points(
            points_frame, thorax_point_indices
        )
        if rbak_idx is not None and t10_idx is not None and centroid_xy is not None:
            rbak_xy = points_frame[rbak_idx, :2]
            t10_xy = points_frame[t10_idx, :2].astype(np.float64)
            d_back = rbak_xy - centroid_xy
            d_back_norm = np.linalg.norm(d_back)
            if d_back_norm > 1e-6:
                d_back = d_back / d_back_norm
            d_right = rbak_xy - t10_xy
            dr_norm = np.linalg.norm(d_right)
            if dr_norm > 1e-6:
                d_right = d_right / dr_norm
    if d_back is None or d_right is None:
        t10_xy, centroid_xy, d_back, d_right = _fallback_lr_ap_axes(
            points_frame,
            walking_axis_x=walking_axis_x,
            left_side_positive_lr=left_side_positive_lr,
        )
    elif t10_xy is None and centroid_xy is not None:
        t10_xy = centroid_xy.copy().astype(np.float64)

    assignments: list[tuple[int, str]] = []
    assigned_pts_global: set[int] = set()
    assigned_lab_global: set[str] = set()
    midline_x: float | None = None
    midline_y: float | None = None
    if centroid_xy is not None:
        midline_x = float(centroid_xy[0])
        midline_y = float(centroid_xy[1])

    if match_head_first:
        head_assignments = _match_head_markers_first(
            points_frame,
            template,
            point_bands,
            label_to_band,
            use_hungarian=use_hungarian,
            max_match_distance=max_match_distance,
            head_lr_ap_threshold_mm=head_lr_ap_threshold_mm,
            walking_axis_x=walking_axis_x,
            left_side_positive_lr=left_side_positive_lr,
            head_flip_lr=head_flip_lr,
            head_anterior_smaller_x=head_anterior_smaller_x,
            head_anterior_larger_x=head_anterior_larger_x,
            head_swap_lfhd_rbhd=head_swap_lfhd_rbhd,
            t10_xy=t10_xy,
            centroid_xy=centroid_xy,
            d_back=d_back,
            d_right=d_right,
        )
        assignments = list(head_assignments)
        assigned_pts_global = {pi for pi, _ in assignments}
        assigned_lab_global = {lab for _, lab in assignments}
        if midline_x is None or midline_y is None:
            if len(head_assignments) >= 2:
                head_xy = np.array([
                    points_frame[pi, :2] for pi, _ in head_assignments
                    if np.isfinite(points_frame[pi]).all()
                ])
                if len(head_xy) >= 2:
                    midline_x = float(np.mean(head_xy[:, 0]))
                    midline_y = float(np.mean(head_xy[:, 1]))
        # C7 and shoulders first (no CLAV); then midline = C7 for trunk
        if centroid_xy is not None and t10_xy is not None and d_back is not None and d_right is not None:
            c7_shoulder_assignments = _match_c7_shoulders_only(
                points_frame,
                template,
                point_bands,
                label_to_band,
                assigned_pts_global,
                assigned_lab_global,
                centroid_xy=centroid_xy,
                t10_xy=t10_xy,
                d_right=d_right,
                left_side_positive_lr=left_side_positive_lr,
                max_match_distance=max_match_distance,
            )
            for pi, lab in c7_shoulder_assignments:
                assignments.append((pi, lab))
                assigned_pts_global.add(pi)
                assigned_lab_global.add(lab)
            # Align head L/R to shoulders: use actual LSHO direction (not d_right) so LFHD is on same side as LSHO
            if head_align_to_shoulders:
                c7_pi = next((pi for pi, lab in assignments if str(lab).strip().upper() == "C7"), None)
                lsho_pi = next((pi for pi, lab in assignments if str(lab).strip().upper() == "LSHO"), None)
                rsho_pi = next((pi for pi, lab in assignments if str(lab).strip().upper() == "RSHO"), None)
                lfhd_pi = next((pi for pi, lab in assignments if str(lab).strip().upper() == "LFHD"), None)
                if c7_pi is not None and lsho_pi is not None and rsho_pi is not None and lfhd_pi is not None and np.isfinite(points_frame[c7_pi]).all() and np.isfinite(points_frame[lsho_pi]).all() and np.isfinite(points_frame[lfhd_pi]).all():
                    c7_xy = points_frame[c7_pi, :2].astype(np.float64)
                    left_dir = points_frame[lsho_pi, :2].astype(np.float64) - c7_xy
                    n_left = np.linalg.norm(left_dir)
                    if n_left > 1.0:
                        left_dir = left_dir / n_left
                        # LFHD should be on same side as LSHO (positive dot with left_dir)
                        side_lfhd = float(np.dot(points_frame[lfhd_pi, :2].astype(np.float64) - c7_xy, left_dir))
                        if side_lfhd < 0:
                            head_keys = {str(lab).strip().upper(): lab for lab in template if str(lab).strip().upper() in HEAD_MARKER_SET}
                            swap = {"LFHD": head_keys.get("RFHD"), "RFHD": head_keys.get("LFHD"), "LBHD": head_keys.get("RBHD"), "RBHD": head_keys.get("LBHD")}
                            for k, v in swap.items():
                                if v is None:
                                    swap = {}
                                    break
                            if swap:
                                for i, (pi, lab) in enumerate(assignments):
                                    u = str(lab).strip().upper()
                                    if u in swap:
                                        assignments[i] = (pi, swap[u])
            # Plan 3: Trunk first (midline = C7), then arms/hands by chain (avoids T-pose Z-band for arms).
            # Midline for trunk = C7 position (used for A/P in trunk markers)
            c7_pi = next((pi for pi, lab in assignments if str(lab).strip().upper() == "C7"), None)
            if c7_pi is not None:
                midline_trunk_xy = points_frame[c7_pi, :2].astype(np.float64)
                if z_band_by_rank:
                    c7_z = float(points_frame[c7_pi, 2])
                    lsho_xy = next((points_frame[pi] for pi, lab in assignments if str(lab).strip().upper() == "LSHO"), None)
                    rsho_xy = next((points_frame[pi] for pi, lab in assignments if str(lab).strip().upper() == "RSHO"), None)
                    trunk_assignments = _match_trunk_from_anatomy_pool(
                        points_frame,
                        template,
                        assigned_pts_global,
                        assigned_lab_global,
                        midline_trunk_xy,
                        c7_z,
                        d_back,
                        d_right,
                        max_match_distance=max_match_distance,
                        left_side_positive_lr=left_side_positive_lr,
                        lsho_xy=lsho_xy,
                        rsho_xy=rsho_xy,
                    )
                else:
                    trunk_assignments = _match_trunk_markers_after_shoulders(
                        points_frame,
                        template,
                        point_bands,
                        label_to_band,
                        assigned_pts_global,
                        assigned_lab_global,
                        midline_trunk_xy,
                        d_back,
                        d_right,
                        max_match_distance=max_match_distance,
                        left_side_positive_lr=left_side_positive_lr,
                    )
                for pi, lab in trunk_assignments:
                    assignments.append((pi, lab))
                    assigned_pts_global.add(pi)
                    assigned_lab_global.add(lab)
            # Arm/hand chain from LSHO/RSHO (skip when z_band_by_rank: band loop will assign bands 8,7,6)
            arm_hand_in_tpl = [lab for lab in template if str(lab).strip().upper() in ARM_HAND_LABELS]
            if arm_hand_in_tpl and not z_band_by_rank:
                lsho_pos = next(
                    (points_frame[pi] for pi, lab in assignments if str(lab).strip().upper() == "LSHO"),
                    None,
                )
                rsho_pos = next(
                    (points_frame[pi] for pi, lab in assignments if str(lab).strip().upper() == "RSHO"),
                    None,
                )
                if lsho_pos is not None and rsho_pos is not None and np.isfinite(lsho_pos).all() and np.isfinite(rsho_pos).all():
                    unassigned_for_chain = [
                        i for i in range(points_frame.shape[0])
                        if i not in assigned_pts_global and np.isfinite(points_frame[i]).all()
                    ]
                    if unassigned_for_chain:
                        chain_assignments = match_arm_hand_chain(
                            points_frame,
                            unassigned_for_chain,
                            lsho_pos,
                            rsho_pos,
                            LEFT_ARM_CHAIN_ORDER,
                            RIGHT_ARM_CHAIN_ORDER,
                            max_match_distance=max_match_distance,
                        )
                        for pi, lab in chain_assignments:
                            assignments.append((pi, lab))
                            assigned_pts_global.add(pi)
                            assigned_lab_global.add(lab)
    else:
        assigned_pts_global = set()
        assigned_lab_global = set()

    # When not match_head_first, run trunk then arm chain after band 10 (Plan 3)
    trunk_chain_done = match_head_first  # True if we already did trunk+chain above

    for b in range(n_bands_actual - 1, -1, -1):  # descending Z: band 11 → 10 → … → 0
        assigned_pts_global = {pi for pi, _ in assignments}
        assigned_lab_global = {lab for _, lab in assignments}
        # Step 1: points in this Z-band are candidates for this band's markers
        point_indices = np.where(point_bands == b)[0]
        point_indices = np.array([i for i in point_indices if i not in assigned_pts_global], dtype=int)
        labels_in_band = [
            lab for lab in template
            if label_to_band.get(lab, -1) == b and lab not in assigned_lab_global
        ]
        if len(point_indices) == 0 or len(labels_in_band) == 0:
            continue
        sub_points = points_frame[point_indices]
        positions = np.array([template[lab] for lab in labels_in_band])
        valid_tpl = np.isfinite(positions).all(axis=1)
        assigned_pts_this_band: set[int] = set()
        assigned_lab_this_band: set[str] = set()

        # Step 2: identify specific markers using L/R, midline (from thorax), and A/P
        if t10_xy is not None and centroid_xy is not None and d_back is not None and d_right is not None:
            n_pts, n_lab = len(point_indices), len(labels_in_band)
            cost = np.full((n_pts, n_lab), np.inf)
            for pi in range(n_pts):
                if not np.isfinite(sub_points[pi]).all():
                    continue
                p_xy = sub_points[pi, :2]
                pt_lr, pt_ap = _classify_point_lr_ap(
                    p_xy, t10_xy, centroid_xy, d_back, d_right
                )
                for li in range(n_lab):
                    if not valid_tpl[li]:
                        continue
                    lab_lr, lab_ap = _classify_label_lr_ap(labels_in_band[li])
                    lr_ok = pt_lr == lab_lr or pt_lr == "mid" or lab_lr == "mid"
                    ap_ok = pt_ap == lab_ap or pt_ap == "mid" or lab_ap == "mid"
                    if lr_ok and ap_ok:
                        dist = np.linalg.norm(sub_points[pi] - positions[li])
                        # Z first criterion, then L/R and A/P (filter above)
                        z_diff = abs(float(sub_points[pi, 2]) - float(positions[li][2]))
                        cost[pi, li] = Z_WEIGHT_FOR_LABELING * z_diff + dist
            try:
                row_ind, col_ind = linear_sum_assignment(cost)
                for r, c in zip(row_ind, col_ind):
                    if not np.isfinite(cost[r, c]):
                        continue
                    dist = np.linalg.norm(sub_points[r] - positions[c])
                    if max_match_distance is not None and dist > max_match_distance:
                        continue
                    assignments.append((int(point_indices[r]), labels_in_band[c]))
                    assigned_pts_this_band.add(int(point_indices[r]))
                    assigned_lab_this_band.add(labels_in_band[c])
            except Exception:
                pass

        unassigned_pts = [i for i in point_indices if i not in assigned_pts_this_band]
        unassigned_lab = [lab for lab in labels_in_band if lab not in assigned_lab_this_band]
        if unassigned_pts and unassigned_lab:
            sub_template = {lab: template[lab].copy() for lab in unassigned_lab}
            sub_pts = points_frame[unassigned_pts]
            sub_asgn = match_markers_to_template(
                sub_pts,
                sub_template,
                use_hungarian=use_hungarian,
                max_match_distance=max_match_distance,
            )
            for local_idx, lab in sub_asgn:
                assignments.append((int(unassigned_pts[local_idx]), lab))
        # Plan 3 (non-head-first path): after band 10 we have C7, LSHO, RSHO; label trunk then arms/hands
        if b == 10 and not trunk_chain_done and t10_xy is not None and centroid_xy is not None and d_back is not None and d_right is not None:
            lab_upper = {str(lab).strip().upper() for lab in assigned_lab_global}
            if "C7" in lab_upper and "LSHO" in lab_upper and "RSHO" in lab_upper:
                c7_pi = next((pi for pi, lab in assignments if str(lab).strip().upper() == "C7"), None)
                if c7_pi is not None and np.isfinite(points_frame[c7_pi]).all():
                    midline_trunk_xy = points_frame[c7_pi, :2].astype(np.float64)
                    if z_band_by_rank:
                        c7_z = float(points_frame[c7_pi, 2])
                        lsho_xy = next((points_frame[pi] for pi, lab in assignments if str(lab).strip().upper() == "LSHO"), None)
                        rsho_xy = next((points_frame[pi] for pi, lab in assignments if str(lab).strip().upper() == "RSHO"), None)
                        trunk_assignments = _match_trunk_from_anatomy_pool(
                            points_frame,
                            template,
                            assigned_pts_global,
                            assigned_lab_global,
                            midline_trunk_xy,
                            c7_z,
                            d_back,
                            d_right,
                            max_match_distance=max_match_distance,
                            left_side_positive_lr=left_side_positive_lr,
                            lsho_xy=lsho_xy,
                            rsho_xy=rsho_xy,
                        )
                    else:
                        trunk_assignments = _match_trunk_markers_after_shoulders(
                            points_frame,
                            template,
                            point_bands,
                            label_to_band,
                            assigned_pts_global,
                            assigned_lab_global,
                            midline_trunk_xy,
                            d_back,
                            d_right,
                            max_match_distance=max_match_distance,
                            left_side_positive_lr=left_side_positive_lr,
                        )
                    for pi, lab in trunk_assignments:
                        assignments.append((pi, lab))
                        assigned_pts_global.add(pi)
                        assigned_lab_global.add(lab)
                    arm_hand_in_tpl = [lab for lab in template if str(lab).strip().upper() in ARM_HAND_LABELS]
                    if arm_hand_in_tpl and not z_band_by_rank:
                        lsho_pos = next((points_frame[pi] for pi, lab in assignments if str(lab).strip().upper() == "LSHO"), None)
                        rsho_pos = next((points_frame[pi] for pi, lab in assignments if str(lab).strip().upper() == "RSHO"), None)
                        if lsho_pos is not None and rsho_pos is not None and np.isfinite(lsho_pos).all() and np.isfinite(rsho_pos).all():
                            unassigned_for_chain = [
                                i for i in range(points_frame.shape[0])
                                if i not in assigned_pts_global and np.isfinite(points_frame[i]).all()
                            ]
                            if unassigned_for_chain:
                                chain_assignments = match_arm_hand_chain(
                                    points_frame,
                                    unassigned_for_chain,
                                    lsho_pos,
                                    rsho_pos,
                                    LEFT_ARM_CHAIN_ORDER,
                                    RIGHT_ARM_CHAIN_ORDER,
                                    max_match_distance=max_match_distance,
                                )
                                for pi, lab in chain_assignments:
                                    assignments.append((pi, lab))
                                    assigned_pts_global.add(pi)
                                    assigned_lab_global.add(lab)
                trunk_chain_done = True
    # Second pass: any unassigned points (e.g. NaN Z) and unassigned labels — global match
    assigned_points = {pi for pi, _ in assignments}
    assigned_labels = {lab for _, lab in assignments}
    unassigned_points = [
        i for i in range(points_frame.shape[0])
        if i not in assigned_points and np.isfinite(points_frame[i]).all()
    ]
    unassigned_template = {
        lab: template[lab] for lab in template
        if lab not in assigned_labels and np.isfinite(template[lab]).all()
    }
    if unassigned_points and unassigned_template:
        sub_points = points_frame[unassigned_points]
        sub_asgn = match_markers_to_template(
            sub_points,
            unassigned_template,
            use_hungarian=use_hungarian,
            max_match_distance=max_match_distance,
        )
        for local_idx, lab in sub_asgn:
            assignments.append((int(unassigned_points[local_idx]), lab))
    return assignments


def match_arm_hand_chain(
    points_frame: np.ndarray,
    unassigned_point_indices: list[int],
    lsho_pos: np.ndarray | None,
    rsho_pos: np.ndarray | None,
    left_chain: tuple[str, ...],
    right_chain: tuple[str, ...],
    *,
    max_match_distance: float | None = None,
) -> list[tuple[int, str]]:
    """
    Match arm/hand labels to unassigned points by proximal-to-distal chain from shoulder.
    For each arm, repeatedly find the unassigned point nearest to the current anchor and assign
    the next label in the chain; anchor moves to that point for the next step.
    """
    assignments: list[tuple[int, str]] = []
    unassigned = set(unassigned_point_indices)
    pts = points_frame

    def chain_one_arm(anchor_pos: np.ndarray | None, chain_order: tuple[str, ...]) -> None:
        if anchor_pos is None or not np.isfinite(anchor_pos).all() or not chain_order:
            return
        anchor = anchor_pos.astype(np.float64)
        for lab in chain_order:
            if not unassigned:
                break
            best_idx, best_dist = None, np.inf
            for pi in unassigned:
                if not np.isfinite(pts[pi]).all():
                    continue
                d = np.linalg.norm(pts[pi] - anchor)
                if d < best_dist:
                    best_dist = d
                    best_idx = pi
            if best_idx is not None and (max_match_distance is None or best_dist <= max_match_distance):
                assignments.append((best_idx, lab))
                unassigned.discard(best_idx)
                anchor = pts[best_idx]

    chain_one_arm(lsho_pos, left_chain)
    chain_one_arm(rsho_pos, right_chain)
    return assignments


def best_frame_by_anchor_cost(
    points_dynamic: np.ndarray,
    template: dict[str, np.ndarray],
    *,
    middle_start: float = DEFAULT_MIDDLE_START,
    middle_end: float = DEFAULT_MIDDLE_END,
    use_hungarian: bool = True,
    visibility_min_fraction: float = DEFAULT_VISIBILITY_MIN_FRACTION,
    loose_cap_for_provisional: float = 600.0,
) -> tuple[int, float]:
    """
    Choose the frame where anchor-only RMS is minimum; return (best_frame, scale_s).
    scale_s = RMS over anchor pairs at best frame (for tiered thresholds).
    Uses provisional full-body match with loose_cap to get anchor pairs per frame.
    """
    anchor_in_template = [
        lab for lab in template
        if str(lab).strip().upper() in WHOLE_BODY_ANCHOR_SET
    ]
    if len(anchor_in_template) < 3:
        best_f = best_frame_for_matching(
            points_dynamic, None,
            middle_start=middle_start, middle_end=middle_end,
            visibility_min_fraction=visibility_min_fraction,
        )
        return best_f, REFERENCE_39_MEAN_RMS_MM

    n_frames = points_dynamic.shape[0]
    start = int(n_frames * middle_start)
    end = int(n_frames * middle_end)
    start = max(0, min(start, n_frames - 1))
    end = max(start + 1, min(end, n_frames))
    candidates = list(range(start, end))
    n_points_d = points_dynamic.shape[1]
    n_valid_per_frame = np.sum(np.isfinite(points_dynamic).all(axis=2), axis=1)
    visibility_ok = n_valid_per_frame >= (visibility_min_fraction * n_points_d)
    candidates_visible = [f for f in candidates if visibility_ok[f]]
    if not candidates_visible:
        candidates_visible = candidates

    def anchor_rms_at_frame(f: int) -> float:
        asgn = match_markers_to_template_z_bands(
            points_dynamic[f], template,
            use_hungarian=use_hungarian,
            max_match_distance=loose_cap_for_provisional,
        )
        anchor_pairs = [
            (pi, lab) for pi, lab in asgn
            if str(lab).strip().upper() in WHOLE_BODY_ANCHOR_SET
        ]
        if len(anchor_pairs) < 3:
            return np.inf
        src = np.array([template[lab] for _, lab in anchor_pairs])
        tgt = np.array([points_dynamic[f, pi] for pi, _ in anchor_pairs])
        R, t = _rigid_transform_3d(src, tgt)
        pred = (R @ src.T).T + t
        rms = np.sqrt(np.mean(np.sum((tgt - pred) ** 2, axis=1)))
        return float(rms)

    best_f = min(candidates_visible, key=anchor_rms_at_frame)
    s = anchor_rms_at_frame(best_f)
    if not np.isfinite(s) or s <= 0:
        s = REFERENCE_39_MEAN_RMS_MM
    return best_f, s


def best_frame_by_trunk_cost(
    points_dynamic: np.ndarray,
    template: dict[str, np.ndarray],
    residual_dynamic: np.ndarray | None,
    *,
    middle_start: float = DEFAULT_MIDDLE_START,
    middle_end: float = DEFAULT_MIDDLE_END,
    use_hungarian: bool = True,
    max_match_distance: float | None = None,
    trunk_labels: list[str] | None = None,
    visibility_min_fraction: float = DEFAULT_VISIBILITY_MIN_FRACTION,
) -> int:
    """
    Choose the frame in the middle range where trunk assignment cost is minimum.
    Only frames where at least visibility_min_fraction of markers are valid are considered.

    For each candidate frame we run a provisional full-body match, then sum
    distances only for pairs whose template label is in trunk_labels. The frame
    with minimum trunk cost is returned. Falls back to best_frame_for_matching
    if trunk_labels has fewer than 3 labels.
    """
    if trunk_labels is None:
        trunk_labels = _trunk_labels_in_template(template)
    n_frames = points_dynamic.shape[0]
    start = int(n_frames * middle_start)
    end = int(n_frames * middle_end)
    start = max(0, min(start, n_frames - 1))
    end = max(start + 1, min(end, n_frames))
    candidates = list(range(start, end))
    trunk_set = {str(l).strip().upper() for l in trunk_labels}
    if len(trunk_set) < 3:
        return best_frame_for_matching(
            points_dynamic, residual_dynamic,
            middle_start=middle_start, middle_end=middle_end,
            visibility_min_fraction=visibility_min_fraction,
        )

    n_points_d = points_dynamic.shape[1]
    n_valid_per_frame = np.sum(np.isfinite(points_dynamic).all(axis=2), axis=1)
    visibility_ok = n_valid_per_frame >= (visibility_min_fraction * n_points_d)
    candidates_visible = [f for f in candidates if visibility_ok[f]]
    if not candidates_visible:
        candidates_visible = candidates

    def trunk_cost(f: int) -> float:
        asgn = match_markers_to_template_z_bands(
            points_dynamic[f], template,
            use_hungarian=use_hungarian,
            max_match_distance=max_match_distance,
        )
        total = 0.0
        n_trunk = 0
        for pi, lab in asgn:
            if str(lab).strip().upper() not in trunk_set:
                continue
            d = np.linalg.norm(points_dynamic[f, pi] - template[lab])
            if np.isfinite(d):
                total += d
                n_trunk += 1
        if n_trunk < 3:
            return np.inf
        return total

    best = min(candidates_visible, key=trunk_cost)
    if np.isinf(trunk_cost(best)):
        return best_frame_for_matching(
            points_dynamic, residual_dynamic,
            middle_start=middle_start, middle_end=middle_end,
            visibility_min_fraction=visibility_min_fraction,
        )
    return best


def procrustes_fit_trunk(
    points_frame: np.ndarray,
    template: dict[str, np.ndarray],
    assignments: list[tuple[int, str]],
    trunk_labels: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit rigid (R, t) from template to points using only trunk-assigned pairs.
    Returns R, t so that aligned_template = R @ template + t.
    """
    trunk_set = {str(l).strip().upper() for l in trunk_labels}
    src_list, tgt_list = [], []
    for pi, lab in assignments:
        if str(lab).strip().upper() not in trunk_set:
            continue
        pos_t = template.get(lab)
        pos_p = points_frame[pi]
        if pos_t is not None and np.isfinite(pos_t).all() and np.isfinite(pos_p).all():
            src_list.append(pos_t)
            tgt_list.append(pos_p)
    if len(src_list) < 3:
        return np.eye(3), np.zeros(3)
    src = np.array(src_list)
    tgt = np.array(tgt_list)
    return _rigid_transform_3d(src, tgt)


def procrustes_fit_anchor(
    points_frame: np.ndarray,
    template: dict[str, np.ndarray],
    assignments: list[tuple[int, str]],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit rigid (R, t) from template to points using only anchor-assigned pairs.
    Returns R, t so that aligned_template = R @ template + t.
    """
    anchor_set = WHOLE_BODY_ANCHOR_SET
    src_list, tgt_list = [], []
    for pi, lab in assignments:
        if str(lab).strip().upper() not in anchor_set:
            continue
        pos_t = template.get(lab)
        pos_p = points_frame[pi]
        if pos_t is not None and np.isfinite(pos_t).all() and np.isfinite(pos_p).all():
            src_list.append(pos_t)
            tgt_list.append(pos_p)
    if len(src_list) < 3:
        return np.eye(3), np.zeros(3)
    src = np.array(src_list)
    tgt = np.array(tgt_list)
    return _rigid_transform_3d(src, tgt)


def _trunk_rms(
    points_frame: np.ndarray,
    template: dict[str, np.ndarray],
    assignments: list[tuple[int, str]],
    trunk_labels: list[str],
) -> float:
    """RMS distance for trunk-assigned pairs only. Returns inf if fewer than 3 trunk pairs."""
    trunk_set = {str(l).strip().upper() for l in trunk_labels}
    dists = []
    for pi, lab in assignments:
        if str(lab).strip().upper() not in trunk_set:
            continue
        pos_t = template.get(lab)
        pos_p = points_frame[pi]
        if pos_t is not None and np.isfinite(pos_t).all() and np.isfinite(pos_p).all():
            dists.append(np.linalg.norm(pos_p - pos_t))
    if len(dists) < 3:
        return np.inf
    return float(np.sqrt(np.mean(np.array(dists) ** 2)))


def propagate_labels_temporal(
    points: np.ndarray,
    initial_assignments: list[tuple[int, str]],
    frame_start: int,
    *,
    max_propagation_distance: float | None = None,
) -> np.ndarray:
    """
    Propagate labels forward and backward from frame_start using nearest neighbor.

    Parameters
    ----------
    points : (n_frames, n_points, 3)
    initial_assignments : list of (point_idx, label) at frame_start
    frame_start : frame index where initial assignment was done
    max_propagation_distance : if set, do not assign when nearest distance > this (mm);
        avoids wrong re-acquisition after dropout or cross-over

    Returns
    -------
    label_per_point_per_frame : (n_frames, n_points) dtype object, empty string for unassigned
    """
    n_frames, n_points, _ = points.shape
    result = np.empty((n_frames, n_points), dtype=object)
    result[:] = ""
    for pi, lab in initial_assignments:
        result[frame_start, pi] = lab
    # Backward
    for f in range(frame_start - 1, -1, -1):
        for pi in range(n_points):
            if result[f + 1, pi] == "":
                continue
            lab = result[f + 1, pi]
            pos_next = points[f + 1, pi]
            if not np.isfinite(pos_next).all():
                result[f, pi] = lab
                continue
            pos_curr = points[f, :, :]
            valid = np.isfinite(pos_curr).all(axis=1)
            dists = np.linalg.norm(pos_curr - pos_next, axis=1)
            dists[~valid] = np.inf
            nearest = np.argmin(dists)
            d = dists[nearest]
            if d < np.inf and (max_propagation_distance is None or d <= max_propagation_distance):
                result[f, nearest] = lab
    # Forward
    for f in range(frame_start + 1, n_frames):
        for pi in range(n_points):
            if result[f - 1, pi] == "":
                continue
            lab = result[f - 1, pi]
            pos_prev = points[f - 1, pi]
            if not np.isfinite(pos_prev).all():
                result[f, pi] = lab
                continue
            pos_curr = points[f, :, :]
            valid = np.isfinite(pos_curr).all(axis=1)
            dists = np.linalg.norm(pos_curr - pos_prev, axis=1)
            dists[~valid] = np.inf
            nearest = np.argmin(dists)
            d = dists[nearest]
            if d < np.inf and (max_propagation_distance is None or d <= max_propagation_distance):
                result[f, nearest] = lab
    return result


def label_body_markers(
    points_dynamic: np.ndarray,
    template: dict[str, np.ndarray],
    *,
    residual_dynamic: np.ndarray | None = None,
    middle_start: float = DEFAULT_MIDDLE_START,
    middle_end: float = DEFAULT_MIDDLE_END,
    visibility_min_fraction: float = DEFAULT_VISIBILITY_MIN_FRACTION,
    use_hungarian: bool = True,
    max_match_distance: float | None = None,
    max_propagation_distance: float | None = None,
    use_whole_body_39: bool = False,
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
    lr_ap_from_walking: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[list[str], np.ndarray]:
    """
    Label body markers in dynamic trial using template and temporal propagation.
    Plan 2: Z-band is prioritized over anchor-based labeling. Anchor path runs only when
    use_whole_body_39 is True and neither z_band_by_gap nor match_head_first is set.
    Plan 1: L/R and A/P from walking (lr_ap_from_walking) are used for the entire Z-band path when provided.
    Plan 3: Trunk (CLAV, RBAK, STRN, T10) is labeled before arms/hands; arms/hands use chain from shoulders (not T-pose Z-band).
    """
    n_frames, n_points, _ = points_dynamic.shape

    # Plan 2: Prefer Z-band over anchor unless explicitly whole-body-39 only (no match_head_first / z_band_by_gap)
    use_anchor_path = (
        use_whole_body_39
        and not z_band_by_gap
        and not match_head_first
    )
    if use_anchor_path:
        best_f, s = best_frame_by_anchor_cost(
            points_dynamic, template,
            middle_start=middle_start, middle_end=middle_end,
            use_hungarian=use_hungarian,
            visibility_min_fraction=visibility_min_fraction,
        )
        pts_f = points_dynamic[best_f]
        # Provisional match with loose cap to get assignments for anchor Procrustes
        loose_cap = 600.0
        assignments_prov = match_markers_to_template_z_bands(
            pts_f, template,
            use_hungarian=use_hungarian,
            max_match_distance=loose_cap,
        )
        R, t = procrustes_fit_anchor(pts_f, template, assignments_prov)
        template_aligned = _apply_rigid_to_template(template, R, t)
        # Tiered per-label cap with minimum so small s does not over-reject (e.g. head)
        per_label_max: dict[str, float] = {}
        for lab in template:
            u = str(lab).strip().upper()
            if u in WHOLE_BODY_ANCHOR_SET:
                per_label_max[lab.strip()] = max(ALPHA_ANCHOR * s, MIN_CAP_ANCHOR_MM)
            elif u in WHOLE_BODY_DISTAL_LABELS:
                per_label_max[lab.strip()] = max(ALPHA_DISTAL * s, MIN_CAP_DISTAL_MM)
            else:
                per_label_max[lab.strip()] = max(ALPHA_MID * s, MIN_CAP_MID_MM)
        assignments = match_markers_to_template(
            pts_f,
            template_aligned,
            use_hungarian=use_hungarian,
            per_label_max_distance=per_label_max,
        )
        # Fallback: assign any unassigned labels to closest unassigned point within loose cap
        assigned_pts = {pi for pi, _ in assignments}
        assigned_labs = {lab for _, lab in assignments}
        unassigned_labs = [lab for lab in template if lab.strip() not in {l.strip() for l in assigned_labs}]
        unassigned_pts = [pi for pi in range(pts_f.shape[0]) if pi not in assigned_pts and np.isfinite(pts_f[pi]).all()]
        fallback_cap = max(ALPHA_DISTAL * s, SUGGESTED_MAX_MATCH_DISTANCE_39_MM, FALLBACK_CAP_MM)
        if unassigned_labs and unassigned_pts:
            for lab in unassigned_labs:
                pos_t = template_aligned.get(lab)
                if pos_t is None or not np.isfinite(pos_t).all():
                    continue
                best_pi, best_d = None, np.inf
                for pi in unassigned_pts:
                    d = np.linalg.norm(pts_f[pi] - pos_t)
                    if d < best_d and d <= fallback_cap:
                        best_d = d
                        best_pi = pi
                if best_pi is not None:
                    assignments.append((best_pi, lab))
                    assigned_pts.add(best_pi)
                    unassigned_pts = [p for p in unassigned_pts if p != best_pi]
                    if not unassigned_pts:
                        break
        prop_cap = ALPHA_PROP * s
        label_per_frame = propagate_labels_temporal(
            points_dynamic,
            assignments,
            best_f,
            max_propagation_distance=prop_cap,
        )
        labels_out = [label_per_frame[best_f, pi] for pi in range(n_points)]
        return labels_out, label_per_frame

    trunk_labels = _trunk_labels_in_template(template)
    use_gap = z_band_by_gap or match_head_first
    if use_gap:
        # Best frame: highest quality among frames with both L/R heel contacts
        best_f = best_frame_heel_contact(
            points_dynamic,
            residual_dynamic,
            middle_start=middle_start,
            middle_end=middle_end,
            visibility_min_fraction=visibility_min_fraction,
            walking_axis_x=walking_axis_x,
        )
    elif len(trunk_labels) >= 3:
        best_f = best_frame_by_trunk_cost(
            points_dynamic, template, residual_dynamic,
            middle_start=middle_start, middle_end=middle_end,
            use_hungarian=use_hungarian,
            max_match_distance=max_match_distance,
            trunk_labels=trunk_labels,
            visibility_min_fraction=visibility_min_fraction,
        )
    else:
        best_f = best_frame_for_matching(
            points_dynamic, residual_dynamic,
            middle_start=middle_start, middle_end=middle_end,
            visibility_min_fraction=visibility_min_fraction,
        )
    pts_f = points_dynamic[best_f]
    # When match_head_first, use Z-band-by-gap so head then RSHO/LSHO/C7 band order is well defined
    use_gap = z_band_by_gap or match_head_first
    if z_band_by_rank:
        boundaries, label_to_band = None, None
    else:
        boundaries, label_to_band = _z_band_boundaries_from_template(
            template, n_bands=N_Z_BANDS, z_band_by_value=z_band_by_value, z_band_by_gap=use_gap
        )
    assignments = match_markers_to_template_z_bands(
        pts_f,
        template,
        boundaries=boundaries,
        label_to_band=label_to_band,
        z_band_by_value=z_band_by_value,
        z_band_by_gap=use_gap,
        z_band_by_rank=z_band_by_rank,
        use_hungarian=use_hungarian,
        max_match_distance=max_match_distance,
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
    # Align template using trunk-only Procrustes, then re-match — unless alignment quality is bad (fallback).
    _TRUNK_RMS_MAX_MM = 150.0
    if len(trunk_labels) >= 3 and assignments:
        initial_rms = _trunk_rms(pts_f, template, assignments, trunk_labels)
        R, t = procrustes_fit_trunk(pts_f, template, assignments, trunk_labels)
        template_aligned = _apply_rigid_to_template(template, R, t)
        if z_band_by_rank:
            boundaries_al, label_to_band_al = None, None
        else:
            boundaries_al, label_to_band_al = _z_band_boundaries_from_template(
                template_aligned, n_bands=N_Z_BANDS, z_band_by_value=z_band_by_value, z_band_by_gap=use_gap
            )
        assignments_aligned = match_markers_to_template_z_bands(
            pts_f,
            template_aligned,
            boundaries=boundaries_al,
            label_to_band=label_to_band_al,
            z_band_by_value=z_band_by_value,
            z_band_by_gap=use_gap,
            z_band_by_rank=z_band_by_rank,
            use_hungarian=use_hungarian,
            max_match_distance=max_match_distance,
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
        aligned_rms = _trunk_rms(pts_f, template_aligned, assignments_aligned, trunk_labels)
        if aligned_rms < _TRUNK_RMS_MAX_MM and aligned_rms < initial_rms:
            # When head L/R was not aligned to shoulders, keep head assignments from the first
            # call so a second run (with aligned template) cannot overwrite them.
            if not head_align_to_shoulders:
                head_from_first = [(pi, lab) for pi, lab in assignments if str(lab).strip().upper() in HEAD_MARKER_SET]
                non_head_aligned = [(pi, lab) for pi, lab in assignments_aligned if str(lab).strip().upper() not in HEAD_MARKER_SET]
                assignments = non_head_aligned + head_from_first
            else:
                assignments = assignments_aligned
    if not assignments:
        return [""] * n_points, np.empty((n_frames, n_points), dtype=object)
    label_per_frame = propagate_labels_temporal(
        points_dynamic,
        assignments,
        best_f,
        max_propagation_distance=max_propagation_distance,
    )
    labels_out = [label_per_frame[best_f, pi] for pi in range(n_points)]
    return labels_out, label_per_frame
