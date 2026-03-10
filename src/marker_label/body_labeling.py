"""Body marker labeling via subject-specific static template and temporal propagation."""

from __future__ import annotations

import logging
import numpy as np
from scipy.optimize import linear_sum_assignment

from .constants import (
    DEFAULT_MIDDLE_END,
    DEFAULT_MIDDLE_START,
    HEAD_MARKERS,
    HEAD_MARKERS_SET,
    C7_SHOULDER_MARKERS,
    C7_SHOULDER_MARKERS_SET,
    CLAV_RBAK_MARKERS,
    CLAV_RBAK_MARKERS_SET,
    STRN_T10_ARM_MARKERS_SET,
    PELVIS_ARM12_MARKERS_SET,
    LEG_FOOT12_MARKERS_SET,
    TRUNK_LABELS,
)
from .constants import (
    SCREENING_BEST_FRAME_VISIBILITY_MIN,
    SCREENING_FOOT_PROXY_N_SMALLEST_Z,
)
from .pelvis import build_pelvis_frame, points_to_pelvis_frame
from . import static_geometry

_TRUNK_SET = {t.upper() for t in TRUNK_LABELS}

# CLAV/RBAK validation error codes (raised when post-assignment checks fail)
CLAVRBAK_RBAK_NOT_POSTERIOR_TO_CLAV = "RBAK_NOT_POSTERIOR_TO_CLAV"
CLAVRBAK_RBAK_NOT_RIGHT_OF_CLAV = "RBAK_NOT_RIGHT_OF_CLAV"
CLAVRBAK_OUTSIDE_SHOULDER_Y_RANGE = "CLAV_OR_RBAK_OUTSIDE_SHOULDER_Y_RANGE"


class CLAVRBAKValidationError(Exception):
    """Raised when CLAV/RBAK labeling fails validation (Y range or RBAK posterior/right of CLAV)."""

    def __init__(self, error_code: str, message: str):
        self.error_code = error_code
        self.message = message
        super().__init__(f"CLAV/RBAK validation failed ({error_code}): {message}")


def _trunk_labels_in_template(template: dict) -> list[str]:
    """Return template keys whose label (case-insensitive) is in TRUNK_LABELS."""
    return [k for k in template if str(k).strip().upper() in _TRUNK_SET]


def assign_head_markers_by_top4_z(
    points_frame: np.ndarray,
    d_back_xy: np.ndarray,
    d_right_xy: np.ndarray,
    head_labels: list[str] | None = None,
) -> list[tuple[int, str]]:
    """
    Assign head markers at one frame: sort points by Z descending, take top 4, then assign
    LFHD, RFHD, LBHD, RBHD by A/P (d_back) and L/R (direct Y comparison, no centroid).

    A/P: smaller projection onto d_back = anterior, larger = posterior.
    L/R: compare candidate points' Y directly. When d_right_xy[1] < 0 (x increases
    during walking), larger Y = left; when d_right_xy[1] > 0 (x decreases), smaller Y = left.

    Parameters
    ----------
    points_frame : (n_points, 3)
    d_back_xy : (2,) xy unit vector toward posterior
    d_right_xy : (2,) xy unit vector toward subject's right (used only for L/R sign)
    head_labels : optional list of 4 labels in order [LFHD, RFHD, LBHD, RBHD]

    Returns
    -------
    assignments : list of (point_idx, label), length min(4, n_valid_points)
    """
    if head_labels is None:
        head_labels = list(HEAD_MARKERS)
    z = points_frame[:, 2]
    valid = np.isfinite(z)
    if np.sum(valid) < 1:
        return []
    order_z = np.argsort(-np.where(valid, z, -np.inf))[:4]
    top4 = [int(i) for i in order_z if np.isfinite(points_frame[i, 2])]
    if len(top4) < 1:
        return []
    xy = np.asarray(points_frame[top4, :2], dtype=np.float64)
    centroid_xy = np.nanmean(xy, axis=0)
    rel = xy - centroid_xy
    dot_back = rel @ d_back_xy
    order_ap = np.argsort(dot_back)
    anterior_local = order_ap[:2]
    posterior_local = order_ap[2:]
    # L/R by direct Y comparison (no centroid): larger Y = left when d_right[1] < 0 (x increases)
    larger_y_is_left = d_right_xy[1] < 0
    out: list[tuple[int, str]] = []
    if len(anterior_local) == 2:
        y_ant = np.array([points_frame[top4[i], 1] for i in anterior_local])
        order_y_ant = np.argsort(-y_ant) if larger_y_is_left else np.argsort(y_ant)
        lr_ant = anterior_local[order_y_ant]
        out.append((top4[lr_ant[0]], head_labels[0]))
        out.append((top4[lr_ant[1]], head_labels[1]))
    elif len(anterior_local) == 1:
        out.append((top4[anterior_local[0]], head_labels[0]))
    if len(posterior_local) == 2:
        y_post = np.array([points_frame[top4[i], 1] for i in posterior_local])
        order_y_post = np.argsort(-y_post) if larger_y_is_left else np.argsort(y_post)
        lr_post = posterior_local[order_y_post]
        out.append((top4[lr_post[0]], head_labels[2]))
        out.append((top4[lr_post[1]], head_labels[3]))
    elif len(posterior_local) == 1:
        out.append((top4[posterior_local[0]], head_labels[2]))
    return out


def assign_c7_shoulders_after_head(
    points_frame: np.ndarray,
    d_right_xy: np.ndarray,
    exclude_pt_indices: set[int],
    template: dict,
) -> list[tuple[int, str]]:
    """
    After head is assigned: among remaining points, sort by Z descending;
    assign 1st = C7, 2nd and 3rd = LSHO/RSHO. L/R by direct Y comparison (no centroid):
    when d_right_xy[1] < 0 (x increases during walking), larger Y = left; else smaller Y = left.

    Parameters
    ----------
    points_frame : (n_points, 3)
    d_right_xy : (2,) xy unit vector toward subject's right (used only for L/R sign)
    exclude_pt_indices : point indices already assigned (e.g. head)
    template : dict label -> position; used to get actual label strings for C7, LSHO, RSHO

    Returns
    -------
    assignments : list of (point_idx, label), length 0--3
    """
    remaining = [i for i in range(points_frame.shape[0]) if i not in exclude_pt_indices]
    z = points_frame[:, 2]
    valid = np.array([np.isfinite(z[i]) for i in remaining])
    if np.sum(valid) < 1:
        return []
    order_z = np.argsort(-np.array([z[i] if np.isfinite(z[i]) else -np.inf for i in remaining]))[:3]
    top3 = [remaining[int(i)] for i in order_z if np.isfinite(points_frame[remaining[i], 2])]
    if len(top3) < 1:
        return []
    # Template labels for C7, LSHO, RSHO (preserve case from template)
    label_c7 = next((lab for lab in template if str(lab).strip().upper() == "C7"), "C7")
    label_lsho = next((lab for lab in template if str(lab).strip().upper() == "LSHO"), "LSHO")
    label_rsho = next((lab for lab in template if str(lab).strip().upper() == "RSHO"), "RSHO")
    out: list[tuple[int, str]] = []
    out.append((top3[0], label_c7))
    if len(top3) >= 3:
        # L/R by direct Y comparison (no centroid): larger Y = left when d_right[1] < 0
        y_sho = points_frame[top3[1:3], 1]
        larger_y_is_left = d_right_xy[1] < 0
        order_y = np.argsort(-y_sho) if larger_y_is_left else np.argsort(y_sho)
        out.append((top3[1 + order_y[0]], label_lsho))
        out.append((top3[1 + order_y[1]], label_rsho))
    elif len(top3) == 2:
        y_shoulder = points_frame[top3[1], 1]
        y_c7 = points_frame[top3[0], 1]
        larger_y_is_left = d_right_xy[1] < 0
        is_left = (y_shoulder > y_c7) if larger_y_is_left else (y_shoulder < y_c7)
        out.append((top3[1], label_lsho if is_left else label_rsho))
    return out


def assign_clav_rbak_after_c7_shoulders(
    points_frame: np.ndarray,
    d_back_xy: np.ndarray,
    d_right_xy: np.ndarray,
    exclude_pt_indices: set[int],
    lsho_idx: int,
    rsho_idx: int,
    template: dict,
) -> list[tuple[int, str]]:
    """
    After C7 and shoulders: among remaining points with Y between the two shoulders
    (min/max of LSHO_y and RSHO_y on this frame), take the two highest Z; assign
    highest Z = CLAV, next-highest Z = RBAK. Then validate: RBAK must be posterior and
    right of CLAV, and both must lie in the shoulder Y range; otherwise raise
    CLAVRBAKValidationError with an error code.

    Parameters
    ----------
    points_frame : (n_points, 3)
    d_back_xy : (2,) xy unit vector toward posterior (used only for validation)
    d_right_xy : (2,) xy unit vector toward subject's right (used only for validation)
    exclude_pt_indices : point indices already assigned (head + C7 + shoulders)
    lsho_idx, rsho_idx : point indices for LSHO and RSHO (used for Y range on this frame)
    template : for label strings CLAV, RBAK

    Returns
    -------
    assignments : list of (point_idx, label), length 0--2

    Raises
    ------
    CLAVRBAKValidationError
        If RBAK is not posterior to CLAV, not right of CLAV, or CLAV/RBAK Y outside shoulder range.
    """
    y_lsho = points_frame[lsho_idx, 1]
    y_rsho = points_frame[rsho_idx, 1]
    if not (np.isfinite(y_lsho) and np.isfinite(y_rsho)):
        return []
    y_min = min(y_lsho, y_rsho)
    y_max = max(y_lsho, y_rsho)
    remaining = [
        i for i in range(points_frame.shape[0])
        if i not in exclude_pt_indices
        and np.isfinite(points_frame[i, 2])
        and y_min <= points_frame[i, 1] <= y_max
    ]
    if len(remaining) < 1:
        return []
    z_vals = np.array([points_frame[i, 2] for i in remaining])
    order_z = np.argsort(-z_vals)[:2]
    top2 = [remaining[int(i)] for i in order_z]
    label_clav = next((lab for lab in template if str(lab).strip().upper() == "CLAV"), "CLAV")
    label_rbak = next((lab for lab in template if str(lab).strip().upper() == "RBAK"), "RBAK")
    out: list[tuple[int, str]] = []
    if len(top2) == 1:
        out.append((top2[0], label_clav))
        return out
    # Assign by Z order only: highest Z = CLAV, next = RBAK
    clav_idx = top2[0]
    rbak_idx = top2[1]
    clav_pt = points_frame[clav_idx]
    rbak_pt = points_frame[rbak_idx]
    # Validation: RBAK must be posterior and right of CLAV; both in shoulder Y range
    if clav_pt[1] < y_min or clav_pt[1] > y_max:
        raise CLAVRBAKValidationError(
            CLAVRBAK_OUTSIDE_SHOULDER_Y_RANGE,
            f"CLAV Y={clav_pt[1]:.1f} outside shoulder Y range [{y_min:.1f}, {y_max:.1f}].",
        )
    if rbak_pt[1] < y_min or rbak_pt[1] > y_max:
        raise CLAVRBAKValidationError(
            CLAVRBAK_OUTSIDE_SHOULDER_Y_RANGE,
            f"RBAK Y={rbak_pt[1]:.1f} outside shoulder Y range [{y_min:.1f}, {y_max:.1f}].",
        )
    xy_clav = np.asarray(clav_pt[:2], dtype=np.float64)
    xy_rbak = np.asarray(rbak_pt[:2], dtype=np.float64)
    centroid_xy = np.nanmean(np.array([xy_clav, xy_rbak]), axis=0)
    rel_clav = xy_clav - centroid_xy
    rel_rbak = xy_rbak - centroid_xy
    dot_back_clav = float(rel_clav @ d_back_xy)
    dot_back_rbak = float(rel_rbak @ d_back_xy)
    dot_right_clav = float(rel_clav @ d_right_xy)
    dot_right_rbak = float(rel_rbak @ d_right_xy)
    if dot_back_rbak <= dot_back_clav:
        raise CLAVRBAKValidationError(
            CLAVRBAK_RBAK_NOT_POSTERIOR_TO_CLAV,
            f"RBAK dot_back={dot_back_rbak:.2f} not > CLAV dot_back={dot_back_clav:.2f} (RBAK must be posterior to CLAV).",
        )
    if dot_right_rbak <= dot_right_clav:
        raise CLAVRBAKValidationError(
            CLAVRBAK_RBAK_NOT_RIGHT_OF_CLAV,
            f"RBAK dot_right={dot_right_rbak:.2f} not > CLAV dot_right={dot_right_clav:.2f} (RBAK must be right of CLAV).",
        )
    out.append((clav_idx, label_clav))
    out.append((rbak_idx, label_rbak))
    return out


def assign_strn_t10_arm4_after_clav_rbak(
    points_frame: np.ndarray,
    d_back_xy: np.ndarray,
    d_right_xy: np.ndarray,
    exclude_pt_indices: set[int],
    lsho_idx: int,
    rsho_idx: int,
    template: dict,
) -> list[tuple[int, str]]:
    """
    After CLAV/RBAK: from remaining points take the 6 highest Z. Two points with Y
    between LSHO and RSHO are STRN (anterior) and T10 (posterior), by direct A/P
    comparison (no centroid). The remaining 4 points: split by L/R (direct Y), then
    within each side higher Z = UPA, lower Z = ELB. All comparisons use point
    values directly (no centroid).
    """
    y_lsho = points_frame[lsho_idx, 1]
    y_rsho = points_frame[rsho_idx, 1]
    if not (np.isfinite(y_lsho) and np.isfinite(y_rsho)):
        return []
    y_min = min(y_lsho, y_rsho)
    y_max = max(y_lsho, y_rsho)
    remaining = [
        i for i in range(points_frame.shape[0])
        if i not in exclude_pt_indices and np.isfinite(points_frame[i, 2])
    ]
    if len(remaining) < 6:
        return []
    z_vals = np.array([points_frame[i, 2] for i in remaining])
    order_z = np.argsort(-z_vals)[:6]
    top6 = [remaining[int(k)] for k in order_z]
    # Template labels (preserve case from template)
    def _lab(name: str) -> str:
        return next((lab for lab in template if str(lab).strip().upper() == name.upper()), name)
    label_strn = _lab("STRN")
    label_t10 = _lab("T10")
    label_lupa = _lab("LUPA")
    label_rupa = _lab("RUPA")
    label_lelb = _lab("LELB")
    label_relb = _lab("RELB")

    # Two points with Y between shoulders → STRN (anterior), T10 (posterior); A/P by direct dot_back
    in_band = [i for i in top6 if y_min <= points_frame[i, 1] <= y_max]
    strn_t10_assignments: list[tuple[int, str]] = []
    if len(in_band) >= 2:
        dot_back = np.array([
            float(points_frame[i, :2] @ d_back_xy) for i in in_band
        ])
        order_ap = np.argsort(dot_back)
        anterior_pt = in_band[order_ap[0]]
        posterior_pt = in_band[order_ap[1]]
        strn_t10_assignments = [(anterior_pt, label_strn), (posterior_pt, label_t10)]
    elif len(in_band) == 1:
        strn_t10_assignments = [(in_band[0], label_strn)]

    # Remaining 4 from top6 (exclude the 2 used for STRN/T10)
    strn_t10_pts = {pi for pi, _ in strn_t10_assignments}
    arm4 = [i for i in top6 if i not in strn_t10_pts]
    if len(arm4) < 4:
        return strn_t10_assignments

    # L/R by direct Y: larger Y = left when d_right[1] < 0
    larger_y_is_left = d_right_xy[1] < 0
    y_arm = np.array([points_frame[i, 1] for i in arm4])
    order_y = np.argsort(-y_arm) if larger_y_is_left else np.argsort(y_arm)
    left_two = [arm4[order_y[0]], arm4[order_y[1]]]
    right_two = [arm4[order_y[2]], arm4[order_y[3]]]

    # Within each side: higher Z = UPA, lower Z = ELB (direct Z comparison)
    z_left = np.array([points_frame[i, 2] for i in left_two])
    order_z_left = np.argsort(-z_left)
    left_assignments = [
        (left_two[order_z_left[0]], label_lupa),
        (left_two[order_z_left[1]], label_lelb),
    ]
    z_right = np.array([points_frame[i, 2] for i in right_two])
    order_z_right = np.argsort(-z_right)
    right_assignments = [
        (right_two[order_z_right[0]], label_rupa),
        (right_two[order_z_right[1]], label_relb),
    ]

    return strn_t10_assignments + left_assignments + right_assignments


def assign_pelvis_arm12_after_strn_t10_arm(
    points_frame: np.ndarray,
    d_back_xy: np.ndarray,
    d_right_xy: np.ndarray,
    exclude_pt_indices: set[int],
    lsho_idx: int,
    rsho_idx: int,
    template: dict,
) -> list[tuple[int, str]]:
    """
    After STRN/T10/arm4: from remaining points take the 12 highest Z.
    - 4 points with Y between LSHO and RSHO -> pelvis: A/P sort -> anterior 2 = LASI,RASI (L/R by Y), posterior 2 = LPSI,RPSI (L/R by Y).
    - Remaining 8: left 4 (Y on left side of LSHO), right 4 (Y on right side of RSHO).
    - Left 4: Z max=LFRM, Z min=LFIN; middle 2 by A/P -> LWRA (anterior), LWRB (posterior).
    - Right 4: Z max=RFRM, Z min=RFIN; middle 2 by A/P -> RWRA, RWRB.
    All comparisons use point values directly (no centroid). Logs error codes on failure.
    """
    logger = logging.getLogger(__name__)
    y_lsho = float(points_frame[lsho_idx, 1])
    y_rsho = float(points_frame[rsho_idx, 1])
    if not (np.isfinite(y_lsho) and np.isfinite(y_rsho)):
        return []
    y_min = min(y_lsho, y_rsho)
    y_max = max(y_lsho, y_rsho)
    larger_y_is_left = d_right_xy[1] < 0

    remaining = [
        i for i in range(points_frame.shape[0])
        if i not in exclude_pt_indices and np.isfinite(points_frame[i, 2])
    ]
    if len(remaining) < 12:
        logger.warning(
            "PELVIS_ARM12_TOO_FEW_POINTS: Fewer than 12 remaining points for pelvis/arm candidates (got %d).",
            len(remaining),
        )
        return []

    z_vals = np.array([points_frame[i, 2] for i in remaining])
    order_z = np.argsort(-z_vals)[:12]
    top12 = [remaining[int(k)] for k in order_z]

    def _lab(name: str) -> str:
        return next((lab for lab in template if str(lab).strip().upper() == name.upper()), name)

    in_band = [i for i in top12 if y_min <= points_frame[i, 1] <= y_max]
    if len(in_band) != 4:
        logger.warning(
            "PELVIS_BAND_NOT_4: Expected 4 pelvis candidates in shoulder Y-band, got %d (y in [%.1f, %.1f]).",
            len(in_band), y_min, y_max,
        )
        return []

    dot_back = np.array([float(points_frame[i, :2] @ d_back_xy) for i in in_band])
    order_ap = np.argsort(dot_back)
    anterior_two = [in_band[order_ap[0]], in_band[order_ap[1]]]
    posterior_two = [in_band[order_ap[2]], in_band[order_ap[3]]]
    y_ant = np.array([points_frame[i, 1] for i in anterior_two])
    y_post = np.array([points_frame[i, 1] for i in posterior_two])
    if larger_y_is_left:
        order_lr_ant = np.argsort(-y_ant)
        order_lr_post = np.argsort(-y_post)
    else:
        order_lr_ant = np.argsort(y_ant)
        order_lr_post = np.argsort(y_post)
    label_lasi = _lab("LASI")
    label_rasi = _lab("RASI")
    label_lpsi = _lab("LPSI")
    label_rpsi = _lab("RPSI")
    pelvis_assignments = [
        (anterior_two[order_lr_ant[0]], label_lasi),
        (anterior_two[order_lr_ant[1]], label_rasi),
        (posterior_two[order_lr_post[0]], label_lpsi),
        (posterior_two[order_lr_post[1]], label_rpsi),
    ]

    pelvis_pts = {i for i in in_band}
    rest8 = [i for i in top12 if i not in pelvis_pts]
    if len(rest8) != 8:
        logger.warning(
            "PELVIS_ARM_MISSING_GROUP: Pelvis/arm partition incomplete: pelvis=4 rest=%d (expected 8).",
            len(rest8),
        )
        return pelvis_assignments

    y_rest = np.array([points_frame[i, 1] for i in rest8])
    if larger_y_is_left:
        order_y = np.argsort(-y_rest)
    else:
        order_y = np.argsort(y_rest)
    left4 = [rest8[order_y[0]], rest8[order_y[1]], rest8[order_y[2]], rest8[order_y[3]]]
    right4 = [rest8[order_y[4]], rest8[order_y[5]], rest8[order_y[6]], rest8[order_y[7]]]

    def assign_arm_side(four_pts: list[int], label_frm: str, label_wra: str, label_wrb: str, label_fin: str) -> list[tuple[int, str]]:
        if len(four_pts) < 4:
            logger.warning("ARM_SIDE_TOO_FEW_FOR_Z: Arm group has fewer than 4 points: got %d.", len(four_pts))
            return []
        z_vals = np.array([points_frame[i, 2] for i in four_pts])
        if np.any(~np.isfinite(z_vals)) or np.ptp(z_vals) == 0:
            logger.warning("ARM_SIDE_INVALID_Z_ORDER: Arm side points have identical or invalid Z.")
            return []
        order_z_desc = np.argsort(-z_vals)
        frm_pt = four_pts[order_z_desc[0]]
        fin_pt = four_pts[order_z_desc[3]]
        mid_two = [four_pts[order_z_desc[1]], four_pts[order_z_desc[2]]]
        dot_mid = np.array([float(points_frame[i, :2] @ d_back_xy) for i in mid_two])
        order_ap_mid = np.argsort(dot_mid)
        anterior_pt = mid_two[order_ap_mid[0]]
        posterior_pt = mid_two[order_ap_mid[1]]
        return [
            (frm_pt, label_frm),
            (anterior_pt, label_wra),
            (posterior_pt, label_wrb),
            (fin_pt, label_fin),
        ]

    label_lfrm = _lab("LFRM")
    label_lwra = _lab("LWRA")
    label_lwrb = _lab("LWRB")
    label_lfin = _lab("LFIN")
    label_rfrm = _lab("RFRM")
    label_rwra = _lab("RWRA")
    label_rwrb = _lab("RWRB")
    label_rfin = _lab("RFIN")
    left_assignments = assign_arm_side(left4, label_lfrm, label_lwra, label_lwrb, label_lfin)
    right_assignments = assign_arm_side(right4, label_rfrm, label_rwra, label_rwrb, label_rfin)
    if len(left_assignments) < 4 or len(right_assignments) < 4:
        return pelvis_assignments + left_assignments + right_assignments

    return pelvis_assignments + left_assignments + right_assignments


def assign_leg_foot12_after_pelvis_arm12(
    points_frame: np.ndarray,
    d_back_xy: np.ndarray,
    d_right_xy: np.ndarray,
    exclude_pt_indices: set[int],
    clav_idx: int,
    template: dict,
) -> list[tuple[int, str]]:
    """After pelvis/arm12: remaining points top 12 by Z. Split L/R by CLAV Y (left 6, right 6). Per side: Z desc -> THI,KNE,TIB,ANK; remaining 2 -> A/P HEE,TOE. Logs error codes on failure."""
    logger = logging.getLogger(__name__)
    if clav_idx < 0 or not np.isfinite(points_frame[clav_idx, 1]):
        logger.warning("LEG_FOOT12_CLAV_INVALID: CLAV marker position invalid or missing; cannot split left/right by CLAV.")
        return []
    clav_y = float(points_frame[clav_idx, 1])
    larger_y_is_left = d_right_xy[1] < 0

    remaining = [i for i in range(points_frame.shape[0]) if i not in exclude_pt_indices and np.isfinite(points_frame[i, 2])]
    if len(remaining) < 12:
        logger.warning("LEG_FOOT12_TOO_FEW_POINTS: Fewer than 12 remaining points for leg/foot candidates (got %d).", len(remaining))
        return []
    z_vals = np.array([points_frame[i, 2] for i in remaining])
    order_z = np.argsort(-z_vals)[:12]
    top12 = [remaining[int(k)] for k in order_z]

    y12 = np.array([points_frame[i, 1] for i in top12])
    if larger_y_is_left:
        left_mask = y12 >= clav_y
    else:
        left_mask = y12 <= clav_y
    n_left = int(np.sum(left_mask))
    n_right = 12 - n_left
    if n_left != 6 or n_right != 6:
        logger.warning(
            "LEG_FOOT12_LR_NOT_6_6: Remaining 12 points do not split into 6 left + 6 right by CLAV Y (got left=%d, right=%d). Using Y-sort fallback.",
            n_left, n_right,
        )
        order_y = np.argsort(-y12) if larger_y_is_left else np.argsort(y12)
        left6 = [top12[order_y[0]], top12[order_y[1]], top12[order_y[2]], top12[order_y[3]], top12[order_y[4]], top12[order_y[5]]]
        right6 = [top12[order_y[6]], top12[order_y[7]], top12[order_y[8]], top12[order_y[9]], top12[order_y[10]], top12[order_y[11]]]
    else:
        left6 = [top12[i] for i in range(12) if left_mask[i]]
        right6 = [top12[i] for i in range(12) if not left_mask[i]]

    def _lab(name: str) -> str:
        return next((lab for lab in template if str(lab).strip().upper() == name.upper()), name)

    def assign_side_six(six_pts: list[int], label_thi: str, label_kne: str, label_tib: str, label_ank: str, label_hee: str, label_toe: str) -> list[tuple[int, str]]:
        if len(six_pts) < 6:
            logger.warning("LEG_FOOT12_SIDE_TOO_FEW: Left or right side has fewer than 6 points (got %d).", len(six_pts))
            return []
        z_side = np.array([points_frame[i, 2] for i in six_pts])
        if np.any(~np.isfinite(z_side)):
            logger.warning("LEG_FOOT12_SIDE_Z_INVALID: Leg side has invalid Z; cannot assign THI, KNE, TIB, ANK in order.")
            return []
        order_z_side = np.argsort(-z_side)
        out = [
            (six_pts[order_z_side[0]], label_thi),
            (six_pts[order_z_side[1]], label_kne),
            (six_pts[order_z_side[2]], label_tib),
            (six_pts[order_z_side[3]], label_ank),
        ]
        foot_two = [six_pts[order_z_side[4]], six_pts[order_z_side[5]]]
        if len(foot_two) != 2:
            logger.warning("LEG_FOOT12_FOOT_NOT_2: Expected 2 foot points remaining on one side, got %d.", len(foot_two))
            return out
        dot_back = np.array([float(points_frame[i, :2] @ d_back_xy) for i in foot_two])
        if not np.all(np.isfinite(dot_back)) or dot_back[0] == dot_back[1]:
            logger.warning("LEG_FOOT12_FOOT_AP_INVALID: Foot side has invalid or identical A/P; cannot assign HEE/TOE.")
            return out
        order_ap = np.argsort(dot_back)
        out.append((foot_two[order_ap[0]], label_hee))
        out.append((foot_two[order_ap[1]], label_toe))
        return out

    left_out = assign_side_six(left6, _lab("LTHI"), _lab("LKNE"), _lab("LTIB"), _lab("LANK"), _lab("LHEE"), _lab("LTOE"))
    right_out = assign_side_six(right6, _lab("RTHI"), _lab("RKNE"), _lab("RTIB"), _lab("RANK"), _lab("RHEE"), _lab("RTOE"))
    if len(left_out) < 6 or len(right_out) < 6:
        return left_out + right_out
    return left_out + right_out


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


def best_frame_by_foot_stability(
    points: np.ndarray,
    *,
    middle_start: float = DEFAULT_MIDDLE_START,
    middle_end: float = DEFAULT_MIDDLE_END,
    visibility_min: float = SCREENING_BEST_FRAME_VISIBILITY_MIN,
    n_smallest_z: int = SCREENING_FOOT_PROXY_N_SMALLEST_Z,
    fallback_fn=None,
) -> int:
    """
    Step 5: Choose frame in middle with ≥visibility_min valid points and minimal |v_foot|.

    Foot-height proxy = mean of the n_smallest_z smallest Z values per frame (feet on ground).
    v_foot(t) = central difference of this proxy; minimize |v_foot| for both feet on ground.

    Parameters
    ----------
    points : (n_frames, n_points, 3)
    middle_start, middle_end : fraction of frames for middle portion
    visibility_min : require this fraction of points valid in the frame (e.g. 0.95)
    n_smallest_z : number of smallest Z values to average for foot proxy
    fallback_fn : callable(points, **kwargs) -> int; used when no frame meets visibility_min

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

    # Per-frame: fraction valid, and foot proxy (mean of n_smallest_z smallest Z)
    valid_per_frame = np.isfinite(points).all(axis=2)  # (n_frames, n_points)
    n_valid_per_frame = np.sum(valid_per_frame, axis=1)
    frac_valid = n_valid_per_frame / n_points if n_points > 0 else np.zeros(n_frames)

    z_foot = np.full(n_frames, np.nan)
    for f in range(n_frames):
        z_vals = points[f, :, 2].copy()
        z_vals[~np.isfinite(z_vals)] = np.nan
        finite = np.isfinite(z_vals)
        if np.sum(finite) >= n_smallest_z:
            smallest = np.partition(z_vals[finite], min(n_smallest_z - 1, np.sum(finite) - 1))[:n_smallest_z]
            z_foot[f] = np.mean(smallest)

    # Central difference for v_foot; at boundaries use one-sided or inf
    v_foot = np.full(n_frames, np.inf)
    for f in range(1, n_frames - 1):
        if np.isfinite(z_foot[f - 1]) and np.isfinite(z_foot[f + 1]):
            v_foot[f] = (z_foot[f + 1] - z_foot[f - 1]) / 2.0

    # Restrict to candidates with visibility >= visibility_min and finite |v_foot|
    good = [
        f for f in candidates
        if frac_valid[f] >= visibility_min and np.isfinite(v_foot[f])
    ]
    if good:
        return int(min(good, key=lambda f: np.abs(v_foot[f])))
    if fallback_fn is not None:
        return fallback_fn(points, middle_start=middle_start, middle_end=middle_end)
    # Last resort: middle frame
    return int(candidates[len(candidates) // 2])


def best_frame_for_matching(
    points: np.ndarray,
    residual: np.ndarray | None,
    *,
    middle_start: float = DEFAULT_MIDDLE_START,
    middle_end: float = DEFAULT_MIDDLE_END,
) -> int:
    """
    Choose frame index in the middle portion with best quality (lowest mean residual or most valid).

    Parameters
    ----------
    points : (n_frames, n_points, 3)
    residual : (n_frames, n_points) or None
    middle_start, middle_end : fraction of frames for middle portion

    Returns
    -------
    frame_idx : int
    """
    n_frames = points.shape[0]
    start = int(n_frames * middle_start)
    end = int(n_frames * middle_end)
    start = max(0, min(start, n_frames - 1))
    end = max(start + 1, min(end, n_frames))
    candidates = list(range(start, end))
    if residual is not None:
        def score(f):
            s = np.nanmean(residual[f, :])
            return s if np.isfinite(s) else np.inf
        best = min(candidates, key=score)
        return best
    # Fallback: frame with most valid markers
    n_valid = np.sum(np.isfinite(points).all(axis=2), axis=1)
    best = max(candidates, key=lambda f: n_valid[f])
    return best


def match_markers_to_template(
    points_frame: np.ndarray,
    template: dict[str, np.ndarray],
    *,
    use_hungarian: bool = True,
    max_match_distance: float | None = None,
) -> list[tuple[int, str]]:
    """
    Match unlabeled points to template (lab or pelvis frame).

    Parameters
    ----------
    points_frame : (n_points, 3) one frame
    template : dict label -> (3,) position
    use_hungarian : if True, use global optimal assignment (min total distance); else greedy
    max_match_distance : if set, reject assignment when distance > this (mm)

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
                    cost[pi, li] = np.linalg.norm(points_frame[pi] - positions[li])
            # Hungarian: assign labels to points (minimize total cost). Rows=points, cols=labels.
            row_ind, col_ind = linear_sum_assignment(cost)
            assignments = []
            for r, c in zip(row_ind, col_ind):
                d = cost[r, c]
                if not np.isfinite(d) or (max_match_distance is not None and d > max_match_distance):
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
            if d < best_dist and (max_match_distance is None or d <= max_match_distance):
                best_dist = d
                best_label = label
        if best_label is not None:
            used_label.add(best_label)
            assignments.append((pi, best_label))
    return assignments


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
) -> int:
    """
    Choose the frame in the middle range where trunk assignment cost is minimum.

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
        )

    def trunk_cost(f: int) -> float:
        asgn = match_markers_to_template(
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

    best = min(candidates, key=trunk_cost)
    if np.isinf(trunk_cost(best)):
        return best_frame_for_matching(
            points_dynamic, residual_dynamic,
            middle_start=middle_start, middle_end=middle_end,
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


def _match_within_z_bands(
    points_frame: np.ndarray,
    template: dict[str, np.ndarray],
    label_to_band: dict[str, int],
    point_bands: np.ndarray,
    *,
    use_hungarian: bool = True,
    max_match_distance: float | None = None,
    exclude_pt_indices: set[int] | None = None,
    exclude_labels: set[str] | None = None,
) -> list[tuple[int, str]]:
    """
    Match points to template labels within each Z-band. Points and labels in the same
    band are matched (Hungarian or greedy). Returns list of (point_idx, label).
    Optionally exclude some point indices and labels (e.g. head already assigned).
    """
    from .constants import N_Z_BANDS

    exclude_pt = exclude_pt_indices or set()
    exclude_lab = exclude_labels or set()
    exclude_lab_upper = {str(l).strip().upper() for l in exclude_lab}
    assignments: list[tuple[int, str]] = []
    for b in range(N_Z_BANDS):
        pt_idx = np.array([i for i in np.where(point_bands == b)[0] if i not in exclude_pt])
        labels_b = [
            lab for lab in template
            if label_to_band.get(lab, -1) == b and str(lab).strip().upper() not in exclude_lab_upper
        ]
        if len(pt_idx) == 0 or len(labels_b) == 0:
            continue
        sub_pts = points_frame[pt_idx]
        sub_tpl = {lab: template[lab] for lab in labels_b}
        asgn = match_markers_to_template(
            sub_pts,
            sub_tpl,
            use_hungarian=use_hungarian,
            max_match_distance=max_match_distance,
        )
        for local_i, lab in asgn:
            assignments.append((int(pt_idx[local_i]), lab))
    return assignments


def label_body_markers(
    points_dynamic: np.ndarray,
    template: dict[str, np.ndarray],
    *,
    residual_dynamic: np.ndarray | None = None,
    middle_start: float = DEFAULT_MIDDLE_START,
    middle_end: float = DEFAULT_MIDDLE_END,
    use_hungarian: bool = True,
    max_match_distance: float | None = None,
    max_propagation_distance: float | None = None,
    label_to_band: dict[str, int] | None = None,
    band_sizes: tuple[int, ...] | None = None,
    segment_geometry: list | None = None,
    use_screening_best_frame: bool = True,
    walking_direction_x: float | None = None,
    d_back_xy: np.ndarray | None = None,
    d_right_xy: np.ndarray | None = None,
) -> tuple[list[str], np.ndarray]:
    """
    Label body markers in dynamic trial using template and temporal propagation.

    When label_to_band is provided (from static Z-band by rank), matching is done
    within each Z-band: dynamic points are assigned to bands by Z rank, then matched
    to template labels in the same band. segment_geometry is for arm/hand labeling
    and gap filling (used in a later step).

    Parameters
    ----------
    points_dynamic : (n_frames, n_points, 3)
    template : label -> (3,) mean position
    residual_dynamic : optional (n_frames, n_points)
    middle_start, middle_end : frame range for best frame
    use_hungarian, max_match_distance, max_propagation_distance : as before
    label_to_band : optional dict from static Z-band by rank (label -> band index)
    band_sizes : when using no-arm-hand z-band, pass Z_BAND_SIZES_NO_ARM_HAND for point band assignment
    segment_geometry : optional list from static segment geometry (arm/hand, gap fill)
    use_screening_best_frame : if True (default), use Step 5: middle range, ≥95%% visibility,
        minimal |v_foot| (foot proxy = mean of 6 smallest Z); fallback to trunk cost or residual.
    walking_direction_x : optional; used only for head assignment when d_back_xy/d_right_xy set.
    d_back_xy, d_right_xy : optional (2,) xy vectors from get_lr_ap_axes_from_walking. When both
        provided, head markers (LFHD, RFHD, LBHD, RBHD) are assigned at best frame by top 4 Z and
        L/R (d_right), A/P (d_back); remaining points/labels then use Z-band matching.

    Returns
    -------
    labels_out : list of str, length n_points
    label_per_frame : (n_frames, n_points) object array
    best_frame : int, 0-based frame index used for axis-based assignment
    """
    n_frames, n_points, _ = points_dynamic.shape
    trunk_labels = _trunk_labels_in_template(template)
    if use_screening_best_frame:
        def _fallback(pts: np.ndarray, **kw) -> int:
            if len(trunk_labels) >= 3:
                return best_frame_by_trunk_cost(
                    pts, template, residual_dynamic,
                    use_hungarian=use_hungarian,
                    max_match_distance=max_match_distance,
                    trunk_labels=trunk_labels,
                    **kw,
                )
            return best_frame_for_matching(pts, residual_dynamic, **kw)
        best_f = best_frame_by_foot_stability(
            points_dynamic,
            middle_start=middle_start,
            middle_end=middle_end,
            fallback_fn=_fallback,
        )
    elif len(trunk_labels) >= 3:
        best_f = best_frame_by_trunk_cost(
            points_dynamic, template, residual_dynamic,
            middle_start=middle_start, middle_end=middle_end,
            use_hungarian=use_hungarian,
            max_match_distance=max_match_distance,
            trunk_labels=trunk_labels,
        )
    else:
        best_f = best_frame_for_matching(
            points_dynamic, residual_dynamic,
            middle_start=middle_start, middle_end=middle_end,
        )

    if label_to_band:
        point_bands = static_geometry.assign_point_bands_by_z_rank(
            points_dynamic[best_f],
            band_sizes=band_sizes,
        )
        # Head markers: assign by top 4 Z and L/R (d_right), A/P (d_back) when axes provided
        head_assignments: list[tuple[int, str]] = []
        head_pt_set: set[int] = set()
        use_head_by_z = (
            d_back_xy is not None
            and d_right_xy is not None
            and len(d_back_xy) == 2
            and len(d_right_xy) == 2
            and any(str(lab).strip().upper() in HEAD_MARKERS_SET for lab in template)
        )
        if use_head_by_z:
            head_labels_ordered = [
                lab for m in HEAD_MARKERS
                for lab in template
                if str(lab).strip().upper() == m
            ]
            head_assignments = assign_head_markers_by_top4_z(
                points_dynamic[best_f],
                d_back_xy,
                d_right_xy,
                head_labels=head_labels_ordered if head_labels_ordered else None,
            )
            head_pt_set = {pi for pi, _ in head_assignments}
        # C7 and shoulders: among remaining points, top 1 Z = C7, next 2 Z = LSHO/RSHO (L/R by d_right)
        c7_shoulder_assignments: list[tuple[int, str]] = []
        priority_pt_set = set(head_pt_set)
        use_c7_shoulder_by_z = (
            use_head_by_z
            and d_right_xy is not None
            and len(d_right_xy) == 2
            and any(str(lab).strip().upper() in C7_SHOULDER_MARKERS_SET for lab in template)
        )
        if use_c7_shoulder_by_z:
            c7_shoulder_assignments = assign_c7_shoulders_after_head(
                points_dynamic[best_f],
                d_right_xy,
                head_pt_set,
                template,
            )
            priority_pt_set = head_pt_set | {pi for pi, _ in c7_shoulder_assignments}
        # CLAV and RBAK: next 2 Z among points with Y between shoulders ± 15 mm; anterior = CLAV, posterior = RBAK
        clav_rbak_assignments: list[tuple[int, str]] = []
        use_clav_rbak_by_z = (
            use_c7_shoulder_by_z
            and len(c7_shoulder_assignments) >= 3
            and d_back_xy is not None
            and len(d_back_xy) == 2
            and any(str(lab).strip().upper() in CLAV_RBAK_MARKERS_SET for lab in template)
        )
        if use_clav_rbak_by_z:
            lsho_idx = next(
                (pi for pi, lab in c7_shoulder_assignments if str(lab).strip().upper() == "LSHO"),
                -1,
            )
            rsho_idx = next(
                (pi for pi, lab in c7_shoulder_assignments if str(lab).strip().upper() == "RSHO"),
                -1,
            )
            if lsho_idx >= 0 and rsho_idx >= 0:
                clav_rbak_assignments = assign_clav_rbak_after_c7_shoulders(
                    points_dynamic[best_f],
                    d_back_xy,
                    d_right_xy,
                    priority_pt_set,
                    lsho_idx,
                    rsho_idx,
                    template,
                )
                priority_pt_set = priority_pt_set | {pi for pi, _ in clav_rbak_assignments}
        # STRN, T10, LUPA, RUPA, LELB, RELB: next 6 Z; Y in shoulder band → STRN/T10 (A/P); rest L/R by Y, then Z = UPA/ELB
        strn_t10_arm_assignments: list[tuple[int, str]] = []
        use_strn_t10_arm = (
            use_clav_rbak_by_z
            and len(clav_rbak_assignments) >= 2
            and d_back_xy is not None
            and len(d_back_xy) == 2
            and d_right_xy is not None
            and len(d_right_xy) == 2
            and any(str(lab).strip().upper() in STRN_T10_ARM_MARKERS_SET for lab in template)
        )
        if use_strn_t10_arm and lsho_idx >= 0 and rsho_idx >= 0:
            strn_t10_arm_assignments = assign_strn_t10_arm4_after_clav_rbak(
                points_dynamic[best_f],
                d_back_xy,
                d_right_xy,
                priority_pt_set,
                lsho_idx,
                rsho_idx,
                template,
            )
            if strn_t10_arm_assignments:
                priority_pt_set = priority_pt_set | {pi for pi, _ in strn_t10_arm_assignments}
        # Pelvis (LASI,RASI,LPSI,RPSI) + arm/hand (LFRM,LWRA,LWRB,LFIN, RFRM,RWRA,RWRB,RFIN): next 12 Z; Y band -> pelvis A/P then L/R; rest left/right 4 each -> Z and A/P
        pelvis_arm12_assignments: list[tuple[int, str]] = []
        use_pelvis_arm12 = (
            use_strn_t10_arm
            and len(strn_t10_arm_assignments) >= 6
            and d_back_xy is not None
            and len(d_back_xy) == 2
            and d_right_xy is not None
            and len(d_right_xy) == 2
            and any(str(lab).strip().upper() in PELVIS_ARM12_MARKERS_SET for lab in template)
        )
        if use_pelvis_arm12 and lsho_idx >= 0 and rsho_idx >= 0:
            pelvis_arm12_assignments = assign_pelvis_arm12_after_strn_t10_arm(
                points_dynamic[best_f],
                d_back_xy,
                d_right_xy,
                priority_pt_set,
                lsho_idx,
                rsho_idx,
                template,
            )
            if pelvis_arm12_assignments:
                priority_pt_set = priority_pt_set | {pi for pi, _ in pelvis_arm12_assignments}
        # Leg/foot (12): remaining points top 12 by Z; split L/R by CLAV Y; per-side Z order THI,KNE,TIB,ANK; remaining 2 per side A/P -> HEE,TOE
        leg_foot12_assignments: list[tuple[int, str]] = []
        clav_idx = next((pi for pi, lab in clav_rbak_assignments if str(lab).strip().upper() == "CLAV"), -1)
        use_leg_foot12 = (
            use_pelvis_arm12
            and len(pelvis_arm12_assignments) >= 12
            and clav_idx >= 0
            and d_back_xy is not None
            and len(d_back_xy) == 2
            and d_right_xy is not None
            and len(d_right_xy) == 2
            and any(str(lab).strip().upper() in LEG_FOOT12_MARKERS_SET for lab in template)
        )
        if use_leg_foot12:
            leg_foot12_assignments = assign_leg_foot12_after_pelvis_arm12(
                points_dynamic[best_f],
                d_back_xy,
                d_right_xy,
                priority_pt_set,
                clav_idx,
                template,
            )
            if leg_foot12_assignments:
                priority_pt_set = priority_pt_set | {pi for pi, _ in leg_foot12_assignments}
        exclude_labels_z = HEAD_MARKERS_SET
        if c7_shoulder_assignments:
            exclude_labels_z = HEAD_MARKERS_SET | C7_SHOULDER_MARKERS_SET
        if clav_rbak_assignments:
            exclude_labels_z = exclude_labels_z | CLAV_RBAK_MARKERS_SET
        if strn_t10_arm_assignments:
            exclude_labels_z = exclude_labels_z | STRN_T10_ARM_MARKERS_SET
        if pelvis_arm12_assignments:
            exclude_labels_z = exclude_labels_z | PELVIS_ARM12_MARKERS_SET
        if leg_foot12_assignments:
            exclude_labels_z = exclude_labels_z | LEG_FOOT12_MARKERS_SET
        band_assignments = _match_within_z_bands(
            points_dynamic[best_f],
            template,
            label_to_band,
            point_bands,
            use_hungarian=use_hungarian,
            max_match_distance=max_match_distance,
            exclude_pt_indices=priority_pt_set if (use_head_by_z or c7_shoulder_assignments or clav_rbak_assignments or strn_t10_arm_assignments or pelvis_arm12_assignments or leg_foot12_assignments) else None,
            exclude_labels=exclude_labels_z if (use_head_by_z or c7_shoulder_assignments or clav_rbak_assignments or strn_t10_arm_assignments or pelvis_arm12_assignments or leg_foot12_assignments) else None,
        )
        assignments = head_assignments + c7_shoulder_assignments + clav_rbak_assignments + strn_t10_arm_assignments + pelvis_arm12_assignments + leg_foot12_assignments + band_assignments
        # Preserve axis/Z-based assignments so Procrustes+Hungarian do not overwrite head, C7, shoulders, CLAV, RBAK, STRN/T10/arm4, pelvis/arm12, leg/foot12
        axis_based_assignments = head_assignments + c7_shoulder_assignments + clav_rbak_assignments + strn_t10_arm_assignments + pelvis_arm12_assignments + leg_foot12_assignments
        # If template has more labels than label_to_band (e.g. 39 vs 27), match remaining points to arm/hand
        assigned_pts = {pi for pi, _ in assignments}
        assigned_labs = {str(lab).strip().upper() for _, lab in assignments}
        unassigned_pts = [pi for pi in range(n_points) if pi not in assigned_pts]
        unassigned_labels = [
            lab for lab in template
            if str(lab).strip().upper() not in assigned_labs
        ]
        if unassigned_pts and unassigned_labels:
            sub_pts = points_dynamic[best_f][unassigned_pts]
            sub_tpl = {lab: template[lab] for lab in unassigned_labels}
            asgn = match_markers_to_template(
                sub_pts,
                sub_tpl,
                use_hungarian=use_hungarian,
                max_match_distance=max_match_distance,
            )
            for local_i, lab in asgn:
                assignments.append((int(unassigned_pts[local_i]), lab))
    else:
        axis_based_assignments = []
    if not label_to_band:
        assignments = match_markers_to_template(
            points_dynamic[best_f],
            template,
            use_hungarian=use_hungarian,
            max_match_distance=max_match_distance,
        )
    # Align template using trunk-only Procrustes, then re-match — unless alignment quality is bad (fallback).
    # Do not overwrite axis/Z-based assignments (head, C7, shoulders, CLAV, RBAK) with Hungarian re-match.
    _TRUNK_RMS_MAX_MM = 150.0  # reject Procrustes if trunk RMS after align exceeds this
    if len(trunk_labels) >= 3 and assignments:
        initial_rms = _trunk_rms(
            points_dynamic[best_f], template, assignments, trunk_labels,
        )
        R, t = procrustes_fit_trunk(
            points_dynamic[best_f], template, assignments, trunk_labels,
        )
        template_aligned = _apply_rigid_to_template(template, R, t)
        if axis_based_assignments:
            # Preserve head, C7, shoulders, CLAV, RBAK; re-match only remaining points to remaining labels
            axis_based_pt_set = {pi for pi, _ in axis_based_assignments}
            axis_based_lab_set = {str(lab).strip().upper() for _, lab in axis_based_assignments}
            remaining_pts = [i for i in range(n_points) if i not in axis_based_pt_set]
            remaining_labs = [
                lab for lab in template
                if str(lab).strip().upper() not in axis_based_lab_set
            ]
            if remaining_pts and remaining_labs:
                sub_pts = points_dynamic[best_f][remaining_pts]
                sub_tpl = {lab: template_aligned[lab] for lab in remaining_labs}
                asgn_rest = match_markers_to_template(
                    sub_pts,
                    sub_tpl,
                    use_hungarian=use_hungarian,
                    max_match_distance=max_match_distance,
                )
                assignments_aligned_rest = [
                    (int(remaining_pts[local_i]), lab) for local_i, lab in asgn_rest
                ]
            else:
                assignments_aligned_rest = []
            assignments_new = list(axis_based_assignments) + assignments_aligned_rest
            aligned_rms = _trunk_rms(
                points_dynamic[best_f], template_aligned, assignments_new, trunk_labels,
            )
            if aligned_rms < _TRUNK_RMS_MAX_MM and aligned_rms < initial_rms:
                assignments = assignments_new
        else:
            assignments_aligned = match_markers_to_template(
                points_dynamic[best_f],
                template_aligned,
                use_hungarian=use_hungarian,
                max_match_distance=max_match_distance,
            )
            aligned_rms = _trunk_rms(
                points_dynamic[best_f], template_aligned, assignments_aligned, trunk_labels,
            )
            if aligned_rms < _TRUNK_RMS_MAX_MM and aligned_rms < initial_rms:
                assignments = assignments_aligned
        # else: keep original assignments (trunk matching failed or did not help)
    if not assignments:
        return [""] * n_points, np.empty((n_frames, n_points), dtype=object), best_f
    label_per_frame = propagate_labels_temporal(
        points_dynamic,
        assignments,
        best_f,
        max_propagation_distance=max_propagation_distance,
    )
    labels_out = [label_per_frame[best_f, pi] for pi in range(n_points)]
    return labels_out, label_per_frame, best_f
