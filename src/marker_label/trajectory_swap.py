"""
Stage 0: Trajectory-based marker swap detection and correction.

Phase 1 scope (current):
    - **Stage 0.0** Velocity discontinuities: frame-to-frame jumps paired with
      sustained displacement vs surrounding windows (swap / tracking glitch hint).
    - Bilateral consistency: PSI markers and hand markers (LWRA, RWRA, etc.)
      where occlusion-driven swaps are most common.
    - Anatomical sanity: wrist markers (catches cross-segment swaps like
      RWRA at pelvis position), plus basic sanity for foot and head.

Excluded from Phase 1:
    - LASI/RASI bilateral check: these define the ML axis convention used by
      this stage. ASI mislabeling is handled by Stage 1 (envelope filter)
      and Stage 2 (pelvis combinatorial swap).
    - Lower body bilateral pairs (LKNE/RKNE, LANK/RANK, etc.): swaps are
      rare in walking trials due to consistent left-right separation, and
      trim_and_combine already validates lower-body markers. Including
      these would risk false positives during phases like obstacle crossing
      where legs cross close to body midline.

Future phases (after Phase 1 validation):
    - Phase 2: Add lower body bilateral pairs and additional anatomical
      constraints if Phase 1 results are stable.
    - Phase 3: Add LASI/RASI check via external reference (not bilateral
      ML check). Requires careful handling to avoid breaking the ML axis
      convention.

Configuration:
    Users can extend bilateral_pairs and anatomical_constraints via config
    if Phase 1 default is too conservative for their use case.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_TRAJECTORY_SWAP_CONFIG: dict[str, Any] = {
    "enable_trajectory_swap_detection": True,
    "bilateral_pairs": [
        ("LPSI", "RPSI"),
        ("LWRA", "RWRA"),
        ("LWRB", "RWRB"),
        ("LFIN", "RFIN"),
    ],
    "bilateral_min_separation_mm": 30.0,
    "bilateral_min_consecutive_frames": 5,
    "anatomical_constraints": [
        ("RWRA", "RFIN", "distance_max", 200),
        ("LWRA", "LFIN", "distance_max", 200),
        ("RWRA", "pelvis_center", "distance_min", 250.0),
        ("LWRA", "pelvis_center", "distance_min", 250.0),
        ("RWRA", "RELB", "distance_range", (200, 350)),
        ("LWRA", "LELB", "distance_range", (200, 350)),
        ("RWRA", "RWRB", "distance_max", 100),
        ("LWRA", "LWRB", "distance_max", 100),
        ("LTOE", "pelvis_center", "distance_min", 400.0),
        ("RTOE", "pelvis_center", "distance_min", 400.0),
        ("LFHD", "pelvis_center", "distance_min", 400.0),
        ("RFHD", "pelvis_center", "distance_min", 400.0),
    ],
    "anatomical_violation_threshold": 2,
    "anatomical_min_consecutive_frames": 5,
    "min_swap_interval_length": 5,
    "merge_intervals_gap_max": 3,
    "donor_match_max_mean_distance_mm": 100.0,
    "donor_match_min_visibility_pct": 0.3,
    "apply_correction_min_confidence": "MEDIUM",
    # Stage 0.0 — velocity discontinuities (frame-to-frame jumps)
    "enable_velocity_discontinuity_detection": True,
    "velocity_jump_threshold_mm": 100.0,
    "velocity_interval_min_displacement_mm": 50.0,
    "velocity_interval_min_frames": 3,
}


def _is_unlabeled_stem(stem: str) -> bool:
    s = str(stem).strip()
    if s.startswith("*"):
        return True
    if re.fullmatch(r"\d+", s):
        return True
    return False


def find_consecutive_intervals(bool_array: Sequence[bool], min_length: int) -> list[tuple[int, int]]:
    """``(start_row, end_row)`` inclusive for each run of True with length >= ``min_length``."""
    intervals: list[tuple[int, int]] = []
    in_run = False
    run_start = 0
    arr = list(bool_array)
    n = len(arr)
    k = int(min_length)
    if k < 1:
        return intervals
    for i, val in enumerate(arr):
        if val and not in_run:
            in_run = True
            run_start = i
        elif not val and in_run:
            if i - run_start >= k:
                intervals.append((run_start, i - 1))
            in_run = False
    if in_run and n - run_start >= k:
        intervals.append((run_start, n - 1))
    return intervals


def _lr_reference_sign_ml(
    markers: Mapping[str, np.ndarray],
    f: int,
    ml_axis: int,
    fallback_sign: int,
) -> int:
    """+1 / -1 for expected sign(left - right) in ML when LASI/RASI exist at ``f``."""
    la = markers.get("LASI")
    ra = markers.get("RASI")
    if la is None or ra is None:
        return int(np.sign(fallback_sign)) or 1
    if not (np.isfinite(la[f]).all() and np.isfinite(ra[f]).all()):
        return int(np.sign(fallback_sign)) or 1
    d = float(la[f, ml_axis] - ra[f, ml_axis])
    if abs(d) < 1e-6:
        return int(np.sign(fallback_sign)) or 1
    return int(np.sign(d))


def detect_bilateral_violations(
    markers: Mapping[str, np.ndarray],
    bilateral_pairs: Sequence[tuple[str, str]],
    ml_axis: int,
    left_ml_sign: int,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    min_sep = float(config["bilateral_min_separation_mm"])
    min_run = int(config["bilateral_min_consecutive_frames"])
    n_frames = len(next(iter(markers.values())))

    for left_m, right_m in bilateral_pairs:
        if left_m not in markers or right_m not in markers:
            continue
        L = markers[left_m]
        R = markers[right_m]
        flags = np.zeros(n_frames, dtype=bool)
        for f in range(n_frames):
            if not (np.isfinite(L[f]).all() and np.isfinite(R[f]).all()):
                continue
            ref_sign = _lr_reference_sign_ml(markers, f, ml_axis, left_ml_sign)
            ml_diff = float(L[f, ml_axis] - R[f, ml_axis])
            if abs(ml_diff) < min_sep:
                continue
            if int(np.sign(ml_diff)) != ref_sign:
                flags[f] = True
        for s_row, e_row in find_consecutive_intervals(flags.tolist(), min_run):
            violations.append(
                {
                    "type": "bilateral_swap",
                    "pair": (left_m, right_m),
                    "start_row": int(s_row),
                    "end_row": int(e_row),
                    "duration": int(e_row - s_row + 1),
                    "evidence": f"ML separation sign mismatch vs LASI–RASI for {e_row - s_row + 1} consecutive rows",
                }
            )
    return violations


def get_marker_or_special(
    markers: Mapping[str, np.ndarray],
    name: str,
    f: int,
) -> np.ndarray | None:
    if name == "pelvis_center":
        lasi = markers.get("LASI")
        rasi = markers.get("RASI")
        if lasi is None or rasi is None:
            return None
        if not (np.isfinite(lasi[f]).all() and np.isfinite(rasi[f]).all()):
            return None
        return 0.5 * (lasi[f] + rasi[f])
    if name not in markers:
        return None
    p = markers[name][f]
    if not np.isfinite(p).all():
        return None
    return p


def check_constraint(
    markers: Mapping[str, np.ndarray],
    constraint: tuple[Any, ...],
    f: int,
) -> bool | None:
    marker_a_name, marker_b_name, ctype, value = constraint[0], constraint[1], constraint[2], constraint[3]
    pos_a = get_marker_or_special(markers, str(marker_a_name), f)
    pos_b = get_marker_or_special(markers, str(marker_b_name), f)
    if pos_a is None or pos_b is None:
        return None
    distance = float(np.linalg.norm(pos_a - pos_b))
    if ctype == "distance_max":
        return distance > float(value)
    if ctype == "distance_min":
        return distance < float(value)
    if ctype == "distance_range":
        lo, hi = float(value[0]), float(value[1])
        return distance < lo or distance > hi
    return None


def detect_anatomical_violations(
    markers: Mapping[str, np.ndarray],
    constraints: Sequence[tuple[Any, ...]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    n_frames = len(next(iter(markers.values())))
    threshold = int(config["anatomical_violation_threshold"])
    min_run = int(config["anatomical_min_consecutive_frames"])
    constraints_by_marker: dict[str, list[tuple[Any, ...]]] = {}
    for c in constraints:
        constraints_by_marker.setdefault(str(c[0]), []).append(tuple(c))
    violations: list[dict[str, Any]] = []

    for marker, marker_constraints in constraints_by_marker.items():
        if marker not in markers:
            continue
        violation_flags = np.zeros(n_frames, dtype=bool)
        for f in range(n_frames):
            count = 0
            for c in marker_constraints:
                r = check_constraint(markers, c, f)
                if r is True:
                    count += 1
            if count >= threshold:
                violation_flags[f] = True
        for s_row, e_row in find_consecutive_intervals(violation_flags.tolist(), min_run):
            violated_summary: list[dict[str, Any]] = []
            for c in marker_constraints:
                key = f"{c[0]}-{c[1]} {c[2]} {c[3]}"
                vc = sum(
                    1
                    for ff in range(s_row, e_row + 1)
                    if check_constraint(markers, c, ff) is True
                )
                if vc > 0:
                    violated_summary.append({"constraint": key, "violated_frames": int(vc)})
            violations.append(
                {
                    "type": "anatomical",
                    "marker": marker,
                    "start_row": int(s_row),
                    "end_row": int(e_row),
                    "duration": int(e_row - s_row + 1),
                    "violated_constraints": violated_summary,
                    "evidence": f"{len(violated_summary)} constraints violated",
                }
            )
    return violations


def detect_velocity_discontinuities(
    markers: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """
    Detect frame-to-frame position jumps above ``velocity_jump_threshold_mm``.

    Pairs consecutive large jumps as jump-in / jump-out when the plateau between
    shows sustained displacement vs surrounding windows; otherwise falls back
    to single-jump handling (trial end or advance one jump).
    """
    thr = float(config["velocity_jump_threshold_mm"])
    min_disp = float(config["velocity_interval_min_displacement_mm"])
    min_frames_iv = int(config["velocity_interval_min_frames"])
    velocity_intervals: list[dict[str, Any]] = []

    for marker_name, traj in markers.items():
        if traj.ndim != 2 or traj.shape[1] != 3:
            continue
        n_frames = int(traj.shape[0])
        jumps: list[tuple[int, float]] = []
        for f in range(1, n_frames):
            if not (np.isfinite(traj[f]).all() and np.isfinite(traj[f - 1]).all()):
                continue
            jump_mag = float(np.linalg.norm(traj[f] - traj[f - 1]))
            if jump_mag > thr:
                jumps.append((f, jump_mag))

        i = 0
        while i < len(jumps):
            f_in, mag_in = jumps[i]

            if i + 1 < len(jumps):
                f_out, mag_out = jumps[i + 1]
                if f_out <= f_in:
                    i += 1
                    continue
                gap_finite = True
                for k in range(f_in, f_out):
                    if not np.isfinite(traj[k]).all():
                        gap_finite = False
                        break
                if not gap_finite:
                    i += 1
                    continue

                interval_positions = traj[f_in:f_out]
                interval_mean = np.nanmean(interval_positions, axis=0)
                pre_start = max(0, f_in - 5)
                pre_end = f_in
                post_start = f_out
                post_end = min(n_frames, f_out + 5)
                pre_positions = traj[pre_start:pre_end]
                post_positions = traj[post_start:post_end]
                if pre_positions.size == 0 or post_positions.size == 0:
                    i += 1
                    continue
                pre_mean = np.nanmean(pre_positions, axis=0)
                post_mean = np.nanmean(post_positions, axis=0)
                surrounding_mean = (pre_mean + post_mean) / 2.0
                displacement = float(np.linalg.norm(interval_mean - surrounding_mean))

                if displacement > min_disp and (f_out - f_in) >= min_frames_iv:
                    velocity_intervals.append(
                        {
                            "type": "velocity_jump",
                            "marker": str(marker_name),
                            "start_row": int(f_in),
                            "end_row": int(f_out - 1),
                            "start_frame": int(f_in),
                            "end_frame": int(f_out - 1),
                            "duration": int(f_out - f_in),
                            "jump_in_magnitude_mm": mag_in,
                            "jump_out_magnitude_mm": mag_out,
                            "sustained_displacement_mm": displacement,
                            "evidence": (
                                f"Jump in at row {f_in} ({mag_in:.0f} mm), "
                                f"jump out at row {f_out} ({mag_out:.0f} mm), "
                                f"sustained displacement {displacement:.0f} mm"
                            ),
                        }
                    )
                    i += 2
                else:
                    i += 1
                continue

            # Single jump — sustained plateau to trial end (skip isolated spike pairs).
            # Use frames strictly before the jump edge (exclude traj[f_in - 1]), otherwise
            # a one-frame spike before f_in biases pre_mean and marks the post-spike flat as displaced.
            pre_end_exclusive = f_in - 1
            if pre_end_exclusive <= 0:
                i += 1
                continue
            pre_start = max(0, pre_end_exclusive - 5)
            pre_positions = traj[pre_start:pre_end_exclusive]
            interval_positions = traj[f_in:]
            finite_mask = np.isfinite(interval_positions).all(axis=1)
            n_fin = int(np.sum(finite_mask))
            if n_fin < min_frames_iv or pre_positions.size == 0:
                i += 1
                continue
            interval_mean = np.nanmean(interval_positions[finite_mask], axis=0)
            pre_mean = np.nanmean(pre_positions, axis=0)
            displacement = float(np.linalg.norm(interval_mean - pre_mean))
            if displacement <= min_disp:
                i += 1
                continue
            f_out = n_frames
            velocity_intervals.append(
                {
                    "type": "velocity_jump",
                    "marker": str(marker_name),
                    "start_row": int(f_in),
                    "end_row": int(f_out - 1),
                    "start_frame": int(f_in),
                    "end_frame": int(f_out - 1),
                    "duration": int(f_out - f_in),
                    "jump_in_magnitude_mm": mag_in,
                    "jump_out_magnitude_mm": None,
                    "sustained_displacement_mm": displacement,
                    "evidence": (
                        f"Jump in at row {f_in}, no paired return jump (extends to trial end)"
                    ),
                }
            )
            i += 1

    return velocity_intervals


def _intervals_overlap(s1: int, e1: int, s2: int, e2: int) -> bool:
    return not (e1 < s2 or e2 < s1)


def merge_violations(
    velocity_intervals: Sequence[Mapping[str, Any]],
    bilateral_violations: Sequence[Mapping[str, Any]],
    anatomical_violations: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """
    Combine velocity (precise boundaries), bilateral, and anatomical intervals.

    Returns ``(merged_intervals, evidence_counts)`` for quality metrics.
    """
    min_len = int(config["min_swap_interval_length"])
    gap_max = int(config["merge_intervals_gap_max"])

    def _ensure_evidence_interval(
        s: int,
        e: int,
        *,
        velocity: list[dict[str, Any]],
        bilateral: list[tuple[str, str]],
        anatomical: list[str],
        sources: list[Mapping[str, Any]],
    ) -> dict[str, Any]:
        ev: set[str] = set()
        if velocity:
            ev.add("velocity")
        if bilateral:
            ev.add("bilateral")
        if anatomical:
            ev.add("anatomical")
        out_anat = list(dict.fromkeys(anatomical))
        for v in velocity:
            mk = str(v.get("marker", ""))
            if mk in ("RWRA", "LWRA") and mk not in out_anat:
                out_anat.append(mk)
        return {
            "start_row": int(s),
            "end_row": int(e),
            "bilateral_pairs": list(dict.fromkeys(bilateral)),
            "anatomical_markers": out_anat,
            "sources": list(sources),
            "velocity_detail": list(velocity),
            "evidence_types": sorted(ev),
        }

    used_bilateral: set[int] = set()
    used_anatomical: set[int] = set()
    raw: list[dict[str, Any]] = []

    for vi in velocity_intervals:
        vs, ve = int(vi["start_row"]), int(vi["end_row"])
        if ve - vs + 1 < min_len:
            continue
        marker = str(vi["marker"])
        bilateral_hits: list[tuple[str, str]] = []
        srcs: list[Mapping[str, Any]] = [vi]
        vel_bucket: list[dict[str, Any]] = [dict(vi)]

        for bi, bv in enumerate(bilateral_violations):
            bs, be = int(bv["start_row"]), int(bv["end_row"])
            if be - bs + 1 < min_len:
                continue
            if not _intervals_overlap(vs, ve, bs, be):
                continue
            left_m, right_m = str(bv["pair"][0]), str(bv["pair"][1])
            if marker not in (left_m, right_m):
                continue
            bilateral_hits.append((left_m, right_m))
            srcs.append(bv)
            used_bilateral.add(bi)

        anatom_hits: list[str] = []
        for ai, av in enumerate(anatomical_violations):
            if str(av["marker"]) != marker:
                continue
            as_, ae = int(av["start_row"]), int(av["end_row"])
            if ae - as_ + 1 < min_len:
                continue
            if not _intervals_overlap(vs, ve, as_, ae):
                continue
            anatom_hits.append(str(av["marker"]))
            srcs.append(av)
            used_anatomical.add(ai)

        raw.append(
            _ensure_evidence_interval(
                vs,
                ve,
                velocity=vel_bucket,
                bilateral=bilateral_hits,
                anatomical=anatom_hits,
                sources=srcs,
            )
        )

    for bi, bv in enumerate(bilateral_violations):
        if bi in used_bilateral:
            continue
        s, e = int(bv["start_row"]), int(bv["end_row"])
        if e - s + 1 < min_len:
            continue
        raw.append(
            _ensure_evidence_interval(
                s,
                e,
                velocity=[],
                bilateral=[tuple(bv["pair"])],
                anatomical=[],
                sources=[bv],
            )
        )

    for ai, av in enumerate(anatomical_violations):
        if ai in used_anatomical:
            continue
        s, e = int(av["start_row"]), int(av["end_row"])
        if e - s + 1 < min_len:
            continue
        raw.append(
            _ensure_evidence_interval(
                s,
                e,
                velocity=[],
                bilateral=[],
                anatomical=[str(av["marker"])],
                sources=[av],
            )
        )

    if not raw:
        return [], {
            "intervals_with_all_three_evidence": 0,
            "intervals_with_velocity_and_one_other": 0,
            "intervals_velocity_only": 0,
            "intervals_anatomical_only": 0,
            "intervals_bilateral_only": 0,
        }

    raw.sort(key=lambda x: (x["start_row"], x["end_row"]))
    merged: list[dict[str, Any]] = [raw[0].copy()]
    for k in ("bilateral_pairs", "anatomical_markers", "sources", "velocity_detail", "evidence_types"):
        merged[-1][k] = list(merged[-1][k])

    for cur in raw[1:]:
        prev = merged[-1]
        if int(cur["start_row"]) <= int(prev["end_row"]) + gap_max + 1:
            prev["end_row"] = max(prev["end_row"], cur["end_row"])
            for p in cur["bilateral_pairs"]:
                if p not in prev["bilateral_pairs"]:
                    prev["bilateral_pairs"].append(p)
            for m in cur["anatomical_markers"]:
                if m not in prev["anatomical_markers"]:
                    prev["anatomical_markers"].append(m)
            prev["sources"].extend(cur["sources"])
            prev["velocity_detail"].extend(cur["velocity_detail"])
            prev["evidence_types"] = sorted(set(prev["evidence_types"]) | set(cur["evidence_types"]))
        else:
            merged.append(cur.copy())
            for k in ("bilateral_pairs", "anatomical_markers", "sources", "velocity_detail", "evidence_types"):
                merged[-1][k] = list(merged[-1][k])

    evidence_counts = {
        "intervals_with_all_three_evidence": 0,
        "intervals_with_velocity_and_one_other": 0,
        "intervals_velocity_only": 0,
        "intervals_anatomical_only": 0,
        "intervals_bilateral_only": 0,
    }
    for block in merged:
        et = set(block["evidence_types"])
        hv = "velocity" in et
        hb = "bilateral" in et
        ha = "anatomical" in et
        if hv and hb and ha:
            evidence_counts["intervals_with_all_three_evidence"] += 1
        elif hv and (hb or ha):
            evidence_counts["intervals_with_velocity_and_one_other"] += 1
        elif hv:
            evidence_counts["intervals_velocity_only"] += 1
        elif ha and hb:
            evidence_counts["intervals_anatomical_only"] += 1
        elif ha:
            evidence_counts["intervals_anatomical_only"] += 1
        elif hb:
            evidence_counts["intervals_bilateral_only"] += 1

    return merged, evidence_counts


def _confidence_rank(c: str) -> int:
    return {"HIGH": 2, "MEDIUM": 1, "LOW": 0}.get(str(c).upper(), 0)


def find_unlabeled_donor(
    target_marker: str,
    start_row: int,
    end_row: int,
    markers: dict[str, np.ndarray],
    unlabeled_markers: dict[str, np.ndarray],
    points: np.ndarray,
    meta: dict[str, Any],
    segment_markers_dict: Mapping[str, Sequence[str]],
    reference_geometry: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[str | None, str]:
    from .trimmed_csv_marker_correction import compute_expected_for_stem

    label_to_idx: dict[str, int] = meta["label_to_marker_idx"]
    max_mean = float(config["donor_match_max_mean_distance_mm"])
    min_vis = float(config["donor_match_min_visibility_pct"])
    n_iv = end_row - start_row + 1
    expected_by_row: dict[int, np.ndarray] = {}
    for f in range(start_row, end_row + 1):
        excl = frozenset({target_marker})
        exp = compute_expected_for_stem(
            points, f, target_marker, segment_markers_dict, reference_geometry, label_to_idx, excl
        )
        if exp is not None and np.isfinite(exp).all():
            expected_by_row[f] = exp
    if len(expected_by_row) < 3:
        return None, "LOW"

    candidates: list[dict[str, Any]] = []
    for u_name, u_traj in unlabeled_markers.items():
        vis_rows = [f for f in range(start_row, end_row + 1) if np.isfinite(u_traj[f]).all()]
        if len(vis_rows) < int(min_vis * n_iv):
            continue
        dists: list[float] = []
        for f in vis_rows:
            if f in expected_by_row:
                dists.append(float(np.linalg.norm(u_traj[f] - expected_by_row[f])))
        if not dists:
            continue
        mean_d = float(np.mean(dists))
        if mean_d > max_mean:
            continue
        candidates.append(
            {
                "name": u_name,
                "mean_distance": mean_d,
                "n_matched_frames": len(dists),
                "visibility_pct": len(vis_rows) / float(n_iv),
            }
        )
    if not candidates:
        return None, "LOW"
    candidates.sort(key=lambda x: (x["mean_distance"], -x["n_matched_frames"]))
    best = candidates[0]
    if best["mean_distance"] < 30.0:
        conf = "HIGH"
    elif best["mean_distance"] < 60.0:
        conf = "MEDIUM"
    else:
        conf = "LOW"
    return str(best["name"]), conf


def identify_swap_operations(
    interval: Mapping[str, Any],
    markers: dict[str, np.ndarray],
    unlabeled_markers: dict[str, np.ndarray],
    points: np.ndarray,
    meta: dict[str, Any],
    segment_markers_dict: Mapping[str, Sequence[str]],
    reference_geometry: Mapping[str, Any],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Build ordered correction operations for one merged interval."""
    ops: list[dict[str, Any]] = []
    s_row = int(interval["start_row"])
    e_row = int(interval["end_row"])
    for left_m, right_m in interval.get("bilateral_pairs", []):
        ops.append({"kind": "swap_pair", "left": left_m, "right": right_m, "start_row": s_row, "end_row": e_row})
    for m in dict.fromkeys(interval.get("anatomical_markers", [])):
        if m == "RWRA":
            donor, dconf = find_unlabeled_donor(
                "RWRA",
                s_row,
                e_row,
                markers,
                unlabeled_markers,
                points,
                meta,
                segment_markers_dict,
                reference_geometry,
                config,
            )
            ops.append(
                {
                    "kind": "reassign_to_label",
                    "displaced": "RWRA",
                    "receiver": "RPSI",
                    "donor": donor,
                    "confidence": dconf,
                    "start_row": s_row,
                    "end_row": e_row,
                }
            )
        elif m == "LWRA":
            donor, dconf = find_unlabeled_donor(
                "LWRA",
                s_row,
                e_row,
                markers,
                unlabeled_markers,
                points,
                meta,
                segment_markers_dict,
                reference_geometry,
                config,
            )
            ops.append(
                {
                    "kind": "reassign_to_label",
                    "displaced": "LWRA",
                    "receiver": "LPSI",
                    "donor": donor,
                    "confidence": dconf,
                    "start_row": s_row,
                    "end_row": e_row,
                }
            )
    return ops


def apply_trajectory_operations(
    points: np.ndarray,
    meta: dict[str, Any],
    markers: dict[str, np.ndarray],
    unlabeled_idx: list[int],
    unlabeled_stem: list[str],
    operations: Sequence[Mapping[str, Any]],
    marker_tiers: Mapping[str, int],
    frames: np.ndarray,
    min_conf: str,
) -> list[dict[str, Any]]:
    from .trimmed_csv_marker_correction import _log_row

    l2i: dict[str, int] = meta["label_to_marker_idx"]
    ul_map = {stem: idx for stem, idx in zip(unlabeled_stem, unlabeled_idx, strict=True)}
    log: list[dict[str, Any]] = []
    min_rank = _confidence_rank(min_conf)

    for op in operations:
        conf = str(op.get("confidence", "MEDIUM"))
        if _confidence_rank(conf) < min_rank:
            continue
        kind = op["kind"]
        s_row = int(op["start_row"])
        e_row = int(op["end_row"])
        if kind == "swap_pair":
            lm, rm = str(op["left"]), str(op["right"])
            if lm not in l2i or rm not in l2i:
                continue
            il, ir = l2i[lm], l2i[rm]
            tier = max(int(marker_tiers.get(lm, 3)), int(marker_tiers.get(rm, 3)))
            for f in range(s_row, e_row + 1):
                pl = points[f, il, :].copy()
                pr = points[f, ir, :].copy()
                if not (np.isfinite(pl).all() and np.isfinite(pr).all()):
                    continue
                points[f, il, :] = pr
                points[f, ir, :] = pl
                markers[lm][f] = points[f, il, :].copy()
                markers[rm][f] = points[f, ir, :].copy()
                fr = int(frames[f])
                log.append(
                    _log_row(
                        fr,
                        lm,
                        0,
                        "trajectory_swap_pair",
                        tier,
                        conf,
                        pl,
                        points[f, il, :],
                        np.full(3, np.nan),
                        f"swap_with={rm};trajectory_stage0",
                    )
                )
                log.append(
                    _log_row(
                        fr,
                        rm,
                        0,
                        "trajectory_swap_pair",
                        tier,
                        conf,
                        pr,
                        points[f, ir, :],
                        np.full(3, np.nan),
                        f"swap_with={lm};trajectory_stage0",
                    )
                )
        elif kind == "reassign_to_label":
            displaced = str(op["displaced"])
            receiver = str(op["receiver"])
            donor = op.get("donor")
            if displaced not in l2i or receiver not in l2i:
                continue
            idd, irec = l2i[displaced], l2i[receiver]
            tier_d = int(marker_tiers.get(displaced, 3))
            tier_r = int(marker_tiers.get(receiver, 2))
            donor_idx = ul_map.get(str(donor)) if donor else None
            for f in range(s_row, e_row + 1):
                p_disp = points[f, idd, :].copy()
                if not np.isfinite(p_disp).all():
                    continue
                prev_rec = points[f, irec, :].copy()
                points[f, irec, :] = p_disp
                markers[receiver][f] = points[f, irec, :].copy()
                new_for_disp = np.full(3, np.nan)
                if donor_idx is not None and donor is not None:
                    du = points[f, donor_idx, :].copy()
                    if np.isfinite(du).all():
                        new_for_disp = du
                        points[f, donor_idx, :] = np.nan
                points[f, idd, :] = new_for_disp
                markers[displaced][f] = points[f, idd, :].copy()
                fr = int(frames[f])
                log.append(
                    _log_row(
                        fr,
                        displaced,
                        0,
                        "trajectory_swap_reassign",
                        tier_d,
                        conf,
                        p_disp,
                        points[f, idd, :],
                        np.full(3, np.nan),
                        f"to={receiver};donor={donor or 'none'};trajectory_stage0",
                    )
                )
                log.append(
                    _log_row(
                        fr,
                        receiver,
                        0,
                        "trajectory_swap_receive",
                        tier_r,
                        conf,
                        prev_rec,
                        points[f, irec, :],
                        np.full(3, np.nan),
                        f"from={displaced};trajectory_stage0",
                    )
                )
    return log


def trajectory_based_correction(
    points: np.ndarray,
    meta: dict[str, Any],
    meta_orig: dict[str, Any],
    setup: Mapping[str, Any],
    reference_geometry: Mapping[str, Any],
    segment_markers_dict: Mapping[str, Sequence[str]],
    marker_tiers: Mapping[str, int],
    unlabeled_indices: Sequence[int],
    unlabeled_stems: Sequence[str],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Stage 0: detect and correct long-running swaps. Mutates ``points`` in place.

    Returns ``(log_entries, summary_dict)``.
    """
    cfg = {**DEFAULT_TRAJECTORY_SWAP_CONFIG}
    for k in DEFAULT_TRAJECTORY_SWAP_CONFIG:
        if k in config:
            cfg[k] = config[k]
    if not bool(cfg.get("enable_trajectory_swap_detection", True)):
        return [], {"enabled": False}

    stems = [str(s) for s in meta["all_stems"]]
    l2i = meta["label_to_marker_idx"]
    markers: dict[str, np.ndarray] = {}
    for s in stems:
        if _is_unlabeled_stem(s) or str(s).startswith("OBSTACLE"):
            continue
        if s not in l2i:
            continue
        markers[s] = points[:, l2i[s], :].copy()
    unlabeled_markers: dict[str, np.ndarray] = {}
    for ui, ust in zip(unlabeled_indices, unlabeled_stems, strict=True):
        unlabeled_markers[str(ust)] = points[:, int(ui), :].copy()

    ml_axis = int(setup["ml_axis"])
    left_ml_sign = int(setup["left_ml_sign"])
    bilateral = [tuple(p) for p in cfg["bilateral_pairs"]]
    anatomical = [tuple(c) for c in cfg["anatomical_constraints"]]

    vel_v: list[dict[str, Any]] = []
    if bool(cfg.get("enable_velocity_discontinuity_detection", True)):
        vel_v = detect_velocity_discontinuities(markers, cfg)

    b_v = detect_bilateral_violations(markers, bilateral, ml_axis, left_ml_sign, cfg)
    a_v = detect_anatomical_violations(markers, anatomical, cfg)
    merged, evidence_counts = merge_violations(vel_v, b_v, a_v, cfg)

    all_ops: list[dict[str, Any]] = []
    for interval in merged:
        all_ops.extend(
            identify_swap_operations(
                interval,
                markers,
                unlabeled_markers,
                points,
                meta,
                segment_markers_dict,
                reference_geometry,
                cfg,
            )
        )
    frames = meta["frames"]
    min_conf = str(cfg.get("apply_correction_min_confidence", "MEDIUM"))
    log = apply_trajectory_operations(
        points,
        meta,
        markers,
        list(unlabeled_indices),
        list(unlabeled_stems),
        all_ops,
        marker_tiers,
        frames,
        min_conf,
    )

    summary = {
        "enabled": True,
        "velocity_violations_detected": len(vel_v),
        "bilateral_violations_detected": len(b_v),
        "anatomical_violations_detected": len(a_v),
        "merged_intervals": len(merged),
        "operations_planned": len(all_ops),
        "log_entries": len(log),
        "bilateral_intervals": b_v,
        "anatomical_intervals": a_v,
        "velocity_intervals": vel_v,
        **evidence_counts,
    }
    return log, summary
