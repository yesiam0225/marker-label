"""Build gap-fill reference geometry from a labeled static CSV."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from marker_label.io import load_c3d
from marker_label.trial_trim import bestframe_sidecar_path, parse_labeled_csv

from .reference import build_robust_reference
from .two_marker_static import static_offsets_for_three_marker_segment


def _load_static_points_and_label_idx(
    static_path: str | Path,
    dynamic_label_to_idx: Mapping[str, int],
) -> tuple[np.ndarray, dict[str, int], np.ndarray, str]:
    """
    Load static trial as (points, label_to_idx, frames, format_tag).

    ``label_to_idx`` maps anatomical names to columns in ``points`` (static file order).
    """
    path = Path(static_path)
    suf = path.suffix.lower()
    if suf == ".csv":
        try:
            _, meta = parse_labeled_csv(path)
        except UnicodeDecodeError as e:
            raise ValueError(
                f"{path.name} is not a UTF-8 labeled flat CSV. "
                "Use a labeled .csv export, or pass a labeled static .c3d with --static-csv."
            ) from e
        return (
            np.asarray(meta["points"], dtype=np.float64),
            dict(meta["label_to_marker_idx"]),
            np.asarray(meta["frames"], dtype=np.int64),
            "csv",
        )
    if suf == ".c3d":
        try:
            d = load_c3d(path)
        except (UnicodeDecodeError, AssertionError, OSError) as e:
            raise ValueError(
                f"Could not read static C3D {path!r}: {e}. "
                "Use the subject's labeled static trial (anatomical marker names in the C3D)."
            ) from e
        labels = [str(lab).strip() for lab in d["labels"]]
        col_by_name = {lab: j for j, lab in enumerate(labels) if lab and not lab.startswith("*")}
        s_idx: dict[str, int] = {}
        for name in dynamic_label_to_idx:
            nn = str(name).strip()
            if nn in col_by_name:
                s_idx[nn] = col_by_name[nn]
        if not s_idx:
            raise ValueError(
                f"No marker names from the dynamic CSV matched labels in static C3D {path.name}. "
                f"Static labels (sample): {labels[:12]!r}"
            )
        pts = np.asarray(d["points"], dtype=np.float64)
        n_frames = int(pts.shape[0])
        ff = int(d.get("first_frame", 1))
        frames = np.arange(ff, ff + n_frames, dtype=np.int64)
        return pts, s_idx, frames, "c3d"
    raise ValueError(
        f"Static reference path must be a labeled .csv or .c3d file, not {path.suffix!r} ({path})"
    )


def load_static_reference_bundle(
    static_csv: str | Path,
    segment_markers_dict: Mapping[str, Sequence[str]],
    dynamic_label_to_idx: Mapping[str, int],
    *,
    n_samples: int = 30,
    lab_vertical: tuple[float, float, float] = (0.0, 1.0, 0.0),
) -> dict[str, Any]:
    """
    Reference dict and auxiliary offsets from the subject static trial.

    Returns keys: ``reference``, ``warnings``, ``two_marker_offsets`` (seg -> target -> tuple),
    ``pelvis_psi_offset`` (mean body offset for LPSI/RPSI from static), ``shoulder_local``.
    """
    static_points, s_idx, frames, fmt = _load_static_points_and_label_idx(
        static_csv, dynamic_label_to_idx
    )
    bf_path = bestframe_sidecar_path(static_csv) if fmt == "csv" else None
    bf = None
    if bf_path is not None and bf_path.is_file():
        try:
            bf = int(bf_path.read_text().strip().split()[0])
        except (ValueError, OSError):
            pass

    reference, warnings = build_robust_reference(
        static_points,
        s_idx,
        segment_markers_dict,
        n_samples=n_samples,
        best_frame=bf,
        csv_path=static_csv if fmt == "csv" else None,
        frames=frames,
    )
    warnings = list(warnings) + [f"static_reference_loaded_from_{fmt}"]

    vert = np.array(lab_vertical, dtype=np.float64)
    two_marker: dict[str, dict[str, tuple[np.ndarray, np.ndarray, str, str]]] = {}
    for seg, names in segment_markers_dict.items():
        seg_names = [str(x).strip() for x in names if str(x).strip() in s_idx]
        offs = static_offsets_for_three_marker_segment(static_points, s_idx, seg_names, vert)
        if offs:
            two_marker[str(seg)] = offs

    pelvis_psi: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    pelvis_psi.update(
        _static_pelvis_psi_offsets(static_points, s_idx, "LPSI", lab_vertical=vert)
    )
    pelvis_psi.update(
        _static_pelvis_psi_offsets(static_points, s_idx, "RPSI", lab_vertical=vert)
    )

    shoulder_local = _static_shoulder_locals(static_points, s_idx)

    return {
        "reference": reference,
        "warnings": warnings,
        "two_marker_offsets": two_marker,
        "pelvis_psi_offset": pelvis_psi,
        "shoulder_local": shoulder_local,
    }


def _static_pelvis_psi_offsets(
    points: np.ndarray,
    label_to_idx: Mapping[str, int],
    target: str,
    *,
    lab_vertical: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Reuse ASIS-only style offset from static mean pelvis (4 markers)."""
    from .asis_only_fill import ap_axis_at_anchor

    names = ("LASI", "RASI", "LPSI", "RPSI")
    if not all(n in label_to_idx for n in names) or target not in label_to_idx:
        return {}
    n = points.shape[0]
    acc: list[int] = []
    for f in range(n):
        if all(np.isfinite(points[f, label_to_idx[m], :]).all() for m in names):
            acc.append(f)
    if not acc:
        return {}
    r = acc[0]
    lasi_p = points[r, label_to_idx["LASI"], :]
    rasi_p = points[r, label_to_idx["RASI"], :]
    center = 0.5 * (lasi_p + rasi_p)
    ml = (rasi_p - lasi_p) / np.linalg.norm(rasi_p - lasi_p)
    ap = ap_axis_at_anchor(points, r, label_to_idx, "LASI", "RASI", "LPSI", "RPSI")
    si = np.cross(ap, ml)
    si = si / np.linalg.norm(si)
    ap_o = np.cross(ml, si)
    ap_o = ap_o / np.linalg.norm(ap_o)
    r_mat = np.stack([ml, si, ap_o], axis=1)
    pt = points[r, label_to_idx[target], :]
    flip = pt - center
    return {target: (r_mat.T @ (pt - center), flip)}


def _static_shoulder_locals(
    points: np.ndarray,
    label_to_idx: Mapping[str, int],
) -> dict[str, np.ndarray]:
    """Mean LSHO/RSHO in thorax-local frame (C7, CLAV, RBAK)."""
    thorax = ("C7", "CLAV", "RBAK")
    shoulders = ("LSHO", "RSHO")
    if not all(m in label_to_idx for m in thorax + shoulders):
        return {}
    n = points.shape[0]
    locals_acc: dict[str, list[np.ndarray]] = {s: [] for s in shoulders}
    from .reference import _global_to_local, _local_coords_from_three

    for f in range(n):
        if not all(np.isfinite(points[f, label_to_idx[m], :]).all() for m in thorax + shoulders):
            continue
        p0 = points[f, label_to_idx["C7"], :]
        p1 = points[f, label_to_idx["CLAV"], :]
        p2 = points[f, label_to_idx["RBAK"], :]
        try:
            origin, basis = _local_coords_from_three(p0, p1, p2)
        except ValueError:
            continue
        for s in shoulders:
            locals_acc[s].append(_global_to_local(points[f, label_to_idx[s], :], origin, basis))
    out: dict[str, np.ndarray] = {}
    for s, lst in locals_acc.items():
        if lst:
            out[s] = np.mean(np.stack(lst, axis=0), axis=0)
    return out
