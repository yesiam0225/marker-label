"""
Post-labeling trial trim: contiguous range around pipeline best frame using segment Kabsch residuals and visibility.

Preserves original ``frame`` and ``time`` values (no renumbering). Best-frame sidecar follows the same convention
as ``marker_label.pipeline`` and ``qc_viewer``: ``<csv>.csv.bestframe`` with a 1-based frame index.

CLI segment presets (see :func:`segment_markers_dict_for_trim_preset`): ``default`` is thigh/shank/foot
only (no pelvis segment; thighs use ASI/THI/KNE). ``lower-body`` adds optional pelvis QC
(LASI/RASI/LTHI triad + leg chains, no PSIS).
``full-body`` restores the previous default: ``segments.SEGMENTS`` with ≥3 markers per entry. ``legs-feet``
is an alias of ``default``.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# --- Defaults (overridable via ``config`` in ``trim_trial``) ---
DEFAULT_RESIDUAL_THRESHOLD_MM = 15.0
DEFAULT_RESIDUAL_THRESHOLD_THIGH_MM = 25.0
DEFAULT_THIGH_SEGMENT_KEYWORDS = ("thigh", "femur")
DEFAULT_VISIBLE_RATIO_THRESHOLD = 0.90
DEFAULT_CONSECUTIVE_BAD_FRAMES = 5
DEFAULT_MIN_TRIAL_LENGTH = 100
DEFAULT_COLLINEARITY_EPS_MM = 1e-3
OBSTACLE_NAME_PREFIXES = ("OBSTACLE_",)

# Preset ``lower-body``: gait / lower-limb kinematics focus — PSIS omitted from pelvis QC.
# ``Pelvis_ASIS_LThigh`` uses LASI–RASI–LTHI as a non-collinear triad for Kabsch (trim metric only).
LOWER_BODY_TRIM_PRESET_SEGMENTS: dict[str, list[str]] = {
    "Pelvis_ASIS_LThigh": ["LASI", "RASI", "LTHI"],
    "L_Thigh": ["LASI", "LTHI", "LKNE"],
    "L_Shank": ["LKNE", "LTIB", "LANK"],
    "L_Foot": ["LANK", "LHEE", "LTOE", "LANK"],
    "R_Thigh": ["RASI", "RTHI", "RKNE"],
    "R_Shank": ["RKNE", "RTIB", "RANK"],
    "R_Foot": ["RANK", "RHEE", "RTOE", "RANK"],
}

# Legs-feet preset keeps only lower-limb chains, but thigh uses ASI->THI->KNE for
# anatomically rigid thigh geometry in gait QC.
LEGS_FEET_TRIM_PRESET_SEGMENTS: dict[str, list[str]] = {
    "L_Thigh": ["LASI", "LTHI", "LKNE"],
    "L_Shank": ["LKNE", "LTIB", "LANK"],
    "L_Foot": ["LANK", "LHEE", "LTOE"],
    "R_Thigh": ["RASI", "RTHI", "RKNE"],
    "R_Shank": ["RKNE", "RTIB", "RANK"],
    "R_Foot": ["RANK", "RHEE", "RTOE"],
}


def _ordered_unique_marker_stems(segments: Mapping[str, Sequence[str]]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for names in segments.values():
        for n in names:
            s = str(n).strip()
            if s.startswith("*") or s in seen:
                continue
            seen.add(s)
            out.append(s)
    return tuple(out)


# Stems used for ``visible_ratio`` when ``visibility_ratio_marker_subset`` is ``legs-feet``.
LEGS_FEET_VISIBILITY_STEMS: tuple[str, ...] = _ordered_unique_marker_stems(LEGS_FEET_TRIM_PRESET_SEGMENTS)

TRIM_SEGMENT_PRESET_CHOICES = frozenset({"default", "full-body", "lower-body", "legs-feet"})


def segment_markers_dict_for_trim_preset(preset: str) -> dict[str, list[str]]:
    """
    Segment definitions for :func:`trim_trial` / CLI.

    Parameters
    ----------
    preset
        ``"default"`` / ``"legs-feet"`` — thighs/shanks/feet only (no pelvis segment). Thigh chains use
        ``LASI->LTHI->LKNE`` and ``RASI->RTHI->RKNE`` for anatomical rigidity.

        ``"lower-body"`` — optional pelvis + leg QC: LASI/RASI/LTHI triad plus the usual leg chains
        (no PSIS on the pelvis triad).

        ``"full-body"`` — all ``segments.SEGMENTS`` entries with at least three markers (skips two-marker
        visualization sticks); same segment set as the historical CLI default before leg-first default.
    """
    key = str(preset).strip().lower().replace("_", "-")
    if key in ("default", "", "legs-feet", "legsfeet"):
        return {k: list(v) for k, v in LEGS_FEET_TRIM_PRESET_SEGMENTS.items()}
    if key in ("full-body", "fullbody"):
        from .segments import SEGMENTS

        return {k: list(v) for k, v in SEGMENTS.items() if len(v) >= 3}
    if key == "lower-body":
        return {k: list(v) for k, v in LOWER_BODY_TRIM_PRESET_SEGMENTS.items()}
    raise ValueError(
        f"Unknown segment preset {preset!r}; choose one of {sorted(TRIM_SEGMENT_PRESET_CHOICES)}"
    )


def kabsch(P_ref: np.ndarray, P_cur: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Rigid transform from reference to current: each row is a point, shape (N, 3).

    Returns ``R`` (3, 3) and ``t`` (3,) such that ``P_cur ≈ P_ref @ R.T + t`` (row-wise).

    Parameters
    ----------
    P_ref, P_cur : (N, 3), must have N >= 3 for a stable rotation.
    """
    if P_ref.shape != P_cur.shape or P_ref.ndim != 2 or P_ref.shape[1] != 3:
        raise ValueError(f"Expected matching (N, 3) arrays, got {P_ref.shape}, {P_cur.shape}")
    n = P_ref.shape[0]
    if n < 3:
        raise ValueError(f"Kabsch needs at least 3 points, got {n}")

    c_ref = np.mean(P_ref, axis=0)
    c_cur = np.mean(P_cur, axis=0)
    X = P_ref - c_ref
    Y = P_cur - c_cur
    H = X.T @ Y
    u, _, vt = np.linalg.svd(H)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    t = c_cur - r @ c_ref
    return r, t


def _segment_residual_threshold_mm(
    segment_name: str,
    *,
    default_mm: float,
    thigh_mm: float,
    thigh_keywords: Sequence[str],
) -> float:
    low = segment_name.lower()
    if any(kw in low for kw in thigh_keywords):
        return float(thigh_mm)
    return float(default_mm)


def bestframe_sidecar_path(csv_path: str | Path) -> Path:
    """``Path(csv_path).with_suffix(Path(csv_path).suffix + '.bestframe')``."""
    p = Path(csv_path)
    return p.with_suffix(p.suffix + ".bestframe")


def load_best_frame_1based(csv_path: str | Path) -> int:
    """
    Read 1-based best frame from ``*.csv.bestframe`` next to the labeled CSV.

    The file is written by the pipeline as ``str(best_frame + 1)`` (1-based frame number).
    """
    path = bestframe_sidecar_path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Best-frame sidecar not found: {path}")
    text = path.read_text().strip().split()
    if not text:
        raise ValueError(f"Empty bestframe file: {path}")
    return int(text[0])


def _read_first_line_raw(fp: io.TextIOWrapper) -> str:
    line = fp.readline()
    if not line:
        return ""
    return line if line.endswith("\n") else line + "\n"


def _normalize_c3d_export_marker_name(raw: str, fallback_index: int) -> str:
    """Strip subject prefix (``DEMO:LFHD``) and map obstacle aliases for the viewer."""
    s = str(raw).strip()
    if not s:
        return f"Point_{fallback_index}"
    if ":" in s:
        s = s.split(":", 1)[1].strip()
    upper = s.upper()
    if upper == "OBS1":
        return "OBSTACLE_L"
    if upper == "OBS2":
        return "OBSTACLE_R"
    return s


def _parse_c3d_export_csv(
    path: Path,
    marker_line: str,
    column_line: str,
    rest: str,
) -> tuple[list[str], dict[str, Any]]:
    """
    Parse C3D-style CSV export: marker names row, then ``Frame``, ``Sub Frame``, ``X/Y/Z``…
    """
    row0 = next(csv.reader(io.StringIO(marker_line)))
    row1 = next(csv.reader(io.StringIO(column_line)))
    if len(row1) < 4:
        raise ValueError(f"C3D export header too short: {path}")

    x_indices = [i for i, c in enumerate(row1) if str(c).strip().upper() == "X"]
    if not x_indices:
        raise ValueError(f"No X columns in C3D export header: {path}")

    all_stems: list[str] = []
    stem_to_triplet: dict[str, tuple[int, int, int]] = {}
    for mi, xi in enumerate(x_indices):
        if xi + 2 >= len(row1):
            raise ValueError(f"Incomplete X/Y/Z triplet at column {xi} in {path}")
        raw_name = row0[xi].strip() if xi < len(row0) else ""
        stem = _normalize_c3d_export_marker_name(raw_name, mi)
        if stem in stem_to_triplet:
            stem = f"{stem}_{mi}"
        stem_to_triplet[stem] = (xi, xi + 1, xi + 2)
        all_stems.append(stem)

    n_markers = len(all_stems)
    all_rows = list(csv.reader(io.StringIO(rest)))
    start = 0
    if all_rows and any(str(c).strip().lower() == "mm" for c in all_rows[0]):
        start = 1
    data_rows = [row for row in all_rows[start:] if any(str(c).strip() for c in row)]

    n_frames = len(data_rows)
    frames = np.empty(n_frames, dtype=np.int64)
    times = np.empty(n_frames, dtype=np.float64)
    points = np.full((n_frames, n_markers, 3), np.nan)

    for r, row in enumerate(data_rows):
        if len(row) < 2:
            raise ValueError(f"Row {r}: expected frame, sub frame, ...")
        try:
            frames[r] = int(float(row[0].strip()))
        except ValueError as e:
            raise ValueError(f"Row {r}: invalid frame value {row[0]!r}") from e
        try:
            times[r] = float(row[1].strip()) if row[1].strip() else np.nan
        except ValueError:
            times[r] = np.nan
        for m, stem in enumerate(all_stems):
            ix, iy, iz = stem_to_triplet[stem]
            for k, j in enumerate((ix, iy, iz)):
                if j < len(row) and row[j].strip():
                    try:
                        points[r, m, k] = float(row[j])
                    except ValueError:
                        pass

    if n_frames > 1:
        diffs = np.diff(frames)
        pos = diffs[diffs > 0]
        if len(pos):
            median_step = float(np.median(pos))
            rate = 100.0 if median_step <= 1.0 else 100.0 / median_step
        else:
            rate = 100.0
    else:
        rate = 0.0

    if n_frames > 0:
        diffs = np.diff(frames)
        if not np.all(diffs == 1):
            raise ValueError(
                f"Column 'frame' must be strictly increasing contiguous integers; "
                f"got diffs {np.unique(diffs).tolist()}"
            )

    ignored_star = [s for s in all_stems if s.strip().startswith("*")]
    qc_stems_ordered: list[str] = []
    seen: set[str] = set()
    for s in all_stems:
        if s.strip().startswith("*"):
            continue
        if s not in seen:
            seen.add(s)
            qc_stems_ordered.append(s)

    label_to_marker_idx = {s: mi for mi, s in enumerate(all_stems)}
    original_header_line = marker_line if marker_line.endswith("\n") else marker_line + "\n"
    original_header_line += column_line if column_line.endswith("\n") else column_line + "\n"

    metadata: dict[str, Any] = {
        "path": path,
        "original_header_line": original_header_line,
        "all_stems": all_stems,
        "stem_to_col_triplet": stem_to_triplet,
        "frames": frames,
        "times": times,
        "points": points,
        "rate": rate,
        "data_rows": data_rows,
        "label_to_marker_idx": label_to_marker_idx,
        "ignored_star_stems": ignored_star,
        "n_frames": n_frames,
        "n_markers": n_markers,
        "csv_format": "c3d_export",
    }
    return qc_stems_ordered, metadata


def parse_labeled_csv(csv_path: str | Path) -> tuple[list[str], dict[str, Any]]:
    """
    Load a labeled flat CSV: ``frame``, ``time``, then ``marker_x/y/z`` triplets.

    Skips marker stems whose stripped name starts with ``*`` for **QC union** bookkeeping
    (columns remain in output when trimming rows only).

    Returns
    -------
    qc_marker_stems : list[str]
        Ordered unique stems used for visibility / segment dictionaries (no ``*`` stems).
    metadata : dict
        Includes ``original_header_line``, ``all_stems``, ``stem_to_col_triplet`` (indices into row),
        ``frames``, ``times``, ``points`` (n_frames, n_markers, 3), ``rate``, ``data_rows``,
        ``label_to_marker_idx`` (stem -> marker column index), ``ignored_star_stems``.
    """
    path = Path(csv_path)
    with open(path, newline="", encoding="utf-8-sig") as f:
        marker_line = _read_first_line_raw(f)
        column_line = _read_first_line_raw(f)
        rest = f.read()
    if not marker_line.strip():
        raise ValueError(f"Missing header: {path}")

    col_cells = next(csv.reader(io.StringIO(column_line)))
    if len(col_cells) >= 2:
        h0 = str(col_cells[0]).strip().lower()
        h1 = str(col_cells[1]).strip().lower().replace(" ", "")
        if h0 == "frame" and h1 in ("subframe", "subframe"):
            return _parse_c3d_export_csv(path, marker_line, column_line, rest)

    # Labeled flat CSV: first line is ``frame, time, marker_x, ...``
    original_header_line = marker_line
    rest = column_line + rest

    reader = csv.reader(io.StringIO(original_header_line))
    header_cells = next(reader)
    if len(header_cells) < 2:
        raise ValueError(f"Header too short: {path}")

    def _norm_cell(c: str) -> str:
        return str(c).strip()

    h0, h1 = _norm_cell(header_cells[0]), _norm_cell(header_cells[1])
    if h0.lower() != "frame" or h1.lower() != "time":
        raise ValueError(f"Expected frame, time in header; got {h0!r}, {h1!r}")

    stem_to_triplet: dict[str, tuple[int, int, int]] = {}
    all_stems: list[str] = []
    i = 2
    while i + 2 < len(header_cells):
        xs = str(header_cells[i])
        ys = str(header_cells[i + 1])
        zs = str(header_cells[i + 2])
        if not (xs.endswith("_x") and ys.endswith("_y") and zs.endswith("_z")):
            break
        stem_x = xs[:-2].strip()
        stem_y = ys[:-2].strip()
        stem_z = zs[:-2].strip()
        if stem_x != stem_y or stem_x != stem_z:
            raise ValueError(
                f"Mismatched marker stems at columns {i},{i+1},{i+2}: "
                f"{stem_x!r}, {stem_y!r}, {stem_z!r}"
            )
        stem_to_triplet[stem_x] = (i, i + 1, i + 2)
        all_stems.append(stem_x)
        i += 3

    n_markers = len(all_stems)
    data_reader = csv.reader(io.StringIO(rest))
    data_rows = [row for row in data_reader if any(c.strip() for c in row)]

    n_frames = len(data_rows)
    frames = np.empty(n_frames, dtype=np.int64)
    times = np.empty(n_frames, dtype=np.float64)
    points = np.full((n_frames, n_markers, 3), np.nan)

    for r, row in enumerate(data_rows):
        if len(row) < 2:
            raise ValueError(f"Row {r}: expected frame, time, ...")
        try:
            frames[r] = int(float(row[0].strip()))
        except ValueError as e:
            raise ValueError(f"Row {r}: invalid frame value {row[0]!r}") from e
        try:
            times[r] = float(row[1].strip()) if row[1].strip() else np.nan
        except ValueError:
            times[r] = np.nan
        for m, stem in enumerate(all_stems):
            ix, iy, iz = stem_to_triplet[stem]
            for k, j in enumerate((ix, iy, iz)):
                if j < len(row) and row[j].strip():
                    try:
                        points[r, m, k] = float(row[j])
                    except ValueError:
                        pass

    if n_frames == 0:
        rate = 0.0
    elif n_frames == 1:
        rate = 0.0
    else:
        dt = float(times[1] - times[0]) if np.isfinite(times[0]) and np.isfinite(times[1]) else 0.0
        rate = 1.0 / dt if dt > 0 else 0.0

    # Validate frame column: sorted, contiguous, matches row order
    if n_frames > 0:
        diffs = np.diff(frames)
        if not np.all(diffs == 1):
            raise ValueError(
                f"Column 'frame' must be strictly increasing contiguous integers; "
                f"got diffs {np.unique(diffs).tolist()}"
            )

    ignored_star = [s for s in all_stems if s.strip().startswith("*")]
    qc_stems_ordered: list[str] = []
    seen: set[str] = set()
    for s in all_stems:
        if s.strip().startswith("*"):
            continue
        if s not in seen:
            seen.add(s)
            qc_stems_ordered.append(s)

    label_to_marker_idx = {s: mi for mi, s in enumerate(all_stems)}

    metadata: dict[str, Any] = {
        "path": path,
        "original_header_line": original_header_line,
        "all_stems": all_stems,
        "stem_to_col_triplet": stem_to_triplet,
        "frames": frames,
        "times": times,
        "points": points,
        "rate": rate,
        "data_rows": data_rows,
        "label_to_marker_idx": label_to_marker_idx,
        "ignored_star_stems": ignored_star,
        "n_frames": n_frames,
        "n_markers": n_markers,
    }
    return qc_stems_ordered, metadata


def _markers_in_segment_dict(segment_markers_dict: Mapping[str, Sequence[str]]) -> set[str]:
    out: set[str] = set()
    for names in segment_markers_dict.values():
        for n in names:
            out.add(str(n).strip())
    return out


def _should_exclude_obstacle_for_visibility(
    stem: str,
    segment_marker_names: set[str],
    *,
    exclude_obstacles_not_in_segments: bool,
) -> bool:
    if not exclude_obstacles_not_in_segments:
        return False
    st = stem.strip()
    if st in segment_marker_names:
        return False
    return any(st.upper().startswith(pfx) for pfx in OBSTACLE_NAME_PREFIXES)


def build_visibility_union_indices(
    metadata: dict[str, Any],
    segment_markers_dict: Mapping[str, Sequence[str]],
    *,
    exclude_obstacles_not_in_segments: bool = True,
) -> tuple[list[int], list[str]]:
    """Deduped marker indices (column order) for visibility ratio, excluding * and optional obstacles."""
    all_stems: list[str] = metadata["all_stems"]
    segment_names = _markers_in_segment_dict(segment_markers_dict)
    label_to_idx: dict[str, int] = metadata["label_to_marker_idx"]
    chosen: list[int] = []
    labels_out: list[str] = []
    seen: set[str] = set()
    for stem in all_stems:
        st = stem.strip()
        if st.startswith("*"):
            continue
        if _should_exclude_obstacle_for_visibility(
            st, segment_names, exclude_obstacles_not_in_segments=exclude_obstacles_not_in_segments
        ):
            continue
        if st not in segment_names:
            continue
        if st in seen:
            continue
        seen.add(st)
        chosen.append(label_to_idx[stem])
        labels_out.append(st)
    return chosen, labels_out


def visibility_indices_for_marker_stems(
    metadata: dict[str, Any],
    stems: Sequence[str],
) -> tuple[list[int], list[str]]:
    """
    Column indices for a fixed marker list (e.g. leg/foot only), in ``stems`` order, deduped.

    Skips stems missing from the CSV. Result may be empty if no stem is present.
    """
    label_to_idx: dict[str, int] = metadata["label_to_marker_idx"]
    chosen: list[int] = []
    labels_out: list[str] = []
    seen: set[str] = set()
    for raw in stems:
        st = str(raw).strip()
        if st.startswith("*") or st in seen:
            continue
        if st not in label_to_idx:
            continue
        seen.add(st)
        chosen.append(label_to_idx[st])
        labels_out.append(st)
    return chosen, labels_out


def resolve_visibility_ratio_marker_indices(
    metadata: dict[str, Any],
    segment_union_indices: Sequence[int],
    segment_union_labels: Sequence[str],
    config: Mapping[str, Any] | None,
) -> tuple[list[int], list[str]]:
    """
    Indices used for per-frame ``visible_ratio`` (finite XYZ / count).

    * ``legs-feet`` (default) — only :data:`LEGS_FEET_VISIBILITY_STEMS` present in the CSV;
      if none match, falls back to the segment union so small/synthetic CSVs still work.
    * ``segment-union`` — same markers as :func:`build_visibility_union_indices`.
    """
    cfg = dict(config or {})
    raw = cfg.get("visibility_ratio_marker_subset", "legs-feet")
    if raw is None:
        mode = "legs-feet"
    else:
        mode = str(raw).strip().lower().replace("_", "-")
    if mode in ("segment-union", "union"):
        return list(segment_union_indices), list(segment_union_labels)
    if mode in ("legs-feet", "legsfeet", "", "default"):
        legs_idx, legs_lab = visibility_indices_for_marker_stems(metadata, LEGS_FEET_VISIBILITY_STEMS)
        if legs_idx:
            return legs_idx, legs_lab
        return list(segment_union_indices), list(segment_union_labels)
    raise ValueError(
        f"Unknown visibility_ratio_marker_subset {raw!r}; use 'legs-feet' or 'segment-union'"
    )


def _orthonormal_frame_from_three_points(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    *,
    collinearity_eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Origin at ``a``, columns of R_global_from_local are [e1, e2, e3] (right-handed).
    ``p_global = R_global_from_local @ p_local + origin``.
    """
    e1 = b - a
    n1 = np.linalg.norm(e1)
    if n1 < collinearity_eps:
        raise ValueError("Degenerate segment: first two reference markers coincide")
    e1 = e1 / n1
    v = c - a
    e3 = np.cross(e1, v)
    n3 = np.linalg.norm(e3)
    if n3 < collinearity_eps:
        raise ValueError(
            "Degenerate segment: first three reference markers are colinear "
            f"(cross norm {n3} < {collinearity_eps})"
        )
    e3 = e3 / n3
    e2 = np.cross(e3, e1)
    r_gl = np.stack([e1, e2, e3], axis=1)
    return r_gl, a.copy()


def compute_reference_geometry(
    metadata: dict[str, Any],
    segment_markers_dict: Mapping[str, Sequence[str]],
    best_idx: int,
    *,
    collinearity_eps_mm: float = DEFAULT_COLLINEARITY_EPS_MM,
) -> dict[str, Any]:
    """
    For each segment, build local coordinates at ``best_idx`` using the first three **finite**
    markers in that segment's list (in order). Stores local coords for all segment markers finite at best.
    """
    points: np.ndarray = metadata["points"]
    label_to_idx: dict[str, int] = metadata["label_to_marker_idx"]
    out: dict[str, Any] = {}
    for seg_name, names in segment_markers_dict.items():
        if len(names) < 3:
            raise ValueError(f"Segment {seg_name!r} needs at least 3 marker names, got {len(names)}")
        p_best = points[best_idx]
        triple: list[tuple[str, np.ndarray]] = []
        for raw in names:
            stem = str(raw).strip()
            if stem not in label_to_idx:
                continue
            idx = label_to_idx[stem]
            q = p_best[idx]
            if np.isfinite(q).all():
                triple.append((stem, q))
            if len(triple) >= 3:
                break
        if len(triple) < 3:
            raise ValueError(
                f"Segment {seg_name!r}: fewer than 3 finite markers at best frame among "
                f"first positions in list {list(names)!r}"
            )
        (_, a), (_, b), (_, c) = triple[0], triple[1], triple[2]
        r_gl, origin = _orthonormal_frame_from_three_points(
            a, b, c, collinearity_eps=collinearity_eps_mm
        )

        marker_names: list[str] = []
        local_coords: list[np.ndarray] = []
        for raw in names:
            stem = str(raw).strip()
            if stem not in label_to_idx:
                continue
            idx = label_to_idx[stem]
            g = p_best[idx]
            if not np.isfinite(g).all():
                continue
            marker_names.append(stem)
            local = r_gl.T @ (g - origin)
            local_coords.append(local)
        if len(marker_names) < 3:
            raise ValueError(
                f"Segment {seg_name!r}: fewer than 3 finite markers at best frame in full segment list"
            )
        out[seg_name] = {
            "R_global_from_local": r_gl,
            "origin_global": origin,
            "marker_names": marker_names,
            "P_local": np.stack(local_coords, axis=0),
        }
    return out


def compute_diagnostics(
    metadata: dict[str, Any],
    reference_geometry: Mapping[str, Any],
    segment_markers_dict: Mapping[str, Sequence[str]],
    visibility_ratio_indices: Sequence[int],
    *,
    residual_threshold_mm: float,
    residual_threshold_thigh_mm: float,
    thigh_segment_keywords: Sequence[str],
    visible_ratio_threshold: float,
) -> list[dict[str, Any]]:
    """Per-frame diagnostics: segment residuals, visibility ratio, bad flags (per-segment thresholds)."""
    points: np.ndarray = metadata["points"]
    n_frames = int(metadata["n_frames"])
    label_to_idx: dict[str, int] = metadata["label_to_marker_idx"]
    frames: np.ndarray = metadata["frames"]

    seg_thresholds = {
        sn: _segment_residual_threshold_mm(
            sn,
            default_mm=residual_threshold_mm,
            thigh_mm=residual_threshold_thigh_mm,
            thigh_keywords=thigh_segment_keywords,
        )
        for sn in segment_markers_dict
    }

    diagnostics: list[dict[str, Any]] = []
    n_vis = len(visibility_ratio_indices)
    if n_vis == 0:
        raise ValueError("visibility_ratio_indices is empty; cannot compute visible_ratio")

    for f in range(n_frames):
        vis_pts = points[f, visibility_ratio_indices, :]
        valid = np.isfinite(vis_pts).all(axis=1)
        visible_count = int(np.sum(valid))
        visible_ratio = float(visible_count / max(n_vis, 1))

        seg_residual: dict[str, float] = {}
        seg_bad: dict[str, bool] = {}
        for seg_name, ref in reference_geometry.items():
            names_ref: list[str] = ref["marker_names"]
            P_loc = ref["P_local"]
            rows_loc: list[np.ndarray] = []
            rows_glob: list[np.ndarray] = []
            for stem, pl in zip(names_ref, P_loc, strict=True):
                if stem not in label_to_idx:
                    continue
                mi = label_to_idx[stem]
                g = points[f, mi, :]
                if not np.isfinite(g).all():
                    continue
                rows_loc.append(np.asarray(pl, dtype=np.float64))
                rows_glob.append(np.asarray(g, dtype=np.float64))
            if len(rows_loc) < 3:
                max_err = np.inf
            else:
                a_loc = np.stack(rows_loc, axis=0)
                a_glob = np.stack(rows_glob, axis=0)
                r, t = kabsch(a_loc, a_glob)
                pred = a_loc @ r.T + t
                err = np.linalg.norm(a_glob - pred, axis=1)
                max_err = float(np.max(err))
            seg_residual[seg_name] = max_err
            thr = seg_thresholds[seg_name]
            seg_bad[seg_name] = max_err > thr

        bad_residual = any(seg_bad.values())
        bad_vis = visible_ratio < visible_ratio_threshold
        is_bad = bad_residual or bad_vis

        diagnostics.append(
            {
                "frame": int(frames[f]),
                "frame_row": f,
                "segment_residual_mm": seg_residual,
                "segment_threshold_mm": {k: float(seg_thresholds[k]) for k in seg_thresholds},
                "segment_bad_residual": seg_bad,
                "visible_count": visible_count,
                "visible_ratio_denominator": int(n_vis),
                "visible_ratio": visible_ratio,
                "visible_ratio_threshold": float(visible_ratio_threshold),
                "bad_residual": bad_residual,
                "bad_visibility": bad_vis,
                "is_bad": is_bad,
            }
        )
    return diagnostics


def _interval_contains_k_consecutive_bad(
    bad: np.ndarray,
    k: int,
    a: int,
    b: int,
) -> bool:
    """True iff inclusive index range [a, b] contains a length-k run of True in ``bad``."""
    if k < 1 or b < a:
        return False
    for j in range(a, b - k + 2):
        if j + k - 1 > b:
            break
        if bool(np.all(bad[j : j + k])):
            return True
    return False


def find_trim_boundary(
    diagnostics: Sequence[Mapping[str, Any]],
    best_idx: int,
    *,
    consecutive_bad_frames: int,
) -> tuple[int, int]:
    """
    Row indices ``trim_start``, ``trim_end`` (half-open) containing ``best_idx``.

    Uses the **maximal** interval containing ``best_idx`` that does not contain ``K`` consecutive
    QC-bad frames (so expansion stops before absorbing a full ``K``-bad block).
    """
    n = len(diagnostics)
    if not (0 <= best_idx < n):
        raise ValueError(f"best_idx {best_idx} out of range for n_frames={n}")
    bad = np.array([bool(d["is_bad"]) for d in diagnostics], dtype=bool)
    k = int(consecutive_bad_frames)
    if k < 1:
        raise ValueError("consecutive_bad_frames must be >= 1")

    if _interval_contains_k_consecutive_bad(bad, k, best_idx, best_idx):
        raise ValueError(
            "Best frame is QC-bad and cannot be part of a valid trim interval under "
            f"consecutive_bad_frames={k}. Relax thresholds or override best_frame."
        )

    left, right = best_idx, best_idx
    while left > 0 and not _interval_contains_k_consecutive_bad(bad, k, left - 1, right):
        left -= 1
    while right < n - 1 and not _interval_contains_k_consecutive_bad(bad, k, left, right + 1):
        right += 1

    trim_start, trim_end = left, right + 1
    if not (trim_start <= best_idx < trim_end):
        raise RuntimeError(
            f"Trim invariant failed: best_idx={best_idx} not in [{trim_start}, {trim_end})"
        )
    return trim_start, trim_end


def save_trimmed_csv(
    metadata: dict[str, Any],
    trim_start_row: int,
    trim_end_row: int,
    output_csv_path: str | Path,
) -> None:
    """Write rows ``[trim_start_row, trim_end_row)`` preserving the original header line."""
    out = Path(output_csv_path)
    rows = metadata["data_rows"]
    header_line = metadata["original_header_line"]
    slice_rows = rows[trim_start_row:trim_end_row]
    with open(out, "w", newline="") as f:
        f.write(header_line)
        w = csv.writer(f)
        for row in slice_rows:
            w.writerow(row)


def save_bestframe_sidecar(output_csv_path: str | Path, best_frame_1based: int) -> None:
    """Write ``*.csv.bestframe`` with the same 1-based convention as the pipeline."""
    path = bestframe_sidecar_path(output_csv_path)
    path.write_text(str(int(best_frame_1based)))


def trim_trial(
    csv_path: str | Path,
    output_csv_path: str | Path,
    segment_markers_dict: Mapping[str, Sequence[str]],
    *,
    best_frame: int | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Trim a labeled CSV to a contiguous range around the pipeline best frame.

    Parameters
    ----------
    csv_path : labeled CSV path; sidecar ``<csv>.bestframe`` supplies 1-based best frame unless
        ``best_frame`` is set.
    output_csv_path : trimmed CSV path.
    segment_markers_dict : segment name -> ordered marker names (≥3 per segment for reference).
    best_frame : optional **1-based** frame index (same as sidecar). Overrides sidecar when set.
    config : optional overrides for thresholds and flags:

        - ``residual_threshold_mm`` (default 15)
        - ``residual_threshold_thigh_mm`` (default 25)
        - ``thigh_segment_keywords`` (default ``('thigh','femur')``)
        - ``visible_ratio_threshold`` (default 0.9)
        - ``consecutive_bad_frames`` (default 5)
        - ``min_trial_length`` (default 100; warn if trim shorter)
        - ``collinearity_eps_mm`` (default 1e-3)
        - ``exclude_obstacles_not_in_segments`` (default True)
        - ``visibility_ratio_marker_subset`` — ``"legs-feet"`` (default): ``visible_ratio`` uses only
          markers in :data:`LEGS_FEET_VISIBILITY_STEMS` present in the CSV (falls back to segment union
          if none match). ``"segment-union"``: ratio over the same marker set as ``build_visibility_union_indices``.

    Returns
    -------
    dict with keys:

        - ``trim_start``, ``trim_end``: **original** ``frame`` column values; ``trim_end`` is exclusive.
        - ``best_frame_1based``: int, same as sidecar / pipeline convention.
        - ``best_frame_idx``: int, 0-based row index into the **original** CSV.
        - ``trim_start_row``, ``trim_end_row``: half-open row indices into original CSV.
        - ``reference_geometry``, ``diagnostics``, ``metadata`` (parsed), ``config_resolved``
    """
    cfg = dict(config or {})
    residual_threshold_mm = float(cfg.get("residual_threshold_mm", DEFAULT_RESIDUAL_THRESHOLD_MM))
    residual_threshold_thigh_mm = float(
        cfg.get("residual_threshold_thigh_mm", DEFAULT_RESIDUAL_THRESHOLD_THIGH_MM)
    )
    thigh_keywords = tuple(
        str(x).lower() for x in cfg.get("thigh_segment_keywords", DEFAULT_THIGH_SEGMENT_KEYWORDS)
    )
    visible_ratio_threshold = float(
        cfg.get("visible_ratio_threshold", DEFAULT_VISIBLE_RATIO_THRESHOLD)
    )
    consecutive_bad = int(cfg.get("consecutive_bad_frames", DEFAULT_CONSECUTIVE_BAD_FRAMES))
    min_trial_len = int(cfg.get("min_trial_length", DEFAULT_MIN_TRIAL_LENGTH))
    col_eps = float(cfg.get("collinearity_eps_mm", DEFAULT_COLLINEARITY_EPS_MM))
    exclude_obs = bool(cfg.get("exclude_obstacles_not_in_segments", True))

    _, metadata = parse_labeled_csv(csv_path)
    frames: np.ndarray = metadata["frames"]
    n_frames = int(metadata["n_frames"])
    if n_frames == 0:
        raise ValueError("CSV has no data rows")

    if best_frame is not None:
        best_1 = int(best_frame)
    else:
        best_1 = load_best_frame_1based(csv_path)

    if best_1 < frames[0] or best_1 > frames[-1]:
        raise ValueError(
            f"best_frame 1-based {best_1} outside CSV frame range [{frames[0]}, {frames[-1]}]"
        )
    best_idx = int(best_1 - int(frames[0]))
    if best_idx < 0 or best_idx >= n_frames or int(frames[best_idx]) != best_1:
        raise ValueError(
            f"best_frame 1-based {best_1} not found in frame column (non-contiguous or missing)"
        )

    for seg, names in segment_markers_dict.items():
        if len(names) < 3:
            raise ValueError(f"Segment {seg!r} must list at least 3 markers")

    vis_indices, vis_labels_union = build_visibility_union_indices(
        metadata,
        segment_markers_dict,
        exclude_obstacles_not_in_segments=exclude_obs,
    )
    if not vis_indices:
        raise ValueError("No markers in visibility union (check segment_markers_dict vs CSV stems)")

    vis_ratio_idx, _vis_ratio_labels = resolve_visibility_ratio_marker_indices(
        metadata,
        vis_indices,
        vis_labels_union,
        cfg,
    )
    if not vis_ratio_idx:
        raise ValueError(
            "No markers for visible_ratio after resolve; check CSV columns vs segment_markers_dict"
        )

    ref_geom = compute_reference_geometry(
        metadata,
        segment_markers_dict,
        best_idx,
        collinearity_eps_mm=col_eps,
    )
    diagnostics = compute_diagnostics(
        metadata,
        ref_geom,
        segment_markers_dict,
        vis_ratio_idx,
        residual_threshold_mm=residual_threshold_mm,
        residual_threshold_thigh_mm=residual_threshold_thigh_mm,
        thigh_segment_keywords=thigh_keywords,
        visible_ratio_threshold=visible_ratio_threshold,
    )

    # Sanity: residuals ~ 0 at best frame
    d0 = diagnostics[best_idx]
    for sn, val in d0["segment_residual_mm"].items():
        if not np.isfinite(val) or val > 1e-5:
            logger.warning("Large residual at best frame for %s: %s mm", sn, val)

    trim_start_row, trim_end_row = find_trim_boundary(
        diagnostics,
        best_idx,
        consecutive_bad_frames=consecutive_bad,
    )

    trim_len = trim_end_row - trim_start_row
    if trim_len < min_trial_len:
        warnings.warn(
            f"Trimmed trial length {trim_len} rows < min_trial_length={min_trial_len}",
            UserWarning,
            stacklevel=2,
        )

    trim_start_frame = int(frames[trim_start_row])
    if trim_end_row <= trim_start_row:
        raise RuntimeError("trim_end_row must be greater than trim_start_row")
    trim_end_frame = int(frames[trim_end_row - 1]) + 1

    if not (trim_start_frame <= best_1 < trim_end_frame):
        raise RuntimeError(
            f"Invariant failed: best_frame_1based={best_1} not in "
            f"[{trim_start_frame}, {trim_end_frame})"
        )

    save_trimmed_csv(metadata, trim_start_row, trim_end_row, output_csv_path)
    save_bestframe_sidecar(output_csv_path, best_1)

    v_sub_raw = cfg.get("visibility_ratio_marker_subset", "legs-feet")
    cfg_resolved = {
        "residual_threshold_mm": residual_threshold_mm,
        "residual_threshold_thigh_mm": residual_threshold_thigh_mm,
        "thigh_segment_keywords": thigh_keywords,
        "visible_ratio_threshold": visible_ratio_threshold,
        "visibility_ratio_marker_subset": str(v_sub_raw),
        "n_visibility_ratio_markers": len(vis_ratio_idx),
        "consecutive_bad_frames": consecutive_bad,
        "min_trial_length": min_trial_len,
        "collinearity_eps_mm": col_eps,
        "exclude_obstacles_not_in_segments": exclude_obs,
    }
    return {
        "trim_start": trim_start_frame,
        "trim_end": trim_end_frame,
        "best_frame_1based": best_1,
        "best_frame_idx": best_idx,
        "trim_start_row": trim_start_row,
        "trim_end_row": trim_end_row,
        "reference_geometry": ref_geom,
        "diagnostics": diagnostics,
        "metadata": metadata,
        "config_resolved": cfg_resolved,
    }


def plot_trim_diagnostics(
    diagnostics: Sequence[Mapping[str, Any]],
    metadata: dict[str, Any],
    *,
    best_frame_1based: int,
    trim_start: int,
    trim_end: int,
    save_path: str | Path | None = None,
) -> Any:
    """
    Three stacked plots vs original ``frame`` column: max segment residual (+ threshold), visible ratio, LASI Z.

    Vertical lines: best frame, trim start, trim end (``trim_end`` is exclusive frame value).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "plot_trim_diagnostics requires matplotlib; install with: pip install matplotlib"
        ) from e

    frames = np.array([int(d["frame"]) for d in diagnostics], dtype=np.int64)
    max_res = np.array(
        [max(d["segment_residual_mm"].values()) for d in diagnostics],
        dtype=np.float64,
    )
    max_thr = np.array(
        [max(d["segment_threshold_mm"].values()) for d in diagnostics],
        dtype=np.float64,
    )
    vis = np.array([float(d["visible_ratio"]) for d in diagnostics])

    label_to_idx: dict[str, int] = metadata["label_to_marker_idx"]
    points: np.ndarray = metadata["points"]
    lasi_key = next((k for k in label_to_idx if k.strip().upper() == "LASI"), None)
    if lasi_key is not None:
        li = label_to_idx[lasi_key]
        pelvis_z = points[:, li, 2]
    else:
        pelvis_z = np.full(len(frames), np.nan)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    ax0, ax1, ax2 = axes

    ax0.plot(frames, max_res, label="max segment residual (mm)")
    ax0.plot(frames, max_thr, "--", label="max segment threshold (mm)")
    ax0.set_ylabel("mm")
    ax0.legend(loc="upper right", fontsize=8)
    ax0.set_title("Trial trim diagnostics")

    ax1.plot(frames, vis, label="visible ratio")
    thr_v = float(diagnostics[0]["visible_ratio_threshold"])
    ax1.axhline(thr_v, color="gray", linestyle="--", label=f"threshold ({thr_v:.2f})")
    ax1.set_ylabel("ratio")
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc="lower right", fontsize=8)

    ax2.plot(frames, pelvis_z, label="LASI z" if lasi_key else "LASI z (missing)")
    ax2.set_ylabel("mm")
    ax2.set_xlabel("frame")
    ax2.legend(loc="upper right", fontsize=8)

    for ax in axes:
        for x, sty in (
            (best_frame_1based, "g"),
            (trim_start, "b"),
            (trim_end, "r"),
        ):
            ax.axvline(x, color=sty, linewidth=0.8, alpha=0.7)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(Path(save_path), dpi=150)
    return fig


def main() -> None:
    """Minimal CLI: trim one CSV using a segment preset or JSON file (default = legs/feet only)."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description=(
            "Trim a labeled CSV around the pipeline best frame (Kabsch segment residuals + visibility). "
            "Reads/writes *.csv.bestframe next to the CSV."
        )
    )
    parser.add_argument("input_csv", help="Labeled CSV path")
    parser.add_argument("-o", "--output", required=True, help="Output trimmed CSV path")
    parser.add_argument(
        "--preset",
        choices=sorted(TRIM_SEGMENT_PRESET_CHOICES),
        default="default",
        help=(
            "Segment set: 'default' / 'legs-feet' = thigh/shank/foot only (no pelvis segment); "
            "'lower-body' = optional pelvis QC (LASI/RASI/LTHI + legs/feet, no PSIS); "
            "'full-body' = SEGMENTS with ≥3 markers (old default). Ignored if --segments-json is set."
        ),
    )
    parser.add_argument(
        "--segments-json",
        default=None,
        help=(
            'JSON object: {"L_Thigh": ["LASI","LTHI","LKNE"], ...}. Overrides --preset when set.'
        ),
    )
    parser.add_argument(
        "--best-frame",
        type=int,
        default=None,
        help="Override best frame (1-based). Default: read input sidecar.",
    )
    parser.add_argument(
        "--diagnostics-json",
        default=None,
        help="Optional path to write per-frame diagnostics JSON",
    )
    parser.add_argument(
        "--plot",
        default=None,
        metavar="PNG",
        help="Optional path to save diagnostic figure (requires matplotlib)",
    )
    parser.add_argument(
        "--min-trial-length",
        type=int,
        default=DEFAULT_MIN_TRIAL_LENGTH,
        help="Warn if trimmed row count is below this (default: %(default)s)",
    )
    parser.add_argument(
        "--visible-ratio-threshold",
        type=float,
        default=DEFAULT_VISIBLE_RATIO_THRESHOLD,
        metavar="F",
        help=(
            "Frame is bad if visible marker fraction is below F (default 0.9). "
            "Numerator/denominator follow --visibility-ratio-marker-subset."
        ),
    )
    parser.add_argument(
        "--visibility-ratio-marker-subset",
        choices=("legs-feet", "segment-union"),
        default="legs-feet",
        help=(
            "Which markers define visible_ratio: 'legs-feet' = only thigh/shank/foot stems present "
            "in the CSV (falls back to segment union if none match); 'segment-union' = all markers "
            "used in segment_markers_dict."
        ),
    )
    args = parser.parse_args()

    if args.segments_json:
        if args.preset != "default":
            print(
                "Warning: --segments-json is set; ignoring --preset "
                f"{args.preset!r}.",
                file=sys.stderr,
            )
        seg = json.loads(Path(args.segments_json).read_text())
        if not isinstance(seg, dict):
            print("segments-json must be a JSON object", file=sys.stderr)
            raise SystemExit(2)
        segment_dict = {str(k): list(v) for k, v in seg.items()}
    else:
        segment_dict = segment_markers_dict_for_trim_preset(args.preset)
        if args.preset == "full-body":
            from .segments import SEGMENTS

            skipped = sorted(k for k, v in SEGMENTS.items() if len(v) < 3)
            if skipped:
                print(
                    "Note: skipping SEGMENTS with fewer than 3 markers (rigid trim needs ≥3): "
                    + ", ".join(skipped),
                    file=sys.stderr,
                )
        if not segment_dict:
            print(
                "No segments available for the chosen preset. Use --segments-json.",
                file=sys.stderr,
            )
            raise SystemExit(2)

    result = trim_trial(
        args.input_csv,
        args.output,
        segment_dict,
        best_frame=args.best_frame,
        config={
            "min_trial_length": int(args.min_trial_length),
            "visible_ratio_threshold": float(args.visible_ratio_threshold),
            "visibility_ratio_marker_subset": str(args.visibility_ratio_marker_subset),
        },
    )
    print(
        json.dumps(
            {
                "trim_start": result["trim_start"],
                "trim_end": result["trim_end"],
                "best_frame_1based": result["best_frame_1based"],
                "trim_start_row": result["trim_start_row"],
                "trim_end_row": result["trim_end_row"],
            },
            indent=2,
        )
    )
    if args.diagnostics_json:
        # Drop non-JSON-serializable metadata
        out_diag = []
        for d in result["diagnostics"]:
            out_diag.append({k: v for k, v in d.items() if k != "frame_row"})
        Path(args.diagnostics_json).write_text(json.dumps(out_diag, indent=2))
    if args.plot:
        plot_trim_diagnostics(
            result["diagnostics"],
            result["metadata"],
            best_frame_1based=result["best_frame_1based"],
            trim_start=result["trim_start"],
            trim_end=result["trim_end"],
            save_path=args.plot,
        )


if __name__ == "__main__":
    main()
