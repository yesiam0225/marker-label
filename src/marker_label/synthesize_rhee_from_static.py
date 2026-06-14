"""
Synthesize missing RHEE from RANK and RTOE using right-foot geometry from a labeled static CSV.

Foot geometry uses ankle, toe, and tibia (when available): x along ankle→toe, y toward tibia
(shin axis in the foot plane), z completing a right-handed basis. Static trial fixes the heel
offset in that basis.

Limitations
-------------
- Assumes the foot does not invert relative to the static medial reference (ankle–tibia
  direction when available) used to orient the basis; large body turns or extreme foot roll
  can bias the heel.
- ``--vertical`` must not be parallel to the ankle-toe segment on typical frames when
  tibia markers are missing (falls back to lab vertical).
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .trial_trim import parse_labeled_csv


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("Zero-length vector in foot basis")
    return v / n


def ankle_tibia_name(ankle_name: str) -> str:
    """``LANK`` -> ``LTIB``, ``RANK`` -> ``RTIB``."""
    s = str(ankle_name).strip()
    if s.endswith("ANK"):
        return s[:-3] + "TIB"
    raise ValueError(f"Expected an ankle marker name (*ANK), got {ankle_name!r}")


def shin_axis_ankle_toe_tibia(
    ankle: np.ndarray,
    toe: np.ndarray,
    tibia: np.ndarray,
) -> np.ndarray | None:
    """
    Ankle→tibia direction with the ankle→toe component removed (shin in the foot plane).

    Returns ``None`` when tibia is colinear with the ankle-toe axis.
    """
    ank = np.asarray(ankle, dtype=np.float64).reshape(3)
    t = np.asarray(toe, dtype=np.float64).reshape(3)
    ti = np.asarray(tibia, dtype=np.float64).reshape(3)
    x_ax = _unit(t - ank)
    shin = ti - ank - float(np.dot(ti - ank, x_ax)) * x_ax
    n = float(np.linalg.norm(shin))
    if n < 1e-9:
        return None
    return shin / n


def medial_reference_ankle_toe_tibia(
    ankle: np.ndarray,
    toe: np.ndarray,
    tibia: np.ndarray,
) -> np.ndarray | None:
    """
    Unit vector in the plane perpendicular to ankle-toe, from ankle toward tibia (medial).

    Foot lateral/medial is defined relative to the shin marker, not lab Y or body center.
    Returns ``None`` when tibia is colinear with the ankle-toe axis.
    """
    ank = np.asarray(ankle, dtype=np.float64).reshape(3)
    t = np.asarray(toe, dtype=np.float64).reshape(3)
    ti = np.asarray(tibia, dtype=np.float64).reshape(3)
    x_ax = _unit(t - ank)
    med = ti - ank - float(np.dot(ti - ank, x_ax)) * x_ax
    n = float(np.linalg.norm(med))
    if n < 1e-9:
        return None
    return med / n


def foot_basis_from_ankle_toe(
    ankle: np.ndarray,
    toe: np.ndarray,
    lab_vertical: np.ndarray,
    *,
    tibia: np.ndarray | None = None,
    z_flip_reference: np.ndarray | None = None,
    shin_axis_reference: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Orthonormal right-handed basis (x, y, z) with origin at ankle.

    x: along ankle -> toe.
    When ``tibia`` is given, y follows the shin axis (ankle -> tibia, perpendicular to x) and
    z = x x y (mediolateral completion). Mediolateral sign is fixed with ``z_flip_reference``
    when provided (typically the static ankle-tibia medial direction).
    ``shin_axis_reference`` locks the shin (y) hemisphere to static when dynamic tibia geometry
    is near-degenerate during obstacle clearance.
    Without tibia, falls back to lab vertical for the foot plane.
    """
    g = np.asarray(lab_vertical, dtype=np.float64).reshape(3)
    g = _unit(g)
    ank = np.asarray(ankle, dtype=np.float64).reshape(3)
    t = np.asarray(toe, dtype=np.float64).reshape(3)
    x_ax = _unit(t - ank)
    used_tibia = False
    shin_ref_u: np.ndarray | None = None
    if shin_axis_reference is not None:
        sr = np.asarray(shin_axis_reference, dtype=np.float64).reshape(3)
        sn = float(np.linalg.norm(sr))
        if sn >= 1e-9:
            shin_ref_u = sr / sn
    if tibia is not None:
        shin_y = shin_axis_ankle_toe_tibia(ank, t, tibia)
        if shin_y is not None:
            used_tibia = True
            if shin_ref_u is not None and float(np.dot(shin_y, shin_ref_u)) < 0.0:
                shin_y = -shin_y
            y_ax = shin_y
            z_ax = _unit(np.cross(x_ax, y_ax))
            y_ax = np.cross(z_ax, x_ax)
    if not used_tibia:
        v_perp = g - float(np.dot(g, x_ax)) * x_ax
        nv = float(np.linalg.norm(v_perp))
        if nv < 1e-9:
            alt = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            v_perp = alt - float(np.dot(alt, x_ax)) * x_ax
            nv = float(np.linalg.norm(v_perp))
            if nv < 1e-9:
                raise ValueError(
                    "Cannot build perpendicular to ankle-toe axis (degenerate with lab vertical)"
                )
        v_perp = v_perp / nv
        z_ax = _unit(np.cross(x_ax, v_perp))
        y_ax = np.cross(z_ax, x_ax)
    if z_flip_reference is not None:
        ref = np.asarray(z_flip_reference, dtype=np.float64).reshape(3)
        if float(np.dot(z_ax, ref)) < 0.0:
            z_ax = -z_ax
            y_ax = np.cross(z_ax, x_ax)
    if shin_ref_u is not None and used_tibia:
        if float(np.dot(y_ax, shin_ref_u)) < 0.0:
            y_ax = -y_ax
            z_ax = _unit(np.cross(x_ax, y_ax))
            y_ax = np.cross(z_ax, x_ax)
    return x_ax, y_ax, z_ax


def static_rhee_coefficients(
    points: np.ndarray,
    label_to_idx: dict[str, int],
    lab_vertical: np.ndarray,
    *,
    rank_name: str = "RANK",
    rtoe_name: str = "RTOE",
    rhee_name: str = "RHEE",
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Mean ankle/toe/heel (and tibia when present) over finite frames; return
    ``(coeffs, heel_side_ref, shin_axis_ref)``.

    coeffs are (3,) such that RHEE ≈ RANK + c0*x + c1*y + c2*z with basis built from mean
    ankle, mean toe, and mean tibia when available.
    """
    ri, toe_i, hi = label_to_idx[rank_name], label_to_idx[rtoe_name], label_to_idx[rhee_name]
    n = points.shape[0]
    acc_a = []
    acc_t = []
    acc_h = []
    for f in range(n):
        a = points[f, ri, :]
        t = points[f, toe_i, :]
        h = points[f, hi, :]
        if np.isfinite(a).all() and np.isfinite(t).all() and np.isfinite(h).all():
            acc_a.append(a)
            acc_t.append(t)
            acc_h.append(h)
    if len(acc_a) < 1:
        raise ValueError(
            f"No frame in static CSV with finite {rank_name}, {rtoe_name}, and {rhee_name}."
        )
    ank_m = np.mean(np.stack(acc_a, axis=0), axis=0)
    toe_m = np.mean(np.stack(acc_t, axis=0), axis=0)
    heel_m = np.mean(np.stack(acc_h, axis=0), axis=0)
    tib_name = ankle_tibia_name(rank_name)
    tib_m: np.ndarray | None = None
    if tib_name in label_to_idx:
        tib_i = label_to_idx[tib_name]
        acc_tb: list[np.ndarray] = []
        for f in range(n):
            a = points[f, ri, :]
            t = points[f, toe_i, :]
            h = points[f, hi, :]
            tb = points[f, tib_i, :]
            if (
                np.isfinite(a).all()
                and np.isfinite(t).all()
                and np.isfinite(h).all()
                and np.isfinite(tb).all()
            ):
                acc_tb.append(tb)
        if not acc_tb:
            for f in range(n):
                tb = points[f, tib_i, :]
                if np.isfinite(tb).all():
                    acc_tb.append(tb)
        if acc_tb:
            tib_m = np.mean(np.stack(acc_tb, axis=0), axis=0)
    basis_kw: dict[str, Any] = {"z_flip_reference": None}
    shin_ref: np.ndarray | None = None
    if tib_m is not None:
        shin_ref = shin_axis_ankle_toe_tibia(ank_m, toe_m, tib_m)
        med_ref = medial_reference_ankle_toe_tibia(ank_m, toe_m, tib_m)
        if med_ref is not None:
            basis_kw["tibia"] = tib_m
            basis_kw["z_flip_reference"] = med_ref
            if shin_ref is not None:
                basis_kw["shin_axis_reference"] = shin_ref
    x_ax, y_ax, z_ax = foot_basis_from_ankle_toe(ank_m, toe_m, lab_vertical, **basis_kw)
    r = np.stack([x_ax, y_ax, z_ax], axis=1)
    rel = heel_m - ank_m
    coeffs = r.T @ rel
    heel_side_ref = (
        basis_kw["z_flip_reference"] if basis_kw["z_flip_reference"] is not None else heel_m - ank_m
    )
    return coeffs, heel_side_ref, shin_ref


def predict_rhee_row(
    rank_xyz: np.ndarray,
    rtoe_xyz: np.ndarray,
    coeffs: np.ndarray,
    lab_vertical: np.ndarray,
    heel_side_ref: np.ndarray,
    *,
    tibia_xyz: np.ndarray | None = None,
    shin_axis_reference: np.ndarray | None = None,
) -> np.ndarray:
    """Single-frame RHEE position (3,) or NaN if inputs invalid."""
    if not (np.isfinite(rank_xyz).all() and np.isfinite(rtoe_xyz).all()):
        return np.full(3, np.nan, dtype=np.float64)
    basis_kw: dict[str, Any] = {"z_flip_reference": heel_side_ref}
    if shin_axis_reference is not None:
        basis_kw["shin_axis_reference"] = shin_axis_reference
    if tibia_xyz is not None and np.isfinite(tibia_xyz).all():
        basis_kw["tibia"] = tibia_xyz
    x_ax, y_ax, z_ax = foot_basis_from_ankle_toe(
        rank_xyz, rtoe_xyz, lab_vertical, **basis_kw
    )
    r = np.stack([x_ax, y_ax, z_ax], axis=1)
    return rank_xyz + r @ coeffs


def _header_cells_from_line(line: str) -> list[str]:
    reader = csv.reader(io.StringIO(line.strip("\n")))
    return next(reader)


def _sync_data_rows_from_points(meta: dict[str, Any]) -> None:
    stem_to_triplet: dict[str, tuple[int, int, int]] = meta["stem_to_col_triplet"]
    all_stems: list[str] = meta["all_stems"]
    points: np.ndarray = meta["points"]
    rows: list[list[str]] = meta["data_rows"]
    for r in range(int(meta["n_frames"])):
        row = rows[r]
        for si, stem in enumerate(all_stems):
            ix, iy, iz = stem_to_triplet[stem]
            for k, j in enumerate((ix, iy, iz)):
                v = points[r, si, k]
                row[j] = "" if not np.isfinite(v) else str(float(v))


def _rhee_header_names(rank_x_cell: str) -> tuple[str, str, str]:
    """Match RANK column stem padding for RHEE_* names."""
    s = str(rank_x_cell).strip()
    if not s.endswith("_x"):
        raise ValueError(f"Expected an *_x header cell for RANK, got {rank_x_cell!r}")
    prefix = s[:-2]
    if "RANK" not in prefix:
        raise ValueError(f"RANK _x cell must contain 'RANK', got {rank_x_cell!r}")
    rhee_prefix = prefix.replace("RANK", "RHEE", 1)
    return f"{rhee_prefix}_x", f"{rhee_prefix}_y", f"{rhee_prefix}_z"


def synthesize_rhee_csv(
    static_csv: str | Path,
    dynamic_csv: str | Path,
    output_csv: str | Path,
    lab_vertical: tuple[float, float, float] = (0.0, 1.0, 0.0),
) -> dict[str, Any]:
    """
    Read static and dynamic labeled CSVs; write ``output_csv`` with RHEE filled or inserted.

    If dynamic already has an RHEE triplet, only rows where RHEE is missing (any coord NaN)
    and RANK/RTOE are finite are overwritten.
    """
    vert = np.array(lab_vertical, dtype=np.float64).reshape(3)
    _, meta_s = parse_labeled_csv(static_csv)
    for name in ("RANK", "RTOE", "RHEE"):
        if name not in meta_s["label_to_marker_idx"]:
            raise ValueError(f"Static CSV missing marker column {name!r}")

    coeffs, heel_ref, shin_ref = static_rhee_coefficients(
        meta_s["points"], meta_s["label_to_marker_idx"], vert
    )

    path_d = Path(dynamic_csv)
    with open(path_d, newline="") as f:
        header_line = f.readline()
        rest = f.read()
    header_cells = _header_cells_from_line(header_line)
    if len(header_cells) < 2 or header_cells[0].strip().lower() != "frame":
        raise ValueError("Dynamic CSV must start with frame, time, ...")

    stem_to_triplet: dict[str, tuple[int, int, int]] = {}
    i = 2
    while i + 2 < len(header_cells):
        xs = str(header_cells[i])
        if not (xs.endswith("_x") and str(header_cells[i + 1]).endswith("_y")):
            break
        stem = xs[:-2].strip()
        stem_to_triplet[stem] = (i, i + 1, i + 2)
        i += 3

    if "RANK" not in stem_to_triplet or "RTOE" not in stem_to_triplet:
        raise ValueError("Dynamic CSV must include RANK and RTOE marker triplets")

    rtib_ix: tuple[int, int, int] | None = None
    if "RTIB" in stem_to_triplet:
        rtib_ix = stem_to_triplet["RTIB"]

    has_rhee = "RHEE" in stem_to_triplet
    rank_ix = stem_to_triplet["RANK"][0]

    if not has_rhee:
        rx_cell = header_cells[rank_ix]
        hx, hy, hz = _rhee_header_names(rx_cell)
        insert_at = rank_ix + 3
        new_header = header_cells[:insert_at] + [hx, hy, hz] + header_cells[insert_at:]
        reader = csv.reader(io.StringIO(rest))
        new_rows: list[list[str]] = []
        for row in reader:
            if not any(c.strip() for c in row):
                continue
            while len(row) < len(header_cells):
                row.append("")
            try:
                rank_xyz = np.array(
                    [float(row[rank_ix]), float(row[rank_ix + 1]), float(row[rank_ix + 2])],
                    dtype=np.float64,
                )
                rtoe_xyz = np.array(
                    [
                        float(row[insert_at]),
                        float(row[insert_at + 1]),
                        float(row[insert_at + 2]),
                    ],
                    dtype=np.float64,
                )
            except ValueError:
                rank_xyz = np.full(3, np.nan, dtype=np.float64)
                rtoe_xyz = np.full(3, np.nan, dtype=np.float64)
            tibia_xyz = None
            if rtib_ix is not None:
                try:
                    tibia_xyz = np.array(
                        [float(row[rtib_ix[0]]), float(row[rtib_ix[1]]), float(row[rtib_ix[2]])],
                        dtype=np.float64,
                    )
                except ValueError:
                    tibia_xyz = None
            pred = predict_rhee_row(
                rank_xyz, rtoe_xyz, coeffs, vert, heel_ref,
                tibia_xyz=tibia_xyz, shin_axis_reference=shin_ref,
            )
            head = row[:insert_at]
            tail = row[insert_at:]
            if np.isfinite(pred).all():
                rhee_cells = [str(float(v)) for v in pred]
            else:
                rhee_cells = ["", "", ""]
            new_rows.append(head + rhee_cells + tail)
        header_cells = new_header
        data_rows = new_rows
    else:
        _, meta_d = parse_labeled_csv(dynamic_csv)
        pts = meta_d["points"]
        li = meta_d["label_to_marker_idx"]
        ri, ti, hi = li["RANK"], li["RTOE"], li["RHEE"]
        tib_i = li.get("RTIB")
        for f in range(int(meta_d["n_frames"])):
            cur = pts[f, hi, :]
            if np.isfinite(cur).all():
                continue
            tibia_xyz = pts[f, tib_i, :] if tib_i is not None else None
            pred = predict_rhee_row(
                pts[f, ri, :],
                pts[f, ti, :],
                coeffs,
                vert,
                heel_ref,
                tibia_xyz=tibia_xyz,
                shin_axis_reference=shin_ref,
            )
            if np.isfinite(pred).all():
                pts[f, hi, :] = pred
        _sync_data_rows_from_points(meta_d)
        header_cells = _header_cells_from_line(meta_d["original_header_line"])
        data_rows = meta_d["data_rows"]

    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header_cells)
        for row in data_rows:
            if len(row) < len(header_cells):
                row = row + [""] * (len(header_cells) - len(row))
            w.writerow(row[: len(header_cells)])

    return {
        "output_csv": str(out),
        "coeffs_ankle_basis": coeffs.tolist(),
        "inserted_rhee_column": not has_rhee,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Add or fill RHEE in a labeled dynamic CSV using RANK, RTOE, and static foot geometry."
        )
    )
    parser.add_argument("static_csv", help="Labeled static CSV with RANK, RTOE, RHEE")
    parser.add_argument("dynamic_csv", help="Labeled dynamic CSV with RANK and RTOE (RHEE optional)")
    parser.add_argument("-o", "--output", required=True, help="Output CSV path")
    parser.add_argument(
        "--vertical",
        type=str,
        default="0,1,0",
        metavar="X,Y,Z",
        help="Lab up direction for foot plane (default 0,1,0). Must not be parallel to ankle-toe.",
    )
    args = parser.parse_args()
    parts = [float(p.strip()) for p in str(args.vertical).split(",")]
    if len(parts) != 3:
        print("Error: --vertical must be three comma-separated numbers.", file=sys.stderr)
        raise SystemExit(2)
    try:
        info = synthesize_rhee_csv(args.static_csv, args.dynamic_csv, args.output, tuple(parts))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(2) from e
    print(f"Wrote {info['output_csv']} (inserted_new_rhee_column={info['inserted_rhee_column']})")


if __name__ == "__main__":
    main()
