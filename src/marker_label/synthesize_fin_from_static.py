"""
Synthesize missing LFIN/RFIN from forearm/wrist markers using static hand geometry.

Default ``forearm`` method: static FIN offset in the RFRM–RWRA–RWRB local frame
(same 3-point basis as shoulder-from-thorax). Falls back to legacy WRA+WRB +
lab-vertical basis when the forearm marker is missing on a frame.

Static may be a labeled flat ``.csv`` or labeled anatomical ``.c3d``.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from marker_label.gap_filling.reference import (
    _global_to_local,
    _local_coords_from_three,
    _local_to_global,
)
from marker_label.gap_filling.static_reference import _load_static_points_and_label_idx
from marker_label.gap_filling.two_marker_static import (
    body_offset_from_static_means,
    predict_from_two_anchors,
)

from .trial_trim import parse_labeled_csv

HandSide = Literal["L", "R"]
FinMethod = Literal["forearm", "wrist", "auto"]

HAND_MARKERS: dict[HandSide, tuple[str, str, str]] = {
    "L": ("LWRA", "LWRB", "LFIN"),
    "R": ("RWRA", "RWRB", "RFIN"),
}

FOREARM_FRAME: dict[HandSide, tuple[str, str, str]] = {
    "L": ("LFRM", "LWRA", "LWRB"),
    "R": ("RFRM", "RWRA", "RWRB"),
}


@dataclass
class FinSideModel:
    forearm_local: np.ndarray | None
    wrist_offset: np.ndarray | None
    wrist_flip: np.ndarray | None
    method_primary: str


def _markers_for_side(side: HandSide) -> tuple[str, str, str, str, str, str]:
    wra, wrb, fin = HAND_MARKERS[side]
    frm, f_wra, f_wrb = FOREARM_FRAME[side]
    return frm, f_wra, f_wrb, wra, wrb, fin


def _mean_finite_rows(
    points: np.ndarray,
    label_to_idx: dict[str, int],
    names: tuple[str, ...],
) -> dict[str, np.ndarray] | None:
    acc: dict[str, list[np.ndarray]] = {n: [] for n in names}
    for f in range(points.shape[0]):
        vecs: dict[str, np.ndarray] = {}
        ok = True
        for n in names:
            p = points[f, label_to_idx[n], :]
            if not np.isfinite(p).all():
                ok = False
                break
            vecs[n] = p
        if ok:
            for n in names:
                acc[n].append(vecs[n])
    if not acc[names[0]]:
        return None
    return {n: np.mean(np.stack(v, axis=0), axis=0) for n, v in acc.items()}


def static_forearm_fin_local(
    points: np.ndarray,
    label_to_idx: dict[str, int],
    *,
    side: HandSide,
) -> np.ndarray:
    frm, wra, wrb, _, _, fin = _markers_for_side(side)
    for name in (frm, wra, wrb, fin):
        if name not in label_to_idx:
            raise ValueError(f"Static trial missing marker {name!r}")
    locals_acc: list[np.ndarray] = []
    for f in range(points.shape[0]):
        if not all(np.isfinite(points[f, label_to_idx[m], :]).all() for m in (frm, wra, wrb, fin)):
            continue
        p0 = points[f, label_to_idx[frm], :]
        p1 = points[f, label_to_idx[wra], :]
        p2 = points[f, label_to_idx[wrb], :]
        try:
            origin, basis = _local_coords_from_three(p0, p1, p2)
        except ValueError:
            continue
        locals_acc.append(_global_to_local(points[f, label_to_idx[fin], :], origin, basis))
    if not locals_acc:
        raise ValueError(f"No static frame with finite {frm}, {wra}, {wrb}, and {fin}.")
    return np.mean(np.stack(locals_acc, axis=0), axis=0)


def static_wrist_fin_offset(
    points: np.ndarray,
    label_to_idx: dict[str, int],
    lab_vertical: np.ndarray,
    *,
    side: HandSide,
) -> tuple[np.ndarray, np.ndarray]:
    _, _, _, wra, wrb, fin = _markers_for_side(side)
    means = _mean_finite_rows(points, label_to_idx, (wra, wrb, fin))
    if means is None:
        raise ValueError(f"No static frame with finite {wra}, {wrb}, and {fin}.")
    off, flip, _ = body_offset_from_static_means(means, wra, wrb, fin, lab_vertical)
    return off, flip


def build_fin_side_model(
    static_points: np.ndarray,
    static_idx: dict[str, int],
    lab_vertical: np.ndarray,
    *,
    side: HandSide,
    method: FinMethod,
) -> FinSideModel:
    forearm_local: np.ndarray | None = None
    wrist_off: np.ndarray | None = None
    wrist_flip: np.ndarray | None = None
    primary = "wrist"

    if method in ("forearm", "auto"):
        try:
            forearm_local = static_forearm_fin_local(static_points, static_idx, side=side)
            primary = "forearm"
        except ValueError:
            if method == "forearm":
                raise

    if method in ("wrist", "auto"):
        wrist_off, wrist_flip = static_wrist_fin_offset(
            static_points, static_idx, lab_vertical, side=side
        )
        if forearm_local is None:
            primary = "wrist"

    if forearm_local is None and wrist_off is None:
        raise ValueError(f"Could not build FIN model for {side} hand.")

    return FinSideModel(
        forearm_local=forearm_local,
        wrist_offset=wrist_off,
        wrist_flip=wrist_flip,
        method_primary=primary,
    )


def predict_fin_forearm_row(
    frm_xyz: np.ndarray,
    wra_xyz: np.ndarray,
    wrb_xyz: np.ndarray,
    local_fin: np.ndarray,
) -> np.ndarray:
    if not (
        np.isfinite(frm_xyz).all()
        and np.isfinite(wra_xyz).all()
        and np.isfinite(wrb_xyz).all()
    ):
        return np.full(3, np.nan, dtype=np.float64)
    try:
        origin, basis = _local_coords_from_three(frm_xyz, wra_xyz, wrb_xyz)
    except ValueError:
        return np.full(3, np.nan, dtype=np.float64)
    return _local_to_global(local_fin, origin, basis)


def predict_fin_wrist_row(
    wra_xyz: np.ndarray,
    wrb_xyz: np.ndarray,
    offset: np.ndarray,
    lab_vertical: np.ndarray,
    flip_reference: np.ndarray,
) -> np.ndarray:
    if not (np.isfinite(wra_xyz).all() and np.isfinite(wrb_xyz).all()):
        return np.full(3, np.nan, dtype=np.float64)
    return predict_from_two_anchors(wra_xyz, wrb_xyz, offset, lab_vertical, flip_reference)


def predict_fin_row_from_model(
    model: FinSideModel,
    frm_xyz: np.ndarray,
    wra_xyz: np.ndarray,
    wrb_xyz: np.ndarray,
    lab_vertical: np.ndarray,
    *,
    method: FinMethod = "auto",
) -> tuple[np.ndarray, str]:
    """Return (position, method_used)."""
    use_forearm = method in ("forearm", "auto") and model.forearm_local is not None
    if use_forearm and np.isfinite(frm_xyz).all():
        pred = predict_fin_forearm_row(frm_xyz, wra_xyz, wrb_xyz, model.forearm_local)
        if np.isfinite(pred).all():
            return pred, "forearm"

    if method in ("wrist", "auto") and model.wrist_offset is not None and model.wrist_flip is not None:
        pred = predict_fin_wrist_row(
            wra_xyz, wrb_xyz, model.wrist_offset, lab_vertical, model.wrist_flip
        )
        if np.isfinite(pred).all():
            return pred, "wrist"

    return np.full(3, np.nan, dtype=np.float64), "none"


def load_static_for_hand(
    static_path: str | Path,
    sides: tuple[HandSide, ...],
) -> tuple[np.ndarray, dict[str, int]]:
    path = Path(static_path)
    need: set[str] = set()
    for side in sides:
        frm, wra, wrb, wra2, wrb2, fin = _markers_for_side(side)
        need.update({frm, wra, wrb, wra2, wrb2, fin})
    if path.suffix.lower() == ".csv":
        _, meta = parse_labeled_csv(path)
        missing = need - set(meta["label_to_marker_idx"])
        if missing:
            raise ValueError(f"Static CSV missing markers: {sorted(missing)}")
        return np.asarray(meta["points"], dtype=np.float64), dict(meta["label_to_marker_idx"])
    keys = {m: 0 for m in need}
    points, label_to_idx, _, _, _ = _load_static_points_and_label_idx(path, keys)
    missing = need - set(label_to_idx)
    if missing:
        raise ValueError(f"Static C3D missing markers: {sorted(missing)}")
    return points, label_to_idx


def _header_cells_from_line(line: str) -> list[str]:
    reader = csv.reader(io.StringIO(line.strip("\n")))
    return next(reader)


def _fin_header_names(wrb_x_cell: str, fin_name: str, wrb_name: str) -> tuple[str, str, str]:
    s = str(wrb_x_cell).strip()
    if not s.endswith("_x"):
        raise ValueError(f"Expected an *_x header cell for {wrb_name}, got {wrb_x_cell!r}")
    prefix = s[:-2]
    if wrb_name not in prefix:
        raise ValueError(f"{wrb_name} _x cell must contain {wrb_name!r}, got {wrb_x_cell!r}")
    fin_prefix = prefix.replace(wrb_name, fin_name, 1)
    return f"{fin_prefix}_x", f"{fin_prefix}_y", f"{fin_prefix}_z"


def _parse_marker_triplets(header_cells: list[str]) -> dict[str, tuple[int, int, int]]:
    stem_to_triplet: dict[str, tuple[int, int, int]] = {}
    i = 2
    while i + 2 < len(header_cells):
        xs = str(header_cells[i])
        if not (xs.endswith("_x") and str(header_cells[i + 1]).endswith("_y")):
            break
        stem = xs[:-2].strip()
        stem_to_triplet[stem] = (i, i + 1, i + 2)
        i += 3
    return stem_to_triplet


def _xyz_from_row(row: list[str], ix: int) -> np.ndarray:
    try:
        return np.array([float(row[ix]), float(row[ix + 1]), float(row[ix + 2])], dtype=np.float64)
    except (IndexError, ValueError):
        return np.full(3, np.nan, dtype=np.float64)


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


def _predict_from_meta_frame(
    meta: dict[str, Any],
    f: int,
    side: HandSide,
    model: FinSideModel,
    vert: np.ndarray,
    *,
    method: FinMethod,
) -> tuple[np.ndarray, str]:
    frm, _, _, wra, wrb, _ = _markers_for_side(side)
    li = meta["label_to_marker_idx"]
    frm_xyz = meta["points"][f, li[frm], :] if frm in li else np.full(3, np.nan)
    return predict_fin_row_from_model(
        model,
        frm_xyz,
        meta["points"][f, li[wra], :],
        meta["points"][f, li[wrb], :],
        vert,
        method=method,
    )


def synthesize_fin_csv(
    static_path: str | Path,
    dynamic_csv: str | Path,
    output_csv: str | Path,
    *,
    sides: tuple[HandSide, ...] = ("L", "R"),
    lab_vertical: tuple[float, float, float] = (0.0, 1.0, 0.0),
    method: FinMethod = "forearm",
    only_if_fin_missing: bool = False,
    fill_nan_in_existing: bool = True,
    replace_existing: bool = False,
) -> dict[str, Any]:
    vert = np.array(lab_vertical, dtype=np.float64).reshape(3)
    static_pts, static_idx = load_static_for_hand(static_path, sides)
    models: dict[HandSide, FinSideModel] = {
        side: build_fin_side_model(static_pts, static_idx, vert, side=side, method=method)
        for side in sides
    }

    path_d = Path(dynamic_csv)
    with open(path_d, newline="") as f:
        header_line = f.readline()
        rest = f.read()
    orig_header = _header_cells_from_line(header_line)
    if len(orig_header) < 2 or orig_header[0].strip().lower() != "frame":
        raise ValueError("Dynamic CSV must start with frame, time, ...")

    orig_triplets = _parse_marker_triplets(orig_header)
    inserted: dict[str, bool] = {HAND_MARKERS[s][2]: False for s in sides}
    methods_used: dict[str, int] = {}

    insert_plan: list[tuple[int, HandSide]] = []
    fill_sides: list[HandSide] = []
    for side in sides:
        _, _, _, wra, wrb, fin = _markers_for_side(side)
        if wra not in orig_triplets or wrb not in orig_triplets:
            raise ValueError(f"Dynamic CSV must include {wra} and {wrb} marker triplets")
        if fin not in orig_triplets:
            insert_plan.append((orig_triplets[wrb][0], side))
            inserted[fin] = True
        elif not only_if_fin_missing and (fill_nan_in_existing or replace_existing):
            fill_sides.append(side)

    reader = csv.reader(io.StringIO(rest))
    rows = [list(r) for r in reader if any(c.strip() for c in r)]

    header = list(orig_header)
    if insert_plan:
        insert_plan.sort(key=lambda t: t[0], reverse=True)
        for _wrb_ix, side in insert_plan:
            frm, _, _, wra, wrb, fin = _markers_for_side(side)
            triplets = _parse_marker_triplets(header)
            wra_ix, wrb_ix_cur = triplets[wra][0], triplets[wrb][0]
            frm_ix = triplets[frm][0] if frm in triplets else None
            hx, hy, hz = _fin_header_names(header[wrb_ix_cur], fin, wrb)
            insert_at = wrb_ix_cur + 3
            model = models[side]
            new_rows: list[list[str]] = []
            for row in rows:
                while len(row) < len(header):
                    row.append("")
                frm_xyz = _xyz_from_row(row, frm_ix) if frm_ix is not None else np.full(3, np.nan)
                pred, used = predict_fin_row_from_model(
                    model,
                    frm_xyz,
                    _xyz_from_row(row, wra_ix),
                    _xyz_from_row(row, wrb_ix_cur),
                    vert,
                    method=method,
                )
                methods_used[used] = methods_used.get(used, 0) + int(np.isfinite(pred).all())
                fin_cells = [str(float(v)) for v in pred] if np.isfinite(pred).all() else ["", "", ""]
                new_rows.append(row[:insert_at] + fin_cells + row[insert_at:])
            rows = new_rows
            header = header[:insert_at] + [hx, hy, hz] + header[insert_at:]

    filled_counts: dict[str, int] = {}

    def _fill_meta(meta_path: Path, sides_to_fill: tuple[HandSide, ...]) -> None:
        nonlocal header, rows, filled_counts
        _, meta = parse_labeled_csv(meta_path)
        pts = meta["points"]
        li = meta["label_to_marker_idx"]
        for side in sides_to_fill:
            frm, _, _, wra, wrb, fin = _markers_for_side(side)
            if fin not in li:
                continue
            model = models[side]
            n = 0
            for f in range(int(meta["n_frames"])):
                if (
                    not replace_existing
                    and np.isfinite(pts[f, li[fin], :]).all()
                ):
                    continue
                pred, used = _predict_from_meta_frame(meta, f, side, model, vert, method=method)
                if np.isfinite(pred).all():
                    pts[f, li[fin], :] = pred
                    methods_used[used] = methods_used.get(used, 0) + 1
                    n += 1
            filled_counts[fin] = n
        _sync_data_rows_from_points(meta)
        header = _header_cells_from_line(meta["original_header_line"])
        rows = meta["data_rows"]

    if fill_sides and not insert_plan:
        _fill_meta(path_d, tuple(fill_sides))
    elif fill_sides and insert_plan:
        tmp = Path(output_csv)
        tmp.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for row in rows:
                while len(row) < len(header):
                    row.append("")
                w.writerow(row[: len(header)])
        _fill_meta(tmp, tuple(fill_sides))

    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows:
            while len(row) < len(header):
                row.append("")
            w.writerow(row[: len(header)])

    return {
        "output_csv": str(out),
        "inserted": inserted,
        "filled_frames": filled_counts,
        "methods_used": methods_used,
        "models": {s: models[s].method_primary for s in sides},
        "sides": list(sides),
    }


def angle_at_vertex(p: np.ndarray, v: np.ndarray, w: np.ndarray) -> float:
    u, x = v - p, w - p
    nu, nx = float(np.linalg.norm(u)), float(np.linalg.norm(x))
    if nu < 1e-9 or nx < 1e-9:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(float(np.dot(u / nu, x / nx)), -1.0, 1.0))))


def compare_fin_methods_on_trial(
    static_path: str | Path,
    dynamic_csv: str | Path,
    *,
    side: HandSide = "R",
) -> dict[str, Any]:
    """Synthesize RFIN/LFIN with wrist vs forearm methods; return angle summaries."""
    static_pts, static_idx = load_static_for_hand(static_path, (side,))
    vert = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    _, meta = parse_labeled_csv(dynamic_csv)
    li = meta["label_to_marker_idx"]
    frm, _, _, wra, wrb, fin = _markers_for_side(side)

    models = {
        "wrist": build_fin_side_model(static_pts, static_idx, vert, side=side, method="wrist"),
        "forearm": build_fin_side_model(static_pts, static_idx, vert, side=side, method="forearm"),
    }

    static_ref = _mean_finite_rows(static_pts, static_idx, (frm, wra, wrb, fin))
    ref_angles: dict[str, float] = {}
    if static_ref:
        O, A, B = static_ref[wra], static_ref[wrb], static_ref[fin]
        ref_angles = {
            "static_WRA": angle_at_vertex(O, B, A),
            "static_WRB": angle_at_vertex(A, B, O),
            "static_FIN": angle_at_vertex(B, O, A),
        }

    out: dict[str, Any] = {"static_ref_angles_deg": ref_angles, "methods": {}}
    for name, model in models.items():
        angs_wra: list[float] = []
        angs_frm: list[float] = []
        n_forearm = n_wrist = n_frames = 0
        for f in range(int(meta["n_frames"])):
            frm_xyz = meta["points"][f, li[frm], :] if frm in li else np.full(3, np.nan)
            wra_xyz = meta["points"][f, li[wra], :]
            wrb_xyz = meta["points"][f, li[wrb], :]
            pred, used = predict_fin_row_from_model(
                model, frm_xyz, wra_xyz, wrb_xyz, vert, method=name  # type: ignore[arg-type]
            )
            if not np.isfinite(pred).all():
                continue
            n_frames += 1
            if used == "forearm":
                n_forearm += 1
            elif used == "wrist":
                n_wrist += 1
            O, A = wra_xyz, wrb_xyz
            B = pred
            angs_wra.append(angle_at_vertex(O, B, A))
            if np.isfinite(frm_xyz).all():
                angs_frm.append(angle_at_vertex(frm_xyz, B, O))
        arr_wra = np.array(angs_wra, dtype=np.float64)
        arr_frm = np.array(angs_frm, dtype=np.float64)
        out["methods"][name] = {
            "n_predicted": n_frames,
            "n_forearm_anchor": n_forearm,
            "n_wrist_fallback": n_wrist,
            "angle_WRA_deg_mean": float(np.nanmean(arr_wra)),
            "angle_WRA_deg_std": float(np.nanstd(arr_wra)),
            "angle_WRA_minus_static_mean": float(np.nanmean(arr_wra) - ref_angles.get("static_WRA", np.nan)),
            "angle_FRM_WRA_FIN_deg_mean": float(np.nanmean(arr_frm)) if arr_frm.size else None,
            "angle_FRM_WRA_FIN_deg_std": float(np.nanstd(arr_frm)) if arr_frm.size else None,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add or fill LFIN/RFIN using forearm-local or legacy wrist static geometry."
    )
    parser.add_argument("static_trial", help="Labeled static .csv or .c3d")
    parser.add_argument("dynamic_csv", help="Labeled dynamic CSV (FIN column optional)")
    parser.add_argument("-o", "--output", help="Output CSV path")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Print wrist vs forearm angle comparison (no output file required)",
    )
    parser.add_argument(
        "--method",
        choices=("forearm", "wrist", "auto"),
        default="forearm",
        help="Primary synthesis method (default: forearm)",
    )
    parser.add_argument("--side", choices=("left", "right", "both"), default="both")
    parser.add_argument("--only-if-fin-missing", action="store_true")
    parser.add_argument("--no-fill-nan", action="store_true")
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Re-synthesize all existing FIN frames",
    )
    parser.add_argument("--vertical", type=str, default="0,1,0", metavar="X,Y,Z")
    args = parser.parse_args()
    parts = [float(p.strip()) for p in str(args.vertical).split(",")]
    if len(parts) != 3:
        print("Error: --vertical must be three comma-separated numbers.", file=sys.stderr)
        raise SystemExit(2)
    side_map = {"left": ("L",), "right": ("R",), "both": ("L", "R")}

    if args.compare:
        for side in side_map[args.side]:
            rep = compare_fin_methods_on_trial(args.static_trial, args.dynamic_csv, side=side)
            print(f"\n=== {side} hand comparison: {Path(args.dynamic_csv).name} ===")
            print("Static reference angles (deg):", rep["static_ref_angles_deg"])
            for m, stats in rep["methods"].items():
                print(f"  [{m}]", stats)
        return

    if not args.output:
        print("Error: -o/--output required unless --compare", file=sys.stderr)
        raise SystemExit(2)

    try:
        info = synthesize_fin_csv(
            args.static_trial,
            args.dynamic_csv,
            args.output,
            sides=side_map[args.side],
            lab_vertical=tuple(parts),
            method=args.method,
            only_if_fin_missing=bool(args.only_if_fin_missing),
            fill_nan_in_existing=not args.no_fill_nan,
            replace_existing=bool(args.replace_existing),
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(2) from e
    print(
        f"Wrote {info['output_csv']} inserted={info['inserted']} "
        f"methods_used={info.get('methods_used', {})} models={info.get('models', {})}"
    )


if __name__ == "__main__":
    main()
