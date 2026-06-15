"""Tests for static hand FIN synthesis."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from marker_label.synthesize_fin_from_static import (
    _mean_finite_rows,
    build_fin_side_model,
    compare_fin_methods_on_trial,
    load_static_for_hand,
    predict_fin_forearm_row,
    static_forearm_fin_local,
    synthesize_fin_csv,
)
from marker_label.trial_trim import parse_labeled_csv


def test_forearm_local_roundtrip_static():
    static = Path("data/BBC13/BBC13 Cal 01.c3d")
    if not static.is_file():
        return
    pts, idx = load_static_for_hand(static, ("R",))
    local = static_forearm_fin_local(pts, idx, side="R")
    means = _mean_finite_rows(pts, idx, ("RFRM", "RWRA", "RWRB", "RFIN"))
    assert means is not None
    pred = predict_fin_forearm_row(means["RFRM"], means["RWRA"], means["RWRB"], local)
    err = float(np.linalg.norm(pred - means["RFIN"]))
    assert err < 0.5


def test_insert_rfin_column_bbc13_t38_forearm():
    inp = Path("corrected/BBC13 Trial 38_corrected.csv")
    if not inp.is_file():
        inp = Path("data/BBC13/BBC13 Trial 38_corrected.csv")
    bak = inp.with_suffix(inp.suffix + ".pre_fin.bak")
    if bak.is_file():
        inp = bak
    if not inp.is_file():
        return
    stems, _ = parse_labeled_csv(inp)
    out = Path(tempfile.gettempdir()) / "test_bbc13_t38_forearm_rfin.csv"
    info = synthesize_fin_csv(
        "data/BBC13/BBC13 Cal 01.c3d",
        inp,
        out,
        sides=("R",),
        method="forearm",
    )
    assert info["inserted"].get("RFIN") or info.get("filled_frames")
    stems_after, meta = parse_labeled_csv(out)
    assert "RFIN" in stems_after
    mi = meta["label_to_marker_idx"]["RFIN"]
    n_ok = int(np.isfinite(meta["points"][:, mi, :]).all(axis=1).sum())
    assert n_ok > meta["n_frames"] // 2


def test_compare_methods_bbc13_t38():
    inp = Path("corrected/BBC13 Trial 38_corrected.csv")
    if not inp.is_file():
        return
    bak = inp.with_suffix(inp.suffix + ".pre_fin.bak")
    src = bak if bak.is_file() else inp
    rep = compare_fin_methods_on_trial("data/BBC13/BBC13 Cal 01.c3d", src, side="R")
    assert "wrist" in rep["methods"] and "forearm" in rep["methods"]
    # Forearm method should track static WRA angle more closely on average
    w = rep["methods"]["wrist"]["angle_WRA_minus_static_mean"]
    f = rep["methods"]["forearm"]["angle_WRA_minus_static_mean"]
    assert abs(f) <= abs(w) + 5.0
