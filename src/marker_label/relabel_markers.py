#!/usr/bin/env python3
"""
relabel_markers.py
==================
Detect, per frame, which trajectory (labeled marker or unlabeled donor) holds
each marker slot of a rigid cluster -- i.e. recover the time-varying ID->marker
correspondence. ID-AGNOSTIC: it never trusts a trajectory's number (those are
reused within a trial and differ across trials); it solves the correspondence
from segment geometry every frame.

Pipeline per cluster (pelvis / head / thorax):
  1. Template: mean rigid shape from clean frames (in-band, sane inter-marker
     distances). Also a chirality reference (signed volume sign) to break the
     left<->right mirror that pure distance-matching cannot.
  2. Per-frame pose solve, trying in order:
       a. COAST  - reuse previous frame's pose (fast path).
       b. TRUSTED- >=3 mutually-consistent in-band labeled members -> Kabsch.
       c. SEARCH - geometric-hashing over candidate points (labeled + donors):
                   find a 3-point correspondence whose triangle matches the
                   template, fit, score by inliers within tol.
     Each hypothesis is scored by inlier count then RMS; rejected unless its
     chirality sign matches the template and (if a previous pose exists) the
     pose step is continuous.
  3. Assignment: with the chosen pose, predict every slot and assign the
     available points by Hungarian (one point per slot) within `assign_gate`.
     Source = 'self' (own label), a donor ID, or 'none'.
  Run forward + backward; keep the higher-confidence solution per frame.

Validity band: y in [min,max] of the OBSTACLE_* markers' y. Points outside it
are never candidates.

Outputs:
  <stem>_labelmap.csv    frame, cluster, slot, source, residual_mm
  <stem>_relabeled.csv   named-marker columns filled from the assigned points
"""
from __future__ import annotations

import argparse
import itertools
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

RIGID_CLUSTERS = {
    "pelvis": ["LASI", "RASI", "LPSI", "RPSI"],
    "head":   ["LFHD", "RFHD", "LBHD", "RBHD"],
    "thorax": ["C7", "CLAV", "T10", "STRN"],
    "larm":   ["LFRM", "LWRA", "LWRB", "LFIN"],
    "rarm":   ["RFRM", "RWRA", "RWRB", "RFIN"],
}

def kabsch(A, B):
    cA, cB = A.mean(0), B.mean(0)
    H = (A - cA).T @ (B - cB)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, cB - R @ cA

def rms(A, B, R, t): return float(np.sqrt(np.mean(np.linalg.norm(A @ R.T + t - B, axis=1) ** 2)))

def signed_vol(p):   # p: dict slot->xyz for 4 ordered slots
    o = p[0]; return float(np.linalg.det(np.array([p[1]-o, p[2]-o, p[3]-o])))

def rigid_fill(clusters, data, cols, relabeled, frame, lo, hi, rms_tol=25.0):
    """Reconstruct a missing cluster member from the rigid pose when >=3 of its
    cluster-mates are present and fit the template well. The result is a SYNTHESIZED
    point (not observed); it is logged with source 'rigidfill' so it stays distinguishable."""
    N = data.shape[0]; out = []
    for cl in clusters:
        RX = {m: relabeled[:, [cols[m]["x"], cols[m]["y"], cols[m]["z"]]] for m in cl.members}
        for f in range(N):
            pres = [m for m in cl.members if not np.isnan(RX[m][f]).any()]
            miss = [m for m in cl.members if m not in pres]
            if len(pres) < 3 or not miss:
                continue
            R, t = kabsch(np.array([cl.tmpl[m] for m in pres]), np.array([RX[m][f] for m in pres]))
            res = float(np.sqrt(np.mean([np.linalg.norm(R @ cl.tmpl[m] + t - RX[m][f]) ** 2 for m in pres])))
            if res > rms_tol:
                continue
            for m in miss:
                pred = R @ cl.tmpl[m] + t
                if lo <= pred[1] <= hi:
                    relabeled[f, [cols[m]["x"], cols[m]["y"], cols[m]["z"]]] = pred
                    out.append((frame[f], cl.name, m, "rigidfill", round(res, 1)))
    return out


def spike_clean(data, cols, named, cluster_members, lo, hi, donors, relabeled, frame,
                spike_tol=150.0, smooth_tol=60.0, donor_tol=30.0, max_bridge=20):
    """Conservative teleport cleaning for markers NOT on a rigid cluster.
    A sample is a teleport only if it is (a) present but out of the obstacle y-band,
    or (b) a there-and-back excursion: the marker's own two-sided constant-velocity
    predictions agree (smooth underlying motion) yet the sample lies far from them.
    Real fast motion is NOT flagged, because then the sample sits between the two
    predictions. Bad samples are de-labeled; where a short continuity bridge exists
    and an unlabeled donor matches the bridged position, the sample is re-labeled."""
    N = data.shape[0]
    targets = [m for m in named if m not in cluster_members]
    rows = []
    for m in targets:
        c = [cols[m]["x"], cols[m]["y"], cols[m]["z"]]
        P = data[:, c]; pr = ~np.isnan(P).any(1); y = P[:, 1]
        ib = pr & (y >= lo) & (y <= hi)
        bad = pr & ~ib                                  # (a) out-of-band present samples
        idx = np.where(ib)[0]                           # (b) there-and-back among in-band
        for k in range(2, len(idx) - 2):
            i, p1, p2, n1, n2 = idx[k], idx[k-1], idx[k-2], idx[k+1], idx[k+2]
            fwd = P[p1] + (P[p1] - P[p2]) / max(p1 - p2, 1) * (i - p1)
            bwd = P[n1] + (P[n1] - P[n2]) / max(n1 - n2, 1) * (i - n1)
            if np.linalg.norm(fwd - bwd) < smooth_tol and \
               np.linalg.norm(P[i] - 0.5 * (fwd + bwd)) > spike_tol:
                bad[i] = True
        if not bad.any():
            continue
        bi = np.where(bad)[0]
        for run in np.split(bi, np.where(np.diff(bi) != 1)[0] + 1):
            a, b = run[0], run[-1]
            before = np.where(ib[:a] & ~bad[:a])[0]
            after = np.where(ib[b+1:] & ~bad[b+1:])[0]
            before = before[-1] if len(before) else None
            after = (after[0] + b + 1) if len(after) else None
            for i in run:
                relabeled[i, c] = np.nan                 # de-label
            if before is not None and after is not None and (after - before) <= max_bridge:
                for i in run:
                    w = (i - before) / (after - before)
                    pred = (1 - w) * P[before] + w * P[after]
                    best = None
                    for dn in donors:
                        q = data[i, [cols[dn]["x"], cols[dn]["y"], cols[dn]["z"]]]
                        if np.isnan(q).any(): continue
                        dd = float(np.linalg.norm(q - pred))
                        if best is None or dd < best[1]: best = (dn, dd, q)
                    if best and best[1] <= donor_tol:
                        relabeled[i, c] = best[2]
                        rows.append((frame[i], "spike", m, best[0], round(best[1], 1)))
                    else:
                        rows.append((frame[i], "spike", m, "gap", ""))
            else:
                for i in run:
                    rows.append((frame[i], "spike", m, "gap", ""))
    return rows


def load_static(path, named):
    """Build clean per-marker reference shape from a labeled static C3D (or CSV).
    Whole-body Procrustes-average over frames removes sway; returns {marker: xyz}."""
    if path.lower().endswith(".c3d"):
        import c3d
        with open(path, "rb") as f:
            r = c3d.Reader(f)
            labels = [l.strip() for l in r.point_labels]
            A = np.array([pts[:, :3].copy() for _, pts, _ in r.read_frames()])
        idx = {l: j for j, l in enumerate(labels)}
        present = [m for m in named if m in idx]
        X = np.stack([A[:, idx[m], :] for m in present], axis=1)        # (F,M,3)
    else:
        df, raw, cols = load(path); d = df.to_numpy(float)
        present = [m for m in named if m in cols]
        X = np.stack([d[:, [cols[m]["x"], cols[m]["y"], cols[m]["z"]]] for m in present], axis=1)
    # drop frames with any missing/zero marker, then Procrustes-average to frame 0
    okf = ~(np.isnan(X).any((1, 2)) | (X == 0).all(2).any(1))
    X = X[okf]
    ref = X[0] - X[0].mean(0); acc = np.zeros_like(ref)
    for f in range(len(X)):
        Y = X[f] - X[f].mean(0)
        U, _, Vt = np.linalg.svd(Y.T @ ref); dt = np.sign(np.linalg.det(Vt.T @ U.T))
        acc += Y @ (Vt.T @ np.diag([1, 1, dt]) @ U.T).T
    mean = acc / len(X)
    return {m: mean[i] for i, m in enumerate(present)}

def load(path):
    df = pd.read_csv(path); raw = list(df.columns)
    clean = [re.sub(r"\s+", "", c) for c in raw]; cols = {}
    for i, c in enumerate(clean):
        m = re.match(r"^(.*)_(x|y|z)$", c)
        if m: cols.setdefault(m.group(1), {})[m.group(2)] = i
    return df, raw, cols

class Cluster:
    def __init__(s, name, members, data, cols, lo, hi):
        s.name, s.lo, s.hi = name, lo, hi
        s.members = [m for m in members if m in cols]
        s.M = len(s.members)
        s.X = {m: data[:, [cols[m]["x"], cols[m]["y"], cols[m]["z"]]] for m in s.members}
        s.tmpl = None; s.chi = None; s.tdist = None
    def ib(s, m, f):
        p = s.X[m][f]; return (not np.isnan(p).any()) and (s.lo <= p[1] <= s.hi)
    def build(s, N, static=None):
        # dynamic clean frames (used for orientation regardless of template source)
        allib = np.array([all(s.ib(m, f) for m in s.members) for f in range(N)])
        g = []
        for f in np.where(allib)[0]:
            ds = [np.linalg.norm(s.X[a][f]-s.X[b][f]) for a,b in itertools.combinations(s.members,2)]
            if ds and min(ds) > 40 and max(ds) < 450: g.append(f)
        g = np.array(g, int)
        if static is not None and all(m in static for m in s.members):
            s.tmpl = {m: static[m] for m in s.members}          # SHAPE from static
        elif len(g) >= 5:
            # Procrustes-average the clean frames (remove body motion) -> motion-invariant SHAPE.
            # A raw positional mean over a moving trial shrinks inter-marker distances and
            # corrupts the template; aligning first is required.
            G = np.stack([np.stack([s.X[m][f] for m in s.members]) for f in g])   # (n,M,3)
            ref = G[0] - G[0].mean(0); acc = np.zeros_like(ref)
            for k in range(len(G)):
                Y = G[k] - G[k].mean(0)
                U, _, Vt = np.linalg.svd(Y.T @ ref); dt = np.sign(np.linalg.det(Vt.T @ U.T))
                acc += Y @ (Vt.T @ np.diag([1, 1, dt]) @ U.T).T
            mean = acc / len(G)
            s.tmpl = {m: mean[i] for i, m in enumerate(s.members)}   # SHAPE bootstrapped
        else:
            return False
        s.tdist = {(a, b): float(np.linalg.norm(s.tmpl[a]-s.tmpl[b]))
                   for a in s.members for b in s.members}
        # ORIENTATION from WHOLE-CLUSTER-clean frames only (every pairwise distance matches
        # the template). A consistent label swap corrupts the whole cluster, so it yields no
        # clean frames here and hence no (wrong) orientation constraint -- which is what lets
        # the solver later correct the swap instead of being locked into it by its own data.
        gc = []
        for f in np.where(allib)[0]:
            if all(abs(np.linalg.norm(s.X[a][f]-s.X[b][f]) - s.tdist[(a, b)]) < 20
                   for a, b in itertools.combinations(s.members, 2)):
                gc.append(f)
        gc = np.array(gc, int)
        # ORIENTATION: keep a medio-lateral ordering constraint only for pairs that are
        # ROBUSTLY separated in y -- large mean separation AND tight spread across the clean
        # frames. Same-side anterior/posterior pairs (e.g. RASI-RPSI) have a looser separation
        # whose sign flips as the pelvis rotates transversely through an obstacle stride; as a
        # hard constraint they wrongly veto a correct pose during rotation (and take any donor
        # recovery in that frame down with them). The strong left-vs-right pairs (LASI-RASI,
        # LPSI-RPSI, head L/R) are tight, do not flip, and alone suffice to break mirror ambiguity.
        s.yorder = {}
        if len(gc) >= 5:
            for a, b in itertools.combinations(s.members, 2):
                dys = s.X[a][gc, 1] - s.X[b][gc, 1]
                mean_dy = float(dys.mean()); spread = float(dys.max() - dys.min())
                if abs(mean_dy) > 30 and dys.min() * dys.max() > 0 and spread < 0.20 * abs(mean_dy):
                    s.yorder[(a, b)] = np.sign(mean_dy)
        return True

def candidates(s, donor_X, donors, f):
    """available points this frame: (label, id, xyz). label==id for labeled members."""
    out = []
    for m in s.members:
        if s.ib(m, f): out.append((m, m, s.X[m][f]))
    for d in donors:
        q = donor_X[d][f]
        if not np.isnan(q).any() and s.lo <= q[1] <= s.hi: out.append(("*", d, q))
    return out

def score_pose(s, R, t, cand, inlier_tol):
    pred = {m: R @ s.tmpl[m] + t for m in s.members}
    C = np.array([c[2] for c in cand])
    cost = np.full((s.M, len(cand)), 1e9)
    for i, m in enumerate(s.members):
        cost[i] = np.linalg.norm(C - pred[m], axis=1)
    ri, cj = linear_sum_assignment(cost)
    asg = {}; res = []
    for i, j in zip(ri, cj):
        if cost[i, j] <= inlier_tol:
            asg[s.members[i]] = (cand[j][0], cand[j][1], cand[j][2]); res.append(cost[i, j])
    return asg, (np.sqrt(np.mean(np.square(res))) if res else 1e9), len(asg)

def orient_ok(s, asg):
    """assignment must respect the dynamic trial's medio-lateral (y) ordering."""
    for (a, b), sign in s.yorder.items():
        if a in asg and b in asg:
            if np.sign(asg[a][2][1] - asg[b][2][1]) != sign:
                return False
    return True

def consistent_subset(s, lab, dtol):
    """largest subset of labeled points whose pairwise distances match the template."""
    items = list(lab.items())                       # [(marker, pos)]
    best = []
    for r in range(len(items), 1, -1):
        for sub in itertools.combinations(items, r):
            ok = all(abs(np.linalg.norm(sub[i][1]-sub[j][1]) - s.tdist[(sub[i][0], sub[j][0])]) < dtol
                     for i, j in itertools.combinations(range(r), 2))
            if ok: return [m for m, _ in sub]
    return [m for m, _ in items][:1]

def gross_outliers(s, f, dtol):
    """Present labeled cluster markers that grossly violate template distances against a
    mutually-consistent core of >=2 OTHER present labels. Used ONLY when the solver produced
    no pose for the frame (too few cluster points to fix a rigid pose), to de-label obvious
    contaminants -- e.g. a label parked on the wrong limb -- that pose-based rejection never
    got to evaluate. Conservative: an outlier must be off-template by >3*dtol to EVERY core
    member, so a noisy-but-valid point (which stays near at least its true neighbours) is kept."""
    lab = {m: s.X[m][f] for m in s.members if s.ib(m, f)}
    if len(lab) < 3:
        return set()
    core = consistent_subset(s, lab, dtol)
    if len(core) < 2:
        return set()
    gross = 3.0 * dtol
    out = set()
    for m in lab:
        if m in core:
            continue
        if all(abs(np.linalg.norm(lab[m] - lab[c]) - s.tdist[(m, c)]) > gross for c in core):
            out.add(m)
    return out

def solve_frame(s, cand, prev_pose, inlier_tol, dtol):
    lab = {m: p for (l, m, p) in cand if l == m}
    donor_pts = [(cid, p) for (l, cid, p) in cand if l == "*"]
    # ---- trusted path: >=3 mutually-consistent labels define the pose; lock them as self ----
    trusted = consistent_subset(s, lab, dtol) if len(lab) >= 3 else []
    if len(trusted) >= 3:
        R, t = kabsch(np.array([s.tmpl[m] for m in trusted]),
                      np.array([lab[m] for m in trusted]))
        asg = {m: ("self", m, lab[m]) for m in trusted}
        open_slots = [m for m in s.members if m not in trusted]
        pool = [(cid, p) for (cid, p) in donor_pts]                    # donors
        pool += [(m, lab[m]) for m in lab if m not in trusted]          # mislabeled labels too
        if open_slots and pool:
            pred = {m: R @ s.tmpl[m] + t for m in open_slots}
            cost = np.full((len(open_slots), len(pool)), 1e9)
            for i, m in enumerate(open_slots):
                for j, (cid, p) in enumerate(pool):
                    cost[i, j] = np.linalg.norm(p - pred[m])
            ri, cj = linear_sum_assignment(cost)
            for i, j in zip(ri, cj):
                if cost[i, j] <= inlier_tol:
                    m = open_slots[i]; cid, p = pool[j]
                    asg[m] = ("*" if cid not in s.members else cid, cid, p)
        # GUARD: a "consistent" trusted subset can actually be a coherent label swap (its
        # mutual distances match by coincidence). The tell of a swap is BOTH a present labeled
        # point that fits nowhere AND a cluster slot left unfilled -- the point belongs in that
        # empty slot but the labels are scrambled. If instead every slot is filled (a teleported
        # label correctly dropped and its slot taken by a donor), the solution is complete and
        # is kept. Only the genuine-swap case falls through to the label-agnostic search.
        explained = {asg[m][1] for m in asg}
        dropped = [m for m in lab if m not in explained]
        unfilled = [m for m in s.members if m not in asg]
        if not (dropped and unfilled):
            r = rms(np.array([s.tmpl[m] for m in trusted]),
                    np.array([lab[m] for m in trusted]), R, t)
            return (R, t), asg, r
    # ---- geometric search: chirality + continuity; reached when <3 trusted OR a present
    #      label was dropped above (swap signature) ----
    hyps = []
    if prev_pose is not None: hyps.append(prev_pose)
    pts = [p for _, _, p in cand]; n = len(pts)
    if n:
        D = np.linalg.norm(np.array(pts)[:, None, :] - np.array(pts)[None, :, :], axis=2)
        found = 0
        for s1, s2 in itertools.permutations(s.members, 2):
            d12 = s.tdist[(s1, s2)]
            for i in range(n):
                for j in range(n):
                    if i == j or abs(D[i, j] - d12) > dtol: continue
                    for s3 in s.members:
                        if s3 in (s1, s2): continue
                        d13, d23 = s.tdist[(s1, s3)], s.tdist[(s2, s3)]
                        for k in range(n):
                            if k in (i, j): continue
                            if abs(D[i, k]-d13) < dtol and abs(D[j, k]-d23) < dtol:
                                hyps.append(kabsch(np.array([s.tmpl[s1], s.tmpl[s2], s.tmpl[s3]]),
                                                   np.array([pts[i], pts[j], pts[k]]))); found += 1
                    if found > 60: break
                if found > 60: break
            if found > 60: break
    best = None
    for (R, t) in hyps:
        asg, r, ninl = score_pose(s, R, t, cand, inlier_tol)
        if ninl < 3 or r > 25 or not orient_ok(s, asg): continue
        cont = 0.0
        if prev_pose is not None:
            cont = np.linalg.norm((R @ s.tmpl[s.members[0]] + t) -
                                  (prev_pose[0] @ s.tmpl[s.members[0]] + prev_pose[1]))
        nself = sum(1 for m in asg if asg[m][1] == m)
        key = (-ninl, -nself, r + 0.15 * cont)
        if best is None or key < best[0]:
            R2, t2 = kabsch(np.array([s.tmpl[m] for m in asg]), np.array([asg[m][2] for m in asg]))
            best = (key, (R2, t2), asg, r)
    return (best[1], best[2], best[3]) if best else (None, {}, 1e9)

def pass_dir(s, donor_X, donors, N, order, inlier_tol, dtol):
    rng = range(N) if order == "fwd" else range(N-1, -1, -1)
    prev = None; result = {}
    for f in rng:
        cand = candidates(s, donor_X, donors, f)
        if not cand: result[f] = ({}, 1e9, None); prev = None; continue
        pose, asg, r = solve_frame(s, cand, prev, inlier_tol, dtol)
        result[f] = (asg, r, pose)
        prev = pose if pose is not None else prev
    return result

def subject_id_from_csv_path(csv_path: str | Path) -> str | None:
    """Extract subject id (e.g. SUBJ01) from ``SUBJ01 Trial 55_trimmed.csv``."""
    m = re.match(r"^([A-Z]+\d+)", Path(csv_path).name)
    return m.group(1) if m else None


def _is_flat_labeled_csv(path: Path) -> bool:
    """True if header starts with frame, time and has marker_x columns."""
    with open(path, newline="") as f:
        header = f.readline().strip().split(",")
    if len(header) < 4:
        return False
    h0, h1 = header[0].strip().lower(), header[1].strip().lower()
    if h0 != "frame" or h1 != "time":
        return False
    return any(c.strip().endswith("_x") for c in header[2:])


def resolve_static_trial(subject_id: str, data_dir: str | Path) -> Path | None:
    """
    Return labeled static trial for ``subject_id`` under ``data_dir/{subject_id}/``.

    Prefers ``Cal 01``, then any ``{subject_id} Cal *.c3d``, then flat ``Cal *.csv``.
    Skips non-flat Vicon export CSVs.
    """
    root = Path(data_dir) / subject_id
    if not root.is_dir():
        return None

    def _ok_csv(p: Path) -> bool:
        return p.is_file() and _is_flat_labeled_csv(p)

    for name in (
        f"{subject_id} Cal 01_labeled.csv",
        f"{subject_id} Cal 01.csv",
        f"{subject_id} Cal 01.c3d",
    ):
        p = root / name
        if not p.is_file():
            continue
        if p.suffix.lower() == ".csv" and not _ok_csv(p):
            continue
        return p

    c3ds = sorted(root.glob(f"{subject_id} Cal *.c3d"), key=lambda p: p.name)
    if c3ds:
        return c3ds[0]

    for p in sorted(root.glob(f"{subject_id} Cal *.csv"), key=lambda x: x.name):
        if _ok_csv(p):
            return p
    return None


def discover_static_map(data_dir: str | Path, subject_ids: list[str]) -> dict[str, Path | None]:
    """Map each subject id to an auto-discovered static path (or None)."""
    out: dict[str, Path | None] = {}
    for sid in subject_ids:
        out[sid] = resolve_static_trial(sid, data_dir)
    return out


def default_output_path(input_csv: str | Path, *, trimmed_to_corrected: bool = True) -> Path:
    """Map ``*_trimmed.csv`` -> ``*_corrected.csv``; else append ``_relabeled`` before ``.csv``."""
    p = Path(input_csv)
    stem = p.stem
    if trimmed_to_corrected and stem.endswith("_trimmed"):
        return p.with_name(stem[: -len("_trimmed")] + "_corrected.csv")
    return p.with_name(stem + "_relabeled.csv")


def relabel_csv(
    csv_path: str | Path,
    *,
    output_csv: str | Path | None = None,
    static_path: str | Path | None = None,
    inlier_tol: float = 35.0,
    dtol: float = 20.0,
    rigidfill: bool = False,
    write_output: bool = True,
) -> dict[str, Path | None]:
    """
    Run rigid-cluster relabeling on one labeled CSV.

    Returns paths for labelmap, optional crosslabel, and output CSV.
    """
    csv_path = Path(csv_path)
    df, raw, cols = load(csv_path)
    data = df.to_numpy(float)
    frame = data[:, 0].astype(int); N = len(df)
    ys = []
    for ob in [m for m in cols if m.startswith("OBSTACLE")]:
        y = data[:, cols[ob]["y"]]; ys += [np.nanmin(y), np.nanmax(y)]
    lo, hi = (min(ys), max(ys)) if ys else (-1e9, 1e9)
    print(f"obstacle y-band = [{lo:.0f}, {hi:.0f}]")
    # walking direction (per trial, +x or -x): net x-travel of the body centroid
    body = [m for cl in RIGID_CLUSTERS.values() for m in cl if m in cols]
    cx = np.nanmedian(np.stack([data[:, cols[m]["x"]] for m in body]), axis=0)
    valid = ~np.isnan(cx)
    if valid.sum() > 2:
        xs = cx[valid]; travel = xs[-1] - xs[0]
        print(f"walking direction: {'+x' if travel > 0 else '-x'} (net x-travel {travel:.0f} mm)")
    donors = [m for m in cols if m.startswith("*") or re.match(r"^\d+$", m)]
    donor_X = {d: data[:, [cols[d]["x"], cols[d]["y"], cols[d]["z"]]] for d in donors}

    rows = []; relabeled = data.copy(); clusters = []
    all_named = [m for cl in RIGID_CLUSTERS.values() for m in cl]
    static = None
    if static_path:
        static = load_static(static_path, all_named)
        print(f"using static reference: {static_path} ({len(static)} markers)")
    for name, members in RIGID_CLUSTERS.items():
        cl = Cluster(name, members, data, cols, lo, hi)
        if cl.M < 3 or not cl.build(N, static):
            print(f"[{name}] skipped (insufficient clean frames)"); continue
        clusters.append(cl)
        fwd = pass_dir(cl, donor_X, donors, N, "fwd", inlier_tol, dtol)
        bwd = pass_dir(cl, donor_X, donors, N, "bwd", inlier_tol, dtol)
        # Blank ONLY out-of-band (teleported) originals; keep in-band originals so the
        # solver can never DELETE a valid point. Confident assignments overwrite below.
        for m in cl.members:
            cidx = [cols[m]["x"], cols[m]["y"], cols[m]["z"]]
            P = data[:, cidx]; yy = P[:, 1]
            oob = (~np.isnan(P).any(1)) & ((yy < lo) | (yy > hi))
            for f in np.where(oob)[0]:
                relabeled[f, cidx] = np.nan
        nslots = 0
        for f in range(N):
            af, rf, pf = fwd[f]; ab, rb, pb = bwd[f]
            nsf = sum(1 for m in af if af[m][1] == m)
            nsb = sum(1 for m in ab if ab[m][1] == m)
            asg, rr, pose = (af, rf, pf) if (nsf, -rf) >= (nsb, -rb) else (ab, rb, pb)
            reassigned = {cid for sl, (lab, cid, q) in asg.items() if cid in cl.members and cid != sl}
            gross = gross_outliers(cl, f, dtol) if pose is None else set()
            for m in cl.members:
                cidx = [cols[m]["x"], cols[m]["y"], cols[m]["z"]]
                if m in asg:
                    lab, cid, q = asg[m]
                    src = "self" if cid == m else cid
                    rows.append((frame[f], name, m, src, round(rr, 1)))
                    relabeled[f, cidx] = q
                    nslots += 1
                elif m in reassigned:
                    relabeled[f, cidx] = np.nan          # its point now lives in another slot
                elif pose is not None:
                    relabeled[f, cidx] = np.nan          # solver had a pose and rejected this point
                elif m in gross:
                    relabeled[f, cidx] = np.nan          # no pose, but a consistent core proves
                    rows.append((frame[f], name, m, "outlier", ""))   # this point is a contaminant
                # else: solver couldn't solve this frame -> keep the in-band original (never delete)
        print(f"[{name}] resolved {nslots} slot-frames")

    lm = pd.DataFrame(rows, columns=["frame", "cluster", "slot", "source", "rms_mm"])

    if rigidfill:
        rf = rigid_fill(clusters, data, cols, relabeled, frame, lo, hi)
        if rf:
            sm = {}
            for (_, cn, mk, _, _) in rf: sm[mk] = sm.get(mk, 0) + 1
            print("\nRigid-body fill (synthesized from >=3 present cluster-mates):")
            for mk, n in sorted(sm.items(), key=lambda kv: -kv[1]):
                print(f"   {mk:5} {n} frames reconstructed")
            lm = pd.concat([lm, pd.DataFrame(rf, columns=lm.columns)], ignore_index=True)
    # ---- spikes on non-rigid markers (de-label; re-label from donor on confident continuity) ----
    cluster_members = {m for c in clusters for m in c.members}
    named_all = [m for m in cols if re.match(r"^[A-Za-z]", m) and not m.startswith("OBSTACLE")
                 and not m.startswith("*")]
    spike_rows = spike_clean(data, cols, named_all, cluster_members, lo, hi, donors,
                             relabeled, frame)
    if spike_rows:
        ns = len(spike_rows); nr = sum(1 for r in spike_rows if r[3] != "gap")
        print(f"\nNon-rigid spike cleaning: {ns} samples de-labeled, {nr} re-labeled from donors")
        sm = {}
        for (fr, _, mk, src, dd) in spike_rows:
            sm.setdefault(mk, [0, 0]); sm[mk][0] += 1; sm[mk][1] += (src != "gap")
        for mk, (tot, rl) in sorted(sm.items(), key=lambda kv: -kv[1][0]):
            print(f"   {mk:5} {tot:3d} samples ({rl} re-labeled from donor)")
        lm = pd.concat([lm, pd.DataFrame(spike_rows, columns=lm.columns)], ignore_index=True)

    lm_path = csv_path.with_name(csv_path.stem + "_labelmap.csv")
    lm.to_csv(lm_path, index=False)
    print(f"wrote {lm_path}")
    cross_path: Path | None = None

    # ---- cross-cluster identity check: a label sitting at another segment's slot ----
    # Fit each cluster's pose from its relabeled (clean) members; test every named marker's
    # ORIGINAL position against foreign slot predictions.
    all_markers = [m for m in cols if re.match(r"^[A-Za-z]", m) and not m.startswith("OBSTACLE")
                   and not m.startswith("*")]
    cross = []
    for cl in clusters:
        RX = {m: relabeled[:, [cols[m]["x"], cols[m]["y"], cols[m]["z"]]] for m in cl.members}
        for f in range(N):
            pres = [m for m in cl.members if not np.isnan(RX[m][f]).any()]
            if len(pres) < 3: continue
            R, t = kabsch(np.array([cl.tmpl[m] for m in pres]), np.array([RX[m][f] for m in pres]))
            pred = {m: R @ cl.tmpl[m] + t for m in cl.members}
            for lab in all_markers:
                if lab in cl.members: continue
                p = data[f, [cols[lab]["x"], cols[lab]["y"], cols[lab]["z"]]]
                if np.isnan(p).any() or not (lo <= p[1] <= hi): continue
                for slot, pp in pred.items():
                    if np.linalg.norm(p - pp) <= 20:           # labeled point IS this foreign slot
                        cross.append((frame[f], lab, cl.name, slot, round(float(np.linalg.norm(p-pp)),1)))
    cluster_members = {m for c in clusters for m in c.members}
    cross_cells = set()
    if cross:
        cdf = pd.DataFrame(cross, columns=["frame", "label", "matches_cluster", "as_slot", "dist_mm"])
        print("\nCross-cluster mislabels (a marker labeled X but sitting at another segment's slot):")
        for (lab, slot), g in cdf.groupby(["label", "as_slot"]):
            fr = sorted(g.frame); segs = []
            for x in fr:
                if segs and x == segs[-1][1] + 1: segs[-1][1] = x
                else: segs.append([x, x])
            sp = ", ".join(f"{a}-{b}" if a != b else f"{a}" for a, b in segs)
            print(f"  '{lab}' is actually {slot}  @ {sp}  (~{g.dist_mm.median():.0f}mm)")
        cross_path = csv_path.with_name(csv_path.stem + "_crosslabel.csv")
        cdf.to_csv(cross_path, index=False)
        for (fr, lab, cn, slot, dd) in cross:
            if lab not in cluster_members:          # cluster members are fixed by their own cluster
                cross_cells.add((lab, fr))

    # print detected ID-change events per slot (source transitions)
    print("\nDetected ID changes (source transitions per slot):")
    for (cl, slot), g in lm.groupby(["cluster", "slot"]):
        g = g.sort_values("frame"); prev = None; segs = []
        for _, r in g.iterrows():
            if r.source != prev: segs.append([r.source, r.frame, r.frame]); prev = r.source
            else: segs[-1][2] = r.frame
        nontrivial = [s for s in segs if s[0] != "self"]
        if nontrivial:
            txt = ", ".join(f"{s[0]}@{s[1]}-{s[2]}" for s in segs if s[0] != "self")
            print(f"  {slot:5}: {txt}")

    out_path: Path | None = None
    if write_output:
        for (lab, fr) in cross_cells:
            i = np.where(frame == fr)[0]
            if len(i):
                relabeled[i[0], [cols[lab]["x"], cols[lab]["y"], cols[lab]["z"]]] = np.nan
        out = pd.DataFrame(relabeled, columns=raw)
        out[raw[0]] = out[raw[0]].astype(int)
        if output_csv is None:
            output_csv = default_output_path(csv_path)
        out_path = Path(output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False, na_rep="")
        print(f"\nwrote {out_path}")

    return {"labelmap": lm_path, "crosslabel": cross_path, "output": out_path}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="Labeled trial CSV (e.g. corrected/*_trimmed.csv)")
    ap.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output corrected CSV (default: *_trimmed.csv -> *_corrected.csv in same folder)",
    )
    ap.add_argument(
        "--apply",
        action="store_true",
        help="Write relabeled CSV (same as default when -o is omitted; kept for script compatibility)",
    )
    ap.add_argument("--static", default=None, help="Labeled static trial (.c3d/.csv) for reference templates")
    ap.add_argument(
        "--static-auto",
        action="store_true",
        help="Use data/{subject_id}/{subject_id} Cal 01.csv or .c3d when --static is omitted",
    )
    ap.add_argument(
        "--data-dir",
        default="data",
        help="Root for --static-auto lookup (default: data)",
    )
    ap.add_argument("--inlier-tol", type=float, default=35.0)
    ap.add_argument("--dtol", type=float, default=20.0)
    ap.add_argument(
        "--rigidfill",
        action="store_true",
        help="Reconstruct missing cluster markers from rigid pose (logged as rigidfill)",
    )
    ap.add_argument(
        "--labelmap-only",
        action="store_true",
        help="Write *_labelmap.csv only; do not write relabeled/corrected CSV",
    )
    args = ap.parse_args()

    static_path = args.static
    if static_path is None and args.static_auto:
        sid = subject_id_from_csv_path(args.csv)
        if sid:
            found = resolve_static_trial(sid, args.data_dir)
            if found:
                static_path = str(found)
            else:
                print(f"warning: no static trial under {args.data_dir}/{sid}/; bootstrapping template from trial")
        else:
            print("warning: could not parse subject id from csv name; bootstrapping template from trial")

    write_out = not args.labelmap_only
    output = args.output
    if write_out and output is None:
        output = str(default_output_path(args.csv))

    relabel_csv(
        args.csv,
        output_csv=output,
        static_path=static_path,
        inlier_tol=args.inlier_tol,
        dtol=args.dtol,
        rigidfill=args.rigidfill,
        write_output=write_out,
    )


if __name__ == "__main__":
    main()
