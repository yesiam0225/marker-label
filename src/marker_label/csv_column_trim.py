"""Remove marker xyz triplets from a labeled CSV by marker stem name rules."""

from __future__ import annotations

import csv
import re
from collections.abc import Sequence
from pathlib import Path


def iter_marker_xyz_triplets(header: Sequence[str]) -> list[tuple[int, int, int, str]]:
    """
    Parse labeled CSV header after ``frame`` and ``time``.

    Returns
    -------
    list of (ix, iy, iz, stem) where stem is the marker name (without ``_x`` / ``_y`` / ``_z``).
    Stops at the first column group that is not a consistent ``_x/_y/_z`` triplet.
    """
    h = [str(c).strip() for c in header]
    if len(h) < 2 or h[0].lower() != "frame" or h[1].lower() != "time":
        raise ValueError(
            "Expected header to start with frame, time (case-insensitive); "
            f"got {h[0]!r}, {h[1]!r}."
        )
    out: list[tuple[int, int, int, str]] = []
    i = 2
    while i + 2 < len(h):
        xs, ys, zs = h[i], h[i + 1], h[i + 2]
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
        out.append((i, i + 1, i + 2, stem_x))
        i += 3
    return out


def stems_matching_any_regex(stems: Sequence[str], patterns: Sequence[re.Pattern[str]]) -> set[str]:
    """Return stems for which ``any(p.search(stem.strip()) for p in patterns)``."""
    matched: set[str] = set()
    for s in stems:
        st = str(s).strip()
        if any(p.search(st) for p in patterns):
            matched.add(st)
    return matched


def trim_labeled_csv_marker_columns(
    input_path: str | Path,
    output_path: str | Path,
    *,
    stem_regexes: Sequence[str],
) -> tuple[int, int, list[str]]:
    """
    Write a new CSV with the same ``frame``/``time`` and marker triplets whose stems
    do **not** match any of ``stem_regexes`` (``re.search`` on stripped stem).

    Returns
    -------
    n_kept_markers, n_dropped_markers, dropped_stems_sorted
    """
    rx_list = [re.compile(p) for p in stem_regexes]
    in_path = Path(input_path)
    with open(in_path, newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"empty CSV: {in_path}")
    header = rows[0]
    triplets = iter_marker_xyz_triplets(header)
    stems = [t[3] for t in triplets]
    drop_stems = stems_matching_any_regex(stems, rx_list)
    keep_idx = [0, 1]
    for ix, iy, iz, stem in triplets:
        if stem in drop_stems:
            continue
        keep_idx.extend([ix, iy, iz])
    # Preserve any trailing columns after last triplet (should not happen for standard export)
    last_trip_end = 2 + 3 * len(triplets)
    if last_trip_end < len(header):
        keep_idx.extend(range(last_trip_end, len(header)))

    new_header = [header[j] for j in keep_idx]
    out_rows = [new_header]
    for row in rows[1:]:
        if len(row) < len(header):
            row = row + [""] * (len(header) - len(row))
        out_rows.append([row[j] for j in keep_idx])

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerows(out_rows)

    n_dropped = len(drop_stems)
    n_kept = len(triplets) - n_dropped
    return n_kept, n_dropped, sorted(drop_stems)


# Presets: common junk in labeled exports (numeric unlabeled + *N placeholders)
PRESET_UNLABELED_PLACEHOLDER_REGEXES: tuple[str, ...] = (
    r"^\*\d+$",  # *115, *116, ...
    r"^\d+$",  # 45, 55, … (loaded index display, digits-only stem)
)
