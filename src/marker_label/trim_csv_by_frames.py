"""
Manual frame-range trimming for labeled flat CSVs.

Keeps rows whose ``frame`` column value lies in an inclusive [start_frame, end_frame] range.
Preserves the original header line and ``frame`` / ``time`` values (no renumbering).

Best-frame sidecar (``*.csv.bestframe``) is copied to the output when present and the stored
value lies within the kept frame range (same convention as :mod:`marker_label.trial_trim`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .trial_trim import (
    bestframe_sidecar_path,
    parse_labeled_csv,
    save_bestframe_sidecar,
    save_trimmed_csv,
)


def trim_labeled_csv_by_frame_range(
    input_csv: str | Path,
    output_csv: str | Path,
    start_frame: int,
    end_frame: int,
) -> dict[str, Any]:
    """
    Keep all rows with ``start_frame <= frame <= end_frame`` (inclusive).

    Parameters
    ----------
    input_csv, output_csv
        Input labeled CSV and output path.
    start_frame, end_frame
        Inclusive bounds matching integers in the ``frame`` column (same convention as
        :func:`marker_label.trial_trim.load_best_frame_1based` / pipeline sidecar).

    Returns
    -------
    dict
        ``trim_start_row``, ``trim_end_row`` (half-open row slice into parsed rows),
        ``start_frame``, ``end_frame``, ``sidecar_written``, ``n_rows_kept``.
    """
    if int(end_frame) < int(start_frame):
        raise ValueError(
            f"end_frame ({end_frame}) must be >= start_frame ({start_frame})"
        )

    _, metadata = parse_labeled_csv(input_csv)
    frames: np.ndarray = metadata["frames"]
    n_frames = int(metadata["n_frames"])

    if n_frames == 0:
        raise ValueError("CSV has no data rows")

    f0 = int(frames[0])
    f_last = int(frames[-1])
    sf, ef = int(start_frame), int(end_frame)

    if sf < f0 or ef > f_last:
        raise ValueError(
            f"Requested frame span [{sf}, {ef}] outside CSV frame span [{f0}, {f_last}]"
        )

    trim_start_row = sf - f0
    trim_end_row = ef - f0 + 1  # exclusive

    if trim_start_row < 0 or trim_end_row > n_frames:
        raise ValueError(
            f"Computed row slice [{trim_start_row}, {trim_end_row}) invalid for n_frames={n_frames}"
        )

    if int(frames[trim_start_row]) != sf or int(frames[trim_end_row - 1]) != ef:
        raise ValueError(
            "Frame column does not align with contiguous indexing; cannot trim by frame span."
        )

    save_trimmed_csv(metadata, trim_start_row, trim_end_row, output_csv)

    sidecar_written = False
    in_side = bestframe_sidecar_path(input_csv)
    if in_side.is_file():
        text = in_side.read_text().strip().split()
        if text:
            bf = int(text[0])
            if sf <= bf <= ef:
                save_bestframe_sidecar(output_csv, bf)
                sidecar_written = True

    n_kept = trim_end_row - trim_start_row
    return {
        "trim_start_row": trim_start_row,
        "trim_end_row": trim_end_row,
        "start_frame": sf,
        "end_frame": ef,
        "sidecar_written": sidecar_written,
        "n_rows_kept": n_kept,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Trim a labeled CSV to an inclusive frame range. "
            "Preserves frame/time values and header. "
            "Copies *.csv.bestframe when the best frame lies in the kept range."
        )
    )
    parser.add_argument("input_csv", help="Input labeled CSV")
    parser.add_argument("-o", "--output", required=True, help="Output CSV path")
    parser.add_argument(
        "--start",
        type=int,
        required=True,
        dest="start_frame",
        help="First frame to keep (inclusive; matches `frame` column)",
    )
    parser.add_argument(
        "--end",
        type=int,
        required=True,
        dest="end_frame",
        help="Last frame to keep (inclusive; matches `frame` column)",
    )
    parser.add_argument(
        "--json-summary",
        action="store_true",
        help="Print a JSON summary to stdout",
    )
    args = parser.parse_args()

    info = trim_labeled_csv_by_frame_range(
        args.input_csv,
        args.output,
        args.start_frame,
        args.end_frame,
    )
    if args.json_summary:
        print(json.dumps(info, indent=2))
    else:
        print(
            f"Wrote {args.output}: frames [{info['start_frame']}, {info['end_frame']}] "
            f"({info['n_rows_kept']} rows), sidecar copied={info['sidecar_written']}"
        )


if __name__ == "__main__":
    main()
