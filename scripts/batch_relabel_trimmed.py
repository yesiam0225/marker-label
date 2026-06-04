#!/usr/bin/env python3
"""Batch-run marker-label-relabel on corrected/*_trimmed.csv files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from marker_label.relabel_markers import (  # noqa: E402
    default_output_path,
    relabel_csv,
    resolve_static_trial,
    subject_id_from_csv_path,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Relabel rigid-cluster markers on all *_trimmed.csv under a directory.",
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default="corrected",
        help="Directory containing *_trimmed.csv (default: corrected)",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Root for subject static trials (default: data)",
    )
    parser.add_argument(
        "--static-auto",
        action="store_true",
        default=True,
        help="Use data/{subject}/{subject} Cal 01.csv|.c3d when available (default: on)",
    )
    parser.add_argument(
        "--no-static-auto",
        action="store_true",
        help="Bootstrap cluster templates from each trial only",
    )
    parser.add_argument("--rigidfill", action="store_true")
    parser.add_argument("--inlier-tol", type=float, default=35.0)
    parser.add_argument("--dtol", type=float, default=20.0)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip when output *_corrected.csv already exists",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    files = sorted(in_dir.glob("*_trimmed.csv"))
    if args.limit:
        files = files[: args.limit]
    if not files:
        print(f"No *_trimmed.csv under {in_dir.resolve()}", file=sys.stderr)
        raise SystemExit(2)

    static_auto = args.static_auto and not args.no_static_auto
    ok = fail = skip = 0
    for inp in files:
        out = default_output_path(inp)
        if args.skip_existing and out.is_file():
            print(f"SKIP (exists): {out}")
            skip += 1
            continue
        static_path = None
        if static_auto:
            sid = subject_id_from_csv_path(inp)
            if sid:
                found = resolve_static_trial(sid, args.data_dir)
                if found:
                    static_path = found
                    print(f"=== {inp.name} static={found} ===")
                else:
                    print(f"=== {inp.name} (no static under {args.data_dir}/{sid}/) ===")
            else:
                print(f"=== {inp.name} (no subject id in filename) ===")
        else:
            print(f"=== {inp.name} ===")
        try:
            relabel_csv(
                inp,
                output_csv=out,
                static_path=static_path,
                inlier_tol=args.inlier_tol,
                dtol=args.dtol,
                rigidfill=args.rigidfill,
                write_output=True,
            )
            ok += 1
        except Exception as e:
            print(f"FAIL {inp}: {e}", file=sys.stderr)
            fail += 1

    print(f"\nDone: {ok} ok, {skip} skipped, {fail} failed, {len(files)} total")


if __name__ == "__main__":
    main()
