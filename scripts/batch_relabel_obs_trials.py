#!/usr/bin/env python3
"""
Batch relabel all trials in obs trials.csv, grouped by subject.

Each subject uses one static reference (from --static-map or auto-discovery under
``data/{subject_id}/``). Processes *_corrected.csv, *_trimmed.csv, and *_labeled.csv
paths listed in the manifest.

Example:
  python3 scripts/batch_relabel_obs_trials.py "corrected/obs trials.csv" \\
    --static-map corrected/subject_static_map.csv \\
    --in-place --backup --rigidfill

Generate a static-map template (edit rows with empty static_path):
  python3 scripts/batch_relabel_obs_trials.py "corrected/obs trials.csv" \\
    --write-static-map corrected/subject_static_map.csv
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from marker_label.relabel_markers import (  # noqa: E402
    default_output_path,
    discover_static_map,
    relabel_csv,
    resolve_static_trial,
)


def load_static_map(path: Path) -> dict[str, Path]:
    """Read ``subject_id,static_path`` CSV; skip blank paths."""
    out: dict[str, Path] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "subject_id" not in reader.fieldnames:
            raise ValueError(f"{path} must have columns: subject_id, static_path")
        sp_col = "static_path" if "static_path" in reader.fieldnames else reader.fieldnames[1]
        for row in reader:
            sid = str(row["subject_id"]).strip()
            raw = str(row.get(sp_col, "")).strip()
            if not sid or not raw:
                continue
            p = Path(raw)
            if not p.is_file():
                raise FileNotFoundError(f"static_path for {sid} not found: {p}")
            out[sid] = p
    return out


def write_static_map_template(
    path: Path,
    subject_ids: list[str],
    data_dir: Path,
    discovered: dict[str, Path | None],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject_id", "static_path", "auto_discovered", "notes"])
        for sid in subject_ids:
            auto = discovered.get(sid)
            w.writerow([
                sid,
                str(auto) if auto else "",
                "yes" if auto else "no",
                "" if auto else f"add path to static trial for {sid}",
            ])
    print(f"Wrote {path}")


def resolve_output_path(
    inp: Path,
    *,
    in_place: bool,
    output_dir: Path | None,
) -> Path:
    if in_place:
        return inp
    if output_dir is not None:
        return output_dir / inp.name
    return default_output_path(inp)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "manifest",
        nargs="?",
        default="corrected/obs trials.csv",
        help="Trial manifest CSV (default: corrected/obs trials.csv)",
    )
    parser.add_argument(
        "--static-map",
        type=Path,
        default=None,
        help="CSV with subject_id, static_path (overrides auto-discovery per subject)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Root for auto static discovery (default: data)",
    )
    parser.add_argument(
        "--write-static-map",
        type=Path,
        default=None,
        metavar="CSV",
        help="Write subject static template and exit",
    )
    parser.add_argument(
        "--subject",
        action="append",
        default=[],
        metavar="ID",
        help="Process only this subject (repeatable, e.g. --subject BBA01)",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite each trial CSV at csv_path (default when not using --output-dir)",
    )
    parser.add_argument(
        "--no-in-place",
        action="store_true",
        help="Write to *_corrected.csv (trimmed) or *_relabeled.csv (others) beside input",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Write all outputs under this directory (implies --no-in-place)",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="With --in-place, copy input to *.pre_relabel.bak before overwrite",
    )
    parser.add_argument("--rigidfill", action="store_true")
    parser.add_argument("--inlier-tol", type=float, default=35.0)
    parser.add_argument("--dtol", type=float, default=20.0)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip trial if output CSV exists and is newer than input",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max trials per subject (for testing)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    import pandas as pd

    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        print(f"Manifest not found: {manifest_path.resolve()}", file=sys.stderr)
        raise SystemExit(2)

    df = pd.read_csv(manifest_path)
    if "csv_path" not in df.columns or "subject_id" not in df.columns:
        print("Manifest must include csv_path and subject_id columns", file=sys.stderr)
        raise SystemExit(2)

    if args.subject:
        keep = set(args.subject)
        df = df[df["subject_id"].isin(keep)]

    subject_ids = sorted(df["subject_id"].unique())
    discovered = discover_static_map(args.data_dir, subject_ids)

    if args.write_static_map:
        write_static_map_template(args.write_static_map, subject_ids, args.data_dir, discovered)
        return

    explicit_map = load_static_map(args.static_map) if args.static_map else {}
    if args.output_dir is not None:
        in_place = False
        args.output_dir.mkdir(parents=True, exist_ok=True)
    else:
        in_place = not args.no_in_place

    by_subject: dict[str, list[tuple[Path, Path]]] = defaultdict(list)
    missing_files: list[str] = []

    for _, row in df.iterrows():
        inp = Path(str(row["csv_path"]))
        if not inp.is_file():
            missing_files.append(str(inp))
            continue
        out = resolve_output_path(inp, in_place=in_place, output_dir=args.output_dir)
        by_subject[str(row["subject_id"])].append((inp, out))

    if missing_files:
        print(f"WARNING: {len(missing_files)} manifest paths missing on disk", file=sys.stderr)
        for p in missing_files[:5]:
            print(f"  {p}", file=sys.stderr)

    ok = fail = skip = 0
    log_path = ROOT / "corrected" / "batch_relabel_log.csv"
    log_rows: list[dict[str, str]] = []

    for sid in sorted(by_subject.keys()):
        static_path = explicit_map.get(sid) or discovered.get(sid)
        if static_path is None:
            static_path = resolve_static_trial(sid, args.data_dir)
        static_str = str(static_path) if static_path else ""
        print(f"\n========== {sid} ({len(by_subject[sid])} trials) static={static_str or 'BOOTSTRAP'} ==========")

        trials = by_subject[sid]
        if args.limit:
            trials = trials[: args.limit]

        for inp, out in trials:
            if args.skip_existing and out.is_file() and out.stat().st_mtime >= inp.stat().st_mtime:
                print(f"SKIP (up-to-date): {out}")
                skip += 1
                continue

            if args.dry_run:
                print(f"DRY-RUN {inp} -> {out}")
                continue

            if in_place and args.backup and inp.is_file():
                bak = inp.with_suffix(inp.suffix + ".pre_relabel.bak")
                if not bak.is_file():
                    shutil.copy2(inp, bak)

            bf = Path(str(inp) + ".bestframe")
            try:
                relabel_csv(
                    inp,
                    output_csv=out,
                    static_path=static_str or None,
                    inlier_tol=args.inlier_tol,
                    dtol=args.dtol,
                    rigidfill=args.rigidfill,
                    write_output=True,
                )
                if out != inp and bf.is_file():
                    out_bf = Path(str(out) + ".bestframe")
                    shutil.copy2(bf, out_bf)
                ok += 1
                log_rows.append({
                    "subject_id": sid,
                    "input": str(inp),
                    "output": str(out),
                    "static": static_str,
                    "status": "ok",
                })
            except Exception as e:
                print(f"FAIL {inp}: {e}", file=sys.stderr)
                fail += 1
                log_rows.append({
                    "subject_id": sid,
                    "input": str(inp),
                    "output": str(out),
                    "static": static_str,
                    "status": f"fail:{type(e).__name__}:{e}",
                })

    if log_rows and not args.dry_run:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["subject_id", "input", "output", "static", "status"])
            w.writeheader()
            w.writerows(log_rows)
        print(f"\nLog: {log_path}")

    print(f"\nDone: {ok} ok, {skip} skipped, {fail} failed")


if __name__ == "__main__":
    main()
