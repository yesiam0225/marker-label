"""Batch gap-fill trials listed in an obs-trials manifest CSV."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from marker_label.relabel_markers import discover_static_map, resolve_static_trial

from .orchestrator import DEFAULT_GAP_FILLING_CONFIG, gap_fill


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


def gap_filled_output_path(input_csv: Path, output_dir: Path) -> Path:
    """``SUBJ01 Trial 05_corrected.csv`` -> ``{output_dir}/SUBJ01 Trial 05_corrected_gap_filled.csv``."""
    return output_dir / f"{input_csv.stem}_gap_filled.csv"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Batch gap-fill all trials in obs trials.csv into a separate output directory. "
            "Inputs at csv_path are not modified."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full-body rigid fill (thorax RBAK, pelvis, arms, legs) + static reference
  marker-label-batch-gap-fill "corrected/obs trials.csv" \\
    --segments-preset full-body \\
    --output-dir corrected/gap_filled_full_body \\
    --static-map corrected/subject_static_map.csv \\
    --skip-existing

  # Gait legs/feet only (default preset)
  marker-label-batch-gap-fill "corrected/obs trials.csv" \\
    --output-dir corrected/gap_filled \\
    --static-map corrected/subject_static_map.csv \\
    --segments-preset legs-feet

  # Dry-run
  marker-label-batch-gap-fill "corrected/obs trials.csv" \\
    --segments-preset full-body --dry-run
""",
    )
    parser.add_argument(
        "manifest",
        nargs="?",
        default="corrected/obs trials.csv",
        help="Trial manifest CSV (default: corrected/obs trials.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("corrected/gap_filled"),
        help="Directory for gap-filled CSVs and sidecars (default: corrected/gap_filled)",
    )
    parser.add_argument(
        "--static-map",
        type=Path,
        default=None,
        help="CSV with subject_id, static_path for per-subject static reference (.c3d or labeled .csv)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Root for auto static discovery when --static-map is omitted (default: data)",
    )
    parser.add_argument(
        "--segments-preset",
        choices=sorted({"full-body", "lower-body", "legs-feet"}),
        default="legs-feet",
        help=(
            "Segment dictionary for rigid fill (default: legs-feet). "
            "Use full-body to include thorax (RBAK/C7/…), pelvis, arms, head."
        ),
    )
    parser.add_argument(
        "--subject",
        action="append",
        default=[],
        metavar="ID",
        help="Process only this subject (repeatable)",
    )
    parser.add_argument(
        "--max-spline-gap",
        type=int,
        default=None,
        metavar="N",
        help=f"Override spline_max_length (default {DEFAULT_GAP_FILLING_CONFIG['spline_max_length']})",
    )
    parser.add_argument("--max-velocity", type=float, default=None, metavar="MM")
    parser.add_argument("--disable-asis-only-pelvis", action="store_true")
    parser.add_argument("--no-continuity-check", action="store_true")
    parser.add_argument(
        "--enable-shoulder-from-thorax",
        action="store_true",
        help="Fill LSHO/RSHO from C7/CLAV/RBAK using static shoulder offsets (needs --static-map)",
    )
    parser.add_argument("--no-two-marker-rigid", action="store_true")
    parser.add_argument("--config-json", type=Path, default=None)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip trial if gap-filled CSV exists and is newer than input",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max trials per subject (for testing)")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    from marker_label.trial_trim import segment_markers_dict_for_trim_preset

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
        df = df[df["subject_id"].isin(set(args.subject))]

    subject_ids = sorted(df["subject_id"].unique())
    discovered = discover_static_map(args.data_dir, subject_ids)
    explicit_map = load_static_map(args.static_map) if args.static_map else {}

    cfg: dict = dict(DEFAULT_GAP_FILLING_CONFIG)
    if args.config_json:
        extra = json.loads(args.config_json.read_text())
        if not isinstance(extra, dict):
            raise SystemExit("--config-json must contain a JSON object")
        cfg.update(extra)
    if args.max_spline_gap is not None:
        cfg["spline_max_length"] = int(args.max_spline_gap)
    if args.max_velocity is not None:
        cfg["max_velocity_mm_per_frame"] = float(args.max_velocity)
    if args.disable_asis_only_pelvis:
        cfg["asis_only_enabled"] = False
    if args.no_continuity_check:
        cfg["validate_continuity"] = False
    if args.no_two_marker_rigid:
        cfg["allow_two_marker_rigid"] = False
    if args.enable_shoulder_from_thorax:
        cfg["enable_shoulder_from_thorax"] = True

    seg = segment_markers_dict_for_trim_preset(args.segments_preset)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    by_subject: dict[str, list[tuple[Path, Path]]] = defaultdict(list)
    missing_files: list[str] = []

    for _, row in df.iterrows():
        inp = Path(str(row["csv_path"]))
        if not inp.is_file():
            missing_files.append(str(inp))
            continue
        out = gap_filled_output_path(inp, args.output_dir)
        by_subject[str(row["subject_id"])].append((inp, out))

    if missing_files:
        print(f"WARNING: {len(missing_files)} manifest paths missing on disk", file=sys.stderr)
        for p in missing_files[:5]:
            print(f"  {p}", file=sys.stderr)

    ok = fail = skip = 0
    log_path = args.output_dir / "batch_gap_fill_log.csv"
    log_rows: list[dict[str, str]] = []
    preset = str(args.segments_preset)

    print(f"segments-preset={preset}  output-dir={args.output_dir}")

    for sid in sorted(by_subject.keys()):
        static_path = explicit_map.get(sid) or discovered.get(sid)
        if static_path is None:
            static_path = resolve_static_trial(sid, args.data_dir)
        static_str = str(static_path) if static_path else ""
        print(f"\n========== {sid} ({len(by_subject[sid])} trials) static={static_str or 'dynamic'} ==========")

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

            try:
                gap_fill(
                    str(inp),
                    str(out),
                    seg,
                    config=cfg,
                    static_csv_path=static_str or None,
                    verbose=bool(args.verbose),
                )
                ok += 1
                log_rows.append({
                    "subject_id": sid,
                    "input": str(inp),
                    "output": str(out),
                    "static": static_str,
                    "segments_preset": preset,
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
                    "segments_preset": preset,
                    "status": f"fail:{type(e).__name__}:{e}",
                })

    if log_rows and not args.dry_run:
        with log_path.open("w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["subject_id", "input", "output", "static", "segments_preset", "status"],
            )
            w.writeheader()
            w.writerows(log_rows)
        print(f"\nLog: {log_path}")

    print(f"\nDone: {ok} ok, {skip} skipped, {fail} failed")


if __name__ == "__main__":
    main()
