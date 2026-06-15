"""Batch synthesize LFIN/RFIN for trials listed in an obs-trials manifest."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

from marker_label.relabel_markers import discover_static_map, resolve_static_trial
from marker_label.synthesize_fin_from_static import HAND_MARKERS, HandSide, synthesize_fin_csv
from marker_label.trial_trim import parse_labeled_csv


def load_static_map(path: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        sp_col = "static_path" if "static_path" in (reader.fieldnames or []) else (reader.fieldnames or [""])[1]
        for row in reader:
            sid = str(row["subject_id"]).strip()
            raw = str(row.get(sp_col, "")).strip()
            if sid and raw:
                p = Path(raw)
                if not p.is_file():
                    raise FileNotFoundError(f"static_path for {sid} not found: {p}")
                out[sid] = p
    return out


def trial_needs_fin(stems: list[str], side: HandSide) -> bool:
    return HAND_MARKERS[side][2] not in stems


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Batch add/fill LFIN/RFIN from subject static trials (obs trials.csv manifest). "
            "Use before full-body gap fill when FIN columns are absent."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  marker-label-batch-synthesize-fin "corrected/obs trials.csv" \\
    --static-map corrected/subject_static_map.csv \\
    --subject BBC13 --side right --only-if-fin-missing --backup
""",
    )
    parser.add_argument("manifest", nargs="?", default="corrected/obs trials.csv")
    parser.add_argument("--static-map", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--subject", action="append", default=[], metavar="ID")
    parser.add_argument(
        "--side",
        choices=("left", "right", "both"),
        default="right",
        help="Hand(s) to synthesize (default: right / RFIN)",
    )
    parser.add_argument(
        "--only-if-fin-missing",
        action="store_true",
        help="Process trial only when FIN column is absent",
    )
    parser.add_argument(
        "--also-fill-nan",
        action="store_true",
        help="When FIN column exists, fill NaN frames where WRA+WRB are visible",
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Re-synthesize all FIN frames (e.g. switch wrist -> forearm method)",
    )
    parser.add_argument(
        "--method",
        choices=("forearm", "wrist", "auto"),
        default="forearm",
        help="FIN synthesis method (default: forearm = RFRM+RWRA+RWRB frame)",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--backup", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    import pandas as pd

    sides = {"left": ("L",), "right": ("R",), "both": ("L", "R")}[args.side]

    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        print(f"Manifest not found: {manifest_path}", file=sys.stderr)
        raise SystemExit(2)

    df = pd.read_csv(manifest_path)
    if args.subject:
        df = df[df["subject_id"].isin(set(args.subject))]

    subject_ids = sorted(df["subject_id"].unique())
    discovered = discover_static_map(args.data_dir, subject_ids)
    explicit = load_static_map(args.static_map) if args.static_map else {}

    in_place = args.output_dir is None
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    ok = skip = fail = 0
    log_rows: list[dict[str, str]] = []
    log_path = (args.output_dir or Path("corrected")) / "batch_synthesize_fin_log.csv"

    for _, row in df.iterrows():
        sid = str(row["subject_id"])
        inp = Path(str(row["csv_path"]))
        if not inp.is_file():
            skip += 1
            continue

        stems, _ = parse_labeled_csv(inp)
        if args.only_if_fin_missing and not any(trial_needs_fin(stems, s) for s in sides):
            skip += 1
            continue

        out = inp if in_place else args.output_dir / inp.name
        static_path = explicit.get(sid) or discovered.get(sid) or resolve_static_trial(sid, args.data_dir)
        if static_path is None:
            print(f"FAIL {inp}: no static trial for {sid}", file=sys.stderr)
            fail += 1
            continue

        if args.dry_run:
            print(f"DRY-RUN {inp} -> {out} static={static_path} sides={sides}")
            continue

        if in_place and args.backup:
            bak = inp.with_suffix(inp.suffix + ".pre_fin.bak")
            if not bak.is_file():
                shutil.copy2(inp, bak)

        try:
            info = synthesize_fin_csv(
                static_path,
                inp,
                out,
                sides=sides,
                method=args.method,
                only_if_fin_missing=bool(args.only_if_fin_missing),
                fill_nan_in_existing=bool(args.also_fill_nan),
                replace_existing=bool(args.replace_existing),
            )
            ok += 1
            log_rows.append({
                "subject_id": sid,
                "input": str(inp),
                "output": str(out),
                "static": str(static_path),
                "inserted": str(info["inserted"]),
                "status": "ok",
            })
            print(f"OK {inp.name} inserted={info['inserted']}")
        except Exception as e:
            fail += 1
            print(f"FAIL {inp}: {e}", file=sys.stderr)
            log_rows.append({
                "subject_id": sid,
                "input": str(inp),
                "output": str(out),
                "static": str(static_path),
                "inserted": "",
                "status": f"fail:{type(e).__name__}:{e}",
            })

        if args.limit and ok >= args.limit:
            break

    if log_rows and not args.dry_run:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["subject_id", "input", "output", "static", "inserted", "status"],
            )
            w.writeheader()
            w.writerows(log_rows)
        print(f"Log: {log_path}")

    print(f"Done: {ok} ok, {skip} skipped, {fail} failed")


if __name__ == "__main__":
    main()
