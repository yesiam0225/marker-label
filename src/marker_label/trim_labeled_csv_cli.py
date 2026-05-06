"""CLI: drop marker columns from a labeled CSV by stem regex or preset."""

from __future__ import annotations

import argparse
import sys

from .csv_column_trim import PRESET_UNLABELED_PLACEHOLDER_REGEXES, trim_labeled_csv_marker_columns


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Remove marker xyz triplets from a labeled CSV when the marker stem "
            "(name before _x/_y/_z) matches a regex. Keeps frame, time."
        ),
    )
    parser.add_argument("input_csv", help="Path to labeled CSV (e.g. *_labeled.csv)")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output CSV path (default: input with _stemtrim before .csv)",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite input file (use with care).",
    )
    parser.add_argument(
        "--preset",
        choices=("unlabeled-placeholder",),
        default=None,
        help=(
            "unlabeled-placeholder: drop stems matching *<digits> or digits-only "
            "(same as export extras for unlabeled channels)."
        ),
    )
    parser.add_argument(
        "--stem-regex",
        action="append",
        default=[],
        metavar="PATTERN",
        help="Python re pattern; drop triplet if re.search matches stripped stem (repeatable).",
    )
    args = parser.parse_args()
    in_csv = args.input_csv
    regexes: list[str] = []
    if args.preset == "unlabeled-placeholder":
        regexes.extend(PRESET_UNLABELED_PLACEHOLDER_REGEXES)
    regexes.extend(args.stem_regex or [])
    if not regexes:
        print(
            "Error: specify --preset unlabeled-placeholder and/or at least one --stem-regex.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    if args.in_place:
        out = in_csv
    elif args.output:
        out = args.output
    else:
        p = in_csv.rsplit(".", 1)
        out = f"{p[0]}_stemtrim.csv" if len(p) == 2 else f"{in_csv}_stemtrim.csv"

    n_kept, n_drop, dropped = trim_labeled_csv_marker_columns(
        in_csv, out, stem_regexes=regexes
    )
    print(
        f"Wrote {out!r}: kept {n_kept} marker(s), dropped {n_drop} marker stem(s): {dropped}"
    )


if __name__ == "__main__":
    main()
