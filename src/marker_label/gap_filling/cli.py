"""CLI: marker-label-gap-fill."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from marker_label.trial_trim import segment_markers_dict_for_trim_preset

from .orchestrator import DEFAULT_GAP_FILLING_CONFIG, gap_fill

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Fill missing marker positions in a corrected labeled CSV.",
    )
    parser.add_argument("input_csv", type=str, help="Marker-corrected CSV (e.g. *_corrected.csv)")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output gap-filled CSV path",
    )
    parser.add_argument(
        "--segments-preset",
        choices=sorted({"full-body", "lower-body", "legs-feet"}),
        default="full-body",
        help="Segment dictionary preset (same as marker-label-trial-trim / correction)",
    )
    parser.add_argument(
        "--max-spline-gap",
        type=int,
        default=None,
        metavar="N",
        help=f"Override spline_max_length (default {DEFAULT_GAP_FILLING_CONFIG['spline_max_length']})",
    )
    g = parser.add_mutually_exclusive_group()
    g.add_argument(
        "--enable-asis-only-pelvis",
        action="store_true",
        default=None,
        help="Enable ASIS-only pelvis fallback for PSI markers (default: on)",
    )
    g.add_argument(
        "--disable-asis-only-pelvis",
        action="store_true",
        help="Disable ASIS-only pelvis fallback",
    )
    parser.add_argument(
        "--max-velocity",
        type=float,
        default=None,
        metavar="MM",
        help=f"Override max_velocity_mm_per_frame (default {DEFAULT_GAP_FILLING_CONFIG['max_velocity_mm_per_frame']})",
    )
    parser.add_argument(
        "--no-continuity-check",
        action="store_true",
        help="Disable continuity validation after fills",
    )
    parser.add_argument(
        "--static-csv",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Subject labeled static trial: flat .csv (frame, time, marker_x/y/z) "
            "or labeled .c3d with anatomical marker names"
        ),
    )
    parser.add_argument(
        "--enable-shoulder-from-thorax",
        action="store_true",
        help="Fill LSHO/RPSI gaps from C7/CLAV/RBAK using static shoulder-local offsets (walking trials)",
    )
    parser.add_argument(
        "--no-two-marker-rigid",
        action="store_true",
        help="Disable 2-visible-marker rigid fill for 3-marker foot/hand segments",
    )
    parser.add_argument(
        "--config-json",
        type=str,
        default=None,
        help="Merge JSON object into gap-fill config",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="DEBUG logging (per-frame decisions)",
    )
    args = parser.parse_args()

    cfg: dict = dict(DEFAULT_GAP_FILLING_CONFIG)
    if args.config_json:
        extra = json.loads(Path(args.config_json).read_text())
        if not isinstance(extra, dict):
            raise SystemExit("--config-json must contain a JSON object")
        cfg.update(extra)

    if args.max_spline_gap is not None:
        cfg["spline_max_length"] = int(args.max_spline_gap)
    if args.max_velocity is not None:
        cfg["max_velocity_mm_per_frame"] = float(args.max_velocity)
    if args.no_continuity_check:
        cfg["validate_continuity"] = False
    if args.no_two_marker_rigid:
        cfg["allow_two_marker_rigid"] = False
    if args.enable_shoulder_from_thorax:
        cfg["enable_shoulder_from_thorax"] = True

    if args.disable_asis_only_pelvis:
        cfg["asis_only_enabled"] = False
    elif args.enable_asis_only_pelvis is True:
        cfg["asis_only_enabled"] = True
    # default True from DEFAULT_GAP_FILLING_CONFIG

    seg = segment_markers_dict_for_trim_preset(args.segments_preset)
    try:
        gap_fill(
            args.input_csv,
            args.output,
            seg,
            config=cfg,
            static_csv_path=args.static_csv,
            verbose=bool(args.verbose),
        )
    except Exception as e:
        logger.exception("Gap fill failed")
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(2) from e
    print(json.dumps({"output": args.output}, indent=2))


if __name__ == "__main__":
    main()
