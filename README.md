# marker-label

Label marker sets in motion capture C3D files using subject-specific static trials. No learned model by default; obstacle markers are detected by stationarity, body markers by template matching and temporal propagation.

## Features

- **Subject-specific**: Uses each subject's labeled static trial to build a body template.
- **Obstacle markers**: Detected first. **Default:** `rod_pair` mode (median motion + rod geometry on simultaneous finite frames; min visibility **0.72**; motion cap **5.0** mm/frame unless disabled). Labels **OBSTACLE_L** / **OBSTACLE_R**. Use `--obstacle-mode legacy` for the older “two lowest mean-motion columns” rule.
- **Body markers**: Template match at a middle high-quality frame, then propagate labels forward and backward.
- **Gap filling**: Optional export of filled trajectories (linear/spline for short/medium gaps, propagation for long).
- **Output**: Both original (NaNs preserved) and filled C3D and CSV (rows = frames, columns = frame, time, `{marker}_x`, `{marker}_y`, `{marker}_z`).

## Install

```bash
pip install -e .
```

Dependencies: `c3d`, `numpy`, `scipy`. For the 3D QC viewer: `pip install -e ".[qc]"` (adds `pyvista`).

## Usage

### CLI

```bash
# Example: label BBA01 using its static (BBA01 Cal 01.c3d) and unlabeled dynamic (BBA01 Trial 05.c3d)
marker-label "path/to/BBA01 Cal 01.c3d" "path/to/BBA01 Trial 05.c3d" -o out/BBA01_trial05
```

Options:
- `-o, --output`: Output path prefix (default: dynamic path + `_labeled`).
- `--no-filled`: Do not export filled C3D/CSV.
- `--obstacle-visibility FRAC`: Min visibility for obstacle candidates (default **0.72**, for `rod_pair`). Use e.g. **0.80** with `--obstacle-mode legacy` if needed.
- `--obstacle-mode {rod_pair,legacy}`: **Default `rod_pair`** (rod geometry + median motion). `legacy` = two lowest **mean**-motion columns among qualified candidates.
- `--obstacle-max-motion-mm MM`: Candidate motion cap (**median** in `rod_pair`, **mean** in `legacy`). **Omitted = default 5.0 mm/frame.** Use `0` to disable the cap.
- `--max-interp-frames N`: Max gap length for spline interpolation (default 10).
- `--static-facing AXIS`, `--dynamic-facing AXIS`: When static and dynamic were captured with the subject facing different lab axes (e.g. static facing **y**, dynamic facing **x**), use `--static-facing y --dynamic-facing x` so the pipeline rotates dynamic to align with static before matching. Values: `x`, `-x`, `y`, `-y`. Output coordinates remain in the original lab frame.
- `--static-unit {mm,m}`, `--dynamic-unit {mm,m}`: Unit of the C3D coordinates (default `mm`). Use `m` if the file is in meters; coordinates are scaled to mm internally so static and dynamic match.
- `--max-match-distance MM`: Reject initial assignment when point–template distance > MM mm (e.g. 300) to avoid bad matches.
- `--max-propagation-distance MM`: Do not propagate a label to the nearest point if it is > MM mm away (e.g. 150) to reduce swaps after dropout or when markers cross.
- `--reference-report JSON`: Use reference analysis (from `marker-label-analyze -o`) to set best-frame search window and optional distance thresholds. See [docs/REFERENCE_GUIDED_LABELING.md](docs/REFERENCE_GUIDED_LABELING.md).
- `--drop-extra-stationary-below-mm MM`: After obstacle detection, drop non-obstacle screened columns whose full-trial mean inter-frame speed is below `MM` mm/frame (and visibility meets the obstacle threshold). **Omitted = use the built-in default (standard processing).** Use `0` only in exceptional cases; document why.
- `--no-drop-extra-stationary`: Turn off that drop entirely (overrides default). **Standard runs leave it on;** use only for exceptional cases and document why.
- `--no-check-screened-count`: Allow screened column count other than 41 (e.g. extra hardware channels).

### Python

```python
from marker_label.pipeline import run_pipeline

# Standard processing: omit ``drop_extra_stationary_motion_max_mm`` (default threshold when two obstacles exist).
run_pipeline(
    "static_labeled.c3d",
    "dynamic_unlabeled.c3d",
    "out/trial_01",
    # obstacle_visibility_min defaults to 0.72 (rod_pair); pass 0.80 for legacy mode
    static_facing_axis="y",   # optional: subject facing y in static
    dynamic_facing_axis="x",  # optional: subject facing x in dynamic
    export_filled=True,
    max_interp_frames=10,
)
```

### Quality inspection

After labeling, check quality of the output CSV:

```bash
marker-label-inspect out/trial_01_labeled.csv
```

Options:
- `--velocity-threshold MM`: Flag velocity jumps above this (mm). Default: 100.
- `--visibility-warn FRAC`: Mark visibility below this as low. Default: 0.80.
- `--max-jumps N`: Max number of velocity jumps to list. Default: 20.

The report shows per-marker visibility, large frame-to-frame velocity jumps (possible swaps or bad assignment), and obstacle stationarity (should be near 0 mm).

From Python:

```python
from marker_label.inspect_quality import run_quality_report, print_quality_report

print_quality_report("out/trial_01_labeled.csv")
# or
report = run_quality_report("out/trial_01_labeled.csv")
# report["visibility"], report["velocity_jumps"], report["obstacle_std"], etc.
```

### 3D QC viewer

View labeled C3D or CSV in 3D with **body segments** (sticks between markers) and playback. Segments are defined in `marker_label.segments.SEGMENTS` (Vicon-style: head, thorax, pelvis, arms, legs). Requires `[qc]`: `pip install -e ".[qc]"`.

```bash
marker-label-view path/to/trial_01_labeled.c3d
# or
marker-label-view path/to/trial_01_labeled.csv
```

Options: `--point-size` (default 12), `--speed` playback multiplier (default 1), `--background` (`white` or `black`), `--segment-color` (default `darkblue`), `--scale FACTOR` (e.g. 1000 if the file is in meters so markers display at mm scale).

To add or change segments, edit `src/marker_label/segments.py`: each segment is an ordered list of marker names; consecutive pairs are drawn as lines (see docstring).

From Python:

```python
from marker_label.qc_viewer import run_viewer, load_data

run_viewer("trial_01_labeled.c3d", point_size=12, playback_speed=1, background="white")
```

### Compare static vs dynamic (same scene)

To see why labeling might fail, view the **static template** (green, labeled) and **dynamic trial** (red, one frame) in one 3D window. Use the slider to scrub through dynamic frames.

```bash
marker-label-compare path/to/static_labeled.c3d path/to/dynamic.c3d
# optional: align orientations (same as pipeline)
marker-label-compare path/to/static.c3d path/to/dynamic.c3d --static-facing y --dynamic-facing x
```

Options: `--frame N`, `--point-size`, `--background`, `--static-facing`, `--dynamic-facing`.

### Reference-subject analysis (`marker-label-analyze`)

Analyze a **manually labeled** static + dynamic pair from a reference subject to see how the static template relates to each dynamic frame (per-frame rigid transform, RMS, and which frame is “closest” to the template). Useful to tune pipeline behavior or inspect typical rotation/translation.

**Input files:** You must provide the **paths** to two C3D files from one **reference subject**, both **manually labeled** with the same marker names (e.g. in Vicon Nexus). Example: reference subject **BBpilot01** — manually labeled static `BBpilot01 Cal 01.c3d`, manually labeled dynamic `BBpilot01 Trial 10.c3d`. (For the **subject you want to label**, e.g. BBA01, you use that subject’s labeled static and unlabeled dynamic with the main `marker-label` pipeline; the analyze tool is for a separate, reference subject.)

```bash
# Example: analyze reference subject BBpilot01 (manually labeled static + dynamic)
marker-label-analyze "path/to/BBpilot01 Cal 01.c3d" "path/to/BBpilot01 Trial 10.c3d"
marker-label-analyze "path/to/BBpilot01 Cal 01.c3d" "path/to/BBpilot01 Trial 10.c3d" --sample 10 -o report.json
```

Options: `--sample N` (analyze every Nth frame; default 1), `--pelvis-frame` (build template in pelvis frame), `--static-unit {mm,m}`, `--dynamic-unit {mm,m}` (default mm; use m if file is in meters), `-o report.json` (write full result as JSON).

## Pipeline summary

1. Load static (labeled) and dynamic (unlabeled) C3D.
2. Initial screening on the dynamic trial (Y range, optional frame trim, visibility) as configured.
3. Detect obstacle markers (default **`rod_pair`**: median motion, rod geometry, min visibility **0.72**; or **`legacy`**: two lowest mean-motion columns among qualified); remove from body labeling set.
4. **Extra stationary drop (standard):** After obstacles are known, drop other screened columns that are highly stationary and visible (full-trial mean inter-frame speed below a default mm/frame threshold). Runs when two obstacles are detected. **Default CLI/API behavior keeps this enabled** (`None` → default threshold). Disabling (`--no-drop-extra-stationary` or `--drop-extra-stationary-below-mm 0`) is for **exceptional** cases only (e.g. extra capture channels, debugging); note the reason in your workflow or PR when you disable it.
5. Build body template from static (mean position per label).
6. Select best frame in dynamic (middle portion, high quality, or a fixed frame if set).
7. Match remaining markers to template; propagate labels temporally (unless column-fixed mode).
8. Build full output: body labels (static order, NaN where missing) + OBSTACLE_L, OBSTACLE_R.
9. Export original and filled C3D + CSV.

For a detailed explanation of the logic and what to check when labeling fails, see [docs/MARKER_LABELING_LOGIC.md](docs/MARKER_LABELING_LOGIC.md).

## Pelvis markers

Vicon full-body pelvis markers: LASI, RASI, LPSI, RPSI (and SACR if present). Used when building template in pelvis frame; currently template matching defaults to lab frame.

## Related projects

Labeled CSV output (`frame`, `{marker}_x/y/z`) is the input contract for downstream gait analysis:

| Repository | Role |
|------------|------|
| [gait-spatiotemporal](https://github.com/gait-spatiotemporal/gait-spatiotemporal) | Rule-based IC/TO detection and spatiotemporal parameters |
| [gait-events-vlm](https://github.com/gait-events-vlm/gait-events-vlm) | VLM-based IC/TO from foot Z plots (experimental / QC) |

## License

See repository.
