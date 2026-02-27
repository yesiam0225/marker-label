# marker-label

Label marker sets in motion capture C3D files using subject-specific static trials. No learned model by default; obstacle markers are detected by stationarity, body markers by template matching and temporal propagation.

## Features

- **Subject-specific**: Uses each subject's labeled static trial to build a body template.
- **Obstacle markers**: Detected first (stationarity + visibility ≥ 80%), then labeled OBSTACLE_L / OBSTACLE_R.
- **Body markers**: Template match at a middle high-quality frame, then propagate labels forward and backward.
- **Gap filling**: Optional export of filled trajectories (linear/spline for short/medium gaps, propagation for long).
- **Output**: Both original (NaNs preserved) and filled C3D and CSV (rows = frames, columns = frame, time, `{marker}_x`, `{marker}_y`, `{marker}_z`).

## Install

```bash
pip install -e .
```

Dependencies: `c3d`, `numpy`, `scipy`.

## Usage

### CLI

```bash
marker-label path/to/static_labeled.c3d path/to/dynamic_unlabeled.c3d -o out/trial_01
```

Options:
- `-o, --output`: Output path prefix (default: dynamic path + `_labeled`).
- `--no-filled`: Do not export filled C3D/CSV.
- `--obstacle-visibility FRAC`: Min visibility for obstacle candidates (default 0.80).
- `--max-interp-frames N`: Max gap length for spline interpolation (default 10).

### Python

```python
from marker_label.pipeline import run_pipeline

run_pipeline(
    "static_labeled.c3d",
    "dynamic_unlabeled.c3d",
    "out/trial_01",
    obstacle_visibility_min=0.80,
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

## Pipeline summary

1. Load static (labeled) and dynamic (unlabeled) C3D.
2. Detect obstacle markers (2 most stationary with ≥80% visibility); remove from set.
3. Build body template from static (mean position per label).
4. Select best frame in dynamic (middle portion, high quality).
5. Match remaining markers to template; propagate labels temporally.
6. Build full output: body labels (static order, NaN where missing) + OBSTACLE_L, OBSTACLE_R.
7. Export original and filled C3D + CSV.

## Pelvis markers

Vicon full-body pelvis markers: LASI, RASI, LPSI, RPSI (and SACR if present). Used when building template in pelvis frame; currently template matching defaults to lab frame.

## License

See repository.
