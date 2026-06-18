# Gait analysis pipeline

Obstacle-crossing gait kinematics and margin-of-stability (MoS) analysis (adult/child × RB/WB × pre/post).

## Layout

```
gait_analysis/
  src/                    # Python modules and CLIs
  data/                   # obs_trials.csv, per_stride_data.csv (symlinks to repo data)
  output/                 # Generated CSVs and plots (gitignored)
  tests/test_smoke.py     # Single-trial smoke test (SUBJ01 T5)
  run_all.py              # Run all four pipelines in sequence
  requirements.txt
```

Trial marker CSVs live in the parent repo at `corrected/` (paths in `obs_trials.csv` use `corrected/SUBJ01 Trial 05_corrected.csv`).

## Setup

```bash
cd gait_analysis
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Data symlinks (already created if you cloned this repo):

```bash
ln -sf "../../corrected/obs trials.csv" data/obs_trials.csv
ln -sf "../../gait_spatiotemporal_out/per_stride_data.csv" data/per_stride_data.csv
```

Optional environment overrides:

- `GAIT_DATA_DIR` — default `gait_analysis/data`
- `GAIT_OUTPUT_DIR` — default `gait_analysis/output`
- `GAIT_TRIAL_DIR` — default repo root (parent of `gait_analysis/`)

## Modules

| File | Role |
|------|------|
| `gait_kinematics.py` | Sagittal joint angles (Newington-Gage HJC), walking-direction normalization, gap fill |
| `gait_mos.py` | Whole-body COM (Dempster), XCOM, MoS at HS / mid-swing / foot-off |
| `batch_kinematics_ensemble.py` | 36 wide CSVs: phase × joint × signal, 0–100% gait cycle |
| `batch_mos.py` | `mos_all_strides.csv` + `mos_subject_condition.csv` |
| `batch_mos_timeseries.py` | MoS AP/ML time-series ensembles (raw + height-normalized) |
| `visualize_mos.py` | Per-trial 3-panel QC plots |

Shared helpers: `trial_paths.resolve_trial_csv`, `paths.py`, `cli_common.py`.

## Run examples

Single pipeline (defaults use `./data/` and `./output/`):

```bash
cd src
python batch_kinematics_ensemble.py --filter-trials SUBJ01:5
python batch_mos.py --filter-trials SUBJ01:5
python batch_mos_timeseries.py --filter-trials SUBJ01:5
python visualize_mos.py --filter-trials SUBJ01:5
```

All four pipelines:

```bash
python run_all.py --filter-trials SUBJ01:5
```

Full dataset (exclude known-bad trials with `--filter-trials` if needed):

```bash
python run_all.py --filter-trials "$(python -c "
import pandas as pd
obs = pd.read_csv('data/obs_trials.csv')
exclude = {('SUBJ02', 57)}  # example: skip unreliable gap fill
pairs = [f\"{r.subject_id}:{r.trial}\" for _, r in obs.iterrows()
         if (r.subject_id, int(r.trial)) not in exclude]
print(','.join(pairs))
")"
```

Or process everything and inspect logs for per-trial failures:

```bash
python run_all.py
```

## Outputs

| Pipeline | Directory | Key files |
|----------|-----------|-----------|
| Kinematics | `output/ensemble_curves/` | `ensemble_<phase>_<joint>_<signal>.csv` (36 files) |
| MoS discrete | `output/mos_output/` | `mos_all_strides.csv`, `mos_subject_condition.csv` |
| MoS time-series | `output/mos_timeseries/` | `ensemble_mos_<phase>_<ap\|ml>_<raw\|normheight>.csv` |
| QC plots | `output/mos_plots/` | `mos_<subject>_T<nn>.png` |

Each ensemble CSV: rows = subject × condition × side × phase; columns = `pct_0` … `pct_100`.

## Sign conventions

- **AP MoS** (Beerse et al. 2024): positive when XCOM is **anterior** to the stance toe.
- **ML MoS** (Hof 2005–like): positive when XCOM is **medial** to the stance heel.
- Gait cycle: **HS → HS** from `per_stride_data.csv` (`hs_start_frame`, `hs_end_frame` as row indices in the trial CSV).
- Mid-swing: swing-toe AP crosses stance-toe AP.

## Tests

```bash
cd gait_analysis
pytest tests/test_smoke.py -v
```

## Verification (example trials)

```bash
cd gait_analysis && python run_all.py --filter-trials SUBJ01:5,SUBJ01:23,SUBJ01:33,SUBJ01:48 && echo "SUCCESS"
```

## Known data issues

Some trials may have unreliable gap fill or marker quality. Exclude them with `--filter-trials SUBJ:NN` when running batch jobs.

## Extra cohort (added trials)

Gap-filled trials under `corrected/added/extra/` use manifest `corrected/added/extra_obs_trials.csv` and spatiotemporal output `gait_spatiotemporal_out/extra/per_stride_data.csv`. Run from repo root with explicit paths:

```bash
cd gait_analysis/src
python batch_kinematics_ensemble.py \
  --obs-csv ../../corrected/added/extra_obs_trials.csv \
  --ps-csv ../../gait_spatiotemporal_out/extra/per_stride_data.csv \
  --trial-dir ../.. \
  --output-dir ../../output/gait_mos_kinematics_extra/ensemble_curves
```

Repeat for `batch_mos.py`, `batch_mos_timeseries.py`, and `visualize_mos.py` with the same `--obs-csv`, `--ps-csv`, `--trial-dir` and output under `output/gait_mos_kinematics_extra/`. Joint peak CSVs require the sibling [gait-mos-kinematics](https://github.com/gait-mos-kinematics/gait-mos-kinematics) package (`batch-kinematics-peaks`). See the root [README.md](../README.md#downstream-gait-analysis).
