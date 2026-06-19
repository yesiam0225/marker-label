# Portfolio demo assets

Regenerate the README GIF with `marker-label-view` styling.

```bash
# From repo root (after pip install -e ".[qc]")
# Direct export from a local trimmed CSV (do not commit source trials)
PYTHONPATH=src python examples/generate_demo_assets.py \
  --direct --input path/to/your_trimmed.csv \
  --x-clip-max 540 --duration-s 1.01 --hide-obstacles --speed 0.5
```

Output: `docs/assets/qc_viewer_demo.gif`

Optional: `--synthetic` for a procedural clip with no human motion capture.

Local CSV under `examples/demo_data/` is gitignored.
