# Marker labeling versions (branches)

## Branches

| Branch | Description |
|--------|-------------|
| **labeling/rule-based-v1** | Saved version: rule-based head labeling (d_back projection for A/P, facing rotation applied to d_back/d_right). Use this when you want the current pipeline with consistent head assignment across trials. |
| **labeling/scratch-v2** | Clean base from original `main`: no rule-based head changes. Use this to try a different labeling approach from scratch. |
| **main** | Primary branch (unchanged; rule-based work is on `labeling/rule-based-v1`). |

## How to switch

```bash
# Use the saved rule-based version
git checkout labeling/rule-based-v1

# Work on a new version from scratch (original code)
git checkout labeling/scratch-v2
```

## Saving the scratch version later

When you have a new approach on `labeling/scratch-v2` that you want to keep:

```bash
git checkout labeling/scratch-v2
git add ...
git commit -m "Describe your new labeling approach"
```

You can create a tag or another branch from that commit to name the version (e.g. `labeling/v2-experiment`).
