# Runbook

## Train a new model
1. Change `configs/train.yaml`.
2. Run `make pipeline`. A failed gate exits non-zero and does not write a model.
3. Compare runs with `make experiments`.
4. Open a pull request with the new `reports/evaluation.json`.

## Troubleshooting
| Symptom | Likely cause | Action |
|---------|--------------|--------|
| `/readyz` returns 503 | Model missing from the image | Confirm `models/sonar_unet.pt` is committed |
| Gate fails after a simulator change | The data got harder | Review the per-class IoU and retrain |
| Demo looks wrong on uploaded images | Real data differs from the simulator | Expected. See limits in `docs/requirements.md` |

## Roll back
Redeploy the previous Render deploy, or the previous `vX.Y.Z` image.
