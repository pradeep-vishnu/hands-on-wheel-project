# Hands On Wheel Project v0.5.0

Full retained workflow: input preview, image/video inference, telemetry, overlays, CSV/JSONL/metrics/manifest, Review Studio, Magic Select correction, immutable annotations, reward-weighted PyTorch fine-tuning, timestamped checkpoints, model inspection, explicit default activation, and checkpoint-backed inference.

## Start
```bash
./scripts/setup.sh
./scripts/run.sh
```
The browser opens automatically. Set `app.open_browser: false` for headless environments.

## Workflow
1. Upload. The latest upload is selected and previewed automatically; manual input selection is respected.
2. Select the baseline or a timestamped checkpoint and run inference.
3. Inspect FPS, CPU, memory, process RSS, live output, CSV, JSONL, metrics and manifest.
4. Review frames, correct five-state labels, optionally replace hand/wheel evidence with Magic Select, and confirm feedback strength.
5. Fine-tune the five-feature HOW decision policy using supervised plus reward-weighted policy loss and entropy regularization.
6. Inspect the timestamped checkpoint and explicitly set it as default or select it for one run.

Magic Select is color-region assistance, not semantic segmentation. MediaPipe remains the hand detector; fine-tuning trains the HOW decision classifier. Never claim improvement without held-out evaluation.
