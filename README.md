# HOW Vision 1.1.0

Complete local-first RGB Hands-On-Wheel workbench.

## Run
```bash
./scripts/setup.sh
./scripts/run.sh
```

## Architecture
- `howvision/inference/types.py`: project-owned frame, detection, feature, prediction and result contracts.
- `io.py`: ordered image/video decoding with canonical IDs and native timestamps.
- `adapters.py`: replaceable MediaPipe hand, ellipse wheel, rule and PyTorch policy adapters.
- `temporal.py`: independently tested confirmation, smoothing, uncertainty and reset.
- `render.py`: standalone overlay renderer.
- `engine.py`: atomic run allocation, stable CSV, rich JSONL, resolved configuration, metrics, logs, overlays, media and manifest.
- `api/`, `frontend/`, and `training/`: review, selected-cue training, epoch comparisons, hyperparameters and model registry.

## Notes
- `UNKNOWN` remains reachable for insufficient evidence.
- The wheel detector is only a replaceable geometric baseline.
- Baseline inference uses rules. Fine-tuned checkpoints use the compact PyTorch policy.
- GPU utilization is `null` unless a supported telemetry adapter supplies it.
