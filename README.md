# HOW Vision v2.0.0

Complete local HOW workbench with inference, review, selective fine-tuning, model lineage and per-session random local ports.

## Run
```bash
./scripts/setup.sh
./scripts/run.sh
```
Each launch binds to an operating-system assigned free localhost port and opens a URL containing a random session token. This avoids reusing stale browser origins from earlier versions.

## Included
- Controlled image/video upload and preview.
- CPU, CUDA and Apple MPS discovery with tested CPU fallback.
- Inference run manifests record requested and actual device.
- Review Studio with label correction, mask points, undo and save guard.
- Fine-tuning cue selection, suggested hyperparameters and editable descriptions.
- Epoch loss, validation accuracy, accuracy delta and before/after example predictions.
- Timestamped PyTorch checkpoints and model evolution metadata.
- Models dashboard with accuracy, device, parameter count and checkpoint size.

OpenCV geometry remains CPU based. Accelerator selection applies to the PyTorch policy training and checkpoint inference seam. CPU fallback is explicit in job metadata.
