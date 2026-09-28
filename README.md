# HOW Vision Final v3.0.0

A local RGB image and video Hands-On-Wheel workbench with inference, review, selective fine-tuning and model lineage.

## Start
```bash
./scripts/setup.sh
./scripts/run.sh
```
Every launch uses a fresh free localhost port and random session query to isolate the browser from older builds.

## Workflow
1. Upload an image or video and select CPU, CUDA, MPS or Auto.
2. Run inference and inspect ordered frames, HOW state, confidence, overlays and CSV output.
3. Review a run using the video-style timeline, reviewed-frame navigation, label controls and mask brushes.
4. Save reviewed cues and select them in Fine-tune.
5. Apply suggested hyperparameters or edit them, then inspect epoch metrics and before/after predictions.
6. Compare timestamped checkpoints in Models.

CUDA or MPS is tested before use. Unsupported or failed accelerators fall back to CPU and the fallback is recorded.
