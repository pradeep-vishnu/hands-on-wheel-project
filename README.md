# HOW Vision v0.2.0

Local-first RGB Hands-On-Wheel performance baseline. It downloads the official MediaPipe Hand Landmarker float16 task model, detects up to two hands, estimates a wheel ellipse, derives contact features, classifies five internal HOW states, applies deterministic temporal smoothing, renders overlays, and writes CSV, JSONL, metrics, logs, configuration and a reproducibility manifest.

The colored hand hull is an approximate landmark region, not pixel-accurate segmentation. The wheel detector is a replaceable geometric baseline. Accuracy is not claimed without reviewed ground truth.

## Install and start

Linux/macOS:
```bash
./scripts/setup.sh
./scripts/run.sh
```

Windows PowerShell:
```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup.ps1
.\scripts\run.ps1
```

Open `http://127.0.0.1:8000`.

`python scripts/check_install.py` installs missing dependencies using the active Python interpreter and downloads the model if absent. Use `--check-only` to avoid installation. Model downloads are cached and staged atomically. Internet is needed for first setup only; runtime is local afterward.

## Test
```bash
python -m pytest
```

## Outputs
Every run creates the next `out_dir/run_####` without overwriting prior work. Outputs include `manifest.json`, `config.yaml`, `predictions.csv`, `predictions.jsonl`, `metrics.json`, `logs/run.log`, `overlays/`, and an output image or `output.mp4`.
