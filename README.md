# Hands On Wheel Project v0.7.0

Full local workflow with image/video inference, performance telemetry, Review Studio, selective fine-tuning, timestamped checkpoints, model activation, CSV/JSONL/metrics/manifests, and automatic browser launch.

## Start
```bash
./scripts/setup.sh
./scripts/run.sh
```

## v0.7 interaction changes
- Removed the Waiting for inference overlay.
- HOW status panel changes green for `HOW_ON`, yellow for `UNKNOWN`, and red for `HOW_OFF`.
- Fine-tune cue cards can be selected or deselected. Deselected frames are dimmed and excluded by the backend training filter.
- Review feedback control is named **Training influence** and explains that it changes the frame's contribution to training loss.
- Review Studio shows input-run thumbnails on the left. Selecting a thumbnail changes the cue immediately.
- Multi-frame/video cues receive an integrated timeline scrubber under the main preview with frame timestamps.
- Label or mask edits open a compact save prompt asking whether to include the frame in fine-tuning.
- Circular hand/wheel brushes, eraser, undo, and immutable original outputs remain.

Fine-tuning updates the compact HOW decision policy, not MediaPipe or a segmentation network. User brush strokes are preserved for future segmentation work.
