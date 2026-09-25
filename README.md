# Hands On Wheel Project v0.6.0

This release retains inference, review, fine-tuning, models, telemetry, output artifacts and automatic browser startup while improving image presentation and mask correction.

## Start
```bash
./scripts/setup.sh
./scripts/run.sh
```

## Changes in v0.6
- Text is no longer burned into inference or review images. HOW status, confidence, checkpoint, frame and performance information are displayed in dedicated UI panels.
- Review Studio uses a fixed 18-pixel circular brush with hand, wheel and eraser modes. Strokes are stored as normalized points and remain separate from original predictions.
- Selecting a run in the Review cue dropdown loads it immediately. There is no Open cue button.
- Fine-tune shows reviewed source-image thumbnails with the user-painted hand and wheel masks.
- The Inference input card has an explicit upload button beneath the preview.
- Raw source frames and rendered evidence overlays are stored separately.
- The newest upload remains selected automatically; manual dropdown selection remains supported.
- Fine-tuning remains real PyTorch reward-weighted policy training. Checkpoints remain timestamped and optional.

The landmark hull and wheel ellipse are baseline evidence, not pixel-accurate segmentation. User brush masks are annotations for future segmentation work; current fine-tuning updates the HOW decision classifier.
