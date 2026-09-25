# Hands On Wheel Project v0.3.0

Local RGB inference, visual review, polygon correction and reward-weighted classifier improvement.

## Start
`./scripts/setup.sh && ./scripts/run.sh`, then open `http://127.0.0.1:8000`.

The setup script installs missing dependencies and downloads the official MediaPipe Hand Landmarker float16 model. Runtime is local after setup.

## Workflow
1. RUN: upload an image/video and execute inference.
2. REVIEW: choose a run and frame, override `BOTH_ON`, `LEFT_ON`, `RIGHT_ON`, `NONE_ON`, or `UNKNOWN`; optionally redraw hand/wheel polygons; assign reward; confirm.
3. LEARN: train a small PyTorch policy classifier from reviewed perception/contact features. The loss combines supervised cross entropy, reward-weighted log policy, and entropy regularization.

This is human-feedback learning over the HOW classifier, not end-to-end reinforcement learning of MediaPipe. Polygon corrections are stored as ground truth for a future trainable segmentation layer. Original predictions are immutable. Models are versioned under `models/versions/how_####` and are not activated automatically. Do not claim improvement without evaluation on held-out reviewed data.
