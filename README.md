# Hands On Wheel Project v0.9.0

Review Studio stabilization plus retained inference, selective fine-tuning and model management.

## Start
```bash
./scripts/setup.sh
./scripts/run.sh
```

## Review interaction
- Unsaved prompt appears only when leaving the current frame or cue through the frame slider, reviewed-frame jump controls, or cue selection.
- Prompt offers Undo last, Discard and continue, and Save and continue.
- Ctrl+Z and Cmd+Z undo the latest label or mask edit.
- Previous/next Reviewed controls jump only among manually saved frames.
- Mouse wheel no longer changes zoom. Zoom uses explicit buttons.
- Move is a toggle. While active, right-click drag pans the frame. Turning Move off resumes the selected paint tool.
- Top-right telemetry HUD reports active/setup state, CPU, RAM, GPU backend and local date/time.
