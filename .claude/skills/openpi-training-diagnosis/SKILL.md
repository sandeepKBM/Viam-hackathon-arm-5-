---
name: openpi-training-diagnosis
description: Diagnose OpenPI fine-tuning or rollout problems with focus on data, checkpoint, chunking, control rate, and train/eval mismatch.
---

# OpenPI Training Diagnosis

Use this skill for OpenPI fine-tuning, rollout diagnosis, or adapter issues.

Rules:
- Identify dataset path, checkpoint path, and active config first.
- Check batch size, action horizon, action chunking, and control frequency.
- Verify first-7 action handling, normalization stats, and any train/eval mismatch.
- Check logging, checkpoint save paths, and resume paths before running anything long.
- Prefer one-batch or one-step diagnostics before real training.
