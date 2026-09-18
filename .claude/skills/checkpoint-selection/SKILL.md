---
name: checkpoint-selection
description: Select the best checkpoint conservatively by reading metrics, logs, and artifact structure without overwriting anything.
---

# Checkpoint Selection

Use this skill when picking a checkpoint for evaluation, warm start, or deployment.

Rules:
- Find all candidate checkpoint directories first.
- Read available metrics, validation logs, and rollout summaries.
- Never overwrite or mutate checkpoint directories.
- Prefer exact evaluation commands over vague recommendations.
- If evidence is weak, state that clearly and rank candidates conservatively.
