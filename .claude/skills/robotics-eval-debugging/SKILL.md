---
name: robotics-eval-debugging
description: Debug MuJoCo, robosuite, OpenPI, OpenVLA, and UR5 evaluation or rollout issues with a short, structured audit.
---

# Robotics Eval Debugging

Use this skill when debugging MuJoCo, robosuite, OpenPI, OpenVLA, or UR5 evaluation runs.

Rules:
- Identify the active config, launcher, environment, policy wrapper, controller, and checkpoint before changing code.
- Check control frequency, action horizon, and action scaling assumptions.
- Identify the smallest safe smoke test before proposing long runs.
- Prefer rollout-wrapper or one-episode smoke checks over training.
- Do not launch long training or collection unless explicitly requested.
