---
name: robotics-runtime-reproducibility
description: Use when debugging imports, MuJoCo assets, checkpoints, CUDA/PyTorch mismatch, env issues, script paths, or machine-specific runtime problems in robotics repos.
---

# Robotics Runtime Reproducibility

- Identify the active shell, conda or venv environment, Python path, and repository root before changing anything.
- Locate required external folders, asset roots, XML/MJCF paths, checkpoint paths, and log directories explicitly.
- Prefer config variables and repo-local paths over ad hoc hardcoded machine-specific fixes.
- Separate runtime/environment fixes from algorithm or controller changes.
- Produce reproducible setup or launch commands and name the exact environment that was used.
- If a path assumption changes, state it clearly instead of silently patching around it.
