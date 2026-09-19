---
name: experiment-evidence-pack
description: Use when running experiments, evaluating policies, generating videos, comparing behavior, or preparing evidence for robotics results or lab testing.
---

# Experiment Evidence Pack

- Capture the exact command, config file, seed, checkpoint or policy path, environment name, and mode (simulation or hardware).
- Record the log path, output directory, and video path if a render or rollout is involved.
- Include a concise before/after metrics summary and any failure cases or abort conditions.
- Prefer timestamped output folders so results can be reproduced and compared later.
- Note what was not validated when the run is only a smoke test.
- Keep evidence focused: enough to prove what changed and what happened, not a dump of every artifact.

## Baselines & comparisons
- Benchmark HEAD-TO-HEAD only against methods in the SAME problem class as yours (same
  centralization, cooperation, constraint, and objective). A method solving a different problem
  (e.g. decentralized/adversarial when yours is centralized/cooperative) is RELATED-WORK
  positioning, not an experimental baseline — say so explicitly rather than comparing raw numbers.
- Tag every external number with provenance: `[measured in-repo]`, `[verified vs PDF, date]`, or
  `[positioned/unverified]`. Never let a positioned paper number read as a measured head-to-head.
- State the axis each baseline stresses (collision / constraint / cooperation / smoothness) so the
  comparison maps to specific claims, not a single aggregate score.

## Training runs (data prep vs GPU)
- Precompute and prep all data on CPU cores (westeros) into ready-to-load shards FIRST, then ship to
  the GPU box (ilab) for training — keeps the GPU compute-bound, not IO-bound. Do not prep/augment
  inside the GPU training loop. Long GPU jobs are owner-run background jobs, not run-and-wait subagents.
