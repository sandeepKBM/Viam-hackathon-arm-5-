---
name: repo-impact-analysis
description: Map the blast radius before multi-file edits using repo graph context, entrypoints, configs, and smoke-test planning.
---

# Repo Impact Analysis

Use this skill before changing multiple files or any core runtime/training path.

Rules:
- List impacted modules and likely call chain before editing.
- Use `.codex_graph/context_pack.md` when available.
- Identify config files, entrypoints, tests, and smoke checks.
- State rollback steps before making risky edits.
- Flag policy, controller, env, runtime, and training changes explicitly.
