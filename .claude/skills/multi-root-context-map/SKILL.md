---
name: multi-root-context-map
description: Use for multi-root robotics projects, symlinked workspaces, GrapeRoot/Dual-Graph mirrors, or benchmark stacks spread across parallel folders.
---

# Multi-Root Context Map

- Read `WORKSPACE_INDEX.md` first if it exists, then map each symlink path back to its original source path.
- List which folder owns simulator code, controller code, benchmark code, configs, scripts, logs, and assets.
- Treat generated workspace mirrors as derived context, not a place to edit source unless the mirror target is the actual file.
- Before cross-folder edits, summarize the dependency chain across repos or sibling folders.
- Avoid assuming there is one git root or one canonical workspace root.
- Prefer short, explicit path references in the final plan so later edits stay local.
