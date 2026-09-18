---
name: token-efficient-debugging
description: Use when debugging huge logs, noisy tests, broad repo exploration, high token usage, or expensive agent sessions. Helps the agent summarize logs, use targeted searches, avoid dumping generated artifacts, and preserve errors/tracebacks.
---

# Token-Efficient Debugging

Trigger:
- huge logs
- failed tests with noisy output
- broad repo exploration
- expensive agent session (any engine)
- explicit token-usage concern

Rules:
1. Start from `.codex_graph/context_pack.md`.
2. Use targeted searches.
3. Summarize large logs before reading them in full (`cdx-agent --summarize-log <file>`).
4. Prefer `git diff --stat` before full `git diff`.
5. Use raw output only when the summary is insufficient.
6. Rerun failing commands without compression if a subtle failure is possible.
7. Explicitly state what was compressed or summarized.
8. Check `cdx-agent --context-budget` for artifact sizes before opening large generated files.
