---
name: cdx-agent-context
description: Use when working in a repo managed by cdx-agent, reading or regenerating .codex_graph context packs, checking staleness, mapping change impact, or choosing which graph artifacts are safe to open at their token cost.
---

# cdx-agent Context & Graph Conventions

Engine-neutral: applies to both Claude Code and Codex sessions launched through cdx-agent.

## The `.codex_graph/` directory

Every cdx-agent-managed repo may carry a `.codex_graph/` folder of generated artifacts. Never edit them by hand; regenerate instead.

| Artifact | What it is | Safe to read whole? |
|---|---|---|
| `context_pack.md` | The human/agent-readable summary: entrypoints, ranked configs, call chains, task-relevant files, risky folders | Yes — always read this first |
| `workspace_context_pack.md` | Multi-repo view: cross-repo import edges, dependency repos, edit policy | Yes |
| `entrypoints.json` | Scored likely entrypoints with `declared_by` provenance | Yes |
| `config_edges.json` | code-file -> config-file reference edges | Usually; check size first |
| `repo_graph.json` | Full node/edge dump (every file's imports, tags, classes) | NO — query it, never dump it |
| `dependency_edges.json`, `dependency_repos.json` | Workspace dependency detection output | Yes |

Check artifact sizes and token estimates before opening: `cdx-agent --context-budget`.

## Reading `context_pack.md`

- The `Generated:` line in the Repo summary tells you the pack's age; a pack older than recent commits or file edits is stale.
- "Likely entrypoints" and "Possible call chain" are the highest-signal sections — chains with `->` arrows are real resolved-import walks, not guesses.
- "Important configs" is ranked (real configs first, vendored/report noise excluded).
- "Task-relevant files" is only task-specific when a task hint was passed; otherwise it is a structural ranking.

## Regenerating

- Full rebuild: `cdx-agent --graph` (run from the repo).
- Task-focused pack: `cdx-agent graph context --repo . --task "your task"`.
- Multi-repo workspace pack: `cdx-agent --workspace-graph`.
- Rebuilds are incremental (scan cache) — cheap to rerun after edits.

## Impact analysis before edits

`cdx-agent graph impact --repo . --files path/to/file.py` returns direct importers, the depth-ranked transitive closure (entrypoint hits flagged), and which code files reference a config. Use it before editing training/eval/controller/policy files, per the working rules.

## Skills and engines

- `cdx-agent skills-list` shows every discovered skill with its roots, engine scoping, and audit severity.
- `cdx-agent skills-audit` reports risky instructions found inside skill directories.
- A skill directory's `agents/<vendor>.yaml` scopes it to one engine (e.g. `agents/openai.yaml` = Codex only); unscoped skills are visible to all engines.

## Engine selection

- Default engine is Claude Code; `cdx-agent --codex` (or `launch --engine codex`) switches back per-invocation.
- `--safe` maps to workspace-write/default permissions; `--full` is the unrestricted mode — prefer `--safe`.
