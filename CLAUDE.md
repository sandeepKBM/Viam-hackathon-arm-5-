# CLAUDE.md — Viam Hackathon Arm 5

Guidance for Claude/Codex agents working in this repo. Read this first.

## What this is

A Python project for a **Viam hackathon** controlling a robot **arm** through the
[Viam Python SDK](https://python.viam.dev/) (`viam-sdk`). The goal is a clean,
modular stack — perception → planning → control → hardware — that we can iterate
on fast during the hackathon and keep readable afterward.

- Language: Python 3.12
- Robot framework: Viam (`viam-sdk==0.80.0`)
- Hardware: **UFactory xArm 5** (5-DOF), via the Viam `Arm` component. DOF is
  known (5); per-joint limits / reach / payload are datasheet-TODO — see
  `arm5/controls/safety.py`. Do not guess those numbers.
- Remote: `origin` → https://github.com/sandeepKBM/Viam-hackathon-arm-5-

## Environment

Use the project-local venv (do **not** use the global conda env for this repo):

```bash
python3 -m venv .venv            # already created
source .venv/bin/activate
pip install -r requirements.txt  # or: pip install viam-sdk
```

Connection credentials come from **environment variables**, never hard-coded:
`ROBOT_ADDRESS`, `ROBOT_API_KEY`, `ROBOT_API_KEY_ID`. A non-secret template lives
at `config/robot.example.json`; copy it to `config/robot.json` (gitignored).

Smoke test the connection before anything else:

```bash
python scripts/connect_smoke_test.py
```

## Layout

```
arm5/                       # main importable package
  vision/                   # perception: cameras, detectors, pose estimation
  hardware/                 # Viam component wrappers: robot client, arm, gripper
  controls/                 # control loops + safety (limits, e-stop, workspace bounds)
  planning/                 # motion & task planning
    base.py                 # Planner ABC, Trajectory/Waypoint types
    classical/              # Viam motion service, IK/RRT, geometric planners
    learned/                # learned policies (framework-agnostic interface)
    soft_logic/             # hand-written rules / heuristics / arbitration we author
config/                     # connection templates (real robot.json is gitignored)
scripts/                    # runnable entrypoints (smoke tests, demos)
tests/                      # import + unit tests
reference/rdk/              # read-only clone of viamrobotics/rdk (gitignored)
.claude/skills/             # robotics skills (see below)
```

### The planning split
- **classical/** — deterministic, model-based: Viam `MotionClient`, inverse
  kinematics, RRT/geometric planning. Reproducible and the default.
- **learned/** — data-driven policies behind a stable `LearnedPolicy` interface
  (`load`, `act(obs) -> action`); framework-agnostic so we can swap backends.
- **soft_logic/** — the glue we write by hand: soft constraints, behavior rules,
  and arbitration that biases or vetoes what the classical/learned planners
  propose. This is where hackathon-specific heuristics go.

## Working rules (robotics)

- **Impact analysis before edits** to controls / planning / hardware — state what
  moves and why before changing it.
- **Never silently change** control rate, action scaling, units, horizon length,
  or joint/workspace limits. Call these out explicitly in the change.
- When editing a **controller**, state the control-rate and action-scaling
  assumptions in the docstring/PR.
- **Safety first on real hardware**: respect `arm5/controls/safety.py` limits; keep
  an e-stop path. Prefer tiny, slow, bounded motions before full runs.
- **No invented peripherals** — use only the arm, gripper, and camera we actually
  have, plus standard Viam services. Ask before adding a sensor/peripheral.
- Prefer a **config/wrapper/adapter fix** over editing `reference/rdk` (it is
  read-only reference; never commit or modify it).
- Keep experiment notes in `docs/status/` or `reports/`, not scattered prose.

## Skills available (`.claude/skills/`)

Robotics skills copied in for this repo — invoke the relevant one:

- `repo-impact-analysis` — blast radius before multi-file edits
- `robotics-change-plan` — plan + safety review + validation + rollback for changes
- `robotics-eval-debugging` — debug rollout/eval issues
- `robotics-runtime-reproducibility` — imports, assets, CUDA/env, machine issues
- `checkpoint-selection` — pick checkpoints conservatively
- `openpi-training-diagnosis` — OpenPI fine-tune / rollout diagnosis
- `experiment-evidence-pack` — evaluate policies, videos, evidence for results
- `multi-root-context-map` — multi-root / symlinked workspaces
- `cdx-agent-context` — read/regenerate `.codex_graph` context packs
- `token-efficient-debugging` — large logs / noisy tests / broad exploration

## Repo graph (cdx-agent)

A `.codex_graph/` context pack is generated for this repo (gitignored). Read
`.codex_graph/context_pack.md` first for entrypoints, configs, and call chains.

**Important:** the vendored `reference/rdk` clone is NOT a scan target — it would
drown the pack in rdk's own entrypoints/configs. `cdx-agent` has no repo-local
ignore for the single-repo graph, so regenerate with `reference` excluded:

```bash
# clean pack (arm5 only) — preferred:
PYTHONPATH=/common/users/ss5772/codex_tools/repo_graph_agent \
  python3 -c "from repo_graph_agent.cli import build_graph; \
  build_graph('$(pwd)', skip_dirs={'reference'})"

# impact analysis before multi-file edits:
cdx-agent graph impact --repo . --files arm5/controls/safety.py
```

Plain `cdx-agent --graph` also works but will re-include `reference/rdk`; if you
run it, regenerate with the command above (or move `reference/` out of the tree).

## Reference

`reference/rdk/` is a shallow clone of https://github.com/viamrobotics/rdk (the Go
robot dev kit) kept for API/behavior reference. It is **gitignored** — re-clone
locally if missing:

```bash
git clone --depth 1 https://github.com/viamrobotics/rdk.git reference/rdk
```
