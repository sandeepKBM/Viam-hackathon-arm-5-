# Viam Hackathon — Arm 5

A modular Python stack for controlling a robot **arm** with the
[Viam Python SDK](https://python.viam.dev/).

Perception → Planning → Control → Hardware, kept as clean, swappable modules.

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# set connection creds (see config/robot.example.json)
export ROBOT_ADDRESS=...        # e.g. my-arm.abcd.viam.cloud
export ROBOT_API_KEY=...
export ROBOT_API_KEY_ID=...

python scripts/connect_smoke_test.py
```

## Modules

| Module | Owns |
| --- | --- |
| `arm5/vision` | cameras, detectors, pose/object estimation |
| `arm5/hardware` | Viam component wrappers: robot client, arm, gripper |
| `arm5/controls` | control loops + safety (limits, e-stop, workspace bounds) |
| `arm5/planning/classical` | Viam motion service, IK, RRT/geometric planners |
| `arm5/planning/learned` | learned policies behind a stable interface |
| `arm5/planning/soft_logic` | hand-written rules / heuristics / arbitration |

See [`CLAUDE.md`](./CLAUDE.md) for the full layout, working rules, and available skills.
