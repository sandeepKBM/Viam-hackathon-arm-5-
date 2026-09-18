# arm5.planning

Owns everything that turns a (start, goal) pair into a `Trajectory` the
control loop can execute. `base.py` defines the shared `Planner` ABC and
`Trajectory`/`Waypoint` data model; three sibling planners implement it:

- `classical/` -- IK/RRT-style motion planning via the Viam `MotionClient`
  (deterministic, model-based).
- `learned/` -- a `LearnedPolicy` interface (load/act) adapted to the
  `Planner` ABC, for a learned (e.g. imitation/RL) policy.
- `soft_logic/` -- hand-written heuristics and rules that score or veto
  candidate trajectories from the other two planners (the "soft logic").

TODO: decide the arbitration policy in `soft_logic/rules.py` (e.g. does
soft logic filter classical output, learned output, or both?), and how a
top-level orchestrator picks which planner runs when.
