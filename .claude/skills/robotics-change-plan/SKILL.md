---
name: robotics-change-plan
description: Use for robotics code changes, controller edits, simulation tweaks, benchmark updates, or cross-folder refactors that need impact analysis, safety review, validation, and rollback.
---

# Robotics Change Plan

- Start with a short impact analysis: active entrypoint, config path, controller/policy/env chain, checkpoint/data/log paths, and likely files to touch.
- State the intended behavior change, risk level, and any safety implications before proposing edits.
- Include a validation plan with at least one smoke test, one log or metric check, and one rollback command.
- Call out what must not change, especially units, control rate, action scaling, horizon length, and checkpoint selection.
- **Render the configuration and have a human look at it before dispatching a run.** When a
  controller drives ONE scalar axis, the run is worthless if that axis points somewhere the
  plant cannot respond to, and that is invisible in the config, the gains and the logs while
  being obvious in a single picture. Produce the picture, show it, and ask which axis is
  wanted rather than inferring it.
  Two guards on the metric you use to pick that axis:
  * Do not rank an axis by a magnitude alone. A large authority number pointed in a direction
    the plant cannot feel is worth nothing; check WHICH DIRECTION the number describes.
  * Check authority at the STATE THE RUN STARTS FROM, not just in general. An alignment
    metric that is degenerate over a whole subspace will happily rate a useless direction as
    ideal; add the projection evaluated at the initial condition.
- **Preflight a sim experiment: the SAFETY MONITOR and the CONTROLLER must resolve in the
  same frame.** If a controller is configured to drive a rotated or diagonal task axis while
  the guard still measures displacement along the original axes, the guard counts intended
  on-axis travel as lateral drift and fires on the motion the controller was told to produce.
  Check frame agreement before dispatching, not after.
  This failure is invisible in every artifact you would normally read. It does not show up as
  a guard trip in the winning run — the search simply retreats to whatever gain stays under
  the mis-measured limit, and reports a converged result with no guard fired. What it looks
  like from outside: a searched gain far weaker than a known-good value, an unrelated bound
  that turns out to be completely inert when swept, and a run that stops improving well short
  of the target.
  Two habits that catch it:
  * Sweep a suspected limiting parameter and confirm it actually MOVES the outcome. A
    parameter whose 20x change alters nothing is not the constraint, however plausible.
  * When a mechanism exists to align the frames, assert it took effect rather than assuming
    the call worked — ordering constraints ("call before reset") make a correctly-written
    call a silent no-op in the wrong position.
- **A gain belongs to the case it was derived for. Never inherit one across a changed case.**
  A gain is valid only for the exact combination it was fitted at — controller, task frame,
  pose, which rows are tracked, and the ROLE the gain plays. Change any of those and the
  number is stale. When deriving a new config from an old one, treat every gain you did not
  re-derive as a bug until you can say where it came from and why it still applies.
  Watch for the two forms this takes, because both are silent — the config parses, the run
  completes, and the value is simply wrong:
  * *Wrong frame or pose*: the gain was scaled by an inertia/Jacobian quantity measured
    somewhere else, so it now describes a different physical axis.
  * *Wrong role*: a gain that was a soft centering/bias term gets reused as a tracking gain
    (or vice versa) because gains are indexed positionally by axis. Re-check how the
    controller indexes its gain tuple before moving an axis between task and barrier roles.
    A conversion rule derived for tracked axes does not apply to barrier axes either.
- **Test the setting production uses, not the convenient default.** A feature that exists to
  change behavior under load, scale, or parallelism must be tested under that condition. A
  suite that only ever exercises the single-worker, small-input, feature-off path will stay
  green while the path that matters is broken — and the breakage can invert the feature's whole
  purpose rather than merely degrade it.
- **Verify the EFFECT, not the invocation.** The recurring failure in this repo is the silent
  no-op: the operation appears to succeed. A parser that ignores a key, an edit whose pattern
  did not match, a call placed after the step that consumes it, a flag that parses and is then
  discarded, a staging command that aborts and leaves an empty commit. None of these raise, and
  several reproduce the pre-fix numbers exactly — indistinguishable from "the fix did not help".
  So check downstream state rather than the return code: re-read through the real parser, assert
  the consumer actually received the value, inspect what the commit contains. Exit codes, hashes
  and green tests are evidence that something ran, not that the intended thing happened.
- **Verify a config edit by re-parsing it, not by reading it.** Load the file through its own
  config parser and assert each field you meant to set. Two independent silent-no-op modes
  are common: an indentation or pattern mismatch so the edit never lands, and a parser that
  never reads the key at all. Both leave a file that looks right and behaves as the default.
- If hardware-facing code is involved, stop at planning unless the user explicitly asks for execution.
- If the change spans multiple folders, map the ownership of each folder before editing.
