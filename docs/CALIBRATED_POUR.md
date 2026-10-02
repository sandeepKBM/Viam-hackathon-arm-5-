# Calibrated bottle → cup pour

**Status (2026-09-19): implemented and tested offline only. Not validated on hardware.**
Every hardware stage (B–G below) is **unrun**. The committed
`config/calibration/pour_setup.json` is an uncalibrated template, so every pour
path fails closed until the calibration procedure is completed on armfarm5.

## 1. Diagnosis

Evidence sources: the live Viam config of armfarm5 (read 2026-09-19), the
overlay your last run saved (`out/sam_regions.png`), and the code at `c8b7d04`.

| Observed failure | Evidence | Root cause | Fix | Test |
|---|---|---|---|---|
| "the arm was above the actual bottle when it was trying to pick it up" | `pour_can.py` side grasp: `pick_z = MIN_Z + 1.5 in` = 217.9 mm, commanded as the **flange** z with a horizontal tool | `MIN_Z` (179.76) is the flange floor for a **downward** tool. It already contains the ~170 mm flange→fingertip length. With the tool horizontal, the pads sit at the flange's own height, ~210 mm above the table, above a 150–200 mm bottle. The flange floor then also silently clamped any lower Z. | Side-grasp height is measured from the calibrated **table plane** for the TCP (pad centre). Flange = TCP ∘ T_flange_tcp⁻¹. The whole envelope is checked against the table, with no silent clamp. | `test_arm_was_above_actual_bottle`, `GraspTests` |
| (same) + pads past the bottle | `POUR_STICKOUT_MM=90` default | A guess with no measurement behind it. The real flange→pad distance is ~170 mm, so the pads ended ~80 mm past the bottle axis and the palm inside it. | Measured TCP (pivot calibration) in `pour_setup.json`. No default. | `test_flange_standoff_comes_from_calibrated_tcp` |
| (same) + wrong object | `out/sam_regions.png`: "bottle" = a lying pump bottle with a rectangular box-fallback mask; the upright FUN TIME bottle was not detected; the script picked the **can** (`pool = cans or bottles`), which was lying down | Source selection preferred cans; a box-fallback mask was accepted; nothing checked that the source was upright | Exactly one validated **upright** source of the requested kind (or explicit pixel selection). Box masks, lying objects and truncated masks are rejected. The bottle+can/no-cup fallback is deleted. | `SceneSelectionTests`, `test_box_shaped_mask_rejected`, `test_lying_bottle_rejected` |
| Localization error in general | Live config: `cam` has **no `align_color_depth`** (module default `false`). The code indexed raw depth-imager pixels with color-image SAM masks and color intrinsics. | The D435 depth imager has a wider FOV and a ~15 mm baseline. Mask pixels read depth from a different spot (tens of mm at the image edges), often the table. | `RGBDFrame` carries alignment state; calibration must verify it (`alignment` step). Per-object edge-registration gate. | `test_deliberately_misregistered_depth_rejected`, `test_baseline_only_misregistration_rejected` |
| (same) | `region_interior`: `if z <= 0: z = table_depth` (whole-frame median) | Objects without depth were placed at table depth along their ray, so tall objects shift outward by h/H × lateral offset | Deleted for bottle/can/cup, in both the new estimators and the sorter's `shapes.py`. Missing depth means unlocalizable and no motion. | `test_zero_depth_under_mask_rejected_no_fallback`, `test_sorter_path_skips_pourables_without_depth` |
| "the cup finding with the correct offset for the camera needs to be implemented" | Cup = mask centroid (or box centre) at the **mode** depth. The camera mount is the shared fragment `xarm850-realsense-original` (83, −14, 18) mm / −97.7°, identical on every armfarm and never calibrated for this cell. | Wrong target point (centroid, not opening) plus an uncalibrated mount | Rim circle + plane fit from the top ring of 3D points, with an interior-below-rim check. Hand-eye calibration of the mount, which the Viam frame system must carry. | `test_cup_rim_center_radius_plane`, `test_cup_box_center_regression`, `test_cup_camera_offset_comes_from_calibration_not_magic_constant` |
| "the action of pouring never succeeded" | The run never got past the grasp. `grab` closed on air, so `holding=False` raised "gripper did not grab". Even if it had grasped, the tilt rotated about the flange with the flange 89 mm from the cup on the wrong side (`tool_x=+1` regardless of `x_sign`), putting the bottle ~240 mm from the cup. | Grasp failure upstream; the pour geometry steered the flange, not the mouth | Mouth pivot path over the rim interior, with the height at each tilt from the clearance constraints. The TCP is compensated around the mouth. | `PourPathTests`, `test_pouring_reaches_done_only_after_full_trajectory` |
| "the pour action is not there at all I guess" | `voice.TASKS` has no pour; the orchestrator skill registry has no pour; `pour_can.py` is standalone | Never wired | One `run_task("pour")` branch and one `pour` skill. Both call `components.pouring.run_pour_request` → `PourController`. | `test_pour_action_is_dispatchable`, `DispatchTests` |

Other confirmations requested:

- **Eye-in-hand.** The live `cam` frame parent is `arm`, i.e. the wrist. The mount type is declared in the calibration file, never inferred.
- **Frame chain.** `pixel → (color intrinsics, aligned depth) → cam → arm end-effector (flange) → world`. `world` = arm base, per the identity `arm` frame. The Viam frame system does cam→world. The code reads `T_world_cam` once per capture via `transform_pose(cam origin → world)` and applies it to every point, so no transform is duplicated.
- **Flange vs TCP.** Viam poses are the **flange**: `arm.get_end_position` and `move_to_position` carry no TCP offset. Evidence: the taught floor of 179.76 mm with the fingertips on the table, and home z of 532.6 mm with the camera ~500 mm above the table. The Viam `gripper` frame (+105 mm) is the gripper origin, not the pad centre, and is not used as the TCP.
- **Transform direction.** Points go cam→world with `T_world_cam`. `test_camera_to_world_direction` catches an inverted transform. With a straight-down camera, T and T⁻¹ nearly coincide (a ~180° rotation is its own inverse), so that test uses a tilted view.
- **Profiles, dims, units, timestamps.** The live config sets 1280×720 color and depth, big-endian `vnd.viam.dep` in mm, one `captured_at` per `GetImages` set. `validate_frames` checks equal dims, intrinsics for that size, the distortion model, units, stream identity, arm stationary (joints before/after), and timestamp span.
- **The 90 mm stickout and the 3.5 in pour offset are guesses.** They were env defaults with no measurement record. Both are removed; the replacements are the measured TCP and the rim geometry.
- **Why the pour was standalone.** The commit message says so ("keep it off the voice/orchestrator path so [the offsets] can be tuned").
- **Module frame origin.** The current `viam:camera:realsense` README says the camera frame origin is the **color** sensor and `extrinsic_parameters` is zero. 0.22.4 still reported the ~15 mm baseline, and armfarm5 runs 0.22.5. This is recorded at calibration time and checked live: a change fails closed.

## 2. Frames and conventions

- Poses are 4×4 `T_a_b` (maps b-coordinates into a), in mm. Viam orientation vectors: `R = Rz(lon)·Ry(lat)·Rz(θ)`; round-tripped exactly in `components/transforms.py`.
- **TCP** = gripper pad centre. +z is the approach, out of the flange, and `gripper.closing_axis_tcp` is the direction the fingers close. `T_flange_tcp` is measured by pivot calibration.
- **Bottle frame B**: origin at the mouth (top centre) and +z along the body axis. `T_tcp_B` is fixed at grasp. Slip is checked by the jaw width after lift.
- **Table**: a plane from the ChArUco board pose, minus the board thickness. Every height is measured from it.
- **Side grasp**: the tool is horizontal, and the held object's axis runs along the remaining tool axis. The planner prefers the camera on top.

## 3. Calibration (hard prerequisite)

Schema: `config/calibration/pour_setup.json`. It holds:
- **Camera:** identity (serial), color/depth profiles, intrinsics, distortion, depth scale, reported extrinsics, and alignment method and metrics.
- **Mount:** mount type, `T_flange_cam`, and world = base.
- **Gripper:** `T_flange_tcp`, pad/finger geometry, jaw mapping, collision envelope, and the bottle-frame definition.
- **Scene:** table plane and the static Viam obstacles.
- **Calibration record:** method, date, sample count, per-sample reprojection and 3D error, residuals, held-out stats, touch-off validation, the Viam frame snippet, and a setup hash.

Fail-closed rules (`components/calibration.py`):
- **Missing or edited file:** refused. The status must be `calibrated`, and the file must be unedited (the hash covers the whole file).
- **Stale:** refused beyond `max_age_days`, which defaults to 7.
- **Undeclared mount:** refused. The mount type must be declared.
- **Unmeasured geometry:** refused. There are no gripper defaults.
- **Unverified alignment:** refused unless alignment has been verified.
- **Too few samples:** refused below 12 hand-eye samples or without held-out validation.
- **Touch-off:** needs at least 9 points within **p95 XY ≤ 10 mm, max XY ≤ 15 mm, p95 Z ≤ 10 mm**. The limits live in code; a file carrying different limits is rejected.
- **Live hardware mismatch:** refused if the camera size, intrinsics (±0.5 px), distortion, reported extrinsics or depth encoding differ, or if the Viam frame system's cam frame differs from `T_flange_cam` by more than 1 mm / 0.3°.

### Viam config changes required (per machine, not in the shared fragment)

The `cam` component lives in the shared fragment used by every armfarm, so apply the changes as `fragment_mods` on armfarm5 only. `hand-eye-solve` prints the exact snippet:

```json
"fragment_mods": [{"fragment_id": "fd2be28c-71e5-4b8a-90a8-a514dbe75ca7", "mods": [
  {"$set": {"components.cam.attributes.align_color_depth": true}},
  {"$set": {"components.cam.frame": {"parent": "arm", "translation": {...}, "orientation": {"type": "ov_degrees", "value": {...}}}}}
]}]
```

### Operator steps (`scripts/calibrate_pour_setup.py`)

Live steps need `VIAM_ALLOW_LIVE=1`. The operator positions the arm by jogging or in teach mode; nothing here moves the arm on its own.

1. `init --mount eye_in_hand`: the mount type is declared, never inferred.
2. Set `align_color_depth: true` (see above), then run `camera`. This records intrinsics, distortion, profiles and reported extrinsics.
3. `alignment --tag-mm 40`: place one ArUco tag on top of a block at least 50 mm tall. The depth under the tag must match the color-PnP plane: median ≤ 3 mm, MAD ≤ 3 mm.
4. `hand-eye-capture --square-mm S --marker-mm M --n 16`: tape a printed ChArUco board flat on the table and measure S and M. Capture at least 12 diverse poses (15–40° tilts, varied yaw).
5. `hand-eye-solve --samples <dir> --board-thickness-mm T`. This uses Park–Martin plus a 3D refinement, with one sample in four held out. It writes `T_flange_cam`, per-sample errors, the table plane and the Viam frame snippet. **Apply the snippet.**
6. `tcp-capture --n 5`: pivot calibration. Touch the pad centre to a fixed point in 5 orientations; the RMS must be ≤ 1 mm.
7. `gripper --closing-axis y --pad-length-mm … --camera-radius-mm …`: enter caliper measurements.
8. `jaw-capture --widths 20,40,60,80`: close on gauge blocks for the width↔position map.
9. `touchoff --n 9 --tag-mm 30`: **stage B**. Vision predicts a tag centre through the live frame system; you touch it with the TCP in teach mode.
10. `finalize`: writes `pour_setup.json` only if every check passes, and records stage B.
11. `objects --trials 27`: **stage C**. 9 positions × 3 arrangements, touch-off on the bottle mouth and the cup rim centre.

`status` prints what is still missing and the stage table at any time.

## 4. The pour primitive (`components/pouring.py`)

```
IDLE → OBSERVE → VALIDATE_CALIBRATION → LOCALIZE_SOURCE_AND_CUP → PLAN_GRASP → PREGRASP → GRASP
→ VERIFY_GRASP → LIFT → REOBSERVE_CUP → PLAN_POUR → PREPOUR → TILT_INCREMENTS → HOLD → UNTILT
→ RETREAT → SAFE_PLACE → DONE          (any state → RECOVER → ABORTED)
```

**Perception**
- **Moondream + SAM propose masks only.** Geometry decides: an upright source (axis from the top surface, radius from the silhouette limbs, height from the table) and a cup opening (rim circle/plane, interior below the rim).
- **Every estimate carries its evidence:** error bounds, quality flags and a rejection reason.
- **Depth is robust across 3–5 frames:** eroded masks, depth-edge exclusion, temporal consistency, valid ratio, connected component, and background/registration checks.

**Grasp**
- A horizontal approach is searched around the source.
- Each candidate is checked for straddle width, finger reach, table and object clearance, TCP in the taught polygon, flange reach, and keep-out around the base.
- Verify: `holding` must be true and the jaw width must match the diameter within 12 mm. After lift, the jaw width must not change (slip check).

**Re-observation**
- After lifting, the wrist camera re-fits the cup from a view where the rigidly held bottle doesn't block the rim.
- The mask comes from depth inside the prior cup's 3D cylinder, with no new detector or VLM call.
- It must agree with the first fit within 10 mm (XY), 6 mm (radius) and 10 mm (Z), or there is no pour.

**Pour path**
- The mouth stays at a fixed point over the rim interior on the source side.
- The bottle rotates about a horizontal axis through the mouth, and the TCP is compensated.
- The mouth height at each tilt angle is the lowest that keeps the bottle and gripper ≥ 15 mm clear of the cup and objects and ≥ 10 mm above the table.
- Position and orientation are interpolated together. Every commanded pose is re-validated right before it is sent, and tracking is checked after every move.

**Guards before every phase**
- robot fault;
- calibration file unchanged;
- perception fresh (≤ 120 s);
- workspace, Z, reach and clearance on each pose.

**Abort / recover**
- A fault or tracking error stops motion and asks for help.
- Otherwise the executed trajectory is **rewound**: untilt, return the source to its pick spot, release, back out, home.
- A failed rewind stops and asks for help.

**Modes and gates**

| Mode | CLI | Needs stages | Liquid |
|---|---|---|---|
| `plan` (dry-run, no motion) | `pour_can.py` | — | — |
| `hover` (D) | `--stage hover` | B, C | — |
| `grasp` (E) | `--stage grasp` | B–D | — |
| `pour_dry` (F) | `--stage pour-dry` | B–E | empty only |
| `pour_liquid` (G) | `--stage pour-liquid --tray --estop --mentor --bounded` | B–F | `POUR_ALLOW_LIQUID=1` |
| `pour` (voice/orchestrator) | — | B–G | `POUR_ALLOW_LIQUID=1` |

Everything live needs `ENABLE_CALIBRATED_POUR=1` (off by default) and `VIAM_ALLOW_LIVE=1`.

### Honest limits

- **Open-loop pouring.** There is no fill sensing: the amount is set by tilt and `hold_s`, capped at 3 s with liquid. The robot never claims the cup is full.
- **Reachability** is only proven by the arm executing. The planner checks reach radius, the keep-out and collisions, not inverse kinematics.
- **Between waypoints** the xArm driver moves in joint space. Steps are ≤ 10 mm and ≤ 5° so the deviation stays small, but it isn't collision-checked.
- **Speed** is the arm's configured `speed_degs_per_sec` (60 today). Lower it for stages E–G.
- **Fault and e-stop state** isn't exposed through this Viam arm API. Tracking error is the runtime proxy, so a human on the e-stop is required for every motion stage.
- **Re-observation** tilts the held bottle by up to 45° dry and 25° with liquid. If no unoccluded view exists, the run aborts and the bottle is put back.
- **Detection.** In your last frame Moondream didn't detect the upright bottle from above. Stage C will show whether the phrase list needs changing.
- **Cans:** the mouth is modelled as the whole top disk, which is conservative. The real opening is off-centre.

## 5. Dry-run, replay, evidence

- `python scripts/pour_can.py` observes, localizes and plans with **no motion**. It writes `out/pour_runs/<stamp>/`: RGB-D frames, masks, estimates, overlay, `plan.json`, top and side plan views, events and summary.
- `python scripts/pour_can.py --replay out/pour_runs/<stamp>` re-runs localization and planning offline from the saved RGB-D, poses and calibration.
- `plan.json` holds TCP and flange poses for every waypoint, mouth and lip positions, clearances, the calibration ID and rejection reasons.

## 6. Voice and orchestrator

- **Voice:** the existing single LLM call now also emits a `pour` task with `source`/`target`, which `resolve_pour_intent` validates deterministically. An unnamed source asks for clarification. "I am thirsty" becomes an implicit pour with source `any`, which still needs exactly one source in view. If pouring isn't enabled and ready, voice says so and suggests "hand me the bottle".
- **Orchestrator:** `task == "pour"` goes through `make_skill_call("pour", …)` and then `execute_call`, reaching the same `run_pour_request`. There is no planner or LLM call.
- **Success** is reported only after `DONE`.

## 7. Staged hardware validation (stop at the first failure)

| Stage | What | Pass rule | Status |
|---|---|---|---|
| A | offline tests + replay | 95 tests green | **run: pass** (see §8) |
| B | calibration validation, teach mode | touch-off ≥ 9 pts, p95 XY ≤ 10, max ≤ 15, p95 Z ≤ 10 mm | **unrun** |
| C | localization, 9 positions × 3 arrangements | each within limits (27/27) | **unrun** |
| D | hover ≥ 30 mm over mouth + rim, no descent | 9 runs, measured error within limits | **unrun** |
| E | empty-container side grasp | ≥ 10 trials, ≥ 9 successes, zero contact | **unrun** |
| F | empty-container pour trajectory | ≥ 10 trials, mouth over interior, ≥ 15 mm clearance, zero collisions, safe aborts | **unrun** |
| G | water, tray, human on e-stop, mentor approval, bounded amount | ≥ 9/10 into the cup, no collision, spills recorded | **unrun** |

Trials are appended to `config/calibration/pour_trials.jsonl` and bound to the setup hash, so recalibrating resets them. Check with `calibrate_pour_setup.py status`.

## 8. Offline results

The tests run on a synthetic ray-cast RGB-D scene: a 640×360 pinhole camera at the home pose through the nominal mount, with 0.6 mm depth noise.

**Localization, before vs after** (XY error to ground truth):

| scene (bottle / cup) | depth | old bottle | new bottle | old cup | new cup |
|---|---|---|---|---|---|
| (300,−60)/(180,−200) | aligned | 8.9 mm | 2.9 mm | 12.9 mm | 0.7 mm |
| same | unaligned (live config) | 28.3 mm | rejected: misregistered | 13.9 mm | rejected |
| (200,−60)/(300,−210) | aligned | 4.6 mm | 1.7 mm | 17.2 mm | 0.9 mm |
| same | unaligned | 20.7 mm | rejected | 19.6 mm | rejected |
| (150,−150)/(300,−100) | aligned | 2.9 mm | 1.4 mm | 15.0 mm | 0.8 mm |
| (260,−40)/(160,−180) | aligned | 8.1 mm | 2.9 mm | 12.6 mm | 0.6 mm |

"Old" is the previous mask centroid at the mode depth. These are synthetic numbers, not hardware accuracy: hardware accuracy is what stages B and C measure.

**Other results:**
- **Calibration chain:** rendering the ChArUco board, detecting it, PnP and the hand-eye solve recover the mount to within 0.07 mm and 0.06°, with a held-out p95 of 0.15 mm.
- **Re-observation:** from a 45° view it agrees with the first observation to 1.7 mm (XY), 0.4 mm (radius) and 1.5 mm (Z).
- **State timing:** a full dry pour takes ~1.7 s of compute, with no real motion. Re-observation view planning is ~1.2 s; each other phase is < 0.2 s. Motion time on hardware is recorded per run in `events.jsonl`.

Run the tests with `python -m unittest discover -s tests -t .` (~30 s).

## 9. Rollback

- **Runtime:** leave `ENABLE_CALIBRATED_POUR` unset, which is the default. No pour path runs and the sorter is unchanged.
- **Code:** run `scripts/rollback_pour.sh`, which restores the touched files to `c8b7d04` and removes the new modules. After this patch is committed, use `git revert <commit>` instead.
