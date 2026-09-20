# Viam-hackathon-arm-5-

UFactory xArm 5 (5-DOF) cell that finds red and yellow blocks on the table, picks them at the taught floor Z, and sorts them into bins.

Machine details live in `machine/config.json`. Do not commit `.env`.

| | |
| --- | --- |
| Machine ID | `25950d11-fdd4-48e8-bebf-ca5857103f46` |
| Machine address | `armfarm5-main.310sld03v2.viam.cloud` |
| Arm | `arm` |
| Camera | `cam` (RealSense `151222070758`) |
| Gripper | `gripper` |
| Obstacles | `table`, `wall-front`, `wall-side`, `ceiling` |
| Arm IP | `192.168.1.233` |

Cell fragment obstacles (mm, world): table `z = -123`, ceiling `z = 1050`, front wall `x = 740`, side wall `y = -500`.

## Setup

1. Copy `.env.example` to `.env`.
2. From the machine **CONNECT** tab in the Viam app, paste `MACHINE_ADDRESS`, `API_KEY`, and `API_KEY_ID`.
3. Install and activate the venv:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`ARM_NAME`, `CAMERA_NAME`, `GRIPPER_NAME`, and `FLOOR_Z` already match this cell.

## Taught poses and safety

Values in `components/constants.py` (also mirrored in `machine/config.json`):

| Pose | How we move | Notes |
| --- | --- | --- |
| **home** | joints | Start / recapture view. TCP ≈ `(281, -87, 533)` |
| **bin1** | joints | Red drop. Slightly outside the workspace polygon |
| **bin2** | joints | Yellow drop. Slightly outside the workspace polygon |
| Pick XY | cartesian | Depth + `transform_pose` into world |
| Pick Z | cartesian | Always the floor: **`179.76 mm`** |

Every `ArmComponent.move_to_position` call is checked against:

- **Z floor** `179.76 mm` — end-effector is never commanded below this
- **Workspace** — taught quad BL → TL → TR → BR: `(-77, 489)`, `(459, 507)`, `(476, -286)`, `(145, -303)`

Bin drops use `go_to("bin1"|"bin2")` (joint moves) so they are not blocked by the polygon.

```sh
python check_workspace.py
python get_joint_positions.py
python go_home.py
```

## Color sort (pick and place)

`sort_blocks.py` is the main routine:

1. Go home, open the gripper, capture color + depth from `cam`.
2. Detect **red** and **yellow** blocks (OpenCV HSV). Yellows first so a stacked yellow comes off before the red under it.
3. Deproject each centroid and transform it into the world frame.
4. For each block:
   - open gripper
   - move to pick XY at travel height (home Z)
   - descend to **floor Z**
   - grab
   - lift
   - go to the taught bin
   - open gripper
   - home

| Color | Bin |
| --- | --- |
| red | bin1 |
| yellow | bin2 |

```sh
python locate_blocks.py          # home + print world XY / workspace check (no grasp)
python sort_blocks.py            # full sort
```

## Polished pick-and-place (fast planner + local IK)

`run_pick_and_place.py` is the full, real-arm-ready loop: connect → perceive
(`VisionComponent`) → plan (`components/policy.py`'s rule-based planner,
default goal `"sort the blocks"`) → execute with retries + declutter
(`components/pipeline.py`, `components/retry.py`) → record to the
self-improving `components/experience_store.py` → print a per-object
summary (picked / decluttered / skipped, retry attempts/escalations, and
the calibrated plan learned for each object so far).

```sh
.venv/bin/python run_pick_and_place.py                       # live robot, default goal
.venv/bin/python run_pick_and_place.py --goal "sort the red blocks"
.venv/bin/python run_pick_and_place.py --detector zeroshot    # swap the perception path
.venv/bin/python run_pick_and_place.py --dry-run              # OFFLINE, no Viam connection
```

**Position control, no RRT per pick.** Every pick runs through
`components/fast_planner.py`: a deterministic 3-waypoint up → over → down
joint-space plan, executed with `move_to_joint_positions` only (never
`move_to_position`) — replacing 3 RRT motion-plan calls per pick with 0.
Declutter (`components/declutter.py`) still clears blockers geometrically
before the pick if needed, and every attempt is recorded so the experience
store's calibrated offsets/retry-budget keep adapting run to run.

**Local IK (`components/ik.py`).** The Viam SDK pinned in this repo exposes
no `compute_inverse_kinematics` (only `Arm.get_kinematics`), so the fast
planner's real IK source is a local numerical solver built on
[`ikpy`](https://github.com/Phylliade/ikpy) against the bundled xArm5 URDF
at `assets/xarm5.urdf` (a copy of `xarm5_sim`'s xacro-expanded
`xarm5_gripper_raw.urdf`, chained `link_base → joint1..5 → link5 → ...
gripper → link_tcp`). It targets the TCP's position plus its wrist (+Z)
axis direction — the fixed top-down grasp orientation every pick uses — and
verifies its own answer via forward kinematics before returning it,
raising `components.ik.UnreachablePoseError` (caught per-object, not fatal
to the whole run) instead of silently handing back a bad joint target for
a pose outside the arm's reach. `pip install ikpy` (already in
`requirements.txt`) is required to build a solver; the rest of the repo
imports fine without it. See `tests/test_ik.py` for the
FK(IK(pose)) round-trip verification (sub-3mm / sub-2° on this cell's
taught workspace) and `--urdf` to point at a different URDF (e.g. one
fetched from a live arm's own `get_kinematics()` via
`components.ik.make_local_ik_fn_from_arm`).

`--dry-run` swaps in a synthetic two-block scene and a mock arm/gripper (no
Viam connection, no camera) so the whole loop — perceive → plan → fast pick
via local IK → record — is verifiable with no robot attached; its
experience-store writes go to a temp file, never `data/experience.json`.

## Vision scripts

```sh
python capture_image.py          # save color frame to out/frame.png
python find_colors.py            # red/yellow bboxes on a saved image → out/colors.png
python find_shapes.py            # home, then classify red triangle/cube/cuboid
```

`components/shapes.py` does HSV color masks and shape labels. `components/vision.py` talks to `cam` and optional Viam vision services.

To run the detector on the machine, add a local module pointing at `module/run.sh` and a vision service:

```json
{
  "name": "vision-1",
  "api": "rdk:service:vision",
  "model": "hack:shape-finder:detector",
  "attributes": { "camera": "cam" }
}
```

## Zero-shot object detection (HuggingFace, experimental)

Open-vocabulary detection: give the model **text prompts** ("red block",
"yellow block") instead of tuning HSV ranges — no training. Default model is
OWLv2 (`google/owlv2-base-patch16-ensemble`); Grounding DINO also works through
the same `zero-shot-object-detection` pipeline. Output is the same
`DetectedShape` the OpenCV path produces, so depth deprojection and the pick
pipeline are unchanged. Torch/transformers are imported lazily — the color-sort
path keeps working without them.

```sh
pip install -r requirements-zeroshot.txt      # torch, transformers, pillow, timm

python capture_image.py                        # grab out/frame.png
python find_objects_zeroshot.py                # offline, on the saved image (no robot)
python find_objects_zeroshot.py out/frame.png "red block" "yellow block" "screwdriver"

python locate_objects_zeroshot.py              # home + live cam + world XY (no grasp)
```

Env knobs (`components/zeroshot.py`): `ZS_MODEL`, `ZS_THRESHOLD` (default `0.1`),
`ZS_PROMPTS` (comma-separated). First run downloads model weights. `VisionComponent`
exposes `locate_objects_zeroshot(prompts=..., threshold=...)` for scripting.

## Lightweight VLM: open-world naming + localization (Moondream, experimental)

Idea: a small VLM **names** what's on the table, and those names drive detection —
so you don't hardcode prompts/HSV. Moondream 2B (`vikhyatk/moondream2`) does
`caption` / `query` / **`detect`** / **`point`** natively, so it can name *and*
localize. Its `detect()` boxes come back as the same `DetectedShape`, so depth
deprojection + the pick pipeline are unchanged.

Two ways to localize (pick per-run):
1. **VLM only** — Moondream `detect()` per label. Simplest.
2. **VLM → OWLv2** — VLM lists labels, OWLv2 (`components/zeroshot.py`) draws the
   boxes. Use when you want the detector's tighter boxes.

**Runs on any device** via transformers, auto-detected: CUDA (fp16) on this
server, Apple Silicon (MPS, fp32), or plain CPU (fp32). Inference can run on a
separate machine (e.g. a Mac) — the Viam SDK connects to the robot remotely, so
the script grabs frames + sends arm commands over the network while the model
runs locally.

```sh
pip install -r requirements-vlm.txt

python capture_image.py                         # grab out/frame.png
python vlm_scene.py                             # caption + list objects + detect (offline)
python vlm_scene.py out/frame.png "red block"   # detect a specific label
python vlm_scene.py out/frame.png --ask "which block is on top?"
```

First run downloads weights (~4GB fp32 / ~2GB fp16). Env (`components/vlm.py`):
`VLM_MODEL`, `VLM_REVISION`, `VLM_DTYPE` (`auto`|`float16`|`float32`),
`VLM_DEVICE` (`cuda`|`mps`|`cpu`), `VLM_LIST_PROMPT`.

**On a weak CPU-only Mac** the 2B is slow (seconds/frame). Options: keep it (fine
for non-realtime picking), set `VLM_MODEL` to a smaller model (e.g. Florence-2),
or use Moondream's own quantized 0.5B via their `moondream` package on Apple
Silicon. `VisionComponent` also exposes `locate_objects_vlm(labels=..., use_zeroshot=...)`.

## Registry modules (config on the machine)

These are **adopted as Viam registry-module config**, not code in this repo --
add/configure them on the machine in the Viam app (or `machine/config.json`),
and our code below just consumes whatever they produce. No new module code is
required to use them; the "consumer" column is what to wire the module's
output into.

| Registry module | Role | Config | Consumed by |
| --- | --- | --- | --- |
| `viam-labs:EfficientDet-COCO` (or `moondream-vision`) | mlmodel + vision **detector** | Detects common objects (cups/cans/person) as a `Vision` service, feeding everything downstream that calls `get_detections_from_camera`. | `components/vision.py`'s `detect()`, `components/zeroshot.py`/`components/vlm.py` as alternates. |
| built-in `detections-to-segments` | **segmenter** | Wraps a detector to also emit per-object 3D point clouds (`get_object_point_clouds`) -- segmenting the scene into one full-object cloud per detection, not just a 2D box. | `components/perception3d.py`'s `get_object_grasps()` -- parses each object's PCD, runs `components/grasp_affordance.py`'s `classify_grasp`, and fixes the partial-view problem (see that module's docstring) by consuming the segmenter's fused/complete cloud instead of one oblique depth frame. |
| `viam-modules:object-tracking` | **tracker** | Wraps a detector to assign each detected object a persistent id that survives across frames. | `components/tracking.py`'s `get_tracked_objects()` -- attaches the id to `LocatedShape.track_id`; `match_to_known()` uses it (falling back to nearest-XY) so the pipeline can skip re-planning/re-capturing an object it's already handling. |
| `viam-labs:opencv` hand-eye calibration | **calibration** | One-time camera<->arm extrinsic calibration; not a runtime service. | Everything that deprojects camera pixels/points into the arm's world frame (`components/shapes.py`'s `transform_pose` calls, `components/perception3d.py`'s geometry centers) implicitly depends on this being accurate -- no code consumes it directly. |
| `filtered_camera` + Data Management | **capture** | Filtered image/data capture and sync, configured on the camera component. | Not consumed by any module here directly -- it's a data-collection path (e.g. for future model retraining), independent of the live perception calls above. |

## Cursor / Viam MCP (optional)

Cursor can drive the cell through `erh:viam-mcp-server`. `.cursor/mcp.json` points at `http://127.0.0.1:8765`.

In the [Viam app](https://app.viam.com), add a generic service with model `erh:viam-mcp-server:mcp-server`:

```json
{
  "components": ["arm", "cam", "gripper"],
  "address": ":8765"
}
```

If the machine is not on this LAN, tunnel it:

```sh
viam machine part tunnel --part=<main-part-id> --local-port=8765 --remote-port=8765
nc -zv 127.0.0.1 8765
```

Then enable the `viam` server in **Cursor Settings → Tools & MCP**. Arm tools look like `arm__end_position` and `arm__move_to_joint_positions`.
