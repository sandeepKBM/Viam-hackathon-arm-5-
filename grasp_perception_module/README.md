# grasp_perception_module

A Viam **generic service** that runs perception **ON the machine** (in-process
with `viam-server`, on the arm's own computer) so heavy payloads -- point
clouds and images -- never cross the network. It reads the LOCAL camera /
vision services, runs detection + grasp localization there, and returns only a
tiny JSON result over `do_command`.

Model: `hackathons:grasp-perception:grasp-service`, API: `rdk:service:generic`.

## Why this exists (measured numbers)

Client code runs on a remote VM (westeros) and reaches the arm only through the
Viam cloud / TURN relay. Profiling over that relay:

| call                          | latency over relay |
|-------------------------------|--------------------|
| `arm.is_moving`, `arm.get_end_position` | ~36 ms (fine) |
| `camera.get_images()`         | ~2,000 ms          |
| `camera.get_point_cloud()`    | **~12,500 ms** for a 14.7 MB cloud  <- bottleneck |

Shipping the point cloud across the relay dominates the loop. This service runs
inside `viam-server` on the arm's box, accesses the camera / point cloud
**locally** (in-process, no relay), classifies grasps there, and returns only a
handful of floats. That turns **~12.5 s -> ~0.04 s**: the 14.7 MB never leaves
the machine.

## What it does (and does NOT do)

Pure **orchestration**. It reinvents no geometry -- it reads the local
segmenter's per-object point clouds and hands them to the repo's existing
perception code:

- `components/perception3d.py` -- `parse_viam_pcd` (PCD parse) +
  `grasps_from_point_cloud_objects` (segmenter -> grasp bridge, with a
  geometry-bbox fallback for sparse/unparseable clouds).
- `components/grasp_affordance.py` -- `classify_grasp`
  (`top_down`/`side`/`inside_outside` geometry), called transitively.
- `components/shapes.py` (`CAMERA_NAME`) and `components/constants.py`
  (`FLOOR_Z`) -- defaults for the camera name and floor height (nothing is
  hardcoded here).

It never returns raw point clouds or images -- only small computed results.

## Files

- `service.py` -- the `GraspPerceptionService` class (the logic).
- `fake_backend.py` -- synthetic camera / segmenter / detector fakes returning
  Viam-style PCD point clouds, plus `build_dependencies(...)` for offline tests.
- `main.py` -- module entrypoint; adds the repo root to `sys.path` and registers
  the model with a Viam `Module`.
- `meta.json`, `run.sh`, `requirements.txt` -- module packaging/deployment.
- `tests/` -- offline unit tests (fakes only; no machine, no network, no torch).

## Config attributes

| attribute     | type   | required | default                | notes                                             |
|---------------|--------|----------|------------------------|---------------------------------------------------|
| `camera`      | string | yes      | --                     | LOCAL camera resource name                        |
| `segmenter`   | string | yes      | --                     | vision service exposing `get_object_point_clouds` |
| `detector`    | string | yes      | --                     | vision service exposing `get_detections`          |
| `floor_z`     | float  | no       | `FLOOR_Z` (179.75673)  | floor height (mm); from `components/constants.py`  |
| `world_frame` | string | no       | `world`                | frame label carried for placement math            |

`validate_config` returns `[camera, segmenter, detector]` as the required
dependency names so `viam-server` wires their LOCAL handles into
`dependencies` before `reconfigure` runs. It rejects missing/blank required
attributes and a non-numeric `floor_z` at config time.

## do_command contract (the network boundary -- keep it exact)

All access is through `do_command({"cmd": ...})`. Every response is a plain
JSON-serializable dict.

- **`{"cmd": "health"}`**
  -> `{"ok": true, "camera": "<name>", "segmenter": "<name>", "detector": "<name>"}`

- **`{"cmd": "detections"}`**
  -> `{"ok": true, "objects": [{"label": str, "world_xyz_mm": [x, y, z], "score": float}, ...]}`
  `score` is the grasp classifier's heuristic confidence (no 2D-detector score
  flows through the point-cloud path; see "Notes / stubs").

- **`{"cmd": "localize", "object": "<label>", "hint_xy": [x, y]?}`**
  - success -> `{"ok": true, "world_xyz_mm": [x, y, z], "grasp_type": "top_down"|"side"|"inside_outside", "approach_vec": [ax, ay, az], "yaw_deg": float, "label": str, "n_points": int}`
  - failure -> `{"ok": false, "error": "<reason>"}`
  Label match is case-insensitive substring. With `hint_xy`, the nearest match
  in XY is chosen; without it, the highest-confidence match.

Unknown/missing `cmd` returns `{"ok": false, "error": ...}` (errors stay inside
the JSON contract rather than raising across the RPC).

## Units / frame assumptions

- Distances **out are millimetres**. `world_xyz_mm` comes straight from
  `ObjectGrasp.center_xyz`, which `components/perception3d.py` sources from the
  segmented object's Viam `Geometry.center` (Viam geometries are mm), falling
  back to `classify_grasp`'s cloud centroid. **This service adds no unit scaling
  of its own** -- it reuses the existing perception3d/shapes convention.
- **Frame:** results are in whatever frame the configured `segmenter` emits its
  geometries in. If the camera is placed in the machine's frame system, those
  centers are already world-frame; otherwise camera-frame. This service holds
  only camera/vision handles (not a `RobotClient`), so it does not run
  `transform_pose` -- frame correctness is a live-machine check.
- No control rate: this is request/response perception, not a control loop. Each
  `do_command` does one capture + classify; the caller sets the cadence by how
  often it polls. Nothing here streams or actuates the arm.

## Deploying as a LOCAL module on the arm's computer

The module code must live **on the arm's own computer** (the box running
`viam-server`), so its perception runs in-process. Copy this repo (or at least
`grasp_perception_module/` plus the `components/` it imports) there, then add a
`local` module + a service to that machine's config:

```json
{
  "modules": [
    {
      "type": "local",
      "name": "grasp-perception",
      "executable_path": "/path/on/arm/viam_5/grasp_perception_module/run.sh"
    }
  ],
  "services": [
    {
      "name": "grasp-service",
      "api": "rdk:service:generic",
      "model": "hackathons:grasp-perception:grasp-service",
      "attributes": {
        "camera": "cam",
        "segmenter": "vision-segment",
        "detector": "shape-detector",
        "floor_z": 179.75673,
        "world_frame": "world"
      },
      "depends_on": ["cam", "vision-segment", "shape-detector"]
    }
  ]
}
```

The referenced `cam` (camera), `vision-segment` and `shape-detector` (vision
services) must already exist in the same machine config -- they are the LOCAL
resources this service reads in-process. `run.sh` bootstraps a local `.venv`
and installs `requirements.txt` (viam-sdk, numpy, opencv-python-headless) on
first launch.

The westeros-side client then calls this over the relay with tiny commands:

```python
svc = Generic.from_robot(machine, "grasp-service")
await svc.do_command({"cmd": "health"})
await svc.do_command({"cmd": "detections"})
await svc.do_command({"cmd": "localize", "object": "cup", "hint_xy": [220, 180]})
```

## Notes / stubs / live-machine checks

- **Detector dependency** is validated, wired into `dependencies`, and reported
  by `health`, but the `detections`/`localize` paths currently source everything
  (label + 3D center + grasp) from the **segmenter's** point clouds -- that is
  what carries 3D geometry. The 2D `detector` handle (`get_detections`) is
  reserved for a future label-enrichment path (`components/zeroshot.py`); it is
  not required for the current contract.
- **PCD units on real hardware:** the offline fakes emit synthetic clouds in mm,
  self-consistent with the mm `Geometry.center`. A real depth camera may emit
  the point cloud in **metres**, while `classify_grasp`'s gripper-width
  thresholds assume mm. If the live segmenter's PCD is in metres, confirm the
  scale before trusting `grasp_type`/`width` (the `world_xyz_mm` centers come
  from the mm `Geometry.center`, so those are unaffected). This is called out as
  a live-machine check, per the repo's "never silently change units" rule.
- **`score`** in `detections` is the grasp heuristic confidence, not a detector
  probability -- see the contract above.

## Running the tests

```
cd /common/users/ss5772/viam_5
python -m pytest grasp_perception_module/tests -q
# or, with the repo venv:
source .venv/bin/activate && python -m pytest grasp_perception_module/tests -q
```

All tests run with the fakes only -- zero hardware, zero network, and with the
`xarm` SDK and torch/CUDA absent.
