# Viam-hackathon-arm-5-

xArm6 cell that finds red/yellow blocks, a soda can, cup, bottle, AirPods, or pen, picks at depth-derived Z, and places into a named target.

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

## Layout

```
components/     robot, vision, gripper, voice helpers
scripts/        one-shot cell commands
voice/          hold-to-talk UI
collector/      live stream + dataset capture
moondream/      loaded detect host + boxes
module/         on-robot Viam vision module
machine/        cell config snapshot
```

## Setup

1. Copy `.env.example` to `.env`.
2. From the machine **CONNECT** tab in the Viam app, paste `MACHINE_ADDRESS`, `API_KEY`, and `API_KEY_ID`.
3. Install and activate the venv:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Taught poses and safety

Values in `components/constants.py` (also mirrored in `machine/config.json`):

| Pose | How we move | Notes |
| --- | --- | --- |
| **home** | joints | Start / recapture view. TCP ≈ `(281, -87, 533)` |
| **bin1** | joints | Default drop for red blocks / can / AirPods. Slightly outside the workspace polygon |
| **bin2** | joints | Default drop for yellow blocks / cup / pen. Slightly outside the workspace polygon |
| **dropoff** | joints | Taught drop-off. TCP ≈ `(14, 324, 111)` — below floor Z, joints only |
| **handoff** | joints | Taught hand-off. TCP ≈ `(110, 330, 205)` |
| Pick XY | cartesian | Depth + `transform_pose` into world |
| Pick Z | cartesian | Always the floor: **`179.76 mm`** |

Every `ArmComponent.move_to_position` call is checked against:

- **Z floor** `179.76 mm`
- **Workspace** BL → TL → TR → BR: `(-77, 489)`, `(459, 507)`, `(476, -286)`, `(145, -303)`

```sh
python scripts/check_workspace.py
python scripts/get_joint_positions.py
python scripts/go_home.py
```

## Pick and sort

`scripts/sort_blocks.py` is the main routine: home, detect the named objects, pick at mode-depth Z, place by spoken target. Red/yellow blocks use HSV; can / cup / bottle / AirPods / pen use Moondream boxes, then **SAM** (`facebook/sam-vit-base`) on each crop so gripper yaw follows `minAreaRect` of the mask.

The same SAM family is on the Viam registry as `viam:sam2-detector` (`sam2` / `sam2-segments`). This cell runs SAM locally so pick does not wait on a farm vision service.

| Object | Default bin |
| --- | --- |
| red | bin1 |
| yellow | bin2 |
| can (soda can) | bin1 |
| cup | bin2 |
| airpods | bin1 |
| pen | bin2 |
| bottle | bin1 |

```sh
python scripts/locate_blocks.py
python scripts/sort_blocks.py
python voice/voice.py                 # http://127.0.0.1:8766
python collector/collect.py           # http://127.0.0.1:8767
python moondream/server.py            # http://127.0.0.1:8768
python scripts/capture_image.py
python scripts/find_colors.py
python scripts/find_shapes.py
```

`python voice/voice.py` is hold-to-talk. Whisper + LLM map speech to `home`, `sort`, `locate`, `capture`, or `quit`. Do not use Globe/Fn twice (emoji picker). Needs `OPENAI_API_KEY`.

Collector writes paired samples into `collector/dataset/` (gitignored): `color/`, `depth/`, `depth_viz/`, `meta/`.

Moondream stays loaded; ping `/api/detect` to classify objects and draw boxes into `collector/dataset/annotated/`.

## Cursor / Viam MCP (optional)

`.cursor/mcp.json` points at `http://127.0.0.1:8765`. Tunnel if needed:

```sh
viam machine part tunnel --part=<main-part-id> --local-port=8765 --remote-port=8765
```
