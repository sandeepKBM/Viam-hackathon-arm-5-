# Viam-hackathon-arm-5-

xArm6 cell that finds red and yellow blocks, picks them at the taught floor Z, and sorts them into bins.

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
| **bin1** | joints | Red drop. Slightly outside the workspace polygon |
| **bin2** | joints | Yellow drop. Slightly outside the workspace polygon |
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

## Color sort

`scripts/sort_blocks.py` is the main routine: home, detect red/yellow, pick at depth-derived Z, place by color.

| Color | Bin |
| --- | --- |
| red | bin1 |
| yellow | bin2 |

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
