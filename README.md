# Viam-hackathon-arm-5-

xArm6 cell that finds red and yellow blocks on the table, picks them at the taught floor Z, and sorts them into bins.

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
