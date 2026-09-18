# arm5.hardware

Owns the connection to the physical/simulated robot and thin wrappers around
its components.

Key files:
- `robot.py` -- `connect()` builds a `viam.robot.client.RobotClient` from
  `ROBOT_ADDRESS` / `ROBOT_API_KEY` / `ROBOT_API_KEY_ID` env vars.
- `arm.py` -- wrapper around the Viam `Arm` component (end position, joint
  positions, move commands).
- `gripper.py` -- wrapper around the Viam `Gripper` component (open, grab,
  stop).

TODO: add reconnect/retry policy, resource-name validation against
`config/robot.example.json`, and a context-manager form of `connect()`.
