# arm5.vision

Owns everything about turning camera frames into information the rest of the
stack can use.

Key files:
- `camera.py` -- thin wrapper around the Viam `Camera` component
  (`get_image`, `get_point_cloud`).
- `detectors.py` -- wrapper around the Viam `VisionClient` service
  (`get_detections`, `get_classifications`).
- `perception.py` -- turns raw detections + camera intrinsics into object
  poses usable by planning.

TODO: pick a concrete intrinsics/extrinsics source, decide on a detection
confidence threshold, and wire real pose-estimation math into
`perception.py`.
