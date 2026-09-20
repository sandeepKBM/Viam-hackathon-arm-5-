"""grasp_perception_module: a Viam generic service that runs perception ON the
arm's own computer, in-process with viam-server.

It reads the LOCAL camera's segmented point clouds and vision services (no
cloud/TURN relay), runs the repo's existing grasp geometry
(`components.perception3d` / `components.grasp_affordance`) there, and returns
only tiny JSON results over `do_command` -- so a ~12.5 s cross-relay
point-cloud fetch becomes a ~0.04 s round trip.

Importing this package never requires a live machine, a network, or torch/CUDA:
the reused geometry is pure numpy (numba optional/lazy), and the vision/camera
resources are accessed through injected handles (see `fake_backend` for the
offline stand-ins used by `grasp_perception_module/tests/`).
"""

from grasp_perception_module.service import GraspPerceptionService

__all__ = ["GraspPerceptionService"]
