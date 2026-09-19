"""Stubs for turning 2D detections + camera intrinsics into 3D object poses.

This module intentionally has no Viam SDK dependency: it operates on plain
detection/intrinsics data so it can be unit tested without a live robot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from viam.components.camera import IntrinsicParameters
from viam.services.vision import Detection


@dataclass
class ObjectPose:
    """Estimated pose of a detected object, in camera (or world) frame."""

    label: str
    x: float
    y: float
    z: float
    confidence: float


def estimate_object_pose(
    detection: Detection,
    intrinsics: IntrinsicParameters,
    depth_m: float,
) -> ObjectPose:
    """Back-project a 2D detection's bounding box center to a 3D point.

    TODO: implement the pinhole back-projection using `intrinsics` and a
    depth estimate (from a depth camera or a point cloud lookup), then
    transform from camera frame into the arm/world frame.
    """
    raise NotImplementedError("TODO: pinhole back-projection + frame transform")


def estimate_object_poses(
    detections: Sequence[Detection],
    intrinsics: IntrinsicParameters,
    depth_m: float,
) -> List[ObjectPose]:
    """Batch version of :func:`estimate_object_pose`.

    TODO: consider per-detection depth lookup instead of a single shared
    `depth_m` once a depth camera / point cloud is wired in.
    """
    return [estimate_object_pose(d, intrinsics, depth_m) for d in detections]
