"""Thin wrapper around the Viam ``Camera`` component."""

from __future__ import annotations

from typing import Tuple

from viam.components.camera import Camera
from viam.media.video import ViamImage
from viam.robot.client import RobotClient


class ArmCamera:
    """Convenience wrapper around a single Viam ``Camera`` resource.

    Usage:
        camera = ArmCamera.from_robot(robot, "cam0")
        image = await camera.get_image()
    """

    def __init__(self, camera: Camera) -> None:
        self._camera = camera

    @classmethod
    def from_robot(cls, robot: RobotClient, name: str) -> "ArmCamera":
        """Look up the named camera resource on ``robot`` and wrap it."""
        return cls(Camera.from_robot(robot, name))

    async def get_image(self) -> ViamImage:
        """Return a single frame from the camera.

        TODO: decide on mime-type filtering and error handling for
        cameras that expose multiple named image streams.
        """
        images, _metadata = await self._camera.get_images()
        return images[0]

    async def get_point_cloud(self) -> Tuple[bytes, str]:
        """Return (point_cloud_bytes, mime_type) from the camera.

        TODO: add a helper to decode this into a numpy/open3d point cloud.
        """
        return await self._camera.get_point_cloud()
