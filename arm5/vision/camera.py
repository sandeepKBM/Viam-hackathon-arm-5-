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
        """Return the first frame from the camera.

        Note: viam-sdk 0.80.0's ``Camera`` has no single-image call; it only
        exposes ``get_images()`` (all named streams). We return the first
        stream here.

        TODO: filter by ``filter_source_names`` / mime type so this returns
        the intended color stream deterministically rather than "the first".
        """
        images, _metadata = await self._camera.get_images()
        if not images:
            raise RuntimeError(
                "Camera.get_images() returned no image streams; cannot return a frame."
            )
        return images[0]

    async def get_point_cloud(self) -> Tuple[bytes, str]:
        """Return (point_cloud_bytes, mime_type) from the camera.

        TODO: add a helper to decode this into a numpy/open3d point cloud.
        """
        return await self._camera.get_point_cloud()
