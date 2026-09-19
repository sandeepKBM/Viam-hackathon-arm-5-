"""Wrapper around the Viam ``VisionClient`` service."""

from __future__ import annotations

from typing import List

from viam.media.video import ViamImage
from viam.robot.client import RobotClient
from viam.services.vision import Classification, Detection, VisionClient


class ArmDetector:
    """Convenience wrapper around a single Viam vision service resource.

    Usage:
        detector = ArmDetector.from_robot(robot, "vision-1")
        detections = await detector.get_detections(image)
    """

    def __init__(self, vision: VisionClient) -> None:
        self._vision = vision

    @classmethod
    def from_robot(cls, robot: RobotClient, name: str) -> "ArmDetector":
        """Look up the named vision service resource on ``robot`` and wrap it."""
        return cls(VisionClient.from_robot(robot, name))

    async def get_detections(self, image: ViamImage) -> List[Detection]:
        """Run object detection on a single image.

        TODO: expose a `from_camera` variant + confidence-threshold filtering.
        """
        return await self._vision.get_detections(image)

    async def get_classifications(self, image: ViamImage, count: int = 1) -> List[Classification]:
        """Run classification on a single image, returning the top ``count`` results.

        TODO: decide default `count` and whether to expose raw scores.
        """
        return await self._vision.get_classifications(image, count)
