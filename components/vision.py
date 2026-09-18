import os
from dataclasses import dataclass
from typing import List, Optional

from viam.components.camera import Camera
from viam.robot.client import RobotClient
from viam.services.vision import VisionClient

from components.shapes import (
    CAMERA_NAME,
    DetectedShape,
    LocatedShape,
    find_shapes,
    locate_block_colors,
    locate_shapes_3d,
)

DETECTOR_NAME = os.environ.get("DETECTOR_NAME", "vision-1")
SEGMENTER_NAME = os.environ.get("SEGMENTER_NAME", "shape-segmenter")
WORLD_FRAME = os.environ.get("WORLD_FRAME", "world")


@dataclass
class Shape2D:
    label: str
    confidence: float
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    @property
    def center_px(self) -> tuple:
        return ((self.x_min + self.x_max) // 2, (self.y_min + self.y_max) // 2)


async def detect(
    machine: RobotClient,
    detector_name: str = DETECTOR_NAME,
    camera_name: str = CAMERA_NAME,
) -> List[Shape2D]:
    detector = VisionClient.from_robot(machine, detector_name)
    dets = await detector.get_detections_from_camera(camera_name)
    return [
        Shape2D(
            label=d.class_name,
            confidence=d.confidence,
            x_min=d.x_min,
            y_min=d.y_min,
            x_max=d.x_max,
            y_max=d.y_max,
        )
        for d in dets
    ]


class VisionComponent:
    def __init__(
        self,
        machine: RobotClient,
        camera_name: str | None = None,
        detector_name: str | None = None,
    ) -> None:
        self.machine = machine
        self.camera_name = camera_name or os.environ.get("CAMERA_NAME", "cam")
        self.detector_name = detector_name or DETECTOR_NAME
        self.camera = Camera.from_robot(machine, self.camera_name)

    async def find_shapes(self) -> List[DetectedShape]:
        return await find_shapes(self.machine, self.camera_name)

    async def locate_shapes(self) -> List[LocatedShape]:
        return await locate_shapes_3d(self.machine, self.camera_name, WORLD_FRAME)

    async def locate_blocks(self, colors: tuple[str, ...] = ("red", "yellow")) -> List[LocatedShape]:
        return await locate_block_colors(
            self.machine, self.camera_name, WORLD_FRAME, colors=colors
        )

    async def detect(self) -> List[Shape2D]:
        return await detect(self.machine, self.detector_name, self.camera_name)
