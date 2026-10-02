import os
import sys
from typing import ClassVar, List, Mapping, Optional, Sequence

from typing_extensions import Self
from viam.components.camera import Camera
from viam.logging import getLogger
from viam.media.video import ViamImage
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import PointCloudObject, ResourceName
from viam.proto.service.vision import Classification, Detection, GetPropertiesResponse
from viam.resource.base import ResourceBase
from viam.resource.registry import Registry, ResourceCreatorRegistration
from viam.resource.types import Model, ModelFamily
from viam.services.vision import CaptureAllResult, Vision
from viam.utils import ValueTypes, struct_to_dict

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from components.shapes import (  # noqa: E402
    _decode_color as decode_color,
    _split_color_depth,
    classify_shapes,
)

LOGGER = getLogger(__name__)


class ShapeDetector(Vision):
    MODEL: ClassVar[Model] = Model(ModelFamily("hack", "shape-finder"), "detector")

    def __init__(self, name: str):
        super().__init__(name)
        self.camera: Optional[Camera] = None
        self.camera_name: str = ""

    @classmethod
    def new(
        cls,
        config: ComponentConfig,
        dependencies: Mapping[ResourceName, ResourceBase],
    ) -> Self:
        instance = cls(config.name)
        instance.reconfigure(config, dependencies)
        return instance

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> Sequence[str]:
        attrs = struct_to_dict(config.attributes)
        camera_name = attrs.get("camera")
        if not camera_name:
            raise ValueError("'camera' attribute is required")
        return [str(camera_name)]

    def reconfigure(
        self,
        config: ComponentConfig,
        dependencies: Mapping[ResourceName, ResourceBase],
    ) -> None:
        attrs = struct_to_dict(config.attributes)
        self.camera_name = str(attrs["camera"])
        self.camera = dependencies[Camera.get_resource_name(self.camera_name)]
        LOGGER.info("ShapeDetector reconfigured: camera=%s", self.camera_name)

    @staticmethod
    def _to_detections(shapes) -> List[Detection]:
        out: List[Detection] = []
        for s in shapes:
            x, y, w, h = s.box
            out.append(
                Detection(
                    x_min=int(x),
                    y_min=int(y),
                    x_max=int(x + w),
                    y_max=int(y + h),
                    confidence=1.0,
                    class_name=s.label,
                )
            )
        return out

    async def _grab_bgr(self):
        assert self.camera is not None, "no camera configured"
        images, _ = await self.camera.get_images()
        bgr, _ = _split_color_depth(images)
        if bgr is None:
            raise RuntimeError("camera returned no decodable color image")
        return bgr

    async def get_detections(
        self,
        image: ViamImage,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> List[Detection]:
        return self._to_detections(classify_shapes(decode_color(image)))

    async def get_detections_from_camera(
        self,
        camera_name: str,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> List[Detection]:
        return self._to_detections(classify_shapes(await self._grab_bgr()))

    async def capture_all_from_camera(
        self,
        camera_name: str,
        return_image: bool = False,
        return_classifications: bool = False,
        return_detections: bool = False,
        return_object_point_clouds: bool = False,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> CaptureAllResult:
        result = CaptureAllResult()
        assert self.camera is not None, "no camera configured"
        images, _ = await self.camera.get_images()
        if return_image:
            result.image = images[0] if images else None
        if return_detections:
            bgr, _ = _split_color_depth(images)
            result.detections = (
                self._to_detections(classify_shapes(bgr)) if bgr is not None else []
            )
        return result

    async def get_classifications(
        self,
        image: ViamImage,
        count: int,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> List[Classification]:
        raise NotImplementedError("shape-finder:detector is a detector")

    async def get_classifications_from_camera(
        self,
        camera_name: str,
        count: int,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> List[Classification]:
        raise NotImplementedError("shape-finder:detector is a detector")

    async def get_object_point_clouds(
        self,
        camera_name: str,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> List[PointCloudObject]:
        raise NotImplementedError("shape-finder:detector does not support point clouds")

    async def get_properties(
        self,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Vision.Properties:
        return GetPropertiesResponse(
            classifications_supported=False,
            detections_supported=True,
            object_point_clouds_supported=False,
        )

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Mapping[str, ValueTypes]:
        return {}


Registry.register_resource_creator(
    Vision.API,
    ShapeDetector.MODEL,
    ResourceCreatorRegistration(ShapeDetector.new, ShapeDetector.validate_config),
)
