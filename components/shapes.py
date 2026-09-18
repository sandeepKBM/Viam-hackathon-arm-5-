import math
import os
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

RED_LOWER_1 = np.array([0, 100, 60])
RED_UPPER_1 = np.array([10, 255, 255])
RED_LOWER_2 = np.array([170, 100, 60])
RED_UPPER_2 = np.array([180, 255, 255])

MIN_AREA_PX = int(os.environ.get("SHAPE_MIN_AREA_PX", 800))
CUBE_AR_MAX = float(os.environ.get("CUBE_AR_MAX", 1.25))
CAMERA_NAME = os.environ.get("CAMERA_NAME", "cam")


@dataclass
class DetectedShape:
    label: str
    cx: int
    cy: int
    area: float
    vertices: int
    aspect_ratio: float
    box: tuple
    color: str = ""
    contour: Optional[np.ndarray] = None
    angle: float = 0.0
    long_px: float = 0.0
    short_px: float = 0.0
    long_p1: tuple = (0.0, 0.0)
    long_p2: tuple = (0.0, 0.0)


COLOR_RANGES = {
    "red": [
        (np.array([0, 90, 50]), np.array([12, 255, 255])),
        (np.array([165, 90, 50]), np.array([180, 255, 255])),
    ],
    "yellow": [(np.array([18, 80, 80]), np.array([38, 255, 255]))],
    "green": [(np.array([40, 50, 40]), np.array([90, 255, 255]))],
    "blue": [(np.array([95, 80, 40]), np.array([130, 255, 255]))],
    "orange": [(np.array([8, 120, 80]), np.array([20, 255, 255]))],
}

BLOCK_MIN_AREA = int(os.environ.get("BLOCK_MIN_AREA_PX", 250))
BLOCK_MAX_AREA = int(os.environ.get("BLOCK_MAX_AREA_PX", 25000))


def red_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, RED_LOWER_1, RED_UPPER_1) | cv2.inRange(
        hsv, RED_LOWER_2, RED_UPPER_2
    )
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def _classify_contour(c: np.ndarray, min_area: int | None = None) -> Optional[DetectedShape]:
    area = cv2.contourArea(c)
    if area < (MIN_AREA_PX if min_area is None else min_area):
        return None

    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    vertices = len(approx)

    (rcx, rcy), (rw, rh), rect_angle = cv2.minAreaRect(c)
    short, long_ = sorted((rw, rh))
    aspect_ratio = long_ / max(short, 1e-6)
    rect_area = rw * rh
    fill = area / rect_area if rect_area > 0 else 0.0
    long_angle = rect_angle if rw >= rh else rect_angle + 90.0
    rad = np.deg2rad(long_angle)
    half = long_ / 2.0
    dx, dy = float(np.cos(rad) * half), float(np.sin(rad) * half)

    if vertices == 3 or fill < 0.65:
        label = "triangle"
    else:
        label = "cube" if aspect_ratio <= CUBE_AR_MAX else "cuboid"

    m = cv2.moments(c)
    if m["m00"] == 0:
        return None
    cx = int(m["m10"] / m["m00"])
    cy = int(m["m01"] / m["m00"])
    x, y, w, h = cv2.boundingRect(c)
    return DetectedShape(
        label=label,
        cx=cx,
        cy=cy,
        area=area,
        vertices=vertices,
        aspect_ratio=aspect_ratio,
        box=(x, y, w, h),
        contour=c,
        angle=float(long_angle),
        long_px=float(long_),
        short_px=float(short),
        long_p1=(float(rcx - dx), float(rcy - dy)),
        long_p2=(float(rcx + dx), float(rcy + dy)),
    )


def classify_shapes(bgr: np.ndarray) -> List[DetectedShape]:
    mask = red_mask(bgr)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shapes = [s for c in contours if (s := _classify_contour(c)) is not None]
    shapes.sort(key=lambda s: s.area, reverse=True)
    return shapes


def _color_mask(hsv: np.ndarray, ranges) -> np.ndarray:
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in ranges:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def _iou(a: tuple, b: tuple) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def _touches_border(box: tuple, shape: tuple, margin: int = 4) -> bool:
    x, y, w, h = box
    H, W = shape[:2]
    return x <= margin or y <= margin or x + w >= W - margin or y + h >= H - margin


def find_block_colors(
    bgr: np.ndarray,
    colors: tuple[str, ...] = ("red", "yellow"),
) -> List[DetectedShape]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    found: List[DetectedShape] = []
    for color in colors:
        ranges = COLOR_RANGES[color]
        mask = _color_mask(hsv, ranges)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < BLOCK_MIN_AREA or area > BLOCK_MAX_AREA:
                continue
            shape = _classify_contour(c, min_area=BLOCK_MIN_AREA)
            if shape is None:
                continue
            shape.color = color
            found.append(shape)

    found.sort(key=lambda s: s.area, reverse=True)
    kept: List[DetectedShape] = []
    for cand in found:
        clash = next(
            (k for k in kept if k.color == cand.color and _iou(cand.box, k.box) > 0.35),
            None,
        )
        if clash is None:
            kept.append(cand)
    kept.sort(key=lambda s: (s.cy, s.cx))
    return kept


def annotate_colors(bgr: np.ndarray, shapes: List[DetectedShape]) -> np.ndarray:
    out = bgr.copy()
    draw = {
        "red": (0, 0, 255),
        "yellow": (0, 255, 255),
        "green": (0, 180, 0),
        "blue": (255, 0, 0),
        "orange": (0, 140, 255),
    }
    for s in shapes:
        color = draw.get(s.color, (255, 255, 255))
        x, y, w, h = s.box
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        p1 = (int(s.long_p1[0]), int(s.long_p1[1]))
        p2 = (int(s.long_p2[0]), int(s.long_p2[1]))
        cv2.line(out, p1, p2, color, 2)
        cv2.putText(
            out,
            f"{s.color} {s.label}",
            (x, y - 8 if y > 20 else y + h + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return out


def _decode_color(vimg) -> np.ndarray:
    buf = np.frombuffer(vimg.data, np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"could not decode color image (mime {vimg.mime_type})")
    return bgr


def _split_color_depth(images):
    color = None
    depth = None
    for im in images:
        mime = (im.mime_type or "").lower()
        name = (getattr(im, "name", "") or "").lower()
        if "dep" in mime or "depth" in name:
            try:
                depth = im.bytes_to_depth_array()
            except Exception:
                pass
        elif color is None:
            try:
                color = _decode_color(im)
            except RuntimeError:
                pass
    return color, depth


async def find_shapes_from_camera(cam) -> List[DetectedShape]:
    images, _ = await cam.get_images()
    color, _ = _split_color_depth(images)
    if color is None:
        raise RuntimeError("camera returned no decodable color image")
    return classify_shapes(color)


async def find_shapes(machine, camera_name: str = CAMERA_NAME) -> List[DetectedShape]:
    from viam.components.camera import Camera

    return await find_shapes_from_camera(Camera.from_robot(machine, camera_name))


@dataclass
class LocatedShape:
    label: str
    x: float
    y: float
    z: float
    shape: Optional[DetectedShape] = None
    color: str = ""
    depth_mm: float = 0.0
    yaw: float = 0.0


async def _color_depth_intrinsics(cam):
    images, _ = await cam.get_images(timeout=60)
    bgr, depth = _split_color_depth(images)
    if bgr is None:
        raise RuntimeError("camera returned no decodable color image")
    props = await cam.get_properties()
    depth_mm = np.asarray(depth) if depth is not None else None
    return bgr, depth_mm, props.intrinsic_parameters


def _mode_valid(depth_mm: np.ndarray, mask: np.ndarray) -> float:
    d = np.asarray(depth_mm)
    vals = d[(mask > 0) & (d > 0)]
    if vals.size == 0:
        return 0.0
    rounded = np.rint(vals.astype(np.float64)).astype(np.int64)
    values, counts = np.unique(rounded, return_counts=True)
    return float(values[int(np.argmax(counts))])


def _sample_depth(depth_mm: np.ndarray, cx: int, cy: int, win: int = 5) -> float:
    h, w = depth_mm.shape[:2]
    d = np.asarray(depth_mm)
    for r in (win, 10, 20, 40):
        y0, y1 = max(0, cy - r), min(h, cy + r + 1)
        x0, x1 = max(0, cx - r), min(w, cx + r + 1)
        mask = np.zeros((h, w), np.uint8)
        mask[y0:y1, x0:x1] = 255
        z = _mode_valid(d, mask)
        if z > 0:
            return z
    return 0.0


def _sample_depth_in_box(depth_mm: np.ndarray, box: tuple) -> float:
    x, y, bw, bh = (int(v) for v in box)
    h, w = depth_mm.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w, x + max(bw, 1)), min(h, y + max(bh, 1))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    mask[y0:y1, x0:x1] = 255
    return _mode_valid(depth_mm, mask)


def _sample_depth_in_block(depth_mm: np.ndarray, shape: DetectedShape) -> float:
    d = np.asarray(depth_mm)
    h, w = d.shape[:2]
    if shape.contour is not None:
        mask = np.zeros((h, w), np.uint8)
        cv2.drawContours(mask, [shape.contour], -1, 255, thickness=-1)
        z = _mode_valid(d, mask)
        if z > 0:
            return z
    return _sample_depth_in_box(d, shape.box)


def deproject(u: float, v: float, z_mm: float, intr) -> tuple:
    x = (u - intr.center_x_px) / intr.focal_x_px * z_mm
    y = (v - intr.center_y_px) / intr.focal_y_px * z_mm
    return x, y, z_mm


def _depth_at(depth_mm: Optional[np.ndarray], shape: DetectedShape) -> float:
    if depth_mm is None:
        return 0.0
    return _sample_depth_in_block(depth_mm, shape)


async def locate_shapes_camera(cam) -> List[LocatedShape]:
    bgr, depth_mm, intr = await _color_depth_intrinsics(cam)
    if depth_mm is None:
        raise RuntimeError("camera returned no depth frame")
    out: List[LocatedShape] = []
    for s in classify_shapes(bgr):
        z = _depth_at(depth_mm, s)
        if z <= 0:
            continue
        x, y, zc = deproject(s.cx, s.cy, z, intr)
        out.append(LocatedShape(label=s.label, x=x, y=y, z=zc, shape=s))
    return out


async def _to_world(machine, camera_name: str, x: float, y: float, z: float, world_frame: str):
    from viam.proto.common import Pose, PoseInFrame

    in_world = await machine.transform_pose(
        PoseInFrame(reference_frame=camera_name, pose=Pose(x=x, y=y, z=z)),
        world_frame,
    )
    return in_world.pose


async def _pixel_to_world(
    machine,
    camera_name: str,
    u: float,
    v: float,
    z_cam: float,
    intr,
    world_frame: str,
):
    x, y, zc = deproject(u, v, z_cam, intr)
    return await _to_world(machine, camera_name, x, y, zc, world_frame)


async def locate_shapes_3d(
    machine,
    camera_name: str = CAMERA_NAME,
    world_frame: str = "world",
) -> List[LocatedShape]:
    from viam.components.camera import Camera

    cam = Camera.from_robot(machine, camera_name)
    located: List[LocatedShape] = []
    for s in await locate_shapes_camera(cam):
        p = await _to_world(machine, camera_name, s.x, s.y, s.z, world_frame)
        located.append(LocatedShape(label=s.label, x=p.x, y=p.y, z=p.z, shape=s.shape))
    return located


async def locate_block_colors(
    machine,
    camera_name: str = CAMERA_NAME,
    world_frame: str = "world",
    colors: tuple[str, ...] = ("red", "yellow"),
) -> List[LocatedShape]:
    from viam.components.camera import Camera

    cam = Camera.from_robot(machine, camera_name)
    bgr, depth_mm, intr = await _color_depth_intrinsics(cam)
    blocks = find_block_colors(bgr, colors=colors)
    if not blocks:
        return []

    table_depth = 0.0
    if depth_mm is not None:
        valid = depth_mm[depth_mm > 0]
        if valid.size:
            table_depth = float(np.median(valid))

    located: List[LocatedShape] = []
    for s in blocks:
        z = _depth_at(depth_mm, s)
        if z <= 0:
            z = table_depth
        if z <= 0:
            print(f"  skip {s.color} px=({s.cx},{s.cy}): no depth")
            continue
        p = await _pixel_to_world(machine, camera_name, s.cx, s.cy, z, intr, world_frame)
        p1 = await _pixel_to_world(
            machine, camera_name, s.long_p1[0], s.long_p1[1], z, intr, world_frame
        )
        p2 = await _pixel_to_world(
            machine, camera_name, s.long_p2[0], s.long_p2[1], z, intr, world_frame
        )
        yaw = math.degrees(math.atan2(p2.y - p1.y, p2.x - p1.x))
        located.append(
            LocatedShape(
                label=s.label,
                x=p.x,
                y=p.y,
                z=p.z,
                shape=s,
                color=s.color,
                depth_mm=z,
                yaw=yaw,
            )
        )
    return located
