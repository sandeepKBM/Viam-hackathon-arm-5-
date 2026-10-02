"""RGB-D samples that carry everything needed to use them correctly.

A color-image mask may only index a depth image that is aligned to the
color stream (same grid, color intrinsics). Viam's RealSense module does
that only with ``align_color_depth: true``; the old code assumed it and
sampled raw depth pixels under SAM masks. Every ``RGBDFrame`` therefore
records its camera model, units, alignment state, timestamp, and the camera
pose from the Viam frame system at capture, and ``validate_frames`` refuses
to hand out anything inconsistent.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

from components.transforms import T_from_json, T_to_json, invert, pose_to_T, rotation_angle_deg

PINHOLE_MODELS = ("", "none", "no_distortion", "pinhole")
KNOWN_MODELS = PINHOLE_MODELS + ("brown_conrady", "inverse_brown_conrady")
STATIONARY_TOL_DEG = 0.05
STATIONARY_TOL_MM = 0.5


class RGBDError(RuntimeError):
    pass


@dataclass(frozen=True)
class CameraModel:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    dist_model: str = ""
    coeffs: tuple = ()

    @staticmethod
    def from_props(props) -> "CameraModel":
        intr = props.intrinsic_parameters
        dist = getattr(props, "distortion_parameters", None)
        model = str(getattr(dist, "model", "") or "").strip().lower()
        coeffs = tuple(float(c) for c in (getattr(dist, "parameters", None) or ()))
        return CameraModel(
            fx=float(intr.focal_x_px),
            fy=float(intr.focal_y_px),
            cx=float(intr.center_x_px),
            cy=float(intr.center_y_px),
            width=int(intr.width_px),
            height=int(intr.height_px),
            dist_model=model,
            coeffs=coeffs,
        )

    @staticmethod
    def from_setup(setup) -> "CameraModel":
        i = setup.intrinsics
        return CameraModel(
            fx=i["fx"], fy=i["fy"], cx=i["cx"], cy=i["cy"],
            width=int(i["width"]), height=int(i["height"]),
            dist_model=setup.distortion["model"], coeffs=tuple(setup.distortion["coeffs"]),
        )

    def as_dict(self) -> dict:
        return {
            "fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy,
            "width": self.width, "height": self.height,
            "dist_model": self.dist_model, "coeffs": list(self.coeffs),
        }

    @property
    def is_pinhole(self) -> bool:
        return self.dist_model in PINHOLE_MODELS or not any(abs(c) > 0 for c in self.coeffs)

    def check_supported(self) -> None:
        if self.dist_model not in KNOWN_MODELS:
            raise RGBDError(f"unsupported distortion model {self.dist_model!r}; refusing to deproject")

    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    def normalized(self, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Pixel -> undistorted normalized image coordinates."""
        self.check_supported()
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        x = (u - self.cx) / self.fx
        y = (v - self.cy) / self.fy
        if self.is_pinhole:
            return x, y
        c = list(self.coeffs) + [0.0] * (5 - len(self.coeffs))
        if self.dist_model == "inverse_brown_conrady":
            # librealsense rs2_deproject_pixel_to_point: coefficients map the
            # distorted normalized point straight to the undistorted one.
            r2 = x * x + y * y
            f = 1 + c[0] * r2 + c[1] * r2 * r2 + c[4] * r2 * r2 * r2
            ux = x * f + 2 * c[2] * x * y + c[3] * (r2 + 2 * x * x)
            uy = y * f + 2 * c[3] * x * y + c[2] * (r2 + 2 * y * y)
            return ux, uy
        pts = np.stack([u.ravel(), v.ravel()], axis=1).reshape(-1, 1, 2)
        und = cv2.undistortPoints(pts, self.K(), np.array(c[:5]), criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 1e-9)).reshape(-1, 2)
        return und[:, 0].reshape(u.shape), und[:, 1].reshape(v.shape)

    def deproject(self, u, v, depth_mm) -> np.ndarray:
        """Pixels + z-depth (mm) -> (N, 3) camera-frame points (mm)."""
        x, y = self.normalized(u, v)
        z = np.asarray(depth_mm, dtype=float)
        return np.stack([x * z, y * z, z], axis=-1).reshape(-1, 3)

    def project(self, P_cam: np.ndarray) -> np.ndarray:
        """(N, 3) camera-frame points -> (N, 2) pixels (pinhole/brown_conrady)."""
        P = np.asarray(P_cam, dtype=float).reshape(-1, 3)
        z = np.where(np.abs(P[:, 2]) < 1e-9, 1e-9, P[:, 2])
        x, y = P[:, 0] / z, P[:, 1] / z
        if not self.is_pinhole:
            c = list(self.coeffs) + [0.0] * (5 - len(self.coeffs))
            if self.dist_model == "brown_conrady":
                r2 = x * x + y * y
                f = 1 + c[0] * r2 + c[1] * r2 * r2 + c[4] * r2 ** 3
                x, y = (
                    x * f + 2 * c[2] * x * y + c[3] * (r2 + 2 * x * x),
                    y * f + 2 * c[3] * x * y + c[2] * (r2 + 2 * y * y),
                )
            else:  # inverse_brown_conrady: invert the direct map iteratively
                tx, ty = x.copy(), y.copy()
                for _ in range(20):
                    ux, uy = self.normalized(tx * self.fx + self.cx, ty * self.fy + self.cy)
                    tx, ty = tx + (x - ux), ty + (y - uy)
                x, y = tx, ty
        return np.stack([x * self.fx + self.cx, y * self.fy + self.cy], axis=1)


@dataclass
class RGBDFrame:
    color: np.ndarray                  # (H, W, 3) BGR uint8
    depth_mm: np.ndarray               # (H, W) float32, 0 = invalid
    model: CameraModel
    aligned_to_color: bool
    depth_units_mm: float
    captured_at: Optional[float]       # epoch seconds from GetImages metadata
    T_world_cam: np.ndarray            # from the Viam frame system at capture
    T_world_flange: Optional[np.ndarray] = None
    joints_before: Optional[list] = None
    joints_after: Optional[list] = None
    color_source: str = "color"
    depth_source: str = "depth"
    depth_encoding: str = ""
    frame_id: str = ""
    extra: dict = field(default_factory=dict)

    def meta(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "model": self.model.as_dict(),
            "aligned_to_color": self.aligned_to_color,
            "depth_units_mm": self.depth_units_mm,
            "captured_at": self.captured_at,
            "T_world_cam": T_to_json(self.T_world_cam),
            "T_world_flange": None if self.T_world_flange is None else T_to_json(self.T_world_flange),
            "joints_before": self.joints_before,
            "joints_after": self.joints_after,
            "color_source": self.color_source,
            "depth_source": self.depth_source,
            "depth_encoding": self.depth_encoding,
            "color_shape": list(self.color.shape),
            "depth_shape": list(self.depth_mm.shape),
            "extra": self.extra,
        }


def validate_frames(frames: Sequence[RGBDFrame], expected: Optional[CameraModel] = None,
                    *, max_span_s: float = 10.0) -> list[str]:
    """Problems that make a frame set unusable (empty list = usable)."""
    problems: list[str] = []
    if not frames:
        return ["no RGB-D frames"]
    ref = frames[0]
    for i, f in enumerate(frames):
        tag = f"frame {i}"
        H, W = f.color.shape[:2]
        if f.depth_mm.shape[:2] != (H, W):
            problems.append(f"{tag}: depth {f.depth_mm.shape[:2]} != color {(H, W)}")
        if (f.model.width, f.model.height) != (W, H):
            problems.append(f"{tag}: intrinsics are for {f.model.width}x{f.model.height}, image is {W}x{H}")
        if not f.aligned_to_color:
            problems.append(f"{tag}: depth is not aligned to the color stream")
        if not (0.05 <= f.depth_units_mm <= 10.0):
            problems.append(f"{tag}: implausible depth units {f.depth_units_mm} mm/count")
        if f.model.dist_model not in KNOWN_MODELS:
            problems.append(f"{tag}: unknown distortion model {f.model.dist_model!r}")
        if f.captured_at is None:
            problems.append(f"{tag}: no capture timestamp")
        if f.model != ref.model:
            problems.append(f"{tag}: camera model changed within the sample set")
        if f.color_source != ref.color_source or f.depth_source != ref.depth_source:
            problems.append(f"{tag}: stream identity changed within the sample set")
        if f.joints_before is not None and f.joints_after is not None:
            dj = max(abs(a - b) for a, b in zip(f.joints_before, f.joints_after))
            if dj > STATIONARY_TOL_DEG:
                problems.append(f"{tag}: arm moved {dj:.2f} deg during capture")
        D = invert(ref.T_world_cam) @ f.T_world_cam
        if np.linalg.norm(D[:3, 3]) > STATIONARY_TOL_MM or rotation_angle_deg(np.eye(3), D[:3, :3]) > STATIONARY_TOL_DEG:
            problems.append(f"{tag}: camera pose differs from frame 0 (arm not stationary)")
    if expected is not None and ref.model != expected:
        problems.append(f"camera model {ref.model.as_dict()} != calibrated {expected.as_dict()}")
    ts = [f.captured_at for f in frames if f.captured_at is not None]
    if len(ts) >= 2 and (max(ts) - min(ts)) > max_span_s:
        problems.append(f"sample set spans {max(ts) - min(ts):.1f} s (> {max_span_s} s)")
    return problems


def depth_edges(depth_mm: np.ndarray, jump_mm: float = 15.0, invalid_borders: bool = False) -> np.ndarray:
    """Boolean map of depth jumps between valid neighbours (optionally also
    valid/invalid borders; random stereo holes are not edges)."""
    d = np.asarray(depth_mm, dtype=np.float32)
    valid = d > 0
    edge = np.zeros(d.shape, dtype=bool)
    for dy, dx in ((0, 1), (1, 0)):
        a = d[: d.shape[0] - dy, : d.shape[1] - dx]
        b = d[dy:, dx:]
        va = valid[: d.shape[0] - dy, : d.shape[1] - dx]
        vb = valid[dy:, dx:]
        jump = va & vb & (np.abs(a - b) > jump_mm)
        if invalid_borders:
            jump |= va ^ vb
        edge[: d.shape[0] - dy, : d.shape[1] - dx] |= jump
        edge[dy:, dx:] |= jump
    return edge


def edge_alignment_score(mask: np.ndarray, depth_mm: np.ndarray, radius_px: int = 3,
                         jump_mm: float = 15.0) -> Optional[float]:
    """Fraction of the mask outline that lies on a depth discontinuity.

    For a raised object on a table, an aligned depth map has a depth jump on
    the color silhouette; misregistered depth puts the jump somewhere else.
    Returns None when the outline is too short to judge.
    """
    m = (np.asarray(mask) > 0).astype(np.uint8)
    outline = m - cv2.erode(m, np.ones((3, 3), np.uint8))
    n = int(outline.sum())
    if n < 40:
        return None
    edges = depth_edges(depth_mm, jump_mm).astype(np.uint8)
    k = 2 * radius_px + 1
    near = cv2.dilate(edges, np.ones((k, k), np.uint8)) > 0
    return float(near[outline > 0].mean())


# ---------------------------------------------------------------------------
# Live capture (Viam)
# ---------------------------------------------------------------------------


def _decode_color(img) -> np.ndarray:
    buf = np.frombuffer(img.data, np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RGBDError(f"could not decode color image ({img.mime_type})")
    return bgr


def _split(images) -> tuple:
    color = depth = None
    csrc = dsrc = denc = ""
    for im in images:
        name = (getattr(im, "name", "") or "").lower()
        mime = (im.mime_type or "").lower()
        if "dep" in mime or name == "depth":
            depth = np.asarray(im.bytes_to_depth_array(), dtype=np.float32)
            dsrc, denc = name or "depth", mime
        elif color is None:
            color = _decode_color(im)
            csrc = name or "color"
    return color, depth, csrc, dsrc, denc


async def frame_system_T(machine, frame: str, to: str = "world") -> np.ndarray:
    """Pose of ``frame``'s origin in ``to``, from the Viam frame system."""
    from viam.proto.common import Pose, PoseInFrame

    res = await machine.transform_pose(
        PoseInFrame(reference_frame=frame, pose=Pose(x=0, y=0, z=0, o_x=0, o_y=0, o_z=1, theta=0)),
        to,
    )
    return pose_to_T(res.pose)


async def camera_mount_from_frame_system(machine, camera_name: str, arm_name: str):
    """(T_parent_cam, parent_name) as configured in the live frame system."""
    parts = await machine.get_frame_system_config()
    for part in parts:
        fr = part.frame
        if fr.reference_frame == camera_name:
            pif = fr.pose_in_observer_frame
            return pose_to_T(pif.pose), pif.reference_frame
    return None, None


async def capture_rgbd_set(
    machine,
    *,
    camera_name: str,
    arm_name: str,
    n_frames: int = 3,
    aligned_to_color: bool,
    depth_units_mm: float = 1.0,
    settle_s: float = 0.2,
) -> tuple[list[RGBDFrame], object]:
    """Capture ``n_frames`` synchronized color+depth pairs with the arm
    stationary. ``aligned_to_color`` must come from the calibration record
    (the camera API does not report it). Returns (frames, properties)."""
    from viam.components.arm import Arm
    from viam.components.camera import Camera

    cam = Camera.from_robot(machine, camera_name)
    arm = Arm.from_robot(machine, arm_name)
    props = await cam.get_properties()
    model = CameraModel.from_props(props)
    frames: list[RGBDFrame] = []
    for i in range(n_frames):
        if settle_s:
            await _sleep(settle_s)
        j0 = list((await arm.get_joint_positions()).values)
        images, meta = await cam.get_images()
        j1 = list((await arm.get_joint_positions()).values)
        T_wc = await frame_system_T(machine, camera_name)
        T_wf = await frame_system_T(machine, arm_name)
        color, depth, csrc, dsrc, denc = _split(images)
        if color is None or depth is None:
            raise RGBDError("camera did not return both color and depth")
        ts = None
        cap = getattr(meta, "captured_at", None)
        if cap is not None and (cap.seconds or cap.nanos):
            ts = cap.seconds + cap.nanos * 1e-9
        frames.append(
            RGBDFrame(
                color=color,
                depth_mm=depth * float(depth_units_mm),
                model=model,
                aligned_to_color=bool(aligned_to_color),
                depth_units_mm=float(depth_units_mm),
                captured_at=ts,
                T_world_cam=T_wc,
                T_world_flange=T_wf,
                joints_before=j0,
                joints_after=j1,
                color_source=csrc,
                depth_source=dsrc,
                depth_encoding=denc,
                frame_id=f"cap{int(time.time() * 1000)}_{i}",
            )
        )
    return frames, props


async def _sleep(s: float) -> None:
    import asyncio

    await asyncio.sleep(s)


def reported_extrinsics(props) -> Optional[dict]:
    ext = getattr(props, "extrinsic_parameters", None)
    if ext is None or not hasattr(props, "HasField") or not props.HasField("extrinsic_parameters"):
        return None
    t = getattr(ext, "translation", None)
    if t is None:
        return None
    return {"translation_mm": [float(t.x), float(t.y), float(t.z)]}


# ---------------------------------------------------------------------------
# Evidence (save / load for replay)
# ---------------------------------------------------------------------------


def save_frames(run_dir: Path, frames: Sequence[RGBDFrame], prefix: str = "obs") -> list[dict]:
    run_dir.mkdir(parents=True, exist_ok=True)
    metas = []
    for i, f in enumerate(frames):
        stem = f"{prefix}_{i:02d}"
        cv2.imwrite(str(run_dir / f"{stem}_color.png"), f.color)
        d = np.clip(np.rint(f.depth_mm), 0, 65535).astype(np.uint16)
        cv2.imwrite(str(run_dir / f"{stem}_depth_mm.png"), d)
        viz = cv2.applyColorMap(cv2.convertScaleAbs(d, alpha=255.0 / max(1.0, float(d.max()))), cv2.COLORMAP_JET)
        cv2.imwrite(str(run_dir / f"{stem}_depth_viz.png"), viz)
        meta = f.meta()
        meta["files"] = {"color": f"{stem}_color.png", "depth_mm": f"{stem}_depth_mm.png"}
        (run_dir / f"{stem}_meta.json").write_text(json.dumps(meta, indent=2))
        metas.append(meta)
    return metas


def load_frames(run_dir: Path, prefix: str = "obs") -> list[RGBDFrame]:
    frames = []
    for meta_path in sorted(Path(run_dir).glob(f"{prefix}_*_meta.json")):
        meta = json.loads(meta_path.read_text())
        color = cv2.imread(str(meta_path.parent / meta["files"]["color"]), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(meta_path.parent / meta["files"]["depth_mm"]), cv2.IMREAD_UNCHANGED)
        if color is None or depth is None:
            raise RGBDError(f"missing image files for {meta_path.name}")
        m = meta["model"]
        frames.append(
            RGBDFrame(
                color=color,
                depth_mm=depth.astype(np.float32),
                model=CameraModel(m["fx"], m["fy"], m["cx"], m["cy"], int(m["width"]), int(m["height"]),
                                  m.get("dist_model", ""), tuple(m.get("coeffs") or ())),
                aligned_to_color=bool(meta["aligned_to_color"]),
                depth_units_mm=float(meta["depth_units_mm"]),
                captured_at=meta.get("captured_at"),
                T_world_cam=T_from_json(meta["T_world_cam"]),
                T_world_flange=None if meta.get("T_world_flange") is None else T_from_json(meta["T_world_flange"]),
                joints_before=meta.get("joints_before"),
                joints_after=meta.get("joints_after"),
                color_source=meta.get("color_source", "color"),
                depth_source=meta.get("depth_source", "depth"),
                depth_encoding=meta.get("depth_encoding", ""),
                frame_id=meta.get("frame_id", meta_path.stem),
                extra=meta.get("extra") or {},
            )
        )
    if not frames:
        raise RGBDError(f"no saved frames with prefix {prefix!r} in {run_dir}")
    return frames


def angle_between_deg(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))
