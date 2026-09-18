from typing import List, Optional, Sequence, Tuple

from viam.proto.common import Pose

from components.constants import FLOOR_Z, MAX_Y, MIN_Z, WORKSPACE_CORNERS

Polygon = Sequence[Tuple[float, float]]


def point_in_polygon(x: float, y: float, poly: Optional[Polygon] = None) -> bool:
    poly = WORKSPACE_CORNERS if poly is None else poly
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def in_workspace(x: float, y: float, poly: Optional[Polygon] = None) -> bool:
    return point_in_polygon(x, y, poly)


def assert_in_workspace(x: float, y: float, poly: Optional[Polygon] = None) -> None:
    if y > MAX_Y:
        raise ValueError(
            f"target y={y:.1f} is above max Y {MAX_Y:.1f}; refusing to move there"
        )
    if not in_workspace(x, y, poly):
        region: List = list(WORKSPACE_CORNERS if poly is None else poly)
        raise ValueError(
            f"target (x={x:.1f}, y={y:.1f}) is outside the workspace {region}"
        )


def clamp_z(z: float, floor: float = MIN_Z) -> float:
    if z < floor:
        print(f"  [safety] clamping z {z:.1f} -> {floor:.1f} (floor)")
        return floor
    return z


def make_pose(
    x: float,
    y: float,
    z: float,
    o_x: float = 0.0,
    o_y: float = 0.0,
    o_z: float = -1.0,
    theta: float = 0.0,
    poly: Optional[Polygon] = None,
    floor: float = MIN_Z,
) -> Pose:
    assert_in_workspace(x, y, poly)
    return Pose(x=x, y=y, z=clamp_z(z, floor), o_x=o_x, o_y=o_y, o_z=o_z, theta=theta)
