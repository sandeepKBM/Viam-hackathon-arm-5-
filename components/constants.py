import os
from typing import Dict, List, Tuple

# End-effector is never commanded below this height (mm).
FLOOR_Z = float(os.environ.get("FLOOR_Z", 179.75673))
MIN_Z = float(os.environ.get("MIN_Z", FLOOR_Z))
# World-frame Z of the table surface from depth (block tops sit above this).
TABLE_Z = float(os.environ.get("TABLE_Z", 0.0))

# Taught workspace corners (mm), perimeter order: BL -> TL -> TR -> BR.
WORKSPACE_CORNERS: List[Tuple[float, float]] = [
    (-77.12904129235005, 489.45863666072296),
    (459.0502534311783, 506.75235394316746),
    (476.3606984754242, -286.36154933311184),
    (144.83694658972408, -303.3956150294798),
]

MAX_Y = float(os.environ.get("MAX_Y", max(y for _, y in WORKSPACE_CORNERS)))

HOME_JOINTS: List[float] = [
    -17.287645921409155,
    21.317320927933277,
    -35.8769517287989,
    0.467687971989073,
    -59.234583350503804,
    -15.887329075509601,
]
HOME_POSE: Dict[str, float] = dict(
    x=281.235302845453,
    y=-86.8692478748763,
    z=532.6488012460089,
    o_x=-0.03189537706111025,
    o_y=0.01726517715312809,
    o_z=-0.999342082862521,
    theta=-26.77995529670437,
)

BIN_1_JOINTS: List[float] = [
    33.80570027808712,
    -34.69020977560377,
    -72.7538841332227,
    -2.4608277251661494,
    -37.60674494742882,
    36.91340219949707,
]
BIN_1_POSE: Dict[str, float] = dict(
    x=502.74634314593396,
    y=333.79597618396895,
    z=351.78864024602717,
    o_x=0.021566090583248657,
    o_y=-0.01709543846470806,
    o_z=-0.9996212531357339,
    theta=142.76045302501078,
)

BIN_2_JOINTS: List[float] = [
    24.40294255530387,
    -29.079711391342293,
    -61.12990620366898,
    -2.4090819985707213,
    -30.831552460613015,
    26.459912080461155,
]
BIN_2_POSE: Dict[str, float] = dict(
    x=496.8360135542832,
    y=223.27703363469834,
    z=333.4097342997685,
    o_x=0.02862079019038093,
    o_y=-0.01067939251907734,
    o_z=-0.9995332915637689,
    theta=159.53911912833792,
)

TAUGHT_JOINTS = {
    "home": HOME_JOINTS,
    "bin1": BIN_1_JOINTS,
    "bin2": BIN_2_JOINTS,
}

COLOR_BINS = {
    "red": "bin1",
    "yellow": "bin2",
}

TRAVEL_Z = float(os.environ.get("TRAVEL_Z", HOME_POSE["z"]))
PICK_ORIENTATION = {
    "o_x": HOME_POSE["o_x"],
    "o_y": HOME_POSE["o_y"],
    "o_z": HOME_POSE["o_z"],
    "theta": HOME_POSE["theta"],
}
