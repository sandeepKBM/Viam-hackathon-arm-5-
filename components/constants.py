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

DROPOFF_JOINTS: List[float] = [
    87.42271941363306,
    -61.25800640155257,
    -28.04985376074048,
    -0.18138427784721722,
    122.00295631341046,
    2.395568896184674,
]
DROPOFF_POSE: Dict[str, float] = dict(
    x=14.341128502784034,
    y=323.93219749948594,
    z=111.26688164189298,
    o_x=0.04226360647420724,
    o_y=0.9988966358085766,
    o_z=-0.020476780462330646,
    theta=-177.50570517189274,
)

HANDOFF_JOINTS: List[float] = [
    56.7004397344565,
    -42.54730081185502,
    -30.55154885558834,
    -81.23994338447831,
    92.57482672722206,
    -4.961535644235534,
]
HANDOFF_POSE: Dict[str, float] = dict(
    x=110.27366250853495,
    y=329.73063061438467,
    z=204.55635869839605,
    o_x=-0.7384054381154712,
    o_y=0.6742446013071756,
    o_z=0.012313674091332993,
    theta=-173.10714195349388,
)

TAUGHT_JOINTS = {
    "home": HOME_JOINTS,
    "bin1": BIN_1_JOINTS,
    "bin2": BIN_2_JOINTS,
    "dropoff": DROPOFF_JOINTS,
    "handoff": HANDOFF_JOINTS,
}
TAUGHT_POSES = {
    "home": HOME_POSE,
    "bin1": BIN_1_POSE,
    "bin2": BIN_2_POSE,
    "dropoff": DROPOFF_POSE,
    "handoff": HANDOFF_POSE,
}

PLACE_ALIASES = {
    "bin1": "bin1",
    "b1": "bin1",
    "1": "bin1",
    "binone": "bin1",
    "firstbin": "bin1",
    "bin2": "bin2",
    "b2": "bin2",
    "2": "bin2",
    "bintwo": "bin2",
    "secondbin": "bin2",
    "dropoff": "dropoff",
    "drop": "dropoff",
    "dropout": "dropoff",
    "dropit": "dropoff",
    "dropitoff": "dropoff",
    "handoff": "handoff",
    "hand": "handoff",
    "handover": "handoff",
    "give": "handoff",
    "giveme": "handoff",
    "givehim": "handoff",
    "giveher": "handoff",
    "givetome": "handoff",
    "givetohim": "handoff",
    "givetoher": "handoff",
    "handitover": "handoff",
    "handittohim": "handoff",
    "handittome": "handoff",
}


def normalize_place(value: str) -> str | None:
    raw = "".join(ch for ch in str(value).lower() if ch.isalnum())
    return PLACE_ALIASES.get(raw)

# These sit on the table; pick TCP at the taught floor instead of lid depth.
FLOOR_PICK_OBJECTS = ("bottle", "can", "pen")
# Extra descent below MIN_Z for thin objects (mm).
PEN_PICK_Z_OFFSET = float(os.environ.get("PEN_PICK_Z_OFFSET", 7))
HSV_OBJECTS = ("red", "yellow")
PICK_OBJECTS = ("red", "yellow", "can", "cup", "airpods", "pen", "bottle")
OBJECT_ALIASES = {
    "red": "red",
    "redblock": "red",
    "redblocks": "red",
    "yellow": "yellow",
    "yellowblock": "yellow",
    "yellowblocks": "yellow",
    "can": "can",
    "soda": "can",
    "sodacan": "can",
    "coke": "can",
    "cocacola": "can",
    "cup": "cup",
    "mug": "cup",
    "airpods": "airpods",
    "airpod": "airpods",
    "airpodscase": "airpods",
    "earbuds": "airpods",
    "earpods": "airpods",
    "pen": "pen",
    "pens": "pen",
    "bottle": "bottle",
    "bottles": "bottle",
    "waterbottle": "bottle",
}
# Detect phrases sent to Moondream for each pick object.
OBJECT_DETECT = {
    "can": ("soda can", "can"),
    "cup": ("cup",),
    "airpods": ("airpods", "earbuds case"),
    "pen": ("pen",),
    "bottle": ("bottle",),
}
OBJECT_BINS = {
    "red": "bin1",
    "yellow": "bin2",
    "can": "bin1",
    "cup": "bin2",
    "airpods": "bin1",
    "pen": "bin2",
    "bottle": "bin1",
}
COLOR_BINS = OBJECT_BINS


def normalize_object(value: str) -> str | None:
    raw = "".join(ch for ch in str(value).lower() if ch.isalnum())
    return OBJECT_ALIASES.get(raw)

TRAVEL_Z = float(os.environ.get("TRAVEL_Z", HOME_POSE["z"]))
PICK_ORIENTATION = {
    "o_x": HOME_POSE["o_x"],
    "o_y": HOME_POSE["o_y"],
    "o_z": HOME_POSE["o_z"],
    "theta": HOME_POSE["theta"],
}
