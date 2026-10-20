"""Immutable FU001 scene inputs and the selected-asset contract.

The ``*_LEGACY`` values below describe the original DexVerse core-scene
evidence kept for provenance.  The canonical B320 runtime uses the reviewed
Poly Haven cabinet declared by ``SELECTED_*``; keeping the two namespaces
explicit prevents an old evidence bundle from silently becoming the selected
asset again.
"""

from __future__ import annotations

from pathlib import Path


TASK_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = TASK_DIR.parents[2]

TASK_ID = "B320-FU001"
RUNTIME_TASK_NAME = "b320_synchronized_double_cabinet_open"
SCENE_SCHEMA = "fu001_core_scene_v1"
SCENE_ROOT = "/World/FU001CoreScene"
CABINET_ROOT = f"{SCENE_ROOT}/DoubleCabinet44781"
HEAD_CAMERA_PATH = "/World/RecordCameras/head_camera"

CABINET_RELATIVE = Path(
    "taskdev/tasks/bench320_fu001_synchronized_double_cabinet_open/"
    "assets/dv_double_cabinet_44781/mobility.usd"
)
# The original 44781 bundle is retained as a historical reference only.  It is
# deliberately not the selected visual/physics source for the repaired FU001.
LEGACY_CABINET_RELATIVE = CABINET_RELATIVE

# Reviewed user-selected asset (Poly Haven Painted Wooden Cabinet).  Runtime
# components live in the canonical B320 asset tree so collection workers do
# not depend on a task-development absolute path; the byte-identical archive
# and source closure remain beside this task's provenance records.
SELECTED_ASSET_ID = "cabinet-polyhaven-painted"
SELECTED_ASSET_RUNTIME_RELATIVE = Path(
    "handson/handson/sim/envs/usd/b320_bimanual_benchmark/assets/"
    "polyhaven_painted_wooden_cabinet/runtime"
)
SELECTED_ASSET_MANIFEST_RELATIVE = Path(
    "handson/handson/sim/envs/usd/b320_bimanual_benchmark/assets/"
    "polyhaven_painted_wooden_cabinet/asset_manifest.json"
)
SELECTED_ASSET_SOURCE_RELATIVE = Path(
    "taskdev/tasks/bench320_fu001_synchronized_double_cabinet_open/"
    "assets/polyhaven_painted_wooden_cabinet/source"
)
SELECTED_ASSET_ARCHIVE_RELATIVE = SELECTED_ASSET_SOURCE_RELATIVE / (
    "painted_wooden_cabinet_4k.gltf.zip"
)
SELECTED_ASSET_ARCHIVE_SHA256 = (
    "dec8c52b933705c3332619faf6ec8618f5e8ca4ff57c24f25016e9876eba0ddf"
)
SELECTED_ASSET_SOURCE_URL = "https://polyhaven.com/a/painted_wooden_cabinet"
SELECTED_ASSET_LICENSE = "CC0 1.0"
SELECTED_ASSET_LICENSE_URL = "https://polyhaven.com/license"
SELECTED_ASSET_COMPONENTS = {
    "body": SELECTED_ASSET_RUNTIME_RELATIVE / "painted_cabinet_body.usd",
    "door_left": SELECTED_ASSET_RUNTIME_RELATIVE
    / "painted_cabinet_door_left.usd",
    "door_right": SELECTED_ASSET_RUNTIME_RELATIVE
    / "painted_cabinet_door_right.usd",
}

# The source cabinet is metre-scale and too tall for the Wuji-1 workspace at
# identity scale.  A uniform source-derived 0.55 scale keeps the full body on
# the 0.89 m tabletop while preserving the visual/collision proportions.
SELECTED_ASSET_SCALE = 0.55
SELECTED_RUNTIME_ORIENTATION_WXYZ = (0.5, 0.5, -0.5, -0.5)
SELECTED_SOURCE_TO_RUNTIME = "(x,y,z) -> (-z,-x,y)"
SELECTED_HINGE_AXIS_SOURCE = "+Y"
SELECTED_HINGE_AXIS_RUNTIME = "+Z"
SELECTED_BODY_SOURCE_MIN_Y_M = -0.0003708677
SELECTED_TABLE_CLEARANCE_M = 0.001
SELECTED_CABINET_ROOT_HOME = (
    -0.15,
    0.0,
    0.89 + SELECTED_TABLE_CLEARANCE_M
    - SELECTED_ASSET_SCALE * SELECTED_BODY_SOURCE_MIN_Y_M,
)
SELECTED_DOOR_SOURCE_PIVOTS = {
    # Source glTF node translations, in metres, before the runtime frame map.
    "cabinet_door_left": (-0.5682284236, 0.5324437022, 0.2114711404),
    "cabinet_door_right": (0.5677881241, 0.5334159732, 0.2114711404),
}
SELECTED_DOOR_HANDLE_SOURCE_POINTS = {
    # Measured points in the selected source mesh, useful for reach/camera
    # checks.  They are not semantic proxies or collision geometry.
    "cabinet_door_left": (-0.02730, 0.51730, 0.23302),
    "cabinet_door_right": (0.04300, 0.51346, 0.25926),
}
# Doors are constrained revolute links: they have no free XYZ translation.
# Their seeded reset variation is angular (10–30% of travel).  The source row
# requests ±30 mm XY furniture-root jitter, but this implementation authors
# the cabinet through a world-fixed articulation root.  IsaacLab cannot safely
# move that fixed root with a rigid-body setter, so the applied range is zero
# until a dedicated fixed-root reset primitive is validated in PhysX.  The
# source range and explicit exemption are recorded separately below; this is
# intentionally a pending gate, not a silently claimed randomization pass.
SELECTED_DOOR_TRANSLATION_RANDOMIZATION_RANGE_M = (0.0, 0.0, 0.0)
SELECTED_SOURCE_FIXTURE_TRANSLATION_RANDOMIZATION_RANGE_M = (
    0.030,
    0.030,
    0.0,
)
SELECTED_FIXTURE_TRANSLATION_RANDOMIZATION_RANGE_M = (
    0.0,
    0.0,
    0.0,
)
SELECTED_FIXTURE_TRANSLATION_RANDOMIZATION_MODE = "fixed_articulated_root_exempt"
ROBOT_RELATIVE = Path("handson/handson/sim/envs/xmls/rokae_wuji1_bimanual.xml")
SOURCE_ROWS_RELATIVE = Path(
    "taskdev/tasks/bench320_fu001_synchronized_double_cabinet_open/"
    "source_rows.authoritative.json"
)
ABC_ENV_RELATIVE = Path("handson/handson/sim/isaaclab/abc_collection_env.py")
CAMERA_PROFILE_RELATIVE = Path("handson/handson/sim/isaaclab/recording_cameras.py")

EXPECTED_SHA256 = {
    str(CABINET_RELATIVE): "9d3caff24dcf4b8465baa0d8186e588da33c1f4ce5cebd864358ababf15e9c46",
    str(CABINET_RELATIVE.parent / "configuration/mobility_base.usd"): "d49014e865264e889bd64182af6b493bb23cc251d87d1833c156fb72233dd93f",
    str(CABINET_RELATIVE.parent / "configuration/mobility_physics.usd"): "cb141c0188ba6c03d55ca5c9c49ba228372641e24575896d2826f2d4dd49127a",
    str(CABINET_RELATIVE.parent / "configuration/mobility_robot.usd"): "5ddec04dd3e11190a48a5a450ebf8bbfca0555b9c87675ae3daf288327e42b07",
    str(CABINET_RELATIVE.parent / "configuration/mobility_sensor.usd"): "75e6ea61cf2d8fda098865629dfbfa44beb7c2cde97726b4dd8a505eba92f8e1",
    str(CABINET_RELATIVE.parent / "config.yaml"): "96d93a6f2232a7aa605375b1e90bd6d11ef9cc83c751ff0b64c7fb170c305a67",
    str(CABINET_RELATIVE.parent / "meta.json"): "0dd38866c35edae121bfb73491e2432f3f904b25b4217852d2369b3280896f7c",
    str(CABINET_RELATIVE.parent / "configuration/materials/textures/texture_0.jpg"): "ac39178048c84f7b229abc893d1a2b0ae42f176c99dffb2da26c4e04111af8de",
    str(CABINET_RELATIVE.parent / "configuration/materials/textures/texture_1.jpg"): "f9b500da9e46f8c1afc63f81299b04c0d98a08bd00ead03a4b4df7e66e18860c",
    str(CABINET_RELATIVE.parent / "configuration/materials/textures/texture_2.png"): "fca29c933bfbec5df7928d871076df9ae70598d378ce42ce973ec95d4310189a",
    str(CABINET_RELATIVE.parent / "configuration/materials/textures/texture_3.jpg"): "7af4d6d5200361e3ca22c0b59f8e61d6dfa5386eea23f5ccee7acdb13f7b2c9c",
    str(CABINET_RELATIVE.parent / "configuration/materials/textures/texture_4.jpg"): "172ef612abe5c96b979e8b1fc45a29c4f2ae5d91aa093b37559deec49762651d",
    str(CABINET_RELATIVE.parent / "configuration/materials/textures/texture_5.png"): "fca29c933bfbec5df7928d871076df9ae70598d378ce42ce973ec95d4310189a",
    str(ROBOT_RELATIVE): "a425da3c71e547fc5f427fc172d0a18de1c51668b22d87ac6628ddba38bf6a63",
    str(SOURCE_ROWS_RELATIVE): "c1bf0990e8ec0ae651ae6d904c6b61e45796c8e11a670f6da782f4b23052e987",
    str(ABC_ENV_RELATIVE): "f78237a065e48e0b9ae354ea39c46346e547da774a830d7a5ec6572c20b95d1c",
    str(CAMERA_PROFILE_RELATIVE): "f6c67996290f451e9552d7612951b6a9c1502ca4614e0f9f7aa538fe11859ad7",
}

DOOR_JOINTS = ("target_joint_revolute", "joint_11")
DOOR_INITIAL_FRACTION = 0.15
CABINET_CENTER_XY_M = (0.40, 0.0)
CABINET_YAW_DEG = 0.0
CABINET_TABLE_CLEARANCE_M = 0.002

PHYSICS_HZ = 120
CONTROL_HZ = 30
SETTLE_STEPS = 120
ZERO_ACTION_SECONDS = 3.0
RENDER_HEIGHT = 480
RENDER_WIDTH = 640
HEAD_SHAPE = (RENDER_HEIGHT, RENDER_WIDTH, 3)

MAX_DOOR_DRIFT_RAD = 0.02
MAX_DOOR_SPEED_RAD_S = 0.08
MAX_ROBOT_CABINET_AABB_OVERLAP_M = 0.002

NATIVE_SOURCE_TASK_OBJECTS = (
    "c10_charger_plug",
    "c10_charging_cable",
    "c10_power_strip",
)
