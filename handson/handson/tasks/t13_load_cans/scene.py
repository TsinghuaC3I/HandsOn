"""Exact assets, scene authoring, and pure semantics for B320-BC012."""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import numpy as np


TASK_ID: Final = "B320-BC012"
TASK_NAME: Final = "t13_load_cans"
TASK_ROOT: Final = "/World/Task"
LEFT_CAN_NAME: Final = "bc012_left_can"
RIGHT_CAN_NAME: Final = "bc012_right_can"
BIN_NAME: Final = "bc012_open_bin"
CAN_NAMES: Final = (LEFT_CAN_NAME, RIGHT_CAN_NAME)

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[2]
ASSET_ROOT: Final = REPOSITORY_ROOT / "sim/envs/usd/t13_load_cans"
CAN_USD: Final = ASSET_ROOT / "tomato_soup_can" / "textured.usd"
CAN_MESH_USD: Final = ASSET_ROOT / "tomato_soup_can" / "Props" / "instanceable_meshes.usd"
CAN_TEXTURE: Final = ASSET_ROOT / "tomato_soup_can" / "textures" / "texture_map.png"
OPEN_BIN_GLB: Final = ASSET_ROOT / "open_bin" / "dexverse_authored_open_bin.glb"

EXPECTED_SHA256: Final = {
    CAN_USD: "c54c63ab31dc0e369ccc27d368f0fa60d1e83c2d358ae207022d5a1317d30f26",
    CAN_MESH_USD: "1b8eb20b782d4907076b4cb927e5f88ea238c9229ad375ab2a616e564f2f4ed4",
    CAN_TEXTURE: "33183a894eb75ae04ae5188a3ae689fe8d06ec6b9f9624da1bf853c2a25e70d2",
    OPEN_BIN_GLB: "eabe5ab04f5c3ec2f80cd81e9deba260eeb35a671b0a31a931e8a7a62440f1e6",
}

TABLE_TOP_Z: Final = 0.89
CAN_SUPPORT_CLEARANCE: Final = 0.002
BIN_SUPPORT_CLEARANCE: Final = 0.002
TEMPLATE_LAYOUT_TRANSLATION_XY: Final = np.asarray([0.0, 0.0], dtype=np.float64)
CAN_ANCHOR_XY: Final = {
    LEFT_CAN_NAME: np.asarray([-0.43, 0.28], dtype=np.float64)
    + TEMPLATE_LAYOUT_TRANSLATION_XY,
    RIGHT_CAN_NAME: np.asarray([-0.43, -0.28], dtype=np.float64)
    + TEMPLATE_LAYOUT_TRANSLATION_XY,
}
BIN_ANCHOR_XY: Final = (
    np.asarray([-0.40, 0.0], dtype=np.float64)
    + TEMPLATE_LAYOUT_TRANSLATION_XY
)
CAN_XY_JITTER: Final = 0.035
BIN_XY_JITTER: Final = 0.025
CAN_YAW_DEGREES: Final = 12.0
RESET_MAX_ATTEMPTS: Final = 128
CAN_CAN_CLEARANCE: Final = 0.11
CAN_BIN_CLEARANCE: Final = 0.095
HAND_CLEARANCE: Final = 0.10
WORKSPACE_LOW_XY: Final = np.asarray([-0.68, -0.44], dtype=np.float64)
WORKSPACE_HIGH_XY: Final = np.asarray([0.16, 0.44], dtype=np.float64)

OPEN_BIN_SOURCE_BOUNDS: Final = np.asarray(
    [[-0.1899999976, 0.0, -0.1400000006], [0.1899999976, 0.1500000060, 0.1400000006]],
    dtype=np.float64,
)
OPEN_BIN_COLLISION_PANELS: Final = {
    "bottom": ((0.0, 0.0, 0.0125), (0.38, 0.28, 0.025)),
    "front": ((0.0, -0.1275, 0.075), (0.38, 0.025, 0.15)),
    "back": ((0.0, 0.1275, 0.075), (0.38, 0.025, 0.15)),
    "left": ((-0.1775, 0.0, 0.075), (0.025, 0.23, 0.15)),
    "right": ((0.1775, 0.0, 0.075), (0.025, 0.23, 0.15)),
}
PARTITION_DIVIDER: Final = ((0.0, 0.0, 0.075), (0.36, 0.012, 0.14))
PARTITION_CENTERS_LOCAL: Final = np.asarray(
    [[0.0, 0.065, 0.075], [0.0, -0.065, 0.075]],
    dtype=np.float64,
)
# Partition centers are ordered by their physical side.  The task deliberately
# cross-loads the cans: the positive-Y (left) can targets the negative-Y
# partition, while the negative-Y (right) can targets the positive-Y partition.
TARGET_PARTITION_INDEX_BY_SIDE: Final = {"left": 1, "right": 0}
GOAL_ZONES: Final = {
    "left": ((0.0, 0.065, 0.035), (0.33, 0.095, 0.010)),
    "right": ((0.0, -0.065, 0.035), (0.33, 0.095, 0.010)),
}

UPRIGHT_COS: Final = float(math.cos(math.radians(20.0)))
MAX_LINEAR_SPEED: Final = 0.04
MAX_ANGULAR_SPEED: Final = 0.50
HAND_CLEAR_DISTANCE: Final = 0.075
HAND_CONTACT_FORCE_THRESHOLD: Final = 0.05
SUPPORT_CONTACT_FORCE_THRESHOLD: Final = 1.0e-4
SYNC_WINDOW_SECONDS: Final = 0.8
STABILITY_STEPS: Final = 15
GRASP_D6_AXES: Final = (
    "transX",
    "transY",
    "transZ",
    "rotX",
    "rotY",
    "rotZ",
)
GRASP_D6_FREE_LIMIT_BY_AXIS: Final = {
    axis: (1000.0 if axis.startswith("trans") else 179.0)
    for axis in GRASP_D6_AXES
}
GRASP_D6_LOCK_LIMIT_BY_AXIS: Final = {
    axis: (1.0e-5 if axis.startswith("trans") else 1.0e-3)
    for axis in GRASP_D6_AXES
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_asset_contract() -> list[str]:
    errors: list[str] = []
    for path, expected in EXPECTED_SHA256.items():
        if not path.is_file():
            errors.append(f"missing exact BC012 asset: {path}")
            continue
        observed = sha256_file(path)
        if observed != expected:
            errors.append(f"asset hash mismatch for {path}: {observed}")
    return errors


def usd_prim_identifier(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_]", "_", str(value)) or "mesh"
    return sanitized if re.match(r"^[A-Za-z_]", sanitized) else f"Mesh_{sanitized}"


def _quat_wxyz_matrix(quaternion: Sequence[float]) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm([w, x, y, z]))
    if norm <= 1.0e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = np.asarray([w, x, y, z], dtype=np.float64) / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def pose_matrix(pose_wxyz: Sequence[float]) -> np.ndarray:
    pose = np.asarray(pose_wxyz, dtype=np.float64).reshape(7)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = _quat_wxyz_matrix(pose[3:7])
    matrix[:3, 3] = pose[:3]
    return matrix


def matrix_to_quat_wxyz(matrix: Sequence[Sequence[float]]) -> np.ndarray:
    """Convert a proper rotation matrix to a normalized wxyz quaternion."""

    rotation = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.asarray(
            [
                0.25 * scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            ],
            dtype=np.float64,
        )
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            scale = math.sqrt(
                1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]
            ) * 2.0
            quaternion = np.asarray(
                [
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                    0.25 * scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                ],
                dtype=np.float64,
            )
        elif index == 1:
            scale = math.sqrt(
                1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]
            ) * 2.0
            quaternion = np.asarray(
                [
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    0.25 * scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                ],
                dtype=np.float64,
            )
        else:
            scale = math.sqrt(
                1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]
            ) * 2.0
            quaternion = np.asarray(
                [
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    0.25 * scale,
                ],
                dtype=np.float64,
            )
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1.0e-12:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    quaternion /= norm
    if quaternion[0] < 0.0:
        quaternion *= -1.0
    return quaternion


def joint_local_frame_pair(
    body0_world: Sequence[Sequence[float]],
    body1_world: Sequence[Sequence[float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Return coincident joint frames in each body's local coordinates."""

    body0 = np.asarray(body0_world, dtype=np.float64).reshape(4, 4)
    body1 = np.asarray(body1_world, dtype=np.float64).reshape(4, 4)
    if not np.all(np.isfinite(body0)) or not np.all(np.isfinite(body1)):
        raise ValueError("BC012 joint bodies require finite world transforms")
    joint_world = body0.copy()
    return np.linalg.inv(body0) @ joint_world, np.linalg.inv(body1) @ joint_world


def _usd_world_pose_matrix(prim: Any, cache: Any) -> np.ndarray:
    transform = cache.GetLocalToWorldTransform(prim)
    translation = transform.ExtractTranslation()
    rotation = transform.ExtractRotationQuat()
    imaginary = rotation.GetImaginary()
    pose = np.asarray(
        [
            float(translation[0]),
            float(translation[1]),
            float(translation[2]),
            float(rotation.GetReal()),
            float(imaginary[0]),
            float(imaginary[1]),
            float(imaginary[2]),
        ],
        dtype=np.float64,
    )
    return pose_matrix(pose)


def partition_centers_world(bin_pose: Sequence[float]) -> np.ndarray:
    matrix = pose_matrix(bin_pose)
    homogeneous = np.column_stack(
        [PARTITION_CENTERS_LOCAL, np.ones(2, dtype=np.float64)]
    )
    return (matrix @ homogeneous.T).T[:, :3]


def _can_metrics(
    pose: Sequence[float],
    velocity: Sequence[float],
    bin_inverse: np.ndarray,
    *,
    positive_y: bool,
    local_center: Sequence[float],
    supported: bool,
    hand_clear: bool,
    hand_contact_force: float,
    owned: bool,
) -> dict[str, Any]:
    can_matrix = pose_matrix(pose)
    center_world = can_matrix @ np.asarray([*local_center, 1.0], dtype=np.float64)
    center_local = (bin_inverse @ center_world)[:3]
    axis_world = can_matrix[:3, 2]
    bin_up = bin_inverse[:3, :3].T[:, 2]
    upright_alignment = float(np.dot(axis_world, bin_up))
    velocity_array = np.asarray(velocity, dtype=np.float64).reshape(6)
    linear_speed = float(np.linalg.norm(velocity_array[:3]))
    angular_speed = float(np.linalg.norm(velocity_array[3:]))

    x_inside = abs(float(center_local[0])) <= 0.132
    if positive_y:
        assigned_y = 0.044 <= float(center_local[1]) <= 0.086
    else:
        assigned_y = -0.086 <= float(center_local[1]) <= -0.044
    z_inside = 0.045 <= float(center_local[2]) <= 0.125
    contained = bool(x_inside and assigned_y and z_inside)
    upright = bool(upright_alignment >= UPRIGHT_COS)
    stable = bool(
        linear_speed <= MAX_LINEAR_SPEED
        and angular_speed <= MAX_ANGULAR_SPEED
    )
    released = bool(
        not owned
        and hand_clear
        and float(hand_contact_force) <= HAND_CONTACT_FORCE_THRESHOLD
    )
    ready = bool(contained and upright and supported and stable and released)
    return {
        "center_local": center_local,
        "contained": contained,
        "upright": upright,
        "upright_alignment": upright_alignment,
        "supported": bool(supported),
        "stable": stable,
        "released": released,
        "hand_clear": bool(hand_clear),
        "hand_contact_force": float(hand_contact_force),
        "owned": bool(owned),
        "linear_speed": linear_speed,
        "angular_speed": angular_speed,
        "ready": ready,
    }


def evaluate_bin_load(
    left_pose: Sequence[float],
    right_pose: Sequence[float],
    bin_pose: Sequence[float],
    *,
    left_velocity: Sequence[float] = (0.0,) * 6,
    right_velocity: Sequence[float] = (0.0,) * 6,
    can_local_center: Sequence[float] = (0.0, 0.0, 0.05),
    left_supported: bool = False,
    right_supported: bool = False,
    left_hand_clear: bool = False,
    right_hand_clear: bool = False,
    left_hand_contact_force: float = math.inf,
    right_hand_contact_force: float = math.inf,
    left_owned: bool = False,
    right_owned: bool = False,
    left_release_frame: int | None = None,
    right_release_frame: int | None = None,
    control_hz: float = 30.0,
) -> dict[str, Any]:
    bin_inverse = np.linalg.inv(pose_matrix(bin_pose))
    left = _can_metrics(
        left_pose,
        left_velocity,
        bin_inverse,
        positive_y=False,
        local_center=can_local_center,
        supported=left_supported,
        hand_clear=left_hand_clear,
        hand_contact_force=left_hand_contact_force,
        owned=left_owned,
    )
    right = _can_metrics(
        right_pose,
        right_velocity,
        bin_inverse,
        positive_y=True,
        local_center=can_local_center,
        supported=right_supported,
        hand_clear=right_hand_clear,
        hand_contact_force=right_hand_contact_force,
        owned=right_owned,
    )
    sync_frames = max(1, int(math.ceil(SYNC_WINDOW_SECONDS * float(control_hz))))
    release_delta = (
        abs(int(left_release_frame) - int(right_release_frame))
        if left_release_frame is not None and right_release_frame is not None
        else math.inf
    )
    synchronized = bool(release_delta <= sync_frames)
    success_now = bool(left["ready"] and right["ready"] and synchronized)
    metrics_vector = np.asarray(
        [
            *left["center_local"],
            *right["center_local"],
            left["upright_alignment"],
            right["upright_alignment"],
            left["linear_speed"],
            right["linear_speed"],
            float(release_delta if math.isfinite(release_delta) else sync_frames + 1),
            float(success_now),
        ],
        dtype=np.float64,
    )
    return {
        "left": left,
        "right": right,
        "release_delta_frames": release_delta,
        "sync_window_frames": sync_frames,
        "synchronized": synchronized,
        "success_now": success_now,
        "metrics_vector": metrics_vector,
    }


def _material(stage: Any, path: str, color: tuple[float, float, float], opacity: float = 1.0) -> Any:
    from pxr import Sdf, UsdShade

    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.42)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.04)
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _bind_material(prim: Any, material: Any) -> None:
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def _set_translate(prim: Any, values: Sequence[float]) -> None:
    from pxr import Gf, UsdGeom

    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(*np.asarray(values, dtype=np.float64).reshape(3))
    )


def _can_source_inventory() -> dict[str, Any]:
    from pxr import Usd, UsdGeom

    source_stage = Usd.Stage.Open(str(CAN_USD), load=Usd.Stage.LoadAll)
    if source_stage is None:
        raise RuntimeError(f"could not open exact BC012 can USD: {CAN_USD}")
    root = source_stage.GetDefaultPrim()
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    bounds = cache.ComputeWorldBound(root).ComputeAlignedRange()
    low = np.asarray(bounds.GetMin(), dtype=np.float64)
    high = np.asarray(bounds.GetMax(), dtype=np.float64)
    extent = high - low
    if (
        not root.IsValid()
        or not np.all(np.isfinite(extent))
        or not (0.04 < extent[0] < 0.12)
        or not (0.04 < extent[1] < 0.12)
        or not (0.07 < extent[2] < 0.16)
    ):
        raise RuntimeError(f"invalid exact BC012 can bounds: {low.tolist()}..{high.tolist()}")
    return {"low": low, "high": high, "extent": extent, "center": 0.5 * (low + high)}


def spawn_bc012_assets(stage: Any) -> dict[str, Any]:
    """Author exact visuals and analytic physics before the first PhysX reset."""

    errors = validate_asset_contract()
    if errors:
        raise RuntimeError("; ".join(errors))

    import trimesh
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, Vt

    inventory = _can_source_inventory()
    can_roots: dict[str, str] = {}
    for name, anchor_xy in CAN_ANCHOR_XY.items():
        root_path = f"{TASK_ROOT}/{name}"
        root = UsdGeom.Xform.Define(stage, root_path).GetPrim()
        root_position = np.asarray(
            [anchor_xy[0], anchor_xy[1], TABLE_TOP_Z + CAN_SUPPORT_CLEARANCE - inventory["low"][2]],
            dtype=np.float64,
        )
        _set_translate(root, root_position)
        visual = UsdGeom.Xform.Define(stage, f"{root_path}/ExactYCBVisual").GetPrim()
        visual.GetReferences().AddReference(str(CAN_USD.resolve()))
        rigid = UsdPhysics.RigidBodyAPI.Apply(root)
        rigid.CreateKinematicEnabledAttr(False)
        UsdPhysics.MassAPI.Apply(root).CreateMassAttr(0.349)
        physx_rigid = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
        physx_rigid.CreateEnableCCDAttr(True)
        physx_rigid.CreateLinearDampingAttr(0.08)
        physx_rigid.CreateAngularDampingAttr(0.12)
        collision = UsdGeom.Cylinder.Define(stage, f"{root_path}/CylinderCollision")
        collision.CreateAxisAttr("Z")
        collision.CreateHeightAttr(float(inventory["extent"][2] * 0.98))
        collision.CreateRadiusAttr(float(min(inventory["extent"][:2]) * 0.49))
        _set_translate(collision.GetPrim(), inventory["center"])
        UsdPhysics.CollisionAPI.Apply(collision.GetPrim()).CreateCollisionEnabledAttr(True)
        UsdGeom.Imageable(collision.GetPrim()).MakeInvisible()
        PhysxSchema.PhysxContactReportAPI.Apply(root).CreateThresholdAttr(0.0)
        can_roots[name] = root_path

    bin_path = f"{TASK_ROOT}/{BIN_NAME}"
    bin_root = UsdGeom.Xform.Define(stage, bin_path).GetPrim()
    _set_translate(bin_root, [BIN_ANCHOR_XY[0], BIN_ANCHOR_XY[1], TABLE_TOP_Z + BIN_SUPPORT_CLEARANCE])
    bin_rigid = UsdPhysics.RigidBodyAPI.Apply(bin_root)
    bin_rigid.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(bin_root).CreateMassAttr(3.0)
    PhysxSchema.PhysxContactReportAPI.Apply(bin_root).CreateThresholdAttr(0.0)

    bin_material = _material(stage, f"{TASK_ROOT}/BC012Looks/OpenBin", (0.055, 0.30, 0.52))
    divider_material = _material(stage, f"{TASK_ROOT}/BC012Looks/Divider", (0.95, 0.62, 0.08))
    goal_materials = {
        "left": _material(stage, f"{TASK_ROOT}/BC012Looks/LeftGoal", (0.15, 0.80, 0.38), 0.35),
        "right": _material(stage, f"{TASK_ROOT}/BC012Looks/RightGoal", (0.96, 0.30, 0.16), 0.35),
    }

    glb_scene = trimesh.load(str(OPEN_BIN_GLB), force="scene")
    measured_bounds = np.asarray(glb_scene.bounds, dtype=np.float64)
    if str(glb_scene.units) != "meters" or not np.allclose(
        measured_bounds, OPEN_BIN_SOURCE_BOUNDS, rtol=0.0, atol=1.0e-7
    ):
        raise RuntimeError(
            f"exact BC012 bin contract mismatch: units={glb_scene.units}, bounds={measured_bounds.tolist()}"
        )
    visual_paths: list[str] = []
    for index, node in enumerate(sorted(glb_scene.graph.nodes_geometry)):
        transform, geometry_name = glb_scene.graph.get(node)
        geometry = glb_scene.geometry[geometry_name].copy()
        geometry.apply_transform(transform)
        source_points = np.asarray(geometry.vertices, dtype=np.float64)
        points = np.column_stack((source_points[:, 0], -source_points[:, 2], source_points[:, 1]))
        mesh_name = f"Mesh_{index:02d}_{usd_prim_identifier(node)}"
        if not re.match(r"^[A-Za-z_]", mesh_name):
            raise RuntimeError(f"invalid generated USD prim identifier: {mesh_name}")
        mesh_path = f"{bin_path}/ExactGLBVisual/{mesh_name}"
        mesh = UsdGeom.Mesh.Define(stage, mesh_path)
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
        faces = np.asarray(geometry.faces, dtype=np.int32)
        mesh.CreateFaceVertexCountsAttr([3] * len(faces))
        mesh.CreateFaceVertexIndicesAttr(faces.reshape(-1).tolist())
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateDoubleSidedAttr(True)
        _bind_material(mesh.GetPrim(), bin_material)
        visual_paths.append(mesh_path)

    collision_paths: list[str] = []
    for panel_name, (center, size) in OPEN_BIN_COLLISION_PANELS.items():
        cube = UsdGeom.Cube.Define(stage, f"{bin_path}/Collisions/{panel_name}")
        cube.CreateSizeAttr(1.0)
        xform = UsdGeom.Xformable(cube.GetPrim())
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(*center))
        xform.AddScaleOp().Set(Gf.Vec3f(*size))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr(True)
        UsdGeom.Imageable(cube.GetPrim()).MakeInvisible()
        collision_paths.append(str(cube.GetPath()))

    divider_center, divider_size = PARTITION_DIVIDER
    divider = UsdGeom.Cube.Define(stage, f"{bin_path}/PartitionDivider")
    divider.CreateSizeAttr(1.0)
    divider_xform = UsdGeom.Xformable(divider.GetPrim())
    divider_xform.ClearXformOpOrder()
    divider_xform.AddTranslateOp().Set(Gf.Vec3d(*divider_center))
    divider_xform.AddScaleOp().Set(Gf.Vec3f(*divider_size))
    UsdPhysics.CollisionAPI.Apply(divider.GetPrim()).CreateCollisionEnabledAttr(True)
    _bind_material(divider.GetPrim(), divider_material)
    collision_paths.append(str(divider.GetPath()))

    for side, (center, size) in GOAL_ZONES.items():
        goal = UsdGeom.Cube.Define(stage, f"{bin_path}/{side.capitalize()}GoalZone")
        goal.CreateSizeAttr(1.0)
        xform = UsdGeom.Xformable(goal.GetPrim())
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(*center))
        xform.AddScaleOp().Set(Gf.Vec3f(*size))
        _bind_material(goal.GetPrim(), goal_materials[side])

    # Physical-only grasping publishes no can-to-palm assist joint paths.
    return {
        "task_id": TASK_ID,
        "task_name": TASK_NAME,
        "can_roots": can_roots,
        "bin_root": bin_path,
        "can_source_low": inventory["low"].copy(),
        "can_source_high": inventory["high"].copy(),
        "can_local_center": inventory["center"].copy(),
        "can_collision_radius": float(min(inventory["extent"][:2]) * 0.49),
        "can_collision_height": float(inventory["extent"][2] * 0.98),
        "glb_visual_paths": tuple(visual_paths),
        "collision_paths": tuple(collision_paths),
        "asset_hashes": {str(path.relative_to(REPOSITORY_ROOT)): sha256_file(path) for path in EXPECTED_SHA256},
    }


def bind_bc012_grasp_joint_owners(
    stage: Any,
    joint_paths: Mapping[str, str],
    can_roots: Mapping[str, str],
) -> dict[str, str]:
    """Atomically author grasp joints after robot spawn and before PhysX."""

    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    owners: dict[str, str] = {}
    transform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    for side in ("left", "right"):
        robot_root = stage.GetPrimAtPath(
            f"/World/HandsOnRobots/{side.capitalize()}Robot"
        )
        palm_name = f"{side}_palm_link"
        candidates = (
            [
                str(prim.GetPath())
                for prim in Usd.PrimRange(robot_root)
                if prim.GetName() == palm_name
                and prim.HasAPI(UsdPhysics.RigidBodyAPI)
            ]
            if robot_root.IsValid()
            else []
        )
        if len(candidates) != 1:
            raise RuntimeError(
                f"BC012 could not preauthor {side} grasp ownership after "
                f"robot spawn: expected one rigid body {palm_name}, "
                f"observed={candidates}"
            )
        joint_path = str(joint_paths.get(side) or "")
        can_name = LEFT_CAN_NAME if side == "left" else RIGHT_CAN_NAME
        can_path = str(can_roots.get(can_name) or "")
        if not joint_path or not can_path:
            raise RuntimeError(
                f"BC012 missing {side} grasp topology paths: "
                f"joint={joint_path!r}, can={can_path!r}"
            )
        palm_path = candidates[0]
        can_prim = stage.GetPrimAtPath(can_path)
        palm_prim = stage.GetPrimAtPath(palm_path)
        body0_frame, body1_frame = joint_local_frame_pair(
            _usd_world_pose_matrix(can_prim, transform_cache),
            _usd_world_pose_matrix(palm_prim, transform_cache),
        )
        body0_quaternion = matrix_to_quat_wxyz(body0_frame[:3, :3])
        body1_quaternion = matrix_to_quat_wxyz(body1_frame[:3, :3])
        joint = UsdPhysics.Joint.Define(stage, joint_path)
        # Publish the complete topology and every runtime-written attribute in
        # the same pre-PhysX stage update. Runtime changes values only.
        joint.CreateBody0Rel().SetTargets([Sdf.Path(can_path)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(palm_path)])
        joint.CreateLocalPos0Attr().Set(
            Gf.Vec3f(*(float(value) for value in body0_frame[:3, 3]))
        )
        joint.CreateLocalRot0Attr().Set(
            Gf.Quatf(*(float(value) for value in body0_quaternion))
        )
        joint.CreateLocalPos1Attr().Set(
            Gf.Vec3f(*(float(value) for value in body1_frame[:3, 3]))
        )
        joint.CreateLocalRot1Attr().Set(
            Gf.Quatf(*(float(value) for value in body1_quaternion))
        )
        joint.CreateJointEnabledAttr(True)
        joint.CreateExcludeFromArticulationAttr(True)
        for axis in GRASP_D6_AXES:
            limit = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), axis)
            free_limit = GRASP_D6_FREE_LIMIT_BY_AXIS[axis]
            limit.CreateLowAttr(-free_limit)
            limit.CreateHighAttr(free_limit)
        owners[side] = palm_path
    return owners
