"""Asset provenance and scene constants for B320-BC007."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import numpy as np


TASK_ID: Final = "B320-BC007"
TASK_NAME: Final = "bench320_bc007_bimanual_long_rod_yaw_reorientation"
SCENE_XML: Final = "arena_arm_hand_bc007_long_rod_rokae_wuji1.xml"
TASK_ROOT: Final = "/World/Task"
ROD_NAME: Final = "bc007_long_rod"
GOAL_NAME: Final = "bc007_long_rod_goal"
ROD_ROOT: Final = f"{TASK_ROOT}/{ROD_NAME}"
GOAL_ROOT: Final = f"{TASK_ROOT}/{GOAL_NAME}"
GRASP_JOINT_PATHS: Final = {
    side: f"{TASK_ROOT}/BC007AssistJoints/{side}_rod_grasp"
    for side in ("left", "right")
}

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[4]
TASKDEV_ROOT: Final = REPOSITORY_ROOT / "taskdev" / "tasks" / TASK_NAME
ASSET_ROOT: Final = TASKDEV_ROOT / "assets"
ROD_USD: Final = ASSET_ROOT / "dexverse_stick" / "stick.usda"
ROD_USD_SHA256: Final = "52584f8606e08d290e514fdb6d63a06f36b31d99908902a7e29c5b5cd2bda516"

TABLE_TOP_Z: Final = 0.89
ROD_LENGTH_M: Final = 0.46
ROD_HALF_LENGTH_M: Final = 0.5 * ROD_LENGTH_M
ROD_RADIUS_M: Final = 0.0225
ROD_COLLISION_SIDES: Final = 12
# The horizontal collider rests on a polygon face, whose support radius is
# smaller than the visible cylinder radius. Use that face for both the first
# authored pose and seeded resets instead of starting 0.77 mm above support.
ROD_SUPPORT_HALF_HEIGHT_M: Final = ROD_RADIUS_M * math.cos(math.pi / ROD_COLLISION_SIDES)
ROD_MASS_KG: Final = 0.22
ROD_LINEAR_DAMPING: Final = 0.25
ROD_ANGULAR_DAMPING: Final = 5.00
# BC007's recording-origin contract must prove that the supported movable rod
# remains awake on every readiness sample.  A zero energy threshold prevents
# automatic sleeping while reset and deliberate manipulation remain physical.
ROD_SLEEP_THRESHOLD: Final = 0.0
ROD_START_CENTER: Final = np.asarray(
    [-0.45, 0.0, TABLE_TOP_Z + ROD_SUPPORT_HALF_HEIGHT_M], dtype=np.float64
)
ROD_TARGET_CENTER_OFFSET: Final = np.asarray([0.05, 0.0, 0.1865], dtype=np.float64)
ROD_START_ORIENTATION_WXYZ: Final = np.asarray(
    [math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0], dtype=np.float64
)
TARGET_YAW_DELTA_DEG: Final = 90.0
RESET_XY_JITTER_M: Final = 0.020
RESET_YAW_JITTER_DEG: Final = 8.0
RESET_MAX_ATTEMPTS: Final = 64
LEFT_ENDPOINT_LOCAL_SIGN: Final = -1.0
RIGHT_ENDPOINT_LOCAL_SIGN: Final = 1.0

HEIGHT_BAND_LOW_M: Final = 1.02
HEIGHT_BAND_HIGH_M: Final = 1.14
TARGET_POSITION_TOLERANCE_M: Final = 0.04
TARGET_YAW_TOLERANCE_DEG: Final = 8.0
ENDPOINT_HEIGHT_DIFFERENCE_M: Final = 0.03
MINIMUM_HAND_SEPARATION_M: Final = 0.25
MAX_LINEAR_SPEED_MPS: Final = 0.08
MAX_ANGULAR_SPEED_RADPS: Final = 0.30
STABILITY_STEPS: Final = 15

TASK_SCENES: Final = {TASK_NAME: SCENE_XML}
SUCCESS_CONDITION_LABELS: Final = {
    TASK_NAME: (
        "Left and right hands continuously own different rod-end regions",
        "The rod is lifted into and kept inside the prescribed height band",
        "The rod completes the required 90-degree horizontal yaw rotation",
        "Target yaw, center position, endpoint levelness, and speed remain stable",
    )
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_asset_contract() -> list[str]:
    if not ROD_USD.is_file():
        return [f"missing exact BC007 rod asset: {ROD_USD}"]
    observed = sha256_file(ROD_USD)
    errors = []
    if observed != ROD_USD_SHA256:
        errors.append(
            f"BC007 rod asset hash mismatch: expected {ROD_USD_SHA256}, got {observed}"
        )
    text = ROD_USD.read_text(encoding="utf-8", errors="replace")
    if 'def Cylinder "geom"' not in text or 'defaultPrim = "stick"' not in text:
        errors.append("BC007 rod source is not the licensed DexVerse stick cylinder")
    return errors


def quat_multiply_wxyz(left: Sequence[float], right: Sequence[float]) -> np.ndarray:
    lw, lx, ly, lz = np.asarray(left, dtype=np.float64).reshape(4)
    rw, rx, ry, rz = np.asarray(right, dtype=np.float64).reshape(4)
    result = np.asarray(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )
    result /= max(float(np.linalg.norm(result)), 1.0e-12)
    if result[0] < 0.0:
        result *= -1.0
    return result


def yaw_quaternion_wxyz(yaw_degrees: float) -> np.ndarray:
    half = 0.5 * math.radians(float(yaw_degrees))
    return np.asarray([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64)


def initial_orientation_wxyz(yaw_jitter_degrees: float) -> np.ndarray:
    return quat_multiply_wxyz(
        yaw_quaternion_wxyz(yaw_jitter_degrees), ROD_START_ORIENTATION_WXYZ
    )


def target_orientation_wxyz(initial_wxyz: Sequence[float]) -> np.ndarray:
    return quat_multiply_wxyz(yaw_quaternion_wxyz(TARGET_YAW_DELTA_DEG), initial_wxyz)


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
        raise ValueError("BC007 grasp-joint bodies require finite world transforms")
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
    matrix = np.eye(4, dtype=np.float64)
    w, x, y, z = pose[3:7] / max(float(np.linalg.norm(pose[3:7])), 1.0e-12)
    matrix[:3, :3] = np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    matrix[:3, 3] = pose[:3]
    return matrix


def _set_xform_pose(prim, position: Sequence[float], orientation_wxyz: Sequence[float]) -> None:
    from pxr import Gf, UsdGeom

    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    translate = xformable.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble)
    orient = xformable.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble)
    position_value = np.asarray(position, dtype=np.float64).reshape(3)
    quaternion = np.asarray(orientation_wxyz, dtype=np.float64).reshape(4)
    translate.Set(Gf.Vec3d(*position_value.tolist()))
    orient.Set(
        Gf.Quatd(
            float(quaternion[0]),
            Gf.Vec3d(*quaternion[1:].tolist()),
        )
    )


def set_goal_marker_pose(stage, position: Sequence[float], orientation_wxyz: Sequence[float]) -> None:
    prim = stage.GetPrimAtPath(GOAL_ROOT)
    if not prim.IsValid():
        raise RuntimeError(f"BC007 goal marker is missing at {GOAL_ROOT}")
    _set_xform_pose(prim, position, orientation_wxyz)


def spawn_bc007_assets(stage) -> dict[str, object]:
    """Reference the existing licensed rod and add task-local physics metadata."""
    errors = validate_asset_contract()
    if errors:
        raise RuntimeError("; ".join(errors))

    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade

    task_root = stage.DefinePrim(TASK_ROOT, "Xform")
    rod_root = stage.DefinePrim(ROD_ROOT, "Xform")
    _set_xform_pose(rod_root, ROD_START_CENTER, ROD_START_ORIENTATION_WXYZ)

    asset_prim = stage.DefinePrim(f"{ROD_ROOT}/asset", "Xform")
    asset_prim.GetReferences().AddReference(str(ROD_USD), "/stick")
    scale_attr = asset_prim.GetAttribute("xformOp:scale")
    if not scale_attr:
        scale_attr = asset_prim.CreateAttribute(
            "xformOp:scale", Sdf.ValueTypeNames.Double3, custom=False
        )
    # This authored DexVerse cylinder's observed USD extent is twice its
    # numeric height scale.  Use the physical half-length here; the live asset
    # gate verifies a 0.46 m composed rod rather than trusting metadata.
    scale_attr.Set(Gf.Vec3d(ROD_RADIUS_M, ROD_RADIUS_M, ROD_HALF_LENGTH_M))
    order_attr = asset_prim.GetAttribute("xformOpOrder")
    if not order_attr:
        order_attr = asset_prim.CreateAttribute(
            "xformOpOrder", Sdf.ValueTypeNames.TokenArray, custom=False
        )
    order_attr.Set(["xformOp:scale"])

    rigid_api = UsdPhysics.RigidBodyAPI.Apply(rod_root)
    rigid_api.CreateRigidBodyEnabledAttr().Set(True)
    mass_api = UsdPhysics.MassAPI.Apply(rod_root)
    mass_api.CreateMassAttr().Set(ROD_MASS_KG)
    transverse_inertia = ROD_MASS_KG * (
        3.0 * ROD_RADIUS_M * ROD_RADIUS_M + ROD_LENGTH_M * ROD_LENGTH_M
    ) / 12.0
    axial_inertia = 0.5 * ROD_MASS_KG * ROD_RADIUS_M * ROD_RADIUS_M
    mass_api.CreateDiagonalInertiaAttr().Set(
        Gf.Vec3f(transverse_inertia, transverse_inertia, axial_inertia)
    )
    mass_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))

    geom_path = f"{ROD_ROOT}/asset/geom"
    geom_prim = stage.GetPrimAtPath(geom_path)
    if not geom_prim.IsValid():
        raise RuntimeError(f"Referenced BC007 cylinder is missing at {geom_path}")
    visual_collision = UsdPhysics.CollisionAPI.Apply(geom_prim)
    visual_collision.CreateCollisionEnabledAttr().Set(False)

    material = UsdShade.Material.Define(stage, f"{TASK_ROOT}/bc007_grip_material")
    material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    material_api.CreateStaticFrictionAttr().Set(1.10)
    material_api.CreateDynamicFrictionAttr().Set(0.90)
    material_api.CreateRestitutionAttr().Set(0.02)
    # The licensed visual remains unchanged.  A single 12-sided convex proxy
    # approximates its round surface to within 0.8 mm while giving PhysX one
    # stable tabletop face instead of an endlessly rolling analytic cylinder.
    side_count = ROD_COLLISION_SIDES
    angles = [
        -0.5 * math.pi - math.pi / side_count + 2.0 * math.pi * i / side_count
        for i in range(side_count)
    ]
    ring = [
        (ROD_RADIUS_M * math.cos(angle), ROD_RADIUS_M * math.sin(angle))
        for angle in angles
    ]
    proxy_geom = UsdGeom.Mesh.Define(stage, f"{ROD_ROOT}/collision_proxy")
    proxy_geom.CreatePointsAttr().Set(
        [
            Gf.Vec3f(x, y, z)
            for z in (-ROD_HALF_LENGTH_M, ROD_HALF_LENGTH_M)
            for x, y in ring
        ]
    )
    proxy_geom.CreateFaceVertexCountsAttr().Set(
        [side_count, side_count, *([4] * side_count)]
    )
    face_indices = [
        *reversed(range(side_count)),
        *range(side_count, 2 * side_count),
    ]
    for i in range(side_count):
        next_i = (i + 1) % side_count
        face_indices.extend([i, next_i, side_count + next_i, side_count + i])
    proxy_geom.CreateFaceVertexIndicesAttr().Set(face_indices)
    proxy_geom.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    proxy_geom.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    proxy_prim = proxy_geom.GetPrim()
    proxy_collision = UsdPhysics.CollisionAPI.Apply(proxy_prim)
    proxy_collision.CreateCollisionEnabledAttr().Set(True)
    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(proxy_prim)
    mesh_collision.CreateApproximationAttr().Set("convexHull")
    proxy_physx = PhysxSchema.PhysxCollisionAPI.Apply(proxy_prim)
    proxy_physx.CreateContactOffsetAttr().Set(0.004)
    proxy_physx.CreateRestOffsetAttr().Set(0.0)
    UsdShade.MaterialBindingAPI.Apply(proxy_prim).Bind(
        material, materialPurpose="physics"
    )

    report_api = PhysxSchema.PhysxContactReportAPI.Apply(rod_root)
    report_api.CreateThresholdAttr().Set(0.0)
    body_api = PhysxSchema.PhysxRigidBodyAPI.Apply(rod_root)
    # A perfectly round tabletop cylinder otherwise keeps rolling on tiny
    # solver perturbations and can produce false strong-contact impulses long
    # after reset.  Damping stabilizes the free body, while the zero sleep
    # threshold keeps the mandatory awake/readback proof honest for action 0.
    body_api.CreateLinearDampingAttr().Set(ROD_LINEAR_DAMPING)
    body_api.CreateAngularDampingAttr().Set(ROD_ANGULAR_DAMPING)
    body_api.CreateSleepThresholdAttr().Set(ROD_SLEEP_THRESHOLD)
    body_api.CreateMaxLinearVelocityAttr().Set(3.0)
    body_api.CreateMaxAngularVelocityAttr().Set(12.0)

    goal_root = stage.DefinePrim(GOAL_ROOT, "Xform")
    target_position = ROD_START_CENTER + ROD_TARGET_CENTER_OFFSET
    target_orientation = target_orientation_wxyz(ROD_START_ORIENTATION_WXYZ)
    _set_xform_pose(goal_root, target_position, target_orientation)
    goal_geom = UsdGeom.Cylinder.Define(stage, f"{GOAL_ROOT}/geom")
    goal_geom.CreateAxisAttr().Set("Z")
    goal_geom.CreateRadiusAttr().Set(ROD_RADIUS_M * 1.18)
    goal_geom.CreateHeightAttr().Set(ROD_LENGTH_M)
    goal_geom.CreateDisplayColorAttr().Set([Gf.Vec3f(0.15, 0.85, 0.25)])
    goal_geom.CreateDisplayOpacityAttr().Set([0.24])

    return {
        "task_root": str(task_root.GetPath()),
        "rod_root": ROD_ROOT,
        "rod_geom": geom_path,
        "goal_root": GOAL_ROOT,
        # The complete disabled joint topology is authored only after the
        # native robot palms exist and before the first PhysX reset.
        "joint_paths": dict(GRASP_JOINT_PATHS),
        "asset_path": str(ROD_USD),
        "asset_sha256": ROD_USD_SHA256,
        "length_m": ROD_LENGTH_M,
        "radius_m": ROD_RADIUS_M,
        "collision_sides": ROD_COLLISION_SIDES,
        "mass_kg": ROD_MASS_KG,
        "linear_damping": ROD_LINEAR_DAMPING,
        "angular_damping": ROD_ANGULAR_DAMPING,
        "sleep_threshold": ROD_SLEEP_THRESHOLD,
    }


def bind_bc007_grasp_joint_owners(
    stage: Any,
    joint_paths: Mapping[str, str],
    rod_root: str,
) -> dict[str, str]:
    """Atomically preauthor disabled contact-gated rod/palm joints."""

    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    rod_path = str(rod_root)
    rod_prim = stage.GetPrimAtPath(rod_path)
    if not rod_prim.IsValid() or not rod_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError(f"BC007 grasp rod body is invalid at {rod_path}")
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
                f"BC007 expected one {palm_name} rigid body, observed={candidates}"
            )
        joint_path = str(joint_paths.get(side) or "")
        if not joint_path:
            raise RuntimeError(f"BC007 missing {side} grasp-joint path")
        palm_path = candidates[0]
        palm_prim = stage.GetPrimAtPath(palm_path)
        body0_frame, body1_frame = joint_local_frame_pair(
            _usd_world_pose_matrix(rod_prim, transform_cache),
            _usd_world_pose_matrix(palm_prim, transform_cache),
        )
        body0_quaternion = matrix_to_quat_wxyz(body0_frame[:3, :3])
        body1_quaternion = matrix_to_quat_wxyz(body1_frame[:3, :3])
        joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
        joint.CreateBody0Rel().SetTargets([Sdf.Path(rod_path)])
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
        joint.CreateJointEnabledAttr(False)
        owners[side] = palm_path
    return owners


__all__ = [
    "ASSET_ROOT",
    "ENDPOINT_HEIGHT_DIFFERENCE_M",
    "GOAL_NAME",
    "GOAL_ROOT",
    "GRASP_JOINT_PATHS",
    "HEIGHT_BAND_HIGH_M",
    "HEIGHT_BAND_LOW_M",
    "LEFT_ENDPOINT_LOCAL_SIGN",
    "MAX_ANGULAR_SPEED_RADPS",
    "MAX_LINEAR_SPEED_MPS",
    "MINIMUM_HAND_SEPARATION_M",
    "RESET_MAX_ATTEMPTS",
    "RESET_XY_JITTER_M",
    "RESET_YAW_JITTER_DEG",
    "RIGHT_ENDPOINT_LOCAL_SIGN",
    "ROD_HALF_LENGTH_M",
    "ROD_LENGTH_M",
    "ROD_LINEAR_DAMPING",
    "ROD_MASS_KG",
    "ROD_ANGULAR_DAMPING",
    "ROD_NAME",
    "ROD_RADIUS_M",
    "ROD_COLLISION_SIDES",
    "ROD_SUPPORT_HALF_HEIGHT_M",
    "ROD_SLEEP_THRESHOLD",
    "ROD_ROOT",
    "ROD_START_CENTER",
    "ROD_START_ORIENTATION_WXYZ",
    "ROD_TARGET_CENTER_OFFSET",
    "ROD_USD",
    "ROD_USD_SHA256",
    "SCENE_XML",
    "STABILITY_STEPS",
    "SUCCESS_CONDITION_LABELS",
    "TABLE_TOP_Z",
    "TARGET_POSITION_TOLERANCE_M",
    "TARGET_YAW_DELTA_DEG",
    "TARGET_YAW_TOLERANCE_DEG",
    "TASK_ID",
    "TASK_NAME",
    "TASK_ROOT",
    "TASK_SCENES",
    "initial_orientation_wxyz",
    "bind_bc007_grasp_joint_owners",
    "joint_local_frame_pair",
    "matrix_to_quat_wxyz",
    "quat_multiply_wxyz",
    "set_goal_marker_pose",
    "spawn_bc007_assets",
    "target_orientation_wxyz",
    "validate_asset_contract",
    "yaw_quaternion_wxyz",
]
