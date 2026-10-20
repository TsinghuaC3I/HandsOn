"""Self-authored USD geometry for the synchronized dual-knob task.

The module deliberately keeps all Isaac/pxr imports inside the authoring function so
task discovery, manifest validation, and predicate tests work in a regular Python
environment.  The same function can author directly into the live stage or into a
standalone USDA layer for asset inspection.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any


DUAL_KNOB_TASK_NAME = "synchronized_dual_knob_setting"
DUAL_KNOB_SCENE_XML = "arena_arm_hand_synchronized_dual_knob_setting_rokae_wuji1.xml"
DUAL_KNOB_ROOT_PATH = "/World/Task/SynchronizedDualKnobPanel"
DUAL_KNOB_BASE_PATH = f"{DUAL_KNOB_ROOT_PATH}/base"
DUAL_KNOB_JOINT_ROOT_PATH = f"{DUAL_KNOB_ROOT_PATH}/joints"
DUAL_KNOB_LEFT_BODY_PATH = f"{DUAL_KNOB_ROOT_PATH}/left_knob"
DUAL_KNOB_RIGHT_BODY_PATH = f"{DUAL_KNOB_ROOT_PATH}/right_knob"
DUAL_KNOB_LEFT_JOINT_PATH = f"{DUAL_KNOB_JOINT_ROOT_PATH}/left_knob_joint"
DUAL_KNOB_RIGHT_JOINT_PATH = f"{DUAL_KNOB_JOINT_ROOT_PATH}/right_knob_joint"

DUAL_KNOB_PANEL_ORIGIN_M = (-0.14, 0.0, 0.915)
DUAL_KNOB_PANEL_HALF_EXTENTS_M = (0.105, 0.245, 0.025)
DUAL_KNOB_CENTER_X_M = -0.005
DUAL_KNOB_CENTER_Y_M = 0.135
DUAL_KNOB_CENTER_Z_M = 0.050
DUAL_KNOB_RADIUS_M = 0.048
DUAL_KNOB_HEIGHT_M = 0.044
DUAL_KNOB_LIMIT_DEG = 100.0
DUAL_KNOB_MASS_KG = 0.085
DUAL_KNOB_BASE_MASS_KG = 2.4
DUAL_KNOB_DRIVE_STIFFNESS = 0.030
DUAL_KNOB_DRIVE_DAMPING = 0.004
DUAL_KNOB_DRIVE_MAX_FORCE = 0.22
DUAL_KNOB_JOINT_FRICTION = 0.010
DUAL_KNOB_CONTACT_THRESHOLD_N = 0.15

DUAL_KNOB_SUCCESS_CONDITION_LABELS = (
    "Left knob is held counter-clockwise at its target",
    "Right knob is held clockwise at its target",
    "The sampled ratio or sum/difference relation is satisfied",
    "Both assigned hands remain in contact",
    "Start and target-entry times are synchronized",
    "The stable setting is held for 15 frames",
)


@dataclass(frozen=True)
class DualKnobAssetContract:
    root_path: str = DUAL_KNOB_ROOT_PATH
    joint_names: tuple[str, str] = ("left_knob_joint", "right_knob_joint")
    revolute_joint_count: int = 2
    rigid_body_count: int = 3
    collision_shape_count: int = 3
    joint_limit_degrees: tuple[float, float] = (
        -DUAL_KNOB_LIMIT_DEG,
        DUAL_KNOB_LIMIT_DEG,
    )
    axis: str = "Z"
    license: str = "project"


ASSET_CONTRACT = DualKnobAssetContract()


def _set_display_color(geom: Any, color: tuple[float, float, float]) -> None:
    from pxr import Gf

    geom.CreateDisplayColorAttr().Set([Gf.Vec3f(*color)])


def _define_box(
    stage: Any,
    path: str,
    *,
    half_extents: tuple[float, float, float],
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rotation_z_degrees: float = 0.0,
    color: tuple[float, float, float],
    collision: bool,
) -> Any:
    from pxr import Gf, UsdGeom, UsdPhysics

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr().Set(2.0)
    xformable = UsdGeom.Xformable(cube)
    xformable.AddTranslateOp().Set(Gf.Vec3d(*translation))
    if rotation_z_degrees:
        xformable.AddRotateZOp().Set(rotation_z_degrees)
    xformable.AddScaleOp().Set(Gf.Vec3d(*half_extents))
    _set_display_color(cube, color)
    if collision:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return cube


def _define_knob(
    stage: Any,
    *,
    body_path: str,
    center_y: float,
    body_color: tuple[float, float, float],
) -> Any:
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    body = UsdGeom.Xform.Define(stage, body_path)
    UsdGeom.Xformable(body).AddTranslateOp().Set(
        Gf.Vec3d(DUAL_KNOB_CENTER_X_M, center_y, DUAL_KNOB_CENTER_Z_M)
    )
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr().Set(DUAL_KNOB_MASS_KG)
    rigid = PhysxSchema.PhysxRigidBodyAPI.Apply(body.GetPrim())
    rigid.CreateLinearDampingAttr().Set(0.08)
    rigid.CreateAngularDampingAttr().Set(0.05)
    rigid.CreateMaxAngularVelocityAttr().Set(12.0)
    report = PhysxSchema.PhysxContactReportAPI.Apply(body.GetPrim())
    report.CreateThresholdAttr().Set(0.0)

    cylinder = UsdGeom.Cylinder.Define(stage, f"{body_path}/grip")
    cylinder.CreateAxisAttr().Set(UsdGeom.Tokens.z)
    cylinder.CreateRadiusAttr().Set(DUAL_KNOB_RADIUS_M)
    cylinder.CreateHeightAttr().Set(DUAL_KNOB_HEIGHT_M)
    _set_display_color(cylinder, body_color)
    UsdPhysics.CollisionAPI.Apply(cylinder.GetPrim())
    collision = PhysxSchema.PhysxCollisionAPI.Apply(cylinder.GetPrim())
    collision.CreateContactOffsetAttr().Set(0.004)
    collision.CreateRestOffsetAttr().Set(0.0)

    # A pointer and radial grip ribs are visual-only children of the rotating body.
    _define_box(
        stage,
        f"{body_path}/pointer",
        half_extents=(0.030, 0.006, 0.004),
        translation=(0.013, 0.0, 0.5 * DUAL_KNOB_HEIGHT_M + 0.004),
        color=(0.97, 0.93, 0.22),
        collision=False,
    )
    for rib_index in range(12):
        angle = 30.0 * rib_index
        angle_rad = math.radians(angle)
        rib_radius = DUAL_KNOB_RADIUS_M - 0.004
        _define_box(
            stage,
            f"{body_path}/rib_{rib_index:02d}",
            half_extents=(0.004, 0.008, 0.5 * DUAL_KNOB_HEIGHT_M + 0.001),
            translation=(
                rib_radius * math.cos(angle_rad),
                rib_radius * math.sin(angle_rad),
                0.0,
            ),
            rotation_z_degrees=angle,
            color=(min(body_color[0] + 0.08, 1.0), min(body_color[1] + 0.08, 1.0), min(body_color[2] + 0.08, 1.0)),
            collision=False,
        )
    return body


def _define_revolute_joint(
    stage: Any,
    *,
    path: str,
    body_path: str,
    center_y: float,
) -> Any:
    from pxr import Gf, PhysxSchema, Sdf, UsdPhysics

    joint = UsdPhysics.RevoluteJoint.Define(stage, path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(DUAL_KNOB_BASE_PATH)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body_path)])
    joint.CreateLocalPos0Attr().Set(
        Gf.Vec3f(DUAL_KNOB_CENTER_X_M, center_y, DUAL_KNOB_CENTER_Z_M)
    )
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateAxisAttr().Set(UsdPhysics.Tokens.z)
    joint.CreateLowerLimitAttr().Set(-DUAL_KNOB_LIMIT_DEG)
    joint.CreateUpperLimitAttr().Set(DUAL_KNOB_LIMIT_DEG)

    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drive.CreateTypeAttr().Set(UsdPhysics.Tokens.force)
    drive.CreateTargetPositionAttr().Set(0.0)
    drive.CreateTargetVelocityAttr().Set(0.0)
    drive.CreateStiffnessAttr().Set(DUAL_KNOB_DRIVE_STIFFNESS)
    drive.CreateDampingAttr().Set(DUAL_KNOB_DRIVE_DAMPING)
    drive.CreateMaxForceAttr().Set(DUAL_KNOB_DRIVE_MAX_FORCE)
    physx_joint = PhysxSchema.PhysxJointAPI.Apply(joint.GetPrim())
    physx_joint.CreateJointFrictionAttr().Set(DUAL_KNOB_JOINT_FRICTION)
    return joint


def author_dual_knob_panel(
    stage: Any,
    root_path: str = DUAL_KNOB_ROOT_PATH,
) -> Path:
    """Author the fixed-base two-DOF calibration panel into ``stage``."""
    if root_path != DUAL_KNOB_ROOT_PATH:
        raise ValueError(
            f"The asset contract requires root_path={DUAL_KNOB_ROOT_PATH!r}; got {root_path!r}"
        )

    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    root = UsdGeom.Xform.Define(stage, root_path)
    UsdGeom.Xformable(root).AddTranslateOp().Set(Gf.Vec3d(*DUAL_KNOB_PANEL_ORIGIN_M))
    root_prim = root.GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(root_prim)
    root_prim.SetCustomDataByKey("handson_assetName", "dual_knob_calibration_panel")
    root_prim.SetCustomDataByKey("handson_assetVersion", "1.0.0")
    root_prim.SetCustomDataByKey("handson_license", "project")
    root_prim.SetCustomDataByKey("handson_authoredFor", DUAL_KNOB_TASK_NAME)

    base = UsdGeom.Xform.Define(stage, DUAL_KNOB_BASE_PATH)
    UsdPhysics.RigidBodyAPI.Apply(base.GetPrim())
    UsdPhysics.MassAPI.Apply(base.GetPrim()).CreateMassAttr().Set(
        DUAL_KNOB_BASE_MASS_KG
    )
    _define_box(
        stage,
        f"{DUAL_KNOB_BASE_PATH}/housing",
        half_extents=DUAL_KNOB_PANEL_HALF_EXTENTS_M,
        color=(0.10, 0.14, 0.18),
        collision=True,
    )
    _define_box(
        stage,
        f"{DUAL_KNOB_BASE_PATH}/left_zone",
        half_extents=(0.0025, 0.082, 0.003),
        translation=(DUAL_KNOB_PANEL_HALF_EXTENTS_M[0] + 0.003, DUAL_KNOB_CENTER_Y_M, 0.0),
        color=(0.16, 0.48, 0.94),
        collision=False,
    )
    _define_box(
        stage,
        f"{DUAL_KNOB_BASE_PATH}/right_zone",
        half_extents=(0.0025, 0.082, 0.003),
        translation=(DUAL_KNOB_PANEL_HALF_EXTENTS_M[0] + 0.003, -DUAL_KNOB_CENTER_Y_M, 0.0),
        color=(0.94, 0.30, 0.18),
        collision=False,
    )

    # Twenty-two high-contrast ticks make joint rotation legible in the main view.
    for side_name, center_y in (
        ("left", DUAL_KNOB_CENTER_Y_M),
        ("right", -DUAL_KNOB_CENTER_Y_M),
    ):
        for tick_index, angle in enumerate(range(-100, 101, 20)):
            angle_rad = math.radians(float(angle))
            tick_radius = DUAL_KNOB_RADIUS_M + 0.020
            _define_box(
                stage,
                f"{DUAL_KNOB_BASE_PATH}/{side_name}_tick_{tick_index:02d}",
                half_extents=(0.010, 0.0025, 0.0025),
                translation=(
                    DUAL_KNOB_CENTER_X_M + tick_radius * math.cos(angle_rad),
                    center_y + tick_radius * math.sin(angle_rad),
                    DUAL_KNOB_PANEL_HALF_EXTENTS_M[2] + 0.004,
                ),
                rotation_z_degrees=float(angle),
                color=(0.82, 0.86, 0.90),
                collision=False,
            )

    _define_knob(
        stage,
        body_path=DUAL_KNOB_LEFT_BODY_PATH,
        center_y=DUAL_KNOB_CENTER_Y_M,
        body_color=(0.08, 0.34, 0.82),
    )
    _define_knob(
        stage,
        body_path=DUAL_KNOB_RIGHT_BODY_PATH,
        center_y=-DUAL_KNOB_CENTER_Y_M,
        body_color=(0.86, 0.18, 0.10),
    )

    UsdGeom.Xform.Define(stage, DUAL_KNOB_JOINT_ROOT_PATH)
    fixed = UsdPhysics.FixedJoint.Define(
        stage, f"{DUAL_KNOB_JOINT_ROOT_PATH}/fixed_base_joint"
    )
    fixed.CreateBody1Rel().SetTargets([Sdf.Path(DUAL_KNOB_BASE_PATH)])
    fixed.CreateLocalPos0Attr().Set(Gf.Vec3f(*DUAL_KNOB_PANEL_ORIGIN_M))
    fixed.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    _define_revolute_joint(
        stage,
        path=DUAL_KNOB_LEFT_JOINT_PATH,
        body_path=DUAL_KNOB_LEFT_BODY_PATH,
        center_y=DUAL_KNOB_CENTER_Y_M,
    )
    _define_revolute_joint(
        stage,
        path=DUAL_KNOB_RIGHT_JOINT_PATH,
        body_path=DUAL_KNOB_RIGHT_BODY_PATH,
        center_y=-DUAL_KNOB_CENTER_Y_M,
    )
    return Path(root_path)


def export_dual_knob_panel_usda(output_path: str | Path) -> Path:
    """Create a portable USDA layer using the exact live-stage authoring code."""
    from pxr import Usd, UsdGeom

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    author_dual_knob_panel(stage)
    stage.GetRootLayer().Save()
    # USD versions differ on whether an exported USDA ends with one or two
    # newlines. Canonicalize the text so the checked-in asset is reproducible
    # byte-for-byte across the HP runtime and local inspections.
    output.write_text(
        output.read_text(encoding="utf-8").rstrip() + "\n",
        encoding="utf-8",
    )
    return output


__all__ = [
    "ASSET_CONTRACT",
    "DUAL_KNOB_BASE_PATH",
    "DUAL_KNOB_CONTACT_THRESHOLD_N",
    "DUAL_KNOB_DRIVE_DAMPING",
    "DUAL_KNOB_DRIVE_MAX_FORCE",
    "DUAL_KNOB_DRIVE_STIFFNESS",
    "DUAL_KNOB_JOINT_FRICTION",
    "DUAL_KNOB_LEFT_BODY_PATH",
    "DUAL_KNOB_LEFT_JOINT_PATH",
    "DUAL_KNOB_LIMIT_DEG",
    "DUAL_KNOB_RIGHT_BODY_PATH",
    "DUAL_KNOB_RIGHT_JOINT_PATH",
    "DUAL_KNOB_ROOT_PATH",
    "DUAL_KNOB_SCENE_XML",
    "DUAL_KNOB_SUCCESS_CONDITION_LABELS",
    "DUAL_KNOB_TASK_NAME",
    "DualKnobAssetContract",
    "author_dual_knob_panel",
    "export_dual_knob_panel_usda",
]
