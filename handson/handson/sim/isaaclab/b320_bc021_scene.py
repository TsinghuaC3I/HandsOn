"""Dedicated asset-backed scene boundary for B320-BC021.

The selected Sketchfab wooden-block closure is normalized in two USD wrappers.
Each wrapper contains one visible source mesh plus one invisible convex-hull
collider derived from the same source mesh.  The runtime adds only the
semantic marked-face visuals and the rigid-body schemas.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .b320_bc021_contract import FINAL_GEOMETRY, TASK_ID, TASK_NAME
from .b320_bimanual_scene import (
    B320_ASSET_ROOT_PATH,
    B320_TABLE_TOP_Z,
    B320SpawnResult,
    _bind_material,
    _set_xform,
    _spawn_worktable,
)


TASK_REQUIREMENT = (
    "Lift both assigned blocks with their assigned hands, then mate their "
    "marked faces in midair and hold the physical contact stable for 0.4 seconds."
)
ASSET_INTEGRATION_STATUS = "integrated_user_selected_asset"
ASSET_CANDIDATE_URL = (
    "https://sketchfab.com/3d-models/"
    "old-fashioned-wooden-blocks-8331752cf38243638b0681f5ea230c28"
)
ASSET_CANDIDATE_LICENSE = "CC BY 4.0"
ASSET_DIRECTORY = (
    Path(__file__).resolve().parent.parent
    / "envs"
    / "usd"
    / "b320_bimanual_benchmark"
    / "assets"
    / "cc_by_curlyfrywhy_old_fashioned_wooden_blocks"
)
ASSET_WRAPPERS = {
    "block_a": ASSET_DIRECTORY / "block_a.usda",
    "block_b": ASSET_DIRECTORY / "block_b.usda",
}
MARKER_THICKNESS_M = 0.0015
# Keep the colored surface visually distinct without making it a millimetre-
# scale spacer between the real collision faces.  The old marker protruded by
# its full 1.5 mm thickness on each block, so two visibly touching markers
# could leave the physical faces about 3 mm apart.
MARKER_SURFACE_OFFSET_M = 0.00005
BLOCK_HOME_POSES_WXYZ = {
    "block_a": (
        -0.32,
        0.20,
        B320_TABLE_TOP_Z + FINAL_GEOMETRY.left_half_extents_m[2],
        1.0,
        0.0,
        0.0,
        0.0,
    ),
    "block_b": (
        -0.32,
        -0.20,
        B320_TABLE_TOP_Z + FINAL_GEOMETRY.right_half_extents_m[2],
        1.0,
        0.0,
        0.0,
        0.0,
    ),
}


def _set_display_color(prim: Any, color: tuple[float, float, float]) -> None:
    from pxr import Gf, UsdGeom

    gprim = UsdGeom.Gprim(prim)
    gprim.CreateDisplayColorAttr().Set([Gf.Vec3f(*map(float, color))])
    gprim.CreateDisplayOpacityAttr().Set([1.0])


def _marked_face_marker_center_local(side: str) -> Any:
    geometry = FINAL_GEOMETRY
    marked_center = geometry.marked_center_local(side)
    marked_normal = geometry.marked_normal_local(side)
    return marked_center + marked_normal * (
        MARKER_SURFACE_OFFSET_M - 0.5 * MARKER_THICKNESS_M
    )


def _spawn_downloaded_block(
    stage: Any,
    sim_utils: Any,
    *,
    name: str,
    home_pose_wxyz: tuple[float, ...],
    side: str,
    marker_color: tuple[float, float, float],
) -> tuple[str, int]:
    from isaaclab.sim import schemas
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics

    source_asset = ASSET_WRAPPERS.get(name)
    if source_asset is None or not source_asset.is_file():
        raise FileNotFoundError(
            f"BC021 selected asset wrapper is missing for {name}: {source_asset}"
        )
    geometry = FINAL_GEOMETRY
    half_extents = geometry.half_extents(side)
    marked_normal = geometry.marked_normal_local(side)
    marked_center = geometry.marked_center_local(side)
    tangent_u, tangent_v = geometry.marked_tangents_local(side)
    patch_half_extents = geometry.marked_patch_half_extents(side)
    root_path = f"{B320_ASSET_ROOT_PATH}/{name}"
    root = UsdGeom.Xform.Define(stage, root_path).GetPrim()
    _set_xform(
        root,
        translation=tuple(float(value) for value in home_pose_wxyz[:3]),
        orientation_wxyz=tuple(float(value) for value in home_pose_wxyz[3:7]),
    )
    root.SetCustomDataByKey("handson:logicalName", name)
    root.SetCustomDataByKey("handson:formalTruth", True)
    root.SetCustomDataByKey("handson:geometryProvisional", False)
    root.SetCustomDataByKey("handson:visibleSource", "downloaded_usd")
    root.SetCustomDataByKey("handson:assetSelectionStatus", ASSET_INTEGRATION_STATUS)
    root.SetCustomDataByKey("handson:candidateURL", ASSET_CANDIDATE_URL)
    root.SetCustomDataByKey("handson:candidateLicense", ASSET_CANDIDATE_LICENSE)
    root.SetCustomDataByKey("handson:sourceAsset", str(source_asset))

    visual = UsdGeom.Xform.Define(stage, f"{root_path}/visual_asset").GetPrim()
    visual.GetReferences().AddReference(
        Sdf.Reference(assetPath=str(source_asset.resolve()), primPath="/Asset")
    )
    visual.SetInstanceable(False)
    visual.SetCustomDataByKey("handson:classification", "task_visible")

    UsdPhysics.RigidBodyAPI.Apply(root)
    schemas.define_rigid_body_properties(
        prim_path=root_path,
        cfg=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.16,
            angular_damping=0.80,
            max_depenetration_velocity=0.6,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=4,
            sleep_threshold=0.005,
            stabilization_threshold=0.002,
        ),
    )
    PhysxSchema.PhysxRigidBodyAPI.Apply(
        root
    ).CreateEnableSpeculativeCCDAttr().Set(True)
    schemas.define_mass_properties(
        prim_path=root_path,
        cfg=sim_utils.MassPropertiesCfg(mass=0.18),
    )
    # The converted source mesh is not density-symmetric even though its
    # measured visual bounds are centered.  Letting PhysX infer the center of
    # mass from that mesh moves the reported rigid-body pose during the first
    # table contact (most noticeably for block_b).  BC021 poses are defined at
    # the measured geometry center, so author the inertial frame explicitly in
    # that same local frame.
    mass_api = UsdPhysics.MassAPI.Apply(root)
    mass_api.CreateMassAttr().Set(0.18)
    mass_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    inertia = (
        0.18 / 3.0 * (float(half_extents[1]) ** 2 + float(half_extents[2]) ** 2),
        0.18 / 3.0 * (float(half_extents[0]) ** 2 + float(half_extents[2]) ** 2),
        0.18 / 3.0 * (float(half_extents[0]) ** 2 + float(half_extents[1]) ** 2),
    )
    mass_api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*inertia))
    mass_api.CreatePrincipalAxesAttr().Set(
        Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0))
    )
    report = PhysxSchema.PhysxContactReportAPI.Apply(root)
    report.CreateThresholdAttr().Set(0.0)
    _bind_material(
        root_path,
        sim_utils=sim_utils,
        static_friction=1.10,
        dynamic_friction=0.90,
    )

    colliders = [
        prim
        for prim in Usd.PrimRange(root)
        if prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    if len(colliders) != 1:
        raise RuntimeError(
            f"BC021 {name} selected asset must expose one invisible collider; "
            f"got {len(colliders)}: {[str(prim.GetPath()) for prim in colliders]}"
        )
    collider = colliders[0]
    collider.SetCustomDataByKey("handson:classification", "formal_collider")
    collider.SetCustomDataByKey("handson:derivedFromVisibleBounds", True)
    collider_visibility = UsdGeom.Imageable(collider).ComputeVisibility()
    if collider_visibility != UsdGeom.Tokens.invisible:
        raise RuntimeError(
            f"BC021 {name} collider is visibly rendered: {collider.GetPath()}"
        )

    marker_path = f"{root_path}/marked_face_visual"
    marker = UsdGeom.Cube.Define(stage, marker_path).GetPrim()
    UsdGeom.Cube(marker).CreateSizeAttr(1.0)
    marker_center = _marked_face_marker_center_local(side)
    _set_xform(
        marker,
        translation=tuple(float(value) for value in marker_center),
        orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
        scale=(
            MARKER_THICKNESS_M,
            2.0 * float(patch_half_extents[0]),
            2.0 * float(patch_half_extents[1]),
        ),
    )
    _set_display_color(marker, marker_color)
    marker.SetCustomDataByKey("handson:classification", "task_visible_marker")
    marker.SetCustomDataByKey("handson:collisionEnabled", False)
    marker.SetCustomDataByKey(
        "handson:surfaceOffsetM",
        MARKER_SURFACE_OFFSET_M,
    )
    if marker.HasAPI(UsdPhysics.CollisionAPI):
        raise RuntimeError(f"BC021 marker unexpectedly has collision: {marker_path}")

    semantic_path = f"{root_path}/marked_face_frame"
    semantic = UsdGeom.Xform.Define(stage, semantic_path).GetPrim()
    _set_xform(
        semantic,
        translation=tuple(float(value) for value in marked_center),
        orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
    )
    semantic.SetCustomDataByKey(
        "handson:normalLocal",
        Gf.Vec3d(*(float(value) for value in marked_normal)),
    )
    semantic.SetCustomDataByKey("handson:tangentULocal", Gf.Vec3d(*tangent_u))
    semantic.SetCustomDataByKey("handson:tangentVLocal", Gf.Vec3d(*tangent_v))
    semantic.SetCustomDataByKey(
        "handson:patchHalfExtentsM",
        Gf.Vec2d(
            float(patch_half_extents[0]),
            float(patch_half_extents[1]),
        ),
    )
    semantic.SetCustomDataByKey("handson:classification", "semantic_frame")
    return root_path, len(colliders)


def spawn_bc021_task_assets(stage: Any, sim_utils: Any) -> B320SpawnResult:
    """Author the isolated BC021 scene from the selected wooden-block USDs."""

    from pxr import UsdGeom

    root = UsdGeom.Xform.Define(stage, B320_ASSET_ROOT_PATH).GetPrim()
    root.SetCustomDataByKey("handson:task", TASK_NAME)
    root.SetCustomDataByKey("handson:taskId", TASK_ID)
    root.SetCustomDataByKey("handson:taskRequirement", TASK_REQUIREMENT)
    root.SetCustomDataByKey("handson:assetIntegrationStatus", ASSET_INTEGRATION_STATUS)
    root.SetCustomDataByKey("handson:geometryContractFinal", True)
    root.SetCustomDataByKey("handson:geometrySource", "FINAL_GEOMETRY")

    table_path = _spawn_worktable(stage, sim_utils, task_name=TASK_NAME)
    result = B320SpawnResult(static_paths=(table_path,))
    for name, side, marker_color in (
        ("block_a", "left", (0.02, 0.20, 0.98)),
        ("block_b", "right", (1.00, 0.18, 0.02)),
    ):
        body_path, collider_count = _spawn_downloaded_block(
            stage,
            sim_utils,
            name=name,
            home_pose_wxyz=BLOCK_HOME_POSES_WXYZ[name],
            side=side,
            marker_color=marker_color,
        )
        result.body_paths[name] = body_path
        result.collider_counts[name] = collider_count
    return result


__all__ = [
    "ASSET_CANDIDATE_LICENSE",
    "ASSET_CANDIDATE_URL",
    "ASSET_INTEGRATION_STATUS",
    "ASSET_DIRECTORY",
    "ASSET_WRAPPERS",
    "BLOCK_HOME_POSES_WXYZ",
    "FINAL_GEOMETRY",
    "TASK_REQUIREMENT",
    "spawn_bc021_task_assets",
]
