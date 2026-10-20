"""Isaac Sim 5.1 runtime for the AS019 core scene only.

No handover trajectory, grasp primitive, or task-success motion is implemented
here.  The live audit holds both robots at their default joint targets and
measures the dynamic screwdriver during a bounded zero-action interval.
"""

from __future__ import annotations

import json
import importlib.metadata
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .scene_contract import (
    AS019_TABLETOP_ARM_HOME_RAD,
    AUTHORITATIVE_EXCEL_ROW,
    AUTHORITATIVE_LEFT_ROLE_ZH,
    AUTHORITATIVE_OBJECTIVE_ZH,
    AUTHORITATIVE_RIGHT_ROLE_ZH,
    AUTHORITATIVE_SOURCE_ROWS_SHA256,
    CONTROL_HZ,
    HEAD_CAMERA_EYE_M,
    HEAD_CAMERA_PRIM_PATH,
    HEAD_CAMERA_RESOLUTION_HW,
    HEAD_CAMERA_TARGET_M,
    MARKER_COLLISION_ENABLED,
    MARKER_ORIENTATION_WXYZ,
    MARKER_POSITION_M,
    MARKER_SIZE_M,
    MARKER_RANDOMIZATION_XY_M,
    MAX_FINAL_ANGULAR_SPEED_RADPS,
    MAX_FINAL_LINEAR_SPEED_MPS,
    MAX_ORIENTATION_DRIFT_RAD,
    MAX_POSITION_DRIFT_M,
    OVERLAY_RELATIVE,
    PHYSICS_HZ,
    PREVIEW_RELATIVE,
    REPOSITORY_ROOT,
    ROBOT_RELATIVE,
    ASSET_MANIFEST_RELATIVE,
    RUNTIME_USD_RELATIVE,
    RUNTIME_COLLIDER_PATHS,
    RUNTIME_PHYSICS_MATERIAL_PATH,
    RUNTIME_RIGID_OWNER,
    RUNTIME_TEXTURE_RELATIVE_BY_SOURCE_MEMBER,
    RUNTIME_VISUAL_MESH_PATHS,
    SCREWDRIVER_COLLIDER_APPROXIMATION,
    SCREWDRIVER_CENTER_OF_MASS_M,
    SCREWDRIVER_CONTACT_OFFSET_M,
    SCREWDRIVER_DEFAULT_FRICTION,
    SCREWDRIVER_EXPECTED_COLLIDER_COUNT,
    SCREWDRIVER_ANGULAR_DAMPING,
    SCREWDRIVER_LINEAR_DAMPING,
    SCREWDRIVER_MASS_KG,
    SCREWDRIVER_MASS_MULTIPLIER_RANGE,
    SCREWDRIVER_ORIENTATION_WXYZ,
    SCREWDRIVER_POSITION_M,
    SCREWDRIVER_REST_OFFSET_M,
    SCREWDRIVER_SAFE_AXIS_WORLD,
    SCREWDRIVER_SCALE_XYZ,
    SCREWDRIVER_STARTS_ASLEEP,
    SCREWDRIVER_SLEEP_THRESHOLD,
    SCREWDRIVER_SOLVER_POSITION_ITERATIONS,
    SCREWDRIVER_SOLVER_VELOCITY_ITERATIONS,
    SCREWDRIVER_STABILIZATION_THRESHOLD,
    SCREWDRIVER_FRICTION_RANGE,
    SCREWDRIVER_RANDOMIZATION_XY_M,
    SCREWDRIVER_RANDOMIZATION_YAW_RAD,
    SETTLE_STEPS,
    SOURCE_ARCHIVE_BYTES,
    SOURCE_ARCHIVE_RELATIVE,
    SOURCE_ARCHIVE_SHA256,
    SOURCE_MEMBER_SHA256,
    SUMMARY_RELATIVE,
    TABLE_TOP_Z_M,
    TASK_DIR,
    TASK_ID,
    TASK_NAME,
    ZERO_ACTION_RELATIVE,
    ZERO_ACTION_SECONDS,
    ZERO_ACTION_STEPS,
    contract_errors,
    frame_metrics,
    sha256_file,
)


_LEFT_FINGER_FRICTION = 3.0
_EXPECTED_ISAAC_SIM_VERSION = "5.1.0.0"


def _isaac_sim_runtime_version() -> tuple[str, str]:
    """Return the canonical Isaac version and the evidence used for it."""

    try:
        return importlib.metadata.version("isaacsim"), "distribution metadata"
    except importlib.metadata.PackageNotFoundError:
        isaac_path = os.environ.get("ISAAC_PATH")
        version_file = Path(isaac_path) / "VERSION" if isaac_path else None
        if version_file is None or not version_file.is_file():
            raise RuntimeError(
                "AS019 cannot determine Isaac Sim version: the isaacsim "
                "distribution metadata and ISAAC_PATH/VERSION are both missing"
            ) from None
        raw_version = version_file.read_text(encoding="utf-8").splitlines()[0].strip()
        core_version = raw_version.split("+", 1)[0].split("-", 1)[0]
        if core_version != "5.1.0":
            raise RuntimeError(
                "AS019 requires Isaac Sim 5.1.0.0, found bundled VERSION "
                f"{raw_version!r}"
            )
        return _EXPECTED_ISAAC_SIM_VERSION, f"bundled VERSION {raw_version!r}"


def _apply_as019_tabletop_arm_home(cfg: Any) -> None:
    """Install the shared mirrored bimanual home in the AS019 scene config."""

    for side in ("right", "left"):
        robot_cfg = getattr(cfg, f"{side}_robot", None)
        init_state = getattr(robot_cfg, "init_state", None)
        joint_pos = getattr(init_state, "joint_pos", None)
        if not isinstance(joint_pos, dict):
            raise RuntimeError(f"AS019 {side} robot joint home is not configurable")
        for index, value in enumerate(AS019_TABLETOP_ARM_HOME_RAD[side], start=1):
            key = f"joint{index}_{side}(_1)?"
            if key not in joint_pos:
                raise RuntimeError(f"AS019 {side} robot home is missing {key}")
            joint_pos[key] = float(value)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def ensure_converted_screwdriver(simulation_app: Any) -> tuple[Path, dict[str, Any]]:
    """Validate and return the committed, source-derived screwdriver asset.

    Asset conversion is an explicit offline build step.  Collection startup
    must never silently rewrite a reviewed visual/collision closure, because
    doing so would change contact geometry without changing the recording
    provenance visible to the operator.
    """

    live_stage_validation = simulation_app is not None
    del simulation_app
    source = REPOSITORY_ROOT / SOURCE_ARCHIVE_RELATIVE
    output = REPOSITORY_ROOT / RUNTIME_USD_RELATIVE
    metadata = REPOSITORY_ROOT / ASSET_MANIFEST_RELATIVE
    errors = contract_errors()
    if errors:
        raise RuntimeError("; ".join(errors))
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    runtime = payload.get("runtime")
    provenance = payload.get("provenance")
    if not isinstance(runtime, dict) or not isinstance(provenance, dict):
        raise RuntimeError("AS019 asset manifest lacks runtime/provenance records")
    required = {
        "path": str(RUNTIME_USD_RELATIVE),
        "sha256": sha256_file(output),
        "size_bytes": output.stat().st_size,
        "rigid_owner": RUNTIME_RIGID_OWNER,
        "visual_mesh_paths": list(RUNTIME_VISUAL_MESH_PATHS),
        "collider_paths": list(RUNTIME_COLLIDER_PATHS),
        "collider_count": SCREWDRIVER_EXPECTED_COLLIDER_COUNT,
        "collision_approximation": SCREWDRIVER_COLLIDER_APPROXIMATION,
        "mass_kg": SCREWDRIVER_MASS_KG,
        "starts_asleep": SCREWDRIVER_STARTS_ASLEEP,
        "physics_material_path": RUNTIME_PHYSICS_MATERIAL_PATH,
        "default_static_friction": SCREWDRIVER_DEFAULT_FRICTION,
        "default_dynamic_friction": SCREWDRIVER_DEFAULT_FRICTION,
        "default_restitution": 0.0,
        "isaac_sim_version": "5.1.0.0",
        "stage_meters_per_unit": 1.0,
        "stage_up_axis": "Z",
        "default_prim": RUNTIME_RIGID_OWNER,
    }
    for key, expected in required.items():
        if runtime.get(key) != expected:
            raise RuntimeError(
                f"AS019 asset manifest runtime {key!r} mismatch: "
                f"{runtime.get(key)!r} != {expected!r}"
            )
    if not np.allclose(
        runtime.get("center_of_mass_m", ()),
        SCREWDRIVER_CENTER_OF_MASS_M,
        atol=1.0e-9,
        rtol=0.0,
    ):
        raise RuntimeError("AS019 asset manifest center of mass mismatch")
    if provenance.get("archive_sha256") != SOURCE_ARCHIVE_SHA256:
        raise RuntimeError("AS019 asset manifest archive hash mismatch")
    members = payload.get("source", {}).get("members", {})
    if isinstance(members, list):
        members = {
            str(record.get("path")): record
            for record in members
            if isinstance(record, dict) and record.get("path")
        }
    if not isinstance(members, dict):
        raise RuntimeError("AS019 asset manifest source members are invalid")
    for member, expected_hash in SOURCE_MEMBER_SHA256.items():
        record = members.get(member)
        actual_hash = record.get("sha256") if isinstance(record, dict) else record
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"AS019 asset manifest member hash mismatch for {member!r}"
            )
    for member, relative_texture in RUNTIME_TEXTURE_RELATIVE_BY_SOURCE_MEMBER.items():
        texture = REPOSITORY_ROOT / relative_texture
        if not texture.is_file():
            raise RuntimeError(f"AS019 runtime texture is missing: {relative_texture}")
        if sha256_file(texture) != SOURCE_MEMBER_SHA256[member]:
            raise RuntimeError(f"AS019 runtime texture hash mismatch: {relative_texture}")
    running_version, version_source = _isaac_sim_runtime_version()
    if running_version != _EXPECTED_ISAAC_SIM_VERSION:
        raise RuntimeError(
            f"AS019 requires Isaac Sim 5.1.0.0, found {running_version!r}"
        )
    print(
        f"[HandsOn/IsaacLab] AS019 Isaac Sim version validated: "
        f"{running_version} ({version_source})",
        flush=True,
    )
    if live_stage_validation:
        from pxr import Usd, UsdGeom

        runtime_stage = Usd.Stage.Open(str(output), load=Usd.Stage.LoadNone)
        if runtime_stage is None:
            raise RuntimeError("AS019 runtime USD cannot be opened")
        default_prim = runtime_stage.GetDefaultPrim()
        if (
            not default_prim.IsValid()
            or str(default_prim.GetPath()) != RUNTIME_RIGID_OWNER
        ):
            raise RuntimeError("AS019 runtime USD defaultPrim mismatch")
        if not np.isclose(UsdGeom.GetStageMetersPerUnit(runtime_stage), 1.0):
            raise RuntimeError("AS019 runtime USD metersPerUnit mismatch")
        if str(UsdGeom.GetStageUpAxis(runtime_stage)) != "Z":
            raise RuntimeError("AS019 runtime USD upAxis mismatch")
    if source.stat().st_size != SOURCE_ARCHIVE_BYTES:
        raise RuntimeError("AS019 source archive byte size mismatch")
    conversion = dict(runtime)
    conversion["output_sha256"] = runtime["sha256"]
    conversion["output_bytes"] = runtime["size_bytes"]
    return output, conversion


def make_scene_cfg(
    runtime_usd: Path,
    *,
    include_cameras: bool = True,
    include_wrist_cameras: bool = False,
):
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg

    from handson.sim.isaaclab.bimanual_assembly_scene import (
        make_bimanual_assembly_scene_cfg,
    )

    cfg = make_bimanual_assembly_scene_cfg(
        num_envs=1,
        env_spacing=2.5,
        physics_backend="physx",
    )
    _apply_as019_tabletop_arm_home(cfg)
    # The table is kinematic, so enabling contact reporting on it does not
    # interfere with dynamic-body sleep.  This is the scene prerequisite for
    # task-owned tool/table and robot/table contact views; the environment must
    # still create and validate those exact filtered views before their values
    # may be used as success or failure evidence.
    cfg.table.spawn.activate_contact_sensors = True
    # The source Wuji MJCF specifies high-friction finger pads, but the shared
    # USD conversion path does not preserve that material.  Bind the authored
    # value only to this task's left-finger collision roots after spawning; do
    # not alter the global robot asset or any collision geometry.
    original_left_spawn = cfg.left_robot.spawn.func

    def _spawn_left_with_as019_finger_material(
        prim_path, spawn_cfg, *spawn_args, **spawn_kwargs
    ):
        left_root = original_left_spawn(
            prim_path, spawn_cfg, *spawn_args, **spawn_kwargs
        )
        from pxr import Usd, UsdPhysics, UsdShade
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        binding_paths: list[str] = []
        for prim in Usd.PrimRange(left_root, Usd.TraverseInstanceProxies()):
            path = str(prim.GetPath())
            if (
                "left_finger" not in path
                or not prim.HasAPI(UsdPhysics.CollisionAPI)
            ):
                continue
            binding_prim = prim
            while binding_prim.IsInstanceProxy():
                binding_prim = binding_prim.GetParent()
            binding_path = str(binding_prim.GetPath())
            if binding_path not in binding_paths:
                binding_paths.append(binding_path)
        material = UsdShade.Material.Define(
            stage,
            f"{left_root.GetPath()}/Looks/as019_left_finger_physics_material",
        )
        material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        material_api.CreateStaticFrictionAttr().Set(_LEFT_FINGER_FRICTION)
        material_api.CreateDynamicFrictionAttr().Set(_LEFT_FINGER_FRICTION)
        material_api.CreateRestitutionAttr().Set(0.0)
        for binding_path in binding_paths:
            UsdShade.MaterialBindingAPI.Apply(
                stage.GetPrimAtPath(binding_path)
            ).Bind(material, materialPurpose="physics")
        return left_root

    cfg.left_robot.spawn.func = _spawn_left_with_as019_finger_material
    cfg.peg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Screwdriver",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(runtime_usd),
            scale=SCREWDRIVER_SCALE_XYZ,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,
                linear_damping=SCREWDRIVER_LINEAR_DAMPING,
                angular_damping=SCREWDRIVER_ANGULAR_DAMPING,
                sleep_threshold=SCREWDRIVER_SLEEP_THRESHOLD,
                stabilization_threshold=SCREWDRIVER_STABILIZATION_THRESHOLD,
                solver_position_iteration_count=(
                    SCREWDRIVER_SOLVER_POSITION_ITERATIONS
                ),
                solver_velocity_iteration_count=(
                    SCREWDRIVER_SOLVER_VELOCITY_ITERATIONS
                ),
                max_linear_velocity=5.0,
                max_angular_velocity=20.0,
                max_depenetration_velocity=1.0,
                enable_gyroscopic_forces=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=SCREWDRIVER_CONTACT_OFFSET_M,
                rest_offset=SCREWDRIVER_REST_OFFSET_M,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=SCREWDRIVER_MASS_KG),
            # AS019 owns filtered PhysX contact views and the converted asset
            # already carries ContactReportAPI.  IsaacLab's generic activation
            # helper forcibly rewrites sleepThreshold to 0.0, preventing the
            # untouched tabletop tool from ever reaching the collection
            # platform's stable-origin gate.  Sleeping bodies wake normally on
            # hand collision, so keep the task contact path without disabling
            # physical sleep.
            activate_contact_sensors=False,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=SCREWDRIVER_POSITION_M,
            rot=SCREWDRIVER_ORIENTATION_WXYZ,
        ),
    )
    cfg.socket = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SafeOrientationMarker",
        spawn=sim_utils.CuboidCfg(
            size=MARKER_SIZE_M,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.04, 0.95, 0.12),
                emissive_color=(0.02, 0.35, 0.04),
                opacity=1.0,
                roughness=0.55,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=MARKER_POSITION_M,
            rot=MARKER_ORIENTATION_WXYZ,
        ),
    )
    if include_cameras:
        cfg.head_camera.prim_path = "{ENV_REGEX_NS}/HeadRecordCamera"
        cfg.head_camera.height = HEAD_CAMERA_RESOLUTION_HW[0]
        cfg.head_camera.width = HEAD_CAMERA_RESOLUTION_HW[1]
    else:
        cfg.head_camera = None
    if not include_cameras or not include_wrist_cameras:
        cfg.wrist_left_camera = None
        cfg.wrist_right_camera = None
    return cfg


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _pose_record(pose: Any, velocity: Any) -> dict[str, Any]:
    pose_np = _to_numpy(pose).reshape(-1)
    velocity_np = _to_numpy(velocity).reshape(-1)
    return {
        "position_m": [float(v) for v in pose_np[:3]],
        "orientation_runtime_order": [float(v) for v in pose_np[3:7]],
        "linear_velocity_mps": [float(v) for v in velocity_np[:3]],
        "angular_velocity_radps": [float(v) for v in velocity_np[3:6]],
    }


def _orientation_delta(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64).reshape(4)
    second = np.asarray(second, dtype=np.float64).reshape(4)
    alignment = float(
        np.clip(
            abs(np.dot(first, second))
            / max(np.linalg.norm(first) * np.linalg.norm(second), 1.0e-12),
            0.0,
            1.0,
        )
    )
    return float(2.0 * math.acos(alignment))


def _hold_robot_defaults(scene: Any) -> None:
    import torch

    for name in ("right_robot", "left_robot"):
        robot = scene[name]
        target = robot.data.default_joint_pos.clone()
        robot.set_joint_position_target(target)
        if not bool(torch.all(torch.isfinite(target))):
            raise RuntimeError(f"{name} default target contains non-finite values")


def _step(scene: Any, simulation: Any, steps: int, *, render_every: int = 0) -> None:
    for index in range(int(steps)):
        _hold_robot_defaults(scene)
        scene.write_data_to_sim()
        simulation.step(render=bool(render_every and index % render_every == 0))
        scene.update(float(simulation.get_physics_dt()))


def _set_head_camera(scene: Any) -> None:
    import torch

    camera = scene["head_camera"]
    device = scene["peg"].data.root_pos_w.device
    eyes = torch.tensor([HEAD_CAMERA_EYE_M], dtype=torch.float32, device=device)
    targets = torch.tensor([HEAD_CAMERA_TARGET_M], dtype=torch.float32, device=device)
    camera.set_world_poses_from_view(eyes=eyes, targets=targets)


def _capture_head(scene: Any, simulation: Any) -> np.ndarray:
    camera = scene["head_camera"]
    _set_head_camera(scene)
    for _ in range(5):
        simulation.render()
        camera.update(1.0 / CONTROL_HZ)
    output = camera.data.output.get("rgb")
    if output is None:
        raise RuntimeError("canonical AS019 head camera has no RGB output")
    frame = _to_numpy(output)[0]
    if frame.ndim != 3 or frame.shape[-1] < 3:
        raise RuntimeError(f"invalid AS019 head frame shape: {frame.shape}")
    frame = frame[..., :3]
    if np.issubdtype(frame.dtype, np.floating):
        maximum = float(np.nanmax(frame)) if frame.size else 0.0
        frame = np.clip(frame * (255.0 if maximum <= 1.0 else 1.0), 0.0, 255.0)
    return np.asarray(frame, dtype=np.uint8)


def _author_overlay(runtime_usd: Path) -> Path:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    path = REPOSITORY_ROOT / OVERLAY_RELATIVE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/World/AS019Core")
    stage.SetDefaultPrim(root.GetPrim())
    root.GetPrim().CreateAttribute("as019:scope", Sdf.ValueTypeNames.String).Set(
        "core_scene_only"
    )
    root.GetPrim().CreateAttribute("as019:sharedBase", Sdf.ValueTypeNames.String).Set(
        "handson.sim.isaaclab.bimanual_assembly_scene"
    )
    root.GetPrim().CreateAttribute("as019:taskComplete", Sdf.ValueTypeNames.Bool).Set(False)

    # Keep task placement on a parent prim.  The converted asset's default
    # prim carries its own identity translate/orient/scale ops; referencing it
    # directly on this prim would compose those ops and make AddTranslateOp()
    # collide with an existing xformOp:translate.
    screwdriver = UsdGeom.Xform.Define(stage, "/World/AS019Core/Screwdriver")
    screwdriver.AddTranslateOp().Set(Gf.Vec3d(*SCREWDRIVER_POSITION_M))
    screwdriver.AddOrientOp().Set(Gf.Quatf(*SCREWDRIVER_ORIENTATION_WXYZ))
    screwdriver.AddScaleOp().Set(Gf.Vec3f(*SCREWDRIVER_SCALE_XYZ))
    screwdriver_asset = UsdGeom.Xform.Define(
        stage, "/World/AS019Core/Screwdriver/Asset"
    )
    screwdriver_asset.GetPrim().GetReferences().AddReference(
        os.path.relpath(runtime_usd, path.parent)
    )

    marker = UsdGeom.Cube.Define(stage, "/World/AS019Core/SafeOrientationMarker")
    marker.CreateSizeAttr(1.0)
    marker.AddTranslateOp().Set(Gf.Vec3d(*MARKER_POSITION_M))
    marker.AddOrientOp().Set(Gf.Quatf(*MARKER_ORIENTATION_WXYZ))
    marker.AddScaleOp().Set(Gf.Vec3f(*(0.5 * np.asarray(MARKER_SIZE_M))))
    marker.GetPrim().CreateAttribute("as019:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(
        MARKER_COLLISION_ENABLED
    )
    material = UsdShade.Material.Define(stage, "/World/AS019Core/Looks/SafeMarker")
    shader = UsdShade.Shader.Define(stage, "/World/AS019Core/Looks/SafeMarker/Preview")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.04, 0.95, 0.12)
    )
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.02, 0.35, 0.04)
    )
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.55)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(marker.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()
    # USD's text writer appends an extra blank line.  Normalize the delivered
    # layer so repository whitespace checks remain reproducible.
    path.write_text(path.read_text(encoding="utf-8").rstrip() + "\n", encoding="utf-8")
    forbidden_shared_prefix = "/" + "mnt" + "/" + "pfs" + "/"
    if forbidden_shared_prefix in path.read_text(encoding="utf-8"):
        raise RuntimeError("delivered overlay contains a hardcoded user path")
    return path


def _stage_inventory(stage: Any, conversion: dict[str, Any]) -> dict[str, Any]:
    from pxr import Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

    screwdriver = stage.GetPrimAtPath("/World/envs/env_0/Screwdriver")
    if not screwdriver.IsValid():
        raise RuntimeError("live AS019 stage is missing the screwdriver root")
    screwdriver_prims = list(
        Usd.PrimRange(screwdriver, Usd.TraverseInstanceProxies())
    )

    def _collision_enabled(prim: Any) -> bool:
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            return False
        value = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
        return value is not False

    collider_paths = [
        str(prim.GetPath()) for prim in screwdriver_prims if _collision_enabled(prim)
    ]
    visible_mesh_paths = [
        str(prim.GetPath())
        for prim in screwdriver_prims
        if prim.IsA(UsdGeom.Mesh)
        and not _collision_enabled(prim)
        and UsdGeom.Imageable(prim).ComputeVisibility(Usd.TimeCode.Default())
        != UsdGeom.Tokens.invisible
    ]
    hidden_collider_paths = [
        str(prim.GetPath())
        for prim in screwdriver_prims
        if _collision_enabled(prim)
        and prim.IsA(UsdGeom.Imageable)
        and UsdGeom.Imageable(prim).ComputeVisibility(Usd.TimeCode.Default())
        == UsdGeom.Tokens.invisible
    ]
    rigid_body_paths = [
        str(prim.GetPath())
        for prim in screwdriver_prims
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
        and UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get() is not False
    ]

    expected_owner = str(conversion["rigid_owner"])

    def _expected_suffix(path: str) -> str:
        if not path.startswith(expected_owner):
            raise RuntimeError(
                f"AS019 asset path {path!r} is outside rigid owner {expected_owner!r}"
            )
        return path[len(expected_owner) :]

    expected_collider_suffixes = {
        _expected_suffix(str(path)) for path in conversion["collider_paths"]
    }
    expected_visual_suffixes = {
        _expected_suffix(str(path)) for path in conversion["visual_mesh_paths"]
    }
    live_collider_suffixes = {
        path.removeprefix(str(screwdriver.GetPath())) for path in collider_paths
    }
    live_visual_suffixes = {
        path.removeprefix(str(screwdriver.GetPath())) for path in visible_mesh_paths
    }

    marker = stage.GetPrimAtPath("/World/envs/env_0/SafeOrientationMarker")
    marker_position: list[float] = []
    marker_orientation: list[float] = []
    marker_world_aabb_size: list[float] = []
    marker_pose_matches_contract = False
    marker_size_matches_contract = False
    marker_material_matches_contract = False
    if marker.IsValid() and marker.IsA(UsdGeom.Xformable):
        marker_transform = UsdGeom.Xformable(
            marker
        ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        translation = marker_transform.ExtractTranslation()
        quaternion = marker_transform.ExtractRotationQuat()
        imaginary = quaternion.GetImaginary()
        marker_position = [float(translation[index]) for index in range(3)]
        marker_orientation = [
            float(quaternion.GetReal()),
            *(float(imaginary[index]) for index in range(3)),
        ]
        actual_quaternion = np.asarray(marker_orientation, dtype=np.float64)
        expected_quaternion = np.asarray(
            MARKER_ORIENTATION_WXYZ, dtype=np.float64
        )
        actual_quaternion /= max(
            float(np.linalg.norm(actual_quaternion)), 1.0e-12
        )
        expected_quaternion /= max(
            float(np.linalg.norm(expected_quaternion)), 1.0e-12
        )
        marker_pose_matches_contract = bool(
            np.allclose(marker_position, MARKER_POSITION_M, atol=1.0e-6)
            and abs(float(np.dot(actual_quaternion, expected_quaternion)))
            >= 1.0 - 1.0e-6
        )
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        )
        aligned_range = cache.ComputeWorldBound(marker).ComputeAlignedRange()
        marker_world_aabb_size = [
            float(value)
            for value in (aligned_range.GetMax() - aligned_range.GetMin())
        ]
        w, x, y, z = expected_quaternion
        expected_rotation = np.asarray(
            [
                [
                    1.0 - 2.0 * (y * y + z * z),
                    2.0 * (x * y - z * w),
                    2.0 * (x * z + y * w),
                ],
                [
                    2.0 * (x * y + z * w),
                    1.0 - 2.0 * (x * x + z * z),
                    2.0 * (y * z - x * w),
                ],
                [
                    2.0 * (x * z - y * w),
                    2.0 * (y * z + x * w),
                    1.0 - 2.0 * (x * x + y * y),
                ],
            ],
            dtype=np.float64,
        )
        expected_world_aabb_size = np.abs(expected_rotation) @ np.asarray(
            MARKER_SIZE_M, dtype=np.float64
        )
        marker_size_matches_contract = bool(
            np.allclose(
                marker_world_aabb_size,
                expected_world_aabb_size,
                atol=1.0e-6,
                rtol=0.0,
            )
        )
        try:
            shader = None
            for candidate in Usd.PrimRange(marker, Usd.TraverseInstanceProxies()):
                material, _relationship = UsdShade.MaterialBindingAPI(
                    candidate
                ).ComputeBoundMaterial()
                if not material:
                    continue
                surface = material.ComputeSurfaceSource()
                source = surface[0] if surface else None
                if source:
                    shader = UsdShade.Shader(source.GetPrim())
                    break

            def _shader_input(name: str) -> Any:
                return shader.GetInput(name).Get() if shader else None

            marker_material_matches_contract = bool(
                np.allclose(
                    _shader_input("diffuseColor"),
                    (0.04, 0.95, 0.12),
                    atol=1.0e-5,
                )
                and np.allclose(
                    _shader_input("emissiveColor"),
                    (0.02, 0.35, 0.04),
                    atol=1.0e-5,
                )
                and float(_shader_input("opacity")) == 1.0
                and np.isclose(float(_shader_input("roughness")), 0.55)
            )
        except (AttributeError, TypeError, ValueError):
            marker_material_matches_contract = False
    marker_colliders = [
        str(prim.GetPath())
        for prim in Usd.PrimRange(marker, Usd.TraverseInstanceProxies())
        if prim.HasAPI(UsdPhysics.CollisionAPI)
        and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
    ]
    active_domes = [
        str(prim.GetPath()) for prim in stage.Traverse() if prim.IsA(UsdLux.DomeLight)
    ]
    active_distants = [
        str(prim.GetPath()) for prim in stage.Traverse() if prim.IsA(UsdLux.DistantLight)
    ]
    cameras = [
        str(prim.GetPath()) for prim in stage.Traverse() if prim.IsA(UsdGeom.Camera)
    ]
    return {
        "screwdriver": {
            "prim_path": "/World/envs/env_0/Screwdriver",
            "scale_xyz": list(SCREWDRIVER_SCALE_XYZ),
            "reset_position_m": list(SCREWDRIVER_POSITION_M),
            "reset_orientation_wxyz": list(SCREWDRIVER_ORIENTATION_WXYZ),
            "safe_tip_axis_world": list(SCREWDRIVER_SAFE_AXIS_WORLD),
            "rigid_body_paths": rigid_body_paths,
            "rigid_body_count": len(rigid_body_paths),
            "visual_mesh_paths": visible_mesh_paths,
            "visual_mesh_count": len(visible_mesh_paths),
            "visual_collision_free": not any(
                path in collider_paths for path in visible_mesh_paths
            ),
            "collider_paths": collider_paths,
            "collider_count": len(collider_paths),
            "colliders_hidden": set(hidden_collider_paths) == set(collider_paths),
            "paths_match_asset_manifest": bool(
                live_collider_suffixes == expected_collider_suffixes
                and live_visual_suffixes == expected_visual_suffixes
            ),
            "collision_approximation": conversion["collision_approximation"],
            "mass_kg": float(conversion["mass_kg"]),
        },
        "marker": {
            "prim_path": str(marker.GetPath()),
            "position_m": marker_position,
            "orientation_wxyz": marker_orientation,
            "size_m": list(MARKER_SIZE_M),
            "world_aabb_size_m": marker_world_aabb_size,
            "collision_enabled": bool(marker_colliders),
            "collider_paths": marker_colliders,
            "pose_matches_contract": marker_pose_matches_contract,
            "size_matches_contract": marker_size_matches_contract,
            "material_matches_contract": marker_material_matches_contract,
        },
        "dome_lights": active_domes,
        "distant_lights": active_distants,
        "camera_paths": cameras,
    }


def _write_manifest(
    *,
    conversion: dict[str, Any],
    overlay: Path,
    preview: Path,
    summary_path: Path,
    zero_action_path: Path,
    summary: dict[str, Any],
) -> Path:
    robot = REPOSITORY_ROOT / ROBOT_RELATIVE
    manifest_path = TASK_DIR / "scene_manifest.json"
    _atomic_json(
        manifest_path,
        {
            "schema": "as019_core_scene_handoff_v1",
            "task_id": TASK_ID,
            "task_name": TASK_NAME,
            "scope": "core_scene_only",
            "isaac_sim_version": "5.1.0.0",
            "authoritative_source": {
                "source_rows_sha256": AUTHORITATIVE_SOURCE_ROWS_SHA256,
                "excel_row": AUTHORITATIVE_EXCEL_ROW,
                "objective_zh": AUTHORITATIVE_OBJECTIVE_ZH,
                "left_role_zh": AUTHORITATIVE_LEFT_ROLE_ZH,
                "right_role_zh": AUTHORITATIVE_RIGHT_ROLE_ZH,
            },
            "components": {
                "robot": {
                    "asset": str(ROBOT_RELATIVE),
                    "sha256": sha256_file(robot),
                    "sides": ["left", "right"],
                    "arm_dof_per_side": 7,
                    "hand_dof_per_side": 20,
                },
                "table": {
                    "source": "handson.sim.isaaclab.bimanual_assembly_scene",
                    "top_height_m": TABLE_TOP_Z_M,
                    "kinematic": True,
                },
                "screwdriver": {
                    "source_archive": str(SOURCE_ARCHIVE_RELATIVE),
                    "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
                    "source_member_sha256": SOURCE_MEMBER_SHA256,
                    "asset_manifest": str(ASSET_MANIFEST_RELATIVE),
                    "asset_manifest_sha256": sha256_file(
                        REPOSITORY_ROOT / ASSET_MANIFEST_RELATIVE
                    ),
                    "runtime_asset": str(RUNTIME_USD_RELATIVE),
                    "runtime_sha256": conversion["output_sha256"],
                    "rigid_owner": conversion["rigid_owner"],
                    "visual_mesh_paths": conversion["visual_mesh_paths"],
                    "collider_paths": conversion["collider_paths"],
                    "scale_xyz": list(SCREWDRIVER_SCALE_XYZ),
                    "reset_position_m": list(SCREWDRIVER_POSITION_M),
                    "reset_orientation_wxyz": list(SCREWDRIVER_ORIENTATION_WXYZ),
                    "safe_tip_axis_world": list(SCREWDRIVER_SAFE_AXIS_WORLD),
                    "collider_count": conversion["collider_count"],
                    "collision_approximation": conversion["collision_approximation"],
                    "canonical_reset_only": True,
                    "randomization_contract": {
                        "xy_m": SCREWDRIVER_RANDOMIZATION_XY_M,
                        "yaw_rad": SCREWDRIVER_RANDOMIZATION_YAW_RAD,
                        "friction_range": list(SCREWDRIVER_FRICTION_RANGE),
                        "mass_multiplier_range": list(SCREWDRIVER_MASS_MULTIPLIER_RANGE),
                    },
                },
                "safe_orientation_marker": {
                    "asset": "procedural_usd_cube",
                    "overlay": str(OVERLAY_RELATIVE),
                    "position_m": list(MARKER_POSITION_M),
                    "orientation_wxyz": list(MARKER_ORIENTATION_WXYZ),
                    "size_m": list(MARKER_SIZE_M),
                    "collision_enabled": MARKER_COLLISION_ENABLED,
                    "collider_count": 0,
                    "randomization_xy_m": MARKER_RANDOMIZATION_XY_M,
                },
                "lighting": {
                    "dome_intensity": 650.0,
                    "distant_intensity": 950.0,
                    "render_mode": "RayTracedLighting",
                    "tonemap": "ACES_op4",
                },
                "head_camera": {
                    "prim_path": HEAD_CAMERA_PRIM_PATH,
                    "resolution_hwc": [*HEAD_CAMERA_RESOLUTION_HW, 3],
                    "preview": str(PREVIEW_RELATIVE),
                },
            },
            "live_core_scene_evidence": {
                "summary": str(summary_path.relative_to(REPOSITORY_ROOT)),
                "summary_sha256": sha256_file(summary_path),
                "zero_action": str(zero_action_path.relative_to(REPOSITORY_ROOT)),
                "zero_action_sha256": sha256_file(zero_action_path),
                "overlay": str(overlay.relative_to(REPOSITORY_ROOT)),
                "overlay_sha256": sha256_file(overlay),
                "preview_sha256": sha256_file(preview),
                "preview_bytes": preview.stat().st_size,
                "head_frame_metrics": summary["head_frame_metrics"],
                "zero_action_seconds": summary["zero_action"]["sim_seconds"],
                "screwdriver_position_drift_m": summary["zero_action"]["position_drift_m"],
                "screwdriver_orientation_drift_rad": summary["zero_action"]["orientation_drift_rad"],
                "object_state_writes_during_action": 0,
            },
            "claims": {
                "core_scene_ready": bool(summary["core_scene_pass"]),
                "motion_or_grasp_validated": False,
                "handover_path_validated": False,
                "task_complete": False,
            },
        },
    )
    return manifest_path


def run_core_scene(*, simulation_app: Any, output_dir: Path) -> dict[str, Any]:
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.scene import InteractiveScene
    from isaaclab.sim import SimulationContext
    import omni.usd

    from handson.sim.isaaclab.bimanual_assembly_scene import (
        apply_scene_visuals,
        fix_robot_bases,
    )
    from handson.sim.isaaclab.compat import initialize_interactive_scene, make_simulation_cfg

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_usd, conversion = ensure_converted_screwdriver(simulation_app)
    sim_cfg = make_simulation_cfg(
        sim_utils,
        physics_backend="physx",
        device="cuda:0",
        dt=1.0 / PHYSICS_HZ,
        render_interval=PHYSICS_HZ // CONTROL_HZ,
        gravity=(0.0, 0.0, -9.81),
        physx_kwargs={
            "solver_type": 1,
            "enable_enhanced_determinism": True,
            "max_position_iteration_count": 32,
            "max_velocity_iteration_count": 4,
            "bounce_threshold_velocity": 0.2,
            "friction_offset_threshold": 0.01,
            "friction_correlation_distance": 0.00625,
        },
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=0.8,
        ),
    )
    simulation = SimulationContext(sim_cfg)
    simulation.set_camera_view(eye=HEAD_CAMERA_EYE_M, target=HEAD_CAMERA_TARGET_M)
    scene = InteractiveScene(make_scene_cfg(runtime_usd))
    initialize_interactive_scene(simulation, scene)
    stage = omni.usd.get_context().get_stage()
    apply_scene_visuals(1)
    fix_robot_bases(1)
    simulation.reset()
    scene.reset()

    for name in ("right_robot", "left_robot"):
        robot = scene[name]
        default_position = robot.data.default_joint_pos.clone()
        default_velocity = torch.zeros_like(default_position)
        robot.write_joint_state_to_sim(default_position, default_velocity)
        robot.set_joint_position_target(default_position)
    _step(scene, simulation, SETTLE_STEPS, render_every=60)

    screwdriver = scene["peg"]
    initial_pose = _to_numpy(screwdriver.data.root_pose_w[0]).astype(np.float64)
    initial_velocity = _to_numpy(screwdriver.data.root_vel_w[0]).astype(np.float64)
    initial = _pose_record(initial_pose, initial_velocity)
    _step(scene, simulation, ZERO_ACTION_STEPS, render_every=120)
    final_pose = _to_numpy(screwdriver.data.root_pose_w[0]).astype(np.float64)
    final_velocity = _to_numpy(screwdriver.data.root_vel_w[0]).astype(np.float64)
    final = _pose_record(final_pose, final_velocity)

    position_drift = float(np.linalg.norm(final_pose[:3] - initial_pose[:3]))
    orientation_drift = _orientation_delta(initial_pose[3:7], final_pose[3:7])
    linear_speed = float(np.linalg.norm(final_velocity[:3]))
    angular_speed = float(np.linalg.norm(final_velocity[3:6]))
    finite = bool(
        np.all(np.isfinite(initial_pose))
        and np.all(np.isfinite(initial_velocity))
        and np.all(np.isfinite(final_pose))
        and np.all(np.isfinite(final_velocity))
    )
    zero_action_pass = bool(
        finite
        and position_drift <= MAX_POSITION_DRIFT_M
        and orientation_drift <= MAX_ORIENTATION_DRIFT_RAD
        and linear_speed <= MAX_FINAL_LINEAR_SPEED_MPS
        and angular_speed <= MAX_FINAL_ANGULAR_SPEED_RADPS
    )
    zero_action = {
        "schema": "as019_zero_action_v1",
        "physics_hz": PHYSICS_HZ,
        "control_hz": CONTROL_HZ,
        "settle_steps": SETTLE_STEPS,
        "zero_action_steps": ZERO_ACTION_STEPS,
        "sim_seconds": ZERO_ACTION_SECONDS,
        "robot_command": "hold_default_joint_targets",
        "object_state_writes": 0,
        "initial": initial,
        "final": final,
        "position_drift_m": position_drift,
        "orientation_drift_rad": orientation_drift,
        "final_linear_speed_mps": linear_speed,
        "final_angular_speed_radps": angular_speed,
        "finite": finite,
        "thresholds": {
            "max_position_drift_m": MAX_POSITION_DRIFT_M,
            "max_orientation_drift_rad": MAX_ORIENTATION_DRIFT_RAD,
            "max_final_linear_speed_mps": MAX_FINAL_LINEAR_SPEED_MPS,
            "max_final_angular_speed_radps": MAX_FINAL_ANGULAR_SPEED_RADPS,
        },
        "pass": zero_action_pass,
        "task_complete": False,
    }
    zero_action_path = REPOSITORY_ROOT / ZERO_ACTION_RELATIVE
    _atomic_json(zero_action_path, zero_action)

    frame = _capture_head(scene, simulation)
    metrics = frame_metrics(frame)
    from PIL import Image

    preview = REPOSITORY_ROOT / PREVIEW_RELATIVE
    preview.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame, mode="RGB").save(preview, format="PNG", compress_level=0)
    overlay = _author_overlay(runtime_usd)
    inventory = _stage_inventory(stage, conversion)
    robot_inventory = {
        side: {
            "arm_dof": sum(
                1 for name in scene[f"{side}_robot"].joint_names if name.startswith("joint")
            ),
            "hand_dof": sum(
                1 for name in scene[f"{side}_robot"].joint_names if "finger" in name
            ),
            "joint_count": len(scene[f"{side}_robot"].joint_names),
            "finite": bool(
                torch.all(torch.isfinite(scene[f"{side}_robot"].data.joint_pos))
            ),
        }
        for side in ("left", "right")
    }
    component_pass = bool(
        inventory["screwdriver"]["collider_count"] == SCREWDRIVER_EXPECTED_COLLIDER_COUNT
        and inventory["screwdriver"]["rigid_body_count"] == 1
        and inventory["screwdriver"]["visual_mesh_count"] == 1
        and inventory["screwdriver"]["visual_collision_free"] is True
        and inventory["screwdriver"]["colliders_hidden"] is True
        and inventory["screwdriver"]["paths_match_asset_manifest"] is True
        and inventory["marker"]["collision_enabled"] is False
        and inventory["marker"]["pose_matches_contract"] is True
        and inventory["marker"]["size_matches_contract"] is True
        and inventory["marker"]["material_matches_contract"] is True
        and inventory["dome_lights"]
        and inventory["distant_lights"]
        and HEAD_CAMERA_PRIM_PATH in inventory["camera_paths"]
        and all(row["arm_dof"] == 7 and row["hand_dof"] == 20 and row["finite"] for row in robot_inventory.values())
    )
    summary = {
        "schema": "as019_core_scene_live_v1",
        "task_id": TASK_ID,
        "task_name": TASK_NAME,
        "simulator": "isaac_sim",
        "simulator_version": "5.1.0.0",
        "physical_gpu": int(os.environ.get("TASKDEV_PHYSICAL_GPU", "0")),
        "logical_gpu": 0,
        "render_mode": "RayTracedLighting",
        "tonemap": "ACES_op4",
        "scene_inventory": inventory,
        "robot_inventory": robot_inventory,
        "head_camera_prim_path": HEAD_CAMERA_PRIM_PATH,
        "head_frame_metrics": metrics,
        "preview": str(PREVIEW_RELATIVE),
        "preview_bytes": preview.stat().st_size,
        "preview_sha256": sha256_file(preview),
        "zero_action": zero_action,
        "component_pass": component_pass,
        "core_scene_pass": bool(component_pass and metrics["pass"] and zero_action_pass),
        "handover_path_executed": False,
        "grasp_executed": False,
        "task_complete": False,
    }
    summary_path = REPOSITORY_ROOT / SUMMARY_RELATIVE
    _atomic_json(summary_path, summary)
    _write_manifest(
        conversion=conversion,
        overlay=overlay,
        preview=preview,
        summary_path=summary_path,
        zero_action_path=zero_action_path,
        summary=summary,
    )
    _atomic_json(output_dir / "RUNNER_CORE_SCENE_PASS.json", {
        "core_scene_pass": summary["core_scene_pass"],
        "task_complete": False,
        "summary": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
    })
    return summary
