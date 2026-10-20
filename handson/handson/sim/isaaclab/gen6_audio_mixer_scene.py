"""GEN6 identity and source-mesh integration for the shared B320 runtime.

The asset bundle must come from inspected licensed source files. This module
never substitutes the training knobs or generates visible placeholder meshes.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any


TASK_NAME = "eval_gen05_set_mixer_knobs"
LABEL = "GEN6 Audio Mixer Dual-Knob Setting"
PROMPT = (
    "Use your left hand to turn the left knob counterclockwise and your right "
    "hand to turn the right knob clockwise, in either order. Set both displayed "
    "targets, release both knobs, and keep the settings stable."
)
ASSET_DIRECTORY = Path(__file__).resolve().parents[1] / "envs" / "usd" / TASK_NAME
ASSET_FILES = (
    "mixer_fixed_visual.usda",
    "knob_visual.usda",
    "mixer_fixed_collision.usda",
    "knob_collision.usda",
    "ATTRIBUTION.md",
)


def asset_geometry() -> dict[str, Any]:
    """Use the same measured frames for scene authoring and CPU observations."""
    return json.loads((ASSET_DIRECTORY / "asset_manifest.json").read_text())["geometry"]


def task_spec_from_reference(reference: Any) -> Any:
    """Change identity/asset source only; preserve physical success semantics."""
    return replace(
        reference,
        task_id="GEN6",
        task_name=TASK_NAME,
        label=LABEL,
        prompt=PROMPT,
        fixture_kind="gen6_audio_mixer",
        source_asset_keys=("gen6_licensed_audio_mixer",),
    )


def validate_asset_files() -> tuple[Path, ...]:
    """Report missing/changed source-derived files before starting Kit."""
    manifest_path = ASSET_DIRECTORY / "asset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "GEN6 audio mixer asset integration is incomplete: "
            f"{manifest_path}. Supply the licensed source model and complete "
            "the asset integration; no placeholder model will be substituted."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("task_id"), manifest.get("task_name")) != ("GEN6", TASK_NAME):
        raise ValueError("GEN6 asset manifest identity does not match the task")
    files = manifest["files"]
    if not set(ASSET_FILES).issubset(files):
        raise ValueError("GEN6 asset manifest is missing a visual/collision/source entry")
    required = [manifest_path]
    for relative, digest in files.items():
        path = ASSET_DIRECTORY / relative
        if not path.is_file():
            raise FileNotFoundError(f"GEN6 asset dependency is missing: {path}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"GEN6 asset dependency checksum mismatch: {relative}")
        required.append(path)
    return tuple(required)


def spawn_task_assets(stage: Any, sim_utils: Any, *, task_spec: Any) -> Any:
    """Bind inspected source visuals/colliders to two real passive revolute DOFs."""
    validate_asset_files()
    from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics
    from .b320_bimanual_scene import (
        B320_ASSET_ROOT_PATH,
        B320SpawnResult,
        _apply_dynamic_body,
        _author_joint,
        _bind_material,
        _fix_articulation_base_to_world,
        _reference_visual,
        _set_xform,
        _spawn_worktable,
    )

    manifest = json.loads((ASSET_DIRECTORY / "asset_manifest.json").read_text())
    geometry = manifest["geometry"]
    root_home = tuple(geometry["root_home_m"])
    centers = geometry["knob_centers_m"]
    knob_orientation = tuple(geometry["knob_orientation_wxyz"])
    table_path = _spawn_worktable(stage, sim_utils, task_name=TASK_NAME)
    base_path = f"{B320_ASSET_ROOT_PATH}/panel_base"
    base = UsdGeom.Xform.Define(stage, base_path).GetPrim()
    _set_xform(base, translation=root_home, orientation_wxyz=(1.0, 0.0, 0.0, 0.0))
    _apply_dynamic_body(stage, sim_utils, path=base_path, mass_kg=2.4)
    articulation_path = _fix_articulation_base_to_world(stage, base_path)
    body_paths = {"panel_base": base_path}
    collider_counts = {}
    static_paths = [table_path]

    for role in ("panel_base", "knob_left", "knob_right"):
        fixed = role == "panel_base"
        body_path = body_paths.get(role, f"{B320_ASSET_ROOT_PATH}/{role}")
        if not fixed:
            side = role.removeprefix("knob_")
            center = centers[side]
            body = UsdGeom.Xform.Define(stage, body_path).GetPrim()
            _set_xform(
                body,
                translation=tuple(root_home[i] + center[i] for i in range(3)),
                orientation_wxyz=knob_orientation,
            )
            _apply_dynamic_body(stage, sim_utils, path=body_path, mass_kg=0.09)
            physx_body = PhysxSchema.PhysxRigidBodyAPI.Apply(body)
            physx_body.CreateLinearDampingAttr().Set(0.08)
            physx_body.CreateAngularDampingAttr().Set(0.40)
            mass = UsdPhysics.MassAPI.Apply(body)
            mass.CreateCenterOfMassAttr().Set(Gf.Vec3f(*geometry["knob_com_m"]))
            mass.CreateDiagonalInertiaAttr().Set(
                Gf.Vec3f(*geometry["knob_diagonal_inertia_kg_m2"])
            )
            mass.CreatePrincipalAxesAttr().Set(
                Gf.Quatf(*geometry["knob_principal_axes_wxyz"])
            )
            joint_path = _author_joint(
                stage,
                path=f"{B320_ASSET_ROOT_PATH}/{side}_knob_joint",
                joint_type="revolute",
                body0=base_path,
                body1=body_path,
                axis="Z",
                local_pos0=tuple(center),
                local_rot0_wxyz=knob_orientation,
                local_rot1_wxyz=(1.0, 0.0, 0.0, 0.0),
                lower=-100.0,
                upper=100.0,
                stiffness=0.0,
                damping=0.10,
                max_force=0.25,
            )
            PhysxSchema.PhysxJointAPI.Apply(
                stage.GetPrimAtPath(joint_path)
            ).CreateJointFrictionAttr().Set(0.065)
        body = stage.GetPrimAtPath(body_path)
        body.SetCustomDataByKey("handson:logicalName", role)
        body.SetCustomDataByKey("handson:formalTruth", True)
        body.SetCustomDataByKey("handson:sourceAsset", manifest["source"]["url"])
        prefix = "mixer_fixed" if fixed else "knob"
        for kind in ("visual", "collision"):
            path = _reference_visual(
                stage,
                path=f"{body_path}/source_{kind}",
                source=ASSET_DIRECTORY / f"{prefix}_{kind}.usda",
                source_prim="/Asset",
                scale=1.0,
                instanceable=False,
            )
            prim = stage.GetPrimAtPath(path)
            prim.SetCustomDataByKey("handson:geometryRole", kind)
            if kind == "visual":
                if fixed:
                    static_paths.append(path)
                continue
            UsdGeom.Imageable(prim).CreateVisibilityAttr().Set("invisible")
            count = 0
            for child in Usd.PrimRange(prim):
                if not child.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                UsdGeom.Imageable(child).CreateVisibilityAttr().Set("invisible")
                _bind_material(
                    str(child.GetPath()),
                    sim_utils=sim_utils,
                    static_friction=0.85,
                    dynamic_friction=0.65,
                    friction_combine_mode="min",
                )
                count += 1
            if not count:
                raise ValueError(f"GEN6 source-derived collision shapes are missing: {path}")
            collider_counts[role] = count
        body_paths[role] = body_path

    return B320SpawnResult(
        body_paths=body_paths,
        articulation_root_paths=(articulation_path,),
        static_paths=tuple(static_paths),
        dof_names=task_spec.dof_names,
        collider_counts=collider_counts,
        evidence={"source": manifest["source"], "geometry": geometry},
    )
