"""GEN3 asset contract. No simulator imports are needed to inspect provenance."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

GEN3_TASK_NAME = "eval_gen02_open_trash_bin"
GEN3_ASSET_DIRECTORY = Path(__file__).resolve().parent.parent / "envs" / "usd" / GEN3_TASK_NAME
GEN3_SOURCE_UID = "d4bed4900def46e7b2f098b0a9ec6209"
GEN3_SOURCE_URL = "https://sketchfab.com/3d-models/trash-can-with-animations-garbage-can-animated-d4bed4900def46e7b2f098b0a9ec6209"
GEN3_LICENSE = "CC-BY-4.0"
GEN3_TABLE_TOP_Z_M = 0.890
GEN3_SUPPORT_GAP_M = 0.00025
GEN3_ASSET_HEIGHT_M = 0.270
GEN3_LID_MASS_KG = 0.150
GEN3_BASE_MASS_KG = 0.700
GEN3_OPEN_THRESHOLD_DEG = 75.0
GEN3_HARD_STOP_DEG = 90.0
GEN3_SUCCESS_LABEL = "Open the trash-bin lid to at least 75 degrees"


def validate_gen3_asset_files() -> tuple[Path, ...]:
    """Check the task's own source/runtime closure, independently of LW archives."""
    manifest_path = GEN3_ASSET_DIRECTORY / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validated = []
    for entry in manifest["files"]:
        path = GEN3_ASSET_DIRECTORY / entry["path"]
        if not path.is_file():
            raise FileNotFoundError(f"GEN3 asset dependency is missing: {path}")
        if path.stat().st_size != entry["size_bytes"]:
            raise ValueError(f"GEN3 asset size mismatch: {path}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"GEN3 asset SHA-256 mismatch: {path}")
        validated.append(path)
    if GEN3_ASSET_DIRECTORY / "runtime" / "trash_bin.usda" not in validated:
        raise ValueError("GEN3 manifest omits the runtime articulation")
    return tuple(validated)


def configure_gen3_dynamics(stage, spec) -> None:
    """Apply the reference passive-lid damping without an opening motor."""
    from pxr import PhysxSchema, UsdPhysics

    for name in (spec.root_body_name, spec.right_body_name):
        prim = stage.GetPrimAtPath(spec.body_path(name))
        mass = UsdPhysics.MassAPI(prim)
        values = (mass.GetMassAttr().Get(), *mass.GetDiagonalInertiaAttr().Get())
        if not all(value is not None and 0 < float(value) < float("inf") for value in values):
            raise ValueError(f"GEN3 source-derived mass properties are invalid: {prim.GetPath()}")
        api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        api.CreateAngularDampingAttr().Set(2.0)
        api.CreateEnableCCDAttr().Set(True)
        api.CreateSolverPositionIterationCountAttr().Set(64)
        api.CreateSolverVelocityIterationCountAttr().Set(8)
        api.CreateMaxDepenetrationVelocityAttr().Set(0.40)
    joint_prim = stage.GetPrimAtPath(f"{spec.right_body_path}/{spec.dof_names[0]}")
    PhysxSchema.PhysxJointAPI.Apply(joint_prim).CreateJointFrictionAttr().Set(0.10)
