#!/usr/bin/env python3
"""Build and zero-action-smoke the B320-BC012 Isaac Sim 5.1 core scene.

This executable composes assets, collisions, the native dual ROKAE/Wuji rig,
table, lights, and head camera. It deliberately exposes no motion phase and
never calls the task environment's action step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT / "handson"))

# AppLauncher must precede all Kit, pxr, Isaac Sim, and project simulator imports.
from isaaclab.app import AppLauncher  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(np.asarray(value, dtype=np.float64)).all())
    except Exception:
        return False


def _tensor_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _aabb_overlap_depth(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64).reshape(2, 3)
    second = np.asarray(second, dtype=np.float64).reshape(2, 3)
    overlap = np.minimum(first[1], second[1]) - np.maximum(first[0], second[0])
    return float(np.min(overlap)) if np.all(overlap > 0.0) else 0.0


def _gpu_namespace_contract(output_dir: Path) -> dict[str, Any]:
    raw_physical = os.environ.get("TASKDEV_PHYSICAL_GPU", "").strip()
    requested_host = os.environ.get("TASKDEV_HOST", "").strip()
    namespace_raw = os.environ.get("TASKDEV_GPU_NAMESPACE_PATH", "").strip()
    namespace_path = (
        Path(namespace_raw).resolve()
        if namespace_raw
        else (output_dir.parent / "gpu_namespace.json").resolve()
    )
    payload = (
        json.loads(namespace_path.read_text(encoding="utf-8"))
        if namespace_path.is_file()
        else {}
    )
    observed = socket.gethostname()
    return {
        "requested_host": requested_host,
        "observed_hostname": observed,
        "physical_gpu": int(raw_physical) if raw_physical.isdigit() else None,
        "logical_gpu": payload.get("logical_gpu"),
        "selected_device": payload.get("selected_device"),
        "namespace_path": str(namespace_path),
        "namespace_sha256": _sha256(namespace_path) if namespace_path.is_file() else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "pass": bool(
            requested_host == "hp-14"
            and raw_physical == "1"
            and payload.get("physical_gpu") == 1
            and payload.get("logical_gpu") == 0
            and payload.get("selected_device") == "/dev/nvidia1"
            and not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        ),
    }


def _vram_snapshot() -> dict[str, Any]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=20,
        ).strip().splitlines()
        rows = []
        for line in output:
            index, total, used, utilization = [int(token.strip()) for token in line.split(",")]
            rows.append(
                {
                    "logical_index": index,
                    "memory_total_mib": total,
                    "memory_used_mib": used,
                    "memory_headroom_fraction": (total - used) / total,
                    "utilization_percent": utilization,
                }
            )
        return {"rows": rows, "pass": bool(rows and rows[0]["memory_headroom_fraction"] > 0.20)}
    except Exception as exc:
        return {"rows": [], "error": f"{type(exc).__name__}: {exc}", "pass": False}


class BC012CoreScene:
    def __init__(
        self,
        *,
        simulation_app: Any,
        app_launcher: Any,
        output_dir: Path,
        device: str,
    ) -> None:
        from handson.sim.isaaclab.abc_collection_env import HandsOnABCIsaacLabEnv
        from taskdev.tasks.bench320_bc012_dual_can_synchronized_bin_load import core_scene_spec as spec

        self.spec = spec
        self.simulation_app = simulation_app
        self.output_dir = output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.task_object_state_writes_after_start = 0
        static_errors = spec.validate_static_contract()
        if static_errors:
            raise RuntimeError("; ".join(static_errors))
        self.env = HandsOnABCIsaacLabEnv(
            "t02_connect_charger_and_cable",
            seed=0,
            control_dt=1.0 / 30.0,
            physics_dt=1.0 / float(spec.PHYSICS_HZ),
            time_limit=15.0,
            render_mode="rgb_array",
            image_obs=True,
            randomize=False,
            randomize_dynamics=False,
            hand_type="wuji1",
            arm_type="rokae",
            launch_isaac=True,
            headless=True,
            render_resolution=spec.HEAD_RESOLUTION_HW,
            hz=30,
            isaac_device=device,
            physics_backend="physx",
            existing_simulation_app=simulation_app,
            existing_app_launcher=app_launcher,
        )
        self.stage = self.env._stage
        self.sim = self.env._sim
        if self.stage is None or self.sim is None:
            raise RuntimeError("native HandsOn Isaac runtime did not expose a stage and simulation")
        self.physics_view = self.env._robots["right"]._physics_sim_view
        if self.physics_view is None:
            raise RuntimeError("native ROKAE PhysX simulation view is unavailable")

    def close(self) -> None:
        self.env.close()

    def _asset_hashes(self) -> dict[str, Any]:
        records = {}
        for path, expected in self.spec.EXPECTED_SHA256.items():
            observed = _sha256(path)
            records[str(path.relative_to(self.spec.REPOSITORY_ROOT))] = {
                "expected_sha256": expected,
                "observed_sha256": observed,
                "bytes": path.stat().st_size,
                "pass": observed == expected,
            }
        return {
            "records": records,
            "pass": all(record["pass"] for record in records.values()),
        }

    def _can_inventory(self) -> dict[str, Any]:
        from pxr import Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.Open(str(self.spec.CAN_USD), load=Usd.Stage.LoadAll)
        if stage is None:
            raise RuntimeError("could not open exact tomato soup can USD")
        root = stage.GetDefaultPrim()
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
            useExtentsHint=True,
        )
        bounds = cache.ComputeWorldBound(root).ComputeAlignedRange()
        low = np.asarray(bounds.GetMin(), dtype=np.float64)
        high = np.asarray(bounds.GetMax(), dtype=np.float64)
        extent = high - low
        return {
            "default_prim": str(root.GetPath()),
            "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
            "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
            "local_aabb_m": [low.tolist(), high.tolist()],
            "extent_m": extent.tolist(),
            "used_layers": sorted(str(layer.realPath) for layer in stage.GetUsedLayers() if layer.realPath),
            "source_collision_count": sum(
                int(prim.HasAPI(UsdPhysics.CollisionAPI)) for prim in stage.TraverseAll()
            ),
            "source_rigid_body_count": sum(
                int(prim.HasAPI(UsdPhysics.RigidBodyAPI)) for prim in stage.TraverseAll()
            ),
            "pass": bool(
                root.IsValid()
                and math.isclose(float(UsdGeom.GetStageMetersPerUnit(stage)), 1.0)
                and str(UsdGeom.GetStageUpAxis(stage)).upper() == "Z"
                and _finite(extent)
                and 0.04 < extent[0] < 0.12
                and 0.04 < extent[1] < 0.12
                and 0.07 < extent[2] < 0.16
            ),
        }

    def _hide_native_c10_objects(self) -> list[str]:
        from pxr import Usd, UsdGeom, UsdPhysics

        hidden = []
        for name in ("c10_charger_plug", "c10_charging_cable", "c10_power_strip"):
            prim = self.env._find_prim(name)
            if prim is not None and prim.IsValid():
                imageable = UsdGeom.Imageable(prim)
                if imageable:
                    imageable.MakeInvisible()
                for descendant in Usd.PrimRange(prim):
                    if descendant.HasAPI(UsdPhysics.CollisionAPI):
                        collision = UsdPhysics.CollisionAPI(descendant)
                        attribute = collision.GetCollisionEnabledAttr()
                        if attribute.IsValid():
                            attribute.Set(False)
                hidden.append(str(prim.GetPath()))
        return hidden

    @staticmethod
    def _material(stage: Any, path: str, color: tuple[float, float, float], opacity: float = 1.0) -> Any:
        from pxr import Sdf, UsdShade

        material = UsdShade.Material.Define(stage, path)
        shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.38)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.08)
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return material

    @staticmethod
    def _bind_material(prim: Any, material: Any) -> None:
        from pxr import UsdShade

        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)

    @staticmethod
    def _set_translate(prim: Any, values: tuple[float, float, float] | list[float]) -> None:
        from pxr import Gf, UsdGeom

        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*values))

    def _compose_can(self, side: str, inventory: dict[str, Any]) -> dict[str, Any]:
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

        source_low = np.asarray(inventory["local_aabb_m"][0], dtype=np.float64)
        source_high = np.asarray(inventory["local_aabb_m"][1], dtype=np.float64)
        extent = source_high - source_low
        anchor = np.asarray(self.spec.CAN_ROOT_POSITIONS_M[side], dtype=np.float64)
        position = anchor.copy()
        position[2] -= source_low[2]
        name = f"{side.capitalize()}Can"
        path = f"{self.spec.SCENE_ROOT}/{name}"
        root = UsdGeom.Xform.Define(self.stage, path).GetPrim()
        self._set_translate(root, position.tolist())
        visual = UsdGeom.Xform.Define(self.stage, f"{path}/ExactYCBVisual").GetPrim()
        visual.GetReferences().AddReference(str(self.spec.CAN_USD.resolve()))
        rigid = UsdPhysics.RigidBodyAPI.Apply(root)
        rigid.CreateKinematicEnabledAttr(False)
        UsdPhysics.MassAPI.Apply(root).CreateMassAttr(0.349)
        physx_rigid = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
        physx_rigid.CreateEnableCCDAttr(True)
        physx_rigid.CreateLinearDampingAttr(0.08)
        physx_rigid.CreateAngularDampingAttr(0.12)
        collision = UsdGeom.Cylinder.Define(self.stage, f"{path}/CylinderCollision")
        collision.CreateAxisAttr("Z")
        collision.CreateHeightAttr(float(extent[2] * 0.98))
        collision.CreateRadiusAttr(float(min(extent[0], extent[1]) * 0.49))
        center = 0.5 * (source_low + source_high)
        self._set_translate(collision.GetPrim(), center.tolist())
        UsdPhysics.CollisionAPI.Apply(collision.GetPrim()).CreateCollisionEnabledAttr(True)
        UsdGeom.Imageable(collision.GetPrim()).MakeInvisible()
        PhysxSchema.PhysxContactReportAPI.Apply(root).CreateThresholdAttr(0.0)
        world_aabb = np.stack((source_low + position, source_high + position))
        return {
            "side": side,
            "prim_path": path,
            "visual_reference": str(self.spec.CAN_USD.relative_to(self.spec.REPOSITORY_ROOT)),
            "position_m": position.tolist(),
            "source_scale_xyz": list(self.spec.CAN_SOURCE_SCALE_XYZ),
            "source_orientation_xyzw": list(self.spec.CAN_SOURCE_ORIENTATION_XYZW),
            "collision": {
                "type": "cylinder",
                "radius_m": float(min(extent[0], extent[1]) * 0.49),
                "height_m": float(extent[2] * 0.98),
                "local_center_m": center.tolist(),
            },
            "initial_world_aabb_m": world_aabb.tolist(),
        }

    def _compose_bin(self) -> dict[str, Any]:
        import trimesh
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, Vt

        root_path = f"{self.spec.SCENE_ROOT}/OpenBin"
        root = UsdGeom.Xform.Define(self.stage, root_path).GetPrim()
        self._set_translate(root, self.spec.OPEN_BIN_ROOT_POSITION_M)
        rigid = UsdPhysics.RigidBodyAPI.Apply(root)
        rigid.CreateKinematicEnabledAttr(True)
        UsdPhysics.MassAPI.Apply(root).CreateMassAttr(3.0)
        bin_material = self._material(
            self.stage,
            f"{self.spec.SCENE_ROOT}/Materials/OpenBin",
            (0.055, 0.30, 0.52),
        )
        divider_material = self._material(
            self.stage,
            f"{self.spec.SCENE_ROOT}/Materials/Divider",
            (0.95, 0.62, 0.08),
        )
        left_goal_material = self._material(
            self.stage,
            f"{self.spec.SCENE_ROOT}/Materials/LeftGoal",
            (0.15, 0.80, 0.38),
            0.35,
        )
        right_goal_material = self._material(
            self.stage,
            f"{self.spec.SCENE_ROOT}/Materials/RightGoal",
            (0.96, 0.30, 0.16),
            0.35,
        )

        scene = trimesh.load(str(self.spec.OPEN_BIN_GLB), force="scene")
        measured_bounds = np.asarray(scene.bounds, dtype=np.float64)
        measured_extent = np.asarray(scene.extents, dtype=np.float64)
        visual_paths = []
        for index, node in enumerate(sorted(scene.graph.nodes_geometry)):
            transform, geometry_name = scene.graph.get(node)
            geometry = scene.geometry[geometry_name].copy()
            geometry.apply_transform(transform)
            source_points = np.asarray(geometry.vertices, dtype=np.float64)
            # Exact source orientation: GLB Y-up to stage Z-up, +90 degrees X.
            points = np.column_stack(
                (source_points[:, 0], -source_points[:, 2], source_points[:, 1])
            )
            mesh_path = (
                f"{root_path}/ExactGLBVisual/"
                f"Mesh_{index:02d}_{self.spec.usd_prim_identifier(node)}"
            )
            mesh = UsdGeom.Mesh.Define(self.stage, mesh_path)
            mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
            faces = np.asarray(geometry.faces, dtype=np.int32)
            mesh.CreateFaceVertexCountsAttr([3] * len(faces))
            mesh.CreateFaceVertexIndicesAttr(faces.reshape(-1).tolist())
            mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            mesh.CreateDoubleSidedAttr(True)
            self._bind_material(mesh.GetPrim(), bin_material)
            visual_paths.append(mesh_path)

        collision_paths = []
        for name, (center, size) in self.spec.OPEN_BIN_COLLISION_PANELS.items():
            cube = UsdGeom.Cube.Define(self.stage, f"{root_path}/Collisions/{name}")
            cube.CreateSizeAttr(1.0)
            xform = UsdGeom.Xformable(cube.GetPrim())
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(Gf.Vec3d(*center))
            xform.AddScaleOp().Set(Gf.Vec3f(*size))
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr(True)
            UsdGeom.Imageable(cube.GetPrim()).MakeInvisible()
            collision_paths.append(str(cube.GetPath()))

        divider_center, divider_size = self.spec.PARTITION_DIVIDER
        divider = UsdGeom.Cube.Define(self.stage, f"{root_path}/PartitionDivider")
        divider.CreateSizeAttr(1.0)
        divider_xform = UsdGeom.Xformable(divider.GetPrim())
        divider_xform.ClearXformOpOrder()
        divider_xform.AddTranslateOp().Set(Gf.Vec3d(*divider_center))
        divider_xform.AddScaleOp().Set(Gf.Vec3f(*divider_size))
        UsdPhysics.CollisionAPI.Apply(divider.GetPrim()).CreateCollisionEnabledAttr(True)
        self._bind_material(divider.GetPrim(), divider_material)
        collision_paths.append(str(divider.GetPath()))

        goal_paths = {}
        for side, (center, size) in self.spec.GOAL_ZONES.items():
            goal = UsdGeom.Cube.Define(self.stage, f"{root_path}/{side.capitalize()}GoalZone")
            goal.CreateSizeAttr(1.0)
            goal_xform = UsdGeom.Xformable(goal.GetPrim())
            goal_xform.ClearXformOpOrder()
            goal_xform.AddTranslateOp().Set(Gf.Vec3d(*center))
            goal_xform.AddScaleOp().Set(Gf.Vec3f(*size))
            self._bind_material(
                goal.GetPrim(), left_goal_material if side == "left" else right_goal_material
            )
            goal_paths[side] = str(goal.GetPath())
        PhysxSchema.PhysxContactReportAPI.Apply(root).CreateThresholdAttr(0.0)
        return {
            "prim_path": root_path,
            "source_glb": str(self.spec.OPEN_BIN_GLB.relative_to(self.spec.REPOSITORY_ROOT)),
            "source_units": str(scene.units),
            "source_bounds_m": measured_bounds.tolist(),
            "source_extent_m": measured_extent.tolist(),
            "position_m": list(self.spec.OPEN_BIN_ROOT_POSITION_M),
            "source_scale_xyz": list(self.spec.OPEN_BIN_SOURCE_SCALE_XYZ),
            "source_orientation_xyzw": list(self.spec.OPEN_BIN_SOURCE_ORIENTATION_XYZW),
            "visual_mesh_paths": visual_paths,
            "collision_paths": collision_paths,
            "goal_zone_paths": goal_paths,
            "authored_panel_count": len(scene.graph.nodes_geometry),
            "collision_shape_count": len(collision_paths),
            "pass": bool(
                str(scene.units) == "meters"
                and len(scene.graph.nodes_geometry) == 5
                and np.allclose(
                    measured_bounds,
                    np.asarray(self.spec.OPEN_BIN_SOURCE_BOUNDS_M),
                    rtol=0.0,
                    atol=1.0e-7,
                )
                and len(collision_paths) == 6
            ),
        }

    def _lights(self) -> dict[str, Any]:
        from pxr import Gf, UsdGeom, UsdLux

        dome = UsdLux.DomeLight.Define(self.stage, f"{self.spec.SCENE_ROOT}/Lights/Dome")
        dome.CreateIntensityAttr(350.0)
        dome.CreateColorAttr(Gf.Vec3f(0.82, 0.88, 1.0))
        distant = UsdLux.DistantLight.Define(
            self.stage, f"{self.spec.SCENE_ROOT}/Lights/Distant"
        )
        distant.CreateIntensityAttr(2200.0)
        distant.CreateAngleAttr(0.7)
        distant.CreateColorAttr(Gf.Vec3f(1.0, 0.94, 0.84))
        distant_xform = UsdGeom.Xformable(distant.GetPrim())
        distant_xform.ClearXformOpOrder()
        distant_xform.AddRotateXYZOp().Set(Gf.Vec3f(-38.0, 28.0, 12.0))
        return {
            "dome": {"prim_path": str(dome.GetPath()), "intensity": 350.0},
            "distant": {"prim_path": str(distant.GetPath()), "intensity": 2200.0},
            "pass": True,
        }

    def _live_state(self, view: Any) -> dict[str, Any]:
        transforms = _tensor_numpy(view.get_transforms()).astype(np.float64).reshape(-1, 7)
        velocities = _tensor_numpy(view.get_velocities()).astype(np.float64).reshape(-1, 6)
        return {
            "transform_xyzw": transforms[0].tolist(),
            "velocity": velocities[0].tolist(),
            "linear_speed_mps": float(np.linalg.norm(velocities[0, :3])),
            "angular_speed_radps": float(np.linalg.norm(velocities[0, 3:])),
            "finite": bool(_finite(transforms) and _finite(velocities)),
        }

    def _render_head(self) -> dict[str, Any]:
        from PIL import Image

        for _ in range(5):
            self.simulation_app.update()
        frames = self.env.render()
        if not isinstance(frames, (list, tuple)) or not frames:
            raise RuntimeError("native head camera returned no frame")
        frame = np.asarray(frames[0])
        if frame.shape != (480, 640, 3) or frame.dtype != np.uint8:
            raise RuntimeError(f"head frame must be uint8 480x640x3, got {frame.shape}/{frame.dtype}")
        path = self.output_dir / "reset_head.png"
        Image.fromarray(frame).save(path, compress_level=0)
        pixel_mean = frame.astype(np.float64).mean(axis=2)
        camera = self.env._cameras.get("head_camera")
        prim_path = str(getattr(camera.cfg, "prim_path", "")) if camera is not None else ""
        metrics = {
            "mean_rgb": float(frame.mean()),
            "std_rgb": float(frame.std()),
            "min_rgb": int(frame.min()),
            "max_rgb": int(frame.max()),
            "black_fraction": float(np.mean(pixel_mean < 10.0)),
        }
        return {
            "prim_path": prim_path,
            "expected_prim_path": self.spec.HEAD_CAMERA_PRIM,
            "resolution_hwc": list(frame.shape),
            "dtype": str(frame.dtype),
            "preview": str(path),
            "preview_sha256": _sha256(path),
            "preview_bytes": path.stat().st_size,
            "metrics": metrics,
            "pass": bool(
                prim_path == self.spec.HEAD_CAMERA_PRIM
                and path.stat().st_size >= 150_000
                and metrics["mean_rgb"] > 30.0
                and metrics["std_rgb"] > 15.0
                and metrics["black_fraction"] < 0.25
            ),
        }

    def run(self) -> dict[str, Any]:
        from pxr import Usd, UsdGeom, UsdPhysics

        started = time.time()
        hashes = self._asset_hashes()
        can_inventory = self._can_inventory()
        hidden_native = self._hide_native_c10_objects()
        root = UsdGeom.Xform.Define(self.stage, self.spec.SCENE_ROOT).GetPrim()
        root.SetActive(True)
        left_can = self._compose_can("left", can_inventory)
        right_can = self._compose_can("right", can_inventory)
        open_bin = self._compose_bin()
        lights = self._lights()

        initial_overlap = {
            "left_can_right_can_m": _aabb_overlap_depth(
                np.asarray(left_can["initial_world_aabb_m"]),
                np.asarray(right_can["initial_world_aabb_m"]),
            ),
        }
        bin_world_bounds = np.asarray(
            [
                [
                    self.spec.OPEN_BIN_ROOT_POSITION_M[0] - 0.19,
                    self.spec.OPEN_BIN_ROOT_POSITION_M[1] - 0.14,
                    self.spec.OPEN_BIN_ROOT_POSITION_M[2],
                ],
                [
                    self.spec.OPEN_BIN_ROOT_POSITION_M[0] + 0.19,
                    self.spec.OPEN_BIN_ROOT_POSITION_M[1] + 0.14,
                    self.spec.OPEN_BIN_ROOT_POSITION_M[2] + 0.15,
                ],
            ]
        )
        for side, can in (("left", left_can), ("right", right_can)):
            initial_overlap[f"{side}_can_open_bin_m"] = _aabb_overlap_depth(
                np.asarray(can["initial_world_aabb_m"]), bin_world_bounds
            )

        export_path = self.output_dir / "bc012_live_scene.usda"
        if not self.stage.Export(str(export_path)):
            raise RuntimeError("failed to export live USD scene")

        self.sim.reset()
        self.physics_view = self.env._robots["right"]._physics_sim_view
        views = {
            "left": self.physics_view.create_rigid_body_view(left_can["prim_path"]),
            "right": self.physics_view.create_rigid_body_view(right_can["prim_path"]),
            "bin": self.physics_view.create_rigid_body_view(open_bin["prim_path"]),
        }
        if any(view is None or int(view.count) != 1 for view in views.values()):
            raise RuntimeError("task rigid-body views are incomplete")
        initial_states = {name: self._live_state(view) for name, view in views.items()}
        max_linear = {name: 0.0 for name in views}
        max_angular = {name: 0.0 for name in views}
        for step in range(self.spec.ZERO_ACTION_STEPS):
            for robot in self.env._robots.values():
                robot.write_data_to_sim()
            self.sim.step(render=bool(step % self.spec.PHYSICS_HZ == 0))
            for robot in self.env._robots.values():
                robot.update(self.env._sim_dt)
            if step % 10 == 0 or step + 1 == self.spec.ZERO_ACTION_STEPS:
                for name, view in views.items():
                    state = self._live_state(view)
                    max_linear[name] = max(max_linear[name], state["linear_speed_mps"])
                    max_angular[name] = max(max_angular[name], state["angular_speed_radps"])
        final_states = {name: self._live_state(view) for name, view in views.items()}
        drift = {}
        for name in views:
            start = np.asarray(initial_states[name]["transform_xyzw"][:3], dtype=np.float64)
            final = np.asarray(final_states[name]["transform_xyzw"][:3], dtype=np.float64)
            drift[name] = {
                "translation_m": float(np.linalg.norm(final - start)),
                "xy_m": float(np.linalg.norm(final[:2] - start[:2])),
                "z_m": float(final[2] - start[2]),
            }

        camera = self._render_head()
        robot = {
            "names": sorted(self.env._robots),
            "arm_dof_per_side": {
                side: len(self.env._arm_joint_ids[side]) for side in ("left", "right")
            },
            "hand_dof_per_side": {
                side: len(self.env._hand_joint_ids[side]) for side in ("left", "right")
            },
            "asset": str(self.spec.ROBOT_MJCF.relative_to(self.spec.REPOSITORY_ROOT)),
        }
        table_prim = self.stage.GetPrimAtPath("/World/Task/table/table")
        task_collisions = sum(
            int(prim.HasAPI(UsdPhysics.CollisionAPI))
            for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate))
            if prim.IsValid() and prim.IsActive()
        )
        stage_joint_count = sum(
            int(prim.IsA(UsdPhysics.Joint)) for prim in self.stage.TraverseAll()
        )
        stability_pass = bool(
            all(state["finite"] for state in final_states.values())
            and final_states["left"]["linear_speed_mps"] < 0.05
            and final_states["right"]["linear_speed_mps"] < 0.05
            and final_states["left"]["angular_speed_radps"] < 0.5
            and final_states["right"]["angular_speed_radps"] < 0.5
            and drift["left"]["xy_m"] < 0.02
            and drift["right"]["xy_m"] < 0.02
            and drift["bin"]["translation_m"] < 1.0e-4
        )
        components_pass = bool(
            robot["names"] == ["left", "right"]
            and all(value == 7 for value in robot["arm_dof_per_side"].values())
            and all(value == 20 for value in robot["hand_dof_per_side"].values())
            and table_prim.IsValid()
            and table_prim.HasAPI(UsdPhysics.RigidBodyAPI)
            and task_collisions >= 8
            and stage_joint_count >= 54
        )
        penetration_pass = all(value == 0.0 for value in initial_overlap.values())
        vram = _vram_snapshot()
        scene_export = {
            "path": str(export_path),
            "sha256": _sha256(export_path),
            "bytes": export_path.stat().st_size,
            "pass": export_path.stat().st_size > 1024,
        }
        core_scene_ready = bool(
            hashes["pass"]
            and can_inventory["pass"]
            and open_bin["pass"]
            and lights["pass"]
            and components_pass
            and penetration_pass
            and stability_pass
            and camera["pass"]
            and vram["pass"]
            and scene_export["pass"]
        )
        return {
            "schema": "bc012_core_scene_live_v1",
            "task_id": self.spec.TASK_ID,
            "scope": "core_scene_only",
            "isaac_sim_version": "5.1",
            "assets": hashes,
            "can_inventory": can_inventory,
            "components": {
                "robot": robot,
                "table": {
                    "prim_path": str(table_prim.GetPath()),
                    "rigid_body": bool(table_prim.HasAPI(UsdPhysics.RigidBodyAPI)),
                },
                "left_can": left_can,
                "right_can": right_can,
                "open_bin": open_bin,
                "lights": lights,
                "head_camera": camera,
                "task_collision_shape_count": task_collisions,
                "stage_joint_count": stage_joint_count,
                "hidden_native_c10_objects": hidden_native,
            },
            "initial_penetration": {
                "aabb_overlap_depth_m": initial_overlap,
                "maximum_m": max(initial_overlap.values()),
                "pass": penetration_pass,
            },
            "zero_action_smoke": {
                "steps": self.spec.ZERO_ACTION_STEPS,
                "physics_hz": self.spec.PHYSICS_HZ,
                "simulated_seconds": self.spec.ZERO_ACTION_STEPS / self.spec.PHYSICS_HZ,
                "action_commands_issued": 0,
                "task_object_state_writes_after_start": self.task_object_state_writes_after_start,
                "initial_states": initial_states,
                "final_states": final_states,
                "max_sampled_linear_speed_mps": max_linear,
                "max_sampled_angular_speed_radps": max_angular,
                "drift": drift,
                "pass": stability_pass,
            },
            "scene_export": scene_export,
            "vram": vram,
            "checks": {
                "asset_hashes": hashes["pass"],
                "can_scale_and_orientation": can_inventory["pass"],
                "bin_scale_and_orientation": open_bin["pass"],
                "dual_rokae_wuji_table_collision_joints": components_pass,
                "lighting": lights["pass"],
                "initial_penetration": penetration_pass,
                "zero_action_stability": stability_pass,
                "head_camera": camera["pass"],
                "vram_headroom": vram["pass"],
                "saved_usd": scene_export["pass"],
            },
            "core_scene_ready": core_scene_ready,
            "motion_executed": False,
            "grasp_tuned": False,
            "success_action_executed": False,
            "task_complete": False,
            "elapsed_wall_seconds": time.time() - started,
        }


def main() -> int:
    args = _parser().parse_args()
    args.headless = True
    args.enable_cameras = True
    args.device = "cuda:0"
    args.width = 640
    args.height = 480
    args.kit_args = " ".join(
        filter(
            None,
            (
                str(getattr(args, "kit_args", "") or ""),
                "--/rtx/rendermode=RayTracedLighting",
                "--/rtx/post/tonemap/op=4",
                "--/rtx/post/tonemap/filmIso=200",
            ),
        )
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    import carb  # noqa: E402

    settings = carb.settings.get_settings()
    settings.set("/rtx/rendermode", "RayTracedLighting")
    settings.set("/rtx/post/tonemap/op", 4)
    settings.set("/rtx/post/tonemap/filmIso", 200.0)
    summary: dict[str, Any] = {
        "schema": "bc012_core_scene_live_v1",
        "task_id": "B320-BC012",
        "core_scene_ready": False,
        "motion_executed": False,
        "grasp_tuned": False,
        "success_action_executed": False,
        "task_complete": False,
    }
    environment = None
    rc = 1
    try:
        namespace = _gpu_namespace_contract(output_dir)
        if not namespace["pass"]:
            raise RuntimeError(f"hp-14 physical GPU1 namespace contract failed: {namespace}")
        environment = BC012CoreScene(
            simulation_app=simulation_app,
            app_launcher=app_launcher,
            output_dir=output_dir,
            device=args.device,
        )
        summary.update(environment.run())
        summary["gpu_namespace"] = namespace
        rc = 0 if summary.get("core_scene_ready") is True else 1
    except Exception as exc:
        summary["exception"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        rc = 1
    finally:
        summary["runner_returncode"] = rc
        summary["task_complete"] = False
        _write_json(output_dir / "core_scene_summary.json", summary)
        marker = output_dir / ("CORE_SCENE_PASS.json" if rc == 0 else "CORE_SCENE_FAIL.json")
        _write_json(
            marker,
            {
                "schema": "bc012_core_scene_live_v1",
                "core_scene_ready": bool(summary.get("core_scene_ready") is True),
                "task_complete": False,
                "summary": "core_scene_summary.json",
                "summary_sha256": _sha256(output_dir / "core_scene_summary.json"),
                "runner_returncode": rc,
            },
        )
        # Isaac Sim 5.1 can stall in framework teardown after MJCF conversion.
        # Bound cleanup exactly like the repository's finite preview runners so
        # a completed smoke test cannot retain the scheduler's GPU lease.
        sys.stdout.flush()
        sys.stderr.flush()
        shutdown_guard = threading.Timer(10.0, lambda: os._exit(rc))
        shutdown_guard.daemon = True
        shutdown_guard.start()
        try:
            if environment is not None:
                try:
                    environment.close()
                except Exception:
                    traceback.print_exc()
                    rc = 1
            simulation_app.close(wait_for_replicator=False)
        finally:
            shutdown_guard.cancel()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
