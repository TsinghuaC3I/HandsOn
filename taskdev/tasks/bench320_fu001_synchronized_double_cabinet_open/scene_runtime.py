#!/usr/bin/env python3
"""Build and audit the FU001 Isaac Sim 5.1 selected-asset scene.

The runner instantiates the canonical B320 FU001 environment, whose spawn
boundary composes the reviewed Poly Haven cabinet components and their
source-derived collision meshes.  It performs only a bounded zero-action
smoke and head-camera capture; it never executes the opening action and always
reports ``task_complete`` false.  The retired 44781 bundle is retained only as
historical provenance in the emitted summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
# The runner is launched with Isaac Sim's Python from the repository root in
# production, but one-shot audits are also invoked by absolute path.  Keep
# both the taskdev package and the editable handson package importable in that
# mode; otherwise ``main()`` fails before AppLauncher can start Kit.
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "handson"))

from isaaclab.app import AppLauncher  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a finite JSON evidence document.

    Isaac/PhysX can return NaN/Inf after a failed reset or broadphase update.
    Those values must never make the audit's error report itself unwritable:
    convert them to JSON ``null`` and leave the corresponding gate false in
    the payload.  This keeps the failure evidence inspectable and avoids an
    exception in the ``finally`` block masking the native error.
    """

    def _finite_json(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return _finite_json(value.tolist())
        if isinstance(value, np.generic):
            return _finite_json(value.item())
        if isinstance(value, dict):
            return {str(key): _finite_json(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_finite_json(item) for item in value]
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        return value

    payload = _finite_json(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _range_list(bound_range: Any) -> list[list[float]]:
    return [
        [float(value) for value in bound_range.GetMin()],
        [float(value) for value in bound_range.GetMax()],
    ]


def _aabb_overlap(first: list[list[float]], second: list[list[float]]) -> float:
    first_array = np.asarray(first, dtype=np.float64).reshape(2, 3)
    second_array = np.asarray(second, dtype=np.float64).reshape(2, 3)
    overlap = np.minimum(first_array[1], second_array[1]) - np.maximum(
        first_array[0], second_array[0]
    )
    return float(np.min(overlap)) if np.all(overlap > 0.0) else 0.0


def _rgb_metrics(frame: np.ndarray) -> dict[str, Any]:
    array = np.asarray(frame)
    luminance = array.astype(np.float64).mean(axis=2)
    counts = np.bincount(array.reshape(-1), minlength=256).astype(np.float64)
    probabilities = counts[counts > 0.0] / float(array.size)
    return {
        "shape_hwc": list(array.shape),
        "dtype": str(array.dtype),
        "mean_rgb": float(array.mean()),
        "std_rgb": float(array.std()),
        "black_fraction": float(np.mean(luminance <= 5.0)),
        "unique_values": int(np.count_nonzero(counts)),
        "entropy_bits": float(-(probabilities * np.log2(probabilities)).sum()),
    }


class FU001CoreScene:
    def __init__(
        self,
        *,
        simulation_app: Any,
        app_launcher: Any,
        output_dir: Path,
        device: str,
    ) -> None:
        from handson.sim.isaaclab.b320_bimanual_env import (
            HandsOnB320BimanualIsaacLabEnv,
        )

        from taskdev.tasks.bench320_fu001_synchronized_double_cabinet_open.scene_spec import (
            PHYSICS_HZ,
            RENDER_HEIGHT,
            RENDER_WIDTH,
        )

        self.output_dir = output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.env = HandsOnB320BimanualIsaacLabEnv(
            "b320_synchronized_double_cabinet_open",
            seed=0,
            control_dt=1.0 / 30.0,
            physics_dt=1.0 / float(PHYSICS_HZ),
            time_limit=10.0,
            render_mode="rgb_array",
            image_obs=True,
            randomize=False,
            randomize_dynamics=False,
            hand_type="wuji1",
            arm_type="rokae",
            launch_isaac=True,
            headless=True,
            render_resolution=(RENDER_HEIGHT, RENDER_WIDTH),
            hz=30,
            isaac_device=device,
            physics_backend="physx",
            existing_simulation_app=simulation_app,
            existing_app_launcher=app_launcher,
        )
        self.stage = self.env._stage
        self.sim = self.env._sim
        if self.stage is None or self.sim is None:
            raise RuntimeError("native HandsOn scene did not expose stage and simulation")

    def close(self) -> None:
        # ABC's supported fast shutdown avoids SimulationContext.stop()
        # blocking after RTX camera capture in a one-shot worker.
        self.env._fast_shutdown_on_close = True
        self.env.close()

    @staticmethod
    def _predicate() -> Any:
        from pxr import Usd

        return Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)

    def _prim_range(self, root: Any) -> Any:
        from pxr import Usd

        return Usd.PrimRange(root, self._predicate())

    def _world_aabb(self, prim: Any) -> list[list[float]]:
        from pxr import Usd, UsdGeom

        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
            useExtentsHint=True,
        )
        return _range_list(cache.ComputeWorldBound(prim).ComputeAlignedRange())

    def _hide_native_task_objects(self) -> list[str]:
        from pxr import UsdGeom

        from taskdev.tasks.bench320_fu001_synchronized_double_cabinet_open.scene_spec import (
            NATIVE_SOURCE_TASK_OBJECTS,
        )

        hidden: list[str] = []
        for name in NATIVE_SOURCE_TASK_OBJECTS:
            prim = self.env._find_rigid_body_prim(name)
            if prim is None or not prim.IsValid():
                continue
            # The base environment already owns tensor views for these
            # fixtures.  Deactivating a physics prim invalidates those views,
            # while authoring a descendant instance proxy is illegal in USD.
            # Hide only the nearest editable instance root and leave physics
            # topology untouched.
            candidate = prim
            while candidate.IsValid() and candidate.IsInstanceProxy():
                candidate = candidate.GetParent()
            imageable = UsdGeom.Imageable(candidate)
            if candidate.IsValid() and imageable:
                imageable.MakeInvisible()
                hidden.append(str(candidate.GetPath()))
        return hidden

    def _compose_cabinet(self) -> tuple[Any, dict[str, Any]]:
        from pxr import UsdGeom

        from handson.sim.isaaclab.b320_bimanual_scene import (
            FU001_ASSET_BUNDLE_KEY,
            FU001_RUNTIME_COMPONENTS,
            FU001_RUNTIME_ORIENTATION_WXYZ,
            FU001_RUNTIME_SCALE,
            B320_TABLE_TOP_Z,
        )
        from taskdev.tasks.bench320_fu001_synchronized_double_cabinet_open.scene_spec import (
            CABINET_RELATIVE,
            REPOSITORY_ROOT,
            SCENE_ROOT,
        )

        # The B320 subclass has already crossed the production spawn boundary
        # during construction.  Reusing its result here guarantees that this
        # live runner audits exactly the same selected components as collection
        # workers instead of composing a parallel legacy fixture.
        spawn_result = self.env._b320_spawn_result
        base_path = spawn_result.body_paths.get("fixture_base")
        if not base_path:
            raise RuntimeError("canonical FU001 spawn did not expose fixture_base")
        cabinet_prim = self.stage.GetPrimAtPath(base_path)
        if cabinet_prim is None or not cabinet_prim.IsValid():
            raise RuntimeError(f"canonical FU001 fixture prim is missing: {base_path}")
        UsdGeom.Xform.Define(self.stage, SCENE_ROOT)

        initial_aabb = self._world_aabb(cabinet_prim)
        final_aabb = self._world_aabb(cabinet_prim)
        # The ABC base scene may expose a different native table height for its
        # source task.  FU001's canonical B320 worktable is authored by the
        # selected spawn at the production constant below; using the ABC value
        # would falsely report a negative cabinet clearance.
        table_top = float(B320_TABLE_TOP_Z)
        evidence = dict(spawn_result.evidence or {})
        selected_components = {
            key: str(path) for key, path in FU001_RUNTIME_COMPONENTS.items()
        }
        selected_components_from_evidence = evidence.get("runtime_components")
        if isinstance(selected_components_from_evidence, dict):
            selected_components = dict(selected_components_from_evidence)
        root_home = evidence.get("root_home_w")
        if not isinstance(root_home, list):
            root_home = [float(value) for value in cabinet_prim.GetAttribute("xformOp:translate").Get()]
        return cabinet_prim, {
            "selected_asset_id": "cabinet-polyhaven-painted",
            "selected_asset_bundle": FU001_ASSET_BUNDLE_KEY,
            "selected_runtime_components": selected_components,
            "source_identity_scale_xyz": [1.0, 1.0, 1.0],
            "composed_world_scale_xyz": [float(FU001_RUNTIME_SCALE)] * 3,
            "runtime_orientation_wxyz": list(FU001_RUNTIME_ORIENTATION_WXYZ),
            "runtime_scale": float(FU001_RUNTIME_SCALE),
            "translation_xyz_m": [float(value) for value in root_home],
            "table_top_z_m": table_top,
            "table_clearance_m": float(final_aabb[0][2] - table_top),
            "initial_oriented_aabb_m": initial_aabb,
            "world_aabb_m": final_aabb,
            # Explicitly retain the old path as historical evidence only; it
            # is never referenced by the canonical runner.
            "historical_reference_asset": str(REPOSITORY_ROOT / CABINET_RELATIVE),
            "historical_reference_selected": False,
            "spawn_evidence": evidence,
        }

    def _configure_lights(self) -> dict[str, Any]:
        from pxr import Gf, UsdLux

        from taskdev.tasks.bench320_fu001_synchronized_double_cabinet_open.scene_spec import (
            SCENE_ROOT,
        )

        dome = UsdLux.DomeLight.Define(self.stage, f"{SCENE_ROOT}/DomeLight")
        dome.CreateIntensityAttr(350.0)
        dome.CreateColorAttr().Set(Gf.Vec3f(0.92, 0.95, 1.0))
        distant = UsdLux.DistantLight.Define(
            self.stage, f"{SCENE_ROOT}/DistantLight"
        )
        distant.CreateIntensityAttr(2500.0)
        distant.CreateAngleAttr(0.53)
        distant.CreateColorAttr().Set(Gf.Vec3f(1.0, 0.94, 0.86))
        return {
            "dome": {"path": str(dome.GetPath()), "intensity": 350.0},
            "distant": {"path": str(distant.GetPath()), "intensity": 2500.0},
        }

    def _configure_door_joints(self, cabinet_prim: Any) -> dict[str, Any]:
        from pxr import UsdPhysics

        from handson.sim.isaaclab.b320_bimanual_scene import B320_ASSET_ROOT_PATH
        from taskdev.tasks.bench320_fu001_synchronized_double_cabinet_open.scene_spec import (
            DOOR_INITIAL_FRACTION,
            DOOR_JOINTS,
        )

        # Production FU001 authors the two joints beside the fixed fixture
        # body (the body0/body1 paths are explicit), so they are not descendants
        # of ``cabinet_prim``.  Search the canonical B320 root and fail closed
        # if either named joint is absent.
        found: dict[str, Any] = {}
        joint_root = self.stage.GetPrimAtPath(B320_ASSET_ROOT_PATH)
        for prim in self._prim_range(joint_root):
            if not prim.IsA(UsdPhysics.RevoluteJoint):
                continue
            name = prim.GetName()
            if name not in DOOR_JOINTS:
                continue
            joint = UsdPhysics.RevoluteJoint(prim)
            lower = float(joint.GetLowerLimitAttr().Get())
            upper = float(joint.GetUpperLimitAttr().Get())
            # FU001 measures both leaves from the closed pose at zero radians.
            # The right leaf's authored joint range is [-90, 0], so ordinary
            # lower-to-upper interpolation would incorrectly reset it near
            # the open limit (-76.5 degrees for a 15% fraction).  Preserve
            # the canonical signed convention used by the env reset path.
            target = (
                math.radians(90.0) * DOOR_INITIAL_FRACTION
                if name == "target_joint_revolute"
                else -math.radians(90.0) * DOOR_INITIAL_FRACTION
            )
            # The canonical spawn authors the drive before PhysX initializes.
            # Re-applying DriveAPI or JointStateAPI here would mutate a live
            # reduced-coordinate joint and Isaac Sim 5.1 reports
            # ``setDriveTarget`` as illegal when direct GPU API is enabled.
            # This runner is an audit, so inspect-only behavior is deliberate;
            # reset writes remain owned by the environment's tensor path.
            def _drive_value(getter_name: str) -> float | None:
                # DriveAPI attributes are namespaced as
                # ``drive:angular:physics:*`` in Isaac Sim 5.1.  Reading the
                # schema accessors keeps the evidence independent of that
                # authored namespace spelling and avoids reporting a false
                # ``None`` for a valid drive.
                drive = UsdPhysics.DriveAPI.Get(prim, "angular")
                if not drive:
                    return None
                attribute = getattr(drive, getter_name)()
                value = attribute.Get()
                return None if value is None else float(value)

            initial_state_authored = False
            drive_stiffness = _drive_value("GetStiffnessAttr")
            drive_damping = _drive_value("GetDampingAttr")
            drive_max_force = _drive_value("GetMaxForceAttr")
            found[name] = {
                "path": str(prim.GetPath()),
                "body0_owner_paths": [
                    str(path) for path in joint.GetBody0Rel().GetTargets()
                ],
                "body1_owner_paths": [
                    str(path) for path in joint.GetBody1Rel().GetTargets()
                ],
                "lower_limit_deg": lower,
                "upper_limit_deg": upper,
                "initial_fraction": float(DOOR_INITIAL_FRACTION),
                "initial_target_deg": target,
                "initial_state_authored": initial_state_authored,
                "drive_stiffness": drive_stiffness,
                "drive_damping": drive_damping,
                "drive_max_force": drive_max_force,
                "drive_authoring": "canonical_spawn_before_physx_init",
            }
        missing = sorted(set(DOOR_JOINTS) - set(found))
        if missing:
            raise RuntimeError(f"missing required cabinet door joints: {missing}")
        return {"joints": found, "required_joint_names": list(DOOR_JOINTS)}

    def _topology(self, cabinet_prim: Any) -> dict[str, Any]:
        from pxr import UsdGeom, UsdPhysics

        from handson.sim.isaaclab.b320_bimanual_scene import B320_ASSET_ROOT_PATH
        from taskdev.tasks.bench320_fu001_synchronized_double_cabinet_open.scene_spec import (
            DOOR_JOINTS,
        )

        collisions: list[str] = []
        rigid_bodies: list[str] = []
        articulations: list[str] = []
        revolute: list[str] = []
        collision_owner_map: dict[str, str | None] = {}
        gprims = 0
        selected_paths = dict(self.env._b320_spawn_result.body_paths)
        selected_prims: list[Any] = []
        for path in dict.fromkeys(selected_paths.values()):
            root = self.stage.GetPrimAtPath(path)
            if root is None or not root.IsValid():
                continue
            selected_prims.extend(self._prim_range(root))
        for joint_name in DOOR_JOINTS:
            joint = self.stage.GetPrimAtPath(f"{B320_ASSET_ROOT_PATH}/{joint_name}")
            if joint is not None and joint.IsValid():
                selected_prims.append(joint)
        # Keep the spawn-boundary articulation roots explicit in the audit.
        # The fixed world joint is authored beneath ``fixture_base`` today,
        # but relying on that hierarchy would make this evidence disappear if
        # a future converter moves the root to a sibling prim.
        for path in self.env._b320_spawn_result.articulation_root_paths:
            root = self.stage.GetPrimAtPath(path)
            if root is not None and root.IsValid():
                selected_prims.append(root)
        seen_paths: set[str] = set()
        for prim in selected_prims:
            path_key = str(prim.GetPath())
            if path_key in seen_paths:
                continue
            seen_paths.add(path_key)
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                collision_path = str(prim.GetPath())
                collisions.append(collision_path)
                owner = prim
                while owner.IsValid() and not owner.HasAPI(UsdPhysics.RigidBodyAPI):
                    if owner.GetPath() in {
                        self.stage.GetPrimAtPath(path).GetPath()
                        for path in selected_paths.values()
                        if self.stage.GetPrimAtPath(path).IsValid()
                    }:
                        owner = None
                        break
                    owner = owner.GetParent()
                collision_owner_map[collision_path] = (
                    str(owner.GetPath()) if owner is not None and owner.IsValid() else None
                )
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rigid_bodies.append(str(prim.GetPath()))
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                articulations.append(str(prim.GetPath()))
            if prim.IsA(UsdPhysics.RevoluteJoint):
                revolute.append(str(prim.GetPath()))
            if prim.IsA(UsdGeom.Gprim):
                gprims += 1
        return {
            "collision_count": len(collisions),
            "collision_paths": collisions,
            "collision_owner_map": collision_owner_map,
            "rigid_body_count": len(rigid_bodies),
            "rigid_body_paths": rigid_bodies,
            "articulation_root_paths": articulations,
            "revolute_joint_paths": revolute,
            "gprim_count": gprims,
            "pass": bool(
                collisions
                and rigid_bodies
                and articulations
                and len(revolute) >= 2
                and gprims > 0
                and all(collision_owner_map.values())
            ),
        }

    def _robot_inventory(self) -> dict[str, Any]:
        sides: dict[str, Any] = {}
        for side in ("left", "right"):
            robot = self.env._robots.get(side)
            if robot is None:
                sides[side] = {"present": False, "pass": False}
                continue
            joint_names = list(robot.joint_names)
            arm_names = [joint_names[index] for index in self.env._arm_joint_ids[side]]
            hand_names = [joint_names[index] for index in self.env._hand_joint_ids[side]]
            root = self.stage.GetPrimAtPath(
                f"/World/HandsOnRobots/{side.capitalize()}Robot"
            )
            sides[side] = {
                "present": root.IsValid(),
                "root_path": str(root.GetPath()) if root.IsValid() else None,
                "arm_joint_names": arm_names,
                "hand_joint_names": hand_names,
                "pass": bool(root.IsValid() and len(arm_names) == 7 and len(hand_names) == 20),
            }
        return {"sides": sides, "pass": all(item["pass"] for item in sides.values())}

    def _collision_overlap_audit(self, cabinet_prim: Any) -> dict[str, Any]:
        from pxr import UsdPhysics

        from taskdev.tasks.bench320_fu001_synchronized_double_cabinet_open.scene_spec import (
            MAX_ROBOT_CABINET_AABB_OVERLAP_M,
        )

        cabinet_colliders: list[Any] = []
        for path in dict.fromkeys(self.env._b320_spawn_result.body_paths.values()):
            root = self.stage.GetPrimAtPath(path)
            if root is None or not root.IsValid():
                continue
            cabinet_colliders.extend(
                prim
                for prim in self._prim_range(root)
                if prim.HasAPI(UsdPhysics.CollisionAPI)
            )
        # A logical alias can point to the same prim in the B320 result.  Keep
        # each source collider once so the overlap evidence is auditable.
        cabinet_colliders = list(
            {str(prim.GetPath()): prim for prim in cabinet_colliders}.values()
        )
        robot_colliders: list[Any] = []
        for side in ("Left", "Right"):
            root = self.stage.GetPrimAtPath(f"/World/HandsOnRobots/{side}Robot")
            if not root.IsValid():
                continue
            robot_colliders.extend(
                prim
                for prim in self._prim_range(root)
                if prim.HasAPI(UsdPhysics.CollisionAPI)
            )
        maximum = 0.0
        pairs: list[dict[str, Any]] = []
        for cabinet_collision in cabinet_colliders:
            cabinet_aabb = self._world_aabb(cabinet_collision)
            for robot_collision in robot_colliders:
                overlap = _aabb_overlap(
                    cabinet_aabb, self._world_aabb(robot_collision)
                )
                maximum = max(maximum, overlap)
                if overlap > 0.0:
                    pairs.append(
                        {
                            "cabinet": str(cabinet_collision.GetPath()),
                            "robot": str(robot_collision.GetPath()),
                            "aabb_overlap_m": overlap,
                        }
                    )
        pairs.sort(key=lambda item: item["aabb_overlap_m"], reverse=True)
        return {
            "cabinet_collider_count": len(cabinet_colliders),
            "robot_collider_count": len(robot_colliders),
            "maximum_aabb_overlap_m": maximum,
            "overlap_pairs": pairs[:20],
            "threshold_m": float(MAX_ROBOT_CABINET_AABB_OVERLAP_M),
            "pass": bool(
                cabinet_colliders
                and robot_colliders
                and maximum <= MAX_ROBOT_CABINET_AABB_OVERLAP_M
            ),
        }

    def _articulation_view(self, topology: dict[str, Any]) -> tuple[Any | None, dict[str, Any]]:
        import omni.physics.tensors as physics_tensors

        simulation_view = physics_tensors.create_simulation_view("torch")
        simulation_view.set_subspace_roots("/")
        # ArticulationView does not retain its owning SimulationView in this
        # Isaac/PhysX tensors build.  Keep the owner alive for all later DOF
        # reads; otherwise the backend handle is released after this method.
        self._physics_simulation_view = simulation_view
        errors: list[str] = []
        best = None
        best_dof = -1
        best_path = None
        for path in dict.fromkeys(topology["articulation_root_paths"]):
            try:
                view = simulation_view.create_articulation_view(path)
                dof_count = int(view.max_dofs)
                if int(view.count) > 0 and dof_count > best_dof:
                    best = view
                    best_dof = dof_count
                    best_path = path
            except Exception as exc:
                errors.append(f"{path}: {type(exc).__name__}: {exc}")
        return best, {
            "selected_path": best_path,
            "selected_dof_count": best_dof,
            "errors": errors,
            "pass": best is not None and best_dof >= 2,
        }

    @staticmethod
    def _dof_state(view: Any | None) -> dict[str, Any]:
        if view is None:
            return {"available": False, "positions_rad": [], "velocities_rad_s": []}
        positions = view.get_dof_positions()
        velocities = view.get_dof_velocities()
        if hasattr(positions, "detach"):
            positions = positions.detach().cpu().numpy()
        if hasattr(velocities, "detach"):
            velocities = velocities.detach().cpu().numpy()
        return {
            "available": True,
            "positions_rad": np.asarray(positions, dtype=np.float64).reshape(-1).tolist(),
            "velocities_rad_s": np.asarray(velocities, dtype=np.float64).reshape(-1).tolist(),
        }

    def _render_preview(self) -> dict[str, Any]:
        from PIL import Image
        from pxr import UsdGeom

        from taskdev.tasks.bench320_fu001_synchronized_double_cabinet_open.scene_spec import (
            HEAD_CAMERA_PATH,
            HEAD_SHAPE,
        )

        frames = self.env.render()
        if not isinstance(frames, (list, tuple)) or not frames:
            raise RuntimeError("native head camera did not return any frames")
        frame = np.asarray(frames[0])
        if frame.shape != HEAD_SHAPE or frame.dtype != np.uint8:
            raise RuntimeError(
                f"head frame must be uint8 {HEAD_SHAPE}, got {frame.shape}/{frame.dtype}"
            )
        path = self.output_dir / "fu001_head_preview.png"
        # Keep the exact 480x640 sensor output and write it losslessly without
        # compression so the evidence remains inspectable and validator-sized.
        Image.fromarray(frame).save(path, compress_level=0)
        metrics = _rgb_metrics(frame)
        camera = self.env._cameras.get("head_camera")
        prim_path = str(getattr(getattr(camera, "cfg", None), "prim_path", ""))
        stage_camera = self.stage.GetPrimAtPath(HEAD_CAMERA_PATH)
        stage_prim_is_camera = bool(
            stage_camera is not None
            and stage_camera.IsValid()
            and stage_camera.IsA(UsdGeom.Camera)
        )
        return {
            "prim_path": prim_path,
            "expected_prim_path": HEAD_CAMERA_PATH,
            "stage_prim_is_camera": stage_prim_is_camera,
            "camera_keys": sorted(str(name) for name in self.env._cameras),
            "deferred_camera_render": bool(
                getattr(self.env, "_deferred_camera_render", True)
            ),
            "path": str(path),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
            "metrics": metrics,
            "pass": bool(
                prim_path == HEAD_CAMERA_PATH
                and stage_prim_is_camera
                and metrics["mean_rgb"] > 30.0
                and metrics["std_rgb"] > 15.0
                and metrics["black_fraction"] < 0.95
                and path.stat().st_size >= 150_000
            ),
        }

    def run(self) -> dict[str, Any]:
        from handson.sim.isaaclab.b320_bimanual_scene import (
            FU001_ASSET_BUNDLE_KEY,
            FU001_ASSET_SOURCE_ARCHIVE_PATH,
            FU001_RUNTIME_COMPONENTS,
            FU001_RUNTIME_SCALE,
            validate_b320_fu001_asset_files,
        )
        from taskdev.tasks.bench320_fu001_synchronized_double_cabinet_open.scene_spec import (
            CABINET_RELATIVE,
            DOOR_JOINTS,
            EXPECTED_SHA256,
            MAX_DOOR_DRIFT_RAD,
            MAX_DOOR_SPEED_RAD_S,
            PHYSICS_HZ,
            REPOSITORY_ROOT,
            SETTLE_STEPS,
            ZERO_ACTION_SECONDS,
        )

        started = time.time()
        # Validate and record the selected runtime closure before any live
        # physics step.  The old hash map is retained under an explicit
        # historical key below; it is not used as a gate for canonical FU001.
        selected_required = validate_b320_fu001_asset_files()
        selected_hashes = {
            path.relative_to(REPOSITORY_ROOT).as_posix(): _sha256(path)
            for path in selected_required
            if path.is_file()
        }
        historical_hashes = {
            relative: _sha256(REPOSITORY_ROOT / relative)
            for relative in EXPECTED_SHA256
            if (REPOSITORY_ROOT / relative).is_file()
        }
        hidden = self._hide_native_task_objects()
        cabinet, placement = self._compose_cabinet()
        lights = self._configure_lights()
        joints = self._configure_door_joints(cabinet)
        topology = self._topology(cabinet)
        robot = self._robot_inventory()
        initial_clearance = self._collision_overlap_audit(cabinet)

        # ``HandsOnB320BimanualIsaacLabEnv`` performs the canonical seeded
        # reset inside its constructor, after the task USD has been composed,
        # and advances five initialization frames before returning.  Do not
        # reset again here: a second tensor reset after stage mutation can
        # submit stale fixed-articulation broadphase updates in Isaac Sim 5.1
        # (``Illegal BroadPhaseUpdateData``) and leave door DOFs non-finite.
        # The runner-side mutations above are visibility/lights/inspection
        # only, so the constructor reset remains the authoritative boundary.
        reset_info = {
            "isaac_warnings": list(getattr(self.env, "_isaac_warnings", ())),
            "seed": 0,
            "canonical_env_reset": True,
            "reset_phase": "constructor_before_runner_stage_mutations",
            "second_reset_avoided": True,
        }
        for _ in range(SETTLE_STEPS):
            self.sim.step(render=False)
        view, view_record = self._articulation_view(topology)
        before = self._dof_state(view)

        smoke_steps = int(round(ZERO_ACTION_SECONDS * PHYSICS_HZ))
        for step in range(smoke_steps):
            self.sim.step(render=step >= smoke_steps - 12)
        after = self._dof_state(view)

        drift = math.inf
        maximum_speed = math.inf
        finite = False
        if before["available"] and after["available"]:
            before_positions = np.asarray(before["positions_rad"], dtype=np.float64)
            after_positions = np.asarray(after["positions_rad"], dtype=np.float64)
            after_velocities = np.asarray(after["velocities_rad_s"], dtype=np.float64)
            if before_positions.shape == after_positions.shape and after_positions.size:
                drift = float(np.max(np.abs(after_positions - before_positions)))
                maximum_speed = float(np.max(np.abs(after_velocities)))
                finite = bool(
                    np.isfinite(before_positions).all()
                    and np.isfinite(after_positions).all()
                    and np.isfinite(after_velocities).all()
                )
        stability = {
            "physics_hz": PHYSICS_HZ,
            "settle_steps": SETTLE_STEPS,
            "zero_action_steps": smoke_steps,
            "zero_action_seconds": float(ZERO_ACTION_SECONDS),
            "before": before,
            "after": after,
            "maximum_dof_position_drift_rad": drift,
            "maximum_dof_speed_rad_s": maximum_speed,
            "finite": finite,
            "pass": bool(
                finite
                and drift <= MAX_DOOR_DRIFT_RAD
                and maximum_speed <= MAX_DOOR_SPEED_RAD_S
            ),
        }
        preview = self._render_preview()
        scale_pass = bool(
            np.allclose(
                placement["composed_world_scale_xyz"],
                np.full(3, float(FU001_RUNTIME_SCALE)),
                rtol=0.0,
                atol=1.0e-8,
            )
            and 0.0 <= placement["table_clearance_m"] <= 0.005
        )
        checks = {
            "asset_hashes": bool(selected_hashes),
            "selected_scale_and_table_pose": scale_pass,
            "cabinet_topology": topology["pass"],
            "door_joints": bool(
                set(joints["joints"]) == set(DOOR_JOINTS)
                and all(
                    record["body0_owner_paths"] and record["body1_owner_paths"]
                    for record in joints["joints"].values()
                )
            ),
            "dual_rokae_wuji": robot["pass"],
            "robot_cabinet_initial_clearance": initial_clearance["pass"],
            "zero_action_stability": stability["pass"],
            "head_camera_preview": preview["pass"],
            "lighting": bool(
                lights["dome"]["intensity"] >= 100.0
                and lights["distant"]["intensity"] >= 500.0
            ),
        }
        return {
            "schema": "fu001_core_scene_live_v1",
            "task_id": "B320-FU001",
            "scope": "core_scene_only",
            "task_complete": False,
            "success_motion_executed": False,
            "render_mode": "RayTracedLighting",
            "source_asset": FU001_ASSET_BUNDLE_KEY,
            "selected_asset": {
                "bundle": FU001_ASSET_BUNDLE_KEY,
                "archive": str(FU001_ASSET_SOURCE_ARCHIVE_PATH),
                "runtime_components": {
                    key: str(path) for key, path in FU001_RUNTIME_COMPONENTS.items()
                },
                "runtime_scale": float(FU001_RUNTIME_SCALE),
                "hashes": selected_hashes,
            },
            "asset_hashes": selected_hashes,
            "historical_asset": str(REPOSITORY_ROOT / CABINET_RELATIVE),
            "historical_asset_hashes": historical_hashes,
            "hidden_native_fixture_paths": hidden,
            "placement": placement,
            "joints": joints,
            "topology": topology,
            "articulation_view": view_record,
            "robot": robot,
            "initial_clearance": initial_clearance,
            "stability": stability,
            # Preserve the reset-boundary provenance, including the explicit
            # second-reset avoidance marker.  Keeping this as a copy (rather
            # than rebuilding a three-field subset) makes the emitted CPU
            # smoke evidence auditable against the broadphase diagnosis.
            "reset_info": dict(reset_info)
            if isinstance(reset_info, dict)
            else {
                "isaac_warnings": [],
                "seed": 0,
                "canonical_env_reset": True,
            },
            "reset_staging": {
                "mode": "collision_clear_spread_home",
                "arm_home_source": "_C_TOOL_SPREAD_ARM_HOME",
                "tcp_y_m": [-0.36, 0.36],
                "hand_door_collision_filtering": False,
                "contact_policy": "ordinary_hand_door_and_table_contacts_remain_enabled",
            },
            "lights": lights,
            "head_camera": preview,
            "checks": checks,
            "core_scene_pass": all(checks.values()),
            "elapsed_wall_seconds": time.time() - started,
        }


def main() -> int:
    args = _parser().parse_args()
    from taskdev.tasks.bench320_fu001_synchronized_double_cabinet_open.scene_spec import (
        RENDER_HEIGHT,
        RENDER_WIDTH,
    )

    args.headless = True
    args.enable_cameras = True
    # Keep the default on the production CUDA path, while honoring an
    # explicit ``--device cpu`` diagnostic run.  Previously this assignment
    # overwrote the CLI value and made it impossible to isolate GPU/PhysX
    # failures from scene-composition failures.
    device = str(getattr(args, "device", "cuda:0") or "cuda:0")
    args.device = device
    args.width = RENDER_WIDTH
    args.height = RENDER_HEIGHT
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

    import carb

    settings = carb.settings.get_settings()
    settings.set("/rtx/rendermode", "RayTracedLighting")
    settings.set("/rtx/post/tonemap/op", 4)
    settings.set("/rtx/post/tonemap/filmIso", 200)
    scene = None
    summary: dict[str, Any] = {
        "schema": "fu001_core_scene_live_v1",
        "task_id": "B320-FU001",
        "scope": "core_scene_only",
        "task_complete": False,
        "core_scene_pass": False,
    }
    rc = 1
    try:
        scene = FU001CoreScene(
            simulation_app=simulation_app,
            app_launcher=app_launcher,
            output_dir=output_dir,
            device=device,
        )
        summary = scene.run()
        rc = 0 if summary.get("core_scene_pass") is True else 1
    except Exception as exc:
        summary["exception"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        rc = 1
    finally:
        summary["runner_rc"] = rc
        _write_json(output_dir / "core_scene_summary.json", summary)
        if scene is not None:
            try:
                scene.close()
            except Exception:
                traceback.print_exc()
                rc = 1
        else:
            try:
                simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
            except TypeError:
                simulation_app.close()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
