"""GPU PBD live proxy for E01 garment collection.

This environment deliberately keeps the high-quality FEM.Cloth/libuipc
implementation in :mod:`e01_fold_garment_env` untouched.  PBD is used here as
an interactive proxy: the canonical garment asset and canonical black Wuji
hands are rendered live, while a guarded particle-to-fingertip attachment
stands in for rigid-cloth friction that Genesis 1.2's legacy PBD coupler does
not implement.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import threading
from typing import Any, Mapping

import gymnasium as gym
import numpy as np

from .e01_fold_garment_env import (
    HandsOnE01GenesisEnv,
    _GENESIS_GARMENT_EULER,
    _GENESIS_GARMENT_SCALE,
    _HAND_DIM,
    _HAND_TABLE_CLEARANCE,
    _HEAD_CAMERA_EYE,
    _HEAD_CAMERA_LOOKAT,
    _IMAGE_SHAPE,
    _LEFT_BUILD_TCP,
    _LEFT_HOME_TCP,
    _RIGHT_BUILD_TCP,
    _RIGHT_HOME_TCP,
    _SUCCESS_STEPS,
    _TABLE_TOP_Z,
    _WUJI_HAND_HOME,
    _WRIST_CAMERA_MOUNT_POSES,
    _WRIST_CAMERA_PALM_POSES,
    _env_float,
    _env_vector,
    _pose_to_matrix,
    _resolve_genesis_garment_obj,
    _to_numpy,
)
from .mjcf_assets import prepare_detached_rokae_wuji_hand_mjcf


_PBD_GENESIS_INIT_LOCK = threading.Lock()
_PBD_GENESIS_INITIALIZED = False

_PBD_PARTICLE_SIZE = 0.010
_PBD_SETTLE_STEPS = 90
_PBD_PINCH_CLOSE_DISTANCE = 0.060
_PBD_PINCH_RELEASE_DISTANCE = 0.085
# The link4 frame lies about 8 mm behind the authored fingertip collision
# surface at table contact, so a 40 mm link-frame gate corresponds to roughly
# 30 mm physical cloth proximity while remaining far from unconditional suction.
_PBD_ATTACH_DISTANCE = 0.040
_PBD_SETTLE_AIR_RESISTANCE = 1.0e-3
_PBD_LIVE_AIR_RESISTANCE = 1.0e-3
_PBD_COLLISION_SAFE_TRANSLATION_STEP = 0.003
_PBD_COLLISION_SAFE_ROTATION_STEP = math.radians(3.0)
_PBD_CONTACT_SLEW_DISTANCE = 0.18
_PBD_HAND_CONTACT_MAX_CORRECTION = 0.002
_PBD_FINGERTIP_RADIUS = 0.0085
_PBD_PALM_HALF_EXTENTS = np.asarray(
    [0.035, 0.026, 0.055], dtype=np.float64
)
# PBD applies Mesh poses about the asset origin, unlike FEM's COM-pivoted
# convention. This placement yields the same centered, z=90-degree shirt pose
# as the offline IPC scene without changing either downloaded mesh.
_PBD_GARMENT_POSITION = (1.010873, -0.000024, 1.005735)
_PBD_DOMAIN_LOWER = np.asarray([-1.0, -1.0, 0.0], dtype=np.float64)
_PBD_DOMAIN_UPPER = np.asarray([1.0, 1.0, 2.0], dtype=np.float64)


def _sphere_contact_penetration(
    nodes: np.ndarray,
    center: np.ndarray,
    *,
    radius: float,
    margin: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return penetration depth and outward normal for an expanded sphere."""
    delta = np.asarray(nodes, dtype=np.float64) - np.asarray(
        center, dtype=np.float64
    ).reshape(1, 3)
    distance = np.linalg.norm(delta, axis=1)
    target_distance = float(radius + margin)
    depth = np.maximum(0.0, target_distance - distance)
    normals = np.zeros_like(delta)
    nonzero = distance > 1.0e-12
    normals[nonzero] = delta[nonzero] / distance[nonzero, None]
    normals[~nonzero, 2] = -1.0
    return depth, normals


def _box_contact_penetration(
    nodes: np.ndarray,
    center: np.ndarray,
    quat: np.ndarray,
    *,
    half_extents: np.ndarray,
    margin: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one-face penetration for a particle-expanded oriented box."""
    transform = _pose_to_matrix(
        np.concatenate(
            [
                np.asarray(center, dtype=np.float64).reshape(3),
                np.asarray(quat, dtype=np.float64).reshape(4),
            ]
        )
    )
    rotation = transform[:3, :3]
    local = (np.asarray(nodes, dtype=np.float64) - transform[:3, 3]) @ rotation
    expanded = np.asarray(half_extents, dtype=np.float64).reshape(3) + float(
        margin
    )
    face_distance = expanded.reshape(1, 3) - np.abs(local)
    inside = np.all(face_distance > 0.0, axis=1)
    axis = np.argmin(face_distance, axis=1)
    rows = np.arange(local.shape[0])
    depth = np.where(inside, face_distance[rows, axis], 0.0)
    local_normals = np.zeros_like(local)
    signs = np.where(local[rows, axis] >= 0.0, 1.0, -1.0)
    local_normals[rows[inside], axis[inside]] = signs[inside]
    return depth, local_normals @ rotation.T


def _make_pbd_garment_surface(gs: Any, asset_path: Path) -> Any:
    """Return the downloaded Cotton_40s_Poplin diffuse/normal material."""
    textures_dir = asset_path.parent / "textures"
    diffuse_path = textures_dir / "Top578_diffuse_1001.png"
    normal_path = textures_dir / "Top578_normal_1001.png"
    if diffuse_path.is_file():
        options: dict[str, Any] = {
            "diffuse_texture": gs.textures.ImageTexture(
                image_path=str(diffuse_path), encoding="srgb"
            ),
            "roughness": 0.9,
        }
        if normal_path.is_file():
            options["normal_texture"] = gs.textures.ImageTexture(
                image_path=str(normal_path), encoding="linear"
            )
        return gs.surfaces.Rough(**options)
    return gs.surfaces.Rough(color=(0.34, 0.56, 0.88, 1.0), roughness=0.9)


def _install_pbd_visual_mesh(
    gs: Any,
    entity: Any,
    *,
    visual_path: Path,
    scale: float,
    position: tuple[float, float, float],
    euler: tuple[float, float, float],
    surface: Any,
) -> bool:
    """Skin an official UV-seamed render mesh to a connected physics mesh.

    OBJ loaders duplicate geometry at genuine UV seams. Feeding that mesh to
    PBD would physically disconnect the shirt, so the simulation entity is
    first created from the geometry-only OBJ. Before ``Scene.build`` we swap
    only its visual vertices/faces to the official UV-preserving OBJ; Genesis'
    existing particle skinning then drives those render vertices.
    """
    if not visual_path.is_file():
        return False

    import genesis.utils.geom as gu

    visual_morph = gs.morphs.Mesh(
        file=str(visual_path),
        scale=float(scale),
        pos=position,
        euler=euler,
    )
    meshes = gs.Mesh.from_morph_surface(visual_morph, surface)
    if len(meshes) != 1:
        raise RuntimeError(
            f"E01 visual garment must contain one mesh, got {len(meshes)}: "
            f"{visual_path}"
        )
    visual_mesh = meshes[0]
    composed_pos, composed_quat = gu.transform_pos_quat_by_trans_quat(
        np.asarray(visual_morph.offset_pos, dtype=np.float32),
        np.asarray(visual_morph.offset_quat, dtype=np.float32),
        np.asarray(visual_morph.pos, dtype=np.float32),
        np.asarray(visual_morph.quat, dtype=np.float32),
    )
    visual_mesh.apply_transform(
        gu.trans_quat_to_T(composed_pos, composed_quat)
    )
    if visual_mesh.uvs is None or len(visual_mesh.uvs) != len(visual_mesh.verts):
        raise RuntimeError(
            f"E01 official visual garment has no usable UVs: {visual_path}"
        )

    entity._vmesh = visual_mesh
    entity._vverts = np.asarray(visual_mesh.verts, dtype=np.float32)
    entity._vfaces = np.asarray(visual_mesh.faces, dtype=np.int32)
    entity._surface = visual_mesh.surface
    return True


class HandsOnE01GenesisPBDLiveEnv(HandsOnE01GenesisEnv):
    """Fast GPU PBD live proxy with conditional pinch attachments.

    It shares E01's detached-hand control, Plan1 reach cache, table guard and
    per-frame pose/joint limits with :class:`HandsOnE01GenesisEnv`.  It is not
    advertised as the high-quality cloth solver; set
    ``HANDSON_GENESIS_E01_SOLVER=ipc`` to select the libuipc environment.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._pbd_genesis_seed = int(
            os.environ.get("HANDSON_GENESIS_SEED", "0")
        )
        self._pbd_particle_size = _env_float(
            "HANDSON_GENESIS_E01_PBD_PARTICLE_SIZE", _PBD_PARTICLE_SIZE
        )
        if not math.isclose(
            self._pbd_particle_size, _PBD_PARTICLE_SIZE, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                "E01 PBD live proxy requires the validated 10 mm particle size; "
                f"got {self._pbd_particle_size:g} m"
            )
        self._pbd_settle_steps_requested = max(
            0,
            int(
                _env_float(
                    "HANDSON_GENESIS_E01_PBD_SETTLE_STEPS", _PBD_SETTLE_STEPS
                )
            ),
        )
        self._pbd_pinch_close_distance = _env_float(
            "HANDSON_GENESIS_E01_PBD_PINCH_CLOSE_DISTANCE",
            _PBD_PINCH_CLOSE_DISTANCE,
        )
        self._pbd_pinch_release_distance = _env_float(
            "HANDSON_GENESIS_E01_PBD_PINCH_RELEASE_DISTANCE",
            _PBD_PINCH_RELEASE_DISTANCE,
        )
        self._pbd_attach_distance = _env_float(
            "HANDSON_GENESIS_E01_PBD_ATTACH_DISTANCE", _PBD_ATTACH_DISTANCE
        )
        self._pbd_live_air_resistance = _env_float(
            "HANDSON_GENESIS_E01_PBD_AIR_RESISTANCE",
            _PBD_LIVE_AIR_RESISTANCE,
        )
        if self._pbd_pinch_release_distance <= self._pbd_pinch_close_distance:
            raise ValueError(
                "PBD pinch release distance must exceed close distance for hysteresis"
            )
        if self._pbd_attach_distance <= 0.0:
            raise ValueError("PBD attachment distance must be positive")

        self._pbd_settled_particle_positions: np.ndarray | None = None
        self._pbd_settled_particle_velocities: np.ndarray | None = None
        self._pbd_settle_steps_completed = 0
        self._pbd_settle_attempts = 0
        self._pbd_settle_max_attempts = max(
            1,
            int(
                _env_float(
                    "HANDSON_GENESIS_E01_PBD_SETTLE_MAX_ATTEMPTS", 5
                )
            ),
        )
        self._pbd_settled_z_span = float("nan")
        self._pbd_rigid_geom_count = 0
        self._pbd_coupled_rigid_geom_count = 0
        self._pbd_visual_asset_path: Path | None = None
        self._pbd_separate_visual_mesh = False
        self._pbd_source_contact_geoms: dict[str, dict[str, Any]] = {}
        self._pbd_bounded_hand_contact_enabled = False
        self._pbd_hand_contact_count = 0
        self._pbd_hand_contact_max_correction = 0.0
        self._pbd_pinch_links: dict[str, tuple[Any, tuple[Any, ...]]] = {}
        self._pbd_attachment_particle: dict[str, int | None] = {
            "right": None,
            "left": None,
        }
        self._pbd_attachment_link_name: dict[str, str] = {
            "right": "",
            "left": "",
        }
        self._pbd_pinch_distance = {
            "right": float("inf"),
            "left": float("inf"),
        }
        self._pbd_nearest_particle_distance = {
            "right": float("inf"),
            "left": float("inf"),
        }
        self._pbd_attachment_events = 0
        self._pbd_domain_boundary_hit = False
        self._pbd_nonfinite_state = False
        super().__init__(*args, **kwargs)
        # Keep Plan1's workspace, static reach cache, table clearance,
        # rotation and finger guards. Legacy rigid-PBD coupling has no CCD, so
        # paths near cloth receive additional collision-safe translation and
        # rotation limits. The Plan1 limits remain active everywhere else.
        self._pbd_plan1_translation_step = float(
            self._max_tcp_translation_step
        )
        self._pbd_plan1_rotation_step = float(self._max_tcp_rotation_step)
        self._pbd_collision_safe_translation_step = _env_float(
            "HANDSON_GENESIS_E01_PBD_MAX_TCP_TRANSLATION_STEP",
            _PBD_COLLISION_SAFE_TRANSLATION_STEP,
        )
        self._pbd_collision_safe_rotation_step = math.radians(
            _env_float(
                "HANDSON_GENESIS_E01_PBD_MAX_TCP_ROTATION_STEP_DEG",
                math.degrees(_PBD_COLLISION_SAFE_ROTATION_STEP),
            )
        )
        self._pbd_contact_slew_distance = _env_float(
            "HANDSON_GENESIS_E01_PBD_CONTACT_SLEW_DISTANCE",
            _PBD_CONTACT_SLEW_DISTANCE,
        )
        self._pbd_active_translation_step = self._pbd_plan1_translation_step

    @staticmethod
    def _initialize_genesis(gs: Any) -> None:
        """Initialize this process on CUDA; PBD live never falls back to CPU."""
        global _PBD_GENESIS_INITIALIZED
        with _PBD_GENESIS_INIT_LOCK:
            if _PBD_GENESIS_INITIALIZED:
                return
            gs.init(
                backend=gs.gpu,
                precision="32",
                seed=int(os.environ.get("HANDSON_GENESIS_SEED", "0")),
                logging_level=os.environ.get(
                    "HANDSON_GENESIS_LOG_LEVEL", "warning"
                ),
            )
            _PBD_GENESIS_INITIALIZED = True

    def _setup_genesis(self) -> None:
        try:
            import genesis as gs
        except ImportError as exc:
            raise RuntimeError(
                "Genesis E01 PBD live runtime is unavailable. Use the isolated "
                "HANDSON_GENESIS_PYTHON environment."
            ) from exc

        self._initialize_genesis(gs)
        self._asset_path = _resolve_genesis_garment_obj()
        garment_scale = _env_float(
            "HANDSON_GENESIS_E01_GARMENT_SCALE", _GENESIS_GARMENT_SCALE
        )
        garment_position = _env_vector(
            "HANDSON_GENESIS_E01_PBD_GARMENT_POSITION",
            _PBD_GARMENT_POSITION,
        )
        garment_euler = _env_vector(
            "HANDSON_GENESIS_E01_GARMENT_EULER", _GENESIS_GARMENT_EULER
        )

        self._scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.physics_dt,
                gravity=(0.0, 0.0, -9.81),
            ),
            rigid_options=gs.options.RigidOptions(enable_self_collision=False),
            pbd_options=gs.options.PBDOptions(
                particle_size=self._pbd_particle_size,
                max_stretch_solver_iterations=4,
                max_bending_solver_iterations=2,
                lower_bound=tuple(_PBD_DOMAIN_LOWER),
                upper_bound=tuple(_PBD_DOMAIN_UPPER),
            ),
            coupler_options=gs.options.LegacyCouplerOptions(rigid_pbd=True),
            # Genesis' generic particle skinning solves unconstrained affine
            # weights from 12 nearby particles.  A cloth surface is locally
            # rank deficient, so those weights can become extremely large and
            # throw only the render mesh across the frame while the simulated
            # particles remain valid.  Two-neighbor line weights avoid that
            # ill-conditioned surface solve while keeping the official UV mesh
            # attached to the 10 mm proxy.  Genesis 1.2's generated skinning
            # kernel cannot compile the degenerate one-neighbor case.
            vis_options=gs.options.VisOptions(n_support_neighbors=2),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=_HEAD_CAMERA_EYE,
                camera_lookat=_HEAD_CAMERA_LOOKAT,
                camera_fov=80,
                refresh_rate=max(30, self.hz),
            ),
            show_viewer=(
                self.render_mode == "human" and self.show_native_viewer
            ),
        )

        # Keep the floor for the canonical visual layout, but only the table
        # participates in PBD coupling.
        self._scene.add_entity(
            morph=gs.morphs.Plane(),
            material=gs.materials.Rigid(needs_coup=False),
            surface=gs.surfaces.Plastic(color=(0.78, 0.80, 0.82, 1.0)),
        )
        self._scene.add_entity(
            morph=gs.morphs.Box(
                pos=(-0.15, 0.0, 0.86),
                size=(0.8, 1.6, 0.06),
                fixed=True,
            ),
            material=gs.materials.Rigid(
                needs_coup=True,
                coup_friction=_env_float(
                    "HANDSON_GENESIS_E01_RIGID_FRICTION", 0.8
                ),
                coup_softness=0.001,
                sdf_cell_size=0.005,
            ),
            surface=gs.surfaces.Plastic(color=(0.72, 0.72, 0.68, 1.0)),
        )

        xml_dir = Path(__file__).resolve().parents[1] / "envs" / "xmls"
        primitive_dir = xml_dir / "wuji_hand1" / "mjcf"
        for side in ("right", "left"):
            # This helper extracts the hand subtree from the canonical complete
            # ROKAE+Wuji asset, retaining its original black mesh visuals, then
            # swaps collision meshes for every shipped primitive collider.
            hand_xml = prepare_detached_rokae_wuji_hand_mjcf(
                xml_dir / f"rokae_wuji1_{side}.xml",
                primitive_dir / f"{side}_primitive.xml",
                side=side,
                collision_profile="pbd_fingertips",
            )
            hand = self._scene.add_entity(
                morph=gs.morphs.MJCF(
                    file=str(hand_xml),
                    pos=(
                        _RIGHT_BUILD_TCP[:3]
                        if side == "right"
                        else _LEFT_BUILD_TCP[:3]
                    ),
                    quat=(
                        _RIGHT_BUILD_TCP[3:7]
                        if side == "right"
                        else _LEFT_BUILD_TCP[3:7]
                    ),
                    requires_jac_and_IK=False,
                ),
                material=gs.materials.Rigid(
                    # Genesis 1.2 Legacy PBD converts penetration correction
                    # directly into velocity and sequentially applies every
                    # overlapping hand collider. Hand contact is therefore
                    # handled by the bounded projection below instead.
                    needs_coup=False,
                    sdf_cell_size=0.005,
                    sdf_min_res=16,
                    sdf_max_res=64,
                    gravity_compensation=1.0,
                ),
            )
            # The original black hand visuals and complete kinematics remain,
            # while the derived MJCF exposes just one palm box and five tip
            # spheres to Legacy PBD.
            self._robots[side] = hand

        garment_surface = _make_pbd_garment_surface(gs, self._asset_path)
        self._cloth = self._scene.add_entity(
            morph=gs.morphs.Mesh(
                file=str(self._asset_path),
                scale=garment_scale,
                pos=garment_position,
                euler=garment_euler,
            ),
            material=gs.materials.PBD.Cloth(
                # 0.18 kg/m^2 is a cotton-like areal mass, but this remains a
                # deliberately labelled live proxy rather than a calibrated
                # high-fidelity cotton constitutive model.
                rho=_env_float("HANDSON_GENESIS_E01_PBD_AREAL_DENSITY", 0.18),
                static_friction=_env_float(
                    "HANDSON_GENESIS_E01_PBD_STATIC_FRICTION", 0.70
                ),
                kinetic_friction=_env_float(
                    "HANDSON_GENESIS_E01_PBD_KINETIC_FRICTION", 0.50
                ),
                stretch_compliance=_env_float(
                    "HANDSON_GENESIS_E01_PBD_STRETCH_COMPLIANCE", 1.0e-7
                ),
                bending_compliance=_env_float(
                    "HANDSON_GENESIS_E01_PBD_BENDING_COMPLIANCE", 1.0e-5
                ),
                stretch_relaxation=_env_float(
                    "HANDSON_GENESIS_E01_PBD_STRETCH_RELAXATION", 0.8
                ),
                bending_relaxation=_env_float(
                    "HANDSON_GENESIS_E01_PBD_BENDING_RELAXATION", 0.4
                ),
                air_resistance=_env_float(
                    # The coarse double-layer proxy needs this damping during
                    # both settling and contact.  The near-cloth 10 mm slew
                    # guard keeps live collider velocity in its stable range.
                    "HANDSON_GENESIS_E01_PBD_SETTLE_AIR_RESISTANCE",
                    _PBD_SETTLE_AIR_RESISTANCE,
                ),
            ),
            surface=garment_surface,
        )
        self._pbd_visual_asset_path = self._asset_path.with_name(
            "TCLC_Top578_visual.obj"
        )
        self._pbd_separate_visual_mesh = _install_pbd_visual_mesh(
            gs,
            self._cloth,
            visual_path=self._pbd_visual_asset_path,
            scale=garment_scale,
            position=garment_position,
            euler=garment_euler,
            surface=garment_surface,
        )

        self._cameras["front"] = self._scene.add_camera(
            res=(_IMAGE_SHAPE[1], _IMAGE_SHAPE[0]),
            pos=_HEAD_CAMERA_EYE,
            lookat=_HEAD_CAMERA_LOOKAT,
            fov=80,
            GUI=False,
        )
        for side in ("left", "right"):
            self._cameras[f"wrist_{side}"] = self._scene.add_camera(
                res=(_IMAGE_SHAPE[1], _IMAGE_SHAPE[0]),
                pos=(0.0, 0.0, 1.0),
                lookat=(0.0, 0.0, 0.0),
                fov=60,
                GUI=False,
            )

        try:
            self._scene.build()
        except Exception as exc:
            raise RuntimeError(
                "Genesis GPU PBD could not build the official TCLC_Top578 "
                "Cotton_40s_Poplin E01 asset: "
                f"{self._asset_path}"
            ) from exc

        self._bind_robot_interfaces()
        self._initialize_robot_home()
        # Settle the official garment without the live hands close to its
        # sleeves.  At the canonical home pose some open-finger primitives are
        # only centimetres from the rest garment; including those contacts in
        # the first coarse PBD frames makes the remeshed double layer
        # nondeterministic.  The hands remain fully coupled once collection
        # starts and are placed at HOME immediately after the cache is made.
        self._set_kinematic_hand_target(
            "right", _RIGHT_BUILD_TCP, _WUJI_HAND_HOME
        )
        self._set_kinematic_hand_target(
            "left", _LEFT_BUILD_TCP, _WUJI_HAND_HOME
        )
        self._attach_wrist_cameras()
        self._pbd_rigid_geom_count = int(self._scene.sim.rigid_solver.n_geoms)
        self._pbd_coupled_rigid_geom_count = sum(
            int(bool(geom.needs_coup))
            for geom in self._scene.sim.rigid_solver.geoms
        )
        if self._pbd_coupled_rigid_geom_count != 1:
            raise RuntimeError(
                "E01 PBD must expose exactly one Legacy-coupled rigid "
                "geometry (the table); got "
                f"{self._pbd_coupled_rigid_geom_count}"
            )
        for side in ("right", "left"):
            palm_geoms = tuple(
                self._robots[side].get_link(f"{side}_palm_link").geoms
            )
            tip_geoms = tuple(
                tuple(
                    self._robots[side]
                    .get_link(f"{side}_finger{finger}_link4")
                    .geoms
                )
                for finger in range(1, 6)
            )
            if len(palm_geoms) != 1 or any(
                len(geoms) != 1 for geoms in tip_geoms
            ):
                raise RuntimeError(
                    f"Incomplete {side} PBD source contact geometry: "
                    f"palm={len(palm_geoms)}, tips="
                    f"{tuple(len(geoms) for geoms in tip_geoms)}"
                )
            self._pbd_source_contact_geoms[side] = {
                "palm": palm_geoms[0],
                "tips": tuple(geoms[0] for geoms in tip_geoms),
            }
        self._pbd_pinch_links = {
            side: (
                self._robots[side].get_link(f"{side}_finger1_link4"),
                tuple(
                    self._robots[side].get_link(
                        f"{side}_finger{finger}_link4"
                    )
                    for finger in range(2, 6)
                ),
            )
            for side in ("right", "left")
        }

        self._pre_settle_and_cache()
        self._activate_live_air_resistance()
        self._set_kinematic_hand_target(
            "right", _RIGHT_HOME_TCP, _WUJI_HAND_HOME
        )
        self._set_kinematic_hand_target(
            "left", _LEFT_HOME_TCP, _WUJI_HAND_HOME
        )
        self._refresh_state(compute_metrics=False, reset_nodal_velocity=True)
        self._right_commanded_tcp_pose = self._right_tcp_pose.copy()
        self._left_commanded_tcp_pose = self._left_tcp_pose.copy()
        self._base_nodes = self._nodes.copy()
        self._reset_nodes = self._nodes.copy()
        self._configure_tracking(self._base_nodes)
        self._last_metrics = self._garment_metrics()
        self._pbd_bounded_hand_contact_enabled = True

    def _activate_live_air_resistance(self) -> None:
        """Apply the validated live drag coefficient after pre-settle."""
        assert self._cloth is not None and self._scene is not None
        self._cloth.material.air_resistance = self._pbd_live_air_resistance
        self._cloth._kernel_add_particles_air_resistance_to_solver(
            f=self._scene.sim.cur_substep_local
        )

    def _pre_settle_and_cache(self) -> None:
        assert self._scene is not None and self._cloth is not None
        positions: np.ndarray | None = None
        velocities: np.ndarray | None = None
        bounds_min: np.ndarray | None = None
        bounds_max: np.ndarray | None = None
        failure = ""
        for attempt in range(1, self._pbd_settle_max_attempts + 1):
            self._pbd_settle_attempts = attempt
            if attempt > 1:
                # GPU self-collision reductions are not bit deterministic.
                # Reset the unchanged downloaded rest mesh and retry instead
                # of accepting a clamped boundary state as a valid garment.
                self._scene.reset()
                self._initialize_robot_home()
                self._set_kinematic_hand_target(
                    "right", _RIGHT_BUILD_TCP, _WUJI_HAND_HOME
                )
                self._set_kinematic_hand_target(
                    "left", _LEFT_BUILD_TCP, _WUJI_HAND_HOME
                )
            for _ in range(self._pbd_settle_steps_requested):
                self._scene.step(update_visualizer=False)
                self._pbd_settle_steps_completed += 1

            positions = _to_numpy(self._cloth.get_particles_pos()).reshape(-1, 3)
            velocities = _to_numpy(self._cloth.get_particles_vel()).reshape(-1, 3)
            if not (
                np.all(np.isfinite(positions))
                and np.all(np.isfinite(velocities))
            ):
                failure = "produced non-finite cloth state"
                continue
            bounds_min = np.min(positions, axis=0)
            bounds_max = np.max(positions, axis=0)
            at_domain_boundary = bool(
                np.any(bounds_min <= _PBD_DOMAIN_LOWER + 1.0e-5)
                or np.any(bounds_max >= _PBD_DOMAIN_UPPER - 1.0e-5)
            )
            if at_domain_boundary:
                failure = "hit its domain boundary"
                continue
            break
        else:
            raise RuntimeError(
                "Genesis PBD pre-settle remained unstable after "
                f"{self._pbd_settle_max_attempts} attempts: {failure}"
            )

        assert positions is not None and velocities is not None
        assert bounds_min is not None and bounds_max is not None

        # A reset should start at rest, not replay the final settling velocity.
        zero_velocities = np.zeros_like(velocities)
        self._cloth.set_particles_vel(zero_velocities)
        self._pbd_settled_particle_positions = positions.astype(
            np.float32, copy=True
        )
        self._pbd_settled_particle_velocities = zero_velocities.astype(
            np.float32, copy=True
        )
        self._pbd_settled_z_span = float(bounds_max[2] - bounds_min[2])

    def _pinch_state(
        self, side: str
    ) -> tuple[float, np.ndarray, Any] | None:
        pair = self._pbd_pinch_links.get(side)
        if pair is None:
            return None
        thumb, opposing = pair
        thumb_pos = _to_numpy(thumb.get_pos()).reshape(-1, 3)[0].astype(
            np.float64
        )
        other_positions = np.asarray(
            [
                _to_numpy(link.get_pos()).reshape(-1, 3)[0]
                for link in opposing
            ],
            dtype=np.float64,
        )
        distances = np.linalg.norm(other_positions - thumb_pos, axis=1)
        closest = int(np.argmin(distances))
        midpoint = 0.5 * (thumb_pos + other_positions[closest])
        return float(distances[closest]), midpoint, thumb

    def _apply_bounded_hand_contact(self) -> None:
        """Project cloth out of the nearest hand shape without adding impulses.

        Each particle receives at most one correction per control frame, even
        when opposing fingertips overlap it. The correction is capped and its
        inward velocity is removed, so penetration cannot be amplified into
        the unbounded velocity produced by Legacy rigid-PBD coupling.
        """
        if (
            not self._pbd_bounded_hand_contact_enabled
            or self._cloth is None
            or not self._pbd_source_contact_geoms
        ):
            self._pbd_hand_contact_count = 0
            self._pbd_hand_contact_max_correction = 0.0
            return
        nodes = np.asarray(self._nodes, dtype=np.float64).reshape(-1, 3)
        if not nodes.size or not np.all(np.isfinite(nodes)):
            return
        velocities = np.asarray(self._velocities, dtype=np.float64).reshape(
            -1, 3
        )
        best_depth = np.zeros(nodes.shape[0], dtype=np.float64)
        best_normal = np.zeros_like(nodes)
        held = {
            int(particle)
            for particle in self._pbd_attachment_particle.values()
            if particle is not None
        }

        def consider(depth: np.ndarray, normals: np.ndarray) -> None:
            if held:
                depth = np.asarray(depth, dtype=np.float64).copy()
                depth[np.fromiter(held, dtype=np.int64)] = 0.0
            deeper = depth > best_depth
            best_depth[deeper] = depth[deeper]
            best_normal[deeper] = normals[deeper]

        margin = 0.5 * self._pbd_particle_size
        for side in ("right", "left"):
            sources = self._pbd_source_contact_geoms[side]
            palm_pos = _to_numpy(
                sources["palm"].get_pos(relative=False)
            ).reshape(-1, 3)[0]
            palm_quat = _to_numpy(
                sources["palm"].get_quat(relative=False)
            ).reshape(-1, 4)[0]
            consider(
                *_box_contact_penetration(
                    nodes,
                    palm_pos,
                    palm_quat,
                    half_extents=_PBD_PALM_HALF_EXTENTS,
                    margin=margin,
                )
            )
            for source in sources["tips"]:
                tip_pos = _to_numpy(
                    source.get_pos(relative=False)
                ).reshape(-1, 3)[0]
                consider(
                    *_sphere_contact_penetration(
                        nodes,
                        tip_pos,
                        radius=_PBD_FINGERTIP_RADIUS,
                        margin=margin,
                    )
                )

        collided = best_depth > 0.0
        self._pbd_hand_contact_count = int(np.count_nonzero(collided))
        if not np.any(collided):
            self._pbd_hand_contact_max_correction = 0.0
            return
        # A kinematic fingertip pressing cloth already supported by the table
        # must displace it sideways, not feed it down into the table's own
        # hard Legacy projection. That opposing pair was another source of
        # unbounded solver velocity.
        near_table = self._nodes[:, 2] <= (
            _TABLE_TOP_Z + margin + _PBD_HAND_CONTACT_MAX_CORRECTION
        )
        downward = collided & near_table & (best_normal[:, 2] < 0.0)
        if np.any(downward):
            horizontal = best_normal[downward].copy()
            horizontal[:, 2] = 0.0
            horizontal_norm = np.linalg.norm(horizontal, axis=1)
            degenerate = horizontal_norm <= 1.0e-12
            horizontal[degenerate] = np.asarray([1.0, 0.0, 0.0])
            horizontal_norm[degenerate] = 1.0
            best_normal[downward] = horizontal / horizontal_norm[:, None]
        correction_length = np.minimum(
            best_depth,
            _PBD_HAND_CONTACT_MAX_CORRECTION,
        )
        correction = best_normal * correction_length[:, None]
        nodes[collided] += correction[collided]
        velocities[collided] = 0.0
        self._nodes = nodes
        self._velocities = velocities
        self._pbd_hand_contact_max_correction = float(
            np.max(correction_length[collided])
        )
        self._cloth.set_particles_pos(nodes.astype(np.float32))
        self._cloth.set_particles_vel(velocities.astype(np.float32))

    def _release_attachment(self, side: str) -> None:
        particle = self._pbd_attachment_particle.get(side)
        if particle is not None and self._cloth is not None:
            particle_indices = np.asarray([particle], dtype=np.int32)
            self._cloth.release_particle(
                particles_idx_local=particle_indices
            )
            # The legacy coupler drives attachments with a corrective
            # velocity one frame late.  Do not feed that residual impulse into
            # the free cloth on the release frame.
            self._cloth.set_particles_vel(
                0.0,
                particles_idx_local=particle_indices,
            )
        self._pbd_attachment_particle[side] = None
        self._pbd_attachment_link_name[side] = ""

    def _release_all_attachments(self) -> None:
        for side in ("right", "left"):
            self._release_attachment(side)
            self._pbd_pinch_distance[side] = float("inf")
            self._pbd_nearest_particle_distance[side] = float("inf")

    def _clear_solver_attachments(self) -> None:
        """Clear the coupler map defensively before a whole-scene reset."""
        if self._cloth is not None:
            # LegacyCoupler.reset() does not clear particle_attach_info.  A
            # full release also protects resets after an interrupted attach
            # call where the Python-side particle bookkeeping was not updated.
            self._cloth.release_particle()
        for side in ("right", "left"):
            self._pbd_attachment_particle[side] = None
            self._pbd_attachment_link_name[side] = ""
            self._pbd_pinch_distance[side] = float("inf")
            self._pbd_nearest_particle_distance[side] = float("inf")

    def _update_pbd_attachments(self) -> None:
        if self._cloth is None or not self._pbd_pinch_links:
            return
        nodes = np.asarray(self._nodes, dtype=np.float64).reshape(-1, 3)
        for side in ("right", "left"):
            state = self._pinch_state(side)
            if state is None:
                continue
            pinch_distance, midpoint, thumb_link = state
            self._pbd_pinch_distance[side] = pinch_distance

            particle = self._pbd_attachment_particle[side]
            if particle is not None:
                self._pbd_nearest_particle_distance[side] = float(
                    np.linalg.norm(nodes[particle] - midpoint)
                )
                if pinch_distance >= self._pbd_pinch_release_distance:
                    self._release_attachment(side)
                continue

            distances = np.linalg.norm(nodes - midpoint, axis=1)
            nearest = int(np.argmin(distances))
            nearest_distance = float(distances[nearest])
            self._pbd_nearest_particle_distance[side] = nearest_distance
            already_held = {
                value
                for other_side, value in self._pbd_attachment_particle.items()
                if other_side != side and value is not None
            }
            if (
                pinch_distance <= self._pbd_pinch_close_distance
                and nearest_distance <= self._pbd_attach_distance
                and nearest not in already_held
            ):
                self._cloth.fix_particles_to_link(
                    int(thumb_link.idx),
                    particles_idx_local=np.asarray([nearest], dtype=np.int32),
                )
                self._pbd_attachment_particle[side] = nearest
                self._pbd_attachment_link_name[side] = str(thumb_link.name)
                self._pbd_attachment_events += 1

    def _path_near_garment(
        self, start: np.ndarray, end: np.ndarray
    ) -> bool:
        """Return whether a palm path enters the cloth contact neighborhood."""
        nodes = np.asarray(self._nodes, dtype=np.float64).reshape(-1, 3)
        if nodes.size == 0 or not np.all(np.isfinite(nodes)):
            return True
        start = np.asarray(start, dtype=np.float64).reshape(3)
        end = np.asarray(end, dtype=np.float64).reshape(3)
        delta = end - start
        length_squared = float(np.dot(delta, delta))
        if length_squared <= 1.0e-12:
            distance = np.linalg.norm(nodes - start, axis=1)
        else:
            alpha = np.clip(
                ((nodes - start) @ delta) / length_squared,
                0.0,
                1.0,
            )
            closest = start + alpha[:, None] * delta
            distance = np.linalg.norm(nodes - closest, axis=1)
        return bool(np.min(distance) <= self._pbd_contact_slew_distance)

    def _apply_action(self, action: Any) -> None:
        # Retain Plan1's 60 mm / 10 degree guards away from the garment. A
        # path entering the no-CCD contact neighborhood is slewed more tightly;
        # the recorded protected pose remains the actual simulated pose.
        right, left = self._split_action(action)
        requested = (right[:3], left[:3])
        current = (
            self._right_commanded_tcp_pose[:3],
            self._left_commanded_tcp_pose[:3],
        )
        near_garment = any(
            self._path_near_garment(
                start,
                np.where(np.isfinite(target), target, start),
            )
            for start, target in zip(current, requested, strict=True)
        )
        self._pbd_active_translation_step = (
            self._pbd_collision_safe_translation_step
            if near_garment
            else self._pbd_plan1_translation_step
        )
        active_rotation_step = (
            self._pbd_collision_safe_rotation_step
            if near_garment
            else self._pbd_plan1_rotation_step
        )
        self._max_tcp_translation_step = self._pbd_active_translation_step
        self._max_tcp_rotation_step = active_rotation_step
        try:
            # The base implementation remains the single source of truth for
            # Plan1 reach, table, rotation-rate and hand-joint guards.
            super()._apply_action(action)
        finally:
            self._max_tcp_translation_step = self._pbd_plan1_translation_step
            self._max_tcp_rotation_step = self._pbd_plan1_rotation_step
        self._update_pbd_attachments()

    def _restore_settled_particle_state(self) -> None:
        if (
            self._cloth is None
            or self._pbd_settled_particle_positions is None
            or self._pbd_settled_particle_velocities is None
        ):
            return
        self._cloth.set_particles_pos(self._pbd_settled_particle_positions)
        self._cloth.set_particles_vel(self._pbd_settled_particle_velocities)

    def reset(self, seed: int | None = None, **_: Any):
        if self._scene is None:
            return super().reset(seed=seed)

        gym.Env.reset(self, seed=seed)
        if seed is not None:
            self._random.seed(seed)
        self.env_step = 0
        self._success_counter = 0
        self._last_metrics = {}
        self.delta_h[:] = 0.0
        self._initial_state[:] = 0.0
        self._right_tcp_pose = _RIGHT_HOME_TCP.copy()
        self._left_tcp_pose = _LEFT_HOME_TCP.copy()
        self._right_commanded_tcp_pose = _RIGHT_HOME_TCP.copy()
        self._left_commanded_tcp_pose = _LEFT_HOME_TCP.copy()
        self._right_hand = _WUJI_HAND_HOME.copy()
        self._left_hand = _WUJI_HAND_HOME.copy()
        self._arm_reach_guard_active = {"right": False, "left": False}
        self._arm_reach_guard_distance = {"right": 0.0, "left": 0.0}
        self._hand_table_guard_active = {"right": False, "left": False}
        self._hand_table_guard_distance = {"right": 0.0, "left": 0.0}
        self._constraint_hold_active = {"right": False, "left": False}
        self._hand_minimum_collision_z = {
            "right": self._hand_collision_guards["right"].minimum_collision_z(
                self._right_commanded_tcp_pose, self._right_hand
            ),
            "left": self._hand_collision_guards["left"].minimum_collision_z(
                self._left_commanded_tcp_pose, self._left_hand
            ),
        }
        if self.randomize:
            warning = (
                "PBD live proxy reset uses its cached official-TCLC_Top578 "
                "pre-settled "
                "state; per-episode garment teleport randomization is disabled"
            )
            if warning not in self._genesis_warnings:
                self._genesis_warnings.append(warning)

        self._clear_solver_attachments()
        self._pbd_attachment_events = 0
        self._pbd_bounded_hand_contact_enabled = False
        self._pbd_hand_contact_count = 0
        self._pbd_hand_contact_max_correction = 0.0
        self._pbd_active_translation_step = self._pbd_plan1_translation_step
        self._pbd_domain_boundary_hit = False
        self._pbd_nonfinite_state = False
        self._scene.reset()
        self._initialize_robot_home()
        # Synchronize the legacy rigid-PBD coupler's previous-link transform
        # before restoring the cached cloth.  Without this priming substep the
        # reset-time BUILD -> HOME teleport is interpreted as a high-speed
        # swept collider on the first live frame.
        self._scene.step(update_visualizer=False)
        self._restore_settled_particle_state()
        self._set_kinematic_hand_target(
            "right", self._right_commanded_tcp_pose, self._right_hand
        )
        self._set_kinematic_hand_target(
            "left", self._left_commanded_tcp_pose, self._left_hand
        )
        self._refresh_state(reset_nodal_velocity=True)
        self._right_commanded_tcp_pose = self._right_tcp_pose.copy()
        self._left_commanded_tcp_pose = self._left_tcp_pose.copy()
        self._reset_nodes = self._nodes.copy()
        self._last_metrics = self._garment_metrics()
        self._pbd_bounded_hand_contact_enabled = True
        info = {"succeed": False, "manual_reset": False}
        info.update(self._task_info())
        return self._compute_observation(), info

    def _refresh_state(
        self,
        *,
        compute_metrics: bool = True,
        elapsed_physics_dt: float | None = None,
        reset_nodal_velocity: bool = False,
    ) -> None:
        del elapsed_physics_dt
        if self._cloth is not None:
            nodes = _to_numpy(self._cloth.get_particles_pos()).reshape(-1, 3)
            velocities = _to_numpy(self._cloth.get_particles_vel()).reshape(-1, 3)
            self._nodes = nodes.astype(np.float64)
            self._velocities = (
                np.zeros_like(self._nodes)
                if reset_nodal_velocity
                else velocities.astype(np.float64)
            )
            self._pbd_nonfinite_state = not bool(
                np.all(np.isfinite(self._nodes))
                and np.all(np.isfinite(self._velocities))
            )
            if not self._pbd_nonfinite_state:
                self._apply_bounded_hand_contact()
                bounds_min = np.min(self._nodes, axis=0)
                bounds_max = np.max(self._nodes, axis=0)
                self._pbd_domain_boundary_hit = bool(
                    np.any(bounds_min <= _PBD_DOMAIN_LOWER + 1.0e-5)
                    or np.any(bounds_max >= _PBD_DOMAIN_UPPER - 1.0e-5)
                )
        for side in ("right", "left"):
            link = self._ee_links.get(side)
            if link is None:
                continue
            pos = _to_numpy(link.get_pos()).reshape(-1, 3)[0]
            quat = _to_numpy(link.get_quat()).reshape(-1, 4)[0]
            pose = np.concatenate([pos, quat]).astype(np.float64)
            if side == "right":
                self._right_tcp_pose = pose
            else:
                self._left_tcp_pose = pose
        if compute_metrics:
            self._last_metrics = self._garment_metrics()

    def _garment_metrics(self) -> dict[str, Any]:
        metrics = super()._garment_metrics()
        if any(
            particle is not None
            for particle in self._pbd_attachment_particle.values()
        ):
            metrics["hands_released"] = False
            metrics["success_now"] = False
        return metrics

    def _compute_failure(self) -> bool:
        # Domain clamps hide a preceding PBD instability by returning finite
        # positions exactly on the configured boundary.  Fail the episode
        # immediately instead of recording a visually corrupted trajectory.
        if self._pbd_nonfinite_state or self._pbd_domain_boundary_hit:
            return True
        return super()._compute_failure()

    def restore_recorded_initial_state(
        self, state_parts: Mapping[str, np.ndarray]
    ) -> dict[str, Any]:
        initial = np.asarray(
            state_parts.get("garment_initial_state", np.zeros(6)),
            dtype=np.float64,
        ).ravel()
        delta = np.asarray(
            state_parts.get("table_delta_height", np.zeros(1)),
            dtype=np.float64,
        ).ravel()
        if np.any(np.abs(initial[:6]) > 1e-8) or np.any(np.abs(delta[:1]) > 1e-8):
            raise ValueError(
                "Genesis PBD live proxy only restores its cached pre-settled "
                "official-TCLC_Top578 state"
            )
        observation, _info = self.reset()
        return observation

    @staticmethod
    def _finite_or_minus_one(value: float) -> float:
        return float(value) if np.isfinite(value) else -1.0

    def _task_info(self) -> dict[str, Any]:
        info = super()._task_info()
        info.update(
            {
                "cloth_solver": "genesis_gpu_pbd_live_proxy",
                "pbd_live_proxy": True,
                "pbd_particle_size_m": float(self._pbd_particle_size),
                "pbd_live_air_resistance": float(
                    self._pbd_live_air_resistance
                ),
                "pbd_collision_safe_translation_step_m": float(
                    self._pbd_collision_safe_translation_step
                ),
                "pbd_collision_safe_rotation_step_deg": float(
                    math.degrees(self._pbd_collision_safe_rotation_step)
                ),
                "pbd_active_translation_step_m": float(
                    self._pbd_active_translation_step
                ),
                "pbd_contact_slew_distance_m": float(
                    self._pbd_contact_slew_distance
                ),
                "plan1_translation_guard_step_m": float(
                    self._pbd_plan1_translation_step
                ),
                "pbd_genesis_seed": int(self._pbd_genesis_seed),
                "pbd_visual_skinning_neighbors": 2,
                "pbd_domain_boundary_hit": bool(
                    self._pbd_domain_boundary_hit
                ),
                "pbd_nonfinite_state": bool(self._pbd_nonfinite_state),
                "pbd_settle_steps": int(self._pbd_settle_steps_completed),
                "pbd_settle_attempts": int(self._pbd_settle_attempts),
                "pbd_settled_z_span": self._finite_or_minus_one(
                    self._pbd_settled_z_span
                ),
                "pbd_full_hand_primitive_collision": False,
                "pbd_hand_contact_proxy": (
                    "bounded_deepest_contact_projection"
                ),
                "pbd_hand_contact_max_correction_m": float(
                    _PBD_HAND_CONTACT_MAX_CORRECTION
                ),
                "pbd_hand_contact_count": int(
                    self._pbd_hand_contact_count
                ),
                "pbd_hand_contact_applied_correction_m": float(
                    self._pbd_hand_contact_max_correction
                ),
                "garment_material": "Cotton_40s_Poplin",
                "pbd_separate_visual_mesh": bool(
                    self._pbd_separate_visual_mesh
                ),
                "pbd_visual_asset": str(
                    self._pbd_visual_asset_path or ""
                ),
                "pbd_rigid_geom_count": int(self._pbd_rigid_geom_count),
                "pbd_coupled_rigid_geom_count": int(
                    self._pbd_coupled_rigid_geom_count
                ),
                "pbd_attachment_active_right": (
                    self._pbd_attachment_particle["right"] is not None
                ),
                "pbd_attachment_active_left": (
                    self._pbd_attachment_particle["left"] is not None
                ),
                "pbd_attachment_particle_right": int(
                    self._pbd_attachment_particle["right"]
                    if self._pbd_attachment_particle["right"] is not None
                    else -1
                ),
                "pbd_attachment_particle_left": int(
                    self._pbd_attachment_particle["left"]
                    if self._pbd_attachment_particle["left"] is not None
                    else -1
                ),
                "pbd_attachment_link_right": self._pbd_attachment_link_name[
                    "right"
                ],
                "pbd_attachment_link_left": self._pbd_attachment_link_name[
                    "left"
                ],
                "pbd_pinch_distance_right": self._finite_or_minus_one(
                    self._pbd_pinch_distance["right"]
                ),
                "pbd_pinch_distance_left": self._finite_or_minus_one(
                    self._pbd_pinch_distance["left"]
                ),
                "pbd_nearest_particle_distance_right": self._finite_or_minus_one(
                    self._pbd_nearest_particle_distance["right"]
                ),
                "pbd_nearest_particle_distance_left": self._finite_or_minus_one(
                    self._pbd_nearest_particle_distance["left"]
                ),
                "pbd_attachment_events": int(self._pbd_attachment_events),
            }
        )
        return info

    def close(self) -> None:
        if self._scene is not None:
            self._release_all_attachments()
        super().close()


__all__ = ["HandsOnE01GenesisPBDLiveEnv"]
