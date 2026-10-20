"""Research-grade surface-deformable runtime for B320-BC025."""

from __future__ import annotations

import math
from typing import Any

from gymnasium import spaces
import numpy as np

from .b320_bimanual_scene import (
    B320_CONTROL_HZ,
    B320_PHYSICS_HZ,
    B320_TABLE_PATH,
    B320_TOWEL_TASK_NAME,
)
from .b320_towel_scene import (
    BC025_TOWEL_BEND_STIFFNESS,
    BC025_TOWEL_COLUMNS,
    BC025_TOWEL_CONTACT_OFFSET,
    BC025_TOWEL_DYNAMIC_FRICTION,
    BC025_TOWEL_ELASTICITY_DAMPING,
    BC025_TOWEL_MAX_DEPENETRATION_VELOCITY,
    BC025_TOWEL_NODE_COUNT,
    BC025_TOWEL_REST_OFFSET,
    BC025_TOWEL_ROWS,
    BC025_TOWEL_SHEAR_STIFFNESS,
    BC025_TOWEL_STRETCH_STIFFNESS,
    BC025_TOWEL_SURFACE_ROOT_PATH,
    BC025_TOWEL_SURFACE_THICKNESS,
    BC025_TOWEL_TARGET_END_DISTANCE_M,
    BC025_TOWEL_TRIANGLE_COUNT,
    BC025_TOWEL_TRANSLATION_BOUNDS_M,
    B320_TOWEL_PLACEMENT,
    B320_TOWEL_TARGET_CENTER_XY,
    B320_TOWEL_TARGET_SIZE_XY,
    bc025_towel_surface_topology,
    b320_towel_surface_metadata,
    spawn_b320_towel_surface,
)
from .b320_bc025_assets import (
    BC025_COLLISION_SOURCE_TITLE,
    BC025_MATERIAL_TITLE,
    BC025_PILLOW_COLLISION_CALIBRATION,
    bc025_static_asset_manifest,
)
from .e01_detached_hand_env import HandsOnE01DetachedHandsIsaacLabEnv
from .e_collection_scene import E01SurfacePhysicsTuning
from .recording_cameras import WRIST_RECORDING_CAMERA_OUTPUT_SIZE
from .recording_origin import DEFAULT_CONTACT_PENETRATION_LIMIT


_TOWEL_CORNER_INDICES = np.asarray(
    [
        0,
        BC025_TOWEL_COLUMNS - 1,
        (BC025_TOWEL_ROWS - 1) * BC025_TOWEL_COLUMNS,
        BC025_TOWEL_NODE_COUNT - 1,
    ],
    dtype=np.int64,
)
_TOWEL_RELEASE_DISTANCE_M = 0.065
_TOWEL_STABLE_SPEED_MPS = 0.04
# Keep recorder/model inputs finite while preserving fail-closed semantics for
# an invalid surface or an as-yet-unreleased corner.  These sentinels are
# above the corresponding success thresholds and the predicate flags remain
# authoritative.
_UNAVAILABLE_RELEASE_DELTA_S = 2.0
_INVALID_SURFACE_STRETCH = 1.0
_INVALID_SURFACE_SPEED_MPS = 1.0
_TOWEL_CORNER_CONTACT_DISTANCE_M = 0.095
_TOWEL_TARGET_CORNER_DISTANCE_TOLERANCE = 0.10

# Isaac Sim 5.1 exposes nodal positions/velocities for a surface-deformable
# body, but its public rigid-contact tensor view does not expose a deformable
# contact manifold.  BC025 therefore uses a deliberately conservative,
# readback-only physical-contact witness: after a proxy approaches the sheet,
# a localized node displacement or velocity response must exceed the global
# surface motion background before the recovery assist can bind.  The assist
# itself is still gated below and can never create this first witness.
_BC025_CONTACT_RESPONSE_RADIUS_M = 0.055
_BC025_CONTACT_RESPONSE_MIN_LOCAL_NODES = 4
_BC025_CONTACT_RESPONSE_MIN_DISPLACEMENT_M = 0.00035
_BC025_CONTACT_RESPONSE_MIN_DISPLACEMENT_EXCESS_M = 0.00020
_BC025_CONTACT_RESPONSE_MIN_SPEED_MPS = 0.025
_BC025_CONTACT_RESPONSE_MIN_SPEED_EXCESS_MPS = 0.010
_BC025_CONTACT_RESPONSE_MIN_PROXY_MOTION_M = 0.00015
_BC025_CONTACT_RESPONSE_MIN_APPROACH_M = 0.00015
_BC025_CONTACT_RESPONSE_CONTACT_MARGIN_M = 0.001
# A response must be observed in the assigned corner neighborhood before the
# assist is armed.  This is tighter than the broad event-distance tolerance so
# a hand cannot touch a different corner, move away, and then obtain an
# assigned-corner latch from proximity alone.
_BC025_ASSIGNED_RESPONSE_DISTANCE_M = 0.075

# The upper edge is row ``BC025_TOWEL_ROWS - 1`` in the task-owned grid.
# These are the only two regions that can satisfy the assigned left/right
# contact requirement; nearest-corner inference is intentionally forbidden.
_BC025_ASSIGNED_CORNER_INDICES = {
    "left": int(_TOWEL_CORNER_INDICES[2]),
    "right": int(_TOWEL_CORNER_INDICES[3]),
}


def _interface_towel_nodes() -> np.ndarray:
    points, _triangles = bc025_towel_surface_topology()
    nodes = np.asarray(points, dtype=np.float64)
    nodes += np.asarray(B320_TOWEL_PLACEMENT, dtype=np.float64)
    return nodes


def _bc025_origin_surface_geometry(nodes: Any, table_bounds: Any) -> dict[str, Any]:
    """Check native surface readback against the actual static tabletop.

    This is geometric support/penetration evidence, not a deformable contact
    manifold or a force measurement. Keep the rigid contact-report gate and
    the shared nodal velocity/drift checks in addition to this check.
    """
    points = np.asarray(nodes, dtype=np.float64)
    bounds = np.asarray(table_bounds, dtype=np.float64)
    finite = bool(
        points.shape == (BC025_TOWEL_NODE_COUNT, 3)
        and bounds.shape == (2, 3)
        and np.all(np.isfinite(points))
        and np.all(np.isfinite(bounds))
        and np.all(bounds[1] > bounds[0])
    )
    if not finite:
        return {"ready": False, "failure_reasons": ["invalid_surface_or_table_readback"]}
    rest_clearance = points[:, 2] - bounds[1, 2] - BC025_TOWEL_REST_OFFSET
    penetration = float(max(0.0, -np.min(rest_clearance)))
    inside = bool(
        np.all(points[:, :2] >= bounds[0, :2])
        and np.all(points[:, :2] <= bounds[1, :2])
    )
    supported_corners = int(
        np.count_nonzero(
            np.abs(rest_clearance[_TOWEL_CORNER_INDICES])
            <= BC025_TOWEL_CONTACT_OFFSET
        )
    )
    failures = []
    if not inside:
        failures.append("surface_outside_table_footprint")
    if penetration > DEFAULT_CONTACT_PENETRATION_LIMIT:
        failures.append("surface_table_penetration")
    if supported_corners < 3:
        failures.append("surface_not_supported_at_reset")
    return {
        "ready": not failures,
        "failure_reasons": failures,
        "surface_node_count": int(len(points)),
        "table_bounds_m": bounds.tolist(),
        "maximum_surface_rest_penetration_m": penetration,
        "penetration_limit_m": DEFAULT_CONTACT_PENETRATION_LIMIT,
        "supported_corner_count": supported_corners,
    }


def _detect_bc025_surface_contact_response(
    nodes: np.ndarray,
    previous_nodes: np.ndarray,
    velocities: np.ndarray,
    proxy_pose: np.ndarray,
    previous_proxy_pose: np.ndarray,
    *,
    proxy_radius: float,
    hand_closed: bool,
) -> tuple[bool, dict[str, Any]]:
    """Detect a localized post-PhysX surface response near one proxy.

    This is intentionally a pure numerical witness rather than a geometric
    proximity predicate.  It compares the local node motion to the rest of
    the sheet and requires the proxy to be in the collision band.  Keeping it
    separate makes the no-Isaac false-positive tests deterministic and makes
    the source of BC025's contact truth explicit.
    """

    points = np.asarray(nodes, dtype=np.float64)
    previous = np.asarray(previous_nodes, dtype=np.float64)
    speed_values = np.asarray(velocities, dtype=np.float64)
    proxy = np.asarray(proxy_pose, dtype=np.float64).reshape(-1)
    previous_proxy = np.asarray(previous_proxy_pose, dtype=np.float64).reshape(-1)
    diagnostics: dict[str, Any] = {
        "response": False,
        "reason": "unknown",
        "local_node_count": 0,
        "local_max_displacement_m": 0.0,
        "local_max_speed_mps": 0.0,
        "background_displacement_m": 0.0,
        "background_speed_mps": 0.0,
        "displacement_excess_m": 0.0,
        "speed_excess_mps": 0.0,
        "proxy_motion_m": 0.0,
        "approach_m": 0.0,
        "proxy_bottom_m": 0.0,
        "local_contact_band_top_m": 0.0,
        "contact_band": False,
    }
    if not hand_closed:
        diagnostics["reason"] = "hand_open"
        return False, diagnostics
    if (
        points.ndim != 2
        or points.shape[1:] != (3,)
        or previous.shape != points.shape
        or speed_values.shape != points.shape
        or points.shape[0] < _BC025_CONTACT_RESPONSE_MIN_LOCAL_NODES
        or not np.all(np.isfinite(points))
        or not np.all(np.isfinite(previous))
        or not np.all(np.isfinite(speed_values))
        or proxy.size < 3
        or previous_proxy.size < 3
        or not np.all(np.isfinite(proxy[:3]))
        or not np.all(np.isfinite(previous_proxy[:3]))
        or not math.isfinite(float(proxy_radius))
        or float(proxy_radius) <= 0.0
    ):
        diagnostics["reason"] = "invalid_readback"
        return False, diagnostics

    distance = np.linalg.norm(points[:, :2] - proxy[:2], axis=1)
    local_indices = np.flatnonzero(distance <= _BC025_CONTACT_RESPONSE_RADIUS_M)
    diagnostics["local_node_count"] = int(local_indices.size)
    if local_indices.size < _BC025_CONTACT_RESPONSE_MIN_LOCAL_NODES:
        diagnostics["reason"] = "no_local_surface_nodes"
        return False, diagnostics

    node_displacement = np.linalg.norm(points - previous, axis=1)
    node_speed = np.linalg.norm(speed_values, axis=1)
    background_indices = np.ones(points.shape[0], dtype=bool)
    background_indices[local_indices] = False
    if int(np.count_nonzero(background_indices)) < _BC025_CONTACT_RESPONSE_MIN_LOCAL_NODES:
        background_indices[:] = True
    background_displacement = float(
        np.percentile(node_displacement[background_indices], 75.0)
    )
    background_speed = float(np.percentile(node_speed[background_indices], 75.0))
    local_displacement = node_displacement[local_indices]
    local_speed = node_speed[local_indices]
    local_max_displacement = float(np.max(local_displacement))
    local_max_speed = float(np.max(local_speed))
    displacement_excess = local_max_displacement - background_displacement
    speed_excess = local_max_speed - background_speed
    proxy_bottom = float(proxy[2] - float(proxy_radius))
    local_contact_band_top = float(
        np.percentile(points[local_indices, 2], 75.0)
        + BC025_TOWEL_CONTACT_OFFSET
        + _BC025_CONTACT_RESPONSE_CONTACT_MARGIN_M
    )
    contact_band = proxy_bottom <= local_contact_band_top
    proxy_motion = float(np.linalg.norm(proxy[:3] - previous_proxy[:3]))
    previous_distance = np.linalg.norm(
        previous[:, :2] - previous_proxy[:2], axis=1
    )
    approach = float(
        np.min(previous_distance) - np.min(distance)
    )
    displacement_hit = bool(
        local_max_displacement >= _BC025_CONTACT_RESPONSE_MIN_DISPLACEMENT_M
        and displacement_excess >= _BC025_CONTACT_RESPONSE_MIN_DISPLACEMENT_EXCESS_M
    )
    speed_hit = bool(
        local_max_speed >= _BC025_CONTACT_RESPONSE_MIN_SPEED_MPS
        and speed_excess >= _BC025_CONTACT_RESPONSE_MIN_SPEED_EXCESS_MPS
    )
    proxy_moved_into_band = bool(
        proxy_motion >= _BC025_CONTACT_RESPONSE_MIN_PROXY_MOTION_M
        or approach >= _BC025_CONTACT_RESPONSE_MIN_APPROACH_M
    )
    # A strong localized response is sufficient even if a controller holds a
    # proxy in place; otherwise require motion/approach so reset settling or a
    # stationary table contact cannot arm the assist.
    strong_response = bool(
        local_max_displacement >= 2.0 * _BC025_CONTACT_RESPONSE_MIN_DISPLACEMENT_M
        or local_max_speed >= 2.0 * _BC025_CONTACT_RESPONSE_MIN_SPEED_MPS
    )
    response = bool(
        contact_band
        and (displacement_hit or speed_hit)
        and (proxy_moved_into_band or strong_response)
    )
    diagnostics.update(
        {
            "response": response,
            "reason": "physical_surface_response"
            if response
            else "awaiting_surface_response"
            if contact_band
            else "outside_contact_band",
            "local_max_displacement_m": local_max_displacement,
            "local_max_speed_mps": local_max_speed,
            "background_displacement_m": background_displacement,
            "background_speed_mps": background_speed,
            "displacement_excess_m": float(displacement_excess),
            "speed_excess_mps": float(speed_excess),
            "proxy_motion_m": proxy_motion,
            "approach_m": approach,
            "proxy_bottom_m": proxy_bottom,
            "local_contact_band_top_m": local_contact_band_top,
            "contact_band": contact_band,
        }
    )
    return response, diagnostics


def _structural_edge_indices() -> tuple[np.ndarray, np.ndarray]:
    first: list[int] = []
    second: list[int] = []
    for row in range(BC025_TOWEL_ROWS):
        for column in range(BC025_TOWEL_COLUMNS - 1):
            index = row * BC025_TOWEL_COLUMNS + column
            first.append(index)
            second.append(index + 1)
    for row in range(BC025_TOWEL_ROWS - 1):
        for column in range(BC025_TOWEL_COLUMNS):
            index = row * BC025_TOWEL_COLUMNS + column
            first.append(index)
            second.append(index + BC025_TOWEL_COLUMNS)
    return np.asarray(first, dtype=np.int64), np.asarray(second, dtype=np.int64)


_TOWEL_EDGE_A, _TOWEL_EDGE_B = _structural_edge_indices()
_TOWEL_TRIANGLES = np.asarray(
    bc025_towel_surface_topology()[1],
    dtype=np.int64,
)


class HandsOnB320TowelIsaacLabEnv(HandsOnE01DetachedHandsIsaacLabEnv):
    """Spread the task-owned Terry Cloth surface with assigned contacts."""

    _TASK_NAME = B320_TOWEL_TASK_NAME
    _TASK_ID = "B320-BC025"
    _SURFACE_LOGICAL_NAME = "b320_towel"
    _SURFACE_ROOT_PATH = BC025_TOWEL_SURFACE_ROOT_PATH
    # The deformable surface has a native node/readback contract, so it must
    # use the same recording-origin gate as rigid B320 tasks.  Leaving this
    # disabled makes the recorder accept an empty ``physics_steps=0`` result
    # without checking node stability, reset drift, or contact validity.
    stable_recording_origin_gate_enabled = True

    def __init__(
        self,
        task_name: str = B320_TOWEL_TASK_NAME,
        **kwargs: Any,
    ) -> None:
        if task_name != B320_TOWEL_TASK_NAME:
            raise ValueError(
                f"B320 towel runtime requires {B320_TOWEL_TASK_NAME!r}, "
                f"got {task_name!r}"
            )
        requested_hand = str(kwargs.get("hand_type", "wuji1")).strip().lower()
        requested_arm = str(kwargs.get("arm_type", "rokae")).strip().lower()
        if requested_hand not in {"wuji1", "wuji", "wuji_hand1", "wuji-hand1"}:
            raise ValueError("B320-BC025 requires hand_type='wuji1'")
        if requested_arm != "rokae":
            raise ValueError("B320-BC025 records against the ROKAE arm contract")
        self._b320_corner_seen = {"left": False, "right": False}
        self._b320_corner_index: dict[str, int | None] = {
            "left": None,
            "right": None,
        }
        self._b320_release_frame: dict[str, int | None] = {
            "left": None,
            "right": None,
        }
        self._b320_towel_metrics: dict[str, Any] = {}
        self._b320_towel_flags = (False,) * 5
        self._b320_assigned_contact_valid = {"left": False, "right": False}
        self._b320_wrong_corner_contact = {"left": False, "right": False}
        self._b320_wrong_corner_contact_seen = {
            "left": False,
            "right": False,
        }
        self._bc025_previous_physx_nodes: np.ndarray | None = None
        # The shared E01 constructor invokes virtual reset hooks while it is
        # composing the first scene.  Suppress only that construction-time
        # sample so the settled baseline remains canonical; later seeded
        # resets are independent of construction history.
        self._bc025_scene_construction_phase = True
        self._bc025_surface_translation_sample_m = np.zeros(
            3, dtype=np.float64
        )
        self._bc025_physical_contact_observed = {
            "left": False,
            "right": False,
        }
        self._bc025_physical_contact_armed = {
            "left": False,
            "right": False,
        }
        self._bc025_contact_response_this_substep = {
            "left": False,
            "right": False,
        }
        self._bc025_contact_response_diagnostics = {
            side: {
                "response": False,
                "reason": "reset",
                "local_node_count": 0,
                "local_max_displacement_m": 0.0,
                "local_max_speed_mps": 0.0,
                "background_displacement_m": 0.0,
                "background_speed_mps": 0.0,
                "displacement_excess_m": 0.0,
                "speed_excess_mps": 0.0,
                "proxy_motion_m": 0.0,
                "approach_m": 0.0,
                "proxy_bottom_m": 0.0,
                "local_contact_band_top_m": 0.0,
                "contact_band": False,
                "assigned_region": False,
                "assigned_distance_m": 0.0,
                "nearest_corner_distance_m": 0.0,
            }
            for side in ("left", "right")
        }
        tuning = E01SurfacePhysicsTuning(
            contact_offset=BC025_TOWEL_CONTACT_OFFSET,
            rest_offset=BC025_TOWEL_REST_OFFSET,
            surface_thickness=BC025_TOWEL_SURFACE_THICKNESS,
            surface_density=720.0,
            dynamic_friction=BC025_TOWEL_DYNAMIC_FRICTION,
            surface_stretch_stiffness=BC025_TOWEL_STRETCH_STIFFNESS,
            surface_shear_stiffness=BC025_TOWEL_SHEAR_STIFFNESS,
            surface_bend_stiffness=BC025_TOWEL_BEND_STIFFNESS,
            elasticity_damping=BC025_TOWEL_ELASTICITY_DAMPING,
            max_linear_velocity=BC025_TOWEL_CONTACT_OFFSET * B320_PHYSICS_HZ,
            max_depenetration_velocity=BC025_TOWEL_MAX_DEPENETRATION_VELOCITY,
            solver_position_iterations=int(
                BC025_PILLOW_COLLISION_CALIBRATION[
                    "solver_position_iterations"
                ]
            ),
        )
        kwargs.setdefault("surface_physics_tuning", tuning)
        kwargs.setdefault("physics_dt", 1.0 / B320_PHYSICS_HZ)
        kwargs.setdefault("hz", B320_CONTROL_HZ)
        kwargs.setdefault("time_limit", 45.0)
        kwargs["hand_type"] = "wuji1"
        kwargs["arm_type"] = "rokae"
        super().__init__(task_name=task_name, **kwargs)
        self._bc025_scene_construction_phase = False
        # E01's default is deterministic for its legacy garment tasks.  BC025
        # explicitly opts into the stricter seeded physical-position contract.
        self._physical_randomization = True

    def stable_recording_origin_task_readiness(self) -> dict[str, Any]:
        """Read the surface tensor and collision fixture without state writes."""
        report: dict[str, Any] = {
            "kind": "bc025_native_surface_geometry_and_rigid_reports",
            "ready": False,
            "deformable_contact_report_available": False,
            "deformable_validation": "native_nodal_geometry",
        }
        if (
            self._e_garment_view is None
            or self._e_readback_failed
            or self._e_failure_reason
        ):
            return {**report, "failure_reasons": ["surface_readback_unavailable_or_failed"]}
        try:
            from pxr import Usd, UsdGeom, UsdPhysics

            table = self._stage.GetPrimAtPath(B320_TABLE_PATH + "/geometry/mesh")
            if (
                not table.IsValid()
                or not table.IsActive()
                or not table.HasAPI(UsdPhysics.CollisionAPI)
            ):
                raise RuntimeError("BC025 tabletop collider is missing")
            if UsdPhysics.CollisionAPI(table).GetCollisionEnabledAttr().Get() is False:
                raise RuntimeError("BC025 tabletop collision is disabled")
            bbox = (
                UsdGeom.BBoxCache(
                    Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]
                )
                .ComputeWorldBound(table)
                .ComputeAlignedRange()
            )
            bounds = np.asarray([bbox.GetMin(), bbox.GetMax()], dtype=np.float64)
            nodes = self._surface_tensor_body(
                self._e_garment_view.get_simulation_nodal_positions(),
                label="origin_nodal_position",
            )
        except Exception as exc:
            return {**report, "failure_reasons": ["surface_or_table_readback_error"], "error": str(exc)}
        report.update(_bc025_origin_surface_geometry(nodes, bounds))
        return report

    def _surface_mesh_metadata(self) -> dict[str, Any]:
        return b320_towel_surface_metadata()

    def _configure_randomizable_table_physics(self) -> tuple[int, bool]:
        """B320-BC025 authors its own fixed table and has no height jitter."""

        return 0, False

    def _interface_surface_nodes(self) -> np.ndarray:
        return _interface_towel_nodes()

    def _spawn_surface_deformable(self) -> str:
        if self._stage is None:
            raise RuntimeError("B320-BC025 stage is unavailable before towel spawn")
        import isaaclab.sim as sim_utils

        return spawn_b320_towel_surface(self._stage, sim_utils)

    def _sample_reset_state(self, *, clear_manual_request: bool) -> None:
        super()._sample_reset_state(clear_manual_request=clear_manual_request)
        if self.randomize and not self._bc025_scene_construction_phase:
            dx = float(
                self._random.uniform(*BC025_TOWEL_TRANSLATION_BOUNDS_M[0])
            )
            dy = float(
                self._random.uniform(*BC025_TOWEL_TRANSLATION_BOUNDS_M[1])
            )
        else:
            dx = 0.0
            dy = 0.0
        # E01 temporarily disables ``randomize`` while sampling its generic
        # reset state.  Install BC025's declared translation after that call so
        # it is the only physical draw and is represented by the inherited
        # garment_initial_state replay field.
        self._e_initial_state[:5] = 0.0
        self._e_initial_state[0] = dx
        self._e_initial_state[1] = dy
        self._e_initial_state[5] = float(self.delta_h[0])
        self._bc025_surface_translation_sample_m = np.asarray(
            [dx, dy, 0.0], dtype=np.float64
        )
        self.delta_h[:] = 0.0
        self._b320_corner_seen = {"left": False, "right": False}
        self._b320_corner_index = {"left": None, "right": None}
        self._b320_release_frame = {"left": None, "right": None}
        self._b320_towel_metrics = {}
        self._b320_towel_flags = (False,) * 5
        self._b320_assigned_contact_valid = {"left": False, "right": False}
        self._b320_wrong_corner_contact = {"left": False, "right": False}
        self._reset_bc025_contact_evidence()

    def _sync_bc025_surface_root_transform(self) -> None:
        """Synchronize the authored body frame with the sampled reset offset.

        PhysX surface nodal tensors are exposed in the global subspace used by
        E01.  Moving only those points leaves the deformable body's USD frame
        at its previous pose; on the next solver step PhysX can then apply a
        uniform frame correction that looks like a spurious high-speed cloth
        event.  BC025 therefore moves its task-owned body Xform first and
        writes the matching world-space nodal state immediately afterwards.
        """

        if self._stage is None:
            return
        from pxr import Gf, UsdGeom

        prim = self._stage.GetPrimAtPath(BC025_TOWEL_SURFACE_ROOT_PATH)
        if not prim.IsValid():
            raise RuntimeError(
                "B320-BC025 cannot synchronize its missing surface root: "
                f"{BC025_TOWEL_SURFACE_ROOT_PATH}"
            )
        xformable = UsdGeom.Xformable(prim)
        translate_ops = [
            op
            for op in xformable.GetOrderedXformOps()
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate
        ]
        if len(translate_ops) > 1:
            raise RuntimeError(
                "B320-BC025 surface root has multiple translate ops; refusing "
                "to guess which body frame PhysX should use"
            )
        translate_op = (
            translate_ops[0]
            if translate_ops
            else xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
        )
        dx = float(self._e_initial_state[0])
        dy = float(self._e_initial_state[1])
        dz = float(self._e_initial_state[5])
        target = np.asarray(B320_TOWEL_PLACEMENT, dtype=np.float64).copy()
        target[:2] += (dx, dy)
        target[2] += dz
        if translate_op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat:
            translate_op.Set(Gf.Vec3f(*map(float, target)))
        else:
            translate_op.Set(Gf.Vec3d(*map(float, target)))
        prim.SetCustomDataByKey(
            "handson:resetTranslationM", (float(dx), float(dy), 0.0)
        )
        prim.SetCustomDataByKey(
            "handson:resetFrameSync",
            "surface_root_translate_then_world_nodal_write",
        )

        # Propagate the USD body-frame change before the tensor reset.  The
        # base implementation forwards once more after writing the nodes;
        # this first forward makes the ordering explicit for the solver.
        if (
            self._sim is not None
            and self._e_garment_view is not None
            and not self._bc025_scene_construction_phase
        ):
            self._sim.forward()

    def _reset_bc025_contact_evidence(self) -> None:
        """Clear all contact witnesses and the pre-assist readback frame."""

        self._bc025_previous_physx_nodes = None
        self._bc025_physical_contact_observed = {
            "left": False,
            "right": False,
        }
        self._bc025_physical_contact_armed = {
            "left": False,
            "right": False,
        }
        self._bc025_contact_response_this_substep = {
            "left": False,
            "right": False,
        }
        self._b320_wrong_corner_contact_seen = {
            "left": False,
            "right": False,
        }
        self._bc025_contact_response_diagnostics = {
            side: {
                "response": False,
                "reason": "reset",
                "local_node_count": 0,
                "local_max_displacement_m": 0.0,
                "local_max_speed_mps": 0.0,
                "background_displacement_m": 0.0,
                "background_speed_mps": 0.0,
                "displacement_excess_m": 0.0,
                "speed_excess_mps": 0.0,
                "proxy_motion_m": 0.0,
                "approach_m": 0.0,
                "proxy_bottom_m": 0.0,
                "local_contact_band_top_m": 0.0,
                "contact_band": False,
                "assigned_region": False,
                "assigned_distance_m": 0.0,
                "nearest_corner_distance_m": 0.0,
            }
            for side in ("left", "right")
        }

    def _apply_task_pose_randomization(self) -> None:
        self._sync_bc025_surface_root_transform()
        super()._apply_task_pose_randomization()
        # This hook is also used by diagnostic/state-guided restore, which does
        # not necessarily pass through ``_sample_reset_state``.
        self._reset_bc025_contact_evidence()

    def _prepare_cloth_contact_assist(
        self,
        nodes: np.ndarray,
        velocities: np.ndarray,
    ) -> None:
        """Build BC025's physical-contact gate from raw PhysX readback.

        The previous frame is captured before any local assistance is written.
        Consequently the first gate transition can only be caused by a
        post-step response from the real kinematic proxy/surface collision,
        never by the recovery velocity that follows this hook.
        """

        current = np.asarray(nodes, dtype=np.float64)
        previous = self._bc025_previous_physx_nodes
        if (
            previous is None
            or previous.shape != current.shape
            or not np.all(np.isfinite(previous))
        ):
            for side in ("left", "right"):
                self._bc025_contact_response_this_substep[side] = False
                self._bc025_contact_response_diagnostics[side] = {
                    "response": False,
                    "reason": "awaiting_previous_physx_frame",
                    "local_node_count": 0,
                    "local_max_displacement_m": 0.0,
                    "local_max_speed_mps": 0.0,
                    "background_displacement_m": 0.0,
                    "background_speed_mps": 0.0,
                    "displacement_excess_m": 0.0,
                    "speed_excess_mps": 0.0,
                    "proxy_motion_m": 0.0,
                    "approach_m": 0.0,
                    "proxy_bottom_m": 0.0,
                    "local_contact_band_top_m": 0.0,
                    "contact_band": False,
                    "assigned_region": False,
                    "assigned_distance_m": 0.0,
                    "nearest_corner_distance_m": 0.0,
                }
            self._bc025_previous_physx_nodes = current.copy()
            return

        for side in ("left", "right"):
            hand_closed = bool(self._contact_assist_grasp_commanded(side))
            proxy = np.asarray(
                self._contact_proxy_last_pose[side],
                dtype=np.float64,
            )
            previous_proxy = np.asarray(
                self._cloth_contact_assist_last_proxy_pose[side],
                dtype=np.float64,
            )
            detected, diagnostics = _detect_bc025_surface_contact_response(
                current,
                previous,
                velocities,
                proxy,
                previous_proxy,
                proxy_radius=float(self._contact_proxy_radius),
                hand_closed=hand_closed,
            )
            assigned_index = _BC025_ASSIGNED_CORNER_INDICES[side]
            assigned_distance = float("inf")
            nearest_corner_distance = float("inf")
            if (
                current.shape == (BC025_TOWEL_NODE_COUNT, 3)
                and np.all(np.isfinite(current))
                and proxy.size >= 3
                and np.all(np.isfinite(proxy[:3]))
            ):
                assigned_distance = float(
                    np.linalg.norm(current[assigned_index] - proxy[:3])
                )
                nearest_corner_distance = float(
                    np.min(
                        np.linalg.norm(
                            current[_TOWEL_CORNER_INDICES] - proxy[:3],
                            axis=1,
                        )
                    )
                )
            assigned_region = bool(
                detected
                and assigned_distance <= _BC025_ASSIGNED_RESPONSE_DISTANCE_M
            )
            wrong_region = bool(
                detected
                and nearest_corner_distance <= _BC025_ASSIGNED_RESPONSE_DISTANCE_M
                and not assigned_region
            )
            diagnostics.update(
                {
                    "assigned_region": assigned_region,
                    "assigned_distance_m": assigned_distance
                    if math.isfinite(assigned_distance)
                    else 0.0,
                    "nearest_corner_distance_m": nearest_corner_distance
                    if math.isfinite(nearest_corner_distance)
                    else 0.0,
                }
            )
            self._bc025_contact_response_this_substep[side] = bool(detected)
            self._bc025_contact_response_diagnostics[side] = diagnostics
            if assigned_region:
                self._bc025_physical_contact_observed[side] = True
                self._bc025_physical_contact_armed[side] = True
            if wrong_region:
                self._b320_wrong_corner_contact_seen[side] = True
            if not hand_closed:
                # Historical assigned-contact evidence remains latched for the
                # ordered event, but a later closed-hand episode must produce a
                # new physical response before assistance can bind again.
                self._bc025_physical_contact_armed[side] = False
        self._bc025_previous_physx_nodes = current.copy()

    def _cloth_contact_assist_allowed(
        self,
        side: str,
        *,
        nodes: np.ndarray,
        velocities: np.ndarray,
        indices: np.ndarray,
        proxy_pose: np.ndarray,
        previous_proxy_pose: np.ndarray,
        contact_active: bool,
        hand_closed: bool,
        grasp_candidate: bool,
    ) -> bool:
        """Only allow BC025 recovery after assigned physical response."""

        del nodes, velocities, indices, previous_proxy_pose, contact_active
        del grasp_candidate
        return bool(
            hand_closed
            and np.all(np.isfinite(np.asarray(proxy_pose, dtype=np.float64)))
            and self._bc025_physical_contact_armed.get(side, False)
        )

    @staticmethod
    def _edge_stretch_ratio(nodes: np.ndarray, baseline: np.ndarray) -> float:
        current_lengths = np.linalg.norm(
            nodes[_TOWEL_EDGE_A] - nodes[_TOWEL_EDGE_B],
            axis=1,
        )
        baseline_lengths = np.linalg.norm(
            baseline[_TOWEL_EDGE_A] - baseline[_TOWEL_EDGE_B],
            axis=1,
        )
        ratios = current_lengths / np.maximum(baseline_lengths, 1.0e-9)
        return max(0.0, float(np.max(ratios)) - 1.0)

    @staticmethod
    def _assigned_corner_end_distance(nodes: np.ndarray) -> float:
        """Return the live 3-D distance between the two assigned upper corners."""

        left = nodes[_BC025_ASSIGNED_CORNER_INDICES["left"]]
        right = nodes[_BC025_ASSIGNED_CORNER_INDICES["right"]]
        return float(np.linalg.norm(right - left))

    @staticmethod
    def _target_bounds() -> tuple[float, float, float, float]:
        center_x, center_y = B320_TOWEL_TARGET_CENTER_XY
        size_x, size_y = B320_TOWEL_TARGET_SIZE_XY
        return (
            center_x - 0.5 * size_x,
            center_x + 0.5 * size_x,
            center_y - 0.5 * size_y,
            center_y + 0.5 * size_y,
        )

    @classmethod
    def _coverage_ratio(cls, nodes: np.ndarray) -> float:
        """Rasterize projected surface triangles over the target rectangle."""

        x_min, x_max, y_min, y_max = cls._target_bounds()
        grid_size = 18
        xs = np.linspace(x_min, x_max, grid_size + 2, dtype=np.float64)[1:-1]
        ys = np.linspace(y_min, y_max, grid_size + 2, dtype=np.float64)[1:-1]
        samples = np.stack(np.meshgrid(xs, ys, indexing="xy"), axis=-1).reshape(-1, 2)
        triangles = nodes[_TOWEL_TRIANGLES, :2]
        a = triangles[:, 0]
        b = triangles[:, 1]
        c = triangles[:, 2]
        v0 = c - a
        v1 = b - a
        dot00 = np.einsum("ij,ij->i", v0, v0)
        dot01 = np.einsum("ij,ij->i", v0, v1)
        dot11 = np.einsum("ij,ij->i", v1, v1)
        denominator = dot00 * dot11 - dot01 * dot01
        valid = np.abs(denominator) > 1.0e-14
        inverse = np.zeros_like(denominator)
        inverse[valid] = 1.0 / denominator[valid]
        covered = 0
        for sample in samples:
            v2 = sample - a
            dot02 = np.einsum("ij,ij->i", v0, v2)
            dot12 = np.einsum("ij,ij->i", v1, v2)
            u = (dot11 * dot02 - dot01 * dot12) * inverse
            v = (dot00 * dot12 - dot01 * dot02) * inverse
            if bool(np.any(valid & (u >= -1.0e-8) & (v >= -1.0e-8) & (u + v <= 1.0 + 1.0e-8))):
                covered += 1
        return float(covered / max(1, samples.shape[0]))

    def _hand_distance_to_towel(self, side: str, nodes: np.ndarray) -> float:
        pose = self._left_tcp_pose if side == "left" else self._right_tcp_pose
        return float(np.min(np.linalg.norm(nodes - pose[:3], axis=1)))

    @staticmethod
    def _surface_pose_from_nodes(nodes: np.ndarray) -> np.ndarray:
        """Build a finite pose from a validated surface-node array."""
        points = np.asarray(nodes, dtype=np.float64)
        # A finite but absurdly large injected node array can overflow the
        # covariance product (1e308 * 1e308) and otherwise leak NaNs through
        # the observation pose.  Treat it as an invalid readback just like a
        # non-finite array and return a finite, clearly non-successful pose.
        fallback = np.asarray(
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64
        )
        if (
            points.ndim != 2
            or points.shape[1] != 3
            or not np.all(np.isfinite(points))
            or np.any(np.abs(points) > 1.0e6)
        ):
            return fallback
        try:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                centroid = np.mean(points, axis=0)
                centered = points[:, :2] - centroid[:2]
                covariance = centered.T @ centered / max(1, points.shape[0] - 1)
            if not np.all(np.isfinite(covariance)):
                return fallback
            _values, vectors = np.linalg.eigh(covariance)
            axis = vectors[:, 1]
            if axis[0] < 0.0:
                axis = -axis
            yaw = math.atan2(float(axis[1]), float(axis[0]))
            result = np.asarray(
                [
                    *centroid,
                    math.cos(0.5 * yaw),
                    0.0,
                    0.0,
                    math.sin(0.5 * yaw),
                ],
                dtype=np.float64,
            )
            return result if np.all(np.isfinite(result)) else fallback
        except Exception:
            return fallback

    def _safe_surface_nodes_for_observation(self) -> np.ndarray:
        """Never expose an injected/failed nodal readback in observations."""
        nodes = np.asarray(self._e_nodes_np, dtype=np.float64)
        if nodes.shape == (BC025_TOWEL_NODE_COUNT, 3) and np.all(
            np.isfinite(nodes)
        ):
            return nodes
        baseline = np.asarray(self._e_base_nodes_np, dtype=np.float64)
        if baseline.shape == (BC025_TOWEL_NODE_COUNT, 3) and np.all(
            np.isfinite(baseline)
        ):
            return baseline.copy()
        return self._interface_surface_nodes().copy()

    def _update_corner_events(self, nodes: np.ndarray) -> None:
        for side in ("left", "right"):
            proxy = np.asarray(
                self._contact_proxy_last_pose[side][:3],
                dtype=np.float64,
            )
            hand_closed = bool(self._contact_assist_grasp_commanded(side))
            active = bool(
                self._cloth_contact_assist_grasp_active.get(side, False)
                and self._cloth_contact_assist_active_nodes.get(side, 0) > 0
                and self._bc025_physical_contact_observed.get(side, False)
            )
            assigned_index = _BC025_ASSIGNED_CORNER_INDICES[side]
            assigned_distance = float("inf")
            nearest_corner_distance = float("inf")
            if np.all(np.isfinite(proxy)):
                assigned_distance = float(
                    np.linalg.norm(nodes[assigned_index] - proxy)
                )
                nearest_corner_distance = float(
                    np.min(np.linalg.norm(nodes[_TOWEL_CORNER_INDICES] - proxy, axis=1))
                )
            assigned_contact = bool(
                active
                and hand_closed
                and assigned_distance <= _TOWEL_CORNER_CONTACT_DISTANCE_M
            )
            self._b320_assigned_contact_valid[side] = assigned_contact
            wrong_corner_now = bool(
                active
                and hand_closed
                and nearest_corner_distance <= _TOWEL_CORNER_CONTACT_DISTANCE_M
                and not assigned_contact
            )
            self._b320_wrong_corner_contact[side] = bool(
                wrong_corner_now
                or self._b320_wrong_corner_contact_seen.get(side, False)
            )
            if wrong_corner_now:
                self._b320_wrong_corner_contact_seen[side] = True
            if assigned_contact and not self._b320_corner_seen[side]:
                self._b320_corner_seen[side] = True
                self._b320_corner_index[side] = assigned_index
            if (
                self._b320_corner_seen[side]
                and not hand_closed
                and self._b320_release_frame[side] is None
                and self._hand_distance_to_towel(side, nodes)
                >= _TOWEL_RELEASE_DISTANCE_M
            ):
                self._b320_release_frame[side] = int(self.env_step)

    def _bc025_contact_state_metrics(self) -> dict[str, Any]:
        """Return finite, serializable hidden-contact state for recording."""

        result: dict[str, Any] = {
            "physical_contact_evidence_source": (
                "localized_surface_nodal_response_after_proxy_approach"
            ),
            "physical_contact_report_scope": (
                "deformable_nodal_readback; no_rigid_contact_manifold_in_isaacsim_5_1"
            ),
        }
        for side in ("left", "right"):
            diagnostics = self._bc025_contact_response_diagnostics.get(side, {})
            result.update(
                {
                    f"physical_contact_observed_{side}": bool(
                        self._bc025_physical_contact_observed.get(side, False)
                    ),
                    f"physical_contact_armed_{side}": bool(
                        self._bc025_physical_contact_armed.get(side, False)
                    ),
                    f"physical_contact_response_{side}": bool(
                        self._bc025_contact_response_this_substep.get(side, False)
                    ),
                    f"wrong_corner_contact_seen_{side}": bool(
                        self._b320_wrong_corner_contact_seen.get(side, False)
                    ),
                    f"physical_contact_response_reason_{side}": str(
                        diagnostics.get("reason", "unknown")
                    ),
                    f"physical_contact_response_displacement_m_{side}": float(
                        diagnostics.get("local_max_displacement_m", 0.0)
                    ),
                    f"physical_contact_response_speed_mps_{side}": float(
                        diagnostics.get("local_max_speed_mps", 0.0)
                    ),
                    f"physical_contact_response_score_{side}": float(
                        max(
                            float(diagnostics.get("displacement_excess_m", 0.0))
                            / max(
                                _BC025_CONTACT_RESPONSE_MIN_DISPLACEMENT_EXCESS_M,
                                1.0e-12,
                            ),
                            float(diagnostics.get("speed_excess_mps", 0.0))
                            / max(
                                _BC025_CONTACT_RESPONSE_MIN_SPEED_EXCESS_MPS,
                                1.0e-12,
                            ),
                            0.0,
                        )
                    ),
                    f"physical_contact_assigned_region_{side}": bool(
                        diagnostics.get("assigned_region", False)
                    ),
                    f"physical_contact_assigned_distance_m_{side}": float(
                        diagnostics.get("assigned_distance_m", 0.0)
                    ),
                }
            )
        return result

    def _metrics(self, *, update_events: bool) -> dict[str, Any]:
        nodes = np.asarray(self._e_nodes_np, dtype=np.float64)
        baseline = np.asarray(self._e_base_nodes_np, dtype=np.float64)
        velocities = np.asarray(self._e_velocities_np, dtype=np.float64)
        valid_state = bool(
            not self._e_readback_failed
            and nodes.shape == (BC025_TOWEL_NODE_COUNT, 3)
            and baseline.shape == nodes.shape
            and np.all(np.isfinite(nodes))
            and np.all(np.isfinite(baseline))
            and velocities.shape == nodes.shape
            and np.all(np.isfinite(velocities))
        )
        if not valid_state:
            self._b320_towel_flags = (False,) * 5
            result = {
                "valid_surface_state": False,
                "max_stretch_ratio": _INVALID_SURFACE_STRETCH,
                "assigned_corner_end_distance_m": 0.0,
                "assigned_corner_end_distance_error_ratio": 1.0,
                "assigned_corner_distance_in_tolerance": False,
                "corner_count_in_target": 0,
                "coverage_ratio": 0.0,
                "release_delta_s": _UNAVAILABLE_RELEASE_DELTA_S,
                "max_nodal_speed_mps": _INVALID_SURFACE_SPEED_MPS,
                "assigned_contact_valid_left": False,
                "assigned_contact_valid_right": False,
                "wrong_corner_contact_left": False,
                "wrong_corner_contact_right": False,
                "success_now": False,
                **self._bc025_contact_state_metrics(),
            }
            self._b320_towel_metrics = result
            return result
        if update_events:
            self._update_corner_events(nodes)
        corners = nodes[_TOWEL_CORNER_INDICES]
        x_min, x_max, y_min, y_max = self._target_bounds()
        inside = (
            (corners[:, 0] >= x_min)
            & (corners[:, 0] <= x_max)
            & (corners[:, 1] >= y_min)
            & (corners[:, 1] <= y_max)
        )
        corner_count = int(np.count_nonzero(inside))
        stretch = self._edge_stretch_ratio(nodes, baseline)
        coverage = self._coverage_ratio(nodes)
        if not np.isfinite(stretch) or not np.isfinite(coverage):
            self._b320_towel_flags = (False,) * 5
            result = {
                "valid_surface_state": False,
                "max_stretch_ratio": _INVALID_SURFACE_STRETCH,
                "assigned_corner_end_distance_m": 0.0,
                "assigned_corner_end_distance_error_ratio": 1.0,
                "assigned_corner_distance_in_tolerance": False,
                "corner_count_in_target": 0,
                "coverage_ratio": 0.0,
                "release_delta_s": _UNAVAILABLE_RELEASE_DELTA_S,
                "max_nodal_speed_mps": _INVALID_SURFACE_SPEED_MPS,
                "assigned_contact_valid_left": False,
                "assigned_contact_valid_right": False,
                "wrong_corner_contact_left": False,
                "wrong_corner_contact_right": False,
                "success_now": False,
                **self._bc025_contact_state_metrics(),
            }
            self._b320_towel_metrics = result
            return result
        different_corners = bool(
            all(self._b320_corner_seen.values())
            and not any(self._b320_wrong_corner_contact_seen.values())
            and self._b320_corner_index["left"]
            == _BC025_ASSIGNED_CORNER_INDICES["left"]
            and self._b320_corner_index["right"]
            == _BC025_ASSIGNED_CORNER_INDICES["right"]
        )
        assigned_end_distance = self._assigned_corner_end_distance(nodes)
        assigned_end_distance_error = abs(
            assigned_end_distance - BC025_TOWEL_TARGET_END_DISTANCE_M
        ) / max(BC025_TOWEL_TARGET_END_DISTANCE_M, 1.0e-9)
        assigned_end_distance_ok = bool(
            np.isfinite(assigned_end_distance_error)
            and assigned_end_distance_error
            <= _TOWEL_TARGET_CORNER_DISTANCE_TOLERANCE
        )
        released = all(value is not None for value in self._b320_release_frame.values())
        release_delta = (
            abs(int(self._b320_release_frame["left"]) - int(self._b320_release_frame["right"]))
            / B320_CONTROL_HZ
            if released
            else _UNAVAILABLE_RELEASE_DELTA_S
        )
        max_speed = float(np.max(np.linalg.norm(velocities, axis=1)))
        if not np.isfinite(max_speed):
            self._b320_towel_flags = (False,) * 5
            result = {
                "valid_surface_state": False,
                "max_stretch_ratio": _INVALID_SURFACE_STRETCH,
                "assigned_corner_end_distance_m": 0.0,
                "assigned_corner_end_distance_error_ratio": 1.0,
                "assigned_corner_distance_in_tolerance": False,
                "corner_count_in_target": 0,
                "coverage_ratio": 0.0,
                "release_delta_s": _UNAVAILABLE_RELEASE_DELTA_S,
                "max_nodal_speed_mps": _INVALID_SURFACE_SPEED_MPS,
                "assigned_contact_valid_left": False,
                "assigned_contact_valid_right": False,
                "wrong_corner_contact_left": False,
                "wrong_corner_contact_right": False,
                "success_now": False,
                **self._bc025_contact_state_metrics(),
            }
            self._b320_towel_metrics = result
            return result
        flags = (
            different_corners,
            stretch <= 0.15 and assigned_end_distance_ok,
            corner_count >= 3,
            coverage > 0.80,
            released and release_delta <= 1.0 and max_speed <= _TOWEL_STABLE_SPEED_MPS,
        )
        self._b320_towel_flags = tuple(bool(value) for value in flags)
        result = {
            "valid_surface_state": True,
            "max_stretch_ratio": stretch,
            "assigned_corner_end_distance_m": assigned_end_distance,
            "assigned_corner_end_distance_error_ratio": assigned_end_distance_error,
            "assigned_corner_distance_in_tolerance": assigned_end_distance_ok,
            "corner_count_in_target": corner_count,
            "coverage_ratio": coverage,
            "release_delta_s": release_delta,
            "max_nodal_speed_mps": max_speed,
            "assigned_contact_valid_left": bool(
                self._b320_assigned_contact_valid["left"]
            ),
            "assigned_contact_valid_right": bool(
                self._b320_assigned_contact_valid["right"]
            ),
            "wrong_corner_contact_left": bool(
                self._b320_wrong_corner_contact["left"]
            ),
            "wrong_corner_contact_right": bool(
                self._b320_wrong_corner_contact["right"]
            ),
            "success_now": bool(all(flags)),
            **self._bc025_contact_state_metrics(),
        }
        self._b320_towel_metrics = result
        return result

    def _build_observation_space(self) -> spaces.Dict:
        state_spaces = {
            "tcp_pose": spaces.Box(-np.inf, np.inf, shape=(14,), dtype=np.float64),
            "gripper_pose": spaces.Box(-np.inf, np.inf, shape=(32,), dtype=np.float64),
            "towel_pose": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float64),
            "towel_corners": spaces.Box(-np.inf, np.inf, shape=(12,), dtype=np.float64),
            "target_bounds": spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float64),
            "spread_metrics": spaces.Box(-np.inf, np.inf, shape=(12,), dtype=np.float64),
            "grasp_state": spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float64),
            "garment_initial_state": spaces.Box(
                -np.inf, np.inf, shape=(6,), dtype=np.float64
            ),
            "table_delta_height": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64),
        }
        observation_spaces: dict[str, spaces.Space] = {
            "state": spaces.Dict(state_spaces),
            "e01_hand_qpos": spaces.Box(-np.inf, np.inf, shape=(40,), dtype=np.float64),
        }
        if self.image_obs:
            head_space = spaces.Box(
                0, 255, shape=self._image_shape, dtype=np.uint8
            )
            wrist_width, wrist_height = WRIST_RECORDING_CAMERA_OUTPUT_SIZE
            wrist_space = spaces.Box(
                0,
                255,
                shape=(wrist_height, wrist_width, 3),
                dtype=np.uint8,
            )
            observation_spaces["images"] = spaces.Dict(
                {
                    "random_camera" if self.randomize else "ego": head_space,
                    "wrist_left": wrist_space,
                    "wrist_right": wrist_space,
                }
            )
        return spaces.Dict(observation_spaces)

    def _compute_observation(self) -> dict[str, Any]:
        metrics = self._metrics(update_events=False)
        tcp, gripper = self._robot_state()
        nodes = self._safe_surface_nodes_for_observation()
        spread = np.zeros(12, dtype=np.float64)
        values = [
            float(metrics["valid_surface_state"]),
            float(metrics["max_stretch_ratio"]),
            float(metrics["corner_count_in_target"]),
            float(metrics["coverage_ratio"]),
            float(metrics["release_delta_s"]),
            float(metrics["max_nodal_speed_mps"]),
            float(metrics["success_now"]),
            float(self._success_counter),
            float(metrics["assigned_corner_end_distance_m"]),
            float(metrics["assigned_corner_end_distance_error_ratio"]),
            float(metrics["assigned_contact_valid_left"]),
            float(metrics["assigned_contact_valid_right"]),
        ]
        spread[: len(values)] = values
        corner_values = [
            -1 if self._b320_corner_index[side] is None else self._b320_corner_index[side]
            for side in ("left", "right")
        ]
        state = {
            "tcp_pose": tcp,
            "gripper_pose": gripper,
            "towel_pose": self._surface_pose_from_nodes(nodes),
            "towel_corners": nodes[_TOWEL_CORNER_INDICES].reshape(-1).astype(np.float64),
            "target_bounds": np.asarray(self._target_bounds(), dtype=np.float64),
            "spread_metrics": spread,
            "grasp_state": np.asarray(
                [
                    float(self._b320_corner_seen["left"]),
                    float(self._b320_corner_seen["right"]),
                    float(corner_values[0]),
                    float(corner_values[1]),
                    float(self._cloth_contact_assist_grasp_active["left"]),
                    float(self._cloth_contact_assist_grasp_active["right"]),
                    float(self._b320_release_frame["left"] is not None),
                    float(self._b320_release_frame["right"] is not None),
                ],
                dtype=np.float64,
            ),
            "garment_initial_state": self._e_initial_state.astype(
                np.float64, copy=True
            ),
            "table_delta_height": self.delta_h.astype(np.float64, copy=True),
        }
        observation: dict[str, Any] = {
            "state": state,
            "e01_hand_qpos": self._e01_recording_hand_qpos(),
        }
        if self.image_obs:
            front, wrist_left, wrist_right = self.render()
            observation["images"] = {
                "random_camera" if self.randomize else "ego": front,
                "wrist_left": wrist_left,
                "wrist_right": wrist_right,
            }
        return observation

    def _compute_success_now(self) -> bool:
        if self._e_failure_reason is not None:
            return False
        return bool(self._metrics(update_events=True)["success_now"])

    def _success_steps_required(self) -> int:
        return 15

    def _success_condition_rows(self) -> list[dict[str, Any]]:
        self._metrics(update_events=False)
        labels = (
            "Assigned hands contacted the upper-left and upper-right towel corners",
            "Towel stretch stays below 15 percent and assigned end distance is within 10 percent",
            "At least three corners are inside the target frame",
            "Projected target coverage exceeds 80 percent",
            "Both hands release within one second and cloth is stable",
        )
        rows = [
            self._condition_row(f"b320_bc025_{index + 1}", label, flag)
            for index, (label, flag) in enumerate(zip(labels, self._b320_towel_flags))
        ]
        rows.append(
            self._condition_row(
                "b320_success_hold",
                "All spread/release conditions hold for 0.5 seconds",
                self._success_counter >= 15,
            )
        )
        return rows

    def _task_info(self) -> dict[str, Any]:
        info = super()._task_info()
        info.update(
            {
                "task_id": self._TASK_ID,
                "task_name": self.task_name,
                "b320_towel_metrics": dict(self._metrics(update_events=False)),
                "b320_corner_indices": dict(self._b320_corner_index),
                "b320_corner_seen": dict(self._b320_corner_seen),
                "b320_release_frames": dict(self._b320_release_frame),
                "b320_assigned_contact_valid": dict(
                    self._b320_assigned_contact_valid
                ),
                "b320_wrong_corner_contact": dict(
                    self._b320_wrong_corner_contact
                ),
                "b320_wrong_corner_contact_seen": dict(
                    self._b320_wrong_corner_contact_seen
                ),
                "b320_physical_contact_observed": dict(
                    self._bc025_physical_contact_observed
                ),
                "b320_physical_contact_armed": dict(
                    self._bc025_physical_contact_armed
                ),
                "b320_contact_response_thresholds": {
                    "radius_m": _BC025_CONTACT_RESPONSE_RADIUS_M,
                    "min_local_nodes": _BC025_CONTACT_RESPONSE_MIN_LOCAL_NODES,
                    "min_displacement_m": _BC025_CONTACT_RESPONSE_MIN_DISPLACEMENT_M,
                    "min_displacement_excess_m": _BC025_CONTACT_RESPONSE_MIN_DISPLACEMENT_EXCESS_M,
                    "min_speed_mps": _BC025_CONTACT_RESPONSE_MIN_SPEED_MPS,
                    "min_speed_excess_mps": _BC025_CONTACT_RESPONSE_MIN_SPEED_EXCESS_MPS,
                    "assigned_region_distance_m": _BC025_ASSIGNED_RESPONSE_DISTANCE_M,
                },
                "b320_surface_only": True,
                "b320_surface_node_count": BC025_TOWEL_NODE_COUNT,
                "b320_surface_triangle_count": BC025_TOWEL_TRIANGLE_COUNT,
                "physical_randomization": bool(self._physical_randomization),
                "b320_surface_translation_randomization_enabled": bool(
                    self.randomize
                ),
                "b320_surface_translation_bounds_m": tuple(
                    tuple(float(value) for value in bounds)
                    for bounds in BC025_TOWEL_TRANSLATION_BOUNDS_M
                ),
                "b320_surface_translation_sample_m": tuple(
                    float(value)
                    for value in self._bc025_surface_translation_sample_m
                ),
                "b320_surface_reset_centroid_m": tuple(
                    float(value)
                    for value in np.mean(self._e_reset_nodes_np, axis=0)
                ),
                "b320_material_source": BC025_MATERIAL_TITLE,
                "b320_collision_source": BC025_COLLISION_SOURCE_TITLE,
                "b320_asset_manifest": bc025_static_asset_manifest(),
                "b320_replay_mode": "diagnostic_state_guided",
                "b320_physical_replay_certified": False,
                "b320_research_level": "R3",
            }
        )
        return info

    def get_physical_replay_reset_contract(self) -> dict[str, Any]:
        contract = dict(super().get_physical_replay_reset_contract())
        contract.update(
            {
                "b320_task_id": self._TASK_ID,
                "b320_surface_root": BC025_TOWEL_SURFACE_ROOT_PATH,
                "b320_surface_only": True,
                "b320_replay_mode": "diagnostic_state_guided",
                "b320_physical_replay_certified": False,
                "b320_physical_contact_evidence_source": (
                    "localized_surface_nodal_response_after_proxy_approach"
                ),
                "b320_surface_translation_randomization_enabled": bool(
                    self.randomize
                ),
                "b320_surface_translation_bounds_m": tuple(
                    tuple(float(value) for value in bounds)
                    for bounds in BC025_TOWEL_TRANSLATION_BOUNDS_M
                ),
                "b320_surface_translation_sample_m": tuple(
                    float(value)
                    for value in self._bc025_surface_translation_sample_m
                ),
                "b320_surface_reset_centroid_m": tuple(
                    float(value)
                    for value in np.mean(self._e_reset_nodes_np, axis=0)
                ),
            }
        )
        return contract


__all__ = ["HandsOnB320TowelIsaacLabEnv"]
