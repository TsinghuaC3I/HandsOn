"""Source-geometry reset and A06-compatible orientation semantics for GEN1.

This module has no Isaac imports, so geometry and predicate checks run on CPU.
All positions use metres. Public poses use XYZ + WXYZ; PhysX reset writes use
XYZ + XYZW. Geometry is measured from the imported source asset, never a proxy.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

TASK_NAME = "eval_gen03_orient_bottle_upright"
BODY_NAME = "gen1_soap_bottle"
TITLE = "EVAL-Gen03 Orient Mustard Bottle Upright"
PROMPT = (
    "Pick up the yellow mustard bottle and place it upright on the table with its bottom "
    "facing down. Release the bottle and keep it upright for 0.5 seconds."
)
XY_BOUNDS = np.asarray([[-0.35, -0.12], [-0.25, 0.0]], dtype=np.float64)
RESET_CENTER_XY = XY_BOUNDS.mean(axis=0)
YAW_BOUNDS_DEG = (-30.0, 30.0)
UPRIGHT_MAX_ANGLE_DEG = 25.0
TABLE_HEIGHT_TOLERANCE_M = 0.040
RELEASE_HOLD_SECONDS = 0.50
SUPPORT_CLEARANCE_M = 0.0005
# A numerical contact-report threshold, not a required operator grip strength.
CONTACT_FORCE_EPS_N = 1e-4
# Reset on the bottle's broad side. Keep movement evidence as well as final
# orientation/support/release checks so an untouched reset cannot succeed.
MIN_MOVE_DISTANCE_M = 0.025
MIN_LIFT_DISTANCE_M = 0.020
SUCCESS_LABELS = (
    "Bottle upright (tilt at most 25 degrees)",
    "Bottle bottom supported by the table",
    "Hand released from the bottle",
)
ASSET_DIR = Path(__file__).resolve().parents[1] / "envs/xmls/assets/eval_gen03_orient_bottle_upright"


@dataclass(frozen=True)
class BottleGeometry:
    collision_vertices: np.ndarray
    bottom_vertices: np.ndarray
    side_rest_quat_wxyz: np.ndarray
    mass_kg: float
    center_of_mass: np.ndarray
    diagonal_inertia: np.ndarray
    principal_axes_wxyz: np.ndarray

    @classmethod
    def from_file(cls, path: Path) -> "BottleGeometry":
        data = json.loads(path.read_text())
        return cls(
            collision_vertices=np.asarray(data["collision_vertices_m"], dtype=np.float64),
            bottom_vertices=np.asarray(data["bottom_vertices_m"], dtype=np.float64),
            side_rest_quat_wxyz=np.asarray(data["side_rest_quat_wxyz"], dtype=np.float64),
            mass_kg=float(data["mass_kg"]),
            center_of_mass=np.asarray(data["center_of_mass_m"], dtype=np.float64),
            diagonal_inertia=np.asarray(data["diagonal_inertia_kg_m2"], dtype=np.float64),
            principal_axes_wxyz=np.asarray(data["principal_axes_wxyz"], dtype=np.float64),
        )


@lru_cache(maxsize=1)
def geometry() -> BottleGeometry:
    return BottleGeometry.from_file(ASSET_DIR / "geometry.json")


def rotation_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("GEN1 requires a finite nonzero quaternion")
    w, x, y, z = q / norm
    return np.asarray([
        [1 - 2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1 - 2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1 - 2*(x*x+y*y)],
    ])


def pose_at_support(
    shape: BottleGeometry, xy: np.ndarray, yaw_deg: float, table_z: float,
) -> np.ndarray:
    half_yaw = math.radians(yaw_deg) * 0.5
    c, s = math.cos(half_yaw), math.sin(half_yaw)
    w, x, y, z = shape.side_rest_quat_wxyz
    quat = np.asarray([c*w-s*z, c*x-s*y, c*y+s*x, c*z+s*w])
    rotated = shape.collision_vertices @ rotation_matrix(quat).T
    height = table_z - float(rotated[:, 2].min()) + SUPPORT_CLEARANCE_M
    return np.concatenate((np.asarray(xy), [height], quat))


def sample_reset_pose(rng, shape: BottleGeometry, table_z: float, *, randomize: bool) -> np.ndarray:
    xy = rng.uniform(XY_BOUNDS[0], XY_BOUNDS[1]) if randomize else RESET_CENTER_XY.copy()
    yaw = float(rng.uniform(*YAW_BOUNDS_DEG)) if randomize else 0.0
    return pose_at_support(shape, xy, yaw, table_z)


def boundary_reset_poses(shape: BottleGeometry, table_z: float) -> list[np.ndarray]:
    """The approved twelve corner/heading layouts, separate from twenty seeds."""
    return [
        pose_at_support(shape, np.asarray([x, y]), yaw, table_z)
        for x in XY_BOUNDS[:, 0]
        for y in XY_BOUNDS[:, 1]
        for yaw in (YAW_BOUNDS_DEG[0], 0.0, YAW_BOUNDS_DEG[1])
    ]


def success_metrics(
    pose_wxyz: np.ndarray, shape: BottleGeometry, table_z: float, *,
    contacts_available: bool, table_force_n: float, hand_force_n: float,
    reset_pose_wxyz: np.ndarray | None = None,
    require_movement: bool = False,
) -> dict[str, float | bool]:
    pose = np.asarray(pose_wxyz, dtype=np.float64)
    rotation = rotation_matrix(pose[3:7])
    alignment = float(np.clip(rotation[2, 2], -1.0, 1.0))
    tilt = math.degrees(math.acos(alignment))
    bottom = shape.bottom_vertices @ rotation.T + pose[:3]
    bottom_error = float(bottom[:, 2].min() - table_z)
    upright = alignment >= math.cos(math.radians(UPRIGHT_MAX_ANGLE_DEG)) - 1e-12
    finite_contacts = bool(np.isfinite(table_force_n) and np.isfinite(hand_force_n))
    supported = bool(
        contacts_available and finite_contacts
        and table_force_n > CONTACT_FORCE_EPS_N
        and abs(bottom_error) <= TABLE_HEIGHT_TOLERANCE_M
    )
    released = bool(
        contacts_available and finite_contacts and hand_force_n <= CONTACT_FORCE_EPS_N
    )
    moved = False
    if reset_pose_wxyz is not None:
        reset_pose = np.asarray(reset_pose_wxyz, dtype=np.float64).reshape(7)
        if np.all(np.isfinite(reset_pose)) and np.all(np.isfinite(pose)):
            displacement = float(np.linalg.norm(pose[:3] - reset_pose[:3]))
            lift = float(pose[2] - reset_pose[2])
            moved = bool(
                displacement >= MIN_MOVE_DISTANCE_M
                or lift >= MIN_LIFT_DISTANCE_M
            )
    movement_ok = bool(moved or not require_movement)
    return {
        "gen1_success_now": bool(upright and supported and released and movement_ok),
        "gen1_upright": bool(upright),
        "gen1_on_table": supported,
        "gen1_hand_released": released,
        "gen1_moved": moved,
        "gen1_movement_required": bool(require_movement),
        "gen1_contacts_available": bool(contacts_available and finite_contacts),
        "gen1_tilt_deg": float(tilt),
        "gen1_bottom_height_error_m": bottom_error,
        "gen1_table_force_n": float(table_force_n),
        "gen1_hand_force_n": float(hand_force_n),
    }


def simulated_step_seconds(env) -> float:
    """Actual PhysX time per control step; the inherited control_dt is nominal.

    A06 currently rounds 0.02 / (1/120) to two physics substeps. Counting the
    nominal 0.02 would accept 0.5 seconds after only 0.4167 simulated seconds.
    Keep the controller unchanged and measure the GEN1 hold on the physics clock.
    """
    dt = float(getattr(env, "_sim_dt", getattr(env, "_physics_dt", 0.0)))
    count = int(getattr(env, "_n_substeps", 0))
    if not np.isfinite(dt) or dt <= 0 or count <= 0:
        raise ValueError("GEN1 requires positive physics timing for its release hold")
    return dt * count


def success_steps(step_seconds: float) -> int:
    if not np.isfinite(step_seconds) or step_seconds <= 0:
        raise ValueError("GEN1 step duration must be finite and positive")
    return max(8, math.ceil(RELEASE_HOLD_SECONDS / step_seconds))


def recording_schema(base: dict) -> dict:
    """Keep hidden progress and reset identity on both sides of transitions."""
    result = dict(base)
    fields = dict(base["fields"])
    fields["gen1_task_state"] = {
        "shape": [4], "dtype": "float64",
        "source": "gen1_action_aligned_task_state",
        "layout": ["success_counter", "contacts_available", "table_force_n", "hand_force_n"],
        "visibility": "hidden_task_logic_and_contact_diagnostics",
        "description": "Consecutive upright/released counter; contact availability and force diagnostics.",
    }
    fields["gen1_reset_pose_wxyz"] = {
        "shape": [7], "dtype": "float64",
        "source": "gen1_environment_rng_authored_reset",
        "layout": ["x", "y", "z", "qw", "qx", "qy", "qz"],
        "description": "Authored episode reset pose, unchanged during physical manipulation.",
    }
    result["fields"] = fields
    result["task_state"] = {
        "task_name": TASK_NAME,
        "field_names": ["gen1_task_state", "gen1_reset_pose_wxyz"],
        "capture_mode": "action_aligned_current_and_next_observation",
        "counter_policy": "Initialize at the episode boundary, then recompute from live physics; no trajectory-state writes after action zero.",
        "contact_force_policy": "Diagnostic measurements, not solver state to restore.",
    }
    return result


def configure_physics(env, sim_utils) -> int:
    """Apply measured source inertia and A06-like contact settings pre-PhysX."""
    from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade

    prim = env._find_rigid_body_prim(BODY_NAME)
    if prim is None:
        raise RuntimeError("GEN1 source bottle rigid body is missing")
    path = str(prim.GetPath())
    sim_utils.make_uninstanceable(path, stage=env._stage)
    sim_utils.update_stage()
    prim = env._stage.GetPrimAtPath(path)
    shape = geometry()
    mass = UsdPhysics.MassAPI.Apply(prim)
    mass.CreateMassAttr().Set(shape.mass_kg)
    mass.CreateCenterOfMassAttr().Set(Gf.Vec3f(*shape.center_of_mass))
    mass.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*shape.diagonal_inertia))
    w, x, y, z = shape.principal_axes_wxyz
    mass.CreatePrincipalAxesAttr().Set(Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z))))
    rigid = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    rigid.CreateLinearDampingAttr().Set(0.40)
    rigid.CreateAngularDampingAttr().Set(2.00)
    # Live support/release reporting requires awake contact impulses. This does
    # not hold the bottle in place or overwrite any velocity during simulation.
    rigid.CreateSleepThresholdAttr().Set(0.0)
    material = UsdShade.Material.Define(env._stage, path + "/GEN1PhysicsMaterial")
    physical = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physical.CreateStaticFrictionAttr().Set(0.8)
    physical.CreateDynamicFrictionAttr().Set(0.6)
    physical.CreateRestitutionAttr().Set(0.0)
    for collider in Usd.PrimRange(prim):
        if not collider.HasAPI(UsdPhysics.CollisionAPI):
            continue
        UsdGeom.Imageable(collider).MakeInvisible()
        contact = PhysxSchema.PhysxCollisionAPI.Apply(collider)
        contact.CreateContactOffsetAttr().Set(0.001)
        contact.CreateRestOffsetAttr().Set(0.0)
        UsdShade.MaterialBindingAPI.Apply(collider).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics"
        )
    return 1


def configure_contact_reporting(env) -> int:
    from pxr import PhysxSchema

    prim = env._find_rigid_body_prim(BODY_NAME)
    if prim is None:
        raise RuntimeError("GEN1 contact reporter cannot find the bottle")
    PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr().Set(0.0)
    return 1


def init_contact_views(env) -> None:
    env._gen1_contact_views = {}
    if not env._robots:
        return
    physics_view = next(iter(env._robots.values()))._physics_sim_view
    bottle = env._find_rigid_body_prim(BODY_NAME)
    table = env._find_rigid_body_prim("table")
    hands = env._ycb02_hand_body_paths()
    if physics_view is None or bottle is None or table is None or not hands:
        env._add_task_state_warning("GEN1 bottle/table/hand contact paths are unavailable")
        return
    views = {}
    for role, filters in (("table", [str(table.GetPath())]), ("hand", hands)):
        view = physics_view.create_rigid_contact_view(
            str(bottle.GetPath()), filter_patterns=filters,
            max_contact_data_count=max(64, len(filters) * 8),
        )
        if view is None or int(view.sensor_count) < 1 or int(view.filter_count) < 1:
            env._add_task_state_warning(f"GEN1 {role} contact view is empty")
            return
        views[role] = view
    env._gen1_contact_views = views
    print("[HandsOn/IsaacLab] GEN1 physical table/hand contact views ready", flush=True)


def current_metrics(env) -> dict[str, float | bool]:
    forces = {}
    views = getattr(env, "_gen1_contact_views", {})
    for role in ("table", "hand"):
        view = views.get(role)
        if view is None:
            continue
        try:
            matrix = view.get_contact_force_matrix(env._sim_dt)
            if hasattr(matrix, "detach"):
                matrix = matrix.detach().cpu().numpy()
            matrix = np.asarray(matrix, dtype=np.float64)
            # PhysX returns [one bottle, filtered bodies, XYZ]. An empty or
            # malformed readback is unknown contact, not a measured zero force.
            if (
                matrix.ndim != 3 or matrix.shape[0] != 1
                or matrix.shape[1] < 1 or matrix.shape[2] != 3
                or not np.all(np.isfinite(matrix))
            ):
                continue
            # Sum magnitudes per filtered body: opposite fingers must not cancel.
            with np.errstate(over="ignore", invalid="ignore"):
                force = float(np.linalg.norm(matrix, axis=-1).sum())
            if np.isfinite(force):
                forces[role] = force
        except Exception:
            # A transient tensor/readback failure clears the success window.
            # It must not terminate an otherwise recoverable manipulation.
            continue
    return success_metrics(
        env._pose_for_role("target"), geometry(), env._current_table_top_z(),
        contacts_available=set(forces) == {"table", "hand"},
        table_force_n=forces.get("table", 0.0),
        hand_force_n=forces.get("hand", 0.0),
        reset_pose_wxyz=getattr(env, "_gen1_sampled_reset_pose", None),
        require_movement=getattr(env, "_gen1_sampled_reset_pose", None) is not None,
    )
