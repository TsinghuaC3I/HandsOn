"""Asset-dependent GEN7 scene geometry; C04 owns the trigger state machine."""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

import numpy as np

TASK_ID = "GEN7"
TASK_NAME = "eval_gen06_spray_shoe"
DISPLAY_NAME = "Spray Shoe Care"
SHOE_NAME = "gen7_shoe"
ASSET_DIR = Path(__file__).resolve().parents[1] / "envs/usd/eval_gen06_spray_shoe"
BOTTLE_XY = np.asarray([-.085, -.22])
SHOE_XY = np.asarray([-.26, .12])
XY_JITTER = .035
YAW_DEGREES = 25.0
PROMPT = (
    "Grasp and aim the spray bottle at the shoe upper before squeezing the trigger. "
    "Keep spraying the shoe to complete the task; a spray that entirely misses the shoe upper fails the attempt."
)
CONDITIONS = (
    "Grasp the bottle with the right hand before pulling the trigger.",
    "Squeeze the trigger with a finger while aiming at the shoe upper.",
    "Keep the spray on the shoe upper for 0.4 seconds.",
    "Spraying only the sole, table, or empty space fails the attempt.",
)
PHASES = ("idle", "armed", "unarmed", "pressing", "pressed")
HIDDEN_ATTRIBUTES = (
    "_c04_trigger_cycle_latched", "_c04_trigger_contact_seen", "_c04_grasp_seen",
    "_c04_trigger_contact_grace", "_c04_grasp_contact_grace", "_c04_grasp_ready_steps",
    "_c04_trigger_armed", "_c04_trigger_pulled", "_c04_water_emitting",
    "_c04_failed_trigger_outside", "_c04_spray_visible", "_success_counter",
)
HIDDEN_SIZE = 1 + len(HIDDEN_ATTRIBUTES)
RECORDING_LAYOUTS = {
    "gen7_hidden_state": ("trigger_phase",) + tuple(name.removeprefix("_c04_").removeprefix("_") for name in HIDDEN_ATTRIBUTES),
    "gen7_reset_parameters": ("bottle_x_m", "bottle_y_m", "bottle_yaw_deg", "shoe_x_m", "shoe_y_m", "shoe_yaw_deg"),
}
RECORDING_ATOLS = {"gen7_hidden_state": 0.0, "gen7_reset_parameters": 1e-9}


def recording_schema(schema):
    if schema is None:
        return None
    result = dict(schema)
    fields = dict(result["fields"])
    for name, layout in RECORDING_LAYOUTS.items():
        fields[name] = {
            "shape": [len(layout)], "dtype": "float64",
            "source": "gen7_action_aligned_task_state", "layout": list(layout),
            "visibility": "hidden_task_logic", "replay_domain": "task_state",
            "physical_replay": True, "replay_restore_supported": False,
            "replay_atol": RECORDING_ATOLS[name],
        }
    result["fields"] = fields
    result["task_state"] = {
        "task_name": TASK_NAME, "field_names": list(RECORDING_LAYOUTS),
        "hidden_field_names": list(RECORDING_LAYOUTS),
        "capture_mode": "action_aligned_current_and_next_observation",
        "replay_policy": "compare_every_step_never_restore",
    }
    return result


@lru_cache(maxsize=1)
def asset_manifest():
    return json.loads((ASSET_DIR / "asset_manifest.json").read_text())


@lru_cache(maxsize=1)
def upper_triangles():
    with np.load(ASSET_DIR / "shoe_target.npz") as data:
        triangles = data["triangles"].astype(np.float64)
    triangles.setflags(write=False)
    return triangles


def ray_triangle_distance(origin, direction, triangles):
    """Closest positive intersection with actual source triangles, in meters."""
    triangles = np.asarray(triangles, dtype=np.float64)
    a = triangles[:, 0]
    edge1 = triangles[:, 1] - a
    edge2 = triangles[:, 2] - a
    cross = np.cross(np.broadcast_to(direction, edge2.shape), edge2)
    determinant = np.einsum("ij,ij->i", edge1, cross)
    valid = np.abs(determinant) > 1e-12
    inverse = np.zeros_like(determinant)
    np.divide(1.0, determinant, out=inverse, where=valid)
    offset = np.asarray(origin) - a
    u = np.einsum("ij,ij->i", offset, cross) * inverse
    q = np.cross(offset, edge1)
    v = np.einsum("j,ij->i", direction, q) * inverse
    t = np.einsum("ij,ij->i", edge2, q) * inverse
    valid &= (u >= -1e-9) & (v >= -1e-9) & (u+v <= 1.0+1e-9) & (t > 1e-6)
    return float(np.min(t[valid])) if np.any(valid) else float("inf")


def spray_aim_metrics(env):
    bottle = env._pose_for_role("target")
    shoe = env._pose_for_role("receptacle")
    # Pose helpers in ABC use [xyz, quaternion wxyz] throughout.
    from .bimanual_assembly_env import _quat_wxyz_to_matrix
    bottle_rotation = _quat_wxyz_to_matrix(bottle[3:7])
    shoe_rotation = _quat_wxyz_to_matrix(shoe[3:7])
    origin = bottle[:3] + bottle_rotation @ np.asarray(asset_manifest()["nozzle_local"])
    forward = bottle_rotation @ np.asarray(asset_manifest()["nozzle_forward_local"])
    query = getattr(env, "_gen7_raycast_closest", None)
    geometry = None
    if query is None and getattr(env, "_stage", None) is not None:
        from .gen7_physics_geometry import PhysicsGeometry
        geometry = getattr(env, "_gen7_physics_geometry", None)
        if geometry is None:
            geometry = env._gen7_physics_geometry = PhysicsGeometry(env)
    hits = 0
    rays = env._c04_spray_sample_directions(forward)
    for direction in rays:
        local_origin = shoe_rotation.T @ (origin-shoe[:3])
        local_direction = shoe_rotation.T @ direction
        distance = ray_triangle_distance(local_origin, local_direction, upper_triangles())
        if not np.isfinite(distance):
            continue
        # A ray that misses the upper cannot succeed. Read all moving
        # occluder poses only when an upper intersection needs checking.
        if query is None and geometry is not None:
            query = geometry.prepare(env).raycast_closest
        if query is None:
            continue
        hit = query(tuple(origin), tuple(direction), distance + .003)
        # Actual closest collision must be the shoe, at the source upper
        # surface. Hands, the bottle, the table and the sole block a ray.
        path = str(hit.get("rigidBody", "")) + "/" + str(hit.get("collision", ""))
        if (hit.get("hit", False) and SHOE_NAME in path
                and abs(float(hit.get("distance", float("inf")))-distance) <= .002):
            hits += 1
    fraction = hits / len(rays)
    aimed = fraction >= 1.0 / 9.0
    return {
        "aimed": aimed,
        "gen7_spray_hits_upper": aimed,
        "gen7_spray_hit_count": hits,
        "gen7_spray_sample_count": len(rays),
        "gen7_spray_hit_fraction": fraction,
        "gen7_occlusion_query_available": query is not None or geometry is not None,
        # Explicit legacy target-role alias used by C04's shared metrics.
        "spray_hits_soil_or_plant": aimed,
    }


def footprint(bounds, xy, yaw):
    bounds = np.asarray(bounds)
    points = np.asarray([[x,y] for x in bounds[:,0] for y in bounds[:,1]])
    angle = np.deg2rad(yaw)
    rotation = np.asarray([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    return points @ rotation.T + xy


def layout_margin(env, xy, yaw):
    """Conservative reset sampling; the final live shape audit remains required."""
    manifest = asset_manifest()
    bounds = {}
    for name, key in (("spray", "bottle_bounds"), (SHOE_NAME, "shoe_bounds")):
        points = footprint(manifest[key], xy[name], yaw[name])
        bounds[name] = np.asarray([points.min(0), points.max(0)])
    a, b = bounds["spray"], bounds[SHOE_NAME]
    separation = np.maximum(np.maximum(a[0]-b[1], b[0]-a[1]), 0)
    margins = [float(np.linalg.norm(separation))-.025]
    points = (*env._reset_robot_body_points("right"), *env._reset_robot_body_points("left"))
    for name, box in bounds.items():
        margins.extend([float(box[0,0]+.55)-.02, float(.25-box[1,0])-.02,
                        float(box[0,1]+.8)-.02, float(.8-box[1,1])-.02])
        max_height = manifest["bottle_bounds" if name=="spray" else "shoe_bounds"][1][2]
        for point in points:
            point = np.asarray(point)
            if point[2] > .89+.02+max_height+.035:
                continue
            delta = np.maximum(np.maximum(box[0]-point[:2], point[:2]-box[1]),0)
            margins.append(float(np.linalg.norm(delta))-.030)
    return min(margins)


def sample_layout(env):
    for _ in range(256 if env.randomize else 1):
        xy = {"spray": BOTTLE_XY.copy(), SHOE_NAME: SHOE_XY.copy()}
        yaw = {"spray": 0.0, SHOE_NAME: 0.0}
        if env.randomize:
            for name in xy:
                xy[name] += env._random.uniform(-XY_JITTER, XY_JITTER, size=2)
                yaw[name] = float(env._random.uniform(-YAW_DEGREES, YAW_DEGREES))
        if layout_margin(env, xy, yaw) >= 0:
            env._gen7_reset_yaws = yaw
            env._gen7_reset_parameters = np.asarray([*xy["spray"], yaw["spray"], *xy[SHOE_NAME], yaw[SHOE_NAME]], dtype=np.float64)
            return xy, yaw["spray"]
    raise RuntimeError("GEN7 could not sample a supported layout with full robot clearance.")


def spray_visual_prim(stage, root_path="/World/Task"):
    from pxr import Usd, UsdGeom
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
        if prim.GetName() == "gen7_spray_cone" and UsdGeom.Imageable(prim):
            return prim
    return stage.GetPrimAtPath(root_path + "/missing_gen7_cone_visual")


def configure_asset_stage(stage, root_path):
    from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade
    root = stage.GetPrimAtPath(root_path)
    # MJCF mesh instances hide their actual collider children from the normal
    # USD traversal. Make only GEN7 collider instances editable before binding
    # the measured shell and the shoe's source triangle representation.
    while True:
        instances = [prim for prim in Usd.PrimRange(root) if prim.IsInstance() and (
            str(prim.GetPath()).startswith(root_path + "/link_2/")
            or (str(prim.GetPath()).startswith(root_path + "/gen7_shoe/")
                and "/collisions" in str(prim.GetPath())))]
        if not instances:
            break
        for prim in instances:
            prim.SetInstanceable(False)
    material = UsdShade.Material.Define(stage, root_path + "/GEN7PhysicsMaterial")
    physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_material.CreateStaticFrictionAttr(.9)
    physics_material.CreateDynamicFrictionAttr(.9)
    physics_material.CreateRestitutionAttr(0.)
    bottle = stage.GetPrimAtPath(root_path + "/link_2/link_2")
    if bottle.IsValid():
        rigid = PhysxSchema.PhysxRigidBodyAPI.Apply(bottle)
        rigid.CreateLinearDampingAttr(.4)
        rigid.CreateAngularDampingAttr(2.)
        rigid.CreateSolverPositionIterationCountAttr(32)
        rigid.CreateSolverVelocityIterationCountAttr(4)
        rigid.CreateEnableCCDAttr(True)
    for articulation in (bottle,stage.GetPrimAtPath(root_path + "/link_2")):
        if articulation.IsValid() and articulation.HasAPI(UsdPhysics.ArticulationRootAPI):
            solver = PhysxSchema.PhysxArticulationAPI.Apply(articulation)
            solver.CreateSolverPositionIterationCountAttr(32)
            solver.CreateSolverVelocityIterationCountAttr(4)
    for prim in Usd.PrimRange(root):
        name = prim.GetName()
        if name == "gen7_spray_cone" and UsdGeom.Imageable(prim):
            UsdGeom.Imageable(prim).MakeInvisible()
        if "gen7_" in name and "collision" in name:
            imageable = UsdGeom.Imageable(prim)
            if imageable:
                imageable.MakeInvisible()
        if "gen7_shoe_collision" in name and prim.IsA(UsdGeom.Mesh):
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set("none")
        if "gen7_" in name and "collision" in name and prim.HasAPI(UsdPhysics.CollisionAPI):
            collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            collision.CreateContactOffsetAttr(.0004)
            collision.CreateRestOffsetAttr(0.)
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose="physics")


def condition_rows(env):
    metrics = env._last_c04_success_metrics
    return [
        env._condition_row("gen7_grasp", "Bottle grasp established", env._c04_hand_gripping_spray()),
        env._condition_row("gen7_trigger", "Trigger physically pressed", bool(metrics.get("spray_trigger_contact_ok", False))),
        env._condition_row("gen7_emitting", "Spray active", env._c04_water_emitting),
        env._condition_row("gen7_target", "Spray reaches the shoe upper", bool(metrics.get("gen7_spray_hits_upper", False))),
        env._condition_row("gen7_hold", "Valid spray held for 0.4 seconds", env._success_counter >= 12),
    ]


def hidden_state(env):
    return np.asarray([PHASES.index(env._c04_trigger_phase)] +
                      [float(getattr(env, key)) for key in HIDDEN_ATTRIBUTES], dtype=np.float64)


def restore_hidden_state(env, state_parts):
    values = np.asarray(state_parts.get("gen7_hidden_state", []), dtype=np.float64).ravel()
    if not values.size:
        return False
    if values.shape != (HIDDEN_SIZE,) or not np.all(np.isfinite(values)):
        raise ValueError("Invalid recorded GEN7 trigger state.")
    phase = int(values[0])
    if not 0 <= phase < len(PHASES):
        raise ValueError("Invalid recorded GEN7 trigger phase.")
    env._c04_trigger_phase = PHASES[phase]
    for key, value in zip(HIDDEN_ATTRIBUTES, values[1:]):
        current = getattr(env, key)
        setattr(env, key, bool(value) if isinstance(current, bool) else int(value))
    if getattr(env, "_stage", None) is not None:
        visible = env._c04_spray_visible
        env._c04_spray_visible = not visible
        env._set_c04_spray_visible(visible)
    env._gen7_reset_parameters = np.asarray(state_parts.get("gen7_reset_parameters", np.zeros(6)), dtype=np.float64).copy()
    return True
