"""GEN2 asset-derived geometry and task-local ABC integration.

The visible surfaces are unchanged D15 components. Success thresholds and
movement/reset latches are inherited from B05 in abc_collection_env.
"""
from pathlib import Path
import math
import xml.etree.ElementTree as ET
import numpy as np

TASK_NAME = 'eval_gen01_cover_lunchbox'
BOX = 'gen2_lunchbox'
LID = 'gen2_lid'
BODY_NAMES = (BOX, LID)
ASSET_PATH = Path(__file__).resolve().parents[2] / 'sim/envs/xmls/eval_gen01_cover_lunchbox_objects.xml'
_SOURCE = ET.parse(ASSET_PATH).getroot()
BOUNDS = {}
INERTIALS = {}
for _name in BODY_NAMES:
    _body = _SOURCE.find(f"./worldbody/body[@name='{_name}']")
    _parts = [
        (np.fromstring(g.get('pos', '0 0 0'), sep=' '), np.fromstring(g.get('size'), sep=' '))
        for g in _body.findall('geom')
    ]
    BOUNDS[_name] = np.array([
        np.min([p-h for p,h in _parts], axis=0),
        np.max([p+h for p,h in _parts], axis=0),
    ])
    _inertial = _body.find('inertial')
    INERTIALS[_name] = dict(
        mass=float(_inertial.get('mass')),
        com=np.fromstring(_inertial.get('pos'), sep=' '),
        diagonal=np.fromstring(_inertial.get('diaginertia'), sep=' '),
        collider_count=len(_parts),
    )
HALF_EXTENTS = {n:(b[1]-b[0])/2 for n,b in BOUNDS.items()}
CENTRES = {n:b.mean(axis=0) for n,b in BOUNDS.items()}
SUPPORT_MIN_Z = {n:float(b[0,2]) for n,b in BOUNDS.items()}
GOAL_LOCAL_Z = float(BOUNDS[BOX][1,2] - BOUNDS[LID][0,2])
DEFAULT_XY = {BOX:np.array([-.20,.155]), LID:np.array([-.275,-.18])}
RESET_XY_BOUNDS = {n:np.array([xy-.025,xy+.025]) for n,xy in DEFAULT_XY.items()}
YAW_BOUNDS_DEG = (-5., 5.)
CONTACT_OFFSET = .001
_YAW = math.radians(5.)
SWEPT_XY = {n:np.array([h[0]*math.cos(_YAW)+h[1]*math.sin(_YAW),
                         h[1]*math.cos(_YAW)+h[0]*math.sin(_YAW)])
            for n,h in HALF_EXTENTS.items()}
CONDITIONS = (
    ('full_coverage', 'Lid covers more than 90% of the lunchbox'),
    ('height_close', 'Lid is at the seated height'),
    ('top_aligned', 'Lid is level with the lunchbox'),
    ('yaw_aligned', 'Lid orientation covers the lunchbox'),
    ('moved', 'Lid has been lifted or moved from its start'),
    ('lid_stable', 'Lid is stable'),
    ('base_stable', 'Lunchbox is stable'),
    ('base_upright', 'Lunchbox is upright'),
    ('base_on_table', 'Lunchbox rests on the table'),
    ('hand_released', 'Right hand is clear of the lid and lunchbox'),
    ('release_stable_hold', 'Lid and box stable and released for 0.5 seconds'),
)
HIDDEN_FIELDS = (
    "lid_start_valid", "lid_start_x", "lid_start_y", "lid_start_z",
    "lid_start_qw", "lid_start_qx", "lid_start_qy", "lid_start_qz",
    "lid_was_moved", "success_counter",
)


def reset_layout_margin(env, positions, hand_points):
    """Conservative source OBB pre-screen; live shape queries remain mandatory."""
    from ...sim.isaaclab import abc_collection_env as abc
    margin = math.inf
    for name, xy in positions.items():
        h = np.array([*SWEPT_XY[name], HALF_EXTENTS[name][2]])
        centre = np.array([*xy, abc._TABLE_TOP_Z + float(env.delta_h[0])
                           - SUPPORT_MIN_Z[name] + CONTACT_OFFSET + CENTRES[name][2]])
        # The full footprint must fit the source table, not just its origin.
        margin = min(margin, float(np.min(np.r_[xy-h[:2]-[-.55,-.8],
                                                 [.25,.8]-xy-h[:2]])))
        for point in hand_points:
            offset = np.abs(np.asarray(point)-centre)-h
            distance = (np.linalg.norm(np.maximum(offset,0))
                        if np.any(offset>0) else np.max(offset))
            margin = min(margin, float(distance)-abc._B05_OBJECT_HAND_CLEARANCE_MARGIN)
    if BOX in positions and LID in positions:
        gaps = np.abs(positions[BOX]-positions[LID])-SWEPT_XY[BOX]-SWEPT_XY[LID]
        margin = min(margin, float(np.max(gaps))-abc._B05_OBJECT_CLEARANCE_MARGIN)
    return float(margin)


def hand_clearance_metrics(env, lid, box):
    from ...sim.isaaclab import abc_collection_env as abc
    minimum = math.inf
    points = tuple(env._a06_hand_points('right'))
    for point in points:
        for name, pose in ((LID,lid),(BOX,box)):
            local = abc._invert_pose_matrix(abc._pose_vector_to_matrix(pose)) @ np.r_[point,1.]
            outside = np.maximum(np.abs(local[:3]-CENTRES[name])-HALF_EXTENTS[name],0.)
            minimum = min(minimum,float(np.linalg.norm(outside)))
    released = bool(points and np.isfinite(minimum) and minimum >= abc._B05_HAND_CLEARANCE_DISTANCE)
    return minimum, released, len(points)


def _coverage_metrics(lid_pose, box_pose):
    """Compute oriented XY coverage from the imported lunchbox/lid bounds."""
    from ...sim.isaaclab import abc_collection_env as abc

    lid = abc._pose_vector_to_matrix(np.asarray(lid_pose, dtype=np.float64).reshape(7))
    box = abc._pose_vector_to_matrix(np.asarray(box_pose, dtype=np.float64).reshape(7))
    box_half = HALF_EXTENTS[BOX][:2]
    lid_half = HALF_EXTENTS[LID][:2]
    corners = np.asarray([
        [-box_half[0], -box_half[1], 0.0, 1.0],
        [ box_half[0], -box_half[1], 0.0, 1.0],
        [ box_half[0],  box_half[1], 0.0, 1.0],
        [-box_half[0],  box_half[1], 0.0, 1.0],
    ])
    corners_in_lid = (abc._invert_pose_matrix(lid) @ (box @ corners.T)).T[:, :2]

    def clip_axis(polygon, axis, boundary, keep_greater):
        if polygon.size == 0:
            return polygon.reshape(0, 2)
        output = []
        previous = polygon[-1]
        previous_inside = previous[axis] >= boundary if keep_greater else previous[axis] <= boundary
        for current in polygon:
            current_inside = current[axis] >= boundary if keep_greater else current[axis] <= boundary
            if current_inside != previous_inside:
                denominator = current[axis] - previous[axis]
                if abs(float(denominator)) > 1e-15:
                    fraction = (boundary - previous[axis]) / denominator
                    output.append(previous + fraction * (current - previous))
            if current_inside:
                output.append(current)
            previous, previous_inside = current, current_inside
        return np.asarray(output, dtype=np.float64).reshape(-1, 2)

    overlap = corners_in_lid
    overlap = clip_axis(overlap, 0, -float(lid_half[0]), True)
    overlap = clip_axis(overlap, 0, float(lid_half[0]), False)
    overlap = clip_axis(overlap, 1, -float(lid_half[1]), True)
    overlap = clip_axis(overlap, 1, float(lid_half[1]), False)
    if len(overlap) >= 3:
        overlap_area = 0.5 * abs(float(
            np.dot(overlap[:, 0], np.roll(overlap[:, 1], -1))
            - np.dot(overlap[:, 1], np.roll(overlap[:, 0], -1))
        ))
    else:
        overlap_area = 0.0
    box_area = float(4.0 * box_half[0] * box_half[1])
    ratio = float(np.clip(overlap_area / box_area, 0.0, 1.0))
    return ratio, bool(ratio > 0.90)


def success_metrics(env):
    """GEN2's live completion predicate using the source asset geometry.

    GEN2 is similar to B05, but its body names and dimensions differ.  Calling
    the B05 predicate directly therefore always measured the wrong pair and,
    before this branch was added, GEN2 never entered a success predicate at all.
    """
    from ...sim.isaaclab import abc_collection_env as abc

    lid = env._pose_for_role("target")
    box = env._pose_for_role("receptacle")
    local = abc._invert_pose_matrix(abc._pose_vector_to_matrix(box)) @ np.r_[lid[:3], 1.0]
    coverage_ratio, full_coverage = _coverage_metrics(lid, box)
    height_error = float(local[2] - GOAL_LOCAL_Z)
    height_close = bool(-0.020 <= height_error <= 0.025)
    top_aligned = bool(env._z_axis_alignment(lid, box) >= math.cos(math.radians(6.0)))
    yaw_aligned = bool(full_coverage)

    if env._lid_start_pose is None:
        env._lid_start_pose = np.asarray(lid, dtype=np.float64).copy()
    move_distance = float(np.linalg.norm(np.asarray(lid[:3]) - env._lid_start_pose[:3]))
    lift_height = float(lid[2] - env._lid_start_pose[2])
    if move_distance >= abc._LID_MIN_MOVE_DISTANCE or lift_height >= abc._LID_MIN_LIFT_HEIGHT:
        env._lid_was_moved = True

    def stable(name):
        velocity = env._physics_body_velocity(name)
        if velocity is None:
            return False
        values = np.asarray(velocity, dtype=np.float64).reshape(-1)
        return bool(
            values.size >= 6 and np.all(np.isfinite(values[:6]))
            and np.linalg.norm(values[:3]) <= abc._LID_STABLE_LINEAR_SPEED
            and np.linalg.norm(values[3:6]) <= abc._LID_STABLE_ANGULAR_SPEED
        )

    lid_stable = stable(LID)
    box_stable = stable(BOX)
    box_upright = bool(env._z_axis_alignment(box) >= abc._B05_BASE_UPRIGHT_MIN_DOT)
    expected_box_z = env._current_table_top_z() - SUPPORT_MIN_Z[BOX] + CONTACT_OFFSET
    box_on_table = bool(abs(float(box[2]) - expected_box_z) <= abc._B05_BASE_TABLE_HEIGHT_TOLERANCE)
    hand_min, hand_released, hand_count = hand_clearance_metrics(env, lid, box)
    success = bool(
        env._lid_was_moved and full_coverage and height_close and top_aligned
        and yaw_aligned and lid_stable and box_stable and box_upright
        and box_on_table and hand_released
    )
    return {
        "b05_full_coverage": full_coverage,
        "b05_coverage_ratio": coverage_ratio,
        "b05_height_close": height_close,
        "b05_height_error_m": height_error,
        "b05_top_aligned": top_aligned,
        "b05_yaw_aligned": yaw_aligned,
        "b05_moved": bool(env._lid_was_moved),
        "b05_lid_stable": lid_stable,
        "b05_base_stable": box_stable,
        "b05_base_upright": box_upright,
        "b05_base_on_table": box_on_table,
        "b05_hand_released": hand_released,
        "b05_hand_min_distance": float(hand_min),
        "b05_hand_point_count": int(hand_count),
        "b05_success_now": success,
    }


def configure_physics(env, sim_utils):
    """Restore source inertials and make the lunchbox a stable fixture.

    The task only asks the operator to manipulate the independent lid.  The
    lunchbox is a supported receptacle, so leaving its imported free joint
    dynamic lets the first table contact spin the very light open shell before
    the recording-origin gate.  Treat the base as a resettable kinematic
    fixture while keeping the lid fully dynamic.
    """
    if env._stage is None:
        return 0
    from pxr import Gf, PhysxSchema, Usd, UsdPhysics
    count = 0
    for name in BODY_NAMES:
        prim = env._find_rigid_body_prim(name)
        if prim is None or not prim.IsValid():
            raise RuntimeError(f'GEN2 missing rigid body: {name}')
        path = str(prim.GetPath())
        sim_utils.make_uninstanceable(path, stage=env._stage)
        sim_utils.update_stage()
        prim = env._stage.GetPrimAtPath(path)
        colliders = [p for p in Usd.PrimRange(prim) if p.HasAPI(UsdPhysics.CollisionAPI)]
        inertial = INERTIALS[name]
        if len(colliders) != inertial['collider_count']:
            raise RuntimeError(f'GEN2 {name}: expected {inertial["collider_count"]} source colliders, found {len(colliders)}')
        # Resolve simultaneous support contacts with eight velocity iterations.
        # The imported one-link articulation opinion overrides body settings,
        # so configure both levels; keep all sleep/damping defaults unchanged.
        solvers = [PhysxSchema.PhysxRigidBodyAPI.Apply(prim)]
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            solvers.append(PhysxSchema.PhysxArticulationAPI.Apply(prim))
        for solver in solvers:
            solver.CreateSolverPositionIterationCountAttr().Set(64)
            solver.CreateSolverVelocityIterationCountAttr().Set(8)
        if name == BOX:
            # The lunchbox is the fixed support in this placement task.  The
            # MJCF freejoint is an importer artefact; remove its articulation
            # opinions before enabling kinematic mode so reset writes remain
            # authoritative and no solver impulse can rotate the base.
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            if prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
                prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
            rigid = UsdPhysics.RigidBodyAPI(prim)
            if not rigid:
                rigid = UsdPhysics.RigidBodyAPI.Apply(prim)
            kinematic = rigid.GetKinematicEnabledAttr()
            if not kinematic.IsValid():
                kinematic = rigid.CreateKinematicEnabledAttr()
            kinematic.Set(True)
        mass = UsdPhysics.MassAPI.Apply(prim)
        mass.CreateMassAttr().Set(inertial['mass'])
        mass.CreateCenterOfMassAttr().Set(Gf.Vec3f(*map(float,inertial['com'])))
        mass.CreatePrincipalAxesAttr().Set(Gf.Quatf(1.,Gf.Vec3f(0.)))
        mass.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*map(float,inertial['diagonal'])))
        for collider in colliders:
            api = PhysxSchema.PhysxCollisionAPI.Apply(collider)
            api.CreateContactOffsetAttr().Set(CONTACT_OFFSET)
            api.CreateRestOffsetAttr().Set(0.)
            count += 1
    return count


def success_rows(env):
    if not env._last_b05_lid_success_metrics:
        env._lid_fit_success()
    metrics = env._last_b05_lid_success_metrics
    rows = []
    for key, label in CONDITIONS:
        if key == 'release_stable_hold':
            ok = env._success_counter >= env._success_steps_required()
        else:
            ok = bool(metrics.get('b05_'+key, False))
        rows.append(env._condition_row('gen2_'+key, label, ok))
    return rows


def success_info(env):
    result = {key.replace('b05_', 'gen2_', 1):value
              for key,value in env._last_b05_lid_success_metrics.items()}
    result.update(gen2_success_counter=int(env._success_counter),
                  gen2_success_steps_required=int(env._success_steps_required()),
                  gen2_release_stable_seconds=.5)
    return result


def hidden_state(env):
    """Replay evidence only; never a policy input or a physical assist."""
    pose = env._lid_start_pose
    return np.r_[
        float(pose is not None),
        np.zeros(7) if pose is None else pose,
        float(env._lid_was_moved),
        float(env._success_counter),
    ].astype(np.float64)


def restore_hidden_state(env, value):
    state = np.asarray(value, dtype=np.float64)
    if state.shape != (len(HIDDEN_FIELDS),) or not np.all(np.isfinite(state)):
        raise ValueError("GEN2 hidden state must be a finite vector of length 10")
    if (
        state[0] not in (0.0, 1.0)
        or state[8] not in (0.0, 1.0)
        or state[9] < 0
        or state[9] != int(state[9])
    ):
        raise ValueError("GEN2 hidden state contains invalid flags or counter")
    if state[0] and not np.isclose(np.linalg.norm(state[4:8]), 1.0, atol=1e-5):
        raise ValueError("GEN2 recorded start quaternion must be normalized")
    env._lid_start_pose = state[1:8].copy() if state[0] else None
    env._lid_was_moved = bool(state[8])
    env._success_counter = int(state[9])
    env._last_b05_lid_success_metrics = {}
