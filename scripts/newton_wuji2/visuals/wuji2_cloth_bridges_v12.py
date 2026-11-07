"""Thin knit joint folds driven by the recorded Wuji2 link transforms.

The sleeves are authored display geometry, not a cloth solver. Native rigid
poses, collision geometry and physical material values remain untouched.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
from scipy.spatial.transform import Rotation

import wuji2_cloth_bridges_v10 as previous

sha = previous.sha
transform_points = previous.transform_points
smooth_normals = previous.smooth_normals


def rigid_bend(joint, parent_world, child_world):
    """Original smooth bend interpolation without its old periodic waves."""
    bind = np.asarray(joint['bind'], dtype=float)
    pivot = np.asarray(joint['pivot'], dtype=float)
    points = np.asarray(joint['points'], dtype=float)
    relative = parent_world[:3, :3].T @ child_world[:3, :3] @ bind[:3, :3].T
    delta = Rotation.from_matrix(relative).as_rotvec()
    rotations = Rotation.from_rotvec(np.asarray(joint['weights'])[:, None] * delta)
    moved = rotations.apply(points - pivot) + pivot
    expected_pivot = transform_points(pivot[None, :], parent_world)[0]
    if np.linalg.norm(expected_pivot - child_world[:3, 3]) > .00015:
        raise RuntimeError(f'Joint pose mismatch for {joint["child"]}')
    return moved, rotations, delta


def fold_displacement(joint, delta):
    """Irregular, localized crests on the compressed half of a thin sleeve.

    Three different widths/offsets replace circumferential sine rings. Their
    crests bend across the tube and vanish on the stretched side and both end
    rings. Maximum added displacement is bounded independently of pose.
    """
    u = np.asarray(joint['u'], dtype=float)
    angle = np.asarray(joint['angles'], dtype=float)
    directions = np.asarray(joint['directions'], dtype=float)
    phase = float(joint.get('soft_fold_phase', joint.get('wrinkle_phase', 0.)))
    envelope = np.sin(np.pi * u) ** 2
    # Force exact zero at attachment rings despite floating-point sin(pi).
    envelope[(u <= 0.) | (u >= 1.)] = 0.
    bend = float(np.linalg.norm(delta))
    if bend <= 1e-10:
        return np.zeros_like(u)
    inward = np.cross(delta / bend, np.array([0., 0., -1.]))
    compression = np.maximum(directions @ inward, 0.) ** 1.35
    activation = np.clip(bend / .95, 0., 1.)

    field = np.zeros_like(u)
    for center, width, strength, tilt in (
        (.285, .067, .74, .050),
        (.505, .088, 1.00, -.037),
        (.735, .059, .66, .032),
    ):
        offset = tilt * np.sin(angle + phase + center * 4.)
        offset += .018 * np.sin(2. * angle - phase)
        distance = (u - center - offset) / width
        crest = np.exp(-.5 * distance * distance)
        # A shallow adjacent trough makes a fold, without sinking deeply into
        # the existing rigid-link glove or exposing its underlying metal.
        trough = .18 * np.exp(-.5 * ((distance + 1.5) / 1.15) ** 2)
        angular = .65 + .35 * np.cos(angle - phase + center * 7.)
        field += strength * angular * (crest - trough)
    field = np.clip(field, -.14, 1.)
    maximum = float(joint.get('soft_fold_max_m', .00065))
    return maximum * activation * compression * envelope * field


def deform(joint, parent_world, child_world):
    moved, rotations, delta = rigid_bend(joint, parent_world, child_world)
    displacement = fold_displacement(joint, delta)
    radial = rotations.apply(np.asarray(joint['directions'], dtype=float))
    offset = radial * displacement[:, None]
    moved += offset * safe_fold_scale(moved, offset, np.asarray(joint['faces']))
    return transform_points(moved, parent_world)


def safe_fold_scale(baseline, offset, faces):
    """Continuous conservative scale before any fold can reverse a triangle.

    On a highly compressed terminal sleeve, its existing triangles can become
    very small. The displaced triangle cross product is quadratic in amplitude.
    Its first positive root against 35% of the original oriented area gives an
    upper bound. A smooth minimum avoids discrete half-amplitude jumps in motion.
    This prevents *added* fold reversals, not pre-existing sleeve intersections.
    """
    triangle, delta = baseline[faces], offset[faces]
    edge1, edge2 = triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0]
    step1, step2 = delta[:, 1] - delta[:, 0], delta[:, 2] - delta[:, 0]
    base = np.cross(edge1, edge2)
    linear = np.cross(step1, edge2) + np.cross(edge1, step2)
    quadratic = np.cross(step1, step2)
    a = np.einsum('ij,ij->i', quadratic, base)
    b = np.einsum('ij,ij->i', linear, base)
    c = .65 * np.einsum('ij,ij->i', base, base)
    roots = np.full((len(faces), 2), np.inf)
    nonlinear = np.abs(a) > 1e-10 * np.maximum(np.abs(b), c)
    discriminant = b * b - 4 * a * c
    real = nonlinear & (discriminant >= 0)
    sqrt = np.sqrt(np.maximum(discriminant[real], 0))
    roots[real, 0] = (-b[real] - sqrt) / (2 * a[real])
    roots[real, 1] = (-b[real] + sqrt) / (2 * a[real])
    falling_line = ~nonlinear & (b < 0)
    roots[falling_line, 0] = -c[falling_line] / b[falling_line]
    roots[roots <= 0] = np.inf
    boundary = float(roots.min())
    if not np.isfinite(boundary):
        return 1.
    # Smooth min(1,boundary), always strictly below the orientation boundary.
    return float((1. + boundary ** -4) ** -.25)


class JointCloth(previous.JointCloth):
    def __init__(self, stage, profile_path, side=None):
        super().__init__(stage, profile_path, side=side)
        # A separately authored material helper keeps sleeves and rigid-link
        # fabric consistent; no scene/exposure/camera mutation is permitted.
        from soft_glove_material_v12 import attach_fabric_mdl
        self.material_report = attach_fabric_mdl(
            stage, self.root + '/Knit', Path(self.profile['bundle']),
            'wuji2_knit_detail_color.png', 'wuji2_knit_detail_normal.png')

    def update(self, poses):
        from pxr import Gf
        for mesh, joint in zip(self.meshes, self.profile['joints']):
            points = deform(joint, poses[joint['parent']], poses[joint['child']])
            normals = smooth_normals(points, np.asarray(joint['faces']))
            mesh.GetPointsAttr().Set([Gf.Vec3f(*x) for x in points])
            mesh.GetNormalsAttr().Set([Gf.Vec3f(*x) for x in normals])
            mesh.GetExtentAttr().Set([Gf.Vec3f(*points.min(axis=0)), Gf.Vec3f(*points.max(axis=0))])
        self.updates += 1


def install(env, profile_path, report_path=None):
    """Use native articulated world poses and update only visual mesh points."""
    from pxr import Usd
    cloth = JointCloth(env._stage, profile_path)
    indexes = {}
    for side, robot in env._robots.items():
        indexes[side] = {str(name): i for i, name in enumerate(robot.body_names)}
    for joint in cloth.profile['joints']:
        for name in (joint['parent'], joint['child']):
            if name not in indexes[joint['side']]:
                raise RuntimeError(f'Native cloth body is missing: {name}')

    def update():
        poses = {}
        for side, robot in env._robots.items():
            source = robot.data.body_pose_w[0].detach().cpu().numpy()
            for name, index in indexes[side].items():
                if '_finger' not in name:
                    continue
                value = source[index]
                quat = value[3:7] if env._isaaclab_uses_xyzw else value[[4, 5, 6, 3]]
                matrix = np.eye(4)
                matrix[:3, :3] = Rotation.from_quat(quat).as_matrix()
                matrix[:3, 3] = value[:3]
                poses[name] = matrix
        cloth.update(poses)

    native_render = env.render

    def render(*args, **kwargs):
        update()
        return native_render(*args, **kwargs)

    update()
    env.render = render
    env._wuji2_joint_cloth_candidate = cloth
    assert not any('Physics' in schema or 'Physx' in schema
                   for prim in Usd.PrimRange(env._stage.GetPrimAtPath(cloth.root))
                   for schema in prim.GetAppliedSchemas())
    if report_path:
        Path(report_path).write_text(json.dumps({
            'profile_sha256': sha(profile_path), 'module_sha256': sha(__file__),
            'sleeve_count': len(cloth.meshes), 'candidate': True,
            'first_update_passed': True, 'physics_schemas': [],
            'deformation': 'localized_irregular_compression_folds_v12',
            'material': cloth.material_report,
            'fold_displacement_bound_m': .00065,
        }, indent=2) + '\n')
    return cloth
