"""Task-local Wuji contact hulls derived from the composed visible hand meshes.

Each visible component gets one GPU-compatible conservative convex collider.
The original robot geometry, joints, mass and inertia remain unchanged.
"""

from __future__ import annotations

import itertools

import numpy as np


def enclosing_contact_hull(points: np.ndarray):
    """Cover every source vertex with at most 32 vertices and 60 triangular faces.

Start with supporting planes in the component's principal frame. Refine
against its exact convex hull until the support error is below 0.5 mm or
the conservative GPU vertex/face budget is reached. This is an outer approximation,
so vertex reduction cannot expose parts of the visible fingertip.
"""
    from scipy.spatial import ConvexHull, HalfspaceIntersection

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Contact hull needs finite 3D source vertices")
    points = np.unique(np.round(points, 9), axis=0)
    center = points.mean(axis=0)
    local = points - center
    if len(points) < 4 or np.linalg.matrix_rank(local) < 3:
        raise ValueError("Hand collision component must have nonzero volume")
    exact = ConvexHull(local)
    _, axes = np.linalg.eigh(local.T @ local)
    directions = np.asarray(
        [p for p in itertools.product((-1.0, 0.0, 1.0), repeat=3)
         if np.count_nonzero(p) in (1,)]
    )
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions @ axes.T

    def polytope(normals):
        support = np.max(local @ normals.T, axis=0) + 5e-5
        vertices = HalfspaceIntersection(
            np.c_[normals, -support], np.zeros(3)
        ).intersections
        hull = ConvexHull(vertices)
        vertices = vertices[hull.vertices]
        return vertices, ConvexHull(vertices)

    vertices, hull = polytope(directions)
    for _ in range(40):
        errors = vertices @ exact.equations[:, :3].T + exact.equations[:, 3]
        support_error = float(errors.max())
        if support_error <= 0.0005:
            break
        plane = int(np.unravel_index(np.argmax(errors), errors.shape)[1])
        proposed = np.vstack((directions, exact.equations[plane, :3]))
        candidate, candidate_hull = polytope(proposed)
        if len(candidate) > 32 or len(candidate_hull.simplices) > 60:
            break
        directions, vertices, hull = proposed, candidate, candidate_hull
    if len(vertices) > 32 or len(hull.simplices) > 60:
        raise ValueError("Hand collider exceeds the GPU convex vertex budget")
    faces = hull.simplices.copy()
    for i, face in enumerate(faces):
        a, b, c = vertices[face]
        if np.dot(np.cross(b - a, c - a), hull.equations[i, :3]) < 0:
            faces[i, 1], faces[i, 2] = faces[i, 2], faces[i, 1]
    coverage_error = float(
        (local @ hull.equations[:, :3].T + hull.equations[:, 3]).max()
    )
    if coverage_error > 1e-7:
        raise ValueError("Reduced collider does not cover its visible mesh")
    return vertices + center, faces, {
        "source_vertices": len(points),
        "hull_vertices": len(vertices),
        "hull_triangles": len(faces),
        "source_coverage_error_m": coverage_error,
        "maximum_support_plane_error_m": float(
            (vertices @ exact.equations[:, :3].T + exact.equations[:, 3]).max()
        ),
    }


def author_hand_contact_hulls(stage, *, contact_offset: float, rest_offset: float):
    """Replace this task's coarse hand collision shapes before physics reset."""
    from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics, Vt

    cache = UsdGeom.XformCache()
    sources, previous = [], []
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        path = str(prim.GetPath()).lower()
        if not path.startswith("/world/handsonrobots/"):
            continue
        if not ("finger" in path or "palm" in path):
            continue
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            previous.append(prim)
            continue
        if not prim.IsA(UsdGeom.Mesh) or "/collisions/" in path:
            continue
        if UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        body = prim
        while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):
            body = body.GetParent()
        if not body:
            continue
        mesh = UsdGeom.Mesh(prim)
        transform = np.asarray(cache.GetLocalToWorldTransform(prim)) @ np.asarray(
            cache.GetLocalToWorldTransform(body).GetInverse()
        )
        points = np.asarray(mesh.GetPointsAttr().Get())
        local = (np.c_[points, np.ones(len(points))] @ transform)[:, :3]
        vertices, faces, audit = enclosing_contact_hull(local)
        sources.append((prim.GetPath(), body.GetPath(), vertices, faces, audit))
    if not sources:
        raise ValueError("No composed Wuji visual components found")
    for prim in previous:
        UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(False)
    audits = []
    for ordinal, (source, body, vertices, faces, audit) in enumerate(sources):
        path = str(body) + f"/flexibleContactHull_{ordinal:02d}"
        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices.astype(np.float32)))
        mesh.CreateFaceVertexCountsAttr([3] * len(faces))
        mesh.CreateFaceVertexIndicesAttr(faces.flatten().tolist())
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateVisibilityAttr("invisible")
        mesh.CreateExtentAttr(
            Vt.Vec3fArray.FromNumpy(
                np.array([vertices.min(0), vertices.max(0)], dtype=np.float32)
            )
        )
        collider = mesh.GetPrim()
        UsdPhysics.CollisionAPI.Apply(collider).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(collider).CreateApproximationAttr("convexHull")
        PhysxSchema.PhysxConvexHullCollisionAPI.Apply(collider).CreateHullVertexLimitAttr(32)
        physics = PhysxSchema.PhysxCollisionAPI.Apply(collider)
        physics.CreateContactOffsetAttr(contact_offset)
        physics.CreateRestOffsetAttr(rest_offset)
        collider.SetCustomDataByKey("handson:contactSource", str(source))
        audits.append({"collider": path, "source": str(source), **audit})
    return audits


def towel_hand_diagnostic(env, side: str) -> str:
    """Read-only geometry diagnostics for the existing two-second teleop log.

    Plane gaps and nearby nodes describe geometry, not measured contact forces
    or proof of a grasp. In particular, closed fingers can remain above a flat
    towel when the Cartesian table-clearance projection lifts the whole hand.
    """
    from pxr import UsdGeom
    from scipy.spatial import ConvexHull

    from . import flexible_cotton_towel as cotton
    from .b320_bimanual_scene import B320_TABLE_TOP_Z

    nodes = getattr(env, "_e_nodes_np", None)
    robot = env._robots.get(side)
    if robot is None or nodes is None or not np.isfinite(nodes).all():
        return "towel_geometry_unavailable"
    cache = getattr(env, "_flexible_diagnostic_hulls", None)
    if cache is None:
        cache = {"right": [], "left": []}
        for row in getattr(env, "_flexible_hand_collider_audit", ()):
            owner = "right" if "/RightRobot/" in row["collider"] else "left"
            prim = env._stage.GetPrimAtPath(row["collider"])
            name = prim.GetParent().GetName()
            index = env._robots[owner].body_names.index(name)
            points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get())
            finger = next(
                (f for f in range(1, 6) if f"finger{f}_link4" in name), None
            )
            planes = ConvexHull(points).equations if finger is not None else None
            cache[owner].append((index, points, finger, planes))
        env._flexible_diagnostic_hulls = cache
    gaps = {}
    lowest = math_inf = float("inf")
    for index, points, finger, planes in cache[side]:
        pose = env._robot_body_pose_matrix(robot, index)
        world = points @ pose[:3, :3].T + pose[:3, 3]
        lowest = min(lowest, float(world[:, 2].min()))
        if finger is not None:
            local = (nodes - pose[:3, 3]) @ pose[:3, :3]
            gap = (local @ planes[:, :3].T + planes[:, 3]).max(axis=1)
            gaps[finger] = np.minimum(gaps.get(finger, math_inf), gap)
    near = cotton.REST_OFFSET + cotton.HAND_REST_OFFSET + 0.0005
    thumb = gaps.get(1, np.full(len(nodes), math_inf))
    opposing = np.minimum.reduce(
        [gaps.get(f, np.full(len(nodes), math_inf)) for f in range(2, 6)]
    )
    between = int(np.count_nonzero((thumb <= near) & (opposing <= near)))
    joint_ids = env._hand_joint_ids[side]
    error = robot.data.joint_pos[0, joint_ids] - env._joint_targets[side][0, joint_ids]
    source = getattr(env, "_towel_source", None)
    revision = source.task_name + "-v1" if source else cotton.MATERIAL_REVISION
    return (
        f"material={revision},"
        f"hand_floor_mm={(lowest - B320_TABLE_TOP_Z) * 1000:.1f},"
        f"thumb_plane_gap_mm={float(thumb.min()) * 1000:.1f},"
        f"opposing_plane_gap_mm={float(opposing.min()) * 1000:.1f},"
        f"between_tip_nodes={between},"
        f"cloth_height_mm={(float(nodes[:, 2].max()) - B320_TABLE_TOP_Z) * 1000:.1f},"
        f"hand_q_error_rad={float(error.abs().max().item()):.3f}"
    )
