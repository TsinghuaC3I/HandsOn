"""Optional visual wrist leads with moving endpoints and a fixed cable length.

This is a geometric harness approximation, not a cable dynamics/contact model.
The near-plug shape stays in palm coordinates; the other endpoint follows the
forearm. No robot meshes, joints, colliders, or physical parameters are changed.
"""
from pathlib import Path
import hashlib
import json
import os

import numpy as np


def _transform(matrix, points):
    return np.einsum('ij,nj->ni', matrix[:3, :3], points) + matrix[:3, 3]


def _unit(vector):
    length = float(np.linalg.norm(vector))
    if not np.isfinite(length) or length < 1e-10:
        raise ValueError('Degenerate cable direction')
    return vector / length


def _hermite(a, b, ta, tb, t):
    return ((2*t**3-3*t**2+1)[:, None]*a + (t**3-2*t**2+t)[:, None]*ta
            + (-2*t**3+3*t**2)[:, None]*b + (t**3-t**2)[:, None]*tb)


def cable_centerline(side_profile, palm_pose, forearm_pose, *, solve_length=True):
    """Return palm-local centerline points; end transforms use current body poses."""
    palm_pose, forearm_pose = np.asarray(palm_pose, float), np.asarray(forearm_pose, float)
    if any(m.shape != (4, 4) or not np.isfinite(m).all() for m in (palm_pose, forearm_pose)):
        raise ValueError('Cable body poses must be finite 4x4 transforms')
    controls = np.asarray(side_profile['palm_controls_m'], float)
    a, b, c, following = controls
    tangents = (b-a, (c-a)*.5, (following-b)*.5)
    t = np.linspace(0, 1, 13)
    prefix = np.vstack([_hermite(a, b, tangents[0], tangents[1], t)[:-1],
                        _hermite(b, c, tangents[1], tangents[2], t)])
    relative = np.linalg.inv(palm_pose) @ forearm_pose
    end = _transform(relative, np.asarray([side_profile['forearm_entry_m']], float))[0]
    outward = _unit(relative[:3, :3] @ np.asarray(side_profile['forearm_outward'], float))
    forward = _unit(relative[:3, :3] @ np.array([0., 0., 1.]))
    # A short automatic handle produces a kink tighter than the tube radius.
    # Keep the measured exit direction but give the bend enough spatial room.
    handle_a = c + _unit(tangents[2])*.04
    handle_b = end + outward*.025 + forward*.025
    u = np.linspace(0, 1, 65)
    bridge = ((1-u)[:, None]**3*c + 3*((1-u)**2*u)[:, None]*handle_a
              + 3*((1-u)*u*u)[:, None]*handle_b + u[:, None]**3*end)
    # The bump has zero value and derivative at either end, so attachment
    # positions and the proximal tangent do not move while slack changes.
    bump = np.sin(np.pi*u)**2
    def curve(amplitude):
        return np.vstack([prefix[:-1], bridge + bump[:, None]*amplitude*outward])
    def length(points):
        return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    base = curve(0)
    if not solve_length:
        return base, {'length_m': length(base), 'slack_amplitude_m': 0.}
    target = float(side_profile['length_m'])
    if target < length(base)-2e-5:
        raise ValueError('Requested cable length cannot span the current anchors')
    lo, hi = 0., .30
    if length(curve(hi)) < target:
        raise ValueError('Requested cable length exceeds routing slack range')
    for _ in range(28):
        mid = (lo+hi)*.5
        if length(curve(mid)) < target:
            lo = mid
        else:
            hi = mid
    result = curve((lo+hi)*.5)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite wrist lead geometry')
    return result, {'length_m': length(result), 'slack_amplitude_m': (lo+hi)*.5}


def tube_mesh(centerline, radius, radial_segments=12):
    """Make a stable round tube using transported cross-section frames."""
    points = np.asarray(centerline, float)
    tangents = np.vstack([points[1]-points[0], points[2:]-points[:-2], points[-1]-points[-2]])
    tangents = np.asarray([_unit(t) for t in tangents])
    axis = np.eye(3)[int(np.argmin(np.abs(tangents[0])))]
    normal = _unit(np.cross(tangents[0], axis))
    frames = []
    for tangent in tangents:
        normal = _unit(normal - tangent*np.dot(normal, tangent))
        frames.append((normal.copy(), np.cross(tangent, normal)))
    frames = np.asarray(frames)
    theta = np.arange(radial_segments)*2*np.pi/radial_segments
    radial = (frames[:, 0, None, :]*np.cos(theta)[None, :, None]
              + frames[:, 1, None, :]*np.sin(theta)[None, :, None])
    vertices = (points[:, None, :] + radius*radial).reshape(-1, 3)
    quads = []
    for i in range(len(points)-1):
        for j in range(radial_segments):
            k = (j+1) % radial_segments
            quads.append([i*radial_segments+j, i*radial_segments+k,
                          (i+1)*radial_segments+k, (i+1)*radial_segments+j])
    # The plugged ends are buried in existing connector/forearm geometry.
    return vertices, np.asarray(quads, np.int32), radial.reshape(-1, 3)


def load_cable_profile(path):
    payload = Path(path).read_bytes()
    profile = json.loads(payload)
    if profile.get('schema') != 'handson.wrist_cable_routing.v1':
        raise ValueError('Unknown wrist cable routing schema')
    if profile.get('hardware') != 'rokae_ar5_5_08_wuji1' or set(profile.get('sides', {})) != {'left', 'right'}:
        raise ValueError('Wrist cable routing requires both Wuji1 arms')
    radius = profile.get('radius_m')
    if isinstance(radius, bool) or not isinstance(radius, (int, float)) or not .001 <= radius <= .008:
        raise ValueError('Invalid wrist cable radius')
    for side, row in profile['sides'].items():
        if not isinstance(row.get('reorient_strain_relief', False), bool):
            raise ValueError('Invalid strain-relief switch')
        for key, shape in [('palm_controls_m', (4, 3)), ('forearm_entry_m', (3,)), ('forearm_outward', (3,))]:
            value = np.asarray(row[key], float)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError('Invalid cable routing coordinates: '+key)
        _unit(np.asarray(row['forearm_outward'], float))
        if not .1 <= float(row['length_m']) <= .8:
            raise ValueError('Invalid wrist cable length')
        if row['palm_link'] != side+'_palm_link' or row['forearm_link'] != 'link5_'+side:
            raise ValueError('Unexpected cable attachment link')
    profile['sha256'] = hashlib.sha256(payload).hexdigest()
    return profile


def strain_relief_mesh(side_profile):
    """Rebuild the estimated sleeve and five ribs on the corrected exit axis."""
    controls = np.asarray(side_profile['palm_controls_m'], float)
    start = controls[0]
    axis = _unit(controls[1]-start)
    vertices, quads, normals = tube_mesh(
        start + np.array([-.008, 0., .013])[:, None]*axis, .006,
        radial_segments=32,
    )
    reference = np.eye(3)[int(np.argmin(np.abs(axis)))]
    u = _unit(np.cross(axis, reference)); v = np.cross(axis, u)
    all_vertices, all_quads, all_normals = [vertices], [quads], [normals]
    offset = len(vertices)
    theta = np.arange(32)*2*np.pi/32
    phi = np.arange(8)*2*np.pi/8
    radial = np.cos(theta)[:, None]*u + np.sin(theta)[:, None]*v
    tube_normals = (np.cos(phi)[None, :, None]*radial[:, None, :]
                    + np.sin(phi)[None, :, None]*axis)
    ring_faces = np.asarray([[i*8+j, ((i+1)%32)*8+j,
        ((i+1)%32)*8+(j+1)%8, i*8+(j+1)%8] for i in range(32) for j in range(8)])
    for index in range(5):
        center = start + axis*(.0025*index)
        ring = center + .0058*radial[:, None, :] + .00055*tube_normals
        all_vertices.append(ring.reshape(-1, 3)); all_normals.append(tube_normals.reshape(-1, 3))
        all_quads.append(ring_faces+offset); offset += 256
    return np.vstack(all_vertices), np.vstack(all_quads), np.vstack(all_normals)


def restore_wrist_cables(environment):
    """Restore the bundle's original lead visibility when reapplying appearance."""
    state = getattr(environment, '_routed_wrist_cables', None)
    if not state:
        return
    from pxr import UsdGeom
    if state['stage'] is environment._stage:
        for path in state['bundle_paths']:
            prim = environment._stage.GetPrimAtPath(path)
            if prim:
                prim.SetInstanceable(False)
        for path, visibility in state['old_visibility'].items():
            prim = environment._stage.GetPrimAtPath(path)
            if prim:
                UsdGeom.Imageable(prim).GetVisibilityAttr().Set(visibility)
        for path in state['mesh_paths'].values():
            environment._stage.RemovePrim(path.rsplit('/', 1)[0])
    environment._routed_wrist_cables = None


def apply_configured_wrist_cables(environment, profile_path=None):
    """Create the optional meshes alongside the bundle, before PhysX reset."""
    path = str(profile_path or os.environ.get('HANDSON_ISAACSIM_WRIST_CABLE_PROFILE', '')).strip()
    if not path:
        restore_wrist_cables(environment)
        return None
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Vt
    from .blender_visual_layer import physics_signature
    profile = load_cable_profile(path)
    report = getattr(environment, 'blender_visual_bundle_report', {}) or {}
    if report.get('bundle_manifest_sha256') != profile.get('visual_bundle_sha256'):
        raise ValueError('Wrist cable profile does not match the applied visual bundle')
    stage = environment._stage
    required = {r[k] for r in profile['sides'].values() for k in ('palm_link', 'forearm_link')}
    found = {}
    for root_path in environment._reference_robot_visual_roots:
        for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
            if prim.GetName() in required and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                if prim.GetName() in found:
                    raise ValueError('Ambiguous cable attachment: '+prim.GetName())
                found[prim.GetName()] = prim
    if set(found) != required:
        raise ValueError('Missing articulated wrist cable attachments')
    prepared = {}
    connector_old = {}
    cache = UsdGeom.XformCache()
    for side, row in profile['sides'].items():
        palm, forearm = found[row['palm_link']], found[row['forearm_link']]
        bundle = palm.GetPath().AppendPath('visuals/BlenderVisuals')
        old = [p.GetPath() for p in Usd.PrimRange(stage.GetPrimAtPath(bundle), Usd.TraverseInstanceProxies())
               if p.GetName() == side+'_WristLead' and p.IsA(UsdGeom.Mesh)]
        if len(old) != 1:
            raise ValueError('Expected exactly one old wrist lead per palm')
        connector_old[side] = []
        if row.get('reorient_strain_relief', False):
            names = {side+'_StrainRelief', *(side+'_ReliefRib'+str(i) for i in range(5))}
            candidates = [p.GetPath() for p in Usd.PrimRange(
                stage.GetPrimAtPath(bundle), Usd.TraverseInstanceProxies())
                if p.GetName() in names and UsdGeom.Imageable(p)]
            # Blender may export an object Xform and a differently named mesh
            # (e.g. ReliefRib/Torus). Select the outer visual object once.
            connector_old[side] = [p for p in candidates
                if not any(p != other and p.HasPrefix(other) for other in candidates)]
            if len(connector_old[side]) != 6:
                raise ValueError('Expected six original strain-relief parts')
        palm_pose = np.array(cache.GetLocalToWorldTransform(palm)).T
        arm_pose = np.array(cache.GetLocalToWorldTransform(forearm)).T
        centers, _ = cable_centerline(row, palm_pose, arm_pose)
        prepared[side] = (palm, bundle, old[0], tube_mesh(centers, profile['radius_m']))
    previous = getattr(environment, '_routed_wrist_cables', None)
    original_visibility = dict(previous['old_visibility']) if previous and previous['stage'] is stage else {}
    for _, _, old, _ in prepared.values():
        original_visibility.setdefault(str(old), UsdGeom.Imageable(stage.GetPrimAtPath(old)).GetVisibilityAttr().Get())
    for paths in connector_old.values():
        for old in paths:
            original_visibility.setdefault(str(old), UsdGeom.Imageable(stage.GetPrimAtPath(old)).GetVisibilityAttr().Get())
    before = physics_signature(stage)
    restore_wrist_cables(environment)
    paths = {}
    for side, (palm, bundle, old, geometry) in prepared.items():
        stage.GetPrimAtPath(bundle).SetInstanceable(False)
        UsdGeom.Imageable(stage.GetPrimAtPath(old)).MakeInvisible()
        visuals = palm.GetPath().AppendChild('visuals')
        root_path = visuals.AppendChild('RoutedWristLead')
        routed = UsdGeom.Xform.Define(stage, root_path)
        routed.GetVisibilityAttr().Set(UsdGeom.Tokens.inherited)
        routed.MakeMatrixXform().Set(UsdGeom.Xformable(stage.GetPrimAtPath(visuals)).GetLocalTransformation().GetInverse())
        mesh = UsdGeom.Mesh.Define(stage, root_path.AppendChild('Cable'))
        vertices, quads, normals = geometry
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices.astype(np.float32)))
        mesh.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.vstack([vertices.min(0), vertices.max(0)]).astype(np.float32)))
        mesh.CreateFaceVertexCountsAttr([4]*len(quads)); mesh.CreateFaceVertexIndicesAttr(quads.ravel().tolist())
        mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals.astype(np.float32)))
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex); mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        material = UsdShade.Material.Define(stage, root_path.AppendChild('Material'))
        shader = UsdShade.Shader.Define(stage, root_path.AppendPath('Material/Surface'))
        shader.CreateIdAttr('UsdPreviewSurface')
        shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.012, .014, .018))
        shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(.58)
        shader.CreateOutput('surface', Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
        if connector_old[side]:
            for old_connector in connector_old[side]:
                UsdGeom.Imageable(stage.GetPrimAtPath(old_connector)).MakeInvisible()
            connector = UsdGeom.Mesh.Define(stage, root_path.AppendChild('StrainRelief'))
            cv, cq, cn = strain_relief_mesh(profile['sides'][side])
            connector.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(cv.astype(np.float32)))
            connector.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.vstack([cv.min(0), cv.max(0)]).astype(np.float32)))
            connector.CreateFaceVertexCountsAttr([4]*len(cq)); connector.CreateFaceVertexIndicesAttr(cq.ravel().tolist())
            connector.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(cn.astype(np.float32)))
            connector.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
            connector.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            cm = UsdShade.Material.Define(stage, root_path.AppendChild('ConnectorMaterial'))
            cs = UsdShade.Shader.Define(stage, root_path.AppendPath('ConnectorMaterial/Surface'))
            cs.CreateIdAttr('UsdPreviewSurface')
            cs.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.007, .009, .012))
            cs.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(.36)
            cs.CreateOutput('surface', Sdf.ValueTypeNames.Token)
            cm.CreateSurfaceOutput().ConnectToSource(cs.ConnectableAPI(), 'surface')
            UsdShade.MaterialBindingAPI.Apply(connector.GetPrim()).Bind(cm)
        paths[side] = str(mesh.GetPath())
    if physics_signature(stage) != before:
        raise RuntimeError('Physics changed while attaching visual wrist leads')
    state = {'stage': stage, 'profile': profile, 'mesh_paths': paths,
             'physics_signature': before, 'indices': {}, 'old_visibility': original_visibility,
             'bundle_paths': [str(item[1]) for item in prepared.values()]}
    environment._routed_wrist_cables = state
    environment._wrist_obstruction_visibility = None
    environment._deferred_robot_body_visibility = None
    return state


def update_wrist_cables(environment):
    """Update only mesh points/normals from current articulation tensors."""
    state = getattr(environment, '_routed_wrist_cables', None)
    if not state or state['stage'] is not environment._stage:
        return
    from pxr import UsdGeom, Vt
    pending = []
    for side, row in state['profile']['sides'].items():
        if not hasattr(environment, '_control_robot'):
            cache = UsdGeom.XformCache()
            from pxr import Usd, UsdPhysics
            bodies = {prim.GetName(): prim for root in environment._reference_robot_visual_roots
                      for prim in Usd.PrimRange(environment._stage.GetPrimAtPath(root))
                      if prim.HasAPI(UsdPhysics.RigidBodyAPI)}
            poses = [np.asarray(cache.GetLocalToWorldTransform(bodies[row[key]])).T
                     for key in ('palm_link', 'forearm_link')]
            centers, metrics = cable_centerline(row, *poses)
            vertices, _, normals = tube_mesh(centers, state['profile']['radius_m'])
            pending.append((side, vertices, normals, metrics))
            continue
        robot = environment._control_robot(side)
        indices = state['indices'].get(side)
        if indices is None:
            indices = []
            for key in ('palm_link', 'forearm_link'):
                matches = [i for i, name in enumerate(robot.body_names) if str(name).split('/')[-1] == row[key]]
                if len(matches) != 1:
                    raise ValueError('Ambiguous runtime cable body '+row[key])
                indices.append(matches[0])
            state['indices'][side] = indices
        poses = [environment._robot_body_pose_matrix(robot, i) for i in indices]
        centers, metrics = cable_centerline(row, *poses)
        vertices, _, normals = tube_mesh(centers, state['profile']['radius_m'])
        pending.append((side, vertices, normals, metrics))
    for side, vertices, normals, metrics in pending:
        mesh = UsdGeom.Mesh(environment._stage.GetPrimAtPath(state['mesh_paths'][side]))
        mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(vertices.astype(np.float32)))
        mesh.GetExtentAttr().Set(Vt.Vec3fArray.FromNumpy(np.vstack([vertices.min(0), vertices.max(0)]).astype(np.float32)))
        mesh.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(normals.astype(np.float32)))
        state[side+'_metrics'] = metrics


# Numerical bridge correction preserves endpoints and the specified cable length.
from .cable_routing import make_solver
_cable_solver_adjustments = []
cable_centerline = make_solver(cable_centerline, _cable_solver_adjustments)
