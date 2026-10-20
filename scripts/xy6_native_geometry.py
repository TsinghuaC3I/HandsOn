#!/usr/bin/env python3
"""XY6 native geometry and short control-response diagnostics.

Adapted from XY4/XY3's validated native probes. Actual tensor poses define
invisible, nonphysical query inputs while Fabric leaves USD poses frozen.
This script does not create successful demonstrations or certify replay.
"""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import re
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'handson')]
from handson.sim.isaaclab.eval_comb06_season_and_stir_scene import OBJECT_NAMES as BODY_NAMES, TASK_NAME

def plain(value):
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value

def shapes(stage):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    cache = UsdGeom.XformCache()
    result = []
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if not prim.IsActive() or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
        if enabled is False:
            continue
        owner = prim
        while owner and not owner.IsPseudoRoot() and not owner.HasAPI(UsdPhysics.RigidBodyAPI):
            owner = owner.GetParent()
        # MJCF's authored primitive extents already include local scaling;
        # BBoxCache can apply it twice. Compute from shape geometry instead.
        extent = UsdGeom.Boundable.ComputeExtentFromPlugins(UsdGeom.Boundable(prim), Usd.TimeCode.Default())
        bound = Gf.BBox3d(Gf.Range3d(Gf.Vec3d(*extent[0]), Gf.Vec3d(*extent[1])),
                         cache.GetLocalToWorldTransform(prim)).ComputeAlignedRange()
        result.append({"path": str(prim.GetPath()), "type": prim.GetTypeName(),
                       "owner": str(owner.GetPath()) if owner and not owner.IsPseudoRoot() else None,
                       "visibility": str(UsdGeom.Imageable(prim).ComputeVisibility()),
                       "bounds": [list(bound.GetMin()), list(bound.GetMax())]})
    return result

def live_shapes(env, label):
    """Query the live tensor poses while Fabric leaves authored USD frozen.

    These invisible GPrims have no physics APIs: they are query inputs only.
    No robot/object state is written and no physics step is advanced.
    """
    from pxr import Gf, Sdf, Tf, UsdGeom, UsdPhysics
    stage = env._stage
    cache = UsdGeom.XformCache()
    poses = {}
    for side, robot in env._robots.items():
        data = robot.data.body_pose_w[0].detach().cpu().numpy()
        for name, pose in zip(robot.body_names, data):
            poses[(side, name)] = pose
    for name in (*BODY_NAMES, 'table'):
        pose = env._physics_body_pose(name)
        if pose is not None:

            prim = env._find_rigid_body_prim(name)
            if prim is None:
                raise RuntimeError(f'Cannot bind live pose for {name}')
            poses[str(prim.GetPath())] = pose
    result = shapes(stage)
    for index, row in enumerate(result):
        source = stage.GetPrimAtPath(row['path'])
        transform = cache.GetLocalToWorldTransform(source)
        owner = row['owner']
        pose = poses.get(owner)
        if owner and '/HandsOnRobots/' in owner:
            side = 'right' if '/RightRobot/' in owner else 'left'
            pose = poses.get((side, owner.rsplit('/', 1)[-1]))
        if pose is not None:
            owner_transform = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(owner))
            actual = Gf.Matrix4d().SetRotate(Gf.Quatd(float(pose[3]), Gf.Vec3d(*map(float, pose[4:]))))
            actual.SetTranslateOnly(Gf.Vec3d(*map(float, pose[:3])))
            # Rigid tensors omit authored asset scale; preserve the source scale.
            actual = Gf.Matrix4d().SetScale(Gf.Transform(owner_transform).GetScale()) * actual
            transform = transform * owner_transform.GetInverse() * actual
        query_path = f'/XY6ShapeQueries/{Tf.MakeValidIdentifier(label)}/shape_{index}'
        proxy = stage.DefinePrim(query_path, source.GetTypeName())
        proxy.GetReferences().AddInternalReference(Sdf.Path(row['path']))
        proxy.SetMetadata('apiSchemas', Sdf.TokenListOp.CreateExplicit([]))
        xform = UsdGeom.Xformable(proxy)
        xform.ClearXformOpOrder()
        xform.AddTransformOp(opSuffix='query').Set(transform)
        UsdGeom.Imageable(proxy).CreateVisibilityAttr().Set('invisible')
        assert not proxy.HasAPI(UsdPhysics.CollisionAPI) and not proxy.HasAPI(UsdPhysics.RigidBodyAPI)
        extent = UsdGeom.Boundable.ComputeExtentFromPlugins(UsdGeom.Boundable(source), 0)
        bounds = Gf.BBox3d(Gf.Range3d(Gf.Vec3d(*extent[0]), Gf.Vec3d(*extent[1])), transform).ComputeAlignedRange()
        row['bounds'] = [list(bounds.GetMin()), list(bounds.GetMax())]
        row['query_path'] = query_path
        row['live_owner_pose'] = pose
    return result

def overlaps(stage, inventory):
    from omni.physx import get_physx_scene_query_interface
    from pxr import PhysicsSchemaTools, Sdf
    query = get_physx_scene_query_interface()
    pairs = set()
    errors = []
    controls = []
    for row in inventory:
        if row["type"] == "Plane":
            continue
        found = []
        center = np.mean(row["bounds"], axis=0).tolist()
        query.overlap_sphere(.01, center, lambda hit: found.append(str(hit.collision)) is None, False)
        if row["path"] in found:
            controls.append({"source": row["path"], "hits": found})
            break
    for row in inventory:
        source = row["path"]
        if row["type"] == "Plane":
            continue
        try:
            encoded = PhysicsSchemaTools.encodeSdfPath(Sdf.Path(row.get('query_path', source)))
            def hit(value):
                target = str(value.collision)
                if target != source:
                    pairs.add(tuple(sorted((source, target))))
                return True
            query.overlap_shape(encoded[0], encoded[1], hit, False)
        except Exception as exc:
            errors.append({"path": source, "error": str(exc)})
    return {"pairs": sorted(pairs), "errors": errors, "positive_controls": controls,
            "positive_control_passed": bool(controls),
            "excluded_query_sources": [r["path"] for r in inventory if r["type"] == "Plane"],
            "query": "PhysX overlap_shape; depth requires contact readback"}

def snapshot(env, obs):
    values = {"objects": {}, "robots": {}}
    for name in BODY_NAMES:
        values["objects"][name] = {"pose": env._physics_body_pose(name), "velocity": env._physics_body_velocity(name)}
    for side, robot in env._robots.items():
        q = robot.data.joint_pos[0].detach().cpu().numpy()
        qd = robot.data.joint_vel[0].detach().cpu().numpy()
        target = robot.data.joint_pos_target[0].detach().cpu().numpy()
        values["robots"][side] = {"names": robot.joint_names, "q": q, "qd": qd,
            "body_names": robot.body_names, "body_poses": robot.data.body_pose_w[0].detach().cpu().numpy(),
            "drive_target": target, "max_target_error": float(np.max(np.abs(q-target))),
            "tcp_actual": robot.data.body_pose_w[0, robot.find_bodies(f"{side}_palm_link")[0][0]].detach().cpu().numpy(), "tcp_command": getattr(env, f"_{side}_tcp_pose")}
    if "physx_contact_header_count" in obs:
        from pxr import PhysicsSchemaTools
        n = int(np.asarray(obs["physx_contact_header_count"]).flat[0])
        contacts = []
        for ids, meta in zip(obs["physx_contact_header_ids"][:n], obs["physx_contact_header_meta"][:n]):
            start, count = int(meta[1]), int(meta[2])
            contacts.append({"paths": [str(PhysicsSchemaTools.intToSdfPath(int(v))) for v in ids[:4]],
                             "points": obs["physx_contact_point_data"][start:start+count]})
        values["contacts"] = contacts
    values["hidden"] = obs.get("xy6_hidden_state")
    def contact_rows(view):
        arrays = [x.detach().cpu().numpy().copy() for x in view.get_contact_data(env._physics_dt)]
        forces, points, normals, separations, counts, starts = arrays
        rows = []
        for i, sensor in enumerate(view.sensor_paths):
            for j, target in enumerate(env._xy6_probe_static_paths):
                count, start = int(counts[i, j]), int(starts[i, j])
                if count:
                    rows.append({'sensor': sensor, 'target': target, 'count': count,
                                 'points': points[start:start+count],
                                 'separations': separations[start:start+count],
                                 'max_penetration': max(0., -float(separations[start:start+count].min())),
                                 'normal_force': float(forces[start:start+count].sum())})
        return rows
    for kind in ('support', 'robot'):
        view = getattr(env, f'_xy6_probe_{kind}_view', None)
        if view is not None:
            values[f'{kind}_static_contacts'] = contact_rows(view)
    return values


def reason(a, b, inventory):
    ra, rb = inventory[a], inventory[b]
    oa, ob = ra['owner'], rb['owner']
    if '/HandsOnRobots/' in a or '/HandsOnRobots/' in b:
        for side in ('right', 'left'):
            root = f'/World/HandsOnRobots/{side.capitalize()}Robot/'
            if not (a.startswith(root) and b.startswith(root)):
                continue
            na, nb = oa.rsplit('/', 1)[-1], ob.rsplit('/', 1)[-1]
            if oa == ob:
                return 'source compound fingertip on one rigid link'
            names = {na, nb}
            if names in ({f'attachment_{side}', f'{side}_palm_link'},
                         {f'link7_{side}', f'{side}_palm_link'}):
                return 'source wrist/palm mounting junction'
            fingers = [re.fullmatch(rf'{side}_finger(\d)_link(\d)', n) for n in (na, nb)]
            if all(fingers) and fingers[0][1] == fingers[1][1] and abs(int(fingers[0][2])-int(fingers[1][2])) == 1:
                return 'source adjacent finger joint shells'
            if f'{side}_palm_link' in names and any(m and int(m[2]) <= 2 for m in fingers):
                return 'source proximal finger/palm housing junction'
            links = [re.fullmatch(rf'link(\d)_{side}', n) for n in (na, nb)]
            if all(links) and (abs(int(links[0][1])-int(links[1][1])) == 1 or {links[0][1], links[1][1]} == {'5', '7'}):
                return 'source arm joint/wrist housing junction'
        return None
    if oa and oa == ob:
        return 'source compound colliders on one rigid body'
    is_table = lambda p: p and ('worktable' in p.lower() or p.endswith('/table'))
    if is_table(oa or a) or is_table(ob or b):
        other = ob if is_table(oa or a) else oa
        if other and other.rsplit('/', 1)[-1] in BODY_NAMES:
            return 'intended tabletop support; native contact depth checked separately'
    return None



def native_pour_geometry(env):
    """Query real pot/lid colliders with hypothetical cap rays; move no bodies."""
    from handson.sim.isaaclab.eval_comb06_season_and_stir_scene import pot_flow_projection, rotation_wxyz
    from handson.sim.isaaclab.b320_bimanual_scene import BC014_JAR_TARGET_HEIGHT_M
    from handson.sim.isaaclab.b320_bimanual_env import _BC014_FLOW_TILT_THRESHOLD_DEG, _BC014_FLOW_MAX_LENGTH_M
    from omni.physx import get_physx_scene_query_interface
    pot=env._pose('dutch_oven'); rotation=rotation_wxyz(pot[3:])
    rows=[]
    for name,u,expected in (('salt_interior',-.035,True),('pepper_interior',.035,True),('outside_rim',.09,False)):
        cap=pot[:3]+rotation@np.array([u,.23,0.])
        # Source jars emit along local +Z. Invert that axis without modifying
        # the actual jar poses or task history.
        jar=np.r_[cap+np.array([0.,0.,BC014_JAR_TARGET_HEIGHT_M]),[0.,1.,0.,0.]]
        sample=pot_flow_projection(jar,pot,jar_height=BC014_JAR_TARGET_HEIGHT_M,
            tilt_threshold=_BC014_FLOW_TILT_THRESHOLD_DEG,maximum_length=_BC014_FLOW_MAX_LENGTH_M)
        clear=env._flow_path_clear(sample) if sample['projects_into_zone'] else False
        valid=bool(sample['projects_into_zone'] and clear)
        rows.append({'name':name,'sample':sample,'native_path_clear':clear,'valid':valid,'expected':expected})
        assert valid==expected,rows[-1]
    lid=env._pose('dutch_oven_lid')
    sample={'cap_position':lid[:3]+np.array([0.,0.,.20]),'cap_axis':np.array([0.,0.,-1.]),'stream_length_m':.30}
    blocked=not env._flow_path_clear(sample)
    hit=get_physx_scene_query_interface().raycast_closest(tuple(sample['cap_position']), (0.,0.,-1.), .30)
    assert blocked and 'dutch_oven_lid' in str(hit.get('rigidBody',hit.get('collision',''))),hit
    hit_record={'hit':bool(hit.get('hit')),'collision':str(hit.get('collision','')),
                'rigid_body':str(hit.get('rigidBody','')),'distance':float(hit.get('distance',0.)),
                'position':list(hit.get('position',[])),'normal':list(hit.get('normal',[]))}
    return {'cap_cases':rows,'real_lid_blocks':blocked,'lid_hit':hit_record,
            'scope':'native ray geometry only; no contact/tilt timers or successful pouring demonstration claimed'}
