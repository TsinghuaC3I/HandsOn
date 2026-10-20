#!/usr/bin/env python3
"""XY5 native geometry and short control-response diagnostics.

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
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'handson')]
from handson.tasks.eval_comb01_cover_pitcher_and_box.layout import BODY_NAMES, TASK_NAME

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
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics
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
        query_path = f'/XY5ShapeQueries/{label}/shape_{index}'
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
    values["hidden"] = obs.get("xy5_hidden_state")
    def contact_rows(view):
        arrays = [x.detach().cpu().numpy().copy() for x in view.get_contact_data(env._physics_dt)]
        forces, points, normals, separations, counts, starts = arrays
        rows = []
        for i, sensor in enumerate(view.sensor_paths):
            for j, target in enumerate(env._xy5_probe_static_paths):
                count, start = int(counts[i, j]), int(starts[i, j])
                if count:
                    rows.append({'sensor': sensor, 'target': target, 'count': count,
                                 'points': points[start:start+count],
                                 'separations': separations[start:start+count],
                                 'max_penetration': max(0., -float(separations[start:start+count].min())),
                                 'normal_force': float(forces[start:start+count].sum())})
        return rows
    for kind in ('support', 'robot'):
        view = getattr(env, f'_xy5_probe_{kind}_view', None)
        if view is not None:
            values[f'{kind}_static_contacts'] = contact_rows(view)
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', default='standard', help='standard, all, seed_N or named boundary')
    parser.add_argument('--steps', '--settle-steps', type=int, default=60)
    parser.add_argument('--tracking', action='store_true')
    parser.add_argument('--render-preview', action='store_true')
    parser.add_argument('--stable-origin', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    from handson.sim.isaaclab.eval_comb01_cover_pitcher_and_box_env import XY5PitcherThenBoxLidEnv
    from handson.sim.isaaclab.abc_collection_env import _TABLE_TOP_Z
    from collection_ui.server import ISAAC_COLLECTION_REALTIME_ENV
    from audit_xy5_cpu_geometry import audit_layouts
    os.environ.update(ISAAC_COLLECTION_REALTIME_ENV)
    os.environ['HANDSON_ENABLE_ISAACSIM_NEWTON'] = '0'
    os.environ['HANDSON_RECORD_PHYSX_GROUND_TRUTH'] = '1'
    if args.stable_origin:
        os.environ['HANDSON_STABLE_RECORDING_ORIGIN'] = '1'
    os.environ['HANDSON_ISAACLAB_DEFER_CAMERA_RENDER'] = '0' if args.render_preview else '1'
    os.environ['HANDSON_ISAACLAB_LIVE_MAIN_CAMERA_ONLY'] = '0'
    report = {'task': TASK_NAME, 'isaacsim': importlib.metadata.version('isaacsim'),
              'scope': 'native geometry, initial stability and short drive response; human task pending', 'events': [], 'rows': []}
    if not report['isaacsim'].startswith('5.1.'):
        raise RuntimeError('XY5 requires Isaac Sim 5.1')
    def save():
        (output/'report.json').write_text(json.dumps(plain(report), indent=2, allow_nan=False)+'\n')
    def shape_event(env, label, first=False):
        inventory = shapes(env._stage) if first else live_shapes(env, label)
        result = overlaps(env._stage, inventory)
        report['events'].append({'label':label,'inventory':inventory,'overlap':result})
        save()
        if result['errors'] or not result['positive_control_passed']:
            raise RuntimeError(f'Native shape query coverage failed: {label}')
        print('XY5_QUERY '+json.dumps({'label':label,'shapes':len(inventory),'pairs':len(result['pairs'])}),flush=True)
    class DiagnosticEnv(XY5PitcherThenBoxLidEnv):
        audit_case = None
        native_loaded = False
        def _configure_task_simulation_settings(self):
            super()._configure_task_simulation_settings()
            from pxr import Usd, UsdPhysics, PhysxSchema
            for p in Usd.PrimRange(self._stage.GetPrimAtPath('/World/HandsOnRobots')):
                if p.HasAPI(UsdPhysics.RigidBodyAPI) and '/CenterRig/' not in str(p.GetPath()):
                    PhysxSchema.PhysxContactReportAPI.Apply(p).CreateThresholdAttr().Set(0.)
            self._sim.get_physics_context().set_enable_scene_query_support(True)
            self._stage.GetRootLayer().Export(str(output/'authored-stage.usda'))
            from omni.physx import get_physx_interface
            get_physx_interface().force_load_physics_from_usd()
            shape_event(self,'authored_before_first_physics',True)
            self.native_loaded = True
        def _sample_reset_state(self, *, clear_manual_request):
            super()._sample_reset_state(clear_manual_request=clear_manual_request)
            if self.audit_case is not None:
                _,height,poses = self.audit_case
                self.delta_h[0] = height-_TABLE_TOP_Z
                self._xy5_layout = {k:v.copy() for k,v in poses.items()}
                self._lid_start_pose = self._xy5_layout['pitcher_lid'].copy()
                self._xy5_b05_start_pose = self._xy5_layout['Box_lid'].copy()
                self._refresh_object_poses()
    env = None; code = 1
    try:
        env = DiagnosticEnv(seed=0,randomize=False,image_obs=args.render_preview,headless=True,
                            physics_backend='physx',render_mode='rgb_array',isaac_device='cuda:0',
                            arm_type='rokae',hand_type='wuji1')
        report['timing'] = {'physics_dt':env._physics_dt,'control_dt':env._control_dt,'substeps':env._n_substeps}
        report['online_cameras'] = list(env._cameras)
        report['reset_rigid_paths'] = [r[0] for r in env._task_reset_body_states]
        from pxr import Usd, UsdPhysics
        env._xy5_probe_static_paths = sorted({
            r['owner'] or r['path'] for r in shapes(env._stage)
            if r['type'] != 'Plane' and (r['owner'] is None or
               bool(UsdPhysics.RigidBodyAPI(env._stage.GetPrimAtPath(r['owner'])).GetKinematicEnabledAttr().Get()))
        })
        assert env._xy5_probe_static_paths, 'No native static support shapes found'
        robot_paths = [str(p.GetPath()) for p in Usd.PrimRange(env._stage.GetPrimAtPath('/World/HandsOnRobots'))
                       if p.HasAPI(UsdPhysics.RigidBodyAPI) and '/CenterRig/' not in str(p.GetPath())]
        object_paths = [str(env._find_rigid_body_prim(name).GetPath()) for name in BODY_NAMES]
        for kind, sensors in (('support', object_paths), ('robot', robot_paths)):
            view = env._robots['right']._physics_sim_view.create_rigid_contact_view(
                sensors, filter_patterns=[env._xy5_probe_static_paths for _ in sensors],
                max_contact_data_count=2048)
            setattr(env, f'_xy5_probe_{kind}_view', view)
            report[f'{kind}_contact_view'] = {'sensors': sensors, 'filters': env._xy5_probe_static_paths,
                                            'sensor_count': view.sensor_count, 'filter_count': view.filter_count}
        cases = [('standard',_TABLE_TOP_Z,None),*audit_layouts()]
        if args.case != 'all': cases=[r for r in cases if r[0] in args.case.split(',')]
        if not cases: raise ValueError('Unknown audit case')
        for case in cases:
            label=case[0]; seed=int(label.split('_')[1]) if label.startswith('seed_') else 0
            env.randomize=label.startswith('seed_')
            env.audit_case=None if env.randomize or label=='standard' else case
            obs,info=env.reset(seed=seed)
            first_layout={k:v.copy() for k,v in env._xy5_layout.items()}
            obs,info=env.reset(seed=seed)
            assert all(np.array_equal(v,env._xy5_layout[k]) for k,v in first_layout.items())
            row={'label':label,'sampled_layout':env._xy5_layout,'repeatable':True,'reset':snapshot(env,obs),'samples':[]}
            report['rows'].append(row);save()
            shape_event(env,label+'_reset')
            if args.stable_origin:
                from handson.sim.isaaclab.recording_origin import stabilize_recording_origin, stable_recording_origin_kwargs_for_task
                profile=stable_recording_origin_kwargs_for_task(TASK_NAME)
                row['origin_profile']=profile
                obs,info=stabilize_recording_origin(env,obs,info,**profile)
                row['recording_origin']=info['physical_recording_origin']
                row['accepted_origin']=snapshot(env,obs);save()
            command_origin=np.r_[env._right_tcp_pose,env._right_hand]
            assert command_origin.shape==env.action_space.shape
            for step in range(args.steps):
                command=command_origin.copy()
                if args.tracking:
                    scale=min(1.,step/12.) if step<36 else max(0.,1.-(step-36)/12.)
                    command[2]+=.025*scale;command[7:]+=.12*scale
                start=time.monotonic()
                obs,reward,terminated,truncated,info=env.step(command)
                if args.tracking or step in (0,1,args.steps-1):
                    row['samples'].append({'step':step,'wall_seconds':time.monotonic()-start,
                                           'physics':snapshot(env,obs),'terminated':terminated,'truncated':truncated,'info':info})
                if terminated or truncated:raise RuntimeError('Diagnostic unexpectedly terminated')
                if args.tracking and step in (35, 48):
                    shape_event(env, label+f'_tracking_{step}')
            row['settled']=snapshot(env,obs)
            shape_event(env,label+'_settled')
            if args.render_preview:
                from PIL import Image
                for _ in range(3):frames=env.render()
                row['frames']=[]
                for name,frame in zip(('main','wrist_left','wrist_right'),frames):
                    frame=np.asarray(frame)
                    assert frame.ndim==3 and frame.std()>5,name
                    file=f'{label}-{name}.png'
                    Image.fromarray(frame.astype(np.uint8)).save(output/file)
                    row['frames'].append({'path':file,'shape':list(frame.shape),'std':float(frame.std())})
            save();print('XY5_SAMPLE '+label,flush=True)
        (output/'environment-schema.json').write_text(json.dumps(plain(env.get_recording_schema()), indent=2)+'\n')
        report['isaac_warnings']=env._isaac_warnings
        report['completed_diagnostic']=True;code=0
    except BaseException:
        report['error']=traceback.format_exc();traceback.print_exc()
    finally:
        save()
        if env is not None:
            env._fast_shutdown_on_close=True
            env.close()
    return code

if __name__=='__main__':
    raise SystemExit(main())
