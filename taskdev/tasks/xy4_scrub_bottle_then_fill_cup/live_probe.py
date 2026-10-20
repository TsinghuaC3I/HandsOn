#!/usr/bin/env python3
"""Native XY4 geometry/control diagnostic, adapted from the qualified XY3 probe.
Reset overrides apply only to boundary layouts; no trajectory state writes or
successful synthetic episode are generated. Preview cameras are explicit.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'handson'), str(Path(__file__).parent)]
from handson.sim.isaaclab.xy4_env import HandsOnXY4IsaacLabEnv
from handson.sim.isaaclab.xy4_contract import TASK_NAME

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

def write(path, value):
    path.write_text(json.dumps(plain(value), indent=2, allow_nan=False) + "\n")

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
    for name in ('hs005_bottle', 'hs005_brush', 'ceramic_mug', 'faucet_base', 'faucet_handle'):
        pose = env._physics_body_pose(name)
        if pose is not None:
            poses[env._b320_spawn_result.body_paths[name]] = pose
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
            # Rigid tensors omit the authored asset scale (the mug is 0.006).
            actual = Gf.Matrix4d().SetScale(Gf.Transform(owner_transform).GetScale()) * actual
            transform = transform * owner_transform.GetInverse() * actual
        query_path = f'/XY4ShapeQueries/{label}/shape_{index}'
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
    for name in ("hs005_bottle", "hs005_brush", "ceramic_mug"):
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
    values["hidden"] = obs.get("xy4_hidden_state")
    view = getattr(env, '_xy4_probe_support_view', None)
    if view is not None:
        def cpu(value):
            return value.detach().cpu().numpy().copy()
        forces, points, normals, separations, counts, starts = map(cpu, view.get_contact_data(env._physics_dt))
        contacts = []
        for index, name in enumerate(('hs005_bottle', 'hs005_brush', 'ceramic_mug')):
            count, start = int(counts[index, 0]), int(starts[index, 0])
            contacts.append({'object': name, 'count': count,
                             'points': points[start:start+count],
                             'separations': separations[start:start+count],
                             'normal_forces': forces[start:start+count]})
        values['support_contacts'] = contacts
    view = getattr(env, '_xy4_probe_robot_view', None)
    if view is not None:
        forces, points, normals, separations, counts, starts = map(cpu, view.get_contact_data(env._physics_dt))
        values['robot_static_contacts'] = []
        for i, sensor in enumerate(view.sensor_paths):
            for j, target in enumerate(env._xy4_probe_static_paths):
                count, start = int(counts[i, j]), int(starts[i, j])
                if count:
                    values['robot_static_contacts'].append({'sensor': sensor, 'target': target, 'count': count,
                        'max_penetration': max(0., -float(separations[start:start+count].min())),
                        'normal_force': float(forces[start:start+count].sum())})
    view = getattr(env, '_xy4_probe_thumb_view', None)
    if view is not None:
        forces, points, normals, separations, counts, starts = map(cpu, view.get_contact_data(env._physics_dt))
        values['thumb_robot_contacts'] = []
        for i, sensor in enumerate(view.sensor_paths):
            for j, target in enumerate(env._xy4_probe_robot_paths):
                count, start = int(counts[i, j]), int(starts[i, j])
                if count:
                    values['thumb_robot_contacts'].append({'sensor': sensor, 'target': target,
                        'max_penetration': max(0., -float(separations[start:start+count].min())),
                        'normal_force': float(forces[start:start+count].sum())})
    return values


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--matrix', action='store_true')
    p.add_argument('--tracking', action='store_true')
    p.add_argument('--tracking-isolation', action='store_true', help='Compare all-finger and isolated thumb pulses')
    p.add_argument('--tracking-z-m', type=float, default=.025, help='Diagnostic TCP Z pulse; zero removes the original faucet approach')
    p.add_argument('--tracking-x-m', type=float, default=0., help='Diagnostic TCP X pulse; negative moves away from the faucet')
    p.add_argument('--steps', type=int, default=60)
    p.add_argument('--settle-steps', type=int, default=0, help='Diagnostic reset-settle comparison; does not change production defaults')
    p.add_argument('--render-preview', action='store_true')
    p.add_argument('--stable-origin', action='store_true', help='Use the unchanged formal recording-origin lifecycle before commands')
    p.add_argument('--origin-timeout', type=float, default=0, help='Short diagnostic failure budget; zero keeps production budget')
    args = p.parse_args()
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    from cpu_audit import layout_cases
    cases = layout_cases() if args.matrix else [{'name': 'standard', 'seed': 0}]
    if args.tracking_isolation:
        assert args.tracking and not args.matrix
        cases.append({'name': 'isolated_thumb', 'seed': 0})
    from collection_ui.server import ISAAC_COLLECTION_REALTIME_ENV
    os.environ.update(ISAAC_COLLECTION_REALTIME_ENV)
    os.environ['HANDSON_RECORD_PHYSX_GROUND_TRUTH'] = '1'
    if args.stable_origin:
        os.environ['HANDSON_STABLE_RECORDING_ORIGIN'] = '1'
    os.environ['HANDSON_ISAACLAB_DEFER_CAMERA_RENDER'] = '0' if args.render_preview else '1'
    os.environ['HANDSON_ISAACLAB_LIVE_MAIN_CAMERA_ONLY'] = '0'
    report = {'source': str(ROOT), 'isaacsim': importlib.metadata.version('isaacsim'),
              'scope': 'actual reset geometry, static stability and small drive response; full operator task pending', 'rows': []}
    report['tracking_tcp_pulse_m'] = [args.tracking_x_m, 0., args.tracking_z_m]
    assert report['isaacsim'].startswith('5.1.')

    class AuditedXY4(HandsOnXY4IsaacLabEnv):
        validation_layout = None

        def _ready_state_settle_physics_steps(self):
            return max(super()._ready_state_settle_physics_steps(), args.settle_steps)

        def _setup_isaaclab(self):
            try:
                return super()._setup_isaaclab()
            except BaseException:
                report['error'] = traceback.format_exc()
                write(output / 'report.json', report)
                raise

        def _sample_ka021_reset_pose(self):
            cup = super()._sample_ka021_reset_pose()
            if self.validation_layout is not None:
                self._xy4_layout = {k: np.asarray(v).copy() for k, v in self.validation_layout.items()}
                self._hs005_bottle_reset_pose = self._xy4_layout['hs005_bottle'].copy()
                self._hs005_brush_reset_pose = self._xy4_layout['hs005_brush'].copy()
                cup = self._xy4_layout['ceramic_mug'].copy()
            return cup

        def _configure_task_simulation_settings(self):
            super()._configure_task_simulation_settings()
            from pxr import Usd, UsdPhysics, PhysxSchema
            robot_paths = [str(p.GetPath()) for p in Usd.PrimRange(self._stage.GetPrimAtPath('/World/HandsOnRobots'))
                           if p.HasAPI(UsdPhysics.RigidBodyAPI) and '/CenterRig/' not in str(p.GetPath())]
            for path in robot_paths:
                PhysxSchema.PhysxContactReportAPI.Apply(self._stage.GetPrimAtPath(path)).CreateThresholdAttr().Set(0.)
            self._sim.get_physics_context().set_enable_scene_query_support(True)
            inventory = shapes(self._stage)
            self._stage.GetRootLayer().Export(str(output / 'authored-stage.usda'))
            from omni.physx import get_physx_interface
            get_physx_interface().force_load_physics_from_usd()
            write(output / 'before-first-physics.json', {'inventory': inventory, 'overlap': overlaps(self._stage, inventory)})

    env = None; code = 1
    try:
        print('XY4_CONSTRUCT', flush=True)
        env = AuditedXY4(hz=30, time_limit=90., launch_isaac=True,
                        image_obs=args.render_preview, headless=True, render_mode='rgb_array',
                        randomize=False, seed=0, isaac_device='cuda:0', hand_type='wuji1', arm_type='rokae')
        report['timing'] = {'physics_dt': env._physics_dt, 'control_dt': env._control_dt, 'substeps': env._n_substeps}
        report['online_cameras'] = list(env._cameras)
        report['reset_rigid_paths'] = [r[0] for r in env._task_reset_body_states]
        report['reset_articulation_paths'] = [r[0] for r in env._task_reset_articulation_states]
        report['articulation_aliases'] = list(env._task_articulation_views)
        from pxr import Usd, UsdPhysics, PhysxSchema
        report['robot_gravity_disabled'] = {str(p.GetPath()): PhysxSchema.PhysxRigidBodyAPI(p).GetDisableGravityAttr().Get()
            for p in Usd.PrimRange(env._stage.GetPrimAtPath('/World/HandsOnRobots')) if p.HasAPI(UsdPhysics.RigidBodyAPI)}
        report['robot_parameters'] = {}
        for side, robot in env._robots.items():
            report['robot_parameters'][side] = {key: getattr(robot.data, key).detach().cpu().numpy()
                for key in ('default_joint_pos', 'joint_pos_limits', 'joint_stiffness', 'joint_damping', 'joint_effort_limits')}
        report['joint_usd_parameters'] = {str(p.GetPath()): {a.GetName(): str(a.Get()) for a in p.GetAttributes()}
            for p in Usd.PrimRange(env._stage.GetPrimAtPath('/World/HandsOnRobots')) if p.IsA(UsdPhysics.RevoluteJoint)}
        report['diagnostic_settle_steps'] = args.settle_steps
        # The inherited recording fallback filters rigid bodies. This extra
        # read-only diagnostic explicitly samples the static sink collider too.
        sensors = [env._b320_spawn_result.body_paths[n] for n in ('hs005_bottle', 'hs005_brush', 'ceramic_mug')]
        support = '/World/Task/B320Benchmark/faucet_fixture/hs005_realistic_kitchen/Sink/Collider'
        env._xy4_probe_support_view = env._robots['right']._physics_sim_view.create_rigid_contact_view(
            sensors, filter_patterns=[[support] for _ in sensors], max_contact_data_count=512)
        report['support_contact_view'] = {'sensors': sensors, 'filter': support,
            'sensor_count': env._xy4_probe_support_view.sensor_count, 'filter_count': env._xy4_probe_support_view.filter_count}
        sensors = [str(p.GetPath()) for p in Usd.PrimRange(env._stage.GetPrimAtPath('/World/HandsOnRobots'))
                   if p.HasAPI(UsdPhysics.RigidBodyAPI) and '/CenterRig/' not in str(p.GetPath())]
        env._xy4_probe_static_paths = [r['path'] for r in shapes(env._stage) if r['owner'] is None and r['type'] != 'Plane']
        env._xy4_probe_robot_view = env._robots['right']._physics_sim_view.create_rigid_contact_view(
            sensors, filter_patterns=[env._xy4_probe_static_paths for _ in sensors], max_contact_data_count=2048)
        if args.tracking:
            env._xy4_probe_robot_paths = sensors
            thumbs = [path for path in sensors if 'finger1_' in path]
            env._xy4_probe_thumb_view = env._robots['right']._physics_sim_view.create_rigid_contact_view(
                thumbs, filter_patterns=[sensors for _ in thumbs], max_contact_data_count=2048)
        assert set(env._robots) == {'right', 'left'}
        assert np.isclose(env._physics_dt, 1/120) and np.isclose(env._control_dt, 1/30) and env._n_substeps == 4
        for case in cases:
            boundary = case['name'].startswith('boundary_')
            env.validation_layout = case['poses_wxyz'] if boundary else None
            env.randomize = case['name'].startswith('seed_')
            obs, info = env.reset(seed=case['seed'] or 0)
            first_layout = {k: v.copy() for k, v in env._xy4_layout.items()}
            obs, info = env.reset(seed=case['seed'] or 0)
            assert all(np.array_equal(v, env._xy4_layout[k]) for k, v in first_layout.items())
            row = {'layout': case['name'], 'seed': case['seed'], 'boundary_reset_override': boundary,
                   'sampled_layout': env._xy4_layout, 'reset': snapshot(env, obs),
                   'park_position_w': env._ka021_cup_park_position_w.copy(),
                   'reset_info': info, 'samples': [], 'peak_body_linear_speed': 0.,
                   'peak_body_angular_speed': 0., 'peak_robot_joint_speed': 0.}
            report['rows'].append(row)
            inventory = live_shapes(env, case['name'] + '_reset')
            row['reset_shapes'] = inventory
            row['reset_overlap'] = overlaps(env._stage, inventory)
            write(output / 'report.json', report)
            if args.stable_origin:
                from handson.sim.isaaclab import recording_origin as origin_module
                profile = origin_module.stable_recording_origin_kwargs_for_task(TASK_NAME)
                if args.origin_timeout:
                    profile['timeout_seconds'] = args.origin_timeout
                row['origin_profile'] = profile
                row['origin_motion_samples'] = []
                row['origin_pose_samples'] = []
                original_pose = origin_module._task_pose_delta
                original_robot = origin_module._robot_motion
                def pose_sample(before, after):
                    value = original_pose(before, after)
                    row['origin_pose_samples'].append(value)
                    return value
                def robot_sample(value):
                    result = original_robot(value)
                    row['origin_motion_samples'].append(result)
                    return result
                origin_module._task_pose_delta = pose_sample
                origin_module._robot_motion = robot_sample
                try:
                    obs, info = origin_module.stabilize_recording_origin(env, obs, info, **profile)
                finally:
                    origin_module._task_pose_delta = original_pose
                    origin_module._robot_motion = original_robot
                row['recording_origin'] = info['physical_recording_origin']
                row['accepted_origin'] = snapshot(env, obs)
            origin = np.r_[env._right_tcp_pose, env._left_tcp_pose, env._right_hand, env._left_hand]
            for step in range(args.steps):
                command = origin.copy()
                if args.tracking:
                    scale = min(1., step/12.) if step < 36 else max(0., 1.-(step-36)/12.)
                    command[0] += args.tracking_x_m*scale; command[7] += args.tracking_x_m*scale
                    command[2] += args.tracking_z_m*scale; command[9] += args.tracking_z_m*scale
                    if case['name'] == 'isolated_thumb':
                        for side, offset in (('right', 14), ('left', 34)):
                            robot = env._robots[side]
                            hand_names = [robot.joint_names[i] for i in env._hand_joint_ids[side]]
                            command[offset + hand_names.index(f'{side}_finger1_joint2')] += .12*scale
                    else:
                        command[14:] += .12*scale
                start = time.monotonic()
                obs, reward, terminated, truncated, info = env.step(command)
                state = snapshot(env, obs)
                for body in state['objects'].values():
                    velocity = body['velocity']
                    if velocity is None: raise RuntimeError('Missing native body velocity')
                    row['peak_body_linear_speed'] = max(row['peak_body_linear_speed'], float(np.linalg.norm(velocity[:3])))
                    row['peak_body_angular_speed'] = max(row['peak_body_angular_speed'], float(np.linalg.norm(velocity[3:])))
                row['peak_robot_joint_speed'] = max(row['peak_robot_joint_speed'], *[float(np.max(np.abs(r['qd']))) for r in state['robots'].values()])
                if args.tracking or step in (0, 1, args.steps - 1):
                    row['samples'].append({'step': step, 'wall_seconds': time.monotonic()-start,
                                           'physics': state, 'terminated': terminated, 'truncated': truncated})
                if terminated or truncated: raise RuntimeError('Static/response diagnostic terminated unexpectedly')
            row['settled_shapes'] = live_shapes(env, case['name'] + '_settled')
            row['settled_overlap'] = overlaps(env._stage, row['settled_shapes'])
            row['settled'] = snapshot(env, obs)
            if args.render_preview:
                from PIL import Image
                for _ in range(3): frames = env.render()
                row['preview_frames'] = []
                for name, frame in zip(('main', 'wrist_left', 'wrist_right'), frames):
                    frame = np.asarray(frame)
                    assert frame.ndim == 3 and frame.std() > 5, name
                    file = f"{case['name']}-{name}.png"
                    Image.fromarray(frame.astype(np.uint8)).save(output / file)
                    row['preview_frames'].append({'path': file, 'shape': list(frame.shape), 'std': float(frame.std())})
            write(output / 'report.json', report)
            print('XY4_SAMPLE ' + json.dumps({'layout': case['name'], 'steps': args.steps,
                                             'peak_body_speed': row['peak_body_linear_speed']}), flush=True)
        write(output / 'environment-schema.json', env.get_recording_schema())
        report['isaac_warnings'] = env._isaac_warnings
        report['completed_diagnostic'] = True; code = 0
    except BaseException:
        report['error'] = traceback.format_exc(); traceback.print_exc()
    finally:
        write(output / 'report.json', report)
        if env is not None:
            env._fast_shutdown_on_close = True
            env.close()
    return code


if __name__ == '__main__':
    raise SystemExit(main())
