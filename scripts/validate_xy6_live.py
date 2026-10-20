#!/usr/bin/env python3
"""XY6 native geometry/reset and short control checks, adapted from XY5.

Uses actual tensor poses for queries, preserves physical task state, and never
labels short diagnostic recordings as successful human demonstrations.
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
from xy6_native_geometry import shapes, live_shapes, overlaps, snapshot, plain, reason, BODY_NAMES, TASK_NAME, native_pour_geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(20)))
    parser.add_argument('--steps', type=int, default=60)
    parser.add_argument('--tracking', action='store_true')
    parser.add_argument('--render-preview', action='store_true')
    parser.add_argument('--save-camera-input', action='store_true',
                        help='Save an unsuccessful short diagnostic for the shared post-renderer')
    parser.add_argument('--stable-origin', action='store_true')
    parser.add_argument('--skip-boundaries', action='store_true')
    parser.add_argument('--only-boundaries', action='store_true')
    args = parser.parse_args()
    assert not (args.save_camera_input and args.render_preview), 'Camera inputs must be captured with cameras deferred'
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    from handson.sim.isaaclab.eval_comb06_season_and_stir_env import HandsOnXY6IsaacLabEnv
    from collection_ui.server import ISAAC_COLLECTION_REALTIME_ENV
    os.environ.update(ISAAC_COLLECTION_REALTIME_ENV)
    os.environ['HANDSON_ENABLE_ISAACSIM_NEWTON'] = '0'
    os.environ['HANDSON_RECORD_PHYSX_GROUND_TRUTH'] = '1'
    if args.stable_origin:
        os.environ['HANDSON_STABLE_RECORDING_ORIGIN'] = '1'
    os.environ['HANDSON_ISAACLAB_DEFER_CAMERA_RENDER'] = '0' if args.render_preview else '1'
    os.environ['HANDSON_ISAACLAB_LIVE_MAIN_CAMERA_ONLY'] = '0'
    report = {'task_id':'XY6','task':TASK_NAME, 'isaacsim_version':importlib.metadata.version('isaacsim'),
              'executable':sys.executable, 'physical_gpu':os.environ.get('TASKDEV_PHYSICAL_GPU'),
              'events':[], 'rows':[], 'operator_verified':False, 'successful_episodes_verified':False}
    assert report['isaacsim_version'].startswith('5.1.')
    assert report['physical_gpu'] == '3', 'Use the authorized private GPU namespace'
    def save():
        (output/'report.json').write_text(json.dumps(plain(report), indent=2, allow_nan=False)+'\n')
    def shape_event(env, label, first=False):
        inventory=shapes(env._stage) if first else live_shapes(env,label)
        query=overlaps(env._stage,inventory)
        by_path={r['path']:r for r in inventory}
        approved, unknown=[],[]
        for a,b in query['pairs']:
            why=reason(a,b,by_path) if a in by_path and b in by_path else None
            (approved if why else unknown).append({'paths':[a,b],'reason':why})
        event={'label':label,'inventory':inventory,'overlap':query,'approved':approved,'unapproved':unknown}
        report['events'].append(event);save()
        assert not query['errors'] and query['positive_control_passed'], event
        for side in ('right','left'):
            for finger in range(1,6):
                for link in range(1,5):
                    assert any(r['owner'] and r['owner'].endswith(f'/{side}_finger{finger}_link{link}') for r in inventory), (side,finger,link)
        assert not unknown, unknown
        print('XY6_QUERY '+json.dumps({'label':label,'shapes':len(inventory),'unapproved':len(unknown)}),flush=True)
    class Probe(HandsOnXY6IsaacLabEnv):
        layout_override=None
        def _sample_xy6_layout(self):
            if self.layout_override is not None:
                return {n:np.asarray(p,dtype=np.float64) for n,p in self.layout_override.items()}
            return super()._sample_xy6_layout()
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
    env=None; code=1
    try:
        env=Probe(seed=0,randomize=True,image_obs=args.render_preview,headless=True,
                  physics_backend='physx',render_mode='rgb_array',isaac_device=args.device,
                  arm_type='rokae',hand_type='wuji1')
        import isaaclab
        lab=next(p/'VERSION' for p in Path(isaaclab.__file__).resolve().parents if (p/'VERSION').is_file())
        report['isaaclab_version']=lab.read_text().strip()
        assert report['isaaclab_version']=='2.3.2'
        assert env._control_dt==1/30 and env._sim_dt==1/120
        report['timing']={'physics_dt':env._physics_dt,'control_dt':env._control_dt,'substeps':env._n_substeps}
        report['online_cameras']=list(env._cameras)
        from pxr import Usd, UsdPhysics
        env._xy6_probe_static_paths=sorted({r['owner'] or r['path'] for r in shapes(env._stage)
            if r['type']!='Plane' and (r['owner'] is None or
                bool(UsdPhysics.RigidBodyAPI(env._stage.GetPrimAtPath(r['owner'])).GetKinematicEnabledAttr().Get()))})
        assert env._xy6_probe_static_paths
        robot_paths=[str(p.GetPath()) for p in Usd.PrimRange(env._stage.GetPrimAtPath('/World/HandsOnRobots'))
                     if p.HasAPI(UsdPhysics.RigidBodyAPI) and '/CenterRig/' not in str(p.GetPath())]
        for kind,sensors in (('support',[env._role_body_path(n) for n in BODY_NAMES]),('robot',robot_paths)):
            view=env._robots['right']._physics_sim_view.create_rigid_contact_view(sensors,
                filter_patterns=[env._xy6_probe_static_paths for _ in sensors],max_contact_data_count=2048)
            setattr(env,f'_xy6_probe_{kind}_view',view)
        cases=[(f'seed_{seed}',seed,None) for seed in args.seeds]
        if args.only_boundaries:cases=[]
        if not args.skip_boundaries:
            cases += [(r['name'],0,r['poses_wxyz']) for r in json.loads((ROOT/'artifacts/xy6/cpu_audit.json').read_text())['boundary_layouts_for_live_validation']]
        for label,seed,layout in cases:
            env.layout_override=layout
            obs,info=env.reset(seed=seed)
            first={n:p.copy() for n,p in env._xy6_reset_poses.items()}
            obs,info=env.reset(seed=seed)
            assert all(np.array_equal(p,env._xy6_reset_poses[n]) for n,p in first.items())
            row={'label':label,'sampled_layout':first,'repeatable':True,'reset':snapshot(env,obs),'samples':[]}
            report['rows'].append(row);save();shape_event(env,label+'_reset')
            if args.stable_origin:
                from handson.sim.isaaclab.recording_origin import stabilize_recording_origin,stable_recording_origin_kwargs_for_task
                profile=stable_recording_origin_kwargs_for_task(TASK_NAME)
                obs,info=stabilize_recording_origin(env,obs,info,**profile)
                row['origin_profile']=profile;row['recording_origin']=info['physical_recording_origin']
                row['accepted_origin']=snapshot(env,obs);save()
            command_origin=np.r_[env._right_tcp_pose,env._left_tcp_pose,env._right_hand,env._left_hand]
            assert command_origin.shape==env.action_space.shape
            camera_inputs=[]
            if args.save_camera_input:
                from handson.tasks.obs_adapters import HandsOnObsAdapter
                from handson.tasks.eval_comb06_season_and_stir.config import TaskConfig
                adapter=HandsOnObsAdapter(env,proprio_keys=TaskConfig().proprio_keys)
                assert not env._cameras and env._deferred_camera_render
            for step in range(args.steps):
                command=command_origin.copy()
                if args.tracking:
                    scale=min(1.,step/12.) if step<36 else max(0.,1.-(step-36)/12.)
                    command[[2,9]]+=.025*scale
                    command[14:]+=.12*scale
                if args.save_camera_input:
                    before=adapter.observation(obs)
                    before={k:np.asarray(before[k]).copy() for k in ('state','xy6_hidden_state','xy6_object_poses')}
                start=time.monotonic()
                obs,reward,terminated,truncated,info=env.step(command)
                if args.save_camera_input:
                    after=adapter.observation(obs)
                    camera_inputs.append({'action':command.copy(),**before,
                        'next_state':np.asarray(after['state']).copy(),
                        'next_xy6_hidden_state':np.asarray(after['xy6_hidden_state']).copy(),
                        'next_xy6_object_poses':np.asarray(after['xy6_object_poses']).copy(),
                        'succeed':bool(info.get('succeed',False)),
                        'terminal':bool(terminated or truncated)})
                if args.tracking or step in (0,1,args.steps-1):
                    row['samples'].append({'step':step,'wall_seconds':time.monotonic()-start,
                        'physics':snapshot(env,obs),'terminated':terminated,'truncated':truncated,'info':info})
                if terminated or truncated:raise RuntimeError('Diagnostic unexpectedly terminated')
                if args.tracking and step in (35,48):shape_event(env,label+f'_tracking_{step}')
            row['settled']=snapshot(env,obs);shape_event(env,label+'_settled')
            if args.tracking:
                row['native_pour_geometry']=native_pour_geometry(env);save()
            if args.render_preview:
                from PIL import Image
                for _ in range(3):frames=env.render()
                row['frames']=[]
                for name,frame in zip(('main','wrist_left','wrist_right'),frames):
                    frame=np.asarray(frame);assert frame.ndim==3 and frame.std()>5,name
                    file=f'{label}-{name}.png';Image.fromarray(frame.astype(np.uint8)).save(output/file)
                    row['frames'].append({'path':file,'shape':list(frame.shape),'std':float(frame.std())})
            if args.save_camera_input:
                from xy6_camera_probe import save_camera_input
                row['camera_input']=save_camera_input(output/(label+'-camera-input'),camera_inputs,env,seed)
            save();print('XY6_SAMPLE '+label,flush=True)
        (output/'environment-schema.json').write_text(json.dumps(plain(env.get_recording_schema()),indent=2)+'\n')
        report['completed_diagnostic']=True;code=0
    except BaseException as exc:
        report['diagnostics']=getattr(exc,'diagnostics',None)
        report['error']=traceback.format_exc();traceback.print_exc()
    finally:
        try:
            save()
        finally:
            if env is not None:
                env._fast_shutdown_on_close=True;env.close()
    return code

if __name__=='__main__':
    raise SystemExit(main())
