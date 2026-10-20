#!/usr/bin/env python3
"""GEN4 authored/settled final-shape audit. Requires an authorized idle GPU.

This is a diagnostic runner, never a substitute for ordinary UI collection,
operator grasp/contact sweeps or fresh saved-trajectory replay.
"""
from __future__ import annotations
import argparse
import copy
from itertools import product
import json
import os
from pathlib import Path
import sys
import traceback
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--boundary', type=int, choices=range(8))
    parser.add_argument('--matrix', action='store_true')
    parser.add_argument('--control-steps', type=int, default=60)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    root_output=args.output
    matrix_report={'task':'GEN4','cases':[],'completed':False}
    if args.matrix:
        args.seed=0
        args.boundary=None
        args.output=root_output/'seed_00'
        args.output.mkdir(exist_ok=True)
    # SimulationApp forwards unknown command-line options to Kit. Keep its
    # data/cache/log folders in this task's own evidence directory.
    private=Path(__file__).resolve().parents[3]/'.gen4-local/kit'
    sys.argv = [sys.argv[0], '--portable-root', str(private), '--/log/file='+str((args.output/'kit.log').resolve())]
    os.environ['HANDSON_ENABLE_ISAACSIM_NEWTON'] = '0'
    os.environ['HANDSON_RECORD_PHYSX_GROUND_TRUTH'] = '1'
    os.environ['HANDSON_ISAACLAB_DEFER_CAMERA_RENDER'] = '1'
    from handson.tasks.eval_gen04_pass_glue_stick import contract as c
    from handson.sim.isaaclab import gen4_env, gen4_scene
    from handson.sim.isaaclab.gen4_env import HandsOnGEN4IsaacLabEnv
    geometry=gen4_scene.load_geometry()
    expected_shapes=72+len(geometry['glue']['collision_paths'])+geometry['tray']['collision_count']+sum(x['collision_count'] for x in geometry['distractors'].values())
    matrix_report['expected_shape_count']=expected_shapes
    if args.matrix or args.boundary is not None:
        original = gen4_scene.sample_layout
        def boundary_layout(*a, **kw):
            layout = original(*a, **kw)
            if args.boundary is None: return layout
            dx, dy, yaw = list(product((-.03,.03),(-.03,.03),(-15,15)))[args.boundary]
            layout[c.GLUE_BODY].update(xy=(np.asarray(c.GLUE_XY)+[dx,dy]).tolist(), yaw_degrees=yaw)
            return layout
        gen4_scene.sample_layout = gen4_env.sample_layout = boundary_layout
    result = {'task':'GEN4', 'seed':args.seed, 'boundary':args.boundary,
        'method':'PhysX final collision-shape queries before first reset and after settling; real contact separation and motion readback',
        'shape_api_source':'https://docs.omniverse.nvidia.com/kit/docs/omni_physics/104.0/source/extensions/omni.physx/docs/index.html',
        'allowlist':['Compound shapes of the same physical owner',
            'Internal overlaps within an explicitly fixed robot mounting assembly',
            'Explicit authored joint/filter collision exclusions',
            'Declared GEN4 body supported by the table; live depth must remain <=2 mm'],
        'phases':[], 'contacts':[], 'motion':[], 'exceptions':[]}
    result_template=copy.deepcopy(result)
    env = None
    subscription = None
    step_number = 0

    def array(value):
        if hasattr(value,'detach'): value=value.detach().cpu().numpy()
        return np.asarray(value)

    def capture_tensor_contacts(environment):
        trace=environment._physx_ground_truth_trace
        snap=trace.snapshot()
        valid=int(array(snap['physx_contact_signature_valid']).ravel()[0])
        count=int(array(snap['physx_contact_signature_pair_count']).ravel()[0])
        canonical=sorted(str(x['path']) for x in trace._contact_report_path_table)
        def path(index):
            return canonical[int(index)-1] if 1<=int(index)<=len(canonical) else f'unknown:{int(index)}'
        for ids,metrics in zip(array(snap['physx_contact_signature_pair_path_indices'])[:count],array(snap['physx_contact_signature_pair_metrics'])[:count]):
            result['contacts'].append({'control_step':step_number,'source':'canonical PhysX contact signature',
                'paths':[path(i) for i in ids], 'penetration_depth_m':float(metrics[0]),
                'normal_impulse_ns':float(metrics[1]),'friction_impulse_ns':float(metrics[2])})
        return {'valid':valid,'pair_count':count,'source':int(array(snap['physx_contact_capture_source']).ravel()[0])}

    def capture(environment, phase):
        from shape_queries import query_final_shapes
        report=query_final_shapes(environment,phase)
        result['phases'].append(report)
        result['expected_shape_count']=expected_shapes
        write()
        if report['shape_count']!=expected_shapes or report['query_errors'] or report['unapproved_pairs'] or not report['callback_count']:
            raise RuntimeError(f"GEN4 {phase}: final-shape audit needs correction; unapproved={len(report['unapproved_pairs'])}, errors={len(report['query_errors'])}, callbacks={report['callback_count']}")

    def write():
        (args.output/'audit.json').write_text(json.dumps(result,indent=2,default=str)+'\n')

    class AuditedEnv(HandsOnGEN4IsaacLabEnv):
        def _apply_task_pose_randomization(self):
            super()._apply_task_pose_randomization()
            before=float(self._sim.current_time)
            self._sim.forward()
            self._refresh_object_poses()
            assert float(self._sim.current_time)==before
            capture(self,'seeded_pose_before_physics')
            result['reset_layout']=self._gen4_reset_layout
            result['reset_physical_poses']={n:self._poses[n].tolist() for n in (c.GLUE_BODY,c.TRAY_BODY,*c.DISTRACTORS)}
            result['reset_robot_states']={side:{key:array(getattr(robot.data,key)).tolist()
                for key in ('joint_pos','joint_vel','joint_pos_target')} for side,robot in self._robots.items()}

        def _configure_task_simulation_settings(self):
            super()._configure_task_simulation_settings()
            import omni.physx
            self._sim.get_physics_context().set_enable_scene_query_support(True)
            # Load/cook final USD into PhysX without advancing time. This hook
            # runs after geometry/contact/material setup and before sim.reset.
            omni.physx.get_physx_interface().force_load_physics_from_usd()
            capture(self,'authored_before_first_sim_reset')

    def check_window(phase):
        nonlocal step_number
        capture(env,phase)
        result['timing']={'control_dt':env._control_dt,'physics_dt':env._physics_dt,
            'substeps':env._n_substeps,'start_time_s':float(env._sim.current_time)}
        assert np.isclose(env._control_dt,1/c.CONTROL_HZ) and np.isclose(env._physics_dt,1/c.PHYSICS_HZ)
        assert env._n_substeps==4
        for step_number in range(1,args.control_steps+1):
            _,_,terminated,truncated,info=env.step(np.zeros(54))
            result['motion'].append({'control_step':step_number,'physics_time_s':float(env._sim.current_time),
                'poses':{n:env._poses[n].tolist() for n in (c.GLUE_BODY,*c.DISTRACTORS)},
                'velocities':{n:np.asarray(env._physics_body_velocity(n)).tolist() for n in (c.GLUE_BODY,*c.DISTRACTORS)},
                'robots':{side:{'joint_velocity_max':float(np.abs(array(robot.data.joint_vel)).max()),
                    'joint_target_error_max':float(np.abs(array(robot.data.joint_pos_target)-array(robot.data.joint_pos)).max())} for side,robot in env._robots.items()},
                'contact_signature':capture_tensor_contacts(env),
                'terminated':bool(terminated),'truncated':bool(truncated)})
            if terminated or truncated: raise RuntimeError('GEN4 idle stability window ended unexpectedly')
        capture(env,'post_two_second_idle_window')
        result['physical_time_s']=float(env._sim.current_time)
        result['timing']['measured_idle_duration_s']=result['physical_time_s']-result['timing']['start_time_s']
        assert np.isclose(result['timing']['measured_idle_duration_s'],args.control_steps/c.CONTROL_HZ,atol=1e-5)
        result['operator_contact_pose_matrix']='pending'
        result['ordinary_platform_and_replay']='pending'

        result['reset_hidden_state']=env._compute_observation()['gen4_handoff_state'].tolist()
        assert result['reset_hidden_state']==[0]*10
        assert all(m['contact_signature']['valid']==1 and m['contact_signature']['pair_count']>=11 for m in result['motion'])
        assert max(x.get('penetration_depth_m',0) for x in result['contacts'])<=.002
        write()
        if args.matrix:
            matrix_report['cases'].append({'name':args.output.name,'seed':args.seed,'boundary':args.boundary,
                'report':str((args.output/'audit.json').relative_to(root_output)),
                'phases':len(result['phases']),'completed':True})
            (root_output/'matrix.json').write_text(json.dumps(matrix_report,indent=2)+'\n')
            print('[GEN4] Reset matrix case complete: '+args.output.name,flush=True)

    try:
        env=AuditedEnv(seed=args.seed,randomize=True,image_obs=False,headless=True,
            render_mode='rgb_array',physics_backend='physx',isaac_device='cuda:0',
            arm_type='rokae',hand_type='wuji1')
        check_window('post_constructor_reset')
        if args.matrix:
            cases=[(i,None,f'seed_{i:02}') for i in range(1,20)]+[(0,i,f'boundary_{i}') for i in range(8)]+[(0,None,'seed_00_repeat')]
            for seed,boundary,name in cases:
                args.seed=seed;args.boundary=boundary
                args.output=root_output/name;args.output.mkdir(exist_ok=True)
                result=copy.deepcopy(result_template)
                result.update(seed=seed,boundary=boundary)
                env.reset(seed=seed)
                check_window('post_manual_seeded_reset')
            matrix_report['completed']=True
            (root_output/'matrix.json').write_text(json.dumps(matrix_report,indent=2)+'\n')
    except BaseException:
        result['exceptions'].append(traceback.format_exc())
        raise
    finally:
        write()
        subscription=None
        if env is not None:
            # Same shutdown path used by the ordinary native Isaac recorder.
            env._fast_shutdown_on_close=True
            env.close()


if __name__=='__main__': main()
