#!/usr/bin/env python3
"""GEN4 physical grasp feasibility probe using only public robot actions.

This is an engineering command trajectory, not human teleoperation evidence.
It does not write robot/object states or replace task success predicates.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=0)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[3]
    sys.argv=[sys.argv[0],'--portable-root',str(root/'.gen4-local/kit'),
        '--/log/file='+str((args.output/'kit.log').resolve())]
    os.environ.update(HANDSON_ENABLE_ISAACSIM_NEWTON='0',
        HANDSON_ISAACLAB_DEFER_CAMERA_RENDER='0',HANDSON_ISAACLAB_LIVE_MAIN_CAMERA_ONLY='0',
        HANDSON_RECORD_PHYSX_GROUND_TRUTH='1')
    from handson.sim.isaaclab.gen4_env import HandsOnGEN4IsaacLabEnv
    from handson.tasks.eval_gen04_pass_glue_stick import contract as c
    from shape_queries import query_final_shapes
    result={'task':'GEN4','seed':args.seed,'command_source':'diagnostic public pose and hand actions',
        'human_teleop':False,'phases':[],'motion':[],'contacts':[],'exceptions':[],'completed':False}
    env=None
    def write():
        (args.output/'contact.json').write_text(json.dumps(result,indent=2)+'\n')
    def array(x):
        return x.detach().cpu().numpy() if hasattr(x,'detach') else np.asarray(x)
    def query(environment,phase,allowed=()):
        row=query_final_shapes(environment,phase,allowed_hand_contacts=allowed)
        result['phases'].append(row);write()
        assert not row['unapproved_pairs'] and not row['query_errors'] and row['callback_count'],f'{phase}: unapproved physical overlap'
    class CheckedEnv(HandsOnGEN4IsaacLabEnv):
        def _configure_task_simulation_settings(self):
            super()._configure_task_simulation_settings()
            import omni.physx
            self._sim.get_physics_context().set_enable_scene_query_support(True)
            omni.physx.get_physx_interface().force_load_physics_from_usd()
            query(self,'authored_before_physics')
        def _apply_task_pose_randomization(self):
            super()._apply_task_pose_randomization()
            before=self._sim.current_time;self._sim.forward()
            assert self._sim.current_time==before
            query(self,'seeded_before_physics')
    try:
        env=CheckedEnv(seed=args.seed,randomize=True,image_obs=True,headless=True,
            render_mode='rgb_array',physics_backend='physx',isaac_device='cuda:0',
            arm_type='rokae',hand_type='wuji1',render_resolution=(576,1024))
        # Cameras stay available for explicitly requested phase snapshots.
        # Skip per-step RGB in this diagnostic so action cadence is measurable.
        env.image_obs=False
        query(env,'post_reset')
        from PIL import Image
        for camera,rgb in zip(('main','wrist_left','wrist_right'),env.render()):
            Image.fromarray(np.asarray(rgb,dtype=np.uint8)).save(args.output/f'reset_{camera}.png')
        target=env._poses[c.GLUE_BODY].copy()
        from handson.sim.isaaclab.bimanual_assembly_env import _matrix_to_quat_wxyz
        robot=env._robots['left']
        actual_home=env._robot_body_pose_matrix(robot,env._ee_body_indices['left'])
        start=np.r_[actual_home[:3,3],_matrix_to_quat_wxyz(actual_home[:3,:3])]
        raised=start.copy();raised[2]=target[2]+.20
        # Rotate the palm into a vertical side grasp. All motion goes through
        # the public absolute-pose action and the ordinary robot controller.
        roll=np.array([[1.,0.,0.],[0.,0.,-1.],[0.,1.,0.]])
        above=start.copy();above[:3]=target[:3]+[-.24,-.057,.20]
        above[3:]=_matrix_to_quat_wxyz(roll@actual_home[:3,:3])
        lowered=above.copy();lowered[2]=target[2]+.05
        grasp=lowered.copy();grasp[0]=target[0]-.10
        lifted=grasp.copy();lifted[2]+=.09
        open_hand=np.zeros(20)
        preshaped=open_hand.copy()
        closed=np.tile([1.0,0.0,1.0,1.0],5)
        result['waypoints']={'start':start.tolist(),'raised':raised.tolist(),'above':above.tolist(),'lowered':lowered.tolist(),'grasp':grasp.tolist(),'lifted':lifted.tolist(),'preshaped_hand':preshaped.tolist(),'closed_hand':closed.tolist()}
        previous_pose=start.copy();previous_hand=open_hand.copy()
        phases=[('raise_before_translation',raised,preshaped,90),
            ('rotate_and_approach',above,preshaped,120),('lower_beside_target',lowered,preshaped,120),
            ('approach_from_side',grasp,preshaped,90),
            ('close_hand',grasp,closed,90),('sustained_grasp',grasp,closed,60),
            ('lift',lifted,closed,90),('hold_lifted',lifted,closed,60)]
        from PIL import Image
        for name,pose,hand,steps in phases:
            contact_frames=0
            for frame in range(1,steps+1):
                alpha=frame/steps
                endpoint=pose.copy()
                if np.dot(previous_pose[3:],endpoint[3:])<0:endpoint[3:]*=-1
                command_pose=(1-alpha)*previous_pose+alpha*endpoint
                command_pose[3:]/=np.linalg.norm(command_pose[3:])
                command_hand=(1-alpha)*previous_hand+alpha*hand
                _,_,terminated,truncated,info=env.step({'left':np.r_[command_pose,command_hand]})
                metrics=env._gen4_metrics()
                contact_frames+=int(metrics['left_target_contact'])
                trace=env._physx_ground_truth_trace;snap=trace.snapshot()
                count=int(array(snap['physx_contact_signature_pair_count']).ravel()[0])
                canonical=sorted(str(x['path']) for x in trace._contact_report_path_table)
                for ids,values in zip(array(snap['physx_contact_signature_pair_path_indices'])[:count],array(snap['physx_contact_signature_pair_metrics'])[:count]):
                    result['contacts'].append({'phase':name,'frame':frame,'paths':[canonical[int(i)-1] if 1<=int(i)<=len(canonical) else str(i) for i in ids],
                        'penetration_depth_m':float(values[0]),'normal_impulse_ns':float(values[1])})
                robot=env._robots['left']
                result['motion'].append({'phase':name,'frame':frame,'time_s':float(env._sim.current_time),
                    'command':np.r_[command_pose,command_hand].tolist(),
                    'glue_pose':env._poses[c.GLUE_BODY].tolist(),'metrics':metrics,
                    'left_palm_world_matrix':env._robot_body_pose_matrix(robot,env._ee_body_indices['left']).tolist(),
                    'left_joint_positions':array(robot.data.joint_pos).tolist(),
                    'left_joint_names':list(robot.joint_names),
                    'terminated':bool(terminated),'truncated':bool(truncated)})
                if terminated or truncated:raise RuntimeError(f'{name}: unexpected episode end: {info}')
            previous_pose=pose.copy();previous_hand=hand.copy()
            write()
            for camera,rgb in zip(('main','wrist_left','wrist_right'),env.render()):
                Image.fromarray(np.asarray(rgb,dtype=np.uint8)).save(args.output/f'{name}_{camera}.png')
            query(env,name,allowed=('left',))
            print('[GEN4] Contact phase: '+json.dumps({'phase':name,'left_contact_frames':contact_frames,'steps':steps,'metrics':metrics}),flush=True)
        result['left_pickup_proven']=env._gen4_handoff.stage>=1
        result['completed']=True
    except BaseException:
        result['exceptions'].append(traceback.format_exc());raise
    finally:
        write()
        if env is not None:
            env._fast_shutdown_on_close=True;env.close()


if __name__=='__main__':main()
