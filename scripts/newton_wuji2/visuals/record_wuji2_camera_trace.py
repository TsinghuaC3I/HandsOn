"""Actual sensor poses alongside photo preview; no camera edits."""
from pathlib import Path
import json

def install(env, output):
    native=env.render
    count=0
    target=Path(output)
    if target.exists():raise RuntimeError('Refusing to overwrite camera pose trace')
    def render(*args,**kwargs):
        nonlocal count
        frames=native(*args,**kwargs)
        cameras={}
        for name,camera in env._cameras.items():
            data=camera.data
            cameras[name]={'position_world_m':data.pos_w[0].detach().cpu().tolist(),
                'quat_wxyz_opengl':data.quat_w_opengl[0].detach().cpu().tolist(),
                'intrinsic_matrix':data.intrinsic_matrices[0].detach().cpu().tolist()}
        mounts={}
        for side,robot in env._robots.items():
            names=list(robot.body_names);name=f'{side}_wrist_fisheye_mount'
            if name in names:mounts[name]=robot.data.body_pose_w[0,names.index(name)].detach().cpu().tolist()
        with target.open('a') as f:f.write(json.dumps({'capture_call':count,'cameras':cameras,'mount_body_pose':mounts,'robot_quaternion_xyzw':env._isaaclab_uses_xyzw})+'\n')
        count+=1
        return frames
    env.render=render
