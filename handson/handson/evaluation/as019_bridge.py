"""Passive API view of AS019's native InteractiveScene; raw reset/step stay bound to raw env."""
from types import SimpleNamespace
from .contracts import require,write_new

class ReadbackView:
    def __init__(self,raw):object.__setattr__(self,'_raw',raw)
    def __setattr__(self,name,value):setattr(self._raw,name,value)
    def __getattr__(self,name):
        raw=self._raw
        if name=='_robots':return {s:raw._scene[s+'_robot'] for s in ('right','left')}
        if name=='_cameras':return {s:raw._scene[s] for s in ('head_camera','wrist_left_camera','wrist_right_camera')}
        if name=='_task_body_views':return {n:o.root_physx_view for n,o in raw._scene.rigid_objects.items()}
        if name=='_task_articulation_views':return {n:(o.root_physx_view,[]) for n,o in raw._scene.articulations.items() if n not in ('right_robot','left_robot')}
        if name=='_task_reset_articulation_states':
            extra=[(n,o) for n,o in raw._scene.articulations.items() if n not in ('right_robot','left_robot')]
            require(not extra,'AS019 passive bridge needs an explicit extra-articulation map')
            return []
        if name=='_target_object_name':
            require('peg' in raw._scene.rigid_objects,'Native screwdriver absent')
            return 'peg'
        if name=='_spec':
            return SimpleNamespace(task_name=raw.task_name,target_objects=tuple(raw._scene.rigid_objects),
                origin='actual native InteractiveScene rigid-object registry; no ABC task spec exists')
        if name in ('_direct_hand_control','_deferred_arm_render','_simplified_post_ik'):
            # The native recorder uses false for absent optional implementation modes.
            # This bridge does not add these attributes to the raw environment.
            return bool(getattr(raw,name,False))
        return getattr(raw,name)
    def _set_deferred_robot_body_hidden(self,hidden):
        require(not hidden,'AS019 has no deferred body-render path')
        return 0


def install(job,output):
    import handson.sim.isaaclab.as019_screwdriver_handover_env as m
    cls=m.AS019ScrewdriverHandoverEnv;native=cls._make_native_scene_cfg
    def make(env,*args,**kwargs):
        previous=(env.image_obs,env._deferred_camera_render)
        try:
            env.image_obs=True;env._deferred_camera_render=False
            return native(env,*args,**kwargs)
        finally:env.image_obs,env._deferred_camera_render=previous
    cls._make_native_scene_cfg=make
    write_new(output/'AS019_SCENE_BRIDGE.json',dict(task=job['task'],
        aliases_source='actual native InteractiveScene',native_reset_and_step_bound_to_raw_environment=True,
        camera_factory_only_image_obs=True,no_teacher_or_guidance=True,
        absent_optional_control_modes='native recorder false default',physics_and_success_unchanged=True))


def wrap(env):
    require(type(env).__name__=='AS019ScrewdriverHandoverEnv','Automatic or guidance AS019 must not run')
    return ReadbackView(env)
