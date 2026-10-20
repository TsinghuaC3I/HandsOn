"""Install approved paper settings before construction and verify live readback."""
import os
from .contracts import require, write_new
from . import settings

def prepare(task, seed, output):
    from handson.sim.isaaclab import b_eval_setting
    b_eval_setting.selected.cache_clear()
    os.environ.pop("HANDSON_B_EVAL_SELECTION", None)
    if task not in settings.task_ids():
        return None
    selection = settings.select_setting(seed, task, 0)
    path = output / "SETTING_SELECTION.json"
    write_new(path, selection)
    os.environ["HANDSON_B_EVAL_SELECTION"] = str(path.resolve())
    return selection

def configure_solver(selection):
    if selection is None:
        return
    import handson.sim.isaaclab.abc_collection_env as module
    solver = selection["physics"]["solver"]
    original_caps = module._physx_solver_iteration_caps
    original_buffers = module._physx_gpu_buffer_kwargs
    def caps(task):
        if task == selection["task"]:
            return solver["max_position_iteration_count"], solver["max_velocity_iteration_count"]
        return original_caps(task)
    def buffers(task):
        values = original_buffers(task)
        if task == selection["task"]:
            values.update({k: v for k, v in solver.items() if k.startswith("gpu_")})
        return values
    module._physx_solver_iteration_caps = caps
    module._physx_gpu_buffer_kwargs = buffers

def verify_b_setting(env, selection):
    """Read live fields; do not invoke success/failure predicates for logging."""
    from handson.sim.isaaclab.b_eval_setting import selected
    require(selected() == selection, 'Native source did not load this setting')
    task, profile = selection['short_id'], selection['setting_id']
    observed = dict(task=env.task_name,setting_id=profile,environment_class=type(env).__module__)
    require(env.task_name == selection['task'], 'B task mismatch')
    if task == 'BC011':
        wanted = {'black_pen_negative_x':'cc_by_black_pen_high_quality/pen.usda',
                  'the_pen_positive_x':'cc_by_the_pen/pen.usda'}[profile]
        observed['pen_asset'] = env._bc011_pen_variant
        require(observed['pen_asset'] == wanted, 'Wrong pen asset/direction')
    elif task == 'D02':
        observed['active_food'] = env._d02_active_food
        require(observed['active_food'] == {'cracker_box':'003_cracker_box',
                'pudding_box':'008_pudding_box'}[profile], 'Wrong microwave food')
    elif task == 'KA005-2':
        import math
        observed['knobs'] = env._ka005_2_reset_sample.as_dict()
        for field in ('initial','target'):
            require(all(abs(math.degrees(a)-b)<1e-6 for a,b in zip(
                observed['knobs'][field+'_rad'],selection['profile'][field+'_degrees'])),
                'Live knob angles differ from the approved joint tuple')
    elif task == 'BC014':
        from handson.sim.isaaclab.b320_bimanual_env import _BC014_FLOW_MAX_POUR_SECONDS
        observed['pour_cap_seconds'] = _BC014_FLOW_MAX_POUR_SECONDS
        require(observed['pour_cap_seconds'] == 4., 'Wrong cumulative pour limit')
    elif task == 'BC017':
        import sys
        observed['drawer_travel'] = sys.modules[type(env).__module__].BC017_DRAWER_TRAVEL_M
        require(abs(observed['drawer_travel']-(.16 if profile=='small_upper_d6' else .248))<1e-9,
                'Wrong drawer geometry/travel')
        suffix = {'small_upper_d6':'b320_bc017_small_d6_env',
                  'large_upper_contact':'b320_bc017_large_upper_contact_env',
                  'large_six_contact':'b320_bc017_env'}[profile]
        require(type(env).__module__.endswith('.'+suffix), 'Wrong drawer contact implementation')
    elif task == 'D03':
        from pxr import UsdPhysics, PhysxSchema
        prim = env._find_rigid_body_prim('d03_target_root')
        require(prim is not None, 'D03 target missing')
        material_path = ('/World/PhysicsMaterials/HandsOnD03B04SmallBlockStability'
                         if profile == 'friction_4_4' else
                         '/World/PhysicsMaterials/HandsOnD03TargetContactSafety')
        material = env._stage.GetPrimAtPath(material_path)
        require(bool(material), 'D03 selected contact material missing')
        api, body = UsdPhysics.MaterialAPI(material), PhysxSchema.PhysxRigidBodyAPI(prim)
        observed['friction'] = [api.GetStaticFrictionAttr().Get(), api.GetDynamicFrictionAttr().Get()]
        require(observed['friction'] == [4., 4. if profile == 'friction_4_4' else 2.5],
                'D03 selected friction differs')
        if profile == 'friction_4_2p5':
            observed['velocity_limits'] = [body.GetMaxDepenetrationVelocityAttr().Get(),
                body.GetMaxLinearVelocityAttr().Get(), body.GetMaxAngularVelocityAttr().Get()]
            require(all(abs(a-b)<1e-6 for a,b in zip(observed['velocity_limits'], [.5,1.,30.])),
                    'D03 selected velocity limits differ')
    observed['verified'] = True
    return observed
