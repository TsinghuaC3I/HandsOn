"""Native reset goal readers. They never modify the environment or sample goals."""
import math

FRUIT_IDS = {'apple': 'apple', '011_banana': 'banana', 'orange': 'orange', 'pear': 'pear'}


def snap(value, allowed):
    value = float(value)
    assert math.isfinite(value)
    candidates = [x for x in allowed if abs(value - x) < 1e-4]
    assert len(candidates) == 1, ('Invalid native goal', value, allowed)
    return candidates[0]


def result(**values):
    return values, dict(values)


def fixed(env):
    return result(mode='fixed_instruction_or_observed_visual_target')


def empty(env):
    return {}, {}


def fruit(env):
    return result(fruit=FRUIT_IDS[env._target_object_name])


def ball(env):
    return result(ball=env._target_object_name.replace('_', ' '))


def fitting_cup(env):
    return result(requested_cup=env._ycb02_active_cup_name.split('_')[0].upper())


def tableware(env):
    return result(requested_tableware=env._target_object_name)


def timer(env):
    return result(minutes=env._ycb07_state.target_minutes, seconds=env._ycb07_state.target_seconds)


def fan(env):
    return result(target_angle_degrees=snap(env._p001_reset.target_open_deg, [60, 90, 120]))


def laptop(env):
    return result(target_angle_degrees=snap(env._fu015_reset.target_open_deg, [30, 60, 90, 120]))


def knife(env):
    if env._as013_target_direction not in (-1, 1):
        raise ValueError('Unknown knife motion')
    return result(motion='extend' if env._as013_target_direction == 1 else 'retract')


def dual_knobs(env):
    left, right = env._ka005_2_reset_sample.target_rad
    return result(left_angle_degrees=snap(math.degrees(left), [40, 60, 80]),
                  right_angle_degrees=snap(-math.degrees(right), [40, 60, 80]))


def fruit_logo(env):
    native_id = str(env._target_object_name)
    assert native_id in FRUIT_IDS
    values = {'fruit': FRUIT_IDS[native_id]}
    readback = dict(native_fruit_id=native_id, human_fruit=values['fruit'],
                    fixed_native_target=getattr(env, '_xy3_fixed_target', None))
    assert not readback['fixed_native_target'], 'Unexpected fixed native XY3 selector'
    return values, readback


def laptop_mouse(env):
    return result(laptop_angle_deg=snap(env._fu015_reset.target_open_deg, [30, 60, 90, 120]))


def mixer_knobs(env):
    state = env._ka005_2_state
    initial = [snap(math.degrees(float(x)), [-80, -60, -40, 0, 40, 60, 80]) for x in state.initial_q]
    targets = [snap(math.degrees(float(x)), [-80, -60, -40, 40, 60, 80]) for x in state.target_q]
    assert len(initial) == len(targets) == 2
    assert 45 <= targets[0] - initial[0] <= 120 and 45 <= initial[1] - targets[1] <= 120, 'Native GEN6 travel not supported'
    assert targets[0] in [40, 60, 80] and targets[1] in [-80, -60, -40]
    values = dict(left_target_deg=targets[0], right_target_deg=targets[1])
    readback = dict(initial_deg_left_right=initial, target_deg_left_right=targets,
                    positive_motion_left_negative_motion_right=True, target_units='signed native joint degrees')
    return values, readback
