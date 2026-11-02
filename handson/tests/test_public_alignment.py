"""Regressions for concrete failures found by the 66-task RTX 5090 audit."""
import hashlib
from pathlib import Path



def test_renamed_transfer_assets_keep_verified_checksums():
    from handson.sim.isaaclab.gen3_hinged_trash_bin_scene import validate_gen3_asset_files
    from handson.sim.isaaclab.gen6_audio_mixer_scene import validate_asset_files
    assert validate_gen3_asset_files()
    assert validate_asset_files()


def test_carton_observer_accepts_the_packaged_native_source():
    from handson.evaluation.force_observer import SOURCE_SHA
    from handson.sim.isaaclab import b320_bimanual_env
    data = Path(b320_bimanual_env.__file__).read_bytes()
    assert hashlib.sha256(data).hexdigest() == SOURCE_SHA
    historical = data.replace(b'handson', b'dexomni').replace(b'HandsOn', b'DexOmni').replace(b'HANDSON', b'DEXOMNI')
    assert hashlib.sha256(historical).hexdigest() == '91a1bd42e67df088bcc755261ba92d1f38043ed15acada073d860328bd3934ec'


def test_paper_placement_and_mouse_factories_use_the_frozen_runtime():
    from handson.tasks import CONFIG_MAPPING
    from handson.tasks.registry import TASKS
    for task in ('t26_place_fruit_in_bucket', 't25_place_cup_on_plate',
                 't24_place_ball_in_bowl', 't14_move_and_click_mouse'):
        env = CONFIG_MAPPING[task]().get_environment(policy_mode=True, launch_isaac=False)
        try:
            assert type(env.unwrapped).__module__ == 'handson.sim.isaaclab.core_box_env'
            assert TASKS[task].trace_environment_module == 'core_box_base'
            assert env.action_space.shape == (52,)
        finally:
            env.close()
