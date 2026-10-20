"""HandsOn MuJoCo environment exports.

The environment modules import MuJoCo at module load time. Keep these exports
lazy so Isaac Sim helpers can import shared env utilities without requiring the
MuJoCo Python package inside Isaac's runtime environment.
"""

from __future__ import annotations

from importlib import import_module


_ENV_EXPORTS = {
    "PandaBimanualAssemblyGymEnv": ".panda_bimanual_assembly_env",
    "PandaBimanualHanoiGymEnv": ".panda_bimanual_hanoi_env",
    "PandaBimanualMicrowaveCookGymEnv": ".panda_bimanual_microwave_cook_env",
    "PandaBimanualPhotographGymEnv": ".panda_bimanual_photograph_env",
    "PandaBimanualUnlockIpadGymEnv": ".panda_bimanual_unlock_ipad_env",
    "PandaC01HammerNailGymEnv": ".panda_08_hammer_nail_env",
    "PandaC02SpoonMarblesGymEnv": ".panda_36_scoop_marbles_env",
    "PandaC04WaterPlantGymEnv": ".panda_50_water_plant_env",
    "PandaC05MoveMouseClickGymEnv": ".panda_14_move_and_click_mouse_env",
    "PandaC06BimanualPhotographGymEnv": ".panda_23_photograph_logo_env",
    "PandaClickMouseGymEnv": ".panda_click_mouse_env",
    "PandaD01BimanualNutBoltAssemblyGymEnv": ".panda_d01_bimanual_nut_bolt_assembly_env",
    "PandaD02BimanualMicrowaveCookGymEnv": ".panda_12_load_and_start_microwave_env",
    "PandaD03BimanualHandoffGymEnv": ".panda_d03_bimanual_handoff_env",
    "PandaD05ScissorsThreeCutsGymEnv": ".panda_43_squeeze_scissors_env",
    "PandaFoldGlassesGymEnv": ".panda_fold_glasses_env",
    "PandaHammerNailGymEnv": ".panda_hammer_nail_env",
    "PandaPickBucketGymEnv": ".panda_pick_bucket_env",
    "PandaPinchTongsGymEnv": ".panda_pinch_tongs_env",
    "PandaWaterPlantGymEnv": ".panda_water_plant_env",
}

__all__ = sorted(_ENV_EXPORTS)


def __getattr__(name: str):
    module_name = _ENV_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
