"""Released native articulation routing and portable asset integrity."""
from pathlib import Path
import hashlib
import pytest
import yaml
from handson.tasks import CONFIG_MAPPING
from handson.sim.isaaclab.lw_native_articulation_contract import TASK_SPECS
from handson.sim.isaaclab.lw_native_articulation_scene import (
    LW_TASK_SPECS, LW_ASSET_DIRECTORY, validate_lw_asset_files,
)

ROOT = Path(__file__).resolve().parents[2]
RELEASED = {
    "t18_open_micropipette_box": ("18", ("Lid001_joint",), True),
    "t49_unfold_ruler": ("49", ("Prop001_joint", "Prop002_joint"), False),
    "eval_gen02_open_trash_bin": ("EVAL-Gen02", ("lid_hinge",), True),
}

EXPECTED_FILE_HASHES = {
    "assets/Foldingfan001/Foldingfan001.usd": (
        164916,
        "f0f3bbdabfc3131a7278b69f6a6216f658effb6f5e5a45e129c5d1eb6a4974f8",
    ),
    "assets/Foldingfan001/texture/T_Foldingfan001_BC001.png": (
        5013557,
        "12978ae73732fe3696e05f4922512d56f4d8a03018cbc309f49235fe941c7ee4",
    ),
    "assets/Foldingfan001/texture/T_Foldingfan001_N001.png": (
        3392385,
        "b2082ebfb2523cd5cb0ece1ebdd70a57697f168183951c48d692ca854e7dff44",
    ),
    "assets/Foldingfan001/texture/T_Foldingfan001_ORM001.png": (
        1963018,
        "d6fd28dbd8c4329b5e76fb9f41f763d02482aa0c94f6ccb39e6fbccbfdcf81c5",
    ),
    "assets/MicropipetteBox001/MicropipetteBox001.usd": (
        2177085,
        "6a8a143fe0a72e3b843c38e095057394ddb1b40e7dd1bf08c5185087bc031255",
    ),
    "assets/MicropipetteBox001/texture/T_MicropipetteBox001_BC001.png": (
        17299,
        "840bc48054dfce74ad6adc25ebe050d9be2b950a7e8bd3562bb0d13b06f1d9e2",
    ),
    "assets/MicropipetteBox001/texture/T_MicropipetteBox001_N001.png": (
        16902,
        "fe7d530dcef44fc4e06f6d6d78bcff2da66f629c86a047ac1947dcbeb9c3067d",
    ),
    "assets/MicropipetteBox001/texture/T_MicropipetteBox001_ORM001.png": (
        1661566,
        "4207a283d263697510d7c2a9c890e2b8aee63958b1f9e812ec865fa63ca2c705",
    ),
    "assets/Ruler035/Ruler035.usd": (
        33679,
        "55f3823f9ae46ba34f5bfcc12332aaa512cf99eb344e2db8bfa92a8825ca0369",
    ),
    "assets/Ruler035/texture/T_Ruler035_BC001.png": (
        2316649,
        "39d63de657476f0e9a90d1a3aa7538a90f7f2654748aaf29c89b06f364178fbf",
    ),
    "assets/Ruler035/texture/T_Ruler035_N001.png": (
        28323,
        "e2f800d12ca234ae2adbc7f8b79fa58998e0bb9a219cd43f31029d8fdfeb1ff8",
    ),
    "assets/Ruler035/texture/T_Ruler035_ORM001.png": (
        28298,
        "3d6fef33e1e7638e39acb8dba212f6cf9ec76b9175b0c0da7ba0b1dc78e23da0",
    ),
}

@pytest.mark.parametrize("task", RELEASED)
def test_public_config_scene_and_checker_agree(task):
    identity, joints, fixed = RELEASED[task]
    config = CONFIG_MAPPING[task]()
    contract = TASK_SPECS[task]
    scene = LW_TASK_SPECS[task]
    assert config.task_id == contract.task_id == scene.task_id == identity
    assert contract.dof_names == scene.dof_names == joints
    assert scene.fixed_root is fixed
    assert scene.source_path.is_file()
    payload = yaml.safe_load((ROOT / "configs/multi_task" / (task + ".yaml")).read_text())
    assert payload["env_name"] == task
    assert payload["prompt"] == config.task_prompt


@pytest.mark.parametrize("task", RELEASED)
@pytest.mark.parametrize("kwargs", [dict(sim_backend="mujoco"), dict(physics_backend="newton"),
    dict(hand_type="wuji2"), dict(arm_type="panda"), dict(hz=29),
    dict(control_dt=0.04), dict(physics_dt=0.01)])
def test_wrong_runtime_is_rejected_before_kit_start(task, kwargs):
    with pytest.raises(ValueError):
        CONFIG_MAPPING[task]().get_native_environment(**kwargs)


def test_native_assets_remain_hash_pinned_after_packaging():
    validated = set(validate_lw_asset_files())
    for relative, (size, digest) in EXPECTED_FILE_HASHES.items():
        path = LW_ASSET_DIRECTORY / relative
        assert path in validated
        assert path.stat().st_size == size
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_retired_fan_prototype_is_not_a_public_task():
    assert "g05_folding_fan_bimanual_open" not in CONFIG_MAPPING
    assert "t48_unfold_fan" in CONFIG_MAPPING
