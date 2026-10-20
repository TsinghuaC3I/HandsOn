from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml

import openpi.models.pi0_config as pi0_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders

with open("config.yaml") as f:
    config = yaml.safe_load(f)

WANDB_ENABLED = config["wandb_enabled"]
PRETRAINED_MODEL_PATH = config["pretrained_model_path"]
PRETRAINED_MODEL_ACTION_DIM_44_PATH = config["pretrained_model_action_dim_44_path"]
PRETRAINED_MODEL_ACTION_DIM_52_PATH = config.get(
    "pretrained_model_action_dim_52_path",
    PRETRAINED_MODEL_ACTION_DIM_44_PATH,
)
PRETRAINED_MODEL_ACTION_DIM_54_PATH = config.get(
    "pretrained_model_action_dim_54_path",
    "/mnt/pfs/luotianwei/handson/checkpoints/pi05_base_action_dim_54/params",
)
DATASET_ROOT = Path(config["dataset_root"])
RAND_FULL_DATASET_ROOT = Path(config["rand_full_dataset_root"])
CKPTS_ROOT = Path(config["ckpts_root"])
RAND_FULL_CKPTS_ROOT = Path(config["rand_full_ckpts_root"])
BATCH_SIZE = config["batch_size"]
SINGLE_ARM_STEPS = config["single_arm_steps"]
DUAL_ARM_STEPS = config["dual_arm_steps"]


@dataclass
class HandsOnConfig:
    name: str
    checkpoint_base_dir: str
    data_root: Path
    single_arm: bool
    base_img_name: str | None = None
    wrist_left_img_name: str | None = None
    wrist_right_img_name: str | None = None
    policy_action_dim: int | None = None
    policy_state_dim: int | None = None
    joint54: bool = False
    hand_absolute: bool = False
    normalized_clip: float | None = None


TrainConfigs: list[HandsOnConfig] = [
    # rand_obj datasets
    HandsOnConfig(
        name="bimanual_assembly",
        checkpoint_base_dir=f"{CKPTS_ROOT}",
        data_root=Path(f"{DATASET_ROOT}/bimanual_assembly"),
        single_arm=False,
    ),
    HandsOnConfig(
        name="bimanual_microwave_cook",
        checkpoint_base_dir=f"{CKPTS_ROOT}",
        data_root=Path(f"{DATASET_ROOT}/bimanual_microwave_cook"),
        single_arm=False,
    ),
    HandsOnConfig(
        name="bimanual_unlock_ipad",
        checkpoint_base_dir=f"{CKPTS_ROOT}",
        data_root=Path(f"{DATASET_ROOT}/bimanual_unlock_ipad"),
        single_arm=False,
    ),
    HandsOnConfig(
        name="bimanual_hanoi",
        checkpoint_base_dir=f"{CKPTS_ROOT}",
        data_root=Path(f"{DATASET_ROOT}/bimanual_hanoi"),
        single_arm=False,
    ),
    HandsOnConfig(
        name="bimanual_photograph",
        checkpoint_base_dir=f"{CKPTS_ROOT}",
        data_root=Path(f"{DATASET_ROOT}/bimanual_photograph"),
        single_arm=False,
    ),
    HandsOnConfig(
        name="fold_glasses",
        checkpoint_base_dir=f"{CKPTS_ROOT}",
        data_root=Path(f"{DATASET_ROOT}/fold_glasses"),
        single_arm=True,
    ),
    HandsOnConfig(
        name="pick_bucket",
        checkpoint_base_dir=f"{CKPTS_ROOT}",
        data_root=Path(f"{DATASET_ROOT}/pick_bucket"),
        single_arm=True,
    ),
    HandsOnConfig(
        name="pick_bucket_wuji1",
        checkpoint_base_dir=f"{CKPTS_ROOT}",
        data_root=DATASET_ROOT.parent
        / "handson_wuji_lerobot_datasets"
        / "pick_bucket",
        single_arm=True,
        policy_action_dim=26,
    ),
    HandsOnConfig(
        name="t24_place_ball_in_bowl_wuji1",
        checkpoint_base_dir=f"{CKPTS_ROOT}",
        data_root=DATASET_ROOT.parent
        / "handson_wuji_lerobot_datasets"
        / "t24_place_ball_in_bowl",
        single_arm=True,
        policy_action_dim=26,
    ),
    HandsOnConfig(
        name="s01_valid_wrist_corrected_wuji1",
        checkpoint_base_dir=f"{CKPTS_ROOT}",
        data_root=Path(
            "/mnt/pfs/luotianwei/handson/datasets/lerobot_v21/"
            "s01_valid_wrist_corrected_single_arm"
        ),
        single_arm=True,
        policy_action_dim=26,
    ),
    HandsOnConfig(
        name="s01_valid_wrist_corrected_unified_bimanual_wuji1",
        checkpoint_base_dir=f"{CKPTS_ROOT}",
        data_root=Path(
            "/mnt/pfs/luotianwei/handson/datasets/lerobot_v21/"
            "s01_valid_wrist_corrected_unified_bimanual"
        ),
        single_arm=False,
        base_img_name="observation.images.front",
        wrist_left_img_name="observation.images.wrist_left",
        wrist_right_img_name="observation.images.wrist_right",
        policy_action_dim=52,
    ),
    HandsOnConfig(
        name="s01_valid_wrist_corrected_joint54_wuji1",
        checkpoint_base_dir=f"{CKPTS_ROOT}",
        data_root=Path(
            "/mnt/pfs/luotianwei/handson/datasets/lerobot_v21/"
            "s01_valid_wrist_corrected_joint54"
        ),
        single_arm=False,
        policy_action_dim=54,
        joint54=True,
        hand_absolute=True,
    ),
    HandsOnConfig(
        name="s01_valid_wrist_corrected_unified_bimanual_v2_wuji1",
        checkpoint_base_dir=f"{CKPTS_ROOT}",
        data_root=Path(
            "/mnt/pfs/luotianwei/handson/datasets/lerobot_v21/"
            "s01_valid_wrist_corrected_unified_bimanual"
        ),
        single_arm=False,
        base_img_name="observation.images.front",
        wrist_left_img_name="observation.images.wrist_left",
        wrist_right_img_name="observation.images.wrist_right",
        policy_action_dim=52,
        normalized_clip=3.0,
    ),
    HandsOnConfig(
        name="physx_v3_actual_eef52_wuji1",
        checkpoint_base_dir=f"{CKPTS_ROOT}",
        data_root=Path(
            "/mnt/pfs/luotianwei/handson/datasets/lerobot_v21/"
            "physx_v3_actual_eef52"
        ),
        single_arm=False,
        base_img_name="observation.images.front",
        wrist_left_img_name="observation.images.wrist_left",
        wrist_right_img_name="observation.images.wrist_right",
        policy_action_dim=52,
        policy_state_dim=52,
        normalized_clip=3.0,
    ),
    HandsOnConfig(
        name="physx_v3_command_eef52_wuji1",
        checkpoint_base_dir=f"{CKPTS_ROOT}",
        data_root=Path(
            "/mnt/pfs/luotianwei/handson/datasets/lerobot_v21/"
            "physx_v3_command_eef52"
        ),
        single_arm=False,
        base_img_name="observation.images.front",
        wrist_left_img_name="observation.images.wrist_left",
        wrist_right_img_name="observation.images.wrist_right",
        policy_action_dim=52,
        policy_state_dim=52,
        normalized_clip=3.0,
    ),
    HandsOnConfig(
        name="physx_v3_actual_joint54_wuji1",
        checkpoint_base_dir=f"{CKPTS_ROOT}",
        data_root=Path(
            "/mnt/pfs/luotianwei/handson/datasets/lerobot_v21/"
            "physx_v3_actual_joint54"
        ),
        single_arm=False,
        base_img_name="observation.images.front",
        wrist_left_img_name="observation.images.wrist_left",
        wrist_right_img_name="observation.images.wrist_right",
        policy_action_dim=54,
        policy_state_dim=54,
        joint54=True,
        hand_absolute=True,
        normalized_clip=3.0,
    ),
    HandsOnConfig(
        name="s01_valid_wrist_corrected_joint54_v2_wuji1",
        checkpoint_base_dir=f"{CKPTS_ROOT}",
        data_root=Path(
            "/mnt/pfs/luotianwei/handson/datasets/lerobot_v21/"
            "s01_valid_wrist_corrected_joint54"
        ),
        single_arm=False,
        policy_action_dim=54,
        joint54=True,
        hand_absolute=True,
        normalized_clip=3.0,
    ),
    HandsOnConfig(
        name="s01_valid_wrist_corrected_joint54_cmd_v3_wuji1",
        checkpoint_base_dir=f"{CKPTS_ROOT}",
        data_root=Path(
            "/mnt/pfs/luotianwei/handson/datasets/lerobot_v21/"
            "s01_valid_wrist_corrected_joint54_cmd_v3"
        ),
        single_arm=False,
        policy_action_dim=54,
        joint54=True,
        hand_absolute=True,
    ),
    HandsOnConfig(
        name="s01_valid_wrist_corrected_joint54_cmd_v3_v2_wuji1",
        checkpoint_base_dir=f"{CKPTS_ROOT}",
        data_root=Path(
            "/mnt/pfs/luotianwei/handson/datasets/lerobot_v21/"
            "s01_valid_wrist_corrected_joint54_cmd_v3"
        ),
        single_arm=False,
        policy_action_dim=54,
        joint54=True,
        hand_absolute=True,
        normalized_clip=3.0,
    ),
    HandsOnConfig(
        name="water_plant",
        checkpoint_base_dir=f"{CKPTS_ROOT}",
        data_root=Path(f"{DATASET_ROOT}/water_plant"),
        single_arm=True,
    ),
    HandsOnConfig(
        name="click_mouse",
        checkpoint_base_dir=f"{CKPTS_ROOT}",
        data_root=Path(f"{DATASET_ROOT}/click_mouse"),
        single_arm=True,
        base_img_name="observation.images.ego_right",
    ),
    HandsOnConfig(
        name="hammer_nail",
        checkpoint_base_dir=f"{CKPTS_ROOT}",
        data_root=Path(f"{DATASET_ROOT}/hammer_nail"),
        single_arm=True,
    ),
    HandsOnConfig(
        name="pinch_tongs",
        checkpoint_base_dir=f"{CKPTS_ROOT}",
        data_root=Path(f"{DATASET_ROOT}/pinch_tongs"),
        single_arm=True,
    ),

    # rand_full datasets
    HandsOnConfig(
        name="bimanual_assembly_rand_full",
        checkpoint_base_dir=f"{RAND_FULL_CKPTS_ROOT}",
        data_root=Path(f"{RAND_FULL_DATASET_ROOT}/bimanual_assembly"),
        single_arm=False,
        base_img_name="observation.images.random_camera",
    ),
    HandsOnConfig(
        name="bimanual_microwave_cook_rand_full",
        checkpoint_base_dir=f"{RAND_FULL_CKPTS_ROOT}",
        data_root=Path(f"{RAND_FULL_DATASET_ROOT}/bimanual_microwave_cook"),
        single_arm=False,
        base_img_name="observation.images.random_camera",
    ),
    HandsOnConfig(
        name="bimanual_unlock_ipad_rand_full",
        checkpoint_base_dir=f"{RAND_FULL_CKPTS_ROOT}",
        data_root=Path(f"{RAND_FULL_DATASET_ROOT}/bimanual_unlock_ipad"),
        single_arm=False,
        base_img_name="observation.images.random_camera",
    ),
    HandsOnConfig(
        name="bimanual_hanoi_rand_full",
        checkpoint_base_dir=f"{RAND_FULL_CKPTS_ROOT}",
        data_root=Path(f"{RAND_FULL_DATASET_ROOT}/bimanual_hanoi"),
        single_arm=False,
        base_img_name="observation.images.random_camera",
    ),
    HandsOnConfig(
        name="bimanual_photograph_rand_full",
        checkpoint_base_dir=f"{RAND_FULL_CKPTS_ROOT}",
        data_root=Path(f"{RAND_FULL_DATASET_ROOT}/bimanual_photograph"),
        single_arm=False,
        base_img_name="observation.images.random_camera",
    ),
    HandsOnConfig(
        name="fold_glasses_rand_full",
        checkpoint_base_dir=f"{RAND_FULL_CKPTS_ROOT}",
        data_root=Path(f"{RAND_FULL_DATASET_ROOT}/fold_glasses"),
        single_arm=True,
        base_img_name="observation.images.random_camera",
    ),
    HandsOnConfig(
        name="pick_bucket_rand_full",
        checkpoint_base_dir=f"{RAND_FULL_CKPTS_ROOT}",
        data_root=Path(f"{RAND_FULL_DATASET_ROOT}/pick_bucket"),
        single_arm=True,
        base_img_name="observation.images.random_camera",
    ),
    HandsOnConfig(
        name="water_plant_rand_full",
        checkpoint_base_dir=f"{RAND_FULL_CKPTS_ROOT}",
        data_root=Path(f"{RAND_FULL_DATASET_ROOT}/water_plant"),
        single_arm=True,
        base_img_name="observation.images.random_camera",
    ),
    HandsOnConfig(
        name="click_mouse_rand_full",
        checkpoint_base_dir=f"{RAND_FULL_CKPTS_ROOT}",
        data_root=Path(f"{RAND_FULL_DATASET_ROOT}/click_mouse"),
        single_arm=True,
        base_img_name="observation.images.random_camera",
    ),
    HandsOnConfig(
        name="hammer_nail_rand_full",
        checkpoint_base_dir=f"{RAND_FULL_CKPTS_ROOT}",
        data_root=Path(f"{RAND_FULL_DATASET_ROOT}/hammer_nail"),
        single_arm=True,
        base_img_name="observation.images.random_camera",
    ),
    HandsOnConfig(
        name="pinch_tongs_rand_full",
        checkpoint_base_dir=f"{RAND_FULL_CKPTS_ROOT}",
        data_root=Path(f"{RAND_FULL_DATASET_ROOT}/pinch_tongs"),
        single_arm=True,
        base_img_name="observation.images.random_camera",
    ),
    HandsOnConfig(
        name="multi_task",
        checkpoint_base_dir=f"{CKPTS_ROOT}",
        data_root=Path("path/to/multi_task/dataset"),
        single_arm=False,
        base_img_name="observation.images.base",
        wrist_left_img_name="observation.images.wrist1",
        wrist_right_img_name="observation.images.wrist2",
    ),
]


def make_single_arm_config(cfg: HandsOnConfig):
    # import in function to avoid circular import
    from .config import DataConfig  # noqa: PLC0415
    from .config import SingleArmDataConfig  # noqa: PLC0415
    from .config import TrainConfig  # noqa: PLC0415

    return TrainConfig(
        name=cfg.name,
        exp_name=cfg.name + datetime.now().strftime("%Y%m%d"),  # noqa: DTZ005
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=30,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            max_token_len=250,
        ),
        data=SingleArmDataConfig(
            root=cfg.data_root,
            repo_id="local_repo",
            base_img_name=cfg.base_img_name,
            base_config=DataConfig(prompt_from_task=True),
            action_dim=cfg.policy_action_dim or 22,
        ),
        batch_size=BATCH_SIZE,
        num_workers=4,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=30,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            max_token_len=250,
        ).get_freeze_filter(),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader(PRETRAINED_MODEL_PATH),
        num_train_steps=SINGLE_ARM_STEPS,
        save_interval=10000,
        wandb_enabled=WANDB_ENABLED,
        checkpoint_base_dir=cfg.checkpoint_base_dir,
    )


def make_dual_arm_config(cfg: HandsOnConfig):
    # import in function to avoid circular import
    from .config import DataConfig  # noqa: PLC0415
    from .config import DexHandDataConfig  # noqa: PLC0415
    from .config import DualArmDataConfig  # noqa: PLC0415
    from .config import TrainConfig  # noqa: PLC0415

    action_dim = cfg.policy_action_dim or 44
    pretrained_path = (
        PRETRAINED_MODEL_ACTION_DIM_54_PATH
        if action_dim == 54
        else (
            PRETRAINED_MODEL_ACTION_DIM_52_PATH
            if action_dim == 52
            else PRETRAINED_MODEL_ACTION_DIM_44_PATH
        )
    )
    data_factory = (
        DexHandDataConfig(
            root=cfg.data_root,
            repo_id="local_repo",
            base_img_name=cfg.base_img_name or "observation.images.front",
            wrist_left_img_name=(
                cfg.wrist_left_img_name or "observation.images.wrist_left"
            ),
            wrist_right_img_name=(
                cfg.wrist_right_img_name or "observation.images.wrist_right"
            ),
            hand_absolute=cfg.hand_absolute,
            normalized_clip=cfg.normalized_clip,
            base_config=DataConfig(prompt_from_task=True),
        )
        if cfg.joint54
        else DualArmDataConfig(
            root=cfg.data_root,
            repo_id="local_repo",
            base_img_name=cfg.base_img_name,
            wrist_left_img_name=cfg.wrist_left_img_name,
            wrist_right_img_name=cfg.wrist_right_img_name,
            action_dim=action_dim,
            state_dim=cfg.policy_state_dim,
            normalized_clip=cfg.normalized_clip,
            base_config=DataConfig(prompt_from_task=True),
        )
    )

    return TrainConfig(
        name=cfg.name,
        exp_name=cfg.name + datetime.now().strftime("%Y%m%d"),  # noqa: DTZ005
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=action_dim,
            action_horizon=30,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            max_token_len=250,
        ),
        data=data_factory,
        batch_size=BATCH_SIZE,
        num_workers=4,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=action_dim,
            action_horizon=30,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            max_token_len=250,
        ).get_freeze_filter(),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader(pretrained_path),
        num_train_steps=DUAL_ARM_STEPS,
        save_interval=10000,
        wandb_enabled=WANDB_ENABLED,
        checkpoint_base_dir=cfg.checkpoint_base_dir,
    )


def get_handson_configs():
    cfgs = []
    for cfg in TrainConfigs:
        if cfg.single_arm:
            cfgs.append(make_single_arm_config(cfg))
        else:
            cfgs.append(make_dual_arm_config(cfg))
    return cfgs
