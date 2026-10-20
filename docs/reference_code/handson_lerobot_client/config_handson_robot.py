from dataclasses import dataclass
from pathlib import Path

from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("handson_robot_config")
@dataclass
class HandsOnRobotConfig(RobotConfig):
    config_path: Path
    seed: int
    randomize: bool
    randomize_dynamics: bool
    pad_state_dim46: bool
