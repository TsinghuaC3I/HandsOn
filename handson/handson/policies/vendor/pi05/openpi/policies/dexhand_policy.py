"""Wuji bimanual joint-space policy transforms."""

from __future__ import annotations

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


ACTION_DIM = 54
ARM_DOF = 7
HAND_DOF = 20


def delta_action_mask(hand_absolute: bool) -> np.ndarray:
    """Use arm deltas and, when requested, absolute dexterous-hand targets."""
    if hand_absolute:
        return transforms.make_bool_mask(
            ARM_DOF,
            -HAND_DOF,
            ARM_DOF,
            -HAND_DOF,
        )
    return transforms.make_bool_mask(ACTION_DIM)


def _parse_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class DexHandInputs(transforms.DataTransformFn):
    """Map real/sim Wuji joint54 rows into Pi0.5's common model inputs."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        if np.shape(data["state"])[-1] != ACTION_DIM:
            raise ValueError(
                f"state dim {np.shape(data['state'])[-1]} != {ACTION_DIM}"
            )
        if "actions" in data and np.shape(data["actions"])[-1] != ACTION_DIM:
            raise ValueError(
                f"actions dim {np.shape(data['actions'])[-1]} != {ACTION_DIM}"
            )
        result = {
            "state": data["state"],
            "image": {
                "base_0_rgb": _parse_image(data["base_image"]),
                "left_wrist_0_rgb": _parse_image(data["left_wrist_image"]),
                "right_wrist_0_rgb": _parse_image(data["right_wrist_image"]),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }
        if "actions" in data:
            result["actions"] = data["actions"]
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        return result


@dataclasses.dataclass(frozen=True)
class DexHandOutputs(transforms.DataTransformFn):
    """Return absolute 54-D joint targets after inverse delta conversion."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :ACTION_DIM])}
