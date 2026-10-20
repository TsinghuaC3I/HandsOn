# coding=utf-8
# Copyright (C) 2026 Tencent.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configuration class for the Hy-VLA policy.

The on-disk format of ``config.json`` is a flat JSON object whose keys
are the dataclass fields (plus a ``"type"`` discriminator).
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Type, TypeVar

from huggingface_hub import hf_hub_download

CONFIG_NAME = "config.json"


class FeatureType(str, Enum):
    STATE = "STATE"
    VISUAL = "VISUAL"
    ENV = "ENV"
    ACTION = "ACTION"


@dataclass
class PolicyFeature:
    type: FeatureType
    shape: tuple[int, ...]

    def __post_init__(self):
        if isinstance(self.type, str):
            self.type = FeatureType(self.type)
        self.shape = tuple(self.shape)


@dataclass
class OptimizerPreset:
    lr: float
    betas: tuple[float, float]
    eps: float
    weight_decay: float


@dataclass
class CosineDecayWithWarmupSchedulerPreset:
    peak_lr: float
    decay_lr: float
    num_warmup_steps: int
    num_decay_steps: int


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {f.name: _to_jsonable(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


T = TypeVar("T", bound="HyVLAConfig")

@dataclass
class HyVLAConfig:
    """Configuration for the Hy-VLA policy."""

    # --- Core I/O structure ------------------------------------------------
    chunk_size: int = 50
    n_action_steps: int = 50

    # --- Image features --------------------------------------------------
    image_features: dict | None = field(default=None)

    max_state_dim: int = 32
    max_action_dim: int = 32

    resize_imgs_with_padding: tuple[int, int] = (224, 224)

    # Number of all-(-1) padding cameras to append when ``image_features``
    # expects more cameras than the batch provides. Default 0 = no padding.
    empty_cameras: int = 0

    # Tokenizer
    tokenizer_max_length: int = 64

    # Action-expert projector width
    proj_width: int = 1024

    # Flow-matching steps
    num_steps: int = 10

    # Attention utils
    use_cache: bool = True
    attention_implementation: str = "eager"  # one of: eager, fa2, flex

    # Finetuning settings
    freeze_vision_encoder: bool = False
    train_expert_only: bool = False
    train_state_proj: bool = True

    # Optimizer / scheduler defaults
    optimizer_lr: float = 2.5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    vis_attn: bool = False

    # Path to the upstream Hy-Embodied VLM (HF repo id or local dir).
    # Used as a fallback when the VLA ckpt is not self-contained, and as
    # the source for raw-VLM bootstrap (``pretrain_source`` in
    # {``scratch``, ``vlm``}).
    vlm_model_path: str = "tencent/HY-Embodied-0.5"

    # Embedded upstream VLM AutoConfig payload. When set, the VLA ckpt is
    # self-contained and builds its inner VLM tower without contacting
    # any external Hy-Embodied repo. ``None`` means fall back to
    # ``vlm_model_path``.
    vlm_config_dict: dict | None = None

    # --- MEM video-encoder switch (paper arXiv:2603.03596v1) --------------
    use_video_encoder: bool = False
    spacetime_layer_stride: int = 4
    past_drop_layer: int | None = None

    # Visual-segment attention mask scope. See
    # ``HyVLA._apply_visual_segment_mask``:
    #   * False -- patch-only: clear cross-image visibility between
    #     image-patch tokens; split rows stay on the causal pathway.
    #   * True  -- full-segment isolation: clear the whole visual
    #     segment's outward visibility and re-enable bidirectional
    #     visibility within each segment. Required to reproduce the
    #     released RoboTwin post-training ckpt.
    visual_segment_isolation: bool = False

    # Class-level constant (no type annotation, so dataclasses treats it
    # as a class variable and does NOT serialize it).
    action_feature = PolicyFeature(type=FeatureType.ACTION, shape=(20,))

    # Path the config was loaded from, populated by from_pretrained.
    pretrained_path: str | None = dataclasses.field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )

        # Default image_features when the user / ckpt did not provide them.
        if self.image_features is None:
            self.image_features = {
                "observation.images.top_head": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
                "observation.images.hand_left": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
                "observation.images.hand_right": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
            }

    def get_optimizer_preset(self) -> OptimizerPreset:
        return OptimizerPreset(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerPreset:
        return CosineDecayWithWarmupSchedulerPreset(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def type(self) -> str:
        # Discriminator written into ``config.json``.
        return "hy"

    # Load-side fields kept on the dataclass for ergonomic kwargs but NOT
    # written to ``config.json`` (avoids baking absolute local paths into
    # released ckpts).
    _NON_PERSISTED_FIELDS = frozenset({"pretrained_path", "vlm_model_path"})

    def _save_pretrained(self, save_directory: str | Path) -> None:
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {}
        for f in fields(self):
            if f.name in self._NON_PERSISTED_FIELDS:
                continue
            payload[f.name] = _to_jsonable(getattr(self, f.name))
        payload["type"] = self.type
        with open(save_directory / CONFIG_NAME, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=4, ensure_ascii=False)

    def save_pretrained(self, save_directory: str | Path) -> None:
        self._save_pretrained(save_directory)

    @classmethod
    def from_pretrained(
        cls: Type[T],
        pretrained_name_or_path: str | Path,
        *,
        force_download: bool = False,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        token: str | bool | None = None,
        **overrides: Any,
    ) -> T:
        model_id = str(pretrained_name_or_path)
        if Path(model_id).is_dir():
            config_file = os.path.join(model_id, CONFIG_NAME)
            if not os.path.exists(config_file):
                raise FileNotFoundError(f"{CONFIG_NAME} not found in {model_id}")
        else:
            config_file = hf_hub_download(
                repo_id=model_id,
                filename=CONFIG_NAME,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                token=token,
                local_files_only=local_files_only,
            )

        with open(config_file, "r", encoding="utf-8") as fp:
            data = json.load(fp)

        allowed = {f.name for f in fields(cls)}
        kwargs: dict[str, Any] = {}
        for k, v in data.items():
            if k == "type":
                continue
            if k not in allowed:
                # Forward compat: silently drop fields the current code
                # version doesn't know about.
                continue
            if k == "image_features" and isinstance(v, dict):
                v = {kk: PolicyFeature(**vv) if isinstance(vv, dict) else vv for kk, vv in v.items()}
            elif k == "resize_imgs_with_padding" and isinstance(v, list):
                v = tuple(v)
            elif k == "optimizer_betas" and isinstance(v, list):
                v = tuple(v)
            kwargs[k] = v
        kwargs.update(overrides)

        instance = cls(**kwargs)
        instance.pretrained_path = model_id
        return instance


__all__ = [
    "FeatureType",
    "PolicyFeature",
    "OptimizerPreset",
    "CosineDecayWithWarmupSchedulerPreset",
    "HyVLAConfig",
]
