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

"""Hy-VLA: a manipulation policy on top of the Hy-Embodied VLA backbone.

The package follows the *transformers single-file modeling* convention,
split into two halves:

* ``hy_vla.hunyuan_vl_mot`` -- vendor copy of the public Hy-Embodied
  release (``tencent/HY-Embodied-0.5``). Importing this sub-package
  registers ``HunYuanVLMoTConfig`` / ``HunYuanVLMoTForConditionalGeneration``
  / ``HunYuanVLMoTProcessor`` into the transformers ``Auto*`` registries
  so loading public HY-Embodied checkpoints works with vanilla
  ``transformers >= 4.57`` without ``trust_remote_code=True``.
* ``hy_vla`` (top level) -- the action-flow expert sitting on top of
  the VLM. Three files: ``configuration_hy_vla`` (``HyVLAConfig``),
  ``modeling_dual_tower`` (``HyDualTower`` -- the VLM + expert dual-tower
  container with a shared-attention forward) and ``modeling_hy_vla``
  (the ``HyVLA`` entrypoint with HuggingFace-style ``from_pretrained``
  / ``save_pretrained``).

``HyVLAConfig`` is a plain dataclass that round-trips itself to
``config.json``; ``HyVLA`` is a plain ``torch.nn.Module`` that
round-trips itself to a single ``model.safetensors`` file via
``safetensors.torch``.
"""

from hy_vla.configuration_hy_vla import (
    HyVLAConfig,
    FeatureType,
    PolicyFeature,
    OptimizerPreset,
    CosineDecayWithWarmupSchedulerPreset,
)
from hy_vla.modeling_hy_vla import HyVLA, HyVLAFlowMatching
from hy_vla.modeling_dual_tower import HyDualTowerConfig, HyDualTower

__all__ = [
    "HyVLAConfig",
    "HyVLA",
    "HyVLAFlowMatching",
    "HyDualTowerConfig",
    "HyDualTower",
    "FeatureType",
    "PolicyFeature",
    "OptimizerPreset",
    "CosineDecayWithWarmupSchedulerPreset",
]
