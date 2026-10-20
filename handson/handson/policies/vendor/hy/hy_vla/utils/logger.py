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

"""Lightweight wandb helpers used by ``hy_vla.train``."""

import os
import json
from pathlib import Path

import wandb
from omegaconf import open_dict


def save_wandb(wandb, save_dir):
    """Persist enough to ``resume`` this run later (entity/project/name/id)."""
    wandb_info = {
        "entity": wandb.run.entity,
        "project": wandb.run.project,
        "name": wandb.run.name,
        "id": wandb.run.id,
    }
    with open(os.path.join(save_dir, "wandb_run_info.json"), "w") as f:
        json.dump(wandb_info, f, indent=2)


def initialize_wandb(cfg):
    """Init the wandb run, transparently resuming when ``cfg.resume_ckpt`` is set.

    Honors ``WANDB_MODE`` env var (``online`` / ``offline`` / ``disabled``);
    defaults to ``online`` to match the historical behaviour. Setting it to
    ``offline`` is the recommended route when the local wandb account has no
    access to the configured ``cfg.wandb_entity`` (403 on upsertBucket).
    """
    if cfg.resume_ckpt:
        save_dir = Path(cfg.resume_ckpt)
        with open(save_dir.parent / "wandb_run_info.json", "r") as f:
            wandb_info = json.load(f)
        cfg.wandb_entity = wandb_info["entity"]
        cfg.wandb_project = wandb_info["project"]
        cfg.exp_name = wandb_info["name"]
        with open_dict(cfg):
            cfg["wandb_id"] = wandb_info["id"]

    wandb_mode = os.environ.get("WANDB_MODE", "online")

    wandb.init(
        entity=cfg.wandb_entity,
        project=cfg.wandb_project,
        name=f"{cfg.exp_name}",
        id=cfg.wandb_id if cfg.resume_ckpt else None,
        mode=wandb_mode,
        resume="must" if cfg.resume_ckpt else None,
    )

    return cfg