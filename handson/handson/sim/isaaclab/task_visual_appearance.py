"""Task-owned object finishes, with shared HandsOn robot and room rendering."""
from __future__ import annotations
import importlib
from pathlib import Path
from typing import Any

PROFILE_NAME = "handson"


def selected_task_visual_profile(environment):
    from .default_appearance import _check_removed_switches
    _check_removed_switches()
    return PROFILE_NAME


def apply_task_visual_appearance(environment):
    """Preserve native object finishes without loading an alternate shared scene."""
    from .blender_visual_layer import physics_signature
    from .default_appearance import physics_material_signature
    selected_task_visual_profile(environment)
    task = str(getattr(environment, "task_name", ""))
    module = Path(__file__).resolve().parents[2] / "tasks" / task / "visual_appearance.py"
    if not task.isidentifier() or not module.is_file():
        return None
    before = physics_signature(environment._stage), physics_material_signature(environment._stage)
    objects = importlib.import_module(f"handson.tasks.{task}.visual_appearance").apply_visual_appearance(environment._stage, task_root_path="/World/Task")
    after = physics_signature(environment._stage), physics_material_signature(environment._stage)
    if before != after:
        raise RuntimeError("Task object appearance changed the physical scene")
    result = {"profile": PROFILE_NAME, "robot": {"links": 0, "owner": "default_appearance"}, "objects": objects}
    environment.task_visual_appearance_report = result
    return result


def update_task_visuals_for_render(environment: Any) -> None:
    """No cable mesh or CPU/GPU transfer in headless state-only collection."""
    if getattr(environment, "_deferred_camera_render", False):
        if getattr(environment, "_headless", False):
            return
        if getattr(environment, "_direct_hand_control", False) and getattr(environment, "_direct_external_preview", False):
            return
        # The operator viewport renders only at its existing cadence. Avoid
        # cable tessellation and tensor readback on intervening control frames.
        divisor = max(1, int(getattr(environment, "_direct_viewport_divisor", 1)))
        if (int(getattr(environment, "_direct_render_counter", 0)) + 1) % divisor:
            return
    if not getattr(environment, "_routed_wrist_cables", None):
        return
    from .wrist_cable_visuals import update_wrist_cables
    update_wrist_cables(environment)
