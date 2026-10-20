"""Stable catalog for optional simulation scenes exposed by collection UI."""

from __future__ import annotations

import os
from typing import Any, Mapping


SIMULATION_SCENE_ENV = "HANDSON_SIMULATION_SCENE"
NO_SIMULATION_SCENE = ""
FU015_MODERN_KITCHEN_SCENE = "fu015_modern_kitchen"
HS005_MODERN_APARTMENT_SINK_SCENE = "hs005_modern_apartment_sink"
HS005_MODERN_KITCHEN_SINK_SCENE = "hs005_modern_kitchen_sink"
HS005_REALISTIC_KITCHEN_SCENE_SINK_SCENE = (
    "hs005_realistic_kitchen_scene_sink"
)
HS005_APPARTEMENT_KITCHEN_SINK_SCENE = "hs005_appartement_kitchen_sink"
HS005_TASK_NAME = "t37_scrub_bottle"

_SCENE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "value": FU015_MODERN_KITCHEN_SCENE,
        "label": "FU015 - Realistic Modern Kitchen",
        "description": "CC BY realistic kitchen visuals, automatically aligned to the task's standard tabletop.",
        "compatible_tasks": None,
    },
    {
        "value": HS005_MODERN_APARTMENT_SINK_SCENE,
        "label": "HS005 · 写实厨房水槽",
        "description": "HS005 的 Modern Apartment 水槽、瓶子、刷子与清洁用品场景。",
        "compatible_tasks": (HS005_TASK_NAME,),
    },
    {
        "value": HS005_MODERN_KITCHEN_SINK_SCENE,
        "label": "HS005 · Modern Kitchen 自带水槽",
        "description": "dylanheyes 的实体厨房；使用该场景自己的双槽、龙头和独立碰撞。",
        "compatible_tasks": (HS005_TASK_NAME,),
    },
    {
        "value": HS005_REALISTIC_KITCHEN_SCENE_SINK_SCENE,
        "label": "HS005 · Realistic Kitchen 自带水槽",
        "description": "Katydid 的写实厨房；使用该场景自己的双槽、龙头和独立碰撞。",
        "compatible_tasks": (HS005_TASK_NAME,),
    },
    {
        "value": HS005_APPARTEMENT_KITCHEN_SINK_SCENE,
        "label": "HS005 · Appartement 自带水槽",
        "description": "SrMonteiro 的实体公寓厨房；使用该场景自己的单槽、龙头和独立碰撞。",
        "compatible_tasks": (HS005_TASK_NAME,),
    },
)

SIMULATION_SCENE_VALUES = frozenset(
    {NO_SIMULATION_SCENE, *(str(spec["value"]) for spec in _SCENE_SPECS)}
)


def simulation_scene_options(task_name: str) -> list[dict[str, str]]:
    """Return UI-safe scene choices compatible with ``task_name``."""

    options: list[dict[str, str]] = []
    for spec in _SCENE_SPECS:
        compatible = spec["compatible_tasks"]
        if compatible is not None and task_name not in compatible:
            continue
        options.append(
            {
                "value": str(spec["value"]),
                "label": str(spec["label"]),
                "description": str(spec["description"]),
            }
        )
    if task_name == "eval_comb06_season_and_stir":
        for option in options:
            if option["value"] == FU015_MODERN_KITCHEN_SCENE:
                option["label"] = "FU015 - Realistic Modern Kitchen"
                option["description"] = (
                    "CC BY realistic kitchen visual layer, aligned to the task's standard tabletop."
                )
    return options


def simulation_scene_is_compatible(scene: str, task_name: str) -> bool:
    """Return whether a known optional scene can be used with one task."""

    if scene == NO_SIMULATION_SCENE:
        return True
    return any(
        scene == spec["value"]
        and (
            spec["compatible_tasks"] is None
            or task_name in spec["compatible_tasks"]
        )
        for spec in _SCENE_SPECS
    )


def selected_simulation_scene(
    environ: Mapping[str, str] | None = None,
    *,
    default: str = NO_SIMULATION_SCENE,
) -> str:
    """Read and strictly validate the optional-scene process contract."""

    source = os.environ if environ is None else environ
    raw = source.get(SIMULATION_SCENE_ENV, default)
    if raw not in SIMULATION_SCENE_VALUES:
        choices = ", ".join(repr(item) for item in sorted(SIMULATION_SCENE_VALUES))
        raise ValueError(
            f"{SIMULATION_SCENE_ENV} must be one of {choices}; got {raw!r}"
        )
    return raw
