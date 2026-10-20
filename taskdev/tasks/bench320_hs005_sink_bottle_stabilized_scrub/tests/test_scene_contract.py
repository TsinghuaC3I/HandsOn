from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from taskdev.tasks.bench320_hs005_sink_bottle_stabilized_scrub.scene_contract import (
    BOTTLE_SOURCE_BOUNDS,
    BOTTLE_VISUAL_SCALE,
    BRUSH_SOURCE_BOUNDS,
    BRUSH_VISUAL_SCALE,
    CONTEXT_PROP_DISH_SOAP_Y,
    CONTEXT_PROP_REAR_X,
    CONTEXT_PROP_SPONGE_Y,
    DISH_SOAP_SOURCE_BOUNDS,
    DISH_SOAP_VISUAL_SCALE,
    SIM_SECONDS,
    SINK_SOURCE_BOUNDS,
    SPONGE_SOURCE_BOUNDS,
    SPONGE_VISUAL_SCALE,
    WARMUP_STEPS,
    ZERO_ACTION_STEPS,
    Aabb,
    initial_layout_pass,
    modern_apartment_bounds,
    placement_contract,
    release_scope,
    rotate_x_90_bounds,
    zero_action_stability_pass,
)


def test_exact_source_scales_and_axis_corrections() -> None:
    bottle = rotate_x_90_bounds(BOTTLE_SOURCE_BOUNDS).scaled(BOTTLE_VISUAL_SCALE)
    sink = modern_apartment_bounds(SINK_SOURCE_BOUNDS, (0.0, 0.0, 0.0))
    assert bottle.extent == pytest.approx(
        (0.07011311317, 0.070113129805, 0.249785753245)
    )
    assert sink.extent == pytest.approx((0.47350564, 0.95081205, 0.11867905))
    brush_extent = Aabb(*BRUSH_SOURCE_BOUNDS).scaled(BRUSH_VISUAL_SCALE).extent
    assert brush_extent == pytest.approx(
        (0.2843313028, 0.0280424783, 0.0280424692)
    )
    soap = rotate_x_90_bounds(DISH_SOAP_SOURCE_BOUNDS).scaled(
        DISH_SOAP_VISUAL_SCALE
    )
    sponge = rotate_x_90_bounds(SPONGE_SOURCE_BOUNDS).scaled(
        SPONGE_VISUAL_SCALE
    )
    assert soap.extent == pytest.approx(
        (0.085020272204, 0.120937472873, 0.2320781491735)
    )
    assert sponge.extent == pytest.approx(
        (0.0441959088744, 0.0998134607664, 0.0176699511)
    )


def test_reset_layout_has_support_gaps_and_no_forbidden_overlap() -> None:
    contract = placement_contract(0.90)
    assert initial_layout_pass(contract)
    assert contract["sink_origin"][:2] == pytest.approx((-0.15, 0.0))
    assert contract["bottle_origin"][:2] == pytest.approx((-0.10, 0.11))
    assert contract["brush_origin"][:2] == pytest.approx((-0.191, -0.125))
    assert contract["dish_soap_origin"][:2] == pytest.approx(
        (CONTEXT_PROP_REAR_X, CONTEXT_PROP_DISH_SOAP_Y)
    )
    assert contract["sponge_origin"][:2] == pytest.approx(
        (CONTEXT_PROP_REAR_X, CONTEXT_PROP_SPONGE_Y)
    )
    assert contract["basin_safe"].extent[:2] == pytest.approx((0.38, 0.40))
    assert contract["support_gaps_m"] == pytest.approx(
        {
            "sink_to_table": 0.005,
            "bottle_to_basin": 0.0,
            "brush_to_basin": 0.0000000082,
            "dish_soap_to_counter": 0.003,
            "sponge_to_counter": 0.0030002504,
        }
    )
    assert min(contract["inside_basin_margins_m"].values()) >= 0.04
    assert set(contract["forbidden_overlap_m"].values()) == {0.0}
    assert contract["sink_visual"].extent == pytest.approx(
        (0.47350564, 0.95081205, 0.11867905)
    )


def test_static_context_props_clear_both_raised_reset_hand_envelopes() -> None:
    contract = placement_contract(0.89)
    # Measured physical TCP centres from the live raised HS005 reset. A
    # conservative 12 cm planar clearance from each context-prop AABB covers
    # the palm and complete open-finger envelope; live scene-query evidence is
    # still required because this geometric check does not replace PhysX.
    hand_xy = ((-0.3714979, -0.2395002), (-0.3714979, 0.2395002))
    for prop_name in ("dish_soap_visual", "sponge_visual"):
        bounds = contract[prop_name]
        for hand_x, hand_y in hand_xy:
            dx = max(bounds.low[0] - hand_x, 0.0, hand_x - bounds.high[0])
            dy = max(bounds.low[1] - hand_y, 0.0, hand_y - bounds.high[1])
            assert math.hypot(dx, dy) >= 0.12


def test_layout_translates_with_live_table_height() -> None:
    low = placement_contract(0.80)
    high = placement_contract(1.05)
    for name in (
        "sink_origin",
        "bottle_origin",
        "brush_origin",
        "dish_soap_origin",
        "sponge_origin",
    ):
        assert high[name][0:2] == low[name][0:2]
        assert high[name][2] - low[name][2] == pytest.approx(0.25)


def test_full_seeded_xy_yaw_envelope_stays_inside_real_basin() -> None:
    contract = placement_contract(0.89)
    basin = contract["basin_safe"]
    bottle = contract["bottle_origin"]
    brush = contract["brush_origin"]

    for bottle_dx in (-0.02, 0.02):
        for bottle_dy in (-0.02, 0.02):
            bottle_xy = (bottle[0] + bottle_dx, bottle[1] + bottle_dy)
            assert bottle_xy[0] - 0.0347 >= basin.low[0]
            assert bottle_xy[0] + 0.0347 <= basin.high[0]
            assert bottle_xy[1] - 0.0347 >= basin.low[1]
            assert bottle_xy[1] + 0.0347 <= basin.high[1]

            for brush_dx in (-0.015, 0.015):
                for brush_dy in (-0.015, 0.015):
                    brush_xy = (brush[0] + brush_dx, brush[1] + brush_dy)
                    assert math.dist(bottle_xy, brush_xy) >= 0.20
                    for yaw_degrees in (-10.0, 10.0):
                        yaw = math.radians(yaw_degrees)
                        cosine, sine = math.cos(yaw), math.sin(yaw)
                        corners = [
                            (
                                brush_xy[0] + x * cosine - y * sine,
                                brush_xy[1] + x * sine + y * cosine,
                            )
                            for x in (
                                BRUSH_SOURCE_BOUNDS[0][0],
                                BRUSH_SOURCE_BOUNDS[1][0],
                            )
                            for y in (
                                BRUSH_SOURCE_BOUNDS[0][1],
                                BRUSH_SOURCE_BOUNDS[1][1],
                            )
                        ]
                        assert min(point[0] for point in corners) >= basin.low[0]
                        assert max(point[0] for point in corners) <= basin.high[0]
                        assert min(point[1] for point in corners) >= basin.low[1]
                        assert max(point[1] for point in corners) <= basin.high[1]


def test_layout_rejects_nonfinite_table_height() -> None:
    with pytest.raises(ValueError):
        placement_contract(math.nan)


def test_zero_action_gate_is_strict_after_bounded_warmup() -> None:
    valid = dict(
        bottle_displacement_m=0.003,
        brush_displacement_m=0.003,
        bottle_linear_speed_m_s=0.001,
        brush_linear_speed_m_s=0.001,
        bottle_angular_speed_rad_s=0.01,
        brush_angular_speed_rad_s=0.01,
        simulated_seconds=SIM_SECONDS,
    )
    assert WARMUP_STEPS == 120
    assert ZERO_ACTION_STEPS == 120
    assert zero_action_stability_pass(**valid)
    assert not zero_action_stability_pass(**{**valid, "simulated_seconds": 0.49})
    assert not zero_action_stability_pass(**{**valid, "bottle_displacement_m": 0.0121})
    assert not zero_action_stability_pass(**{**valid, "brush_linear_speed_m_s": 0.021})


def test_release_scope_cannot_claim_motion_or_task_completion() -> None:
    scope = release_scope()
    assert scope == {
        "core_scene_ready": False,
        "scrub_path_validated": False,
        "grasp_validated": False,
        "action_tuning_performed": False,
        "task_complete": False,
    }


def test_scene_manifest_is_hash_bound_and_scope_guarded() -> None:
    task_dir = Path(__file__).resolve().parents[1]
    manifest = json.loads((task_dir / "scene_manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_authority"]["sha256"] == (
        "c1bf0990e8ec0ae651ae6d904c6b61e45796c8e11a670f6da782f4b23052e987"
    )
    assert manifest["foundation_evidence"]["status"] == (
        "historical_pending_collision_revalidation"
    )
    assert manifest["foundation_evidence"]["valid_for_current_scene"] is False
    assert manifest["foundation_evidence"]["prelaunch_worktree_clean"] is False
    assert manifest["claims"]["core_scene_ready"] is True
    assert manifest["claims"]["grasp_validated"] is False
    assert manifest["claims"]["scrub_path_validated"] is False
    assert manifest["claims"]["task_complete"] is False


def test_portable_layer_declares_all_collision_and_body_owners() -> None:
    task_dir = Path(__file__).resolve().parents[1]
    scene = (task_dir / "scene/hs005_assets.usda").read_text(encoding="utf-8")
    assert scene.count('"PhysicsCollisionAPI"') == 20
    assert scene.count('"PhysicsRigidBodyAPI"') == 2
    assert "../assets/modern_apartment_kitchen/modern_apartment_kitchen.usd" in scene
    assert "../assets/cc_by_lukemodels75_water_bottle/water_bottle_visual.usd" in scene
    assert "../assets/cc_by_amhyde_cleaning_brush/cleaning_brush_visual.usd" in scene
    assert (
        "../assets/cc_by_mike_nov_dishwashing_dispenser/"
        "dishwashing_dispenser_visual.usd"
    ) in scene
    assert "../assets/cc_by_requital_dish_sponge/dish_sponge_visual.usd" in scene
    sink_layer = scene.split('def Xform "Sink"', 1)[1].split(
        'def Xform "Bottle"', 1
    )[0]
    assert 'def Cube "Collider"' in sink_layer
    assert 'float3 xformOp:scale = (0.42, 0.72, 0.02)' in sink_layer
    assert 'double3 xformOp:translate = (0.0, 0.0, -0.01)' in sink_layer
    assert '"PhysicsMeshCollisionAPI"' not in sink_layer
    scene_collision = scene.split('def Xform "SceneCollision"', 1)[1].split(
        'def Xform "Bottle"', 1
    )[0]
    assert (
        'physics_policy = "full_height_basin_and_open_countertop_ring_v1"'
        in scene_collision
    )
    for name in (
        "front_wall",
        "back_wall",
        "left_wall",
        "right_wall",
        "front_countertop",
        "back_countertop",
        "left_countertop",
        "right_countertop",
    ):
        assert f'def Cube "{name}"' in scene_collision
    assert "point3f physics:centerOfMass = (0.05, 0.0, 0.0)" in scene
    assert "float3 physics:diagonalInertia = (0.000012, 0.00074, 0.00074)" in scene
    assert 'def Material "LightCountertop"' in scene
    assert 'def Material "StainlessSink"' in scene
    assert scene.count(
        'rel material:binding = </HS005Assets/Looks/StainlessSink>'
    ) == 2
    bottle_layer = scene.split('def Xform "Bottle"', 1)[1].split(
        'def Xform "Brush"', 1
    )[0]
    assert "float physxRigidBody:linearDamping = 0.40" in bottle_layer
    assert "float physxRigidBody:angularDamping = 2.00" in bottle_layer
    assert 'def Cube "BrushRest"' not in scene
    assert 'def Cube "stability_foot"' in scene
    assert scene.count('token visibility = "invisible"') == 20
    assert "task_complete = false" in scene


def test_downloaded_visuals_are_hash_bound_render_only_conversions() -> None:
    task_dir = Path(__file__).resolve().parents[1]
    expected = {
        "cc_by_lukemodels75_water_bottle": ("water_bottle", 3),
        "cc_by_amhyde_cleaning_brush": ("cleaning_brush", 5),
        "cc_by_mike_nov_dishwashing_dispenser": ("dishwashing_dispenser", 11),
        "cc_by_requital_dish_sponge": ("dish_sponge", 1),
    }
    for directory, (asset_id, mesh_count) in expected.items():
        asset_dir = task_dir / "assets" / directory
        metadata = json.loads((asset_dir / "conversion.json").read_text(encoding="utf-8"))
        output = asset_dir / metadata["output"]["filename"]
        assert metadata["asset_id"] == asset_id
        assert metadata["source"]["license"] == "CC BY 4.0"
        assert metadata["output"]["mesh_count"] == mesh_count
        assert metadata["output"]["forbidden_schema_count"] == 0
        assert metadata["output"]["sha256"] == hashlib.sha256(
            output.read_bytes()
        ).hexdigest()
