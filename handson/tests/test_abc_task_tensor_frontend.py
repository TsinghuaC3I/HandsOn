from __future__ import annotations

import sys
import types

from handson.sim.isaaclab.abc_collection_env import (
    _A03_ANALYTIC_BALL_MASS_RADIUS,
    _A03_RESET_STABILITY_BODIES,
    _A03_SUPPORT_LOCAL_MIN_Z,
    _create_torch_task_physics_sim_view,
    _is_independent_task_rigid_body_path,
    _physx_gpu_buffer_kwargs,
)


def test_task_state_view_is_torch_and_bound_to_current_stage(monkeypatch) -> None:
    events: list[tuple[object, ...]] = []

    class FakeView:
        def set_subspace_roots(self, path: str) -> None:
            events.append(("roots", path))

    fake_view = FakeView()
    tensors = types.ModuleType("omni.physics.tensors")

    def create_simulation_view(
        frontend: str,
        *,
        stage_id: int,
    ) -> FakeView:
        events.append(("create", frontend, stage_id))
        return fake_view

    tensors.create_simulation_view = create_simulation_view
    usd = types.ModuleType("omni.usd")
    usd.get_context = lambda: types.SimpleNamespace(get_stage_id=lambda: 73)
    physics = types.ModuleType("omni.physics")
    physics.__path__ = []
    physics.tensors = tensors
    omni = types.ModuleType("omni")
    omni.__path__ = []
    omni.physics = physics
    omni.usd = usd
    monkeypatch.setitem(sys.modules, "omni", omni)
    monkeypatch.setitem(sys.modules, "omni.physics", physics)
    monkeypatch.setitem(sys.modules, "omni.physics.tensors", tensors)
    monkeypatch.setitem(sys.modules, "omni.usd", usd)

    result = _create_torch_task_physics_sim_view()

    assert result is fake_view
    assert events == [
        ("create", "torch", 73),
        ("roots", "/"),
    ]


def test_articulation_links_are_not_cached_as_independent_rigid_bodies() -> None:
    p002_root = "/World/Task/P002Door"
    roots = {p002_root}

    assert not _is_independent_task_rigid_body_path(p002_root, roots)
    assert not _is_independent_task_rigid_body_path(
        f"{p002_root}/Door042_Door",
        roots,
    )
    assert not _is_independent_task_rigid_body_path(
        f"{p002_root}/Door042_Handle002/Handle002_joint",
        roots,
    )
    assert _is_independent_task_rigid_body_path(
        "/World/Task/P002Doorish/free_body",
        roots,
    )
    assert _is_independent_task_rigid_body_path("/World/Task/table/table", roots)


def test_root_link_sibling_layout_remains_articulation_only() -> None:
    roots = {"/World/Task/Fixture/root_link"}

    assert not _is_independent_task_rigid_body_path(
        "/World/Task/Fixture/moving_link",
        roots,
    )
    assert _is_independent_task_rigid_body_path(
        "/World/Task/independent_body",
        roots,
    )
    assert _is_independent_task_rigid_body_path(
        "/World/Task/Fixture/free_body",
        set(),
    )


def test_a03_uses_bounded_physx_buffers_without_changing_partition_count() -> None:
    profile = _physx_gpu_buffer_kwargs("t24_place_ball_in_bowl")

    assert profile["gpu_found_lost_pairs_capacity"] == 2**16
    assert profile["gpu_found_lost_aggregate_pairs_capacity"] == 2**16
    assert profile["gpu_total_aggregate_pairs_capacity"] == 2**16
    assert profile["gpu_collision_stack_size"] == 2**22
    assert profile["gpu_heap_capacity"] == 2**22
    assert profile["gpu_temp_buffer_capacity"] == 2**20
    assert profile["gpu_max_num_partitions"] == 8


def test_a03_reset_stability_tunes_all_three_rolling_balls() -> None:
    assert set(_A03_RESET_STABILITY_BODIES) == {
        "tennis_ball",
        "golf_ball",
        "mini_soccer_ball",
    }


def test_a03_analytic_ball_mass_and_inertia_match_source_spheres() -> None:
    expected = {
        "tennis_ball": (0.058, 0.034, 2.68192e-5),
        "golf_ball": (0.046, 0.022, 8.9056e-6),
        "mini_soccer_ball": (0.070, 0.038, 4.0432e-5),
    }

    assert set(_A03_ANALYTIC_BALL_MASS_RADIUS) == set(expected)
    for name, (mass, radius, inertia) in expected.items():
        configured_mass, configured_radius = _A03_ANALYTIC_BALL_MASS_RADIUS[name]
        assert configured_mass == mass
        assert configured_radius == radius
        calculated_inertia = 0.4 * configured_mass * configured_radius**2
        assert abs(calculated_inertia - inertia) < 1.0e-15


def test_a03_reset_support_uses_prepared_analytic_sphere_radii() -> None:
    assert _A03_SUPPORT_LOCAL_MIN_Z["tennis_ball"] == -0.034
    assert _A03_SUPPORT_LOCAL_MIN_Z["golf_ball"] == -0.022
    assert _A03_SUPPORT_LOCAL_MIN_Z["mini_soccer_ball"] == -0.038
