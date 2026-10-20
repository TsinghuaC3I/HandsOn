from __future__ import annotations

import sys
import types

import pytest

from handson.sim.isaaclab import compat
from handson.sim.isaaclab.abc_collection_env import HandsOnABCIsaacLabEnv


def test_newton_simulation_cfg_forwards_deterministic_collision_order(
    monkeypatch,
) -> None:
    class CollisionCfg:
        def __init__(self, *, broad_phase: str) -> None:
            self.broad_phase = broad_phase

        def to_pipeline_args(self) -> dict[str, object]:
            return dict(self.__dict__)

    class NewtonCfg:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    module = types.SimpleNamespace(
        NewtonCollisionPipelineCfg=CollisionCfg,
        MJWarpSolverCfg=lambda **kwargs: types.SimpleNamespace(**kwargs),
        NewtonCfg=NewtonCfg,
    )
    sim_utils = types.SimpleNamespace(
        SimulationCfg=lambda **kwargs: types.SimpleNamespace(**kwargs)
    )
    monkeypatch.setattr(compat, "_load_lab3_physx_cfg", lambda: object())
    monkeypatch.setattr(
        compat.importlib.metadata,
        "version",
        lambda _name: "3.8.0",
    )
    monkeypatch.setattr(
        compat.importlib,
        "import_module",
        lambda name: module if name == "isaaclab_newton.physics" else None,
    )

    config = compat.make_simulation_cfg(
        sim_utils,
        physics_backend="newton",
        newton_deterministic_contacts=True,
        device="cuda:0",
        dt=1.0 / 120.0,
    )

    collision = config.physics.kwargs["collision_cfg"]
    assert collision.broad_phase == "explicit"
    assert collision.deterministic is True


def test_a01_runtime_rejects_a_nondeterministic_newton_pipeline(
    monkeypatch,
) -> None:
    env = HandsOnABCIsaacLabEnv.__new__(HandsOnABCIsaacLabEnv)
    env.task_name = "t26_place_fruit_in_bucket"
    env._physics_backend = "newton"
    physics = types.ModuleType("isaaclab_newton.physics")
    gpu_to_gpu = object()
    physics.NewtonManager = types.SimpleNamespace(
        _collision_pipeline=types.SimpleNamespace(deterministic=False),
        _solver=types.SimpleNamespace(_deterministic=gpu_to_gpu),
    )
    package = types.ModuleType("isaaclab_newton")
    package.physics = physics
    monkeypatch.setitem(sys.modules, "isaaclab_newton", package)
    monkeypatch.setitem(sys.modules, "isaaclab_newton.physics", physics)
    warp = types.ModuleType("warp")
    warp.DeterministicMode = types.SimpleNamespace(GPU_TO_GPU=gpu_to_gpu)
    warp.config = types.SimpleNamespace(deterministic=gpu_to_gpu)
    monkeypatch.setitem(sys.modules, "warp", warp)

    with pytest.raises(RuntimeError, match="t26_place_fruit_in_bucket"):
        env._verify_newton_determinism()


def test_a01_runtime_records_the_live_deterministic_pipeline(monkeypatch) -> None:
    env = HandsOnABCIsaacLabEnv.__new__(HandsOnABCIsaacLabEnv)
    env.task_name = "t26_place_fruit_in_bucket"
    env._physics_backend = "newton"
    physics = types.ModuleType("isaaclab_newton.physics")
    gpu_to_gpu = object()
    physics.NewtonManager = types.SimpleNamespace(
        _collision_pipeline=types.SimpleNamespace(
            deterministic=True,
            broad_phase="explicit",
        ),
        _solver=types.SimpleNamespace(_deterministic=gpu_to_gpu),
    )
    package = types.ModuleType("isaaclab_newton")
    package.physics = physics
    monkeypatch.setitem(sys.modules, "isaaclab_newton", package)
    monkeypatch.setitem(sys.modules, "isaaclab_newton.physics", physics)
    warp = types.ModuleType("warp")
    warp.DeterministicMode = types.SimpleNamespace(GPU_TO_GPU=gpu_to_gpu)
    warp.config = types.SimpleNamespace(deterministic=gpu_to_gpu)
    monkeypatch.setitem(sys.modules, "warp", warp)

    env._verify_newton_determinism()

    assert env._newton_determinism_contract == {
        "collision_pipeline_deterministic": True,
        "collision_broad_phase": "explicit",
        "warp_deterministic_mode": "gpu_to_gpu",
        "solver_deterministic_mode": "gpu_to_gpu",
        "cross_gpu_ready": False,
    }
