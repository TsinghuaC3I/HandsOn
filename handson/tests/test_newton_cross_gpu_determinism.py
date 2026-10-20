from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import types

import pytest

from handson.sim.isaaclab import compat
from handson.sim.isaaclab import newton_determinism as determinism
from handson.sim.isaaclab.abc_collection_env import (
    _canonical_newton_broad_phase,
)


def test_cross_gpu_profile_rejects_a_prequalified_stack(monkeypatch) -> None:
    versions = {
        "newton": "1.2.0",
        "warp-lang": "1.12.0",
    }
    monkeypatch.setattr(
        determinism,
        "_package_version",
        lambda name: versions.get(name, "1.0.0"),
    )

    with pytest.raises(determinism.NewtonCrossGpuUnavailable, match="qualified"):
        determinism.configure_newton_determinism("cross_gpu_v1")


def test_cross_gpu_profile_enables_warp_gpu_to_gpu_without_overclaiming(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        determinism,
        "_package_version",
        lambda _name: "99.0.0",
    )
    monkeypatch.delitem(sys.modules, "newton", raising=False)
    gpu_to_gpu = object()
    fake_warp = types.SimpleNamespace(
        DeterministicMode=types.SimpleNamespace(
            GPU_TO_GPU=gpu_to_gpu,
            RUN_TO_RUN=object(),
            NOT_GUARANTEED=object(),
        ),
        config=types.SimpleNamespace(deterministic=None),
    )
    monkeypatch.setattr(
        determinism.importlib,
        "import_module",
        lambda name: fake_warp if name == "warp" else None,
    )

    contract = determinism.configure_newton_determinism("cross_gpu_v1")

    assert fake_warp.config.deterministic is gpu_to_gpu
    assert contract["cross_gpu_ready"] is False
    assert contract["warp_deterministic_mode"] == "gpu_to_gpu"
    assert contract["solver_deterministic_mode"] == "unconfigured"
    assert contract["bitwise_cross_gpu_claim"] is False


def test_gpu_to_gpu_profile_rejects_warp_without_the_mode(monkeypatch) -> None:
    monkeypatch.setattr(
        determinism,
        "_package_version",
        lambda _name: "99.0.0",
    )
    fake_warp = types.SimpleNamespace(
        config=types.SimpleNamespace(deterministic=None),
    )
    monkeypatch.setattr(
        determinism.importlib,
        "import_module",
        lambda name: fake_warp if name == "warp" else None,
    )

    with pytest.raises(
        determinism.NewtonCrossGpuUnavailable,
        match="GPU_TO_GPU",
    ):
        determinism.configure_newton_determinism("local_debug_v1")


def test_distinct_gpu_model_does_not_treat_a_second_same_model_card_as_cross_gpu() -> None:
    source = {
        "gpu_name": "NVIDIA RTX 4090",
        "gpu_arch": "sm_89",
        "gpu_uuid": "GPU-A",
    }
    same_model = {
        "gpu_name": "NVIDIA RTX 4090",
        "gpu_arch": "sm_89",
        "gpu_uuid": "GPU-B",
    }
    different_model = {
        "gpu_name": "NVIDIA RTX 5090",
        "gpu_arch": "sm_120",
        "gpu_uuid": "GPU-C",
    }

    assert not determinism.is_distinct_gpu_model(source, same_model)
    assert determinism.is_distinct_gpu_model(source, different_model)


def test_velocity_projection_runtime_requires_and_hashes_both_patched_modules(
    monkeypatch,
    tmp_path,
) -> None:
    marker = b"HANDSON_B03_NEWTON_QD_PROJECTION_V1"
    paths = {}
    for index, module_name in enumerate(
        determinism._NEWTON_QD_PROJECTION_MODULES
    ):
        path = tmp_path / f"projection_{index}.py"
        path.write_bytes(marker + b"\n" + module_name.encode("utf-8"))
        paths[module_name] = path
    monkeypatch.setattr(
        determinism,
        "_newton_module_source_path",
        lambda name: paths[name],
    )
    contract = determinism.qualify_newton_velocity_projection_runtime()

    assert contract["algorithm"] == (
        "candidate_qd_clamp_before_semi_implicit_qpos"
    )
    assert contract["integrator"] == "implicitfast"
    assert len(contract["combined_sha256"]) == 64
    assert set(contract["module_sha256"]) == set(paths)


def test_velocity_projection_runtime_fails_closed_without_patch_marker(
    monkeypatch,
    tmp_path,
) -> None:
    source = tmp_path / "unpatched.py"
    source.write_text("# pristine upstream source\n", encoding="utf-8")
    monkeypatch.setattr(
        determinism,
        "_newton_module_source_path",
        lambda _name: source,
    )
    with pytest.raises(
        determinism.NewtonCrossGpuUnavailable,
        match="qd projection patch",
    ):
        determinism.qualify_newton_velocity_projection_runtime()


def test_velocity_projection_binds_loaded_module_and_is_not_stale_cached(
    monkeypatch,
    tmp_path,
) -> None:
    marker = "# HANDSON_B03_NEWTON_QD_PROJECTION_V1\n"
    loaded_paths = {}
    decoy_root = tmp_path / "decoy"
    for index, module_name in enumerate(
        determinism._NEWTON_QD_PROJECTION_MODULES
    ):
        loaded_path = tmp_path / f"loaded_{index}.py"
        loaded_path.write_text(marker + f"loaded = {index}\n", encoding="utf-8")
        loaded_paths[module_name] = loaded_path
        module = types.ModuleType(module_name)
        module.__file__ = str(loaded_path)
        monkeypatch.setitem(sys.modules, module_name, module)

        decoy_path = decoy_root / Path(*module_name.split(".")).with_suffix(
            ".py"
        )
        decoy_path.parent.mkdir(parents=True, exist_ok=True)
        decoy_path.write_text(marker + "decoy = True\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(decoy_root))

    first = determinism.qualify_newton_velocity_projection_runtime()
    loaded_paths[determinism._NEWTON_QD_PROJECTION_MODULES[0]].write_text(
        marker + "loaded = 'changed'\n",
        encoding="utf-8",
    )
    second = determinism.qualify_newton_velocity_projection_runtime()

    assert first["combined_sha256"] != second["combined_sha256"]
    for module_name, path in loaded_paths.items():
        assert second["module_sha256"][module_name] == hashlib.sha256(
            path.read_bytes()
        ).hexdigest()


def test_software_identity_binds_the_repository_physics_fingerprint() -> None:
    source = {"b03_physics_source_sha256": "a" * 64}
    changed = {"b03_physics_source_sha256": "b" * 64}

    assert determinism.software_identity(source) != (
        determinism.software_identity(changed)
    )


def test_software_identity_binds_the_installed_qd_projection_sources() -> None:
    source = {"newton_qd_projection": {"combined_sha256": "a" * 64}}
    changed = {"newton_qd_projection": {"combined_sha256": "b" * 64}}

    assert determinism.software_identity(source) != (
        determinism.software_identity(changed)
    )


def test_software_identity_binds_generated_robot_asset_hashes() -> None:
    source = {
        "b03_generated_robot_assets": {
            "right": {"generated_mjcf_sha256": "a" * 64}
        }
    }
    changed = {
        "b03_generated_robot_assets": {
            "right": {"generated_mjcf_sha256": "b" * 64}
        }
    }

    assert determinism.software_identity(source) != (
        determinism.software_identity(changed)
    )


def test_software_identity_binds_both_gpu_to_gpu_modes() -> None:
    source = {
        "warp_deterministic_mode": "gpu_to_gpu",
        "solver_deterministic_mode": "gpu_to_gpu",
    }
    changed = {
        **source,
        "solver_deterministic_mode": "run_to_run",
    }

    assert determinism.software_identity(source) != (
        determinism.software_identity(changed)
    )


def test_cross_gpu_semantic_identity_ignores_host_and_card_metadata() -> None:
    source = {
        "python": "3.12.13",
        "platform": "Linux-local",
        "gpu_name": "NVIDIA GeForce RTX 4090",
        "gpu_arch": "sm_89",
        "gpu_uuid": "GPU-LOCAL",
        "gpu_pci_bus_id": "0000:01:00.0",
        "cuda_driver_version": [13, 0],
        "nvidia_driver_version": "580.159.04",
        "b03_physics_source_sha256": "a" * 64,
    }
    replay = {
        **source,
        "platform": "Linux-cloud",
        "gpu_name": "NVIDIA GeForce RTX 5090",
        "gpu_arch": "sm_120",
        "gpu_uuid": "GPU-CLOUD",
        "gpu_pci_bus_id": "0000:63:00.0",
    }

    assert determinism.software_identity(source) == (
        determinism.software_identity(replay)
    )
    differences = determinism.runtime_environment_differences(source, replay)
    assert differences["platform"] == {
        "source": "Linux-local",
        "replay": "Linux-cloud",
    }
    assert differences["gpu_uuid"]["source"] == "GPU-LOCAL"


def test_cross_gpu_semantic_identity_keeps_driver_and_toolkit_strict() -> None:
    source = {
        "cuda_driver_version": [13, 0],
        "cuda_toolkit_version": [12, 9],
        "nvidia_driver_version": "580.159.04",
    }

    for field, changed_value in (
        ("cuda_driver_version", [13, 1]),
        ("cuda_toolkit_version", [13, 0]),
        ("nvidia_driver_version", "581.00.00"),
    ):
        changed = {**source, field: changed_value}
        assert determinism.software_identity(source) != (
            determinism.software_identity(changed)
        )


def test_same_machine_environment_projection_keeps_card_and_driver_strict() -> None:
    source = {
        "platform": "Linux-local",
        "device": "cuda:0",
        "gpu_name": "NVIDIA GeForce RTX 4090",
        "gpu_arch": "sm_89",
        "gpu_uuid": "GPU-A",
        "gpu_pci_bus_id": "0000:01:00.0",
        "cuda_driver_version": [13, 0],
        "nvidia_driver_version": "580.159.04",
    }
    changed = {**source, "gpu_uuid": "GPU-B"}

    assert determinism.runtime_environment_identity(source) != (
        determinism.runtime_environment_identity(changed)
    )


def test_newton_broad_phase_identity_is_stable_across_processes() -> None:
    BroadPhaseExplicit = type("BroadPhaseExplicit", (), {})

    assert _canonical_newton_broad_phase(BroadPhaseExplicit()) == "explicit"
    assert _canonical_newton_broad_phase(" Explicit ") == "explicit"


def test_b03_simulation_cfg_forwards_deterministic_collision_order(monkeypatch) -> None:
    class CollisionCfg:
        def __init__(self, *, broad_phase: str) -> None:
            self.broad_phase = broad_phase

        def to_pipeline_args(self):
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
    monkeypatch.setattr(compat.importlib.metadata, "version", lambda _name: "3.8.0")
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
