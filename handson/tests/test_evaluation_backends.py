"""Backend selection cannot silently use PhysX resets, state or task routes."""
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from handson.evaluation import backends, cameras, cli, native


@pytest.mark.parametrize("task", sorted(backends.NEWTON_TASKS))
def test_newton_factories_keep_paper_hand_and_physical_30hz(task):
    row = native.check_task(task, physics_backend="newton")
    assert row["physics_backend"] == "newton"
    assert row["hand_type"] == "wuji1"
    assert row["physics_dt"] * row["n_substeps"] == pytest.approx(1 / 30)


def test_unsupported_newton_task_fails_before_any_worker(monkeypatch, capsys):
    monkeypatch.setattr(cli.subprocess, "Popen", lambda *a, **k: pytest.fail("worker started"))
    monkeypatch.setattr(sys, "argv", ["eval", "--physics-backend", "newton", "--task", "t03_cover_box"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert "not supported by the newton evaluator" in capsys.readouterr().err


def test_bounded_policy_is_not_the_hold_diagnostic(monkeypatch, tmp_path):
    received = {}
    def run(task, output, **kwargs):
        received.update(kwargs)
        return {}
    monkeypatch.setattr(native, "run_episode", run)
    monkeypatch.setattr(sys, "argv", ["eval", "--worker", "--task", "t09_insert_peg",
        "--physics-backend", "newton", "--max-policy-steps", "90", "--output", str(tmp_path)])
    cli.main()
    assert received["policy"]["transport"] == "openpi_websocket"
    assert received["diagnostic_steps"] == 0
    assert received["max_policy_steps"] == 90
    assert received["physics_backend"] == "newton"


def test_newton_reset_reads_native_measured_state(monkeypatch, tmp_path):
    from handson.evaluation import eef52
    monkeypatch.setattr(eef52, "live_state", lambda env: (np.zeros(52), {"measured": True}))
    receipt = backends.record_newton_reset(SimpleNamespace(env_step=0), tmp_path, {"reset": 1})
    assert receipt["paper_physx_origin_applied"] is False
    assert json.loads((tmp_path / "NEWTON_RESET.json").read_text())["state"] == [0] * 52


def test_newton_capture_uses_live_manager():
    value = SimpleNamespace(numpy=lambda: np.array([1., 2.]))
    manager = SimpleNamespace(get_state_0=lambda: SimpleNamespace(body_q=value, joint_q=value),
                              get_model=lambda: SimpleNamespace(body_label=["body"]))
    env = SimpleNamespace(
        env_step=3, _sim=SimpleNamespace(current_time_step_index=24, physics_manager=manager), _control_dt=1/30,
        _physics_dt=1/240, _n_substeps=8, _newton_solver_settings={"solver": "mujoco"})
    row = backends.capture_newton(env)
    assert row["native_state"] == {"body_q": [1., 2.], "joint_q": [1., 2.]}
    assert row["physics_tick"] == 24
    assert row["recorded_state_used"] is False


@pytest.mark.parametrize("counter", [
    SimpleNamespace(current_time_step_index=72),
    SimpleNamespace(get_physics_step_count=lambda: 72),
])
def test_step_accounting_uses_real_counter_on_both_runtimes(counter):
    from handson.evaluation.readback import physics_tick
    assert physics_tick(SimpleNamespace(_sim=counter)) == 72


@pytest.mark.parametrize("lab3", [False, True])
def test_camera_buffers_support_both_lab_apis(lab3):
    annotator = SimpleNamespace(get_data=lambda: np.zeros((2, 2, 3), dtype=np.uint8))
    if lab3:
        camera = SimpleNamespace(_render_data=SimpleNamespace(
            render_product_paths=["/camera"], annotators={"rgb": annotator}))
    else:
        camera = SimpleNamespace(render_product_paths=["/camera"], _rep_registry={"rgb": [annotator]})
    assert cameras.render_product_paths(camera) == ["/camera"]
    assert cameras.has_frame(camera)
