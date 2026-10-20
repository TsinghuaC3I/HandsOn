"""Release configuration and process failures must fail clearly without Kit."""
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from handson.evaluation import cli
from handson.tasks import CONFIG_MAPPING


def test_shipped_configs_only_reference_released_tasks():
    root = Path(__file__).resolve().parents[2]
    for directory, count in (("multi_task", 66), ("rand_obj", 8), ("rand_full", 8)):
        paths = sorted((root / "configs" / directory).glob("*.yaml"))
        assert len(paths) == count
        names = [yaml.safe_load(path.read_text())["env_name"] for path in paths]
        assert len(set(names)) == len(names)
        assert set(names) <= set(CONFIG_MAPPING)
        if directory == "multi_task":
            assert set(names) == set(CONFIG_MAPPING)


@pytest.mark.parametrize("contents", ["", "[]", "env_name: 3", "env_name: removed_task", "env_name: ["])
def test_bad_yaml_reports_usage_error_before_starting_worker(tmp_path, monkeypatch, capsys, contents):
    config = tmp_path / "task.yaml"
    config.write_text(contents)
    monkeypatch.setattr(sys, "argv", ["handson-eval", "--config", str(config), "--check"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert "error:" in capsys.readouterr().err


@pytest.mark.parametrize("returncode,exitcode", [(-6, 134), (-9, 137), (0, 1), (2, 2)])
def test_incomplete_worker_is_infrastructure_failure(tmp_path, monkeypatch, returncode, exitcode):
    from handson.robots import assets
    monkeypatch.setattr(assets, "require_benchmark_assets", lambda: None)
    output = tmp_path / "episode"
    child = SimpleNamespace(returncode=returncode, stdout=io.StringIO("worker log\n"), wait=lambda: returncode)
    monkeypatch.setattr(cli.subprocess, "Popen", lambda *args, **kwargs: child)
    monkeypatch.setattr(sys, "argv", ["handson-eval", "--task", "t03_cover_box",
                                    "--diagnostic-steps", "90", "--output", str(output)])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == exitcode
    receipt = json.loads((output / "ERROR.json").read_text())
    assert receipt["counts_as_model_failure"] is False
    assert receipt["child_exit_code"] == returncode
    assert not (output / "RESULT.json").exists()
    assert (output / "runtime.log").read_text() == "worker log\n"


def test_missing_robot_assets_fail_before_worker_or_output_creation(tmp_path, monkeypatch, capsys):
    from handson.robots import assets
    def unavailable():
        raise FileNotFoundError("Bundled ROKAE description is missing; reinstall")
    monkeypatch.setattr(assets, "require_benchmark_assets", unavailable)
    monkeypatch.setattr(cli.subprocess, "Popen", lambda *a, **k: pytest.fail("worker started"))
    output = tmp_path / "episode"
    monkeypatch.setattr(sys, "argv", ["handson-eval", "--task", "t03_cover_box",
                                    "--diagnostic-steps", "90", "--output", str(output)])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert "reinstall" in capsys.readouterr().err
    assert not output.exists()
