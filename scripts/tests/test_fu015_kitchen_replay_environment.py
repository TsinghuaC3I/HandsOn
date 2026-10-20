from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


FU015_TASK_NAME = "t16_open_laptop"
KITCHEN_ENV = "HANDSON_FU015_KITCHEN_PREVIEW"


def _run_restore(
    demo_dir: Path,
    *,
    task_name: str = FU015_TASK_NAME,
    inherited: str = "1",
) -> subprocess.CompletedProcess[str]:
    program = r'''
import json
import os
import sys
from pathlib import Path

from scripts.isaac_render_arm_replay import (
    _FU015_KITCHEN_PREVIEW_ENV,
    _restore_recorded_fu015_kitchen_preview,
)

value = _restore_recorded_fu015_kitchen_preview(Path(sys.argv[1]), sys.argv[2])
print(json.dumps({
    "value": value,
    "present": _FU015_KITCHEN_PREVIEW_ENV in os.environ,
    "environment": os.environ.get(_FU015_KITCHEN_PREVIEW_ENV),
}))
'''
    environment = os.environ.copy()
    environment[KITCHEN_ENV] = inherited
    return subprocess.run(
        [sys.executable, "-c", program, str(demo_dir), task_name],
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
        env=environment,
    )


@pytest.mark.parametrize("recorded,inherited", [("1", "0"), ("0", "1")])
def test_fu015_camera_replay_restores_the_recorded_kitchen_flag(
    tmp_path: Path,
    recorded: str,
    inherited: str,
) -> None:
    (tmp_path / "replay_manifest.json").write_text(
        json.dumps(
            {
                "format": "handson.replay_manifest",
                "environment": {"variables": {KITCHEN_ENV: recorded}},
            }
        ),
        encoding="utf-8",
    )

    result = _run_restore(tmp_path, inherited=inherited)

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload == {
        "value": recorded,
        "present": True,
        "environment": recorded,
    }


def test_legacy_fu015_manifest_defaults_the_kitchen_to_off(
    tmp_path: Path,
) -> None:
    (tmp_path / "replay_manifest.json").write_text(
        json.dumps({"format": "handson.replay_manifest", "recording": {}}),
        encoding="utf-8",
    )

    result = _run_restore(tmp_path, inherited="1")

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "value": "0",
        "present": True,
        "environment": "0",
    }


def test_non_fu015_replay_restores_the_recorded_kitchen_flag(
    tmp_path: Path,
) -> None:
    (tmp_path / "replay_manifest.json").write_text(
        json.dumps(
            {
                "format": "handson.replay_manifest",
                "environment": {"variables": {KITCHEN_ENV: "1"}},
            }
        ),
        encoding="utf-8",
    )
    result = _run_restore(
        tmp_path,
        task_name="a01_pick_place",
        inherited="0",
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "value": "1",
        "present": True,
        "environment": "1",
    }


@pytest.mark.parametrize(
    "manifest_text,error_fragment",
    [
        ("not json", "valid manifest"),
        ("{}", "non-empty mapping"),
        (
            json.dumps(
                {
                    "format": "handson.replay_manifest",
                    "environment": None,
                }
            ),
            "environment must be a mapping",
        ),
        (
            json.dumps(
                {
                    "format": "handson.replay_manifest",
                    "environment": {"variables": []},
                }
            ),
            "environment.variables must be a mapping",
        ),
        (
            json.dumps(
                {
                    "format": "handson.replay_manifest",
                    "environment": {
                        "variables": {KITCHEN_ENV: "true"},
                    },
                }
            ),
            "expected '0' or '1'",
        ),
        (
            json.dumps(
                {
                    "format": "handson.replay_manifest",
                    "environment": {"variables": {KITCHEN_ENV: True}},
                }
            ),
            "expected '0' or '1'",
        ),
    ],
)
def test_fu015_camera_replay_rejects_invalid_recorded_environment(
    tmp_path: Path,
    manifest_text: str,
    error_fragment: str,
) -> None:
    (tmp_path / "replay_manifest.json").write_text(
        manifest_text,
        encoding="utf-8",
    )

    result = _run_restore(tmp_path)

    assert result.returncode != 0
    assert error_fragment in result.stderr


def test_fu015_camera_replay_requires_a_manifest(tmp_path: Path) -> None:
    result = _run_restore(tmp_path)

    assert result.returncode != 0
    assert "valid manifest" in result.stderr
