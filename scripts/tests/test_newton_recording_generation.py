from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from scripts import record_demos_zarr as recorder


def test_newton_recordings_use_a_separate_non_physx_generation(
    tmp_path: Path,
) -> None:
    environment = SimpleNamespace(
        unwrapped=SimpleNamespace(_physics_backend="newton")
    )
    generation = recorder._dataset_generation_for_env(environment)

    assert generation.name == "nvidia_newton_v1"
    assert generation.directory_suffix == "__newton_v1"
    assert generation.is_newton_v1 is True
    assert generation.is_physx_v3 is False
    assert generation.training_status == "newton_backend_requires_separate_validation"

    marker = recorder._write_dataset_generation_marker(
        tmp_path,
        generation,
        episode_id="t26_place_fruit_in_bucket_demo_1_timestamp__newton_v1",
    )
    assert marker == tmp_path / "DATASET_NEWTON_V1.json"
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["is_newton_v1_data"] is True
    assert payload["is_physx_v3_data"] is False
