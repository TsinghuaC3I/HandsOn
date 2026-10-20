"""Instrument the normal P0-01 runner with per-frame PhysX separation reads."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

import live_smoke as runner


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


class _InstrumentedEnvironment:
    def __init__(self, environment: Any, output: Path) -> None:
        self._environment = environment
        self._output = output
        self._control_steps = 0
        self._active_contact_samples = 0
        self._sample_errors: list[str] = []
        self._by_view: dict[str, dict[str, float | int]] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._environment, name)

    @property
    def unwrapped(self) -> Any:
        return self._environment.unwrapped

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        return self._environment.reset(*args, **kwargs)

    def step(self, action: Any) -> Any:
        value = self._environment.step(action)
        self._control_steps += 1
        self._sample_separations()
        return value

    def _sample_separations(self) -> None:
        raw = self._environment.unwrapped
        for key, view in sorted(raw._p001_contact_views.items()):
            # This unfiltered view is intentionally created only for aggregate
            # net-force diagnostics; PhysX does not expose contact-data
            # separations for it.
            if key == "right_end_net":
                continue
            try:
                _forces, _points, _normals, separations, counts, starts = (
                    view.get_contact_data(raw._sim_dt)
                )
                separation_values = _numpy(separations).reshape(-1)
                contact_counts = _numpy(counts).astype(np.int64).reshape(-1)
                contact_starts = _numpy(starts).astype(np.int64).reshape(-1)
                active: list[np.ndarray] = []
                for count, start in zip(contact_counts, contact_starts):
                    if int(count) <= 0:
                        continue
                    values = separation_values[
                        int(start) : int(start) + int(count)
                    ]
                    values = values[np.isfinite(values)]
                    if values.size:
                        active.append(values)
                if not active:
                    continue
                values = np.concatenate(active)
                minimum = float(np.min(values))
                maximum_penetration = max(0.0, -minimum)
                record = self._by_view.setdefault(
                    key,
                    {
                        "active_contact_samples": 0,
                        "minimum_separation_m": math.inf,
                        "maximum_penetration_m": 0.0,
                    },
                )
                record["active_contact_samples"] = int(
                    record["active_contact_samples"]
                ) + int(values.size)
                record["minimum_separation_m"] = min(
                    float(record["minimum_separation_m"]), minimum
                )
                record["maximum_penetration_m"] = max(
                    float(record["maximum_penetration_m"]),
                    maximum_penetration,
                )
                self._active_contact_samples += int(values.size)
            except Exception as exc:
                message = f"{key}: {type(exc).__name__}: {exc}"
                if message not in self._sample_errors:
                    self._sample_errors.append(message)

    def close(self) -> Any:
        self._write_certificate()
        return self._environment.close()

    def _write_certificate(self) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        implementation_files = {
            "contract": repo_root
            / "handson/handson/sim/isaaclab/p001_contract.py",
            "environment": repo_root
            / "handson/handson/sim/isaaclab/p001_collection_env.py",
            "runner": Path(__file__).with_name("live_smoke.py"),
            "instrumentation": Path(__file__),
        }
        maximum_penetration = max(
            (
                float(record["maximum_penetration_m"])
                for record in self._by_view.values()
            ),
            default=math.nan,
        )
        result = {
            "schema": "p001_physx_penetration_certificate_v1",
            "control_steps_sampled": self._control_steps,
            "active_contact_samples": self._active_contact_samples,
            "minimum_separation_m": min(
                (
                    float(record["minimum_separation_m"])
                    for record in self._by_view.values()
                ),
                default=math.nan,
            ),
            "maximum_penetration_m": maximum_penetration,
            "threshold_m": 0.005,
            "passed": bool(
                self._control_steps > 0
                and self._active_contact_samples > 0
                and not self._sample_errors
                and math.isfinite(maximum_penetration)
                and maximum_penetration <= 0.005
            ),
            "by_contact_view": self._by_view,
            "sample_errors": self._sample_errors,
            "implementation_sha256": {
                name: _sha256(path) for name, path in implementation_files.items()
            },
        }
        self._output.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._output.with_suffix(self._output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self._output)


_original_get_environment = runner.TaskConfig.get_environment


def _instrumented_get_environment(self: Any, *args: Any, **kwargs: Any) -> Any:
    output_value = os.environ.get("P001_PENETRATION_OUTPUT", "").strip()
    if not output_value:
        raise RuntimeError("P001_PENETRATION_OUTPUT is required")
    environment = _original_get_environment(self, *args, **kwargs)
    return _InstrumentedEnvironment(environment, Path(output_value))


runner.TaskConfig.get_environment = _instrumented_get_environment


if __name__ == "__main__":
    runner.main()
