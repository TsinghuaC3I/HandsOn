#!/usr/bin/env python3
"""Headless, deterministic GPU qualification for the E01 PhysX runtime.

This is deliberately separate from ``preview_isaaclab_task.py``.  The preview
only proves that a USD scene can advance; this harness exercises the actual
54D detached-Wuji environment boundary, native surface-deformable tensor
readback, reset settling, bounded tracker input, and contact motion.

The program refuses to launch Kit while another CUDA compute process is using
the selected GPU unless ``--allow-busy-gpu`` is passed explicitly.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
HANDSON_ROOT = REPO_ROOT / "handson"
if str(HANDSON_ROOT) not in sys.path:
    sys.path.insert(0, str(HANDSON_ROOT))

from handson.sim.isaaclab.e_collection_scene import (
    E01_CONTACT_OFFSET,
    E01_PHYSICS_HZ,
    selected_garmentlab_preset_metadata,
)

CONTROL_DT = 1.0 / 30.0
PHYSICS_DT = 1.0 / E01_PHYSICS_HZ
POKE_SPEEDS_M_S = (0.03, 0.05, 0.06)


def _selected_garment_mesh_count(key: str) -> int:
    return int(selected_garmentlab_preset_metadata()[key])


class StressFailure(RuntimeError):
    """One qualification invariant failed."""


@dataclass(frozen=True)
class Thresholds:
    min_simulation_node_count: int = field(
        default_factory=lambda: _selected_garment_mesh_count("sim_vertex_count")
    )
    max_simulation_node_count: int = field(
        default_factory=lambda: _selected_garment_mesh_count("sim_vertex_count")
    )
    min_simulation_triangle_count: int = field(
        default_factory=lambda: _selected_garment_mesh_count("sim_triangle_count")
    )
    max_simulation_triangle_count: int = field(
        default_factory=lambda: _selected_garment_mesh_count("sim_triangle_count")
    )
    expected_visual_vertex_count: int = field(
        default_factory=lambda: _selected_garment_mesh_count("visual_vertex_count")
    )
    expected_visual_triangle_count: int = field(
        default_factory=lambda: _selected_garment_mesh_count("visual_triangle_count")
    )
    max_nodal_speed_m_s: float = (
        E01_CONTACT_OFFSET * E01_PHYSICS_HZ + 0.06
    )
    max_control_node_step_m: float = 0.015
    max_idle_drift_m: float = 0.005
    max_reset_drift_m: float = 0.005
    min_contact_response_m: float = 0.0001
    min_pinch_vertical_response_m: float = 0.001
    min_dual_pull_xy_response_m: float = 0.001
    max_hand_linear_speed_m_s: float = 0.55
    max_hand_angular_speed_rad_s: float = 5.5
    max_public_readback_position_error_m: float = 1e-6
    max_public_readback_rotation_error_deg: float = 0.01
    max_reset_command_position_error_m: float = 0.005
    max_reset_command_rotation_error_deg: float = 2.0
    max_p95_step_ms: float = 25.0
    max_p99_step_ms: float = 33.34
    max_realtime_overrun_fraction: float = 0.01


@dataclass
class PhaseMetrics:
    name: str
    step_count: int = 0
    simulated_seconds: float = 0.0
    wall_seconds: float = 0.0
    max_nodal_speed_m_s: float = 0.0
    max_control_node_step_m: float = 0.0
    max_response_m: float = 0.0
    max_vertical_response_m: float = 0.0
    max_xy_response_m: float = 0.0
    max_hand_linear_speed_m_s: float = 0.0
    max_hand_angular_speed_rad_s: float = 0.0
    max_public_readback_position_error_m: float = 0.0
    max_public_readback_rotation_error_deg: float = 0.0
    max_tracking_translation_error_m: float = 0.0
    max_tracking_rotation_error_deg: float = 0.0
    max_tracking_saturation_substeps: int = 0
    force_saturation_steps: int = 0
    torque_saturation_steps: int = 0
    passed: bool = False
    failure: str | None = None


@dataclass
class StressReport:
    status: str
    device: str
    gpu: dict[str, Any]
    thresholds: dict[str, Any]
    simulation_node_count: int | None = None
    simulation_triangle_count: int | None = None
    visual_vertex_count: int | None = None
    visual_triangle_count: int | None = None
    phases: list[dict[str, Any]] = field(default_factory=list)
    reset_cycles: int = 0
    reset_max_node_drift_m: float = 0.0
    reset_min_settle_seconds_observed: float | None = None
    reset_max_settle_seconds_observed: float | None = None
    reset_min_stable_steps_observed: int | None = None
    step_count: int = 0
    simulated_seconds: float = 0.0
    wall_seconds: float = 0.0
    step_p50_ms: float = math.inf
    step_p95_ms: float = math.inf
    step_p99_ms: float = math.inf
    realtime_overrun_fraction: float = 1.0
    failure: str | None = None


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return math.inf
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _cuda_index(device: str) -> int:
    match = re.fullmatch(r"cuda:(\d+)", str(device).strip().lower())
    if match is None:
        raise ValueError(
            f"E01 GPU stress requires an explicit CUDA device such as cuda:0; got {device!r}"
        )
    return int(match.group(1))


def parse_compute_process_rows(output: str) -> list[dict[str, Any]]:
    """Parse ``nvidia-smi --query-compute-apps`` CSV without importing CUDA."""

    rows: list[dict[str, Any]] = []
    for raw_line in str(output).splitlines():
        line = raw_line.strip()
        if not line or line.lower().startswith("no running"):
            continue
        parts = [part.strip() for part in line.split(",", 2)]
        if len(parts) != 3:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        try:
            memory_mib: int | None = int(parts[2])
        except ValueError:
            memory_mib = None
        rows.append(
            {"pid": pid, "process_name": parts[1], "used_memory_mib": memory_mib}
        )
    return rows


def parse_pmon_process_rows(output: str) -> list[dict[str, Any]]:
    """Parse one ``nvidia-smi pmon`` sample, including graphics-only owners."""

    rows: list[dict[str, Any]] = []
    for raw_line in str(output).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            gpu_index = int(parts[0])
            pid = int(parts[1])
        except ValueError:
            continue
        rows.append(
            {
                "gpu_index": gpu_index,
                "pid": pid,
                "type": parts[2],
                "command": parts[-1],
            }
        )
    return rows


def gpu_preflight(device: str, *, allow_busy_gpu: bool) -> dict[str, Any]:
    """Refuse a racy Kit launch when collection already owns the GPU."""

    index = _cuda_index(device)
    try:
        identity = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(index),
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10.0,
        ).stdout.strip()
        process_output = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(index),
                "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10.0,
        ).stdout
        pmon_output = subprocess.run(
            ["nvidia-smi", "pmon", "-i", str(index), "-c", "1"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10.0,
        ).stdout
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        raise StressFailure(f"GPU preflight could not query nvidia-smi: {exc}") from exc

    identity_parts = [part.strip() for part in identity.split(",", 3)]
    processes = parse_compute_process_rows(process_output)
    compute_pids = {int(row["pid"]) for row in processes}
    # A MuJoCo/Genesis collection may own only an EGL graphics context and is
    # absent from --query-compute-apps.  Catch those Python/Kit workloads while
    # allowing the workstation's ordinary Xorg/browser graphics clients.
    graphics_workloads = [
        row
        for row in parse_pmon_process_rows(pmon_output)
        if int(row["pid"]) not in compute_pids
        and "G" in str(row["type"])
        and re.search(
            r"python|kit|isaac|mujoco|genesis|handson",
            str(row["command"]),
            flags=re.IGNORECASE,
        )
    ]
    gpu = {
        "index": index,
        "name": identity_parts[1] if len(identity_parts) > 1 else "unknown",
        "memory_total_mib": (
            int(identity_parts[2])
            if len(identity_parts) > 2 and identity_parts[2].isdigit()
            else None
        ),
        "driver_version": identity_parts[3] if len(identity_parts) > 3 else "unknown",
        "preexisting_compute_processes": processes,
        "preexisting_graphics_workloads": graphics_workloads,
    }
    if (processes or graphics_workloads) and not allow_busy_gpu:
        compute_rendered = ", ".join(
            f"pid={row['pid']} {row['process_name']} ({row['used_memory_mib']} MiB)"
            for row in processes
        )
        graphics_rendered = ", ".join(
            f"pid={row['pid']} {row['command']} ({row['type']})"
            for row in graphics_workloads
        )
        rendered = "; ".join(
            value
            for value in (
                f"compute: {compute_rendered}" if compute_rendered else "",
                f"graphics workload: {graphics_rendered}" if graphics_rendered else "",
            )
            if value
        )
        raise StressFailure(
            "Selected GPU already has active workloads; refusing to launch Kit: "
            f"{rendered}. Finish the active collection first."
        )
    return gpu


def make_action(
    right_pose: np.ndarray,
    left_pose: np.ndarray,
    right_hand: np.ndarray,
    left_hand: np.ndarray,
) -> np.ndarray:
    """Pack the public E01 54D layout: right/left pose, right/left 20D hand."""

    action = np.concatenate(
        [
            np.asarray(right_pose, dtype=np.float64).reshape(7),
            np.asarray(left_pose, dtype=np.float64).reshape(7),
            np.asarray(right_hand, dtype=np.float64).reshape(20),
            np.asarray(left_hand, dtype=np.float64).reshape(20),
        ]
    )
    if action.shape != (54,):
        raise AssertionError(f"Internal E01 stress action has shape {action.shape}")
    return action


def _normalized_lerp_quat(start: np.ndarray, target: np.ndarray, fraction: float) -> np.ndarray:
    q0 = np.asarray(start, dtype=np.float64).reshape(4)
    q1 = np.asarray(target, dtype=np.float64).reshape(4)
    q0 = q0 / max(float(np.linalg.norm(q0)), 1e-12)
    q1 = q1 / max(float(np.linalg.norm(q1)), 1e-12)
    if float(np.dot(q0, q1)) < 0.0:
        q1 = -q1
    result = (1.0 - fraction) * q0 + fraction * q1
    return result / max(float(np.linalg.norm(result)), 1e-12)


def quaternion_angle_wxyz(first: np.ndarray, second: np.ndarray) -> float:
    """Shortest angular distance between two public/Isaac WXYZ quaternions."""

    q0 = np.asarray(first, dtype=np.float64).reshape(4)
    q1 = np.asarray(second, dtype=np.float64).reshape(4)
    if not np.all(np.isfinite(q0)) or not np.all(np.isfinite(q1)):
        return math.inf
    norm0 = float(np.linalg.norm(q0))
    norm1 = float(np.linalg.norm(q1))
    if norm0 <= 1e-12 or norm1 <= 1e-12:
        return math.inf
    dot = float(np.clip(abs(np.dot(q0 / norm0, q1 / norm1)), 0.0, 1.0))
    return 2.0 * math.acos(dot)


def motion_step_count(
    start_poses: Mapping[str, np.ndarray],
    target_poses: Mapping[str, np.ndarray],
    start_hands: Mapping[str, np.ndarray],
    target_hands: Mapping[str, np.ndarray],
    *,
    linear_speed_m_s: float,
    hand_speed_rad_s: float = 1.2,
) -> int:
    if not math.isfinite(linear_speed_m_s) or linear_speed_m_s <= 0.0:
        raise ValueError("linear_speed_m_s must be finite and positive")
    if not math.isfinite(hand_speed_rad_s) or hand_speed_rad_s <= 0.0:
        raise ValueError("hand_speed_rad_s must be finite and positive")
    linear_distance = max(
        float(
            np.linalg.norm(
                np.asarray(target_poses[side])[:3]
                - np.asarray(start_poses[side])[:3]
            )
        )
        for side in ("right", "left")
    )
    hand_distance = max(
        float(
            np.max(
                np.abs(
                    np.asarray(target_hands[side])
                    - np.asarray(start_hands[side])
                )
            )
        )
        for side in ("right", "left")
    )
    # Distances come from repeated binary64 pose arithmetic.  Subtracting a
    # tiny dimensionless tolerance prevents an exact 18-step motion from
    # becoming 19 solely because its quotient is 18.000000000000014.
    linear_steps = int(
        math.ceil(linear_distance / (linear_speed_m_s * CONTROL_DT) - 1e-12)
    )
    hand_steps = int(
        math.ceil(hand_distance / (hand_speed_rad_s * CONTROL_DT) - 1e-12)
    )
    return max(
        1,
        linear_steps,
        hand_steps,
    )


class E01StressRunner:
    def __init__(self, env: Any, thresholds: Thresholds) -> None:
        self.env = env
        self.thresholds = thresholds
        self.step_durations: list[float] = []
        self.phases: list[PhaseMetrics] = []
        self.phase: PhaseMetrics | None = None
        self.phase_reference_nodes: np.ndarray | None = None
        self.previous_nodes: np.ndarray | None = None
        self.reset_reference_nodes: np.ndarray | None = None
        self.reset_max_node_drift_m = 0.0
        self.reset_settle_seconds_observed: list[float] = []
        self.reset_stable_steps_observed: list[int] = []
        self.simulation_node_count: int | None = None
        self.simulation_triangle_count: int | None = None
        self.visual_vertex_count: int | None = None
        self.visual_triangle_count: int | None = None

    def current_action(self) -> np.ndarray:
        return make_action(
            self.env._right_tcp_pose,
            self.env._left_tcp_pose,
            self.env._right_hand,
            self.env._left_hand,
        )

    def _surface_state(self) -> tuple[np.ndarray, np.ndarray]:
        nodes = np.asarray(self.env._e_nodes_np, dtype=np.float64)
        velocities = np.asarray(self.env._e_velocities_np, dtype=np.float64)
        expected = int(self.env._e_expected_node_count)
        if not (
            self.thresholds.min_simulation_node_count
            <= expected
            <= self.thresholds.max_simulation_node_count
        ):
            raise StressFailure(
                "E01 native simulation node count is outside the qualified "
                f"band: {expected}"
            )
        if nodes.shape != (expected, 3):
            raise StressFailure(
                f"wrong E01 garment topology: nodes={nodes.shape}, expected=({expected}, 3)"
            )
        if velocities.shape != nodes.shape:
            raise StressFailure(
                f"wrong E01 velocity topology: velocities={velocities.shape}, nodes={nodes.shape}"
            )
        if not np.all(np.isfinite(nodes)) or not np.all(np.isfinite(velocities)):
            raise StressFailure("E01 surface tensor contains NaN/Inf")
        return nodes.copy(), velocities.copy()

    def _validate_runtime_contract(self, info: Mapping[str, Any]) -> None:
        required = {
            "control_runtime": "isaac_physx_detached_canonical_wuji",
            "simulated_arm_count": 0,
            "simulated_hand_count": 2,
            "online_arm_ik": False,
            "hand_root_control_hz": float(E01_PHYSICS_HZ),
            "control_command_hz": 30.0,
            "hand_interpolation_substeps": int(
                round(CONTROL_DT / PHYSICS_DT)
            ),
            "physical_randomization": False,
        }
        for key, expected in required.items():
            if info.get(key) != expected:
                raise StressFailure(
                    f"runtime contract mismatch: {key}={info.get(key)!r}, expected={expected!r}"
                )
        node_count = int(info.get("garment_node_count", -1))
        triangle_count = int(
            info.get("garment_simulation_triangle_count", -1)
        )
        visual_vertices = int(info.get("garment_visual_vertex_count", -1))
        visual_triangles = int(
            info.get("garment_visual_triangle_count", -1)
        )
        target_triangles = int(
            info.get("garment_simulation_target_triangle_count", -1)
        )
        if not (
            self.thresholds.min_simulation_node_count
            <= node_count
            <= self.thresholds.max_simulation_node_count
        ):
            raise StressFailure(
                "runtime exposed an unqualified E01 native simulation mesh: "
                f"nodes={node_count}"
            )
        if not (
            self.thresholds.min_simulation_triangle_count
            <= triangle_count
            <= self.thresholds.max_simulation_triangle_count
        ):
            raise StressFailure(
                "runtime exposed an unqualified E01 native simulation mesh: "
                f"triangles={triangle_count}"
            )
        if target_triangles != self.thresholds.max_simulation_triangle_count:
            raise StressFailure(
                "runtime reports a different native PhysX topology: "
                f"{target_triangles}"
            )
        if (
            visual_vertices != self.thresholds.expected_visual_vertex_count
            or visual_triangles
            != self.thresholds.expected_visual_triangle_count
        ):
            raise StressFailure(
                "runtime no longer uses the selected GarmentLab visual asset: "
                f"visual={visual_vertices}v/{visual_triangles}t"
            )
        self.simulation_node_count = node_count
        self.simulation_triangle_count = triangle_count
        self.visual_vertex_count = visual_vertices
        self.visual_triangle_count = visual_triangles

    def _validate_public_tcp_observation(
        self,
        observation: Mapping[str, Any],
    ) -> tuple[float, float]:
        """Verify the public WXYZ pose is the actual PhysX link readback.

        This catches both a right/left layout regression and an accidental
        XYZW/WXYZ conversion.  A sign-flipped quaternion is accepted because
        it represents the same physical orientation.
        """

        try:
            public_tcp = np.asarray(
                observation["state"]["tcp_pose"], dtype=np.float64
            ).reshape(14)
        except (KeyError, TypeError, ValueError) as exc:
            raise StressFailure(
                "public E01 observation is missing state.tcp_pose[14]"
            ) from exc
        actual_tcp = np.concatenate(
            [
                np.asarray(self.env._detached_sim_pose["right"], dtype=np.float64),
                np.asarray(self.env._detached_sim_pose["left"], dtype=np.float64),
            ]
        )
        if not np.all(np.isfinite(public_tcp)) or not np.all(np.isfinite(actual_tcp)):
            raise StressFailure("public/PhysX TCP pose contains NaN/Inf")
        max_position_error = 0.0
        max_rotation_error = 0.0
        for offset, side in ((0, "right"), (7, "left")):
            position_error = float(
                np.linalg.norm(public_tcp[offset : offset + 3] - actual_tcp[offset : offset + 3])
            )
            rotation_error = quaternion_angle_wxyz(
                public_tcp[offset + 3 : offset + 7],
                actual_tcp[offset + 3 : offset + 7],
            )
            max_position_error = max(max_position_error, position_error)
            max_rotation_error = max(max_rotation_error, rotation_error)
            if position_error > self.thresholds.max_public_readback_position_error_m:
                raise StressFailure(
                    f"{side} public TCP position differs from PhysX readback by "
                    f"{position_error:.9f} m"
                )
            if math.degrees(rotation_error) > self.thresholds.max_public_readback_rotation_error_deg:
                raise StressFailure(
                    f"{side} public WXYZ orientation differs from PhysX readback by "
                    f"{math.degrees(rotation_error):.9f} deg"
                )
        return max_position_error, math.degrees(max_rotation_error)

    def _validate_state(
        self,
        info: Mapping[str, Any],
        *,
        previous_nodes: np.ndarray | None,
        observation: Mapping[str, Any],
    ) -> np.ndarray:
        if info.get("failure_reason") is not None:
            raise StressFailure(
                f"environment latched failure_reason={info.get('failure_reason')!r}; "
                f"diagnostics={info.get('garment_failure_diagnostics')!r}"
            )
        if info.get("garment_state_valid") is not True:
            raise StressFailure("environment reports garment_state_valid=False")

        public_position_error, public_rotation_error = (
            self._validate_public_tcp_observation(observation)
        )
        nodes, velocities = self._surface_state()
        speed = np.linalg.norm(velocities, axis=1)
        max_speed = float(np.max(speed))
        configured_limit = min(
            self.thresholds.max_nodal_speed_m_s,
            float(info.get("garment_nodal_speed_failure_limit", math.inf)),
        )
        if max_speed > configured_limit + 1e-6:
            raise StressFailure(
                f"nodal speed {max_speed:.6f} m/s exceeds {configured_limit:.6f} m/s"
            )

        control_step = 0.0
        if previous_nodes is not None:
            control_step = float(
                np.max(np.linalg.norm(nodes - previous_nodes, axis=1))
            )
            if control_step > self.thresholds.max_control_node_step_m + 1e-6:
                raise StressFailure(
                    f"control-frame node displacement {control_step:.6f} m exceeds "
                    f"{self.thresholds.max_control_node_step_m:.6f} m"
                )

        for side in ("right", "left"):
            pose = np.asarray(self.env._detached_sim_pose[side], dtype=np.float64)
            velocity = np.asarray(
                self.env._detached_sim_velocity[side], dtype=np.float64
            )
            hand = np.asarray(self.env._detached_sim_hand[side], dtype=np.float64)
            command_pose = np.asarray(
                self.env._plan1_command_filter.poses[side], dtype=np.float64
            )
            command_hand = np.asarray(
                self.env._plan1_command_filter.hands[side], dtype=np.float64
            )
            if not all(
                np.all(np.isfinite(value))
                for value in (pose, velocity, hand, command_pose, command_hand)
            ):
                raise StressFailure(f"{side} hand readback/command contains NaN/Inf")
            linear_speed = float(np.linalg.norm(velocity[:3]))
            angular_speed = float(np.linalg.norm(velocity[3:6]))
            if linear_speed > self.thresholds.max_hand_linear_speed_m_s + 1e-6:
                raise StressFailure(
                    f"{side} palm linear speed {linear_speed:.6f} m/s exceeds cap"
                )
            if angular_speed > self.thresholds.max_hand_angular_speed_rad_s + 1e-6:
                raise StressFailure(
                    f"{side} palm angular speed {angular_speed:.6f} rad/s exceeds cap"
                )

        if self.phase is not None:
            phase = self.phase
            phase.max_nodal_speed_m_s = max(phase.max_nodal_speed_m_s, max_speed)
            phase.max_control_node_step_m = max(
                phase.max_control_node_step_m, control_step
            )
            phase.max_public_readback_position_error_m = max(
                phase.max_public_readback_position_error_m,
                public_position_error,
            )
            phase.max_public_readback_rotation_error_deg = max(
                phase.max_public_readback_rotation_error_deg,
                public_rotation_error,
            )
            if self.phase_reference_nodes is not None:
                delta = nodes - self.phase_reference_nodes
                phase.max_response_m = max(
                    phase.max_response_m,
                    float(np.max(np.linalg.norm(delta, axis=1))),
                )
                phase.max_vertical_response_m = max(
                    phase.max_vertical_response_m,
                    float(np.max(delta[:, 2])),
                )
                phase.max_xy_response_m = max(
                    phase.max_xy_response_m,
                    float(np.max(np.linalg.norm(delta[:, :2], axis=1))),
                )
            for side in ("right", "left"):
                velocity = np.asarray(
                    self.env._detached_sim_velocity[side], dtype=np.float64
                )
                phase.max_hand_linear_speed_m_s = max(
                    phase.max_hand_linear_speed_m_s,
                    float(np.linalg.norm(velocity[:3])),
                )
                phase.max_hand_angular_speed_rad_s = max(
                    phase.max_hand_angular_speed_rad_s,
                    float(np.linalg.norm(velocity[3:6])),
                )
                phase.max_tracking_translation_error_m = max(
                    phase.max_tracking_translation_error_m,
                    float(
                        info.get(
                            f"palm_tracking_translation_error_m_{side}", 0.0
                        )
                    ),
                )
                phase.max_tracking_rotation_error_deg = max(
                    phase.max_tracking_rotation_error_deg,
                    float(
                        info.get(
                            f"palm_tracking_rotation_error_deg_{side}", 0.0
                        )
                    ),
                )
                phase.max_tracking_saturation_substeps = max(
                    phase.max_tracking_saturation_substeps,
                    int(
                        info.get(
                            f"palm_tracking_saturation_substeps_{side}", 0
                        )
                    ),
                )
            if any(
                bool(info.get(f"palm_force_saturated_{side}", False))
                for side in ("right", "left")
            ):
                phase.force_saturation_steps += 1
            if any(
                bool(info.get(f"palm_torque_saturated_{side}", False))
                for side in ("right", "left")
            ):
                phase.torque_saturation_steps += 1
        return nodes

    def reset(self, seed: int) -> tuple[Any, Mapping[str, Any]]:
        observation, info = self.env.reset(seed=seed)
        self._validate_runtime_contract(info)
        if info.get("garment_reset_settle_converged") is not True:
            raise StressFailure(
                "reset did not converge: "
                f"timed_out={info.get('garment_reset_settle_timed_out')}, "
                f"seconds={info.get('garment_reset_settle_seconds')}, "
                f"rms={info.get('garment_reset_settle_rms_speed')}, "
                f"max={info.get('garment_reset_settle_max_speed')}"
            )
        settle_steps = int(info.get("garment_reset_settle_steps", -1))
        settle_seconds = float(info.get("garment_reset_settle_seconds", math.nan))
        stable_steps = int(info.get("garment_reset_settle_stable_steps", -1))
        minimum_seconds = float(
            info.get("garment_reset_settle_min_seconds", math.nan)
        )
        stable_seconds = float(
            info.get("garment_reset_settle_stable_seconds", math.nan)
        )
        stable_window_drift = float(
            info.get("garment_reset_settle_window_drift", math.nan)
        )
        stable_window_drift_limit = float(
            info.get("garment_reset_settle_max_window_drift", math.nan)
        )
        if not all(
            math.isfinite(value) and value > 0.0
            for value in (settle_seconds, minimum_seconds, stable_seconds)
        ):
            raise StressFailure(
                "reset settle dwell metadata is missing/nonfinite: "
                f"settle={settle_seconds}, min={minimum_seconds}, "
                f"stable={stable_seconds}"
            )
        minimum_steps = int(math.ceil(minimum_seconds / PHYSICS_DT - 1e-12))
        stable_steps_required = int(
            math.ceil(stable_seconds / PHYSICS_DT - 1e-12)
        )
        if settle_steps < minimum_steps or settle_seconds + 1e-12 < minimum_seconds:
            raise StressFailure(
                "reset converged before its minimum physical dwell: "
                f"steps={settle_steps}/{minimum_steps}, "
                f"seconds={settle_seconds:.6f}/{minimum_seconds:.6f}"
            )
        if stable_steps < stable_steps_required:
            raise StressFailure(
                "reset lacks the required consecutive stable window: "
                f"stable_steps={stable_steps}/{stable_steps_required}"
            )
        if (
            not math.isfinite(stable_window_drift)
            or not math.isfinite(stable_window_drift_limit)
            or stable_window_drift < 0.0
            or stable_window_drift_limit <= 0.0
            or stable_window_drift > stable_window_drift_limit + 1e-9
        ):
            raise StressFailure(
                "reset stable-window drift contract failed: "
                f"drift={stable_window_drift}, "
                f"limit={stable_window_drift_limit}"
            )
        if not math.isclose(
            settle_seconds,
            settle_steps * PHYSICS_DT,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise StressFailure(
                "reset settle step/time metadata disagree: "
                f"steps={settle_steps}, seconds={settle_seconds:.9f}"
            )
        self.reset_settle_seconds_observed.append(settle_seconds)
        self.reset_stable_steps_observed.append(stable_steps)

        nodes = self._validate_state(
            info,
            previous_nodes=None,
            observation=observation,
        )
        public_tcp = np.asarray(
            observation["state"]["tcp_pose"], dtype=np.float64
        ).reshape(14)
        for offset, side in ((0, "right"), (7, "left")):
            command = np.asarray(
                self.env._plan1_command_filter.poses[side], dtype=np.float64
            )
            position_error = float(
                np.linalg.norm(public_tcp[offset : offset + 3] - command[:3])
            )
            rotation_error_deg = math.degrees(
                quaternion_angle_wxyz(
                    public_tcp[offset + 3 : offset + 7], command[3:7]
                )
            )
            if position_error > self.thresholds.max_reset_command_position_error_m:
                raise StressFailure(
                    f"{side} reset palm is {position_error:.6f} m from its public command"
                )
            if rotation_error_deg > self.thresholds.max_reset_command_rotation_error_deg:
                raise StressFailure(
                    f"{side} reset palm WXYZ orientation is {rotation_error_deg:.6f} deg "
                    "from its public command"
                )
        if self.reset_reference_nodes is None:
            self.reset_reference_nodes = nodes.copy()
        else:
            drift = float(
                np.max(
                    np.linalg.norm(nodes - self.reset_reference_nodes, axis=1)
                )
            )
            self.reset_max_node_drift_m = max(self.reset_max_node_drift_m, drift)
            if drift > self.thresholds.max_reset_drift_m + 1e-6:
                raise StressFailure(
                    f"reset node drift {drift:.6f} m exceeds "
                    f"{self.thresholds.max_reset_drift_m:.6f} m"
                )
        self.previous_nodes = nodes
        return observation, info

    def step(self, action: np.ndarray) -> Mapping[str, Any]:
        before = time.perf_counter()
        observation, _reward, terminated, truncated, info = self.env.step(action)
        elapsed = time.perf_counter() - before
        self.step_durations.append(elapsed)
        if self.phase is not None:
            self.phase.step_count += 1
            self.phase.simulated_seconds += CONTROL_DT
            self.phase.wall_seconds += elapsed
        nodes = self._validate_state(
            info,
            previous_nodes=self.previous_nodes,
            observation=observation,
        )
        self.previous_nodes = nodes
        if terminated or truncated:
            raise StressFailure(
                f"unexpected episode termination: terminated={terminated}, "
                f"truncated={truncated}, succeed={info.get('succeed')}, "
                f"failure={info.get('failure_reason')}"
            )
        return info

    def phase_run(
        self,
        name: str,
        callback: Callable[[], None],
        *,
        min_response_m: float | None = None,
        min_vertical_response_m: float | None = None,
        min_xy_response_m: float | None = None,
        max_response_m: float | None = None,
    ) -> None:
        if self.phase is not None:
            raise AssertionError("nested stress phases are not supported")
        nodes, _ = self._surface_state()
        metrics = PhaseMetrics(name=name)
        self.phase = metrics
        self.phase_reference_nodes = nodes
        try:
            callback()
            if min_response_m is not None and metrics.max_response_m < min_response_m:
                raise StressFailure(
                    f"{name} did not measurably contact cloth: response="
                    f"{metrics.max_response_m:.6f} m < {min_response_m:.6f} m"
                )
            if (
                min_vertical_response_m is not None
                and metrics.max_vertical_response_m < min_vertical_response_m
            ):
                raise StressFailure(
                    f"{name} did not lift cloth: vertical_response="
                    f"{metrics.max_vertical_response_m:.6f} m < "
                    f"{min_vertical_response_m:.6f} m"
                )
            if (
                min_xy_response_m is not None
                and metrics.max_xy_response_m < min_xy_response_m
            ):
                raise StressFailure(
                    f"{name} did not pull cloth: xy_response="
                    f"{metrics.max_xy_response_m:.6f} m < "
                    f"{min_xy_response_m:.6f} m"
                )
            if max_response_m is not None and metrics.max_response_m > max_response_m:
                raise StressFailure(
                    f"{name} drifted {metrics.max_response_m:.6f} m > "
                    f"{max_response_m:.6f} m"
                )
            metrics.passed = True
        except BaseException as exc:
            metrics.failure = str(exc)
            raise
        finally:
            self.phases.append(metrics)
            self.phase = None
            self.phase_reference_nodes = None

    def hold(self, seconds: float) -> None:
        steps = max(1, int(math.ceil(float(seconds) / CONTROL_DT)))
        for _ in range(steps):
            self.step(self.current_action())

    def drive(
        self,
        target_poses: Mapping[str, np.ndarray],
        target_hands: Mapping[str, np.ndarray] | None = None,
        *,
        linear_speed_m_s: float,
        hand_speed_rad_s: float = 1.2,
    ) -> None:
        start_poses = {
            "right": np.asarray(self.env._right_tcp_pose, dtype=np.float64).copy(),
            "left": np.asarray(self.env._left_tcp_pose, dtype=np.float64).copy(),
        }
        start_hands = {
            "right": np.asarray(self.env._right_hand, dtype=np.float64).copy(),
            "left": np.asarray(self.env._left_hand, dtype=np.float64).copy(),
        }
        final_poses = {
            side: np.asarray(target_poses.get(side, start_poses[side]), dtype=np.float64).copy()
            for side in ("right", "left")
        }
        final_hands = {
            side: np.asarray(
                (target_hands or {}).get(side, start_hands[side]), dtype=np.float64
            ).copy()
            for side in ("right", "left")
        }
        environment_control_dt = float(self.env._control_dt)
        max_translation_step = float(
            self.env._contact_max_tcp_translation_step
        )
        if (
            not math.isfinite(environment_control_dt)
            or environment_control_dt <= 0.0
            or not math.isfinite(max_translation_step)
            or max_translation_step <= 0.0
        ):
            raise StressFailure(
                "E01 environment reports an invalid control interval or "
                "translation-step limit"
            )
        # The Plan1 boundary clamps every submitted tracker command to the
        # environment's contact-safe translation step.  Count against that
        # effective speed, otherwise a request such as 0.12 m/s gets only half
        # the required frames when the live command boundary is capped at
        # 0.002 m per 1/30 s (0.06 m/s).
        effective_linear_speed_m_s = min(
            float(linear_speed_m_s),
            max_translation_step / environment_control_dt,
        )
        steps = motion_step_count(
            start_poses,
            final_poses,
            start_hands,
            final_hands,
            linear_speed_m_s=effective_linear_speed_m_s,
            hand_speed_rad_s=hand_speed_rad_s,
        )
        for index in range(1, steps + 1):
            fraction = index / steps
            poses: dict[str, np.ndarray] = {}
            hands: dict[str, np.ndarray] = {}
            for side in ("right", "left"):
                pose = (
                    (1.0 - fraction) * start_poses[side]
                    + fraction * final_poses[side]
                )
                pose[3:7] = _normalized_lerp_quat(
                    start_poses[side][3:7], final_poses[side][3:7], fraction
                )
                poses[side] = pose
                hands[side] = (
                    (1.0 - fraction) * start_hands[side]
                    + fraction * final_hands[side]
                )
            self.step(
                make_action(
                    poses["right"],
                    poses["left"],
                    hands["right"],
                    hands["left"],
                )
            )

    def contact_poses(self) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Choose repeatable cloth-relative hover/contact palm poses."""

        nodes, _ = self._surface_state()
        center_x = float(np.median(nodes[:, 0]))
        center_y = float(np.median(nodes[:, 1]))
        table_top = 0.89 + float(self.env.delta_h[0])
        contact: dict[str, np.ndarray] = {}
        hover: dict[str, np.ndarray] = {}
        for side, percentile in (("right", 38.0), ("left", 62.0)):
            pose = np.asarray(
                self.env._plan1_command_filter.poses[side], dtype=np.float64
            ).copy()
            x = float(np.clip(center_x, -0.36, 0.12))
            y_value = float(np.percentile(nodes[:, 1], percentile))
            if side == "right":
                y = float(np.clip(min(y_value, center_y - 0.025), -0.30, -0.035))
            else:
                y = float(np.clip(max(y_value, center_y + 0.025), 0.035, 0.30))
            pose[:2] = (x, y)

            probe = pose.copy()
            probe[2] = 0.0
            collision_guard = self.env._plan1_command_filter.collision_guards[side]
            collision_bottom_offset = collision_guard.minimum_collision_z(
                probe,
                np.asarray(self.env._plan1_command_filter.hands[side]),
            )
            desired_collision_bottom = table_top + 0.0015
            pose[2] = desired_collision_bottom - collision_bottom_offset
            contact[side] = pose.copy()
            hover[side] = pose.copy()
            hover[side][2] += 0.030
        return hover, contact

    def closed_hands(self, fraction: float) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {}
        for side in ("right", "left"):
            current = np.asarray(self.env._right_hand if side == "right" else self.env._left_hand)
            ranges = np.asarray(
                self.env._plan1_command_filter.joint_ranges[side], dtype=np.float64
            )
            result[side] = current + float(fraction) * (ranges[:, 1] - current)
        return result


def _run_suite(
    runner: E01StressRunner,
    *,
    idle_seconds: float,
    reset_cycles: int,
) -> None:
    thresholds = runner.thresholds
    runner.reset(seed=0)

    runner.phase_run(
        "idle",
        lambda: runner.hold(idle_seconds),
        max_response_m=thresholds.max_idle_drift_m,
    )

    for speed in POKE_SPEEDS_M_S:
        runner.reset(seed=0)

        def poke(speed_m_s: float = speed) -> None:
            hover, contact = runner.contact_poses()
            runner.drive(hover, linear_speed_m_s=0.12)
            runner.hold(0.30)
            runner.drive(
                {"right": contact["right"], "left": hover["left"]},
                linear_speed_m_s=speed_m_s,
            )
            runner.hold(0.75)
            tangent = {side: pose.copy() for side, pose in contact.items()}
            tangent["right"][0] += 0.012
            runner.drive(
                {"right": tangent["right"], "left": hover["left"]},
                linear_speed_m_s=speed_m_s,
            )
            runner.hold(0.35)
            runner.drive(hover, linear_speed_m_s=0.12)
            runner.hold(0.35)

        runner.phase_run(
            f"poke_{speed:.2f}m_s",
            poke,
            min_response_m=thresholds.min_contact_response_m,
        )

    runner.reset(seed=0)

    def pinch_lift() -> None:
        hover, contact = runner.contact_poses()
        opened = {
            "right": np.asarray(runner.env._right_hand, dtype=np.float64).copy(),
            "left": np.asarray(runner.env._left_hand, dtype=np.float64).copy(),
        }
        runner.drive(hover, linear_speed_m_s=0.12)
        runner.drive(contact, linear_speed_m_s=0.08)
        runner.hold(0.50)
        closed = runner.closed_hands(0.68)
        runner.drive(
            contact,
            closed,
            linear_speed_m_s=0.05,
            hand_speed_rad_s=0.9,
        )
        runner.hold(0.60)
        lifted = {side: pose.copy() for side, pose in contact.items()}
        for pose in lifted.values():
            pose[2] += 0.030
        runner.drive(lifted, closed, linear_speed_m_s=0.05)
        runner.hold(0.75)
        runner.drive(lifted, opened, linear_speed_m_s=0.05)
        runner.drive(hover, opened, linear_speed_m_s=0.10)
        runner.hold(0.40)

    runner.phase_run(
        "pinch_lift_release",
        pinch_lift,
        min_response_m=thresholds.min_contact_response_m,
        min_vertical_response_m=thresholds.min_pinch_vertical_response_m,
    )

    runner.reset(seed=0)

    def dual_pull() -> None:
        hover, contact = runner.contact_poses()
        opened = {
            "right": np.asarray(runner.env._right_hand, dtype=np.float64).copy(),
            "left": np.asarray(runner.env._left_hand, dtype=np.float64).copy(),
        }
        runner.drive(hover, linear_speed_m_s=0.12)
        runner.drive(contact, linear_speed_m_s=0.08)
        closed = runner.closed_hands(0.55)
        runner.drive(contact, closed, linear_speed_m_s=0.05)
        runner.hold(0.60)
        pulled = {side: pose.copy() for side, pose in contact.items()}
        pulled["right"][1] -= 0.020
        pulled["left"][1] += 0.020
        runner.drive(pulled, closed, linear_speed_m_s=0.05)
        runner.hold(0.75)
        runner.drive(pulled, opened, linear_speed_m_s=0.05)
        runner.drive(hover, opened, linear_speed_m_s=0.10)
        runner.hold(0.40)

    runner.phase_run(
        "dual_pull_release",
        dual_pull,
        min_response_m=thresholds.min_contact_response_m,
        min_xy_response_m=thresholds.min_dual_pull_xy_response_m,
    )

    runner.reset(seed=0)

    def tracker_faults() -> None:
        before = {
            side: np.asarray(runner.env._plan1_command_filter.poses[side]).copy()
            for side in ("right", "left")
        }
        jump = runner.current_action()
        jump[:3] += np.asarray([1.0, -1.0, 0.5])
        jump[7:10] += np.asarray([1.0, 1.0, 0.5])
        jump[3:7] = np.asarray([0.0, 1.0, 0.0, 0.0])
        jump[10:14] = np.asarray([0.0, 0.0, 1.0, 0.0])
        runner.step(jump)
        for side in ("right", "left"):
            after = np.asarray(runner.env._plan1_command_filter.poses[side])
            translation = float(np.linalg.norm(after[:3] - before[side][:3]))
            if translation > 0.004 + 1e-9:
                raise StressFailure(
                    f"{side} tracker jump bypassed 4 mm guard: {translation:.6f} m"
                )
            before_quat = before[side][3:7] / np.linalg.norm(before[side][3:7])
            after_quat = after[3:7] / np.linalg.norm(after[3:7])
            rotation = 2.0 * math.acos(
                float(np.clip(abs(np.dot(before_quat, after_quat)), 0.0, 1.0))
            )
            if rotation > math.radians(2.0) + 1e-9:
                raise StressFailure(
                    f"{side} tracker jump bypassed 2 degree guard: "
                    f"{math.degrees(rotation):.6f} deg"
                )

        held = {
            side: np.asarray(runner.env._plan1_command_filter.poses[side]).copy()
            for side in ("right", "left")
        }
        runner.step(np.full(54, np.nan, dtype=np.float64))
        for side in ("right", "left"):
            after = np.asarray(runner.env._plan1_command_filter.poses[side])
            if not np.allclose(after, held[side], rtol=0.0, atol=1e-12):
                raise StressFailure(f"{side} NaN tracker packet was not held")
        runner.hold(1.0)

    runner.phase_run("tracker_jump_nan", tracker_faults)

    for cycle in range(reset_cycles):
        runner.reset(seed=cycle)


def _write_report(path: Path | None, report: StressReport) -> None:
    payload = json.dumps(asdict(report), ensure_ascii=False, indent=2, sort_keys=True)
    if path is not None:
        path = path.expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload + "\n", encoding="utf-8")
    print(payload, flush=True)


def _print_plan(thresholds: Thresholds, *, idle_seconds: float, reset_cycles: int) -> None:
    print(
        json.dumps(
            {
                "launches_kit": False,
                "control_hz": 30,
                "physics_hz": E01_PHYSICS_HZ,
                "phases": [
                    f"idle_{idle_seconds:g}s_simulated",
                    *[f"poke_{speed:.2f}m_s" for speed in POKE_SPEEDS_M_S],
                    "pinch_lift_release",
                    "dual_pull_release",
                    "tracker_jump_nan",
                    f"reset_x{reset_cycles}",
                ],
                "thresholds": asdict(thresholds),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--idle-seconds", type=float, default=30.0)
    parser.add_argument("--reset-cycles", type=int, default=10)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument(
        "--allow-busy-gpu",
        action="store_true",
        help="Explicitly bypass the default refusal when CUDA compute owners exist.",
    )
    parser.add_argument(
        "--print-plan",
        action="store_true",
        help="Print phases/thresholds and exit before nvidia-smi, CUDA, or Kit imports.",
    )
    parser.add_argument("--max-p95-step-ms", type=float, default=25.0)
    parser.add_argument("--max-p99-step-ms", type=float, default=33.34)
    parser.add_argument("--max-overrun-fraction", type=float, default=0.01)
    parser.add_argument("--max-idle-drift-mm", type=float, default=5.0)
    parser.add_argument("--max-reset-drift-mm", type=float, default=5.0)
    parser.add_argument("--min-contact-response-mm", type=float, default=0.1)
    parser.add_argument("--min-lift-response-mm", type=float, default=1.0)
    parser.add_argument("--min-pull-response-mm", type=float, default=1.0)
    return parser


def _thresholds_from_args(args: argparse.Namespace) -> Thresholds:
    values = (
        args.max_p95_step_ms,
        args.max_p99_step_ms,
        args.max_overrun_fraction,
        args.max_idle_drift_mm,
        args.max_reset_drift_mm,
        args.min_contact_response_mm,
        args.min_lift_response_mm,
        args.min_pull_response_mm,
    )
    if not all(math.isfinite(float(value)) and float(value) >= 0.0 for value in values):
        raise ValueError("all stress thresholds must be finite and non-negative")
    return Thresholds(
        max_p95_step_ms=float(args.max_p95_step_ms),
        max_p99_step_ms=float(args.max_p99_step_ms),
        max_realtime_overrun_fraction=float(args.max_overrun_fraction),
        max_idle_drift_m=float(args.max_idle_drift_mm) / 1000.0,
        max_reset_drift_m=float(args.max_reset_drift_mm) / 1000.0,
        min_contact_response_m=float(args.min_contact_response_mm) / 1000.0,
        min_pinch_vertical_response_m=float(args.min_lift_response_mm) / 1000.0,
        min_dual_pull_xy_response_m=float(args.min_pull_response_mm) / 1000.0,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.idle_seconds <= 0.0 or not math.isfinite(args.idle_seconds):
        raise ValueError("--idle-seconds must be finite and positive")
    if args.reset_cycles < 1 or args.reset_cycles > 100:
        raise ValueError("--reset-cycles must be in [1, 100]")
    thresholds = _thresholds_from_args(args)
    if args.print_plan:
        _print_plan(
            thresholds,
            idle_seconds=args.idle_seconds,
            reset_cycles=args.reset_cycles,
        )
        return 0

    gpu = gpu_preflight(args.device, allow_busy_gpu=args.allow_busy_gpu)
    # Recheck as late as possible before importing the environment that owns
    # AppLauncher.  This narrows (but cannot entirely eliminate) the race
    # between the read-only preflight and CUDA context creation.
    gpu = gpu_preflight(args.device, allow_busy_gpu=args.allow_busy_gpu)
    sys.path.insert(0, str(HANDSON_ROOT))
    from handson.sim.isaaclab.e01_detached_hand_env import (  # noqa: PLC0415
        HandsOnE01DetachedHandsIsaacLabEnv,
    )

    report = StressReport(
        status="running",
        device=args.device,
        gpu=gpu,
        thresholds=asdict(thresholds),
        reset_cycles=args.reset_cycles,
    )
    env = None
    runner = None
    suite_start = time.perf_counter()
    exit_code = 1
    try:
        env = HandsOnE01DetachedHandsIsaacLabEnv(
            "e01_fold_garment",
            launch_isaac=True,
            headless=True,
            image_obs=False,
            render_mode="rgb_array",
            randomize=False,
            hz=0,
            control_dt=CONTROL_DT,
            physics_dt=PHYSICS_DT,
            time_limit=900.0,
            isaac_device=args.device,
            seed=0,
        )
        runner = E01StressRunner(env, thresholds)
        _run_suite(
            runner,
            idle_seconds=args.idle_seconds,
            reset_cycles=args.reset_cycles,
        )
        durations_ms = [duration * 1000.0 for duration in runner.step_durations]
        report.phases = [asdict(phase) for phase in runner.phases]
        report.simulation_node_count = runner.simulation_node_count
        report.simulation_triangle_count = runner.simulation_triangle_count
        report.visual_vertex_count = runner.visual_vertex_count
        report.visual_triangle_count = runner.visual_triangle_count
        report.reset_max_node_drift_m = runner.reset_max_node_drift_m
        report.reset_min_settle_seconds_observed = min(
            runner.reset_settle_seconds_observed
        )
        report.reset_max_settle_seconds_observed = max(
            runner.reset_settle_seconds_observed
        )
        report.reset_min_stable_steps_observed = min(
            runner.reset_stable_steps_observed
        )
        report.step_count = len(durations_ms)
        report.simulated_seconds = len(durations_ms) * CONTROL_DT
        report.wall_seconds = time.perf_counter() - suite_start
        report.step_p50_ms = float(statistics.median(durations_ms))
        report.step_p95_ms = _percentile(durations_ms, 95.0)
        report.step_p99_ms = _percentile(durations_ms, 99.0)
        report.realtime_overrun_fraction = float(
            np.mean(np.asarray(durations_ms) > CONTROL_DT * 1000.0)
        )
        if report.step_p95_ms > thresholds.max_p95_step_ms:
            raise StressFailure(
                f"step p95 {report.step_p95_ms:.3f} ms exceeds "
                f"{thresholds.max_p95_step_ms:.3f} ms"
            )
        if report.step_p99_ms > thresholds.max_p99_step_ms:
            raise StressFailure(
                f"step p99 {report.step_p99_ms:.3f} ms exceeds "
                f"{thresholds.max_p99_step_ms:.3f} ms"
            )
        if (
            report.realtime_overrun_fraction
            > thresholds.max_realtime_overrun_fraction
        ):
            raise StressFailure(
                "real-time overrun fraction "
                f"{report.realtime_overrun_fraction:.4f} exceeds "
                f"{thresholds.max_realtime_overrun_fraction:.4f}"
            )
        report.status = "passed"
        exit_code = 0
    except BaseException as exc:
        report.status = "failed"
        report.failure = f"{type(exc).__name__}: {exc}"
        if runner is not None:
            report.phases = [asdict(phase) for phase in runner.phases]
            report.simulation_node_count = runner.simulation_node_count
            report.simulation_triangle_count = runner.simulation_triangle_count
            report.visual_vertex_count = runner.visual_vertex_count
            report.visual_triangle_count = runner.visual_triangle_count
            report.reset_max_node_drift_m = runner.reset_max_node_drift_m
            if runner.reset_settle_seconds_observed:
                report.reset_min_settle_seconds_observed = min(
                    runner.reset_settle_seconds_observed
                )
                report.reset_max_settle_seconds_observed = max(
                    runner.reset_settle_seconds_observed
                )
            if runner.reset_stable_steps_observed:
                report.reset_min_stable_steps_observed = min(
                    runner.reset_stable_steps_observed
                )
            report.step_count = len(runner.step_durations)
            report.simulated_seconds = len(runner.step_durations) * CONTROL_DT
            if runner.step_durations:
                durations_ms = [value * 1000.0 for value in runner.step_durations]
                report.step_p50_ms = float(statistics.median(durations_ms))
                report.step_p95_ms = _percentile(durations_ms, 95.0)
                report.step_p99_ms = _percentile(durations_ms, 99.0)
                report.realtime_overrun_fraction = float(
                    np.mean(np.asarray(durations_ms) > CONTROL_DT * 1000.0)
                )
        report.wall_seconds = time.perf_counter() - suite_start
        print(f"[E01 GPU STRESS] FAILED: {report.failure}", file=sys.stderr, flush=True)
    finally:
        try:
            _write_report(args.json_out, report)
        finally:
            if env is not None:
                # Kit shutdown occasionally blocks on extension teardown.  The
                # timer only terminates this harness process; it never signals any
                # external collection process.
                shutdown_guard = threading.Timer(10.0, lambda: os._exit(exit_code))
                shutdown_guard.daemon = True
                shutdown_guard.start()
                try:
                    env.close()
                finally:
                    shutdown_guard.cancel()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
