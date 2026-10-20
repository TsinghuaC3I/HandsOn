"""B03 action-aligned NVIDIA Newton readback and drive-target trace.

Newton and PhysX expose the same Isaac Lab articulation/task buffers, but
Newton does not expose PhysX contact-report buffers.  Reuse the mature,
name-mapped readback implementation while publishing a separate ``newton_*``
schema and omitting fields that would falsely claim PhysX contact provenance.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from typing import Any

import numpy as np

from ...data.b03_newton_validity import (
    B03_NEWTON_HAND_PEG_MAX_SOLID_PENETRATION_M,
    b03_newton_source_validity_gate_contract,
)
from .physx_ground_truth import (
    PHYSX_STATE_LAYOUT,
    ROBOT_SIDES,
    PhysxGroundTruthTrace,
    _first_row,
    _numpy,
)
from .newton_determinism import capture_newton_runtime_identity


NEWTON_GROUND_TRUTH_ENV = "HANDSON_RECORD_NEWTON_GROUND_TRUTH"
B03_HAND_PEG_MAX_SOLID_PENETRATION_M = (
    B03_NEWTON_HAND_PEG_MAX_SOLID_PENETRATION_M
)

_B03_CONTACT_FIELDS = (
    "physx_hand_peg_contact_valid",
    "physx_hand_peg_contact_count",
    "physx_hand_peg_min_solid_separation",
    "physx_hand_peg_max_solid_penetration",
)


def newton_ground_truth_enabled() -> bool:
    """Return whether B03's isolated Newton trace is enabled."""

    return os.environ.get(NEWTON_GROUND_TRUTH_ENV, "0").strip().lower() not in {
        "",
        "0",
        "false",
        "no",
        "off",
    }


def newton_execution_config(env: Any) -> dict[str, Any]:
    """Return the exact requested/effective Newton graph execution contract."""

    requested = bool(getattr(env, "_newton_use_cuda_graph", False))
    captured = bool(getattr(env, "_newton_cuda_graph_captured", False))
    return {
        "cuda_graph_requested": requested,
        "use_cuda_graph": bool(requested and captured),
        "cuda_graph_captured": captured,
        "cuda_graph_scope": str(
            getattr(env, "_newton_cuda_graph_scope", "disabled")
        ),
        "cuda_graph_capture_mode": str(
            getattr(env, "_newton_cuda_graph_capture_mode", "disabled")
        ),
        "capture_completed_before_episode_reset": bool(
            getattr(env, "_newton_graph_captured_before_last_reset", False)
        ),
        "newton_manager_num_substeps": 1,
        "newton_manager_decimation": int(
            getattr(env, "_newton_manager_decimation", 1)
        ),
    }


def _newton_name(name: str) -> str:
    return name.replace("physx_", "newton_", 1)


def _physx_name(name: str) -> str:
    return name.replace("newton_", "physx_", 1)


def _newton_stream_token(stream: Any) -> tuple[str, int]:
    """Return a stable identity for one Warp CUDA stream wrapper."""

    device = str(getattr(stream, "device", ""))
    raw_handle = getattr(stream, "cuda_stream", None)
    try:
        handle = int(raw_handle) if raw_handle else 0
    except (TypeError, ValueError):
        handle = id(raw_handle)
    return device, handle


@dataclass(frozen=True)
class _NewtonRingField:
    """One fixed-layout CUDA source and its owned frame readback storage."""

    shape: tuple[int, ...]
    size: int
    dtype: Any
    device: Any
    device_ring: Any
    host_ring: Any


class _NewtonGpuFrameRing:
    """Stage substeps on-device and perform one synchronized frame readback.

    The ring deliberately knows nothing about the published trace schema.  It
    owns exact byte copies of every registered CUDA source, which makes it
    small enough to test without Isaac Sim while keeping stream ordering and
    completion proof in one place.
    """

    def __init__(
        self,
        wp: Any,
        *,
        slot_count: int,
        sources: Mapping[str, Any],
        stream: Any,
    ) -> None:
        if int(slot_count) <= 0:
            raise ValueError("Newton GPU trace ring needs at least one slot")
        if not sources:
            raise RuntimeError("Newton GPU trace has no CUDA sources")
        self.wp = wp
        self.slot_count = int(slot_count)
        self._fields: dict[str, _NewtonRingField] = {}
        self._capture_cursor = 0
        self._frame_stream_token: tuple[str, int] | None = None
        self._initialization_stream_token = _newton_stream_token(stream)
        self._has_materialized_frame = False
        self._frame_sync_count = 0
        self._last_d2h_copy_count = 0

        expected_device: Any | None = None
        for name, raw_source in sources.items():
            source = self._as_warp_array(raw_source)
            self._validate_cuda_source(name, source)
            if expected_device is None:
                expected_device = source.device
            elif str(source.device) != str(expected_device):
                raise RuntimeError(
                    "Newton GPU trace sources span CUDA devices: "
                    f"expected={expected_device}, {name}={source.device}"
                )
            size = int(source.size)
            shape = tuple(int(value) for value in source.shape)
            device_ring = wp.empty(
                shape=self.slot_count * size,
                dtype=source.dtype,
                device=source.device,
            )
            host_ring = wp.empty(
                shape=self.slot_count * size,
                dtype=source.dtype,
                device="cpu",
                pinned=True,
            )
            self._fields[str(name)] = _NewtonRingField(
                shape=shape,
                size=size,
                dtype=source.dtype,
                device=source.device,
                device_ring=device_ring,
                host_ring=host_ring,
            )

        assert expected_device is not None
        self.device = expected_device
        marker_dtype = wp.int32
        zero_markers = np.zeros(self.slot_count, dtype=np.int32)
        expected_markers = np.arange(
            1,
            self.slot_count + 1,
            dtype=np.int32,
        )
        self._zero_markers = wp.array(
            zero_markers,
            dtype=marker_dtype,
            device=self.device,
        )
        self._expected_markers = wp.array(
            expected_markers,
            dtype=marker_dtype,
            device=self.device,
        )
        self._completion_device = wp.empty(
            shape=self.slot_count,
            dtype=marker_dtype,
            device=self.device,
        )
        self._completion_host = wp.empty(
            shape=self.slot_count,
            dtype=marker_dtype,
            device="cpu",
            pinned=True,
        )
        # Marker initialization is enqueued on the same current stream used
        # by the first ``begin`` call.  It intentionally does not synchronize;
        # the frame's single terminal synchronization covers it as well.

    def _as_warp_array(self, value: Any) -> Any:
        raw = value
        try:
            raw = raw.warp
        except AttributeError:
            try:
                raw = raw.torch
            except AttributeError:
                pass
        if hasattr(raw, "detach"):
            contiguous = getattr(raw, "is_contiguous", None)
            if callable(contiguous) and not bool(contiguous()):
                raise RuntimeError(
                    "Newton GPU trace refuses a non-contiguous Torch source"
                )
            raw = self.wp.from_torch(raw)
        return raw

    @staticmethod
    def _validate_cuda_source(name: str, source: Any) -> None:
        required = ("device", "dtype", "shape", "size", "is_contiguous")
        missing = [field for field in required if not hasattr(source, field)]
        if missing:
            raise RuntimeError(
                f"Newton GPU trace source {name!r} is not a Warp array; "
                f"missing={missing}"
            )
        if not bool(getattr(source.device, "is_cuda", False)):
            raise RuntimeError(
                f"Newton GPU trace source {name!r} is not CUDA-backed: "
                f"{source.device}"
            )
        if not bool(source.is_contiguous):
            raise RuntimeError(
                f"Newton GPU trace source {name!r} is non-contiguous"
            )
        if int(source.size) <= 0:
            raise RuntimeError(
                f"Newton GPU trace source {name!r} is empty"
            )

    def _require_stream(self, stream: Any) -> None:
        token = _newton_stream_token(stream)
        if self._frame_stream_token is None:
            raise RuntimeError("Newton GPU trace frame was not started")
        if token != self._frame_stream_token:
            raise RuntimeError(
                "Newton GPU trace current stream changed inside a control "
                f"frame: expected={self._frame_stream_token}, actual={token}"
            )

    def begin(self, stream: Any) -> None:
        if str(getattr(stream, "device", "")) != str(self.device):
            raise RuntimeError(
                "Newton GPU trace stream/device mismatch: "
                f"stream={getattr(stream, 'device', None)}, "
                f"sources={self.device}"
            )
        token = _newton_stream_token(stream)
        if (
            not self._has_materialized_frame
            and token != self._initialization_stream_token
        ):
            raise RuntimeError(
                "Newton GPU trace stream changed between lazy allocation and "
                f"its first frame: initialized={self._initialization_stream_token}, "
                f"actual={token}"
            )
        self._capture_cursor = 0
        self._frame_sync_count = 0
        self._last_d2h_copy_count = 0
        self._frame_stream_token = token
        self.wp.copy(
            self._completion_device,
            self._zero_markers,
            count=self.slot_count,
            stream=stream,
        )

    def stage(self, sources: Mapping[str, Any], stream: Any) -> int:
        self._require_stream(stream)
        slot = self._capture_cursor
        if slot >= self.slot_count:
            raise RuntimeError(
                "Newton GPU trace received more substeps than its ring can hold"
            )
        if set(sources) != set(self._fields):
            missing = sorted(set(self._fields) - set(sources))
            extra = sorted(set(sources) - set(self._fields))
            raise RuntimeError(
                "Newton GPU trace source set changed inside an episode: "
                f"missing={missing}, extra={extra}"
            )
        for name, field in self._fields.items():
            source = self._as_warp_array(sources[name])
            self._validate_cuda_source(name, source)
            if (
                tuple(int(value) for value in source.shape) != field.shape
                or int(source.size) != field.size
                or source.dtype != field.dtype
                or str(source.device) != str(field.device)
            ):
                raise RuntimeError(
                    "Newton GPU trace source layout changed: "
                    f"name={name}, shape={tuple(source.shape)}, "
                    f"dtype={source.dtype}, device={source.device}"
                )
            self.wp.copy(
                field.device_ring,
                source,
                dest_offset=slot * field.size,
                count=field.size,
                stream=stream,
            )
        # This D2D marker is ordered after every field copy on the exact same
        # stream.  A complete CPU marker sequence therefore proves all eight
        # slots were staged, rather than merely trusting a Python counter.
        self.wp.copy(
            self._completion_device,
            self._expected_markers,
            dest_offset=slot,
            src_offset=slot,
            count=1,
            stream=stream,
        )
        self._capture_cursor += 1
        return slot

    def materialize(self, stream: Any) -> dict[str, np.ndarray]:
        self._require_stream(stream)
        if self._capture_cursor != self.slot_count:
            raise RuntimeError(
                "Newton GPU trace frame is incomplete: "
                f"captured={self._capture_cursor}, expected={self.slot_count}"
            )
        d2h_count = 0
        for field in self._fields.values():
            self.wp.copy(
                field.host_ring,
                field.device_ring,
                count=self.slot_count * field.size,
                stream=stream,
            )
            d2h_count += 1
        self.wp.copy(
            self._completion_host,
            self._completion_device,
            count=self.slot_count,
            stream=stream,
        )
        d2h_count += 1
        # The sole control-frame synchronization.  No host array is examined
        # before it completes.
        self.wp.synchronize_stream(stream)
        self._frame_sync_count += 1
        self._last_d2h_copy_count = d2h_count

        completion = np.asarray(self._completion_host).reshape(-1).copy()
        expected = np.arange(1, self.slot_count + 1, dtype=np.int32)
        if completion.shape != expected.shape or not np.array_equal(
            completion,
            expected,
        ):
            raise RuntimeError(
                "Newton GPU trace completion sentinel mismatch: "
                f"actual={completion.tolist()}, expected={expected.tolist()}"
            )

        result: dict[str, np.ndarray] = {}
        for name, field in self._fields.items():
            host = np.asarray(field.host_ring)
            if host.shape[0] != self.slot_count * field.size:
                raise RuntimeError(
                    "Newton GPU trace pinned mirror has an invalid shape: "
                    f"name={name}, shape={host.shape}"
                )
            component_shape = tuple(int(value) for value in host.shape[1:])
            result[name] = host.reshape(
                self.slot_count,
                *field.shape,
                *component_shape,
            ).copy()
        if self._frame_sync_count != 1:
            raise RuntimeError(
                "Newton GPU trace performed an unexpected number of frame "
                f"synchronizations: {self._frame_sync_count}"
            )
        self._has_materialized_frame = True
        return result

    @property
    def frame_sync_count(self) -> int:
        return int(self._frame_sync_count)

    @property
    def last_d2h_copy_count(self) -> int:
        return int(self._last_d2h_copy_count)


class NewtonGroundTruthTrace(PhysxGroundTruthTrace):
    """Record B03 Newton state without pretending contacts came from PhysX."""

    _OMITTED_PREFIXES = (
        "physx_contact_",
        "physx_friction_anchor_",
        "physx_substep_contact_",
        "physx_substep_friction_anchor_",
        "physx_deformable_",
        "physx_substep_deformable_",
    )
    _SUBSTEP_CAPTURE_FIELDS = (
        *PhysxGroundTruthTrace._SUBSTEP_CAPTURE_FIELDS,
        *_B03_CONTACT_FIELDS,
    )

    _ENDPOINT_REUSED_ROBOT_FIELDS = (
        "physx_robot_joint_position",
        "physx_robot_joint_velocity",
        "physx_robot_joint_effort",
        "physx_robot_joint_valid",
        "physx_robot_joint_effort_valid",
        "physx_tcp_state",
        "physx_tcp_valid",
    )
    _ENDPOINT_ALWAYS_FIELDS = (
        *_ENDPOINT_REUSED_ROBOT_FIELDS,
        "physx_robot_joint_target",
        "physx_robot_joint_target_valid",
        "physx_robot_root_state",
        "physx_robot_root_valid",
        "physx_command_tcp_pose",
        "physx_command_hand",
        "physx_command_hand_valid",
        "physx_pose_commanded",
        "physx_pose_filter_state",
        "physx_pose_filter_valid",
        *_B03_CONTACT_FIELDS,
        "physx_sim_time",
        "physx_control_step",
    )

    def __init__(self, env: Any):
        # These caches contain only immutable model metadata or owned NumPy
        # copies.  They never retain live CUDA/Warp views across substeps.
        self._newton_contact_static_model: Any | None = None
        self._newton_contact_static_metadata_cache: (
            tuple[np.ndarray, np.ndarray, tuple[str, ...]] | None
        ) = None
        self._newton_wp: Any | None = None
        self._newton_manager: Any | None = None
        self._newton_gpu_batch: _NewtonGpuFrameRing | None = None
        self._newton_gpu_batch_requested = False
        self._newton_cpu_substeps: list[dict[str, np.ndarray]] = []
        self._newton_frame_active = False
        self._newton_frame_materialized = False
        self._newton_frame_env_step = -1
        self._newton_frame_origin_id = -1
        self._newton_last_capture_state_write_count = -1
        self._newton_effort_source_by_side: dict[str, str] = {}
        self._newton_root_source_by_side: dict[str, str] = {}
        self._newton_latest_rigid_states: dict[str, np.ndarray] = {}
        self._newton_latest_robot_states: dict[str, dict[str, np.ndarray]] = {}
        self._newton_latest_state_env_step = -1
        self._newton_latest_state_write_count = -1
        self._newton_latest_origin_id = -1
        self._clear_newton_endpoint_cache()
        super().__init__(env)
        self._newton_gpu_batch_requested = bool(
            getattr(env, "task_name", None) == "t09_insert_peg"
            and getattr(env, "_physics_backend", None) == "newton"
        )

    def _clear_newton_endpoint_cache(self) -> None:
        self._newton_endpoint_cache: dict[str, np.ndarray] | None = None
        self._newton_endpoint_cache_substep_count = 0
        self._newton_endpoint_cache_env_step = -1
        self._newton_endpoint_cache_state_write_count = -1

    @staticmethod
    def _newton_as_warp_array(wp: Any, value: Any) -> Any:
        raw = value
        try:
            raw = raw.warp
        except AttributeError:
            try:
                raw = raw.torch
            except AttributeError:
                pass
        if hasattr(raw, "detach"):
            contiguous = getattr(raw, "is_contiguous", None)
            if callable(contiguous) and not bool(contiguous()):
                raise RuntimeError(
                    "Newton GPU trace refuses a non-contiguous Torch source"
                )
            raw = wp.from_torch(raw)
        return raw

    def _newton_current_stream(
        self,
        sources: Mapping[str, Any] | None = None,
    ) -> Any:
        wp = self._newton_wp
        if wp is None:
            raise RuntimeError("Newton GPU trace has no Warp runtime")
        batch = self._newton_gpu_batch
        if batch is not None:
            return wp.get_stream(batch.device)
        if not sources:
            raise RuntimeError("Newton GPU trace cannot resolve its CUDA device")
        first = self._newton_as_warp_array(wp, next(iter(sources.values())))
        return wp.get_stream(first.device)

    def _select_robot_cuda_sources(self) -> dict[str, Any]:
        sources: dict[str, Any] = {}
        for entry in self.robots:
            side = entry.side
            robot = entry.robot
            if robot is None:
                raise RuntimeError(
                    f"Newton GPU trace is missing the {side} robot"
                )
            if entry.tcp_body_index is None:
                raise RuntimeError(
                    f"Newton GPU trace is missing the {side} TCP body index"
                )
            target = getattr(robot.data, "joint_pos_target", None)
            if target is None:
                raise RuntimeError(
                    f"Newton GPU trace is missing the {side} applied target"
                )
            sources[f"robot/{side}/joint_position"] = robot.data.joint_pos
            sources[f"robot/{side}/joint_velocity"] = robot.data.joint_vel
            sources[f"robot/{side}/joint_target"] = target
            sources[f"robot/{side}/body_state"] = robot.data.body_state_w

            effort_name = self._newton_effort_source_by_side.get(side)
            if effort_name is None:
                for candidate in ("applied_torque", "computed_torque"):
                    if getattr(robot.data, candidate, None) is not None:
                        effort_name = candidate
                        break
                if effort_name is None:
                    raise RuntimeError(
                        f"Newton GPU trace is missing {side} joint effort"
                    )
                self._newton_effort_source_by_side[side] = effort_name
            effort = getattr(robot.data, effort_name, None)
            if effort is None:
                raise RuntimeError(
                    f"Newton GPU trace lost {side}.{effort_name}"
                )
            sources[f"robot/{side}/joint_effort"] = effort

            root_name = self._newton_root_source_by_side.get(side)
            if root_name is None:
                for candidate in ("root_state_w", "root_link_state_w"):
                    if getattr(robot.data, candidate, None) is not None:
                        root_name = candidate
                        break
                if root_name is None:
                    raise RuntimeError(
                        f"Newton GPU trace is missing the {side} robot root"
                    )
                self._newton_root_source_by_side[side] = root_name
            root = getattr(robot.data, root_name, None)
            if root is None:
                raise RuntimeError(
                    f"Newton GPU trace lost {side}.{root_name}"
                )
            sources[f"robot/{side}/root_state"] = root
        return sources

    def _select_task_cuda_sources(self) -> dict[str, Any]:
        sources: dict[str, Any] = {}
        for index, entry in enumerate(self.rigid_bodies):
            prefix = f"rigid/{index}"
            if entry.source == "physx_tensor_view":
                sources[f"{prefix}/transform"] = entry.handle.get_transforms()
                native_view = getattr(entry.handle, "_view", None)
                is_floating = getattr(native_view, "is_floating_base", None)
                if is_floating is None or bool(is_floating):
                    sources[f"{prefix}/velocity"] = (
                        entry.handle.get_velocities()
                    )
            else:
                sources[f"{prefix}/root_state"] = (
                    entry.handle.data.root_state_w
                )
        for index, entry in enumerate(self.articulations):
            prefix = f"articulation/{index}"
            sources[f"{prefix}/root_transform"] = (
                entry.view.get_root_transforms()
            )
            sources[f"{prefix}/root_velocity"] = (
                entry.view.get_root_velocities()
            )
            if self.max_articulation_dofs > 0 and entry.dof_count > 0:
                sources[f"{prefix}/dof_position"] = (
                    entry.view.get_dof_positions()
                )
                sources[f"{prefix}/dof_velocity"] = (
                    entry.view.get_dof_velocities()
                )
        return sources

    def _select_contact_cuda_sources(self) -> dict[str, Any]:
        manager = self._newton_manager
        if manager is None:
            raise RuntimeError("Newton GPU trace has no NewtonManager")
        contacts = manager.get_contacts()
        state = manager.get_state_0()
        model = manager.get_model()
        if contacts is None or state is None or model is None:
            raise RuntimeError(
                "Newton GPU trace model/contact/state buffers are missing"
            )
        if (
            self._newton_contact_static_model is not None
            and self._newton_contact_static_model is not model
        ):
            raise RuntimeError(
                "Newton finalized model changed without rebuilding the trace"
            )
        self._newton_contact_static_model = model
        return {
            "contact/count": contacts.rigid_contact_count,
            "contact/shape0": contacts.rigid_contact_shape0,
            "contact/shape1": contacts.rigid_contact_shape1,
            "contact/point0": contacts.rigid_contact_point0,
            "contact/point1": contacts.rigid_contact_point1,
            "contact/normal": contacts.rigid_contact_normal,
            "contact/margin0": contacts.rigid_contact_margin0,
            "contact/margin1": contacts.rigid_contact_margin1,
            "contact/body_q": state.body_q,
            "contact/model_shape_body": model.shape_body,
            "contact/model_shape_margin": model.shape_margin,
        }

    def _newton_cuda_sources(self) -> dict[str, Any]:
        sources = self._select_robot_cuda_sources()
        sources.update(self._select_task_cuda_sources())
        sources.update(self._select_contact_cuda_sources())
        return sources

    def _initialize_newton_gpu_batch_capture(self) -> None:
        """Allocate the B03 ring before the first recorded control frame."""

        try:
            import warp as wp
            from isaaclab_newton.physics import NewtonManager
        except Exception as exc:
            raise RuntimeError(
                "B03 Newton recording requires Warp/Newton GPU readback"
            ) from exc
        self._newton_wp = wp
        self._newton_manager = NewtonManager
        model = NewtonManager.get_model()
        if model is None:
            raise RuntimeError("B03 Newton recording has no finalized model")
        # Shape/body tables stay CUDA-resident here.  They join the first
        # frame's ring readback so lazy setup never adds a hidden synchronous
        # ``.numpy()`` call to that action interval.
        self._newton_contact_static_model = model
        sources = self._newton_cuda_sources()
        stream = self._newton_current_stream(sources)
        self._newton_gpu_batch = _NewtonGpuFrameRing(
            wp,
            slot_count=max(1, int(getattr(self.env, "_n_substeps", 1))),
            sources=sources,
            stream=stream,
        )

    def _sample_newton_substep_robot_fields(self) -> dict[str, np.ndarray]:
        """Sample only robot fields published in the substep schema.

        The shared PhysX sampler also reads both robot roots.  Root state is an
        endpoint-only field, so doing that on every Newton physics substep
        adds two blocking CUDA readbacks whose results are immediately
        discarded by ``_SUBSTEP_CAPTURE_FIELDS``.
        """

        shape = (len(ROBOT_SIDES), self.max_robot_joints)
        position = np.zeros(shape, dtype=np.float64)
        velocity = np.zeros(shape, dtype=np.float64)
        target = np.zeros(shape, dtype=np.float64)
        effort = np.zeros(shape, dtype=np.float64)
        valid = np.zeros(shape, dtype=np.uint8)
        target_valid = np.zeros(shape, dtype=np.uint8)
        effort_valid = np.zeros(shape, dtype=np.uint8)
        tcp_state = np.zeros((len(ROBOT_SIDES), 13), dtype=np.float64)
        tcp_valid = np.zeros((len(ROBOT_SIDES),), dtype=np.uint8)
        joint_targets = getattr(self.env, "_joint_targets", {})

        for side_index, entry in enumerate(self.robots):
            robot = entry.robot
            if robot is None:
                continue
            try:
                actual_position = _first_row(robot.data.joint_pos).reshape(-1)
                actual_velocity = _first_row(robot.data.joint_vel).reshape(-1)
                count = min(
                    entry.joint_count,
                    actual_position.size,
                    actual_velocity.size,
                    self.max_robot_joints,
                )
                finite = (
                    np.isfinite(actual_position[:count])
                    & np.isfinite(actual_velocity[:count])
                )
                position[side_index, :count][finite] = actual_position[:count][
                    finite
                ]
                velocity[side_index, :count][finite] = actual_velocity[:count][
                    finite
                ]
                valid[side_index, :count] = finite.astype(np.uint8)
            except Exception:
                pass

            for source_name in ("applied_torque", "computed_torque"):
                try:
                    effort_values = _first_row(
                        getattr(robot.data, source_name)
                    ).reshape(-1)
                except Exception:
                    continue
                effort_count = min(effort_values.size, self.max_robot_joints)
                finite = np.isfinite(effort_values[:effort_count])
                effort[side_index, :effort_count][finite] = effort_values[
                    :effort_count
                ][finite]
                effort_valid[side_index, :effort_count] = finite.astype(
                    np.uint8
                )
                break

            if isinstance(joint_targets, Mapping) and entry.side in joint_targets:
                try:
                    target_values = _first_row(
                        joint_targets[entry.side]
                    ).reshape(-1)
                    target_count = min(target_values.size, self.max_robot_joints)
                    finite = np.isfinite(target_values[:target_count])
                    target[side_index, :target_count][finite] = target_values[
                        :target_count
                    ][finite]
                    target_valid[side_index, :target_count] = finite.astype(
                        np.uint8
                    )
                except Exception:
                    pass

            if entry.tcp_body_index is not None:
                try:
                    body_states = _numpy(robot.data.body_state_w)
                    tcp = body_states[0, entry.tcp_body_index, :13]
                    if tcp.shape == (13,) and np.all(np.isfinite(tcp)):
                        tcp_state[side_index] = tcp
                        tcp_valid[side_index] = 1
                except Exception:
                    pass

        return {
            "physx_robot_joint_position": position,
            "physx_robot_joint_velocity": velocity,
            "physx_robot_joint_target": target,
            "physx_robot_joint_target_valid": target_valid,
            "physx_robot_joint_valid": valid,
            "physx_robot_joint_effort": effort,
            "physx_robot_joint_effort_valid": effort_valid,
            "physx_tcp_state": tcp_state,
            "physx_tcp_valid": tcp_valid,
        }

    def _initialize_contact_report_capture(self) -> None:
        """Newton contacts are not represented by PhysX report APIs."""

    def _initialize_tensor_contact_capture(self) -> None:
        """Do not construct PhysX tensor contact views for a Newton world."""

    @staticmethod
    def _warp_numpy(value: Any) -> np.ndarray:
        numpy = getattr(value, "numpy", None)
        if not callable(numpy):
            raise RuntimeError("Newton contact buffer has no NumPy readback")
        return np.asarray(numpy())

    @staticmethod
    def _transform_point(transform: np.ndarray, point: np.ndarray) -> np.ndarray:
        """Apply a Newton xyz+xyzw body transform to one local point."""

        value = np.asarray(transform, dtype=np.float64).reshape(7)
        local = np.asarray(point, dtype=np.float64).reshape(3)
        quaternion = value[3:7]
        norm = float(np.linalg.norm(quaternion))
        if norm <= 1.0e-12:
            return value[:3] + local
        x, y, z, w = quaternion / norm
        rotation = np.asarray(
            [
                [
                    1.0 - 2.0 * (y * y + z * z),
                    2.0 * (x * y - z * w),
                    2.0 * (x * z + y * w),
                ],
                [
                    2.0 * (x * y + z * w),
                    1.0 - 2.0 * (x * x + z * z),
                    2.0 * (y * z - x * w),
                ],
                [
                    2.0 * (x * z - y * w),
                    2.0 * (y * z + x * w),
                    1.0 - 2.0 * (x * x + y * y),
                ],
            ],
            dtype=np.float64,
        )
        return value[:3] + rotation @ local

    @staticmethod
    def _is_hand_shape(label: str) -> bool:
        lower = label.lower()
        return (
            ("/rightrobot/" in lower or "/leftrobot/" in lower)
            and ("_col2" in lower or "collision" in lower)
        )

    @staticmethod
    def _is_b03_peg_shape(label: str) -> bool:
        lower = label.lower()
        return (
            "industreal_round_peg_8mm" in lower
            and "industreal_tray_insert_round_peg_8mm" not in lower
        )

    def _contact_static_metadata(
        self,
        model: Any,
    ) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
        """Read immutable Newton shape metadata once per finalized model."""

        cached_model = getattr(self, "_newton_contact_static_model", None)
        cached = getattr(
            self,
            "_newton_contact_static_metadata_cache",
            None,
        )
        if cached_model is model and cached is not None:
            return cached

        shape_body = (
            self._warp_numpy(model.shape_body).reshape(-1).copy()
        )
        shape_margin = (
            self._warp_numpy(model.shape_margin).reshape(-1).copy()
        )
        raw_labels = getattr(model, "shape_label", None)
        labels = (
            tuple(str(label) for label in raw_labels)
            if raw_labels is not None
            else ()
        )
        # Make accidental mutation fail locally instead of silently changing
        # the contact classifier for the rest of an episode.
        shape_body.setflags(write=False)
        shape_margin.setflags(write=False)
        cached = (shape_body, shape_margin, labels)
        self._newton_contact_static_model = model
        self._newton_contact_static_metadata_cache = cached
        return cached

    def _contact_fields_from_arrays(
        self,
        *,
        count: Any,
        shape0: Any,
        shape1: Any,
        point0: Any,
        point1: Any,
        normal: Any,
        margin0: Any,
        margin1: Any,
        body_q: Any,
        strict: bool,
    ) -> dict[str, np.ndarray]:
        """Derive B03 contact evidence from already-owned host arrays."""

        result = {
            "physx_hand_peg_contact_valid": np.asarray([0], dtype=np.uint8),
            "physx_hand_peg_contact_count": np.asarray([0], dtype=np.int64),
            "physx_hand_peg_min_solid_separation": np.asarray(
                [0.0],
                dtype=np.float64,
            ),
            "physx_hand_peg_max_solid_penetration": np.asarray(
                [0.0],
                dtype=np.float64,
            ),
        }
        raw_count = np.asarray(count).reshape(-1)
        if raw_count.shape != (1,):
            raise RuntimeError(
                "Newton contact count must contain exactly one value"
            )
        contact_count = int(raw_count[0])
        shape0_array = np.asarray(shape0).reshape(-1)
        shape1_array = np.asarray(shape1).reshape(-1)
        capacity = min(shape0_array.size, shape1_array.size)
        if contact_count < 0 or contact_count > capacity:
            raise RuntimeError(
                "Newton contact count is outside its fixed capacity: "
                f"count={contact_count}, capacity={capacity}"
            )

        point0_array = np.asarray(point0)
        point1_array = np.asarray(point1)
        normal_array = np.asarray(normal)
        margin0_array = np.asarray(margin0).reshape(-1)
        margin1_array = np.asarray(margin1).reshape(-1)
        body_q_array = np.asarray(body_q)
        dynamic_capacities = {
            "point0": int(point0_array.shape[0]),
            "point1": int(point1_array.shape[0]),
            "normal": int(normal_array.shape[0]),
            "margin0": int(margin0_array.size),
            "margin1": int(margin1_array.size),
        }
        undersized = {
            name: value
            for name, value in dynamic_capacities.items()
            if value < contact_count
        }
        if undersized:
            raise RuntimeError(
                "Newton contact buffers disagree on capacity: "
                f"count={contact_count}, undersized={undersized}"
            )
        if contact_count and not all(
            np.all(np.isfinite(value[:contact_count]))
            for value in (
                point0_array,
                point1_array,
                normal_array,
                margin0_array,
                margin1_array,
            )
        ):
            raise RuntimeError(
                "Newton active contact evidence contains a non-finite value"
            )

        model = self._newton_contact_static_model
        metadata = self._newton_contact_static_metadata_cache
        if model is None or metadata is None:
            raise RuntimeError("Newton contact model metadata is unavailable")
        shape_body, shape_margin, labels = metadata
        if strict and not labels:
            raise RuntimeError("Newton finalized model has no shape labels")

        separations: list[float] = []
        for index in range(contact_count):
            first = int(shape0_array[index])
            second = int(shape1_array[index])
            if (
                first < 0
                or second < 0
                or first >= len(labels)
                or second >= len(labels)
                or first >= shape_body.size
                or second >= shape_body.size
                or first >= shape_margin.size
                or second >= shape_margin.size
            ):
                if strict:
                    raise RuntimeError(
                        "Newton contact references an invalid shape id: "
                        f"shape0={first}, shape1={second}, "
                        f"shape_count={len(labels)}"
                    )
                continue
            is_hand_peg = (
                self._is_hand_shape(labels[first])
                and self._is_b03_peg_shape(labels[second])
            ) or (
                self._is_hand_shape(labels[second])
                and self._is_b03_peg_shape(labels[first])
            )
            if not is_hand_peg:
                continue
            first_body = int(shape_body[first])
            second_body = int(shape_body[second])
            for body_index in (first_body, second_body):
                if body_index >= int(body_q_array.shape[0]):
                    raise RuntimeError(
                        "Newton contact references an invalid body id: "
                        f"body={body_index}, body_count={body_q_array.shape[0]}"
                    )
                if body_index >= 0 and not np.all(
                    np.isfinite(body_q_array[body_index])
                ):
                    raise RuntimeError(
                        "Newton contact body transform is non-finite: "
                        f"body={body_index}"
                    )
            first_world = (
                np.asarray(point0_array[index], dtype=np.float64)
                if first_body < 0
                else self._transform_point(
                    body_q_array[first_body],
                    point0_array[index],
                )
            )
            second_world = (
                np.asarray(point1_array[index], dtype=np.float64)
                if second_body < 0
                else self._transform_point(
                    body_q_array[second_body],
                    point1_array[index],
                )
            )
            direction = np.asarray(
                normal_array[index],
                dtype=np.float64,
            ).reshape(3)
            norm = float(np.linalg.norm(direction))
            if norm <= 1.0e-12:
                continue
            direction /= norm
            point_distance = float(
                np.dot(direction, second_world - first_world)
            )
            effective_radius = (
                float(margin0_array[index])
                - float(shape_margin[first])
                + float(margin1_array[index])
                - float(shape_margin[second])
            )
            separation = point_distance - effective_radius
            if not np.isfinite(separation):
                raise RuntimeError(
                    "Newton contact solid separation is non-finite"
                )
            separations.append(float(separation))

        result["physx_hand_peg_contact_valid"][0] = 1
        result["physx_hand_peg_contact_count"][0] = len(separations)
        if separations:
            minimum = float(min(separations))
            result["physx_hand_peg_min_solid_separation"][0] = minimum
            result["physx_hand_peg_max_solid_penetration"][0] = max(
                0.0,
                -minimum,
            )
        return result

    def _sample_contact_fields(self) -> dict[str, np.ndarray]:
        """Read Newton-native B03 hand/peg contact and solid penetration."""

        result = {
            "physx_hand_peg_contact_valid": np.asarray([0], dtype=np.uint8),
            "physx_hand_peg_contact_count": np.asarray([0], dtype=np.int64),
            "physx_hand_peg_min_solid_separation": np.asarray(
                [0.0],
                dtype=np.float64,
            ),
            "physx_hand_peg_max_solid_penetration": np.asarray(
                [0.0],
                dtype=np.float64,
            ),
        }
        if getattr(self.env, "task_name", None) != "t09_insert_peg":
            return result
        try:
            from isaaclab_newton.physics import NewtonManager

            contacts = NewtonManager.get_contacts()
            model = NewtonManager.get_model()
            state = NewtonManager.get_state_0()
            if contacts is None or model is None or state is None:
                return result
            count = self._warp_numpy(contacts.rigid_contact_count)
            shape0 = self._warp_numpy(contacts.rigid_contact_shape0)
            shape1 = self._warp_numpy(contacts.rigid_contact_shape1)
            point0 = self._warp_numpy(contacts.rigid_contact_point0)
            point1 = self._warp_numpy(contacts.rigid_contact_point1)
            normal = self._warp_numpy(contacts.rigid_contact_normal)
            margin0 = self._warp_numpy(contacts.rigid_contact_margin0)
            margin1 = self._warp_numpy(contacts.rigid_contact_margin1)
            self._contact_static_metadata(model)
            body_q = self._warp_numpy(state.body_q)
        except Exception:
            return result
        try:
            return self._contact_fields_from_arrays(
                count=count,
                shape0=shape0,
                shape1=shape1,
                point0=point0,
                point1=point1,
                normal=normal,
                margin0=margin0,
                margin1=margin1,
                body_q=body_q,
                strict=False,
            )
        except Exception:
            return result

    def _sample_deformable_fields(self) -> dict[str, np.ndarray]:
        return {}

    @classmethod
    def _is_published_field(cls, name: str) -> bool:
        return not name.startswith(cls._OMITTED_PREFIXES)

    @property
    def field_specs(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for name, raw_spec in super().field_specs.items():
            if not self._is_published_field(name):
                continue
            spec = dict(raw_spec)
            source = str(spec.get("source", ""))
            spec["source"] = source.replace("PhysX", "Newton")
            dtype = np.dtype(spec["dtype"])
            published_name = _newton_name(name)
            spec["physical_replay"] = not any(
                token in published_name
                for token in (
                    "_command_",
                    "_pose_commanded",
                    "_pose_filter_",
                )
            )
            if dtype.kind in "biu":
                replay_atol = 0.0
            elif "target" in name or "command" in name:
                replay_atol = 1.0e-7
            elif "joint_velocity" in name:
                replay_atol = 2.0e-3
            elif "joint_position" in name:
                replay_atol = 2.0e-4
            elif "effort" in name:
                replay_atol = 5.0e-2
            elif name.endswith("_state"):
                # The certifier compares rigid state semantically (position,
                # quaternion angle, and both velocity components).  This
                # scalar remains a conservative fallback for older readers.
                replay_atol = 5.0e-3
                spec["replay_metric"] = "rigid_state_v1"
                spec["replay_tolerance"] = {
                    "position_m": 2.0e-4,
                    "quaternion_angle_rad": float(np.deg2rad(0.2)),
                    "linear_velocity_m_s": 2.0e-3,
                    "angular_velocity_rad_s": 5.0e-3,
                }
            elif "velocity" in name:
                replay_atol = 2.0e-3
            else:
                replay_atol = 2.0e-4
            spec["replay_atol"] = replay_atol
            result[published_name] = spec
        contact_specs = {
            "newton_hand_peg_contact_valid": {
                "shape": [1],
                "dtype": "uint8",
                "layout": "1=Newton contact buffers were readable",
                "source": "NewtonManager.get_contacts",
            },
            "newton_hand_peg_contact_count": {
                "shape": [1],
                "dtype": "int64",
                "source": "Newton rigid contact shape pairs",
            },
            "newton_hand_peg_min_solid_separation": {
                "shape": [1],
                "dtype": "float64",
                "units": "m",
                "source": "Newton contact points minus primitive radii",
            },
            "newton_hand_peg_max_solid_penetration": {
                "shape": [1],
                "dtype": "float64",
                "units": "m",
                "source": "max(0, -minimum solid separation)",
                "maximum_allowed": B03_HAND_PEG_MAX_SOLID_PENETRATION_M,
            },
        }
        substep_count = max(1, int(getattr(self.env, "_n_substeps", 1)))
        for name, raw_spec in contact_specs.items():
            spec = dict(raw_spec)
            # Contact counts and ordering remain diagnostic.  Cross-GPU
            # certification enforces readable contacts, a real grasp, and the
            # penetration safety bound, but does not require an identical
            # number of redundant manifold points at every frame.
            spec["physical_replay"] = False
            spec["replay_atol"] = (
                0.0 if np.dtype(spec["dtype"]).kind in "biu" else 1.0e-6
            )
            result[name] = spec
            substep = dict(spec)
            substep["shape"] = [substep_count, *spec["shape"]]
            substep["source"] = "per_physics_substep:" + str(spec["source"])
            result[name.replace("newton_", "newton_substep_", 1)] = substep
        return result

    def _rename_snapshot(
        self,
        snapshot: Mapping[str, Any],
    ) -> dict[str, np.ndarray]:
        return {
            _newton_name(name): np.asarray(value).copy()
            for name, value in snapshot.items()
            if self._is_published_field(name)
        }

    @staticmethod
    def _require_finite_newton_array(name: str, value: Any) -> np.ndarray:
        result = np.asarray(value)
        if not np.all(np.isfinite(result)):
            raise RuntimeError(
                f"Newton GPU trace field {name!r} contains non-finite values"
            )
        return result

    def _decode_batched_robot_fields(
        self,
        raw: Mapping[str, np.ndarray],
        slot: int,
    ) -> dict[str, np.ndarray]:
        joint_shape = (len(ROBOT_SIDES), self.max_robot_joints)
        position = np.zeros(joint_shape, dtype=np.float64)
        velocity = np.zeros(joint_shape, dtype=np.float64)
        target = np.zeros(joint_shape, dtype=np.float64)
        effort = np.zeros(joint_shape, dtype=np.float64)
        valid = np.zeros(joint_shape, dtype=np.uint8)
        target_valid = np.zeros(joint_shape, dtype=np.uint8)
        effort_valid = np.zeros(joint_shape, dtype=np.uint8)
        tcp_state = np.zeros((len(ROBOT_SIDES), 13), dtype=np.float64)
        tcp_valid = np.zeros((len(ROBOT_SIDES),), dtype=np.uint8)
        root_state = np.zeros((len(ROBOT_SIDES), 13), dtype=np.float64)
        root_valid = np.zeros((len(ROBOT_SIDES),), dtype=np.uint8)

        for side_index, entry in enumerate(self.robots):
            side = entry.side
            count = int(entry.joint_count)
            if count <= 0 or count > self.max_robot_joints:
                raise RuntimeError(
                    f"Newton GPU trace has invalid {side} joint count {count}"
                )
            decoded: dict[str, np.ndarray] = {}
            for label in (
                "joint_position",
                "joint_velocity",
                "joint_target",
                "joint_effort",
            ):
                key = f"robot/{side}/{label}"
                if key not in raw:
                    raise RuntimeError(
                        f"Newton GPU trace is missing required source {key}"
                    )
                values = np.asarray(raw[key][slot]).reshape(-1)
                if values.size < count:
                    raise RuntimeError(
                        "Newton GPU trace robot buffer is undersized: "
                        f"name={key}, size={values.size}, expected={count}"
                    )
                decoded[label] = self._require_finite_newton_array(
                    key,
                    values[:count],
                ).astype(np.float64, copy=False)
            position[side_index, :count] = decoded["joint_position"]
            velocity[side_index, :count] = decoded["joint_velocity"]
            target[side_index, :count] = decoded["joint_target"]
            effort[side_index, :count] = decoded["joint_effort"]
            valid[side_index, :count] = 1
            target_valid[side_index, :count] = 1
            effort_valid[side_index, :count] = 1

            body_key = f"robot/{side}/body_state"
            body_states = np.asarray(raw[body_key][slot]).reshape(-1, 13)
            tcp_index = int(entry.tcp_body_index)
            if tcp_index < 0 or tcp_index >= body_states.shape[0]:
                raise RuntimeError(
                    "Newton GPU trace TCP body index is outside body_state: "
                    f"side={side}, index={tcp_index}, "
                    f"body_count={body_states.shape[0]}"
                )
            tcp = self._require_finite_newton_array(
                body_key,
                body_states[tcp_index],
            )
            tcp_state[side_index] = tcp
            tcp_valid[side_index] = 1

            root_key = f"robot/{side}/root_state"
            roots = np.asarray(raw[root_key][slot]).reshape(-1, 13)
            if roots.shape[0] < 1:
                raise RuntimeError(
                    f"Newton GPU trace root buffer {root_key} is empty"
                )
            root = self._require_finite_newton_array(root_key, roots[0])
            root_state[side_index] = root
            root_valid[side_index] = 1

        return {
            "physx_robot_joint_position": position,
            "physx_robot_joint_velocity": velocity,
            "physx_robot_joint_target": target,
            "physx_robot_joint_target_valid": target_valid,
            "physx_robot_joint_valid": valid,
            "physx_robot_root_state": root_state,
            "physx_robot_root_valid": root_valid,
            "physx_robot_joint_effort": effort,
            "physx_robot_joint_effort_valid": effort_valid,
            "physx_tcp_state": tcp_state,
            "physx_tcp_valid": tcp_valid,
        }

    def _decode_batched_rigid_fields(
        self,
        raw: Mapping[str, np.ndarray],
        slot: int,
    ) -> dict[str, np.ndarray]:
        if not self.rigid_bodies:
            return {}
        states = np.zeros((len(self.rigid_bodies), 13), dtype=np.float64)
        valid = np.zeros((len(self.rigid_bodies),), dtype=np.uint8)
        body_is_active = getattr(
            self.env,
            "_physx_task_rigid_body_is_active",
            None,
        )
        for index, entry in enumerate(self.rigid_bodies):
            if callable(body_is_active) and not bool(body_is_active(entry.name)):
                continue
            prefix = f"rigid/{index}"
            if entry.source == "physx_tensor_view":
                pose = np.asarray(raw[f"{prefix}/transform"][slot]).reshape(
                    -1,
                    7,
                )[0]
                velocity_key = f"{prefix}/velocity"
                speed = (
                    np.zeros(6, dtype=np.float64)
                    if velocity_key not in raw
                    else np.asarray(raw[velocity_key][slot]).reshape(-1, 6)[0]
                )
                pose = self._require_finite_newton_array(
                    f"{prefix}/transform",
                    pose,
                )
                speed = self._require_finite_newton_array(
                    velocity_key,
                    speed,
                )
                state = np.concatenate(
                    [pose[:3], pose[6:7], pose[3:6], speed]
                )
            else:
                state = np.asarray(raw[f"{prefix}/root_state"][slot]).reshape(
                    -1,
                    13,
                )[0]
                state = self._require_finite_newton_array(
                    f"{prefix}/root_state",
                    state,
                )
            states[index] = np.asarray(state, dtype=np.float64)
            valid[index] = 1
        return {
            "physx_task_rigid_body_state": states,
            "physx_task_rigid_body_valid": valid,
        }

    def _decode_batched_articulation_fields(
        self,
        raw: Mapping[str, np.ndarray],
        slot: int,
    ) -> dict[str, np.ndarray]:
        if not self.articulations:
            return {}
        count = len(self.articulations)
        roots = np.zeros((count, 13), dtype=np.float64)
        root_valid = np.zeros((count,), dtype=np.uint8)
        has_dofs = self.max_articulation_dofs > 0
        if has_dofs:
            dof_shape = (count, self.max_articulation_dofs)
            dof_position = np.zeros(dof_shape, dtype=np.float64)
            dof_velocity = np.zeros(dof_shape, dtype=np.float64)
            dof_valid = np.zeros(dof_shape, dtype=np.uint8)

        for index, entry in enumerate(self.articulations):
            prefix = f"articulation/{index}"
            pose = np.asarray(raw[f"{prefix}/root_transform"][slot]).reshape(
                -1,
                7,
            )[0]
            speed = np.asarray(raw[f"{prefix}/root_velocity"][slot]).reshape(
                -1,
                6,
            )[0]
            pose = self._require_finite_newton_array(
                f"{prefix}/root_transform",
                pose,
            )
            speed = self._require_finite_newton_array(
                f"{prefix}/root_velocity",
                speed,
            )
            roots[index] = np.concatenate(
                [pose[:3], pose[6:7], pose[3:6], speed]
            )
            root_valid[index] = 1
            if not has_dofs or entry.dof_count <= 0:
                continue
            position = np.asarray(
                raw[f"{prefix}/dof_position"][slot]
            ).reshape(-1)
            velocity = np.asarray(
                raw[f"{prefix}/dof_velocity"][slot]
            ).reshape(-1)
            dof_count = int(entry.dof_count)
            if position.size < dof_count or velocity.size < dof_count:
                raise RuntimeError(
                    f"Newton articulation {entry.name} DOF buffer is undersized"
                )
            position = self._require_finite_newton_array(
                f"{prefix}/dof_position",
                position[:dof_count],
            )
            velocity = self._require_finite_newton_array(
                f"{prefix}/dof_velocity",
                velocity[:dof_count],
            )
            dof_position[index, :dof_count] = position
            dof_velocity[index, :dof_count] = velocity
            dof_valid[index, :dof_count] = 1
        result = {
            "physx_task_articulation_root_state": roots,
            "physx_task_articulation_root_valid": root_valid,
        }
        if has_dofs:
            result.update(
                {
                    "physx_task_articulation_dof_position": dof_position,
                    "physx_task_articulation_dof_velocity": dof_velocity,
                    "physx_task_articulation_dof_valid": dof_valid,
                }
            )
        return result

    def _decode_batched_contact_fields(
        self,
        raw: Mapping[str, np.ndarray],
        slot: int,
    ) -> dict[str, np.ndarray]:
        return self._contact_fields_from_arrays(
            count=raw["contact/count"][slot],
            shape0=raw["contact/shape0"][slot],
            shape1=raw["contact/shape1"][slot],
            point0=raw["contact/point0"][slot],
            point1=raw["contact/point1"][slot],
            normal=raw["contact/normal"][slot],
            margin0=raw["contact/margin0"][slot],
            margin1=raw["contact/margin1"][slot],
            body_q=raw["contact/body_q"][slot],
            strict=True,
        )

    def _install_batched_contact_static_metadata(
        self,
        raw: Mapping[str, np.ndarray],
    ) -> None:
        model = self._newton_contact_static_model
        if model is None:
            raise RuntimeError("Newton GPU trace has no finalized contact model")
        shape_body = np.asarray(
            raw["contact/model_shape_body"][0]
        ).reshape(-1).copy()
        shape_margin = np.asarray(
            raw["contact/model_shape_margin"][0]
        ).reshape(-1).copy()
        labels = tuple(
            str(label) for label in getattr(model, "shape_label", ())
        )
        if (
            not labels
            or shape_body.shape != shape_margin.shape
            or shape_body.size != len(labels)
            or not np.all(np.isfinite(shape_margin))
        ):
            raise RuntimeError(
                "Newton finalized shape metadata is incomplete: "
                f"bodies={shape_body.shape}, margins={shape_margin.shape}, "
                f"labels={len(labels)}"
            )
        shape_body.setflags(write=False)
        shape_margin.setflags(write=False)
        cached = self._newton_contact_static_metadata_cache
        if cached is not None and (
            cached[2] != labels
            or not np.array_equal(cached[0], shape_body)
            or not np.array_equal(cached[1], shape_margin)
        ):
            raise RuntimeError(
                "Newton finalized shape metadata changed inside an episode"
            )
        self._newton_contact_static_metadata_cache = (
            shape_body,
            shape_margin,
            labels,
        )

    def _decode_batched_substep(
        self,
        raw: Mapping[str, np.ndarray],
        slot: int,
        cpu_sample: Mapping[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        sample: dict[str, np.ndarray] = {}
        sample.update(self._decode_batched_robot_fields(raw, slot))
        sample.update(self._decode_batched_rigid_fields(raw, slot))
        sample.update(self._decode_batched_articulation_fields(raw, slot))
        sample.update(self._decode_batched_contact_fields(raw, slot))
        sample.update(
            {
                name: np.asarray(value).copy()
                for name, value in cpu_sample.items()
            }
        )
        return sample

    def _publish_latest_batched_state(
        self,
        endpoint: Mapping[str, np.ndarray],
    ) -> None:
        rigid_states = np.asarray(
            endpoint.get(
                "physx_task_rigid_body_state",
                np.zeros((0, 13), dtype=np.float64),
            )
        )
        rigid_valid = np.asarray(
            endpoint.get(
                "physx_task_rigid_body_valid",
                np.zeros((0,), dtype=np.uint8),
            )
        ).astype(bool)
        self._newton_latest_rigid_states = {
            entry.name: np.asarray(rigid_states[index]).copy()
            for index, entry in enumerate(self.rigid_bodies)
            if index < rigid_valid.size and rigid_valid[index]
        }

        position = np.asarray(endpoint["physx_robot_joint_position"])
        velocity = np.asarray(endpoint["physx_robot_joint_velocity"])
        target = np.asarray(endpoint["physx_robot_joint_target"])
        valid = np.asarray(endpoint["physx_robot_joint_valid"]).astype(bool)
        self._newton_latest_robot_states = {
            entry.side: {
                "position": position[index].copy(),
                "velocity": velocity[index].copy(),
                "target": target[index].copy(),
                "valid": valid[index].copy(),
                "tcp_state": np.asarray(
                    endpoint["physx_tcp_state"][index]
                ).copy(),
            }
            for index, entry in enumerate(self.robots)
        }
        self._newton_latest_state_env_step = self._newton_frame_env_step
        self._newton_latest_state_write_count = int(
            self._newton_last_capture_state_write_count
        )
        self._newton_latest_origin_id = self._newton_frame_origin_id

    def _latest_batched_state_is_fresh(self) -> bool:
        current_step = int(getattr(self.env, "env_step", 0))
        return bool(
            self._newton_latest_state_env_step >= 0
            and current_step
            in {
                self._newton_latest_state_env_step,
                self._newton_latest_state_env_step + 1,
            }
            and int(getattr(self.env, "_trajectory_state_write_count", 0))
            == self._newton_latest_state_write_count
            and id(getattr(self.env, "_physical_recording_origin", None))
            == self._newton_latest_origin_id
        )

    def latest_task_rigid_body_state(
        self,
        name: str,
    ) -> np.ndarray | None:
        """Return the owned final-substep B03 state without a new D2H copy."""

        if not self._latest_batched_state_is_fresh():
            return None
        direct = self._newton_latest_rigid_states.get(str(name))
        if direct is not None:
            return np.asarray(direct).copy()
        view = getattr(self.env, "_task_body_views", {}).get(name)
        if view is None:
            return None
        for entry in self.rigid_bodies:
            if entry.handle is view:
                state = self._newton_latest_rigid_states.get(entry.name)
                return None if state is None else np.asarray(state).copy()
        return None

    def latest_robot_joint_state(
        self,
        side: str,
    ) -> dict[str, np.ndarray] | None:
        """Return an owned prior/final-frame joint diagnostic on the CPU."""

        if not self._latest_batched_state_is_fresh():
            return None
        state = self._newton_latest_robot_states.get(str(side))
        if state is None:
            return None
        return {name: np.asarray(value).copy() for name, value in state.items()}

    def invalidate_latest_batched_state(self) -> None:
        """Drop action-frame CPU caches before an out-of-band reset write."""

        self._newton_latest_rigid_states = {}
        self._newton_latest_robot_states = {}
        self._newton_latest_state_env_step = -1
        self._newton_latest_state_write_count = -1
        self._newton_latest_origin_id = -1
        self._newton_frame_active = False
        self._newton_frame_materialized = False
        self._newton_cpu_substeps = []
        self._substep_snapshots = []
        self._clear_newton_endpoint_cache()

    def _materialize_newton_gpu_frame(self) -> None:
        if self._newton_frame_materialized:
            return
        batch = self._newton_gpu_batch
        if batch is None:
            raise RuntimeError("B03 Newton GPU trace ring was not initialized")
        expected = max(1, int(getattr(self.env, "_n_substeps", 1)))
        if len(self._newton_cpu_substeps) != expected:
            raise RuntimeError(
                "Newton GPU trace CPU sidecar is incomplete: "
                f"captured={len(self._newton_cpu_substeps)}, expected={expected}"
            )
        current_writes = int(
            getattr(self.env, "_trajectory_state_write_count", 0)
        )
        if current_writes != self._newton_last_capture_state_write_count:
            raise RuntimeError(
                "Newton state was written after the final captured substep; "
                "refusing a stale physical endpoint"
            )
        stream = self._newton_current_stream()
        raw = batch.materialize(stream)
        self._install_batched_contact_static_metadata(raw)
        complete_samples = [
            self._decode_batched_substep(raw, slot, cpu_sample)
            for slot, cpu_sample in enumerate(self._newton_cpu_substeps)
        ]
        if len(complete_samples) != expected:
            raise RuntimeError("Newton GPU trace decoded an incomplete frame")
        self._substep_snapshots = [
            {
                name: np.asarray(sample[name]).copy()
                for name in self._SUBSTEP_CAPTURE_FIELDS
                if name in sample
            }
            for sample in complete_samples
        ]
        endpoint = {
            name: np.asarray(value).copy()
            for name, value in complete_samples[-1].items()
        }
        self._newton_endpoint_cache = endpoint
        self._newton_endpoint_cache_substep_count = expected
        self._newton_endpoint_cache_env_step = self._newton_frame_env_step
        self._newton_endpoint_cache_state_write_count = current_writes
        self._publish_latest_batched_state(endpoint)
        self._newton_frame_materialized = True

    def finish_substep_capture(self) -> None:
        """Complete B03's one frame readback before CPU task consumers run."""

        if (
            getattr(self, "_newton_frame_active", False)
            and self._newton_gpu_batch is not None
        ):
            self._materialize_newton_gpu_frame()

    def begin_substep_capture(self) -> None:
        """Start one Newton substep trace and invalidate any older endpoint."""

        if (
            getattr(self, "_newton_gpu_batch_requested", False)
            and getattr(self, "_newton_frame_active", False)
        ):
            raise RuntimeError(
                "Newton GPU trace began a new frame before consuming the prior one"
            )
        super().begin_substep_capture()
        self._clear_newton_endpoint_cache()
        if getattr(self, "_newton_gpu_batch_requested", False):
            if getattr(self, "_newton_gpu_batch", None) is None:
                self._initialize_newton_gpu_batch_capture()
            batch = self._newton_gpu_batch
            assert batch is not None
            self._newton_cpu_substeps = []
            self._newton_frame_active = True
            self._newton_frame_materialized = False
            self._newton_frame_env_step = int(getattr(self.env, "env_step", 0))
            self._newton_frame_origin_id = id(
                getattr(self.env, "_physical_recording_origin", None)
            )
            self._newton_last_capture_state_write_count = int(
                getattr(self.env, "_trajectory_state_write_count", 0)
            )
            batch.begin(self._newton_current_stream())

    def capture_substep(self) -> None:
        """Capture one Newton substep without unused robot-root readbacks."""

        max_count = max(1, int(getattr(self.env, "_n_substeps", 1)))
        if getattr(self, "_newton_gpu_batch", None) is not None:
            if not getattr(self, "_newton_frame_active", False):
                raise RuntimeError(
                    "Newton GPU trace capture_substep called without begin"
                )
            if len(self._newton_cpu_substeps) >= max_count:
                raise RuntimeError(
                    "Newton executed more substeps than the recording schema "
                    f"declares ({max_count}); refusing to truncate the "
                    "physical replay trajectory"
                )
            sources = self._newton_cuda_sources()
            stream = self._newton_current_stream()
            slot = self._newton_gpu_batch.stage(sources, stream)
            if slot != len(self._newton_cpu_substeps):
                raise RuntimeError(
                    "Newton GPU trace slot/cursor mismatch: "
                    f"slot={slot}, cpu={len(self._newton_cpu_substeps)}"
                )
            cpu_sample = {
                name: np.asarray(value).copy()
                for name, value in self._sample_command_fields().items()
            }
            cpu_sample["physx_sim_time"] = np.asarray(
                [self._sim_time()],
                dtype=np.float64,
            )
            self._newton_cpu_substeps.append(cpu_sample)
            self._newton_last_capture_state_write_count = int(
                getattr(self.env, "_trajectory_state_write_count", 0)
            )
            return
        if len(self._substep_snapshots) >= max_count:
            raise RuntimeError(
                "Newton executed more substeps than the recording schema "
                f"declares ({max_count}); refusing to truncate the physical "
                "replay trajectory"
            )
        sample: dict[str, np.ndarray] = {}
        sample.update(self._sample_newton_substep_robot_fields())
        sample.update(self._sample_command_fields())
        sample.update(self._sample_rigid_body_fields())
        sample.update(self._sample_articulation_fields())
        sample.update(self._sample_contact_fields())
        sample["physx_sim_time"] = np.asarray(
            [self._sim_time()],
            dtype=np.float64,
        )

        owned_sample = {
            name: np.asarray(value).copy()
            for name, value in sample.items()
            if self._is_published_field(name)
        }
        self._substep_snapshots.append(
            {
                name: np.asarray(owned_sample[name]).copy()
                for name in self._SUBSTEP_CAPTURE_FIELDS
                if name in owned_sample
            }
        )
        # Keep a separate owned copy.  ``substep_snapshot(consume=True)`` may
        # clear its list, and returned observations must never alias storage
        # that a later control step can overwrite.
        self._newton_endpoint_cache = {
            name: np.asarray(value).copy()
            for name, value in owned_sample.items()
        }
        self._newton_endpoint_cache_substep_count = len(
            self._substep_snapshots
        )
        self._newton_endpoint_cache_env_step = int(
            getattr(self.env, "env_step", 0)
        )
        self._newton_endpoint_cache_state_write_count = int(
            getattr(self.env, "_trajectory_state_write_count", 0)
        )

    def _sample_newton_endpoint_mutable_robot_fields(
        self,
    ) -> dict[str, np.ndarray]:
        """Read endpoint-only roots and callback-mutable drive targets."""

        shape = (len(ROBOT_SIDES), self.max_robot_joints)
        target = np.zeros(shape, dtype=np.float64)
        target_valid = np.zeros(shape, dtype=np.uint8)
        root_state = np.zeros((len(ROBOT_SIDES), 13), dtype=np.float64)
        root_valid = np.zeros((len(ROBOT_SIDES),), dtype=np.uint8)
        joint_targets = getattr(self.env, "_joint_targets", {})

        for side_index, entry in enumerate(self.robots):
            robot = entry.robot
            if robot is None:
                continue
            for source_name in ("root_state_w", "root_link_state_w"):
                try:
                    root = _first_row(
                        getattr(robot.data, source_name)
                    ).reshape(-1)[:13]
                except Exception:
                    continue
                if root.shape == (13,) and np.all(np.isfinite(root)):
                    root_state[side_index] = root
                    root_valid[side_index] = 1
                    break
            if isinstance(joint_targets, Mapping) and entry.side in joint_targets:
                try:
                    values = _first_row(joint_targets[entry.side]).reshape(-1)
                    count = min(values.size, self.max_robot_joints)
                    finite = np.isfinite(values[:count])
                    target[side_index, :count][finite] = values[:count][finite]
                    target_valid[side_index, :count] = finite.astype(np.uint8)
                except Exception:
                    pass
        return {
            "physx_robot_joint_target": target,
            "physx_robot_joint_target_valid": target_valid,
            "physx_robot_root_state": root_state,
            "physx_robot_root_valid": root_valid,
        }

    def _cached_endpoint_snapshot(self) -> dict[str, np.ndarray] | None:
        """Return a complete endpoint using the final physical substep."""

        cache = getattr(self, "_newton_endpoint_cache", None)
        expected_substeps = max(1, int(getattr(self.env, "_n_substeps", 1)))
        if (
            not isinstance(cache, dict)
            or self._newton_endpoint_cache_substep_count != expected_substeps
            or len(self._substep_snapshots) != expected_substeps
            or int(getattr(self.env, "env_step", 0))
            != self._newton_endpoint_cache_env_step + 1
            or int(getattr(self.env, "_trajectory_state_write_count", 0))
            != self._newton_endpoint_cache_state_write_count
        ):
            return None

        result = {
            name: np.asarray(value).copy()
            for name, value in cache.items()
        }
        if getattr(self, "_newton_gpu_batch", None) is None:
            result.update(self._sample_newton_endpoint_mutable_robot_fields())
        # Commands and filters may be changed by the post-substep callback;
        # refresh them without touching simulator state tensors.
        result.update(self._sample_command_fields())
        if self.has_rng_state:
            result.update(self._sample_rng_fields())
        result["physx_sim_time"] = np.asarray(
            [self._sim_time()],
            dtype=np.float64,
        )
        result["physx_control_step"] = np.asarray(
            [float(getattr(self.env, "env_step", 0))],
            dtype=np.float64,
        )

        required = set(self._ENDPOINT_ALWAYS_FIELDS)
        if self.has_rng_state:
            required.update(
                {
                    "physx_rng_seed",
                    "physx_rng_mt19937_keys",
                    "physx_rng_mt19937_position",
                    "physx_rng_mt19937_has_gauss",
                    "physx_rng_mt19937_cached_gaussian",
                }
            )
        if self.rigid_bodies:
            required.update(
                {
                    "physx_task_rigid_body_state",
                    "physx_task_rigid_body_valid",
                }
            )
        if self.articulations:
            required.update(
                {
                    "physx_task_articulation_root_state",
                    "physx_task_articulation_root_valid",
                }
            )
        if self.max_articulation_dofs > 0:
            required.update(
                {
                    "physx_task_articulation_dof_position",
                    "physx_task_articulation_dof_velocity",
                    "physx_task_articulation_dof_valid",
                }
            )
        if not required.issubset(result):
            return None
        return result

    def snapshot(self) -> dict[str, np.ndarray]:
        batch_active = bool(
            getattr(self, "_newton_gpu_batch", None) is not None
            and getattr(self, "_newton_frame_active", False)
        )
        if batch_active:
            self._materialize_newton_gpu_frame()
        cached = self._cached_endpoint_snapshot()
        if batch_active and cached is None:
            raise RuntimeError(
                "B03 Newton GPU trace could not publish its owned endpoint"
            )
        return self._rename_snapshot(
            super().snapshot() if cached is None else cached
        )

    def substep_snapshot(
        self,
        *,
        consume: bool = False,
    ) -> dict[str, np.ndarray]:
        """Return Newton-named, fixed-shape samples for the last control step."""

        if (
            getattr(self, "_newton_gpu_batch", None) is not None
            and getattr(self, "_newton_frame_active", False)
        ):
            self._materialize_newton_gpu_frame()
        snapshots = self._substep_snapshots
        result: dict[str, np.ndarray] = {}
        specs = self.field_specs
        for source_name in self._SUBSTEP_CAPTURE_FIELDS:
            if not self._is_published_field(source_name):
                continue
            newton_source_name = _newton_name(source_name)
            substep_name = newton_source_name.replace(
                "newton_",
                "newton_substep_",
                1,
            )
            spec = specs.get(substep_name)
            if spec is None:
                continue
            value = np.zeros(
                tuple(int(size) for size in spec["shape"]),
                dtype=np.dtype(spec["dtype"]),
            )
            for index, snapshot in enumerate(snapshots[: value.shape[0]]):
                if source_name in snapshot:
                    value[index] = snapshot[source_name]
            result[substep_name] = value
        result["newton_substep_count"] = np.asarray(
            [len(snapshots)],
            dtype=np.int64,
        )
        if consume:
            self._substep_snapshots = []
            self._clear_newton_endpoint_cache()
            self._newton_frame_active = False
        return result

    @property
    def gpu_batch_telemetry(self) -> dict[str, int | bool]:
        batch = self._newton_gpu_batch
        return {
            "enabled": batch is not None,
            "frame_materialized": bool(
                self._newton_frame_materialized
                and self._latest_batched_state_is_fresh()
            ),
            "frame_sync_count": (
                0 if batch is None else batch.frame_sync_count
            ),
            "d2h_copy_count": (
                0 if batch is None else batch.last_d2h_copy_count
            ),
        }

    def _newton_physics_config(self) -> dict[str, Any]:
        determinism = dict(
            getattr(self.env, "_newton_determinism_contract", {})
        )
        solver_settings = dict(
            getattr(self.env, "_newton_solver_settings", {})
        )
        njmax = solver_settings.get("njmax")
        nconmax = solver_settings.get("nconmax")
        hand_target_slew_rate = float(
            getattr(self.env, "_newton_hand_target_slew_rate_rad_s", 0.0)
        )
        hand_target_slew_enabled = bool(
            getattr(self.env, "_newton_hand_target_slew_enabled", False)
        )
        physics_dt = float(getattr(self.env, "_physics_dt", 0.0))
        return {
            "backend": "nvidia_newton_mujoco_warp",
            "device": str(getattr(self.env, "_isaac_device", "")),
            "physics_dt": physics_dt,
            "live_physics_dt": float(
                getattr(
                    self.env,
                    "_sim_dt",
                    getattr(self.env, "_physics_dt", 0.0),
                )
            ),
            "control_dt": float(getattr(self.env, "_control_dt", 0.0)),
            "n_substeps": int(getattr(self.env, "_n_substeps", 0)),
            "solver": {
                "solver": str(solver_settings.get("solver", "newton")),
                "integrator": str(
                    solver_settings.get("integrator", "implicitfast")
                ),
                "cone": str(solver_settings.get("cone", "pyramidal")),
                "iterations": int(solver_settings.get("iterations", 100)),
                "line_search_iterations": int(
                    solver_settings.get("ls_iterations", 50)
                ),
                "ccd_iterations": int(
                    solver_settings.get("ccd_iterations", 50)
                ),
                "tolerance": float(
                    solver_settings.get("tolerance", 1.0e-8)
                ),
                "njmax": njmax,
                "nconmax": nconmax,
                "capacity_policy": (
                    "fixed_explicit"
                    if njmax is not None and nconmax is not None
                    else "newton_1_4_finalized_model_estimate"
                ),
                "mujoco_read_only_sensor_pipeline": "disabled",
                "use_mujoco_cpu": bool(
                    solver_settings.get("use_mujoco_cpu", False)
                ),
                "use_mujoco_contacts": bool(
                    solver_settings.get("use_mujoco_contacts", False)
                ),
                "use_newton_actuators": True,
                "deterministic_mode": determinism.get(
                    "solver_deterministic_mode"
                ),
            },
            "execution": newton_execution_config(self.env),
            "hand_target_shaping": {
                "algorithm": "per_physics_substep_slew_v1",
                "enabled": hand_target_slew_enabled,
                "scope": "b03_articulated_newton",
                "rate_rad_s": hand_target_slew_rate,
                "maximum_delta_per_physics_substep_rad": (
                    hand_target_slew_rate * physics_dt
                ),
                "reference": "previous_applied_drive_target",
                "raw_hand_command_mutated": False,
                "recording": "actual_applied_per_substep_target",
                "replay": "exact_recorded_pre_substep_targets_bypass_slew",
            },
            "joint_velocity_projection": dict(
                getattr(
                    self.env,
                    "_newton_velocity_projection_contract",
                    {},
                )
            ),
            "asset_identity": {
                side: dict(identity)
                for side, identity in dict(
                    getattr(self.env, "_newton_b03_asset_identity", {})
                ).items()
            },
            "contact_model": dict(
                getattr(self.env, "_newton_b03_contact_contract", {})
            ),
            "determinism": determinism,
        }

    def schema(self) -> dict[str, Any]:
        physics_config = self._newton_physics_config()
        runtime_identity = capture_newton_runtime_identity(
            str(getattr(self.env, "_isaac_device", "cuda:0")),
            determinism_contract=getattr(
                self.env,
                "_newton_determinism_contract",
                {},
            ),
        )
        runtime_identity["b03_generated_robot_assets"] = {
            side: dict(identity)
            for side, identity in dict(
                getattr(self.env, "_newton_b03_asset_identity", {})
            ).items()
        }
        return {
            "name": "handson.b03_newton_ground_truth",
            "version": 3,
            "description": (
                "B03 action-aligned NVIDIA Newton readback. Current fields "
                "describe state before action[t], next_* fields describe state "
                "after action[t], and newton_substep_* stores every live "
                "physics substep and its exact robot position-drive targets."
            ),
            "fields": self.field_specs,
            "newton": {
                "bitwise_checkpoint": False,
                "requires_fresh_process_certification": True,
                "requires_distinct_gpu_model_certification": True,
                "cross_gpu_replay_profile": "cross_gpu_v1",
                "cross_gpu_certification_runs_required": 3,
                "allows_recorded_t0_state_restore": False,
                "certification_forbids_state_writes": True,
                "certification_forbids_state_writes_after_action_start": True,
                "replay_input": (
                    "exact_recorded_per_newton_substep_robot_position_targets"
                ),
                "capture_mode": "read_only_live_isaaclab_tensor_snapshot",
                "contact_capture": (
                    "Newton-native B03 hand/peg pair count and solid-surface "
                    "penetration; no PhysX contact-report provenance"
                ),
                "hand_peg_max_solid_penetration_m": (
                    B03_HAND_PEG_MAX_SOLID_PENETRATION_M
                ),
                "source_validity_gate": (
                    b03_newton_source_validity_gate_contract()
                ),
                "state_layout": list(PHYSX_STATE_LAYOUT),
                "robot_sides": [entry.side for entry in self.robots],
                "robot_joint_names": {
                    entry.side: list(entry.joint_names) for entry in self.robots
                },
                "rigid_body_names": [entry.name for entry in self.rigid_bodies],
                "rigid_body_sources": [
                    entry.source.replace("physx", "newton")
                    for entry in self.rigid_bodies
                ],
                "articulation_names": [
                    entry.name for entry in self.articulations
                ],
                "articulation_dof_names": {
                    entry.name: list(entry.dof_names)
                    for entry in self.articulations
                },
                "physics_config": physics_config,
                "physics_dt": physics_config["physics_dt"],
                "control_dt": physics_config["control_dt"],
                "n_substeps": physics_config["n_substeps"],
                "determinism": dict(physics_config["determinism"]),
                "source_runtime_identity": runtime_identity,
            },
        }

    @staticmethod
    def _as_physx_schema(recorded_schema: Mapping[str, Any]) -> dict[str, Any]:
        converted = dict(recorded_schema)
        environment_schema = converted.get("environment_schema")
        if isinstance(environment_schema, Mapping):
            environment_schema = dict(environment_schema)
            newton = environment_schema.get("newton")
            if not isinstance(newton, Mapping):
                raise ValueError(
                    "Recording schema does not contain environment_schema.newton"
                )
            environment_schema["physx"] = dict(newton)
            converted["environment_schema"] = environment_schema
            return converted
        newton = converted.get("newton")
        if not isinstance(newton, Mapping):
            raise ValueError("Recording schema does not contain newton metadata")
        converted["physx"] = dict(newton)
        return converted

    def apply_robot_joint_targets(
        self,
        snapshot: Mapping[str, Any],
        recorded_schema: Mapping[str, Any],
    ) -> int:
        """Apply recorded Newton drive targets without teleporting live state."""

        converted_snapshot = {
            _physx_name(name): value for name, value in snapshot.items()
        }
        return super().apply_robot_joint_targets(
            converted_snapshot,
            self._as_physx_schema(recorded_schema),
        )
