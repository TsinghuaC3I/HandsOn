from __future__ import annotations

import sys
import types

import numpy as np

from handson.sim.isaaclab.newton_ground_truth import (
    NewtonGroundTruthTrace,
    _NewtonGpuFrameRing,
)


class _CountingBuffer:
    def __init__(self, value) -> None:
        self.value = np.asarray(value)
        self.shape = self.value.shape
        self.numpy_calls = 0

    def numpy(self) -> np.ndarray:
        self.numpy_calls += 1
        return self.value.copy()


class _FakeCudaDevice:
    is_cuda = True

    def __str__(self) -> str:
        return "cuda:0"


class _FakeCpuDevice:
    is_cuda = False

    def __str__(self) -> str:
        return "cpu"


class _FakeStream:
    def __init__(self, handle: int = 17) -> None:
        self.device = _FakeCudaDevice()
        self.cuda_stream = handle


class _FakeWarpArray:
    def __init__(self, value, *, device, dtype=None) -> None:
        self.value = np.asarray(value, dtype=dtype)
        self.shape = self.value.shape
        self.size = self.value.size
        self.dtype = self.value.dtype
        self.device = device
        self.is_contiguous = True

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        value = self.value if dtype is None else self.value.astype(dtype)
        return value.copy() if copy else value


class _FakeWarp:
    int32 = np.dtype(np.int32)

    def __init__(self) -> None:
        self.operations: list[str] = []
        self.sync_count = 0

    @staticmethod
    def empty(*, shape, dtype, device, pinned=False):
        del pinned
        resolved_device = (
            _FakeCpuDevice() if str(device) == "cpu" else device
        )
        return _FakeWarpArray(
            np.zeros(shape, dtype=np.dtype(dtype)),
            device=resolved_device,
        )

    @staticmethod
    def array(value, *, dtype, device):
        return _FakeWarpArray(
            value,
            dtype=np.dtype(dtype),
            device=device,
        )

    def copy(
        self,
        destination,
        source,
        *,
        dest_offset=0,
        src_offset=0,
        count=0,
        stream=None,
    ) -> None:
        assert stream is not None
        count = int(count) if count else int(source.size)
        destination.value.reshape(-1)[dest_offset : dest_offset + count] = (
            source.value.reshape(-1)[src_offset : src_offset + count]
        )
        if destination.device.is_cuda and source.device.is_cuda:
            self.operations.append("d2d")
        elif not destination.device.is_cuda and source.device.is_cuda:
            self.operations.append("d2h")
        else:
            self.operations.append("other")

    def synchronize_stream(self, stream) -> None:
        assert stream is not None
        self.operations.append("sync")
        self.sync_count += 1


def _fake_cuda_array(value) -> _FakeWarpArray:
    return _FakeWarpArray(value, device=_FakeCudaDevice())


def test_newton_contact_static_model_metadata_is_read_once(monkeypatch) -> None:
    contacts = types.SimpleNamespace(
        rigid_contact_count=_CountingBuffer([1]),
        rigid_contact_shape0=_CountingBuffer([0]),
        rigid_contact_shape1=_CountingBuffer([1]),
        rigid_contact_point0=_CountingBuffer([[0.0, 0.0, 0.0]]),
        rigid_contact_point1=_CountingBuffer([[-0.0004, 0.0, 0.0]]),
        rigid_contact_normal=_CountingBuffer([[1.0, 0.0, 0.0]]),
        rigid_contact_margin0=_CountingBuffer([0.003]),
        rigid_contact_margin1=_CountingBuffer([0.0005]),
    )
    shape_body = _CountingBuffer([0, 1])
    shape_margin = _CountingBuffer([0.003, 0.0005])
    model = types.SimpleNamespace(
        shape_label=[
            "/World/HandsOnRobots/RightRobot/right_finger1_link1_col2",
            "/World/Task/Geometry/industreal_round_peg_8mm/collision",
        ],
        shape_body=shape_body,
        shape_margin=shape_margin,
    )
    state = types.SimpleNamespace(
        body_q=_CountingBuffer(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )
    )

    class Manager:
        get_contacts = staticmethod(lambda: contacts)
        get_model = staticmethod(lambda: model)
        get_state_0 = staticmethod(lambda: state)

    package = types.ModuleType("isaaclab_newton")
    package.__path__ = []
    physics = types.ModuleType("isaaclab_newton.physics")
    physics.NewtonManager = Manager
    monkeypatch.setitem(sys.modules, "isaaclab_newton", package)
    monkeypatch.setitem(sys.modules, "isaaclab_newton.physics", physics)

    trace = object.__new__(NewtonGroundTruthTrace)
    trace.has_task_controller_state = False
    trace.env = types.SimpleNamespace(task_name="t09_insert_peg")

    first = trace._sample_contact_fields()
    second = trace._sample_contact_fields()

    assert first["physx_hand_peg_contact_valid"].tolist() == [1]
    assert second["physx_hand_peg_contact_valid"].tolist() == [1]
    assert shape_body.numpy_calls == 1
    assert shape_margin.numpy_calls == 1
    assert contacts.rigid_contact_count.numpy_calls == 2
    assert state.body_q.numpy_calls == 2


def test_newton_substep_robot_sampler_does_not_read_endpoint_roots() -> None:
    class RobotData:
        joint_pos = np.asarray([[0.1, 0.2, 0.3]])
        joint_vel = np.asarray([[1.0, 2.0, 3.0]])
        applied_torque = np.asarray([[4.0, 5.0, 6.0]])
        body_state_w = np.arange(26, dtype=np.float64).reshape(1, 2, 13)

        @property
        def root_state_w(self):
            raise AssertionError("substep sampler read endpoint-only root state")

        @property
        def root_link_state_w(self):
            raise AssertionError("substep sampler read endpoint-only root state")

    robot = types.SimpleNamespace(data=RobotData())
    trace = object.__new__(NewtonGroundTruthTrace)
    trace.has_task_controller_state = False
    trace.env = types.SimpleNamespace(
        _joint_targets={
            "right": np.asarray([[0.4, 0.5, 0.6]]),
            "left": np.asarray([[0.7, 0.8, 0.9]]),
        }
    )
    trace.max_robot_joints = 3
    trace.robots = tuple(
        types.SimpleNamespace(
            side=side,
            robot=robot,
            joint_count=3,
            tcp_body_index=1,
        )
        for side in ("right", "left")
    )

    fields = trace._sample_newton_substep_robot_fields()

    assert "physx_robot_root_state" not in fields
    assert "physx_robot_root_valid" not in fields
    np.testing.assert_allclose(
        fields["physx_robot_joint_velocity"],
        [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
    )
    np.testing.assert_allclose(
        fields["physx_robot_joint_target"],
        [[0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
    )


def test_newton_endpoint_reuses_owned_last_substep_without_aliasing() -> None:
    trace = object.__new__(NewtonGroundTruthTrace)
    trace.has_task_controller_state = False
    trace.env = types.SimpleNamespace(
        _n_substeps=1,
        _control_dt=0.1,
        env_step=0,
        _trajectory_state_write_count=0,
    )
    trace.max_robot_joints = 1
    trace.hand_command_dim = 1
    trace.max_contact_headers = 1
    trace.max_contact_points = 1
    trace.max_friction_anchors = 1
    trace.robots = ()
    trace.rigid_bodies = ()
    trace.articulations = ()
    trace.max_articulation_dofs = 0
    trace.deformable_shape = None
    trace.has_rng_state = False
    trace._substep_snapshots = []
    trace._clear_newton_endpoint_cache()

    robot_sample = {
        "physx_robot_joint_position": np.asarray([[1.0], [2.0]]),
        "physx_robot_joint_velocity": np.asarray([[3.0], [4.0]]),
        "physx_robot_joint_target": np.asarray([[5.0], [6.0]]),
        "physx_robot_joint_target_valid": np.ones((2, 1), dtype=np.uint8),
        "physx_robot_joint_valid": np.ones((2, 1), dtype=np.uint8),
        "physx_robot_joint_effort": np.asarray([[7.0], [8.0]]),
        "physx_robot_joint_effort_valid": np.ones((2, 1), dtype=np.uint8),
        "physx_tcp_state": np.arange(26, dtype=np.float64).reshape(2, 13),
        "physx_tcp_valid": np.ones(2, dtype=np.uint8),
    }
    command_sample = {
        "physx_command_tcp_pose": np.zeros((2, 7), dtype=np.float64),
        "physx_command_hand": np.zeros((2, 1), dtype=np.float64),
        "physx_command_hand_valid": np.ones((2, 1), dtype=np.uint8),
        "physx_pose_commanded": np.ones(2, dtype=np.uint8),
        "physx_pose_filter_state": np.zeros((2, 14), dtype=np.float64),
        "physx_pose_filter_valid": np.ones(2, dtype=np.uint8),
    }
    contact_sample = {
        "physx_hand_peg_contact_valid": np.ones(1, dtype=np.uint8),
        "physx_hand_peg_contact_count": np.asarray([2], dtype=np.int64),
        "physx_hand_peg_min_solid_separation": np.asarray([-0.0002]),
        "physx_hand_peg_max_solid_penetration": np.asarray([0.0002]),
    }
    calls = {"contact": 0, "commands": 0, "endpoint_robot": 0}

    def sample_contact():
        calls["contact"] += 1
        return contact_sample

    def sample_commands():
        calls["commands"] += 1
        return command_sample

    def sample_endpoint_robot():
        calls["endpoint_robot"] += 1
        return {
            "physx_robot_joint_target": np.asarray([[9.0], [10.0]]),
            "physx_robot_joint_target_valid": np.ones((2, 1), dtype=np.uint8),
            "physx_robot_root_state": np.zeros((2, 13), dtype=np.float64),
            "physx_robot_root_valid": np.ones(2, dtype=np.uint8),
        }

    trace._sample_newton_substep_robot_fields = lambda: robot_sample
    trace._sample_command_fields = sample_commands
    trace._sample_rigid_body_fields = lambda: {}
    trace._sample_articulation_fields = lambda: {}
    trace._sample_contact_fields = sample_contact
    trace._sample_newton_endpoint_mutable_robot_fields = sample_endpoint_robot
    trace._sample_robot_fields = lambda: (_ for _ in ()).throw(
        AssertionError("endpoint unexpectedly fell back to a full robot sample")
    )

    trace.capture_substep()
    robot_sample["physx_robot_joint_velocity"][0, 0] = 1234.0
    trace.env.env_step = 1

    first = trace.snapshot()
    expected_endpoint_fields = {
        name
        for name in trace.field_specs
        if name.startswith("newton_")
        and not name.startswith("newton_substep_")
    }
    assert set(first) == expected_endpoint_fields
    assert first["newton_robot_joint_velocity"][0, 0] == 3.0
    np.testing.assert_allclose(
        first["newton_robot_joint_target"],
        [[9.0], [10.0]],
    )
    assert calls == {"contact": 1, "commands": 2, "endpoint_robot": 1}

    first["newton_robot_joint_velocity"][0, 0] = -99.0
    second = trace.snapshot()
    assert second["newton_robot_joint_velocity"][0, 0] == 3.0
    assert calls == {"contact": 1, "commands": 3, "endpoint_robot": 2}


def test_newton_gpu_ring_stages_eight_owned_slots_with_one_sync() -> None:
    wp = _FakeWarp()
    stream = _FakeStream()
    source = _fake_cuda_array([0.0, 0.0])
    ring = _NewtonGpuFrameRing(
        wp,
        slot_count=8,
        sources={"joint_position": source},
        stream=stream,
    )

    ring.begin(stream)
    for slot in range(8):
        source.value[:] = [float(slot), float(slot) + 0.25]
        assert ring.stage({"joint_position": source}, stream) == slot

    assert wp.sync_count == 0
    assert "d2h" not in wp.operations
    result = ring.materialize(stream)

    np.testing.assert_allclose(
        result["joint_position"],
        np.asarray([[slot, slot + 0.25] for slot in range(8)]),
    )
    assert wp.sync_count == 1
    assert ring.frame_sync_count == 1
    assert ring.last_d2h_copy_count == 2  # one field plus sentinel

    # Returned snapshots own their memory; a later device/host overwrite
    # cannot mutate an observation already handed to the recorder.
    source.value[:] = -1.0
    ring._fields["joint_position"].host_ring.value[:] = -2.0
    assert result["joint_position"][7, 0] == 7.0


def test_newton_gpu_ring_rejects_incomplete_frame_before_readback() -> None:
    wp = _FakeWarp()
    stream = _FakeStream()
    source = _fake_cuda_array([1.0])
    ring = _NewtonGpuFrameRing(
        wp,
        slot_count=8,
        sources={"q": source},
        stream=stream,
    )
    ring.begin(stream)
    for _ in range(7):
        ring.stage({"q": source}, stream)

    with np.testing.assert_raises_regex(RuntimeError, "incomplete"):
        ring.materialize(stream)
    assert wp.sync_count == 0
    assert "d2h" not in wp.operations


def test_newton_gpu_ring_completion_sentinel_fails_closed() -> None:
    wp = _FakeWarp()
    stream = _FakeStream()
    source = _fake_cuda_array([1.0])
    ring = _NewtonGpuFrameRing(
        wp,
        slot_count=8,
        sources={"q": source},
        stream=stream,
    )
    ring.begin(stream)
    for _ in range(8):
        ring.stage({"q": source}, stream)
    ring._completion_device.value[3] = 0

    with np.testing.assert_raises_regex(RuntimeError, "sentinel mismatch"):
        ring.materialize(stream)
    assert wp.sync_count == 1


def test_newton_gpu_ring_rejects_stream_change_without_syncing() -> None:
    wp = _FakeWarp()
    stream = _FakeStream(handle=17)
    source = _fake_cuda_array([1.0])
    ring = _NewtonGpuFrameRing(
        wp,
        slot_count=8,
        sources={"q": source},
        stream=stream,
    )
    ring.begin(stream)

    with np.testing.assert_raises_regex(RuntimeError, "stream changed"):
        ring.stage({"q": source}, _FakeStream(handle=18))
    assert wp.sync_count == 0


def test_newton_batched_robot_nonfinite_fails_closed() -> None:
    trace = object.__new__(NewtonGroundTruthTrace)
    trace.has_task_controller_state = False
    trace.max_robot_joints = 2
    trace.robots = tuple(
        types.SimpleNamespace(
            side=side,
            joint_count=2,
            tcp_body_index=0,
        )
        for side in ("right", "left")
    )
    raw: dict[str, np.ndarray] = {}
    for side in ("right", "left"):
        for name in (
            "joint_position",
            "joint_velocity",
            "joint_target",
            "joint_effort",
        ):
            raw[f"robot/{side}/{name}"] = np.ones(
                (1, 1, 2),
                dtype=np.float32,
            )
        raw[f"robot/{side}/body_state"] = np.ones(
            (1, 1, 1, 13),
            dtype=np.float32,
        )
        raw[f"robot/{side}/root_state"] = np.ones(
            (1, 1, 13),
            dtype=np.float32,
        )
    raw["robot/right/joint_velocity"][0, 0, 1] = np.nan

    with np.testing.assert_raises_regex(RuntimeError, "non-finite"):
        trace._decode_batched_robot_fields(raw, 0)


def test_newton_reset_snapshot_never_reuses_inactive_action_ring() -> None:
    trace = object.__new__(NewtonGroundTruthTrace)
    trace.has_task_controller_state = False
    trace.env = types.SimpleNamespace(env_step=0, _control_dt=0.1)
    trace._newton_gpu_batch = object()
    trace._newton_frame_active = False
    trace.has_rng_state = False
    trace._substep_snapshots = []
    trace._clear_newton_endpoint_cache()
    trace._materialize_newton_gpu_frame = lambda: (_ for _ in ()).throw(
        AssertionError("reset snapshot tried to materialize the prior action")
    )
    trace._sample_robot_fields = lambda: {
        "physx_robot_joint_position": np.asarray([[42.0]])
    }
    trace._sample_command_fields = lambda: {}
    trace._sample_rigid_body_fields = lambda: {}
    trace._sample_articulation_fields = lambda: {}
    trace._sample_contact_fields = lambda: {}
    trace._sample_deformable_fields = lambda: {}

    snapshot = trace.snapshot()

    assert snapshot["newton_robot_joint_position"].tolist() == [[42.0]]
    assert snapshot["newton_control_step"].tolist() == [0.0]
