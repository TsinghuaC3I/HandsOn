"""Replay-only command activation metadata, separate from the EEF52 policy API."""
import numpy as np

from ..eef52 import SIDES, decode_command


def _flags(value):
    flags = np.asarray(value)
    if flags.shape != (2,) or not np.isin(flags, [0, 1]).all():
        raise ValueError('Recorded pose-command flags must be two binary values in right,left order')
    return flags.astype(bool, copy=True)


class RecordedPoseControl:
    """Preserve the source's latched pose activation during historical replay.

    EEF52 always decodes into valid target poses. Some source recordings begin
    with hand-only commands, while their stored arm targets are still inactive.
    Use separately verified source flags to retain that native no-pose-command
    state. Zero pose7 is the native input sentinel, not a valid EEF52 rotation.

    This is a richer replay input, not a change to the trained-policy interface.
    It neither writes physical state nor substitutes measured or future poses.
    The initial controller flags must already have been restored with t0.
    """

    def __init__(self, native, initial_pose_commanded):
        self.native = native
        self.expected = _flags(initial_pose_commanded)
        self.last_command = None
        self._check_native(self.expected)

    def _check_native(self, expected):
        actual = _flags([getattr(self.native, f'_{side}_pose_commanded') for side in SIDES])
        if not np.array_equal(actual, expected):
            raise ValueError('Native pose-command state differs from the recorded replay state')

    def step(self, action, pose_commanded):
        flags = _flags(pose_commanded)
        self._check_native(self.expected)
        if np.any(self.expected & ~flags):
            raise ValueError('A no-command input cannot clear an already activated pose controller')
        command = decode_command(action)
        for index, side in enumerate(SIDES):
            if not flags[index]:
                command[side][:7] = 0.0
        self.last_command = command
        result = self.native.step(command)
        self._check_native(flags)
        self.expected = flags
        return result


class RecordedJointTargetControl:
    """Apply recorded motor setpoints at native substeps, without scene-state writes.

    This replay-only input is richer than EEF52. It does not predict actions or
    change the policy evaluator. The caller supplies the verified episode schema
    and the motor targets corresponding to each unchanged public EEF command.
    """

    def __init__(self, native, physics_schema):
        self.native = native
        self.trace = native._physx_ground_truth_trace
        self.schema = {'physx': physics_schema}
        self.applied_substeps = 0
        self.max_target_error = 0.0

    def step(self, action, targets, valid):
        native = self.native
        targets, valid = np.asarray(targets), np.asarray(valid)
        expected = (native._n_substeps, 2, 27)
        if targets.shape != expected or valid.shape != expected:
            raise ValueError('Recorded joint targets do not match the native substep layout')
        if not np.isfinite(targets).all() or not np.isin(valid, [0, 1]).all():
            raise ValueError('Invalid recorded joint targets or validity mask')
        hook = '_handson_pre_physics_substep_callback'
        if getattr(native, hook, None) is not None:
            raise ValueError('Another substep callback is already installed')
        calls = 0

        def apply(index, count):
            nonlocal calls
            if count != expected[0] or index != calls:
                raise ValueError('Native substep sequence differs from the recording')
            packet = dict(physx_robot_joint_target=targets[index],
                          physx_robot_joint_target_valid=valid[index])
            mask = valid[index].astype(bool)
            if not mask.any():
                raise ValueError('Recorded substep has no valid motor targets')
            applied = self.trace.apply_robot_joint_targets(packet, self.schema)
            if applied != int(mask.sum()):
                raise ValueError('Recorded motor targets were not all applied')
            actual = self.trace.snapshot()['physx_robot_joint_target']
            error = float(np.max(np.abs(actual[mask] - targets[index][mask])))
            if not np.isfinite(error) or error > 1e-7:
                raise ValueError('Native motor target readback differs from the recording')
            self.max_target_error = max(self.max_target_error, error)
            calls += 1
            self.applied_substeps += 1

        before = getattr(native, '_before_control_step', None)
        if before is not None:
            before()
        native._apply_action(decode_command(action))
        setattr(native, hook, apply)
        try:
            result = native.step_with_current_joint_targets()
            if calls != expected[0]:
                raise ValueError('Native step did not execute every recorded substep')
            return result
        finally:
            setattr(native, hook, None)
