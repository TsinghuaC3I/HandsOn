"""Finite-mass articulated-drive replay helpers.

This package deliberately contains no kinematic proxy or state-teleport path.
It reuses the task's original robot articulations and changes only their
recorded position-drive targets before each physics substep.
"""

from .drive_path import (
    ArticulatedDriveControlStep,
    RecordedArticulatedDrivePath,
    run_articulated_drive_control_step,
    validate_recorded_physics_configuration,
)

__all__ = (
    "ArticulatedDriveControlStep",
    "RecordedArticulatedDrivePath",
    "run_articulated_drive_control_step",
    "validate_recorded_physics_configuration",
)
