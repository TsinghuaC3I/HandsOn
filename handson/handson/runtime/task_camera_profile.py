"""Select task camera dimensions before any simulator/camera module import.

The recorder imports scene helpers before constructing TaskConfig. Camera
constants have already been frozen by then, so the task factory is too late.
This module deliberately has no simulation imports and does not parse flags
on behalf of absl (which still validates the complete command line).
"""
from collections.abc import MutableMapping, Sequence

BC007_TASK_NAME = "bench320_bc007_bimanual_long_rod_yaw_reorientation"
WRIST_OUTPUT_SIZE_ENV = "HANDSON_ISAACLAB_WRIST_CAMERA_OUTPUT_SIZE"


def configure_collection_task_camera_profile(
    argv: Sequence[str], environment: MutableMapping[str, str]
) -> None:
    task_name = None
    args = iter(argv)
    for arg in args:
        if arg == "--":
            break
        if arg == "--exp_name":
            task_name = next(args, None)
        elif arg.startswith("--exp_name="):
            task_name = arg.partition("=")[2]
    if task_name == BC007_TASK_NAME:
        # BC007's mandatory dimensions override generic fleet defaults. The
        # existing factory guard remains responsible for stale-import errors.
        environment[WRIST_OUTPUT_SIZE_ENV] = "640x480"
