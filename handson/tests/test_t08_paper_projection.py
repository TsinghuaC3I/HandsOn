"""Reset visibility is independent of the calibrated observation camera."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from handson.sim.isaaclab import abc_collection_env as module


def test_reset_projection_matches_archived_source_and_survives_render_calibration():
    oracle = json.loads(
        (Path(__file__).parent / "fixtures/t08_reset_projection.json").read_text()
    )
    method = module.HandsOnABCIsaacLabEnv._c01_head_ndc
    expected = np.asarray(oracle["expected_ndc"])
    actual = method(oracle["points"])
    # The archived float64 oracle was generated on x86; ARM arithmetic can
    # differ by one ULP. Calibration invariance below remains exact.
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-15)
    with (
        patch.object(module, "HEAD_RECORDING_CAMERA_EYE", (10.0, 20.0, 30.0)),
        patch.object(module, "_HEAD_RECORDING_CAMERA_QUAT_WXYZ", (1.0, 0.0, 0.0, 0.0)),
        patch.object(module, "_HEAD_RECORDING_CAMERA_FOVY", 90.0),
    ):
        np.testing.assert_array_equal(method(oracle["points"]), actual)
