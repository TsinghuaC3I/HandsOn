from __future__ import annotations

import hashlib
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "handson"))

from handson.sim.isaaclab.p005_contract import (  # noqa: E402
    EXPECTED_DOF_NAMES,
    INSERTION_HOLD_STEPS,
    KEY_AXIS_ERROR_DEG,
    KEY_AXIS_LOCAL_M,
    KEY_COUPLING_AXIS_ERROR_DEG,
    KEY_INSERTION_DEPTH_M,
    KEY_LATERAL_ERROR_M,
    KEY_TIP_LOCAL_M,
    ASSET_SCALE,
    KEY_RING_CENTER_LOCAL_M,
    PADLOCK_LIFT_MIN_M,
    PADLOCK_KEYHOLE_LOCAL_M,
    PADLOCK_KEYWAY_INWARD_LOCAL_M,
    KEY_XY_BOUNDS_M,
    PADLOCK_XY_BOUNDS_M,
    RESET_MIN_HAND_CLEARANCE_M,
    RESET_MIN_OBJECT_CLEARANCE_M,
    KEY_REST_QUATERNION_WXYZ,
    KEY_REST_ROOT_Z_M,
    KEY_TURN_TARGET_DEG,
    LOCK_CYLINDER_DOF,
    KEY_BODY_NAME,
    PADLOCK_BODY_NAME,
    PADLOCK_ROOT_QUATERNION_WXYZ,
    PADLOCK_ROOT_XYZ_M,
    P005PhysicalReadback,
    P005StageTracker,
    SHACKLE_PREUNLOCK_HOLD_STEPS,
    SHACKLE_PREUNLOCK_MAX_M,
    SHACKLE_HINGE_DOF,
    SHACKLE_SLIDE_DOF,
    SHACKLE_SLIDE_TARGET_M,
    SUCCESS_HOLD_STEPS,
    TABLE_TOP_Z_M,
    TURN_HOLD_STEPS,
    cylinder_coupling_geometry_ok,
    evaluate_physical_readback,
    insertion_geometry_ok,
    key_reset_quaternion_wxyz,
    sample_reset_state,
)


def _neutral_initial_readback() -> P005PhysicalReadback:
    return P005PhysicalReadback(
        key_insertion_depth_m=0.0,
        key_lateral_error_m=0.0,
        key_axis_error_deg=0.0,
        lock_cylinder_angle_deg=0.0,
        shackle_slide_m=0.0,
        shackle_hinge_deg=0.0,
        shackle_linear_speed_mps=0.0,
        shackle_angular_speed_radps=0.0,
        key_linear_speed_mps=0.0,
        key_angular_speed_radps=0.0,
        key_bottom_z_m=1.0,
        padlock_lift_m=0.0,
        right_key_contact_force_n=0.0,
        left_padlock_contact_force_n=0.0,
        left_shackle_contact_force_n=0.0,
        right_shackle_contact_force_n=0.0,
        robot_table_contact_force_n=0.0,
        max_penetration_m=0.0,
        finite_state=True,
    )


def test_recording_origin_rejects_negative_raw_locked_dof_offsets() -> None:
    from handson.sim.isaaclab.p005_collection_env import HandsOnP005IsaacLabEnv

    environment = object.__new__(HandsOnP005IsaacLabEnv)
    environment._physical_readback = _neutral_initial_readback
    environment._contact_reading = lambda _label: (1.0, 0.0)
    environment._p005_failure_latched = False
    environment._p005_tracker = SimpleNamespace(success=False)
    environment._authored_reset_pose_report = lambda: {
        "passed": True,
        "bodies": {},
    }
    raw_positions = {
        LOCK_CYLINDER_DOF: 0.0,
        SHACKLE_SLIDE_DOF: 0.0,
        SHACKLE_HINGE_DOF: 0.0,
    }
    environment._dof_reading = lambda name: (raw_positions[name], 0.0)

    assert environment.stable_recording_origin_task_readiness()["ready"] is True

    raw_positions[LOCK_CYLINDER_DOF] = math.radians(-2.0)
    assert environment.stable_recording_origin_task_readiness()["ready"] is False
    raw_positions[LOCK_CYLINDER_DOF] = 0.0
    raw_positions[SHACKLE_SLIDE_DOF] = -0.002
    assert environment.stable_recording_origin_task_readiness()["ready"] is False
    raw_positions[SHACKLE_SLIDE_DOF] = 0.0
    raw_positions[SHACKLE_HINGE_DOF] = math.radians(-2.0)
    assert environment.stable_recording_origin_task_readiness()["ready"] is False


def test_authored_key_and_padlock_pose_gate_uses_this_seed_target() -> None:
    from handson.sim.isaaclab.p005_collection_env import HandsOnP005IsaacLabEnv

    environment = object.__new__(HandsOnP005IsaacLabEnv)
    environment._p005_reset = sample_reset_state(11)
    key_pose = np.asarray(
        [
            environment._p005_reset.key_x_m,
            environment._p005_reset.key_y_m,
            environment._p005_reset.key_z_m,
            *key_reset_quaternion_wxyz(environment._p005_reset.key_yaw_deg),
        ],
        dtype=np.float64,
    )
    poses = {
        KEY_BODY_NAME: key_pose,
        PADLOCK_BODY_NAME: np.concatenate(
            [
                np.asarray(
                    [
                        environment._p005_reset.padlock_x_m,
                        environment._p005_reset.padlock_y_m,
                        environment._p005_reset.padlock_z_m,
                    ],
                    dtype=np.float64,
                ),
                PADLOCK_ROOT_QUATERNION_WXYZ,
            ]
        ),
    }
    environment._body_pose_wxyz = lambda name: poses[name].copy()

    assert environment._authored_reset_pose_report()["passed"] is True
    poses[KEY_BODY_NAME][2] += 0.005001
    assert environment._authored_reset_pose_report()["passed"] is False


def test_key_geometry_uses_the_side_keyway_frame() -> None:
    from handson.sim.isaaclab.p005_collection_env import HandsOnP005IsaacLabEnv

    environment = object.__new__(HandsOnP005IsaacLabEnv)
    half_sqrt_two = math.sqrt(0.5)
    # Rz(-90°) maps the key's local -Y shaft axis to world -X, the side
    # keyway insertion direction, while local +Z remains the turn reference.
    key_rotation = np.asarray(
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    keyhole = PADLOCK_ROOT_XYZ_M + PADLOCK_KEYHOLE_LOCAL_M
    inserted_tip = keyhole + 0.015 * PADLOCK_KEYWAY_INWARD_LOCAL_M
    key_root = inserted_tip - key_rotation @ KEY_TIP_LOCAL_M
    poses = {
        KEY_BODY_NAME: np.asarray(
            [*key_root, half_sqrt_two, 0.0, 0.0, -half_sqrt_two],
            dtype=np.float64,
        ),
        PADLOCK_BODY_NAME: np.concatenate(
            [PADLOCK_ROOT_XYZ_M, PADLOCK_ROOT_QUATERNION_WXYZ]
        ),
    }
    environment._body_pose_wxyz = lambda name: poses[name].copy()

    depth, lateral, axis_error, _ = environment._key_geometry()
    assert depth == pytest.approx(0.015, abs=1.0e-9)
    assert lateral == pytest.approx(0.0, abs=1.0e-9)
    assert axis_error == pytest.approx(0.0, abs=1.0e-9)


def test_key_geometry_reports_clockwise_turn_from_side_view() -> None:
    from handson.sim.isaaclab.p005_collection_env import HandsOnP005IsaacLabEnv

    s = math.sqrt
    # Viewed from +X looking into the slot, top (+Z) moves to screen-right
    # (+Y): Rx(-60 degrees) followed by the shaft alignment Rz(-90).
    key_rotation = np.asarray(
        [[0.0, 1.0, 0.0], [-0.5, 0.0, s(0.75)], [s(0.75), 0.0, 0.5]],
        dtype=np.float64,
    )
    keyhole = PADLOCK_ROOT_XYZ_M + PADLOCK_KEYHOLE_LOCAL_M
    key_tip = keyhole + 0.015 * PADLOCK_KEYWAY_INWARD_LOCAL_M
    key_root = key_tip - key_rotation @ KEY_TIP_LOCAL_M
    poses = {
        KEY_BODY_NAME: np.asarray(
            [*key_root, 0.6123724357, -0.3535533906, -0.3535533906, -0.6123724357],
            dtype=np.float64,
        ),
        PADLOCK_BODY_NAME: np.concatenate(
            [PADLOCK_ROOT_XYZ_M, PADLOCK_ROOT_QUATERNION_WXYZ]
        ),
    }
    environment = object.__new__(HandsOnP005IsaacLabEnv)
    environment._body_pose_wxyz = lambda name: poses[name].copy()
    assert environment._key_geometry()[3] == pytest.approx(60.0, abs=1.0e-8)


def test_live_runtime_reports_the_key_table_support_pair() -> None:
    source = (
        REPO_ROOT
        / "handson/handson/sim/isaaclab/p005_collection_env.py"
    ).read_text(encoding="utf-8")
    runtime_init = source.split("def _init_task_specific_runtime", 1)[1].split(
        "def _validate_task_initial_state_before_ready", 1
    )[0]
    assert '"robot_table"' not in runtime_init
    assert "hand_only=False" not in runtime_init
    compact = " ".join(source.split())
    assert (
        'self._create_contact_view( physics_view, "key_support", '
        "body_paths[KEY_BODY_NAME], [str(table.GetPath())], )"
    ) in compact
    validator = (
        REPO_ROOT
        / "taskdev/tasks/p005_key_insert_turn_unlock_open_shackle/"
        "live_physics_acceptance.py"
    ).read_text(encoding="utf-8")
    assert 'environment._p005_contact_paths.get("key_support")' in validator
    assert '== {"sensor_count": 1, "filter_count": 1}' in validator
    assert "key_support_pair_exact" in validator


def test_priority_zero_validator_samples_each_substep_before_telemetry_downsample() -> None:
    source = (
        REPO_ROOT
        / "taskdev/tasks/p005_key_insert_turn_unlock_open_shackle/"
        "live_physics_acceptance.py"
    ).read_text(encoding="utf-8")
    loop = source.split("for step in range(stability_steps):", 1)[1]
    every_frame, _telemetry = loop.split(
        "if step % 12 == 0 or step == stability_steps - 1:",
        1,
    )
    assert "readback = environment._physical_readback()" in every_frame
    assert "environment._contact_reading(" in every_frame
    assert '"key_support"' in every_frame
    assert "support_samples += 1" in every_frame
    assert "max_penetration = max(" in every_frame
    assert "max_support_penetration = max(" in every_frame
    assert "nan_count +=" in every_frame
    assert "current_task_body_poses = {" in every_frame
    assert (
        "task_body_names = (KEY_BODY_NAME, PADLOCK_BODY_NAME, SHACKLE_BODY_NAME)"
        in source
    )
    assert "max_task_body_translation_drift_m[name] = max(" in every_frame
    assert "max_task_body_rotation_drift_rad[name] = max(" in every_frame
    assert "max_lock_cylinder_drift_deg = max(" in every_frame
    assert "max_shackle_slide_drift_m = max(" in every_frame
    assert "max_shackle_hinge_drift_deg = max(" in every_frame
    assert "max_task_dof_step_translation_m = max(" in every_frame
    assert "max_task_dof_step_rotation_rad = max(" in every_frame
    assert (
        "current_task_slide_speed_mps = abs("
        in every_frame
    )
    assert "current_task_dof_velocities[SHACKLE_SLIDE_DOF]" in every_frame
    assert "current_task_rotational_dof_speed_radps = max(" in every_frame
    assert "static_pose_covered_steps += int(frame_static_pose)" in every_frame
    assert "PRIORITY_ZERO_LINEAR_SPEED_LIMIT_MPS = 0.001" in source
    assert "PRIORITY_ZERO_ANGULAR_SPEED_LIMIT_RADPS = 0.002" in source
    assert "STATIC_POSE_TRANSLATION_STEP_LIMIT_M = 1.0e-6" in source
    assert "STATIC_POSE_ROTATION_STEP_LIMIT_RAD = 1.0e-5" in source
    assert "velocity_or_static_pose_pass" in source
    assert "PRIORITY_ZERO_MAX_PENETRATION_M = 0.0005" in source
    assert "supported_samples == support_samples" in source
    assert "allow_nan=False" in source


def test_reset_truth_is_read_before_the_extra_zero_action() -> None:
    source = (
        REPO_ROOT
        / "taskdev/tasks/p005_key_insert_turn_unlock_open_shackle/"
        "live_physics_acceptance.py"
    ).read_text(encoding="utf-8")
    reset_loop = source.split("for seed in range(int(args.reset_count)):", 1)[1]
    reset_gate = source.split("reset_pass = bool(", 1)[1].split(
        "reachability_pass = bool(", 1
    )[0]
    raw_readback = reset_loop.index("raw_readback = environment._physical_readback()")
    raw_body_velocity = reset_loop.index("raw_task_body_velocities = {")
    raw_dof_velocity = reset_loop.index("raw_task_dof_velocities = {")
    zero_action = reset_loop.index("environment.step(")
    assert raw_readback < raw_body_velocity < raw_dof_velocity < zero_action
    assert "raw_readback.finite_state" in reset_loop
    assert "post_zero_readback.finite_state" in reset_loop
    for field in (
        '"key_linear_speed_mps"',
        '"key_angular_speed_radps"',
        '"shackle_linear_speed_mps"',
        '"shackle_angular_speed_radps"',
        '"raw_task_linear_speed_mps"',
        '"raw_task_angular_speed_radps"',
        '"post_zero_task_linear_speed_mps"',
        '"post_zero_task_angular_speed_radps"',
        '"lock_cylinder_angle_deg"',
        '"shackle_slide_m"',
        '"shackle_hinge_deg"',
    ):
        assert field in reset_loop
        assert f"row[{field}]" in reset_gate
    assert '"initial_zero_velocity_write_performed": False' in source
    assert '"final_zero_velocity_write_performed": False' in source


TASK_XML = (
    REPO_ROOT
    / "handson/handson/sim/envs/xmls/p005_key_padlock_objects.xml"
)
ARENA_XML = (
    REPO_ROOT
    / "handson/handson/sim/envs/xmls/arena_arm_hand_p005_key_unlock_padlock_rokae_wuji1.xml"
)
ASSETS = {
    "handson/handson/sim/envs/xmls/lightwheel_ycb/key_berkeley_meshes/visuals/key_berkeley_meshes.obj":
        "de74f64a1a9e0ec898a69499c9a5ce3f5dcbaaddd2aad790b6913c6675aef702",
    "handson/handson/sim/envs/xmls/lightwheel_ycb/key_berkeley_meshes/visuals/T_key_berkeley_meshes_BC001.png":
        "b2abe7ba584672b3f81fb9aeedddbc9a2d822e12cbdeb7c4e5d3a7507e80269d",
    "handson/handson/sim/envs/xmls/assets/p005_ycb_key_ring/visuals/key_berkeley_meshes_single_ring.obj":
        "a622ccc5dc6a9673247a65321b2ab8603d77bb4ec8896b2531df7c89011e105d",
    "handson/handson/sim/envs/xmls/lightwheel_ycb/padlock/visuals/padlock.obj":
        "94076bad2ec4e931a4cdf8dce8f702e1ada75e7ec9ad4ef409a117ab4699c285",
    "handson/handson/sim/envs/xmls/lightwheel_ycb/padlock/visuals/padlock_prop.obj":
        "48e4f7cee055c0b67fabe62e0caf65bf55111259de906c50c8012fa3c7e8457d",
    "handson/handson/sim/envs/xmls/lightwheel_ycb/padlock/visuals/T_padlock_BC001.png":
        "588f351213c177176a84f558ffe52a9db0ccffdc834e59ad53da9882576d31d6",
}
ASSETS["handson/handson/sim/envs/xmls/assets/p005_ycb_key_ring/visuals/key_berkeley_meshes_single_operable_key.obj"] = "a79c53904762dec0c7c220a2868a9175e1aa913e891f366e8148eb2159ec731e"
ASSETS.update(
    {
        "handson/handson/sim/envs/xmls/lightwheel_ycb/key_berkeley_meshes/collision/"
        f"key_berkeley_meshes_Collider{index}_m.obj": digest
        for index, digest in enumerate(
            (
                "6422c33e61ab57653a39639703d41b00ee0b3dfd6ec78ecdb984d8b04b4d20ef",
                "7121e0cebe08012d8ae2c2ec10d55df7794f300cf5cc57add6eddde53d269035",
                "f0bde70d45efc05e5a38d0a540094f4bb3ae62b73fc69029a9ec29445b6d5ff6",
                "5aa4564186af9e5d2c757b6058b116dfc5b7726386f992beb72491d06c25ec44",
                "06f544d99f9c2585242610dab1fe907446794069b1fd39b419bc87c4ec46cdd6",
                "94c4a0c780cf21979e2e017e61eb54c95a3f89be3c68ee5d9d4ca096e74eaccf",
                "2866de4b44e9837a933bfb1e57bbd08ed7911f7e2e854afc267aeee0696c3679",
                "f41a1fe980b3a9eadb2cb4d451bf5828a373d5c5f495fd8b2dfe6553bcc2b5fd",
                "9306416ac607a2cac77fab56600b5f360daf6bac67808dc58c2983a152c24962",
                "e1810ea403c1f39bc95c932f3b2f9a9396d1d41ac77fcd5e4c97b1a158465077",
                "bd30a584384807f9075daf1a97a7bdf0e7905709b0811cee778b840ddd7e7aed",
                "1aac04355cb173f0944b1759093918e06b7f5c8ed95f60144d4f5f903aca1b85",
                "911e91c4c60d69e1dfff3d43fb591a1d9f3301fc20f5e5d2deb17bf551840fd3",
                "e2302e91ca443cbb7896e51ce30a6f5ff7db634ca0377f2cff68dff3f97d2459",
                "a401b4044174a4a7169ec57437d144a0483f1ca014095838c72a0ed8d017cb88",
                "a993dd20ccfbf4e8b5a9fff86f0100a42286df8a5ea7f587e409c96659d02e53",
            ),
            start=1,
        )
    }
)
ASSETS.update(
    {
        "handson/handson/sim/envs/xmls/assets/p005_ycb_key_ring/collision/"
        f"key_berkeley_meshes_Collider{index}_m.obj": digest
        for index, digest in enumerate(
            (
                "a32f3ac1f8f120fad5dc796f8af2af5b004d4572fbd9a15f8737db437406cbbc",
                "6d7a0b8fd28d3d2724b755716b85ca372182a5b61067877982fec62c4a26ce64",
                "47597467f0325a68edab830a92bd9dfada291f992f8b08c1dff0211046640f72",
                "dee5427d57df3b0ba77897980b272d025737b9cccbebb1e3c744a3ad5693d6b2",
                "307fc240844851d86b4cb429371f4697ac55ae7bc2bb3362bdebc633cef774a7",
                "7c4a0c79ab79f00bc0208eebe62daed0db5df2e553bad9d334eddf96b91babfd",
                "eae36b742b72f3d04ae5118eca650cd9527ed50b851b39e6ddda48c65eefd6bf",
                "f6acd844d77dc63c95182122e8ac4aafb63e262e7189d8e7954bb72ded03c78b",
                "165995c7d746f9c819112b6617af259fb221fd6b0be60530ae7ffcabca7550a3",
                "f01df452c64325233f8a8870071d800c51e03ad34466a9ff8f1b5d0e1fd11b8f",
                "78ab9516922c047ffa0a932db6c87f7f763ec4028e99d39d76fabb899abc328c",
                "a8c61cc36f33d57afeccbc3604269f5d6848ac62064ebc518e6e35cf11411c6d",
                "8e10b1addf3a214c2a341a4cc93a4113f71e6dbcfd8a4a83ceeb6d349231feb8",
                "8a07656fc68a0fbb192eb34810cefd157a6bfd9d5d1f7d61ac9bbb96fc758e87",
                "c4260b502ba90e837573539a5512234b27d5a395f21c302408e901551f9752b1",
                "bb9c86ddd07db369ed092b13d9d3945590198c74364a531475312cc72e9d0fa2",
            ),
            start=1,
        )
    }
)


def _readback(**overrides) -> P005PhysicalReadback:
    values = dict(
        key_insertion_depth_m=0.0,
        key_lateral_error_m=0.0,
        key_axis_error_deg=0.0,
        lock_cylinder_angle_deg=0.0,
        shackle_slide_m=0.0,
        shackle_hinge_deg=0.0,
        shackle_linear_speed_mps=0.0,
        shackle_angular_speed_radps=0.0,
        key_linear_speed_mps=0.0,
        key_angular_speed_radps=0.0,
        key_bottom_z_m=0.895,
        padlock_lift_m=0.04,
        right_key_contact_force_n=0.10,
        left_padlock_contact_force_n=0.10,
        left_shackle_contact_force_n=0.0,
        right_shackle_contact_force_n=0.0,
        robot_table_contact_force_n=0.0,
        max_penetration_m=0.0,
        finite_state=True,
    )
    values.update(overrides)
    return P005PhysicalReadback(**values)


def test_asset_blobs_are_real_and_hash_locked() -> None:
    for relative, expected_sha in ASSETS.items():
        path = REPO_ROOT / relative
        payload = path.read_bytes()
        assert len(payload) > 100
        assert not payload.startswith(b"version https://git-lfs.github.com/spec/v1")
        if path.suffix == ".obj":
            text = payload.decode("utf-8")
            assert "\nv " in text
            assert "\nf " in text
        else:
            assert len(payload) > 1000
        assert hashlib.sha256(payload).hexdigest() == expected_sha


def test_mjcf_uses_relative_assets_and_separate_joint_bodies() -> None:
    root = ET.parse(TASK_XML).getroot()
    assert all(not Path(node.attrib["file"]).is_absolute() for node in root.findall(".//*[@file]"))
    joint_names = {node.attrib.get("name") for node in root.findall(".//joint")}
    assert EXPECTED_DOF_NAMES <= joint_names
    slide_body = root.find(".//body[@name='p005_shackle_slide_body']")
    shackle_body = root.find(".//body[@name='p005_shackle']")
    assert slide_body is not None and shackle_body is not None
    assert slide_body.find("./joint[@name='p005_shackle_slide_joint']") is not None
    assert shackle_body.find("./joint[@name='p005_shackle_hinge_joint']") is not None
    arena = ET.parse(ARENA_XML).getroot()
    includes = {node.attrib["file"] for node in arena.findall("include")}
    assert "rokae_wuji1_bimanual.xml" in includes
    assert "p005_key_padlock_objects.xml" in includes


def test_key_ring_is_single_rear_component_at_task_scale() -> None:
    root = ET.parse(TASK_XML).getroot()
    key_body = root.find(".//body[@name='p005_key_root_body']")
    assert key_body is not None
    assert float(key_body.attrib["pos"].split()[2]) == pytest.approx(
        KEY_REST_ROOT_Z_M, abs=1.0e-9
    )
    np.testing.assert_allclose(
        np.fromstring(key_body.attrib["quat"], sep=" "),
        KEY_REST_QUATERNION_WXYZ,
        atol=1.0e-9,
    )
    visuals = root.findall(".//geom[@name='p005_key_visual']")
    assert len(visuals) == 1
    assert visuals[0].attrib["mesh"] == "p005_key_visual_mesh"
    meshes = {mesh.attrib["name"]: mesh for mesh in root.findall(".//mesh")}
    expected_scale = f"{ASSET_SCALE:g} {ASSET_SCALE:g} {ASSET_SCALE:g}"
    assert meshes["p005_key_visual_mesh"].attrib["scale"] == expected_scale
    key_collision_meshes = [
        mesh for name, mesh in meshes.items() if name.startswith("p005_key_collider")
    ]
    assert len(key_collision_meshes) == 15
    assert "p005_key_collider03" not in meshes
    assert all(mesh.attrib.get("scale") == "1 1 1" for mesh in key_collision_meshes)
    assert all(
        mesh.attrib["file"].startswith("assets/p005_ycb_key_ring/collision/")
        for mesh in key_collision_meshes
    )
    assert key_body.find("./site[@name='p005_key_bow']") is not None
    np.testing.assert_allclose(
        np.fromstring(
            key_body.find("./site[@name='p005_key_bow']").attrib["pos"], sep=" "
        ),
        KEY_RING_CENTER_LOCAL_M,
        atol=1.0e-9,
    )
    assert len(key_body.findall("./geom[@class='p005_key_collision']")) == 15
    assert not root.findall(".//geom[@name='p005_key_ring_visual']")
    assert not any(
        "ring_collision" in str(geom.attrib.get("name", ""))
        for geom in root.findall(".//geom")
    )


def test_free_padlock_has_side_keyway_and_standard_pose() -> None:
    root = ET.parse(TASK_XML).getroot()
    padlock = root.find(".//body[@name='p005_padlock_body']")
    assert padlock is not None
    np.testing.assert_allclose(
        np.fromstring(padlock.attrib["pos"], sep=" "),
        PADLOCK_ROOT_XYZ_M,
        atol=1.0e-9,
    )
    quaternion = np.fromstring(padlock.attrib["quat"], sep=" ")
    np.testing.assert_allclose(
        quaternion,
        PADLOCK_ROOT_QUATERNION_WXYZ,
        atol=1.0e-9,
    )
    assert padlock.find("./freejoint[@name='p005_padlock_root']") is not None
    cylinder = root.find(".//body[@name='p005_lock_cylinder']")
    assert cylinder is not None
    assert cylinder.attrib["pos"] == "0.029400000 0 -0.009720000"
    assert cylinder.find("./joint[@name='p005_lock_cylinder_yaw']").attrib["axis"] == "-1 0 0"
    assert not padlock.findall("./geom[@class='p005_keyway_collision']")
    assert len(cylinder.findall("./geom[@class='p005_keyway_collision']")) == 4


def test_seeded_reset_distribution_is_bounded_and_deterministic() -> None:
    samples = [sample_reset_state(seed) for seed in range(50)]
    assert samples == [sample_reset_state(seed) for seed in range(50)]
    assert len({(sample.key_x_m, sample.key_y_m, sample.key_yaw_deg) for sample in samples}) == 50
    assert len({(sample.padlock_x_m, sample.padlock_y_m) for sample in samples}) == 50
    assert all(
        KEY_XY_BOUNDS_M[0, 0] <= sample.key_x_m <= KEY_XY_BOUNDS_M[0, 1]
        and KEY_XY_BOUNDS_M[1, 0] <= sample.key_y_m <= KEY_XY_BOUNDS_M[1, 1]
        for sample in samples
    )
    assert all(
        PADLOCK_XY_BOUNDS_M[0, 0] <= sample.padlock_x_m <= PADLOCK_XY_BOUNDS_M[0, 1]
        and PADLOCK_XY_BOUNDS_M[1, 0] <= sample.padlock_y_m <= PADLOCK_XY_BOUNDS_M[1, 1]
        for sample in samples
    )
    assert all(
        math.dist(
            (sample.key_x_m, sample.key_y_m),
            (sample.padlock_x_m, sample.padlock_y_m),
        )
        >= RESET_MIN_OBJECT_CLEARANCE_M
        for sample in samples
    )
    assert all(0.95 <= sample.key_friction <= 1.15 for sample in samples)
    assert all(abs(sample.key_yaw_deg) <= 10.0 for sample in samples)
    assert all(sample.key_z_m == KEY_REST_ROOT_Z_M for sample in samples)
    np.testing.assert_allclose(
        key_reset_quaternion_wxyz(0.0), KEY_REST_QUATERNION_WXYZ, atol=1.0e-12
    )


def test_seeded_reset_avoids_current_home_hand_positions() -> None:
    hand_centers = (
        np.asarray([-0.10, -0.16, 1.07]),
        np.asarray([-0.10, 0.16, 1.07]),
    )
    for seed in range(50):
        sample = sample_reset_state(seed, forbidden_centers=hand_centers)
        for object_xy in (
            (sample.key_x_m, sample.key_y_m),
            (sample.padlock_x_m, sample.padlock_y_m),
        ):
            assert min(
                math.dist(object_xy, center[:2]) for center in hand_centers
            ) >= RESET_MIN_HAND_CLEARANCE_M


def test_free_key_uses_strict_every_frame_static_pose_proof() -> None:
    source = (
        REPO_ROOT
        / "handson/handson/sim/isaaclab/recording_origin.py"
    ).read_text(encoding="utf-8")
    assert '"p005_key_insert_turn_unlock_open_shackle": (1.0e-6, 1.0e-5)' in source


def test_end_to_end_acceptance_enables_and_checks_the_platform_origin_gate() -> None:
    source = (
        REPO_ROOT
        / "taskdev/tasks/p005_key_insert_turn_unlock_open_shackle/live_end_to_end_acceptance.py"
    ).read_text(encoding="utf-8")
    assert 'os.environ["HANDSON_STABLE_RECORDING_ORIGIN"] = "1"' in source
    assert 'reset_info.get("physical_recording_origin")' in source
    assert "P0-05 platform-equivalent acceptance requires" in source


def test_ordered_success_requires_all_physical_stages() -> None:
    tracker = P005StageTracker()
    inserted = _readback(
        key_insertion_depth_m=KEY_INSERTION_DEPTH_M + 0.001,
        key_lateral_error_m=KEY_LATERAL_ERROR_M - 0.001,
        key_axis_error_deg=KEY_AXIS_ERROR_DEG - 1.0,
    )
    for _ in range(INSERTION_HOLD_STEPS):
        tracker.update(inserted)
    assert tracker.key_inserted_seen
    turned = _readback(
        key_insertion_depth_m=KEY_INSERTION_DEPTH_M + 0.001,
        key_lateral_error_m=KEY_LATERAL_ERROR_M - 0.001,
        key_axis_error_deg=KEY_AXIS_ERROR_DEG - 1.0,
        lock_cylinder_angle_deg=KEY_TURN_TARGET_DEG + 1.0,
    )
    for _ in range(TURN_HOLD_STEPS):
        tracker.update(turned)
    assert tracker.key_turned_seen
    opened = _readback(
        key_insertion_depth_m=KEY_INSERTION_DEPTH_M + 0.001,
        key_lateral_error_m=KEY_LATERAL_ERROR_M - 0.001,
        key_axis_error_deg=KEY_AXIS_ERROR_DEG - 1.0,
        lock_cylinder_angle_deg=KEY_TURN_TARGET_DEG + 1.0,
        shackle_slide_m=SHACKLE_SLIDE_TARGET_M + 0.001,
        left_padlock_contact_force_n=0.10,
    )
    for _ in range(SUCCESS_HOLD_STEPS):
        tracker.update(opened)
    assert tracker.success
    assert evaluate_physical_readback(opened, tracker)["success_now"]


def test_shackle_unlock_requires_completed_insertion_and_key_turn() -> None:
    tracker = P005StageTracker()
    assert tracker.shackle_unlock_allowed is False

    inserted = _readback(
        key_insertion_depth_m=KEY_INSERTION_DEPTH_M + 0.001,
        key_lateral_error_m=KEY_LATERAL_ERROR_M - 0.001,
        key_axis_error_deg=KEY_AXIS_ERROR_DEG - 1.0,
    )
    for _ in range(INSERTION_HOLD_STEPS):
        tracker.update(inserted)
    assert tracker.key_inserted_seen is True
    assert tracker.key_turned_seen is False
    assert tracker.shackle_unlock_allowed is False

    turned = _readback(
        key_insertion_depth_m=KEY_INSERTION_DEPTH_M + 0.001,
        key_lateral_error_m=KEY_LATERAL_ERROR_M - 0.001,
        key_axis_error_deg=KEY_AXIS_ERROR_DEG - 1.0,
        lock_cylinder_angle_deg=KEY_TURN_TARGET_DEG + 1.0,
    )
    for _ in range(TURN_HOLD_STEPS):
        tracker.update(turned)
    assert tracker.key_turned_seen is True
    assert tracker.shackle_unlock_allowed is True


def test_turn_or_open_cannot_skip_insertion_and_preunlock_pull_fails() -> None:
    tracker = P005StageTracker()
    skip = _readback(
        lock_cylinder_angle_deg=KEY_TURN_TARGET_DEG + 10.0,
        shackle_slide_m=SHACKLE_SLIDE_TARGET_M + 0.002,
        left_padlock_contact_force_n=0.2,
    )
    for _ in range(30):
        tracker.update(skip)
    assert not tracker.key_inserted_seen
    assert not tracker.key_turned_seen
    assert not tracker.success
    assert tracker.preunlock_shackle_motion
    assert evaluate_physical_readback(skip, tracker)["failed"]


def test_failure_gates_are_strict() -> None:
    for bad in (
        _readback(max_penetration_m=0.006),
        _readback(key_bottom_z_m=0.86),
        _readback(key_linear_speed_mps=6.0),
        _readback(finite_state=False),
    ):
        tracker = P005StageTracker()
        tracker.update(bad)
        assert evaluate_physical_readback(bad, tracker)["failed"]

    preunlock_pull = _readback(
        shackle_slide_m=SHACKLE_PREUNLOCK_MAX_M + 0.001
    )
    tracker = P005StageTracker()
    for _ in range(SHACKLE_PREUNLOCK_HOLD_STEPS - 1):
        tracker.update(preunlock_pull)
        assert not evaluate_physical_readback(preunlock_pull, tracker)["failed"]
    tracker.update(preunlock_pull)
    assert evaluate_physical_readback(preunlock_pull, tracker)["failed"]


def test_robot_table_contact_is_recoverable_and_not_a_failure_gate() -> None:
    tracker = P005StageTracker()
    metrics = evaluate_physical_readback(
        _readback(robot_table_contact_force_n=math.inf),
        tracker,
    )

    assert metrics["robot_table_contact_gating_enabled"] is False
    assert metrics["failed"] is False


def test_right_hand_key_contact_does_not_fail_by_itself() -> None:
    tracker = P005StageTracker()
    metrics = evaluate_physical_readback(
        _readback(
            right_key_contact_force_n=8.0,
            left_padlock_contact_force_n=0.0,
            padlock_lift_m=0.0,
        ),
        tracker,
    )

    assert metrics["failed"] is False


def test_live_penetration_gate_excludes_expected_hand_contacts() -> None:
    source = (
        REPO_ROOT
        / "handson/handson/sim/isaaclab/p005_collection_env.py"
    ).read_text(encoding="utf-8")
    assert 'readings["key_lock"][1], readings["shackle_lock"][1]' in source
    assert "max(value[1] for value in readings.values())" not in source


def test_cylinder_contact_window_does_not_relax_task_alignment_gate() -> None:
    edge_contact = _readback(
        key_insertion_depth_m=KEY_INSERTION_DEPTH_M + 0.001,
        key_lateral_error_m=KEY_LATERAL_ERROR_M - 0.001,
        key_axis_error_deg=(KEY_AXIS_ERROR_DEG + KEY_COUPLING_AXIS_ERROR_DEG) / 2,
    )
    assert cylinder_coupling_geometry_ok(edge_contact)
    assert not insertion_geometry_ok(edge_contact)
