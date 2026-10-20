from __future__ import annotations

import ast
import hashlib
import json
import math
import re
import struct
from pathlib import Path

import numpy as np
import pytest
import yaml

from taskdev.tasks.t16_open_laptop.core_scene_runtime import (
    aggregate_checks,
    decoded_rgb_metrics,
    head_content_pass,
    quaternion_angle_rad,
)
from taskdev.tasks.t16_open_laptop.core_scene_spec import (
    EXPECTED_COLLISION_PATH_SUFFIXES,
    HEAD_SHAPE_HWC,
    HINGE_LOWER_DEG,
    HINGE_UPPER_DEG,
    INITIAL_HINGE_DEG,
    PLACEMENT_YAW_DEG,
    SCALE_XYZ,
    SOURCE_AABB_M,
    TARGET_OPEN_DEG,
    ZERO_ACTION_SECONDS,
    ZERO_ACTION_STEPS,
    initial_fraction_of_travel,
)


TASK_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = TASK_DIR.parents[2]


def test_task_package_init_is_zero_bytes() -> None:
    assert (TASK_DIR / "__init__.py").read_bytes() == b""


def test_exact_asset_bundle_and_robot_exist() -> None:
    required = (
        TASK_DIR / "assets/dv_laptop/model_computer_9.usd",
        TASK_DIR / "assets/dv_laptop/resource/material.usd",
        TASK_DIR / "assets/dv_laptop/resource/v_0/link_default_0.usd",
        TASK_DIR / "assets/dv_laptop/resource/img/18c93ae8fc20d972508941b6b33d5501.png",
        TASK_DIR / "assets/dv_laptop/resource/img/90f4fa946fc6a4c4878739260b1a0321.png",
        REPOSITORY_ROOT / "handson/handson/sim/envs/xmls/rokae_wuji1_bimanual.xml",
    )
    assert all(path.is_file() and path.stat().st_size > 0 for path in required)


def test_scale_pose_and_safe_initial_hinge_contract() -> None:
    assert SCALE_XYZ == (1.0, 1.0, 1.0)
    assert PLACEMENT_YAW_DEG == -90.0
    assert HINGE_LOWER_DEG < INITIAL_HINGE_DEG < HINGE_UPPER_DEG
    assert initial_fraction_of_travel() == pytest.approx(0.20)
    assert 0.10 <= initial_fraction_of_travel() <= 0.30
    assert abs(INITIAL_HINGE_DEG - TARGET_OPEN_DEG) > 70.0
    assert ZERO_ACTION_SECONDS == pytest.approx(0.25)
    assert ZERO_ACTION_STEPS == 30


def test_measured_source_bounds_and_collider_partition_are_complete() -> None:
    low, high = SOURCE_AABB_M
    size = tuple(high[index] - low[index] for index in range(3))
    assert size == pytest.approx((0.3282400914, 0.2300044975, 0.0174805816))
    assert len(EXPECTED_COLLISION_PATH_SUFFIXES) == 13
    assert len(set(EXPECTED_COLLISION_PATH_SUFFIXES)) == 13
    assert sum(path.startswith("E_body_1/") for path in EXPECTED_COLLISION_PATH_SUFFIXES) == 8
    assert sum(path.startswith("E_displayer_5/") for path in EXPECTED_COLLISION_PATH_SUFFIXES) == 5


def test_head_content_gate_rejects_black_and_accepts_structured_rgb() -> None:
    black = np.zeros(HEAD_SHAPE_HWC, dtype=np.uint8)
    assert head_content_pass(decoded_rgb_metrics(black), 200_000) is False
    rows, cols, _ = HEAD_SHAPE_HWC
    y = np.arange(rows, dtype=np.uint16)[:, None]
    x = np.arange(cols, dtype=np.uint16)[None, :]
    frame = np.empty(HEAD_SHAPE_HWC, dtype=np.uint8)
    frame[..., 0] = ((x + y) % 256).astype(np.uint8)
    frame[..., 1] = ((2 * x + y) % 256).astype(np.uint8)
    frame[..., 2] = ((x + 3 * y) % 256).astype(np.uint8)
    assert head_content_pass(decoded_rgb_metrics(frame), 200_000) is True


def test_quaternion_drift_uses_shortest_arc() -> None:
    first = np.asarray([0, 0, 0, 0, 0, 0, 1], dtype=np.float64)
    same = np.asarray([0, 0, 0, 0, 0, 0, -1], dtype=np.float64)
    quarter = np.asarray(
        [0, 0, 0, 0, 0, math.sin(math.pi / 4), math.cos(math.pi / 4)],
        dtype=np.float64,
    )
    assert quaternion_angle_rad(first, same) == pytest.approx(0.0)
    assert quaternion_angle_rad(first, quarter) == pytest.approx(math.pi / 2)


def test_aggregate_checks_fails_closed() -> None:
    assert aggregate_checks({"a": True, "b": True}) == (True, [])
    assert aggregate_checks({"a": True, "b": False}) == (False, ["b"])


def test_manifest_is_explicitly_core_scene_only() -> None:
    manifest = json.loads((TASK_DIR / "scene_manifest.json").read_text(encoding="utf-8"))
    assert manifest["scope"] == "core_scene_only"
    assert manifest["claims"]["task_complete"] is False
    assert manifest["claims"]["path_or_action_tuned"] is False
    assert manifest["claims"]["core_scene_ready"] is False
    assert manifest["claims"]["zero_action_stable"] is True
    assert manifest["claims"]["zero_action_scope"] == "laptop_table_standalone_only"
    assert manifest["claims"]["dual_arm_runtime_validated"] is False
    assert manifest["claims"]["head_camera_validated"] is False
    assert manifest["components"]["laptop"]["scale_xyz"] == [1.0, 1.0, 1.0]
    assert manifest["components"]["laptop"]["collision_shape_count"] == 13
    camera = manifest["components"]["head_camera"]
    assert camera["design_preview"] == "previews/core_scene_design.png"
    assert camera["design_preview_kind"] == "cpu_schematic_not_camera_evidence"


def test_pure_usd_source_audit_matches_runtime_contract() -> None:
    audit = json.loads(
        (TASK_DIR / "evidence/source_asset_audit.json").read_text(encoding="utf-8")
    )
    assert audit["audit_method"] == "pure_usd_cpu"
    assert audit["isaac_or_kit_started"] is False
    assert audit["authored_scale_xyz"] == list(SCALE_XYZ)
    assert audit["hinge"]["axis"] == "X"
    assert audit["hinge"]["lower_limit_deg"] == HINGE_LOWER_DEG
    assert audit["hinge"]["upper_limit_deg"] == HINGE_UPPER_DEG
    assert audit["collision_shape_count"] == len(EXPECTED_COLLISION_PATH_SUFFIXES)
    assert audit["collision_paths"] == list(EXPECTED_COLLISION_PATH_SUFFIXES)
    assert audit["task_complete"] is False


def test_design_preview_is_labelled_640x480_png() -> None:
    preview = TASK_DIR / "previews/core_scene_design.png"
    payload = preview.read_bytes()
    assert payload.startswith(b"\x89PNG\r\n\x1a\n")
    assert struct.unpack(">II", payload[16:24]) == (640, 480)
    assert len(payload) >= 150_000
    svg = (TASK_DIR / "previews/core_scene_design.svg").read_text(encoding="utf-8")
    assert "CPU SCHEMATIC" in svg
    assert "LIVE RTX HEAD PREVIEW PENDING" in svg


def test_scheduler_owned_laptop_physics_evidence_passes_in_scope() -> None:
    evidence_path = TASK_DIR / "evidence/laptop_physics_live.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    marker = json.loads(
        (TASK_DIR / "evidence/LAPTOP_PHYSICS_PASS.json").read_text(encoding="utf-8")
    )
    namespace = json.loads(
        (TASK_DIR / "evidence/scheduler_gpu_namespace.json").read_text(encoding="utf-8")
    )
    blocker = json.loads(
        (TASK_DIR / "evidence/full_core_scene_blocker.json").read_text(encoding="utf-8")
    )
    assert all(evidence["checks"].values())
    assert evidence["standalone_laptop_physics_pass"] is True
    assert evidence["full_core_scene_pass"] is False
    assert evidence["task_complete"] is False
    assert evidence["zero_action"]["steps"] == 30
    assert evidence["zero_action"]["base_translation_drift_m"] <= 0.0001
    assert evidence["zero_action"]["final_hinge_error_rad"] <= 0.001
    assert evidence["contacts"]["max_penetration_m"] == 0.0
    assert marker["standalone_laptop_physics_pass"] is True
    assert marker["full_core_scene_pass"] is False
    assert marker["summary_sha256"] == hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    assert namespace["physical_gpu"] == 4
    assert namespace["logical_gpu"] == 0
    assert blocker["production_head_frame_available"] is False
    assert blocker["full_core_scene_pass"] is False
    assert blocker["task_complete"] is False


def test_validation_report_records_complete_acceptance_scope() -> None:
    manifest_path = TASK_DIR / "task.yaml"
    report = yaml.safe_load((TASK_DIR / "validation_report.yaml").read_text(encoding="utf-8"))
    assert report["manifest_sha256"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    checks = report["checks"]
    assert all(
        checks[name]["status"] == "pass"
        for name in (
            "schema",
            "assets",
            "reachability",
            "contact",
            "reset",
            "success",
            "camera",
            "stability",
            "collection",
        )
    )
    assert checks["success"]["metrics"]["task_complete"] is True


def test_task_manifest_uses_true_articulation_contract() -> None:
    task = yaml.safe_load((TASK_DIR / "task.yaml").read_text(encoding="utf-8"))
    assert task["task"]["taxonomy"]["family"] == "articulation"
    assert task["robot"]["arms"]["count"] == 2
    assert task["robot"]["hands"] == {
        "model": "wuji_gen1",
        "generation": 1,
        "count": 2,
        "asset": "handson/handson/sim/envs/xmls/rokae_wuji1_bimanual.xml",
    }
    assert task["assets"][0]["scale_m"] == [1.0, 1.0, 1.0]
    predicates = {item["id"]: item for item in task["success"]["predicates"]}
    assert predicates["screen_angle"]["parameters"]["target_deg"] == 105.0
    assert task["success"]["hold_steps"] == 15


def test_runner_obeys_app_import_and_validator_source_contracts() -> None:
    path = TASK_DIR / "core_scene_runner.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = [
        (node.lineno, alias.name)
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    ] + [
        (node.lineno, node.module or "")
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    ]
    app_line = min(line for line, module in imports if module == "isaaclab.app")
    post_app_modules = ("pxr", "carb", "omni.physx", "handson.sim.isaaclab")
    assert all(line > app_line for line, module in imports if module.startswith(post_app_modules))
    assert "args.width = 640" in source
    assert "args.height = 480" in source
    assert "RayTracedLighting" in source
    assert 'settings.set("/rtx/post/tonemap/op", 4)' in source
    assert "UsdLux.DomeLight" in source and "UsdLux.DistantLight" in source
    assert "dome.CreateIntensityAttr(300.0)" in source
    assert "distant.CreateIntensityAttr(2500.0)" in source
    assert "omni.isaac." not in source
    assert re.search(r"/home/[A-Za-z0-9_.-]+/", source) is None
    assert re.search(r"(?:10|192\.168)\.\d+\.\d+\.\d+", source) is None
    assert "environment.step(" not in source
    assert "sim.step(render=False)" in source
    assert "sim.render_mode = sim.RenderMode.NO_RENDERING" in source
    assert "sim.render_mode = production_render_mode" in source
    assert 'output_dir / "CORE_SCENE_PARTIAL.json"' in source
    assert '"physical_core_pass": physical_core_pass' in source
    assert '"task_complete": False' in source
    assert '"path_or_action_tuning_executed": False' in source
    assert "except SystemExit:" in source


def test_standalone_physics_smoke_is_fail_closed_and_render_free() -> None:
    source = (TASK_DIR / "laptop_physics_smoke.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = [
        (node.lineno, alias.name)
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    ] + [
        (node.lineno, node.module or "")
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    ]
    app_line = min(line for line, module in imports if module == "isaaclab.app")
    post_app_modules = ("pxr", "omni.physx", "isaaclab.sim")
    assert all(line > app_line for line, module in imports if module.startswith(post_app_modules))
    assert 'args.enable_cameras = False' in source
    assert 'args.device = "cpu"' in source
    assert 'device="cpu"' in source
    assert "sim.step(render=False)" in source
    assert 'UsdGeom.Cube.Define(stage, f"{TABLE_PATH}/Geometry")' in source
    assert "UsdPhysics.RigidBodyAPI.Apply(table.GetPrim())" not in source
    assert '"full_core_scene_pass": False' in source
    assert '"task_complete": False' in source
    assert '"path_or_action_tuning_executed": False' in source
    assert "environment.step(" not in source
    assert "omni.isaac." not in source
    assert "except SystemExit:" in source
