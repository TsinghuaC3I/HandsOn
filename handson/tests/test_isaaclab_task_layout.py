from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
import xml.etree.ElementTree as ET

import numpy as np
import pytest


_PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "handson"


def _load_task_preview_module(tmp_path: Path):
    module_name = "handson.sim.isaaclab._task_preview_layout_test"
    scene_module_name = "handson.sim.isaaclab.bimanual_assembly_scene"

    isaaclab = types.ModuleType("isaaclab")
    isaaclab_sim = types.ModuleType("isaaclab.sim")
    isaaclab.sim = isaaclab_sim

    scene = types.ModuleType(scene_module_name)
    scene._CACHE_DIR = tmp_path
    scene._XML_DIR = tmp_path
    scene._prepare_robot_mjcf = lambda *args, **kwargs: tmp_path / "robot.xml"
    scene._repair_imported_visual_materials = lambda *args, **kwargs: None
    scene.spawn_center_rig = lambda prim_path="/World/HandsOnRobots/CenterRig": prim_path

    source = _PACKAGE_ROOT / "sim" / "isaaclab" / "task_preview.py"
    spec = importlib.util.spec_from_file_location(module_name, source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    saved_modules = {
        name: sys.modules.get(name)
        for name in ("isaaclab", "isaaclab.sim", scene_module_name, module_name)
    }
    try:
        sys.modules["isaaclab"] = isaaclab
        sys.modules["isaaclab.sim"] = isaaclab_sim
        sys.modules[scene_module_name] = scene
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for name, previous in saved_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def test_collection_task_only_placeholder_contains_no_robot_physics(
    tmp_path: Path,
) -> None:
    task_preview = _load_task_preview_module(tmp_path)

    placeholder = task_preview._prepare_task_only_robot_placeholder(tmp_path)
    root = ET.parse(placeholder).getroot()

    assert root.attrib["model"] == "handson_collection_task_only"
    assert root.find("compiler") is None
    assert root.find("worldbody") is None
    assert root.find("actuator") is None
    assert root.find("sensor") is None
    assert task_preview.COLLECTION_TASK_ONLY_ROBOT_TASKS >= frozenset(
        {
            "t02_connect_charger_and_cable",
            "t12_load_and_start_microwave",
            "t17_open_latched_door",
            "p003_bimanual_folding_ruler_shape_configuration",
            "user_make_pizza_long_horizon",
            *task_preview.D_LONG_TASK_NAMES,


            "t06_fit_device_cover",
            "eval_flex02_fold_towel_twice",
            *task_preview.LW_TASK_NAMES,
            *task_preview.B320_TASK_NAMES,
        }
    )


def test_d07_first_physx_safe_scene_omits_imported_robot_pair(
    tmp_path: Path,
    monkeypatch,
) -> None:
    task_preview = _load_task_preview_module(tmp_path)
    task_preview._CACHE_DIR = tmp_path
    task_preview._XML_DIR = _PACKAGE_ROOT / "sim" / "envs" / "xmls"

    def fail_full_robot_build():
        raise AssertionError("D07 first-PhysX scene must not build imported robots")

    monkeypatch.setattr(
        task_preview,
        "_prepare_bimanual_robot_mjcf",
        fail_full_robot_build,
    )
    prepared = task_preview.prepare_task_mjcf(
        "d07_organize_office_desk",
        first_physx_safe=True,
    )
    root = ET.parse(prepared).getroot()
    body_names = {
        body.attrib.get("name", "")
        for body in root.iter("body")
    }
    joint_names = {
        joint.attrib.get("name", "")
        for joint in root.iter("joint")
    }

    assert {"d07_drawer", "d07_keyboard", "d07_mouse"} <= body_names
    assert {"base_left", "base_right"}.isdisjoint(body_names)
    assert {"joint1_left", "joint1_right"}.isdisjoint(joint_names)
    assert prepared.name == (
        "d07_organize_office_desk_scene_first_physx_safe_isaac.xml"
    )
    cache_dir = task_preview.task_usd_cache_dir(
        "d07_organize_office_desk",
        prepared,
        first_physx_safe=True,
    )
    assert cache_dir.name == "d07_organize_office_desk_first_physx_safe"


def test_bc004_d08_aligned_scene_has_an_isolated_usd_cache_namespace(
    tmp_path: Path,
) -> None:
    task_preview = _load_task_preview_module(tmp_path)
    task_preview._CACHE_DIR = tmp_path
    task_mjcf = tmp_path / "t10_lift_carton_scene_isaac.xml"
    task_mjcf.write_text("<mujoco/>", encoding="utf-8")

    cache_dir = task_preview.task_usd_cache_dir(
        "t10_lift_carton",
        task_mjcf,
        first_physx_safe=True,
    )

    assert task_preview._BC004_D08_ALIGNED_USD_CACHE_REVISION in cache_dir.name
    assert cache_dir.name.endswith("_first_physx_safe")


def test_d09_corrected_cabinet_has_an_isolated_usd_cache_namespace(
    tmp_path: Path,
) -> None:
    task_preview = _load_task_preview_module(tmp_path)
    task_preview._CACHE_DIR = tmp_path
    task_mjcf = tmp_path / "t31_prepare_coffee_scene_isaac.xml"
    task_mjcf.write_text("<mujoco/>", encoding="utf-8")

    cache_dir = task_preview.task_usd_cache_dir(
        "t31_prepare_coffee",
        task_mjcf,
        first_physx_safe=True,
    )

    assert task_preview._D09_CABINET_USD_CACHE_REVISION in cache_dir.name
    assert cache_dir.name.endswith("_first_physx_safe")


def test_imported_task_site_groups_are_hidden_without_disabling_prims(
    tmp_path: Path, monkeypatch
) -> None:
    task_preview = _load_task_preview_module(tmp_path)

    class FakePrim:
        def __init__(self, name: str, *, valid: bool = True) -> None:
            self.name = name
            self.valid = valid
            self.visibility = "inherited"
            self.transform = (1.0, 2.0, 3.0)
            self.prims = []

        def GetName(self) -> str:
            return self.name

        def IsValid(self) -> bool:
            return self.valid

    root = FakePrim("Task")
    body = FakePrim("cup")
    cup_sites = FakePrim("sites")
    lid_sites = FakePrim("sites")
    similarly_named_visual = FakePrim("sites_visual")
    root.prims = [root, body, cup_sites, lid_sites, similarly_named_visual]

    class FakeStage:
        def GetPrimAtPath(self, path: str):
            assert path == "/World/Task"
            return root

    class FakeImageable:
        def __init__(self, prim: FakePrim) -> None:
            self.prim = prim

        def __bool__(self) -> bool:
            return True

        def MakeInvisible(self) -> None:
            self.prim.visibility = "invisible"

    pxr = types.ModuleType("pxr")
    pxr.Usd = types.SimpleNamespace(PrimRange=lambda prim: prim.prims)
    pxr.UsdGeom = types.SimpleNamespace(Imageable=FakeImageable)
    monkeypatch.setitem(sys.modules, "pxr", pxr)

    original_transforms = {id(prim): prim.transform for prim in root.prims}
    hidden = task_preview._hide_imported_task_sites(FakeStage(), "/World/Task")

    assert hidden == 2
    assert cup_sites.visibility == "invisible"
    assert lid_sites.visibility == "invisible"
    assert body.visibility == "inherited"
    assert similarly_named_visual.visibility == "inherited"
    assert {id(prim): prim.transform for prim in root.prims} == original_transforms


def test_a01_layout_is_separated_inside_the_right_arm_workspace(tmp_path: Path) -> None:
    task_preview = _load_task_preview_module(tmp_path)
    root = ET.parse(
        _PACKAGE_ROOT / "sim" / "envs" / "xmls" / "a_series_ycb_task_objects.xml"
    ).getroot()

    task_preview._apply_ycb_task_layout(root, "t26_place_fruit_in_bucket")

    bodies = {body.attrib["name"]: body for body in root.iter("body")}
    assert set(bodies) == {"011_banana", "apple", "orange", "pear"}
    assert "spoon" not in bodies
    assert "a_marbles" not in bodies
    assert task_preview._YCB_TASK_LAYOUTS["t26_place_fruit_in_bucket"] == {
        "bucket": (-0.35, 0.08),
        "011_banana": (-0.37, -0.265),
        "apple": (-0.285, -0.265),
        "orange": (-0.37, -0.14),
        "pear": (-0.285, -0.08),
    }
    assert bodies["011_banana"].attrib["pos"] == "-0.370000 -0.265000 0.910000"
    assert bodies["apple"].attrib["pos"] == "-0.285000 -0.265000 0.928000"
    assert bodies["orange"].attrib["pos"] == "-0.370000 -0.140000 0.928000"
    assert bodies["pear"].attrib["pos"] == "-0.285000 -0.080000 0.942000"

    # A01 runtime reset uses per-object bounds inside this imported safe layout.
    max_jitter = 0.08
    for x, y in task_preview._YCB_TASK_LAYOUTS["t26_place_fruit_in_bucket"].values():
        assert -0.50 <= x - max_jitter <= x + max_jitter <= -0.03
        assert -0.43 <= y - max_jitter <= y + max_jitter <= 0.20


def test_a06_cup_starts_sideways_on_a_collision_facet(tmp_path: Path) -> None:
    task_preview = _load_task_preview_module(tmp_path)
    root = ET.parse(
        _PACKAGE_ROOT / "sim" / "envs" / "xmls" / "a_series_ycb_task_objects.xml"
    ).getroot()

    task_preview._apply_ycb_task_layout(root, "t19_orient_cup_upright")

    cup = root.find(".//body[@name='a_cup']")
    assert cup is not None
    assert cup.attrib["pos"] == "-0.300000 -0.060000 0.915122"
    quat_wxyz = np.fromstring(cup.attrib["quat"], sep=" ")
    expected = np.asarray(task_preview._A06_SIDE_REST_QUAT_WXYZ)
    np.testing.assert_allclose(quat_wxyz, expected, atol=1e-10)
    np.testing.assert_allclose(np.linalg.norm(quat_wxyz), 1.0, atol=1e-8)

    w, x, y, z = quat_wxyz
    rotation = np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    # Gravity points through the selected three-point support face in local
    # coordinates, rather than through the mouth rim alone.
    local_down = rotation.T @ np.asarray([0.0, 0.0, -1.0])
    np.testing.assert_allclose(
        local_down,
        np.asarray([0.02159739, -0.99467988, -0.10072482]),
        atol=1e-7,
    )

    collider_dir = (
        _PACKAGE_ROOT
        / "sim"
        / "envs"
        / "xmls"
        / "lightwheel_ycb"
        / "a_cups"
        / "collision"
    )
    vertices = []
    for collider in sorted(collider_dir.glob("*.obj")):
        for line in collider.read_text(encoding="utf-8").splitlines():
            if line.startswith("v "):
                vertices.append([float(value) for value in line.split()[1:4]])
    world_vertices = np.asarray(vertices) @ rotation.T
    minimum_z = float(np.min(world_vertices[:, 2]))
    support = world_vertices[
        np.isclose(world_vertices[:, 2], minimum_z, rtol=0.0, atol=1.0e-8)
    ]
    assert support.shape == (3, 3)
    np.testing.assert_allclose(
        task_preview._A06_SIDE_REST_Z_OFFSET + minimum_z,
        0.0005,
        atol=1.0e-9,
    )

    source_com = np.asarray([0.00013567, 0.00011084, -0.01130477])
    world_com = rotation @ source_com
    barycentric = np.linalg.solve(
        np.vstack([support[:, :2].T, np.ones(3)]),
        np.asarray([world_com[0], world_com[1], 1.0]),
    )
    # Gravity through the source center of mass lands comfortably inside the
    # actual three-collider support triangle, not merely on its boundary.
    assert np.min(barycentric) > 0.15

    # The opening axis is tilted only enough to put both ends on the table; it
    # remains far outside the task's upright success cone.
    local_z_world_z = 1.0 - 2.0 * (x * x + y * y)
    assert 0.09 < local_z_world_z < 0.11
    assert local_z_world_z < np.cos(np.deg2rad(25.0))


def test_a02_plate_is_a_resettable_free_body() -> None:
    root = ET.parse(
        _PACKAGE_ROOT / "sim" / "envs" / "xmls" / "t25_place_cup_on_plate_plate.xml"
    ).getroot()

    plate = root.find(".//body[@name='plate']")
    assert plate is not None
    assert plate.find("freejoint").attrib["name"] == "plate_joint"
    collision = plate.find("geom[@name='plate_collision']")
    assert collision is not None
    assert collision.attrib["mass"] == "0.18"


def test_a02_filter_keeps_only_task_mug_and_plate(tmp_path: Path) -> None:
    task_preview = _load_task_preview_module(tmp_path)
    root = ET.parse(
        _PACKAGE_ROOT / "sim" / "envs" / "xmls" / "t25_place_cup_on_plate_objects.xml"
    ).getroot()
    sensor = ET.SubElement(root, "sensor")
    ET.SubElement(sensor, "framepos", {"objname": "c_cups"})
    ET.SubElement(sensor, "framepos", {"objname": "025_mug"})

    removed = task_preview._apply_a02_task_filter(root, "t25_place_cup_on_plate")

    bodies = {body.attrib["name"] for body in root.iter("body")}
    assert removed == 4
    assert bodies == {"025_mug"}
    assert [element.attrib["objname"] for element in sensor] == ["025_mug"]


def test_a02_filter_does_not_change_other_tasks(tmp_path: Path) -> None:
    task_preview = _load_task_preview_module(tmp_path)
    root = ET.parse(
        _PACKAGE_ROOT / "sim" / "envs" / "xmls" / "t25_place_cup_on_plate_objects.xml"
    ).getroot()

    assert task_preview._apply_a02_task_filter(root, "t24_place_ball_in_bowl") == 0
    assert len(root.findall(".//body")) == 5


def test_c06_isaac_reset_places_camera_flat_on_table(tmp_path: Path) -> None:
    task_preview = _load_task_preview_module(tmp_path)
    root = ET.parse(
        _PACKAGE_ROOT / "sim" / "envs" / "xmls" / "camera.xml"
    ).getroot()

    assert task_preview._apply_c06_stable_camera_reset_pose(
        root, "t23_photograph_logo"
    ) == 1
    camera = root.find(".//body[@name='link_19']")
    assert camera is not None
    assert camera.attrib["pos"].split()[2] == "0.955000"
    assert camera.attrib["quat"] == "1 0 0 0"


def test_c12_isaac_objects_start_in_direct_table_contact(tmp_path: Path) -> None:
    task_preview = _load_task_preview_module(tmp_path)
    positions = task_preview._TASK_BODY_POSITION_OVERRIDES[
        "t11_line_up_blocks"
    ]

    assert positions["c12_set_square"][2] == 0.9000
    for name in ("c12_block_0", "c12_block_1", "c12_block_2"):
        assert positions[name][2] == 0.9115


def test_b04_small_blocks_use_exact_box_collisions(tmp_path: Path) -> None:
    task_preview = _load_task_preview_module(tmp_path)
    root = ET.parse(
        _PACKAGE_ROOT / "sim" / "envs" / "xmls" / "a_series_ycb_task_objects.xml"
    ).getroot()

    assert task_preview._apply_b04_analytic_small_block_collisions(
        root, "t44_stack_blocks"
    ) == 3
    for name, half_extent in task_preview._B04_SMALL_BLOCK_HALF_EXTENTS.items():
        collision = root.find(
            f".//body[@name='{name}']/geom[@name='{name}_collision_1']"
        )
        assert collision is not None
        assert collision.attrib["type"] == "box"
        assert "mesh" not in collision.attrib
        expected = f"{half_extent:.6f} {half_extent:.6f} {half_extent:.6f}"
        assert collision.attrib["size"] == expected


def test_b04_drops_black_visual_collision_but_keeps_stable_collision(
    tmp_path: Path,
) -> None:
    task_preview = _load_task_preview_module(tmp_path)
    root = ET.parse(
        _PACKAGE_ROOT / "sim" / "envs" / "xmls" / "a_series_ycb_task_objects.xml"
    ).getroot()

    assert task_preview._remove_b04_wood_block_visual_collision(
        root, "t44_stack_blocks"
    ) == 1
    wood_body = root.find(".//body[@name='036_wood_block']")
    assert wood_body is not None
    assert wood_body.find("geom[@name='036_wood_block_collision_1']") is None
    assert wood_body.find("geom[@name='036_wood_block_visual']") is not None
    assert wood_body.find("geom[@name='036_wood_block_stable_collision']") is not None


def test_a01_scanned_fruit_hulls_become_stable_analytic_collisions(
    tmp_path: Path,
) -> None:
    task_preview = _load_task_preview_module(tmp_path)
    root = ET.parse(
        _PACKAGE_ROOT / "sim" / "envs" / "xmls" / "a_series_ycb_task_objects.xml"
    ).getroot()
    # The production full scene includes bucket.xml separately.  Add its root
    # here so this focused fragment test also proves the explicit source
    # inertial is installed for that imported free body.
    worldbody = root.find("worldbody")
    assert worldbody is not None
    ET.SubElement(worldbody, "body", {"name": "bucket"})

    assert task_preview._apply_a01_analytic_fruit_collisions(
        root, "t26_place_fruit_in_bucket"
    ) == 4
    expected = {
        "011_banana": (
            "box",
            "0.0333585 0.0973900 0.019722",
            "0.07",
            (
                0.00023038795229333336,
                0.0000350407588125,
                0.0002472773711858334,
            ),
        ),
        "apple": (
            "sphere",
            "0.038",
            "0.10",
            (0.00005776, 0.00005776, 0.00005776),
        ),
        "orange": (
            "sphere",
            "0.037",
            "0.11",
            (0.000060236, 0.000060236, 0.000060236),
        ),
        "pear": (
            "box",
            "0.028 0.028 0.052",
            "0.11",
            (
                0.0001278933333333333,
                0.0001278933333333333,
                0.00005749333333333334,
            ),
        ),
    }
    for name, (shape_type, size, mass, diagonal_inertia) in expected.items():
        body = root.find(f".//body[@name='{name}']")
        assert body is not None
        collisions = root.findall(f".//body[@name='{name}']/geom")
        collisions = [
            geom
            for geom in collisions
            if geom.attrib.get("name", "").startswith(f"{name}_collision_")
        ]
        assert collisions[0].attrib["type"] == shape_type
        assert collisions[0].attrib["size"] == size
        assert collisions[0].attrib["mass"] == mass
        assert "mesh" not in collisions[0].attrib
        assert len(collisions) == 1

        inertials = body.findall("inertial")
        assert len(inertials) == 1
        inertial = inertials[0]
        assert inertial.attrib["mass"] == mass
        assert inertial.attrib["pos"] == "0 0 0"
        assert inertial.attrib["quat"] == "1 0 0 0"
        np.testing.assert_allclose(
            [
                float(value)
                for value in inertial.attrib["diaginertia"].split()
            ],
            diagonal_inertia,
            rtol=0.0,
            atol=0.0,
        )

    bucket = root.find(".//body[@name='bucket']")
    assert bucket is not None
    bucket_inertials = bucket.findall("inertial")
    assert len(bucket_inertials) == 1
    assert bucket_inertials[0].attrib == (
        task_preview._A01_ANALYTIC_FRUIT_INERTIALS["bucket"]
    )


def test_a01_apple_obj_uvs_match_the_source_texture_bottom_left_origin() -> None:
    from PIL import Image

    visual_dir = (
        _PACKAGE_ROOT
        / "sim"
        / "envs"
        / "xmls"
        / "lightwheel_ycb"
        / "apple"
        / "visuals"
    )
    uv = np.asarray(
        [
            [float(value) for value in line.split()[1:3]]
            for line in (visual_dir / "apple.obj").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.startswith("vt ")
        ]
    )
    pixels = np.asarray(
        Image.open(visual_dir / "T_apple_BC001.png").convert("RGB")
    )
    height, width = pixels.shape[:2]
    x = np.clip(np.rint(uv[:, 0] * (width - 1)).astype(int), 0, width - 1)
    bottom_left_y = np.clip(
        np.rint((1.0 - uv[:, 1]) * (height - 1)).astype(int),
        0,
        height - 1,
    )
    incorrectly_flipped_y = np.clip(
        np.rint(uv[:, 1] * (height - 1)).astype(int),
        0,
        height - 1,
    )
    source_samples = pixels[bottom_left_y, x].mean(axis=1)
    flipped_samples = pixels[incorrectly_flipped_y, x].mean(axis=1)

    # Every authored bottom-left OBJ UV lands on scanned apple color.  Isaac's
    # OBJ import already accounts for that convention; manually flipping the
    # atlas sends most vertices into its black padding.
    assert np.all(source_samples > 10.0)
    assert np.mean(flipped_samples > 10.0) < 0.40


def test_a03_balls_and_bowl_start_inside_the_recording_workspace(
    tmp_path: Path,
) -> None:
    task_preview = _load_task_preview_module(tmp_path)
    root = ET.parse(
        _PACKAGE_ROOT / "sim" / "envs" / "xmls" / "ball_to_bowl_objects.xml"
    ).getroot()

    task_preview._apply_task_body_position_overrides(root, "t24_place_ball_in_bowl")

    bodies = {body.attrib["name"]: body for body in root.iter("body")}
    assert bodies["024_bowl"].attrib["pos"] == "-0.220000 0.140000 0.920000"
    assert bodies["tennis_ball"].attrib["pos"] == "-0.380000 -0.240000 0.954000"
    assert bodies["golf_ball"].attrib["pos"] == "-0.300000 -0.180000 0.941000"
    assert (
        bodies["mini_soccer_ball"].attrib["pos"]
        == "-0.200000 -0.100000 0.958000"
    )
    assert all(
        -0.40 <= float(bodies[name].attrib["pos"].split()[0]) <= -0.12
        and -0.30 <= float(bodies[name].attrib["pos"].split()[1]) <= 0.30
        for name in ("tennis_ball", "golf_ball", "mini_soccer_ball")
    )
    expected_ball_radii = {
        "tennis_ball": "0.034",
        "golf_ball": "0.022",
        "mini_soccer_ball": "0.038",
    }
    for name, radius in expected_ball_radii.items():
        collision = bodies[name].find(f"geom[@name='{name}_collision_1']")
        assert collision is not None
        assert collision.attrib["type"] == "sphere"
        assert collision.attrib["size"] == radius
        assert "mesh" not in collision.attrib


def test_lid_tasks_use_the_rokae_reachable_layouts(tmp_path: Path) -> None:
    task_preview = _load_task_preview_module(tmp_path)
    expected = {
        "t04_cover_pitcher": {
            "019_pitcher_base": "-0.300000 0.100000 1.008000",
            "pitcher_lid": "-0.340000 -0.200000 0.914000",
        },
        "t03_cover_box": {
            "Clear_Box": "-0.200000 0.150000 0.964500",
            "Box_lid": "-0.360000 0.150000 0.900000",
        },
    }

    for task_name, positions in expected.items():
        root = ET.parse(
            _PACKAGE_ROOT
            / "sim"
            / "envs"
            / "xmls"
            / "a_series_ycb_task_objects.xml"
        ).getroot()
        task_preview._apply_ycb_task_layout(root, task_name)
        bodies = {body.attrib["name"]: body for body in root.iter("body")}
        for name, position in positions.items():
            assert bodies[name].attrib["pos"] == position


def test_b01_isaac_lid_insert_collider_fits_inside_the_pitcher_mouth(
    tmp_path: Path,
) -> None:
    task_preview = _load_task_preview_module(tmp_path)
    root = ET.parse(
        _PACKAGE_ROOT
        / "sim"
        / "envs"
        / "xmls"
        / "a_series_ycb_task_objects.xml"
    ).getroot()

    task_preview._apply_04_cover_pitcher_collision_repair(root, "t04_cover_pitcher")

    bodies = {body.attrib["name"]: body for body in root.iter("body")}
    base_geoms = {
        geom.attrib["name"]: geom
        for geom in bodies["019_pitcher_base"].findall("geom")
        if "name" in geom.attrib
    }
    lid_geoms = {
        geom.attrib["name"]: geom
        for geom in bodies["pitcher_lid"].findall("geom")
        if "name" in geom.attrib
    }

    assert base_geoms["pitcher_base_lid_seat"].attrib["contype"] == "0"
    assert base_geoms["pitcher_base_lid_seat"].attrib["conaffinity"] == "0"
    assert lid_geoms["pitcher_lid_collision_1"].attrib["contype"] == "0"
    assert lid_geoms["pitcher_lid_collision_1"].attrib["conaffinity"] == "0"
    assert lid_geoms["pitcher_lid_bottom_seat"].attrib["type"] == "cylinder"
    assert lid_geoms["pitcher_lid_bottom_seat"].attrib["size"] == "0.043 0.010"
    assert lid_geoms["pitcher_lid_bottom_seat"].attrib["pos"] == "0 0 -0.0135045"


def test_reported_assembly_objects_are_moved_out_from_under_the_left_hand(
    tmp_path: Path,
) -> None:
    task_preview = _load_task_preview_module(tmp_path)
    cases = (

        (
            "t09_insert_peg",
            "industreal_round_peg_8mm.xml",
            {
                "industreal_round_peg_8mm": "-0.300000 -0.420000 0.940000",
            },
        ),
        (
            "t09_insert_peg",
            "industreal_tray_insert_round_peg_8mm.xml",
            {
                "industreal_tray_insert_round_peg_8mm": (
                    "-0.100000 0.000000 0.920000"
                ),
            },
        ),
    )

    for task_name, source_name, expected in cases:
        root = ET.parse(
            _PACKAGE_ROOT / "sim" / "envs" / "xmls" / source_name
        ).getroot()
        task_preview._apply_task_body_position_overrides(root, task_name)
        bodies = {body.attrib["name"]: body for body in root.iter("body")}
        for name, position in expected.items():
            assert bodies[name].attrib["pos"] == position


def test_other_single_arm_default_assets_are_also_brought_into_reach(
    tmp_path: Path,
) -> None:
    task_preview = _load_task_preview_module(tmp_path)
    cases = (
        (
            "t25_place_cup_on_plate",
            "t25_place_cup_on_plate_objects.xml",
            {
                "025_mug": "-0.340000 -0.200000 0.963765",
            },
        ),
        (
            "t50_water_plant",
            "spray.xml",
            {
                "link_2": "-0.300000 -0.220000 0.955000",
            },
        ),
        (
            "t14_move_and_click_mouse",
            "mouse_nolight.xml",
            {
                "link_3": "-0.380000 -0.240000 0.892000",
            },
        ),
        (
            "t14_move_and_click_mouse",
            "display.xml",
            {
                "display_link_1": "0.100000 0.000000 1.160000",
            },
        ),
    )

    for task_name, source_name, expected in cases:
        root = ET.parse(
            _PACKAGE_ROOT / "sim" / "envs" / "xmls" / source_name
        ).getroot()
        task_preview._apply_task_body_position_overrides(root, task_name)
        bodies = {body.attrib["name"]: body for body in root.iter("body")}
        for body_name, position in expected.items():
            assert bodies[body_name].attrib["pos"] == position


def test_c04_plant_uses_mujoco_water_plant_source() -> None:
    root = ET.parse(
        _PACKAGE_ROOT / "sim" / "envs" / "xmls" / "plant.xml"
    ).getroot()
    materials = {material.attrib["name"]: material for material in root.findall(".//material")}
    meshes = {mesh.attrib["name"]: mesh for mesh in root.findall(".//mesh") if "name" in mesh.attrib}
    textures = {texture.attrib["name"]: texture for texture in root.findall(".//texture")}
    default_geoms = {
        default.attrib["class"]: default.find("geom")
        for default in root.findall(".//default")
        if "class" in default.attrib and default.find("geom") is not None
    }

    assert "model_1_vis" in meshes
    assert "blinn1SG" in materials
    assert "image0" in textures
    assert {
        geom.attrib.get("mesh")
        for geom in root.iter("geom")
        if geom.attrib.get("class") == "plant_visual"
        and geom.attrib.get("type") == "mesh"
    } == {
        "model_0_vis",
        "model_1_vis",
        "model_2_vis",
        "model_3_vis",
    }

    target_regions = [
        geom
        for geom in root.iter("geom")
        if geom.attrib.get("class") == "plant_visual"
        and geom.attrib.get("type") == "cylinder"
        and geom.attrib.get("group") == "5"
    ]
    assert len(target_regions) == 1
    target_region = target_regions[0]
    assert target_region.attrib["rgba"] == "0 0.5 0 0.5"
    assert target_region.attrib["size"] == "0.2 0.2"
    assert "name" not in target_region.attrib
    assert default_geoms["plant_collision"].attrib["rgba"] == "0.5 0 0 0.5"


def test_c04_isaac_makes_leaf_texture_backgrounds_transparent(tmp_path: Path) -> None:
    task_preview = _load_task_preview_module(tmp_path)
    from PIL import Image

    for file_name in ("image0.png", "image1.png", "image2.png"):
        image = Image.new("RGB", (2, 1))
        image.putpixel((0, 0), (255, 255, 255))
        image.putpixel((1, 0), (20, 140, 30))
        image.save(tmp_path / file_name)

    root = ET.Element("mujoco")
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", {"name": "image0", "file": "image0.png"})
    ET.SubElement(asset, "texture", {"name": "image1", "file": "image1.png"})
    ET.SubElement(asset, "texture", {"name": "image2", "file": "image2.png"})

    changed = task_preview._apply_c04_plant_texture_alpha(
        root,
        tmp_path,
        "t50_water_plant",
    )

    textures = {
        texture.attrib["name"]: texture.attrib["file"]
        for texture in root.findall(".//texture")
    }
    assert changed == 2
    assert textures["image0"].endswith("_isaac_alpha.png")
    assert textures["image1"].endswith("_isaac_alpha.png")
    assert textures["image2"] == "image2.png"

    alpha_image = Image.open(tmp_path / textures["image0"])
    assert alpha_image.getpixel((0, 0))[3] == 0
    assert alpha_image.getpixel((1, 0))[3] == 255


def test_c04_isaac_removes_only_the_mujoco_hidden_target_region(tmp_path: Path) -> None:
    task_preview = _load_task_preview_module(tmp_path)
    root = ET.parse(
        _PACKAGE_ROOT / "sim" / "envs" / "xmls" / "plant.xml"
    ).getroot()

    removed = task_preview._remove_c04_isaac_hidden_visuals(
        root,
        "t50_water_plant",
    )

    plant_geoms = list(next(body for body in root.iter("body") if body.attrib.get("name") == "plant"))
    assert removed == 1
    assert not any(
        geom.tag == "geom"
        and geom.attrib.get("class") == "plant_visual"
        and geom.attrib.get("type") == "cylinder"
        and geom.attrib.get("group") == "5"
        for geom in plant_geoms
    )
    assert any(
        geom.tag == "geom"
        and geom.attrib.get("class") == "plant_collision"
        and geom.attrib.get("size") == "0.1 0.15"
        for geom in plant_geoms
    )
    assert {
        geom.attrib.get("mesh")
        for geom in plant_geoms
        if geom.tag == "geom" and geom.attrib.get("type") == "mesh"
    } == {"model_0_vis", "model_1_vis", "model_2_vis", "model_3_vis"}


def test_mocap_target_helper_geoms_are_removed_without_removing_bodies(
    tmp_path: Path,
) -> None:
    task_preview = _load_task_preview_module(tmp_path)
    root = ET.fromstring(
        """
        <mujoco>
          <worldbody>
            <body name="target" mocap="true">
              <geom name="target" type="box" contype="0" conaffinity="0"/>
              <site name="target_site"/>
            </body>
            <body name="target_right" mocap="true">
              <inertial pos="0 0 0" mass="1e-6" diaginertia="1e-12 1e-12 1e-12"/>
              <geom name="target_right" type="box" contype="0" conaffinity="0"/>
            </body>
            <body name="target_left" mocap="true">
              <geom name="target_left" type="box" contype="0" conaffinity="0"/>
              <geom name="left_debug_collision" type="sphere"/>
            </body>
            <body name="target_box" mocap="true">
              <geom name="target" type="box" contype="0" conaffinity="0"/>
            </body>
            <body name="target_non_mocap">
              <geom name="target_right" type="box" contype="0" conaffinity="0"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )

    removed = task_preview._remove_mocap_target_helper_geoms(root)

    assert removed == 3
    assert {
        body.attrib["name"]
        for body in root.iter("body")
        if body.attrib.get("mocap") == "true"
    } == {"target", "target_right", "target_left", "target_box"}
    named_geoms = {
        geom.attrib["name"]
        for geom in root.iter("geom")
        if geom.attrib.get("name")
    }
    assert named_geoms == {"left_debug_collision", "target", "target_right"}


def test_all_isaac_task_scene_mocap_target_helpers_can_be_stripped(
    tmp_path: Path,
) -> None:
    task_preview = _load_task_preview_module(tmp_path)
    helper_names = task_preview._MOCAP_TARGET_HELPER_GEOMS

    for spec in task_preview.TASK_SPECS.values():
        root = ET.parse(
            _PACKAGE_ROOT / "sim" / "envs" / "xmls" / spec.scene_xml
        ).getroot()
        task_preview._remove_mocap_target_helper_geoms(root)
        for body in root.iter("body"):
            body_name = body.attrib.get("name", "")
            if body.attrib.get("mocap") != "true" or body_name not in helper_names:
                continue
            assert not any(
                geom.tag == "geom" and geom.attrib.get("name") == body_name
                for geom in body
            ), spec.task_name


def test_c06_isaac_removes_only_the_green_target_region_visual(
    tmp_path: Path,
) -> None:
    task_preview = _load_task_preview_module(tmp_path)
    root = ET.fromstring(
        """
        <mujoco>
          <worldbody>
            <body name="table">
              <geom name="photograph_target_region_visual"
                    contype="0" conaffinity="0"/>
              <geom name="photograph_logo_disk"
                    contype="0" conaffinity="0"/>
              <geom name="photograph_target_region_visual_collision"
                    contype="1" conaffinity="1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )

    removed = task_preview._remove_c06_target_region_visual(
        root,
        "t23_photograph_logo",
    )

    assert removed == 1
    assert {
        geom.attrib["name"]
        for geom in root.iter("geom")
    } == {
        "photograph_logo_disk",
        "photograph_target_region_visual_collision",
    }


def test_c06_target_region_removal_does_not_touch_other_tasks(
    tmp_path: Path,
) -> None:
    task_preview = _load_task_preview_module(tmp_path)
    root = ET.fromstring(
        """
        <mujoco>
          <worldbody>
            <geom name="photograph_target_region_visual"
                  contype="0" conaffinity="0"/>
          </worldbody>
        </mujoco>
        """
    )

    assert task_preview._remove_c06_target_region_visual(
        root,
        "t25_place_cup_on_plate",
    ) == 0
    assert [
        geom.attrib["name"]
        for geom in root.iter("geom")
    ] == ["photograph_target_region_visual"]


def test_c_tool_scene_filter_keeps_only_current_task_bodies(
    tmp_path: Path,
) -> None:
    task_preview = _load_task_preview_module(tmp_path)
    root = ET.parse(
        _PACKAGE_ROOT / "sim" / "envs" / "xmls" / "c_tool_interaction_objects.xml"
    ).getroot()

    removed = task_preview._apply_c_tool_task_filter(
        root,
        "c08_tongs_ice_cubes",
    )

    bodies = {body.attrib.get("name") for body in root.iter("body")}
    assert removed > 0
    assert {"c08_tongs", "c08_glass_cup", "c08_ice_0", "c08_ice_3"} <= bodies
    assert "c08_bowl" not in bodies
    assert "c07_pitcher" not in bodies
    assert "c09_stapler" not in bodies
    assert "c12_set_square" not in bodies


def test_interrupted_task_usd_cache_is_rejected(tmp_path: Path) -> None:
    task_preview = _load_task_preview_module(tmp_path)
    task_mjcf = tmp_path / "c08_tongs_ice_cubes_scene_isaac.xml"
    usd_dir = tmp_path / "usd" / "tasks" / "v87" / "c08_tongs_ice_cubes"
    configuration = usd_dir / "configuration"
    configuration.mkdir(parents=True)
    (usd_dir / ".asset_hash").write_text("interrupted")
    (usd_dir / f"{task_mjcf.stem}.usd").write_bytes(b"x" * 492)
    (configuration / f"{task_mjcf.stem}_base.usd").write_bytes(b"x" * 492)

    assert task_preview._task_usd_cache_is_incomplete(usd_dir, task_mjcf)

    (usd_dir / f"{task_mjcf.stem}.usd").write_bytes(
        b"x" * task_preview._TASK_USD_MIN_ROOT_BYTES
    )
    (configuration / f"{task_mjcf.stem}_base.usd").write_bytes(
        b"x" * task_preview._TASK_USD_MIN_BASE_BYTES
    )
    assert not task_preview._task_usd_cache_is_incomplete(usd_dir, task_mjcf)


def test_d11_task_mjcf_contract_rejects_the_legacy_layout(tmp_path: Path) -> None:
    task_preview = _load_task_preview_module(tmp_path)
    source = (
        _PACKAGE_ROOT
        / "sim"
        / "envs"
        / "xmls"
        / "d_long_horizon_objects.xml"
    )

    current_root = ET.parse(source).getroot()
    task_preview._validate_d11_task_mjcf_contract(
        current_root,
        source_path=source,
    )
    task_preview._validate_dlong_task_mjcf_contract(
        current_root,
        "d11_presoak_pan",
        source_path=source,
    )

    legacy_root = ET.fromstring(ET.tostring(current_root))
    pan = next(
        body for body in legacy_root.iter("body")
        if body.attrib.get("name") == "d11_pan"
    )
    pan.attrib["pos"] = "-.28 -.48 .925"
    visual = pan.find("geom[@name='d11_pan_visual']")
    assert visual is not None
    visual.attrib.update({"type": "box"})
    visual.attrib.pop("mesh", None)

    with pytest.raises(ValueError, match="D11 task MJCF contract failed"):
        task_preview._validate_d11_task_mjcf_contract(
            legacy_root,
            source_path=tmp_path / "legacy_d11.xml",
        )

    wrong_units_root = ET.fromstring(ET.tostring(current_root))
    compiler = wrong_units_root.find("compiler")
    assert compiler is not None
    compiler.attrib["angle"] = "degree"
    with pytest.raises(ValueError, match="D-long task MJCF contract failed"):
        task_preview._validate_dlong_task_mjcf_contract(
            wrong_units_root,
            "d11_presoak_pan",
            source_path=tmp_path / "degree_d11.xml",
        )


@pytest.mark.parametrize(
    ("task_name", "expected_ranges"),
    (
        (
            "t31_prepare_coffee",
            {
                "d09_cabinet_door_joint": (-1.45, 0.0),
                "d09_cabinet_right_door_joint": (0.0, 1.45),
            },
        ),
        (
            "d10_load_dishwasher",
            {
                "d10_door_joint": (-1.45, 0.0),
                "d10_rack_joint": (-0.20, 0.0),
            },
        ),
    ),
)
def test_dlong_task_only_mjcf_preserves_radian_joint_ranges(
    tmp_path: Path,
    task_name: str,
    expected_ranges: dict[str, tuple[float, float]],
) -> None:
    mujoco = pytest.importorskip("mujoco")
    task_preview = _load_task_preview_module(tmp_path / "loader")
    task_preview._CACHE_DIR = tmp_path / "cache"
    task_preview._XML_DIR = _PACKAGE_ROOT / "sim" / "envs" / "xmls"

    prepared = task_preview.prepare_task_mjcf(
        task_name,
        first_physx_safe=True,
    )
    root = ET.parse(prepared).getroot()
    compiler = root.find("compiler")
    assert compiler is not None
    assert compiler.attrib.get("angle") == "radian"

    joints = {j.get("name"): j for j in root.iter("joint")}
    for joint_name, expected in expected_ranges.items():
        np.testing.assert_allclose(np.fromstring(joints[joint_name].get("range"), sep=" "), expected)


def test_b03_task_mjcf_and_usd_key_are_stable_across_cache_roots(
    tmp_path: Path,
) -> None:
    task_preview = _load_task_preview_module(tmp_path / "loader")
    source_xml_dir = _PACKAGE_ROOT / "sim" / "envs" / "xmls"
    task_preview._XML_DIR = source_xml_dir

    def generate(cache_dir: Path) -> tuple[Path, Path]:
        task_preview._CACHE_DIR = cache_dir

        def prepare_robot(side: str, **_kwargs) -> Path:
            output = cache_dir / "mjcf" / f"test_robot_{side}.xml"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                "<mujoco><worldbody><body name='test_robot_"
                f"{side}'><geom type='sphere' size='0.01'/></body>"
                "</worldbody></mujoco>",
                encoding="utf-8",
            )
            return output

        task_preview._prepare_robot_mjcf = prepare_robot
        task_mjcf = task_preview.prepare_task_mjcf("t09_insert_peg")
        usd_dir = task_preview.task_usd_cache_dir(
            "t09_insert_peg",
            task_mjcf,
        )
        return task_mjcf, usd_dir

    first_mjcf, first_usd = generate(tmp_path / "cache-a")
    first_payload = first_mjcf.read_bytes()
    # Reconstructing the same cache must be a byte-for-byte no-op.
    repeated_mjcf, repeated_usd = generate(tmp_path / "cache-a")
    second_mjcf, second_usd = generate(
        tmp_path / "different" / "cache" / "depth"
    )

    assert repeated_mjcf == first_mjcf
    assert repeated_usd == first_usd
    assert repeated_mjcf.read_bytes() == first_payload
    assert second_mjcf.read_bytes() == first_payload
    assert first_usd.name == second_usd.name
    assert len(first_usd.name) == 64

    root = ET.parse(second_mjcf).getroot()
    peg_socket_meshes = [
        mesh.attrib["file"]
        for mesh in root.iter("mesh")
        if mesh.attrib.get("name")
        in {
            "industreal_round_peg_8mm_mesh",
            "industreal_tray_insert_round_peg_8mm_mesh",
        }
    ]
    texture_references = [
        texture.attrib["file"]
        for texture in root.iter("texture")
        if texture.attrib.get("file")
    ]
    assert len(peg_socket_meshes) == 2
    assert len(texture_references) == 3
    assert all(
        reference.startswith("task_assets/")
        and ".." not in Path(reference).parts
        and not Path(reference).is_absolute()
        for reference in peg_socket_meshes + texture_references
    )


def test_b03_task_usd_cache_key_changes_with_referenced_asset_bytes(
    tmp_path: Path,
) -> None:
    task_preview = _load_task_preview_module(tmp_path / "loader")
    asset = tmp_path / "asset.obj"
    asset.write_bytes(b"mesh-v1")
    task_mjcf = tmp_path / "t09_insert_peg_scene_isaac.xml"
    task_mjcf.write_text(
        '<mujoco><asset><mesh name="peg" file="asset.obj"/></asset></mujoco>',
        encoding="utf-8",
    )

    baseline = task_preview.task_usd_cache_dir(
        "t09_insert_peg",
        task_mjcf,
    )
    asset.write_bytes(b"mesh-v2")
    changed = task_preview.task_usd_cache_dir(
        "t09_insert_peg",
        task_mjcf,
    )

    assert baseline != changed
    assert baseline.parent == changed.parent


def test_b03_synced_content_cache_needs_no_converter_metadata(
    tmp_path: Path,
) -> None:
    task_preview = _load_task_preview_module(tmp_path / "loader")
    task_preview.is_lab3 = lambda: True
    task_mjcf = tmp_path / "t09_insert_peg_scene_isaac.xml"
    task_mjcf.write_text("<mujoco/>", encoding="utf-8")
    usd_dir = tmp_path / ("a" * 64)
    asset_dir = usd_dir / task_mjcf.stem
    payloads = asset_dir / "payloads"
    payloads.mkdir(parents=True)
    (asset_dir / f"{task_mjcf.stem}.usda").write_bytes(
        b"x" * task_preview._TASK_USD_MIN_ROOT_BYTES
    )
    (payloads / "base.usda").write_bytes(
        b"x" * task_preview._TASK_USD_MIN_BASE_BYTES
    )

    # Exact USD trees can be synced to another host without config.yaml or
    # .asset_hash; the directory key already binds the complete source bytes.
    assert not task_preview._task_usd_cache_is_incomplete(usd_dir, task_mjcf)

    (usd_dir / f"{task_mjcf.stem}_1").mkdir()
    assert task_preview._task_usd_cache_is_incomplete(usd_dir, task_mjcf)


def test_c_tool_scene_xml_parses_and_declares_all_new_task_roots(
    tmp_path: Path,
) -> None:
    task_preview = _load_task_preview_module(tmp_path)
    xml_path = _PACKAGE_ROOT / "sim" / "envs" / "xmls" / "c_tool_interaction_objects.xml"
    root = ET.parse(xml_path).getroot()
    bodies = {body.attrib.get("name") for body in root.iter("body")}

    for task_name, expected_bodies in task_preview._C_TOOL_TASK_BODIES.items():
        assert set(expected_bodies) <= bodies, task_name

    assets = {mesh.attrib.get("name") for mesh in root.find("asset").findall("mesh")}
    assert "c08_tongs_fixed_shell" not in assets
    assert "c08_bowl_ycb" not in assets
    assert "c08_glass_cup_ycb" in assets
    assert "c12_wood_block_ycb" in assets

    c08_top_level = {
        body.attrib.get("name")
        for body in root.find("worldbody").findall("body")
        if body.attrib.get("name", "").startswith("c08_")
    }
    assert c08_top_level == {
        "c08_tongs",
        "c08_tongs_cradle",
        "c08_glass_cup",
        "c08_ice_0",
        "c08_ice_1",
        "c08_ice_2",
        "c08_ice_3",
    }
    assert root.find(".//body[@name='c09_stapler_top']/joint[@name='c09_stapler_top_joint']") is not None
    stapler_base = root.find(".//geom[@name='c09_stapler_base']")
    stapler_top = root.find(".//geom[@name='c09_stapler_top_geom']")
    assert stapler_base is not None
    assert stapler_top is not None
    assert stapler_base.attrib["class"] == "ctool_collision"
    assert stapler_base.attrib["rgba"] == "0 0 0 0"
    assert stapler_top.attrib["class"] == "ctool_collision"
    assert stapler_top.attrib["rgba"] == "0 0 0 0"
    assert root.find(".//geom[@name='c09_stapler_top_cover']").attrib["type"] == "capsule"
    assert root.find(".//geom[@name='c09_stapler_base_shell']") is None
    assert root.find(".//geom[@name='c09_stapler_top_shell']") is None
    for paper_index in range(2):
        grip_tab = root.find(
            f".//geom[@name='c09_paper_{paper_index}_grip_tab']"
        )
        assert grip_tab is not None
        expected_tab_position = (
            "-0.060 -0.038 0.006"
            if paper_index == 0
            else "-0.060 0.038 0.006"
        )
        assert grip_tab.attrib["pos"] == expected_tab_position
        # The colored tab is a visible software pinch affordance.  It must not
        # collide with the other sheet when the papers are stacked.
        assert grip_tab.attrib["class"] == "ctool_visual"
        assert "mass" not in grip_tab.attrib
        assert grip_tab.attrib.get("material") in {"ctool_red", "ctool_blue"}

        grip_tab_collision = root.find(
            f".//geom[@name='c09_paper_{paper_index}_grip_tab_collision']"
        )
        assert grip_tab_collision is not None
        assert grip_tab_collision.attrib["pos"] == expected_tab_position
        assert grip_tab_collision.attrib["size"] == grip_tab.attrib["size"]
        assert grip_tab_collision.attrib["class"] == "ctool_collision"
        assert float(grip_tab_collision.attrib["mass"]) > 0.0

        corner = root.find(
            f".//geom[@name='c09_paper_{paper_index}_corner']"
        )
        assert corner is not None
        assert corner.attrib["pos"] == "0.060 0.038 0.0025"

    # The two raised, visible grip tabs occupy opposite paper corners, so a
    # 4 mm stacked-sheet separation cannot make their solid colliders overlap.
    tab0 = root.find(".//geom[@name='c09_paper_0_grip_tab']")
    tab1 = root.find(".//geom[@name='c09_paper_1_grip_tab']")
    tab0_y = float(tab0.attrib["pos"].split()[1])
    tab1_y = float(tab1.attrib["pos"].split()[1])
    tab_half_y = float(tab0.attrib["size"].split()[1])
    assert abs(tab1_y - tab0_y) > 2.0 * tab_half_y

    assert root.find(".//body[@name='c09_alignment_guide']") is None
    stapler = root.find(".//body[@name='c09_stapler']")
    assert stapler is not None
    guide_long = stapler.find("geom[@name='c09_alignment_guide_long']")
    guide_short = stapler.find("geom[@name='c09_alignment_guide_short']")
    assert guide_long is not None
    assert guide_short is not None
    assert guide_long.attrib["pos"] == "0 -0.082 -0.001"
    assert guide_long.attrib["size"] == "0.058 0.002 0.001"
    assert guide_short.attrib["pos"] == "-0.056 -0.158 -0.001"
    assert guide_short.attrib["size"] == "0.002 0.078 0.001"
    assert root.find(".//body[@name='c09_work_zone']") is None
    for target_name in ("c09_staple_target_a", "c09_staple_target_b"):
        target = root.find(f".//geom[@name='{target_name}']")
        assert target is not None
        assert target.attrib["pos"] == "0.080 0 0.003"
    assert (
        root.find(".//geom[@name='c09_staple_done']").attrib["fromto"]
        == "0.074 0 0.005 0.086 0 0.005"
    )
    assert float(
        root.find(".//geom[@name='c09_paper_0_geom']").attrib["size"].split()[2]
    ) >= 0.0015
    assert root.find(".//geom[@name='c10_socket_plate_center']") is not None
    assert root.find(".//geom[@name='c10_socket_capture_marker']") is not None
    assert (
        root.find(".//geom[@name='c10_cable_tail_a_collision']").attrib["class"]
        == "ctool_visual"
    )
    assert root.find(".//geom[@name='c10_power_strip_body']") is None
    for name in (
        "c10_power_strip_bottom",
        "c10_power_strip_left",
        "c10_power_strip_right",
        "c10_power_strip_center_front",
        "c10_power_strip_center_back",
    ):
        assert root.find(f".//geom[@name='{name}']") is not None
    assert root.find(".//geom[@name='c10_charger_head']") is None
    for name in (
        "c10_charger_head_front",
        "c10_charger_head_rear_top",
        "c10_charger_head_rear_bottom",
        "c10_charger_head_rear_left",
        "c10_charger_head_rear_right",
    ):
        assert root.find(f".//geom[@name='{name}']") is not None
    for suffix, expected_y in (("a", 0.008), ("b", -0.008)):
        prong = root.find(f".//geom[@name='c10_charger_prong_{suffix}']")
        assert prong is not None
        assert float(prong.attrib["pos"].split()[1]) == expected_y
        assert [float(value) for value in prong.attrib["size"].split()] == [
            0.022,
            0.002,
            0.002,
        ]
    assert [
        float(value)
        for value in root.find(
            ".//geom[@name='c10_cable_metal_tip']"
        ).attrib["size"].split()
    ] == [0.008, 0.0075, 0.003]
    for collision_name, visual_name, expected_class in (
        (
            "c10_cable_strain_relief_collision",
            "c10_cable_strain_relief",
            "ctool_collision",
        ),
        ("c10_cable_tail_a_collision", "c10_cable_tail_a", "ctool_visual"),
        ("c10_cable_tail_b_collision", "c10_cable_tail_b", "ctool_visual"),
    ):
        collision = root.find(f".//geom[@name='{collision_name}']")
        visual = root.find(f".//geom[@name='{visual_name}']")
        assert collision is not None
        assert visual is not None
        assert collision.attrib["class"] == expected_class
        assert collision.attrib["type"] == "capsule"
        assert collision.attrib["fromto"] == visual.attrib["fromto"]
        assert float(collision.attrib["size"]) < float(visual.attrib["size"])
        assert collision.attrib["rgba"] == "0 0 0 0"
        assert float(collision.attrib["mass"]) <= 0.006
        assert visual.attrib["class"] == "ctool_visual"
    assert root.find(".//geom[@name='c08_tongs_fixed_arm_visual']") is not None
    assert root.find(".//geom[@name='c08_tongs_fixed_scallop_2']") is not None
    c08_spring_collision = root.find(
        ".//geom[@name='c08_tongs_spring_collision']"
    )
    assert c08_spring_collision is not None
    assert c08_spring_collision.attrib["type"] == "box"
    assert c08_spring_collision.attrib["pos"] == "-0.128 0 0.012"
    assert c08_spring_collision.attrib["size"] == "0.011 0.029 0.011"
    tongs_joint = root.find(".//joint[@name='c08_tongs_joint']")
    assert tongs_joint.attrib["axis"] == "0 0 1"
    assert float(tongs_joint.attrib["springref"]) == 0.121765
    assert float(tongs_joint.attrib["stiffness"]) >= 1.0
    assert float(tongs_joint.attrib["damping"]) >= 0.12
    assert float(tongs_joint.attrib["frictionloss"]) >= 0.012
    for index in range(5):
        assert root.find(f".//geom[@name='c08_tongs_spring_coil_{index}']") is not None
    for slot in range(5):
        assert root.find(f".//geom[@name='c11_prompt_slot_{slot}']") is not None
    key_positions = np.asarray(
        [
            [
                float(value)
                for value in root.find(f".//body[@name='{name}']").attrib["pos"].split()
            ]
            for name in (
                "c11_key_red",
                "c11_key_yellow",
                "c11_key_blue",
                "c11_key_green",
            )
        ]
    )
    assert np.ptp(key_positions[:, 0]) == 0.0
    assert np.ptp(key_positions[:, 1]) >= 0.18
    xylophone_z = float(
        root.find(".//body[@name='c11_xylophone']").attrib["pos"].split()[2]
    )
    assert xylophone_z >= 0.896
    assert (
        root.find(
            ".//body[@name='c11_xylophone']/freejoint"
            "[@name='c11_xylophone_joint']"
        )
        is not None
    )
    mallet = root.find(".//body[@name='c11_mallet']")
    assert mallet is not None
    assert float(mallet.attrib["pos"].split()[2]) == pytest.approx(0.902)
    mallet_quat = np.fromstring(mallet.attrib["quat"], sep=" ")
    mallet_roll = 2.0 * np.arctan2(mallet_quat[1], mallet_quat[0])
    assert np.rad2deg(mallet_roll) == pytest.approx(3.014, abs=0.01)
    for collision_name in (
        "c11_mallet_handle",
        "c11_mallet_head",
        "c11_key_red_geom",
        "c11_key_yellow_geom",
        "c11_key_blue_geom",
        "c11_key_green_geom",
    ):
        collision = root.find(f".//geom[@name='{collision_name}']")
        assert collision is not None
        assert collision.attrib["class"] == "ctool_collision"
        assert collision.attrib["rgba"] == "0 0 0 0"
    assert root.find(".//geom[@name='c11_mallet_handle_visual']").attrib["type"] == "capsule"
    assert root.find(".//geom[@name='c11_mallet_head_visual']").attrib["type"] == "capsule"
    for key_name in (
        "c11_key_red",
        "c11_key_yellow",
        "c11_key_blue",
        "c11_key_green",
    ):
        assert root.find(f".//body[@name='{key_name}']/freejoint") is not None
    assert root.find(".//geom[@name='c12_set_square_hypotenuse']") is not None
    baseline_y = float(
        root.find(".//body[@name='c12_baseline_pad']").attrib["pos"].split()[1]
    )
    assert all(
        float(root.find(f".//body[@name='c12_block_{index}']").attrib["pos"].split()[1])
        < baseline_y
        for index in range(3)
    )


def test_c06_isaac_removes_camera_coordinate_axes_and_unused_live_screen(tmp_path: Path) -> None:
    task_preview = _load_task_preview_module(tmp_path)
    root = ET.parse(
        _PACKAGE_ROOT / "sim" / "envs" / "xmls" / "camera.xml"
    ).getroot()

    removed = task_preview._remove_c06_camera_debug_visuals(
        root,
        "t23_photograph_logo",
    )

    named_geoms = {
        geom.attrib.get("name")
        for geom in root.iter("geom")
        if geom.attrib.get("name")
    }
    assert removed == 4
    assert not named_geoms & task_preview._C06_CAMERA_DEBUG_GEOMS
    assert "camera_live_screen" not in named_geoms
    assert "camera_body_collision" in named_geoms
    assert "camera_shutter_button_collision" in named_geoms


def test_c06_isaac_adds_valid_inertials_to_visual_only_camera_links(
    tmp_path: Path,
) -> None:
    task_preview = _load_task_preview_module(tmp_path)
    root = ET.parse(
        _PACKAGE_ROOT / "sim" / "envs" / "xmls" / "camera.xml"
    ).getroot()

    applied = task_preview._apply_c06_camera_auxiliary_inertials(
        root,
        "t23_photograph_logo",
    )

    camera_root = next(
        body
        for body in root.iter("body")
        if body.attrib.get("name") == "link_19"
    )
    child_bodies = {
        body.attrib["name"]: body for body in camera_root.findall("body")
    }
    assert applied == 19
    assert child_bodies["link_0"].find("inertial").attrib["mass"] == "1e-6"
    assert child_bodies["link_20"].find("inertial").attrib["mass"] == "1e-6"
    assert child_bodies["link_7"].find("inertial").attrib["mass"] == "0.003"


def test_c06_isaac_enables_an_enlarged_movable_shutter_collider(
    tmp_path: Path,
) -> None:
    task_preview = _load_task_preview_module(tmp_path)
    root = ET.parse(
        _PACKAGE_ROOT / "sim" / "envs" / "xmls" / "camera.xml"
    ).getroot()

    configured = task_preview._configure_c06_shutter_collision(
        root,
        "t23_photograph_logo",
    )

    shutter = next(
        geom
        for geom in root.iter("geom")
        if geom.attrib.get("name") == "camera_shutter_button_collision"
    )
    assert configured == 1
    assert shutter.attrib["size"] == "0.015 0.0005"
    assert shutter.attrib["contype"] == "1"
    assert shutter.attrib["conaffinity"] == "1"
    assert shutter.attrib["margin"] == "0.002"


def test_c04_isaac_preserves_compiled_spray_inertials(tmp_path: Path) -> None:
    task_preview = _load_task_preview_module(tmp_path)
    root = ET.parse(
        _PACKAGE_ROOT / "sim" / "envs" / "xmls" / "spray.xml"
    ).getroot()

    applied = task_preview._apply_c04_spray_inertials(root, "t50_water_plant")

    bodies = {body.attrib.get("name"): body for body in root.iter("body")}
    assert applied == 5
    assert bodies["link_2"].find("inertial").attrib == {
        "pos": "0 -0.04 -0.008",
        "quat": "0.707106781186548 0.707106781186548 0 0",
        "mass": "0.106864",
        "diaginertia": "0.0002484944213333334 0.0002484944213333334 0.000041035776",
    }
    assert bodies["link_1"].find("inertial").attrib["mass"] == "0.012"
    assert bodies["link_0"].find("inertial").attrib["mass"] == "0.0128"
    assert bodies["cone_vis"].find("inertial").attrib["mass"] == "1e-6"


def test_c04_isaac_uses_one_broad_root_support_collision(tmp_path: Path) -> None:
    task_preview = _load_task_preview_module(tmp_path)
    root = ET.parse(
        _PACKAGE_ROOT / "sim" / "envs" / "xmls" / "spray.xml"
    ).getroot()

    removed = task_preview._apply_c04_stable_spray_root_collision(
        root,
        "t50_water_plant",
    )

    bodies = {body.attrib.get("name"): body for body in root.iter("body")}
    root_supports = [
        geom
        for geom in bodies["link_2"].findall("geom")
        if geom.attrib.get("class") == "spray/collision"
    ]
    scanned_root_hulls = [
        geom
        for geom in bodies["link_1"].findall("geom")
        if geom.attrib.get("class") == "spray/collision"
    ]
    trigger_collisions = [
        geom
        for geom in bodies["link_0"].findall("geom")
        if geom.attrib.get("class") == "spray/collision"
    ]
    assert removed == 12
    assert len(root_supports) == 1
    assert root_supports[0].attrib["type"] == "box"
    assert scanned_root_hulls == []
    assert len(trigger_collisions) == 1


def test_c04_material_mapping_keeps_leaf_soil_and_pot_textures_separate(tmp_path: Path) -> None:
    task_preview = _load_task_preview_module(tmp_path)
    texture_dir = tmp_path / "texture_aliases"
    texture_dir.mkdir()
    for file_name in ("image0.png", "image1.png"):
        (texture_dir / file_name).touch()
        (texture_dir / file_name.replace(".png", "_isaac_alpha.png")).touch()
    for file_name in ("image2.png", "image3.png"):
        (texture_dir / file_name).touch()

    root = ET.Element("mujoco")
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", {"name": "image0", "file": "texture_aliases/image0.png"})
    ET.SubElement(asset, "texture", {"name": "image1", "file": "texture_aliases/image1.png"})
    ET.SubElement(asset, "texture", {"name": "image2", "file": "texture_aliases/image2.png"})
    ET.SubElement(asset, "texture", {"name": "image3", "file": "texture_aliases/image3.png"})
    ET.SubElement(asset, "material", {"name": "blinn1SG", "texture": "image0"})
    ET.SubElement(asset, "material", {"name": "blinn2SG", "texture": "image1"})
    ET.SubElement(asset, "material", {"name": "blinn3SG", "texture": "image2"})
    ET.SubElement(asset, "material", {"name": "blinn4SG", "texture": "image3"})
    mjcf_path = tmp_path / "t50_water_plant_scene_isaac.xml"
    ET.ElementTree(root).write(mjcf_path, encoding="utf-8", xml_declaration=True)

    material_textures = task_preview._c04_material_texture_paths(
        mjcf_path,
        "t50_water_plant",
    )

    assert set(material_textures) == {"blinn1SG", "blinn2SG", "blinn3SG", "blinn4SG"}
    assert material_textures["blinn1SG"].name == "image0_isaac_alpha.png"
    assert material_textures["blinn2SG"].name == "image1_isaac_alpha.png"
    assert material_textures["blinn3SG"].name == "image2.png"
    assert material_textures["blinn4SG"].name == "image3.png"
    assert task_preview._c04_model_material_name(
        "/World/Task/plant/visuals/model_0_vis/Mesh"
    ) == "blinn2SG"
    assert task_preview._c04_model_material_name(
        "/World/Task/plant/visuals/model_1_vis/Mesh"
    ) == "blinn1SG"
    assert task_preview._c04_model_material_name(
        "/World/Task/plant/visuals/model_2_vis/Mesh"
    ) == "blinn4SG"
    assert task_preview._c04_model_material_name(
        "/World/Task/plant/visuals/model_3_vis/Mesh"
    ) == "blinn3SG"
    assert task_preview._c04_model_material_name(
        "/World/Task/plant/collisions/model_0_col/Mesh"
    ) is None


def test_c04_spray_uses_mujoco_water_plant_source() -> None:
    root = ET.parse(
        _PACKAGE_ROOT / "sim" / "envs" / "xmls" / "spray.xml"
    ).getroot()

    materials = {material.attrib["name"]: material for material in root.findall(".//material")}
    meshes = {mesh.attrib["name"]: mesh for mesh in root.findall(".//mesh") if "name" in mesh.attrib}
    geoms = {geom.attrib["name"]: geom for geom in root.iter("geom") if "name" in geom.attrib}
    default_geoms = {
        default.attrib["class"]: default.find("geom")
        for default in root.findall(".//default")
        if "class" in default.attrib and default.find("geom") is not None
    }

    assert not any(name.startswith("c04_ycb_windex") for name in meshes)
    assert not any(name.startswith("c04_ycb_windex") for name in materials)
    assert not any(name.startswith("spray_ycb") for name in geoms)
    assert not any(name.startswith("c04_spray") for name in materials)

    visual_mesh_geoms = [
        geom
        for geom in root.iter("geom")
        if geom.attrib.get("class") == "spray/visual"
        and geom.attrib.get("type") == "mesh"
        and geom.attrib.get("mesh", "").startswith("original-")
    ]
    assert visual_mesh_geoms
    assert all(
        geom.attrib["material"] == f"{geom.attrib['mesh']}_material_0_0"
        for geom in visual_mesh_geoms
    )

    assert geoms["spray"].attrib["rgba"] == "0.95 0.55 0.7 1"
    assert geoms["spray"].attrib["material"] == "pink_plastic_tex"
    assert geoms["cone_visual"].attrib["rgba"] == "0.2 0.4 1.0 0.5"
    assert "rgba" not in default_geoms["spray/collision"].attrib

    mesh_collision_geoms = [
        geom
        for geom in root.iter("geom")
        if geom.attrib.get("class") == "spray/collision"
        and geom.attrib.get("type") == "mesh"
    ]
    assert mesh_collision_geoms
    assert not any(name.startswith("spray_") for name in geoms)

    bodies = {body.attrib.get("name"): body for body in root.iter("body")}
    trigger_joint = bodies["link_0"].find("joint")
    assert trigger_joint is not None
    assert trigger_joint.attrib["name"] == "joint_0"
    assert trigger_joint.attrib["stiffness"] == "1"
    assert trigger_joint.attrib["damping"] == "2"
    assert "frictionloss" not in trigger_joint.attrib
    assert "armature" not in trigger_joint.attrib

    motors = [
        motor
        for actuator in root.findall("actuator")
        for motor in actuator.findall("motor")
    ]
    assert any(motor.attrib.get("joint") == "joint_0" for motor in motors)

    original_materials = [
        material
        for name, material in materials.items()
        if name.startswith("original-") and name.endswith("_material_0_0")
    ]
    assert original_materials
    assert all(material.attrib["rgba"] == "0.6 0.55 0.5 1" for material in original_materials)
    assert all(material.attrib["specular"] == "0.05" for material in original_materials)
    assert all(material.attrib["shininess"] == "0.1" for material in original_materials)


def test_preview_uses_raised_arm_home_for_imported_inactive_left_hand(
    tmp_path: Path,
) -> None:
    task_preview = _load_task_preview_module(tmp_path)

    default_home = task_preview._task_arm_home(None)
    a01_home = task_preview._task_arm_home("t26_place_fruit_in_bucket")
    b05_home = task_preview._task_arm_home("t03_cover_box")
    c04_home = task_preview._task_arm_home("t50_water_plant")
    c05_home = task_preview._task_arm_home("t14_move_and_click_mouse")
    d06_home = task_preview._task_arm_home("t42_sort_objects")
    d07_home = task_preview._task_arm_home("d07_organize_office_desk")
    d08_home = task_preview._task_arm_home("d08_toast_bread")
    d09_home = task_preview._task_arm_home("t31_prepare_coffee")
    d11_home = task_preview._task_arm_home("d11_presoak_pan")
    c09_home = task_preview._task_arm_home("c09_staple_paper_sheets")
    c10_home = task_preview._task_arm_home("t02_connect_charger_and_cable")
    c11_home = task_preview._task_arm_home("c11_xylophone_sequence")
    bc014_home = task_preview._task_arm_home(
        "t30_pour_seasonings"
    )

    assert a01_home == task_preview._A01_CLEAR_ARM_HOME
    assert b05_home["right"] == default_home["right"]
    assert b05_home["left"] != default_home["left"]
    assert c05_home["right"] == default_home["right"]
    assert c05_home["left"] == default_home["left"]
    assert b05_home["left"] == task_preview._RAISED_ARM_HOME["left"]
    assert c04_home["right"] == default_home["right"]
    assert c04_home["left"] == task_preview._RAISED_ARM_HOME["left"]
    assert d06_home == task_preview._D06_LOWERED_ARM_HOME
    assert c09_home == d06_home
    assert c10_home == d06_home
    assert c11_home == d06_home
    assert d07_home["right"] == default_home["right"]
    assert d07_home["left"] == task_preview._RAISED_ARM_HOME["left"]
    assert d08_home == default_home
    assert d09_home == d08_home
    mirror_signs = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0)
    assert d08_home["left"] == tuple(
        sign * value
        for sign, value in zip(mirror_signs, d08_home["right"], strict=True)
    )
    assert d11_home == task_preview._C_TOOL_SPREAD_ARM_HOME
    assert bc014_home == task_preview._C_TOOL_SPREAD_ARM_HOME


def test_c05_isaac_adds_valid_mousepad_inertial(tmp_path: Path) -> None:
    task_preview = _load_task_preview_module(tmp_path)
    root = ET.parse(
        _PACKAGE_ROOT / "sim" / "envs" / "xmls" / "arena_arm_hand_monitor_mouse_rokae_wuji1.xml"
    ).getroot()

    applied = task_preview._apply_c05_mousepad_inertial(
        root,
        "t14_move_and_click_mouse",
    )

    mousepad = next(
        body for body in root.iter("body") if body.attrib.get("name") == "mousepad"
    )
    assert applied == 1
    assert mousepad.find("inertial").attrib == {
        "pos": "0 0 0",
        "mass": "1e-6",
        "diaginertia": "1e-12 1e-12 1e-12",
    }


def test_d05_isaac_adds_valid_visual_link_inertial(tmp_path: Path) -> None:
    task_preview = _load_task_preview_module(tmp_path)
    root = ET.parse(
        _PACKAGE_ROOT / "sim" / "envs" / "xmls" / "d05_scissors_objects.xml"
    ).getroot()

    applied = task_preview._apply_d05_scissors_auxiliary_inertial(
        root, "t43_squeeze_scissors"
    )

    assert applied == 1
    link = next(
        body
        for body in root.iter("body")
        if body.attrib.get("name") == "d05_scissors_link"
    )
    assert link.find("inertial").attrib == {
        "pos": "0 0 0",
        "mass": "1e-6",
        "diaginertia": "1e-12 1e-12 1e-12",
    }


def test_c05_isaac_fixes_display_to_tabletop(tmp_path: Path) -> None:
    task_preview = _load_task_preview_module(tmp_path)
    root = ET.parse(
        _PACKAGE_ROOT / "sim" / "envs" / "xmls" / "display.xml"
    ).getroot()

    removed = task_preview._fix_c05_display(root, "t14_move_and_click_mouse")

    display = next(
        body for body in root.iter("body") if body.attrib.get("name") == "display_link_1"
    )
    assert removed == 2
    assert not list(display.iter("freejoint"))
    assert not list(display.iter("joint"))
    assert not [
        element
        for element in root.iter()
        if element.attrib.get("joint") in {"display_root", "joint_0"}
    ]


def test_preview_success_condition_labels_are_ascii(
    tmp_path: Path,
) -> None:
    task_preview = _load_task_preview_module(tmp_path)

    for task_name in (
        "t04_cover_pitcher",
        "t09_insert_peg",
        "t23_photograph_logo",
        "t17_open_latched_door",
    ):
        for label in task_preview.success_condition_labels(task_name):
            label.encode("ascii")


def test_c01_wood_is_compact_enough_for_one_hand_support(
    tmp_path: Path,
) -> None:
    task_preview = _load_task_preview_module(tmp_path)
    root = ET.parse(
        _PACKAGE_ROOT / "sim" / "envs" / "xmls" / "c01_wood_nail.xml"
    ).getroot()
    task_preview._apply_task_body_position_overrides(root, "t08_hammer_nail")

    bodies = {body.attrib["name"]: body for body in root.iter("body")}
    geoms = {geom.attrib["name"]: geom for geom in root.iter("geom")}
    sites = {site.attrib["name"]: site for site in root.iter("site")}

    assert bodies["wood"].attrib["pos"] == "-0.350000 0.200000 0.916500"
    inertial = bodies["wood"].find("inertial")
    assert inertial is not None
    assert inertial.attrib["pos"] == "0 0 0"
    assert inertial.attrib["mass"] == "0.085"
    assert inertial.attrib["diaginertia"] == (
        "0.00004905208333 0.00008517708333 0.0001055416667"
    )
    wood_collisions = [
        geom
        for geom in bodies["wood"].findall("geom")
        if geom.attrib.get("contype", "1") != "0"
    ]
    assert [geom.attrib["name"] for geom in wood_collisions] == [
        "wood_collision"
    ]
    assert geoms["wood_collision"].attrib["size"] == "0.050 0.035 0.0225"
    assert geoms["wood_collision"].attrib["type"] == "box"
    assert geoms["wood_visual"].attrib["type"] == "mesh"
    assert sites["wood_top"].attrib["pos"] == "0 0 0.0225"
    assert bodies["nail"].attrib["pos"] == "0 0 0.0695"


def test_c01_hammer_uses_matching_ycb_visual_and_convex_collisions() -> None:
    root = ET.parse(
        _PACKAGE_ROOT / "sim" / "envs" / "xmls" / "hammer.xml"
    ).getroot()

    textures = {texture.attrib["name"]: texture for texture in root.findall(".//texture")}
    materials = {material.attrib["name"]: material for material in root.findall(".//material")}
    meshes = {mesh.attrib["name"]: mesh for mesh in root.findall(".//mesh")}
    geoms = {geom.attrib["name"]: geom for geom in root.iter("geom")}

    assert textures["c01_ycb_hammer_tex"].attrib["file"] == (
        "lightwheel_ycb/hammer/visuals/T_hammer_BC001.png"
    )
    assert materials["c01_ycb_hammer_mat"].attrib["texture"] == "c01_ycb_hammer_tex"
    assert meshes["c01_ycb_hammer_vis"].attrib["file"] == (
        "lightwheel_ycb/hammer/visuals/optimized_visuals/hammer.obj"
    )
    assert geoms["hammer_visual"].attrib["mesh"] == "c01_ycb_hammer_vis"
    assert geoms["hammer_visual"].attrib["material"] == "c01_ycb_hammer_mat"
    assert geoms["hammer_visual"].attrib["contype"] == "0"
    assert geoms["hammer_visual"].attrib["mass"] == "0"

    collision_names = (
        "handle",
        "hammer_collision_02",
        "hammer_collision_03",
        "hammer_collision_04",
        "head",
        "hammer_collision_06",
        "neck",
        "face",
        "hammer_collision_09",
        "hammer_collision_10",
        "claw",
        "hammer_collision_12",
    )
    visual = geoms["hammer_visual"]
    assert visual.attrib["quat"] == "0.7071068 0.7071068 0 0"
    for index, geom_name in enumerate(collision_names, start=1):
        mesh_name = f"c01_ycb_hammer_col_{index:02d}"
        assert meshes[mesh_name].attrib["file"] == (
            "lightwheel_ycb/hammer/collision/"
            f"hammer_Collider{index}_m.obj"
        )
        collider = geoms[geom_name]
        assert collider.attrib["type"] == "mesh"
        assert collider.attrib["mesh"] == mesh_name
        assert collider.attrib["pos"] == visual.attrib["pos"]
        assert collider.attrib["quat"] == visual.attrib["quat"]
        assert collider.attrib["rgba"] == "0 0 0 0"
        assert collider.attrib["group"] == "3"


def test_c01_wood_and_nail_use_matching_ycb_assets() -> None:
    root = ET.parse(
        _PACKAGE_ROOT / "sim" / "envs" / "xmls" / "c01_wood_nail.xml"
    ).getroot()

    textures = {texture.attrib["name"]: texture for texture in root.findall(".//texture")}
    materials = {material.attrib["name"]: material for material in root.findall(".//material")}
    meshes = {mesh.attrib["name"]: mesh for mesh in root.findall(".//mesh")}
    geoms = {geom.attrib["name"]: geom for geom in root.iter("geom")}

    assert textures["c01_ycb_wood_block_tex"].attrib["file"] == (
        "lightwheel_ycb/wood_block/visuals/T_wood_block_BC001.png"
    )
    assert materials["c01_ycb_wood_block_mat"].attrib["texture"] == (
        "c01_ycb_wood_block_tex"
    )
    assert meshes["c01_ycb_wood_block_vis"].attrib["file"] == (
        "lightwheel_ycb/wood_block/visuals/wood_block.obj"
    )
    assert geoms["wood_visual"].attrib["mesh"] == "c01_ycb_wood_block_vis"
    assert geoms["wood_visual"].attrib["material"] == "c01_ycb_wood_block_mat"
    assert geoms["wood_collision"].attrib["type"] == "box"

    assert textures["c01_ycb_nail_tex"].attrib["file"] == (
        "lightwheel_ycb/a_Nails/visuals/T_a_Nails_BC001.png"
    )
    assert materials["c01_ycb_nail_mat"].attrib["texture"] == "c01_ycb_nail_tex"
    assert meshes["c01_ycb_nail_vis"].attrib["file"] == (
        "lightwheel_ycb/a_Nails/visuals/a_Nails.obj"
    )
    assert geoms["nail_visual"].attrib["mesh"] == "c01_ycb_nail_vis"
    assert geoms["nail_visual"].attrib["material"] == "c01_ycb_nail_mat"
    assert meshes["c01_ycb_nail_collision"].attrib["file"] == (
        "lightwheel_ycb/a_Nails/collision/a_Nails_Collider1_m.obj"
    )
    assert meshes["c01_ycb_nail_collision"].attrib["scale"] == (
        meshes["c01_ycb_nail_vis"].attrib["scale"]
    )
    assert geoms["nail_head"].attrib["type"] == "mesh"
    assert geoms["nail_head"].attrib["mesh"] == "c01_ycb_nail_collision"
    assert geoms["nail_head"].attrib["pos"] == geoms["nail_visual"].attrib["pos"]
    assert geoms["nail_head"].attrib["quat"] == geoms["nail_visual"].attrib["quat"]
    assert geoms["nail_head"].attrib["rgba"] == "0 0 0 0"
    assert geoms["nail_head"].attrib["group"] == "3"
    assert geoms["nail_shaft"].attrib["type"] == "cylinder"
    assert geoms["nail_shaft"].attrib["mass"] == "0"
