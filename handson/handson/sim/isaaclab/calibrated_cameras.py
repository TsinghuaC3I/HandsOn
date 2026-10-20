"""HandsOn main camera; import-safe until the existing AppLauncher starts Kit."""
import copy
import hashlib
import json
from pathlib import Path
from .compat import quat_wxyz_to_isaaclab


def profile():
    path = Path(__file__).with_name("camera_calibration.json")
    value = json.loads(path.read_text())
    assert value["version"] == "handson"
    return value


def install(AppLauncher=None):
    """Patch the camera constructor after Kit starts, before scene creation."""
    if AppLauncher is None:
        from isaaclab.app import AppLauncher
    if getattr(AppLauncher, "_handson_camera_installed", False):
        return
    launch = AppLauncher.__init__

    def initialize(app, launcher_args=None, **kwargs):
        if profile().get("lens_model") == "opencvPinhole":
            # USD schema registration must happen before the first stage/registry
            # is created. Enabling the extension after SimulationApp is too late.
            addition = " --enable omni.usd.schema.omni_lens_distortion"
            launcher_args = copy.deepcopy(launcher_args)
            if isinstance(launcher_args, dict):
                launcher_args["kit_args"] = launcher_args.get("kit_args", "")+addition
            elif launcher_args is not None:
                launcher_args.kit_args = getattr(launcher_args, "kit_args", "")+addition
            else:
                kwargs["kit_args"] = kwargs.get("kit_args", "")+addition
        launch(app, launcher_args, **kwargs)
        from isaaclab.sensors import Camera
        if getattr(Camera, "_handson_camera_installed", False):
            return
        original = Camera.__init__
        calibration = profile()
        Camera._handson_calibration = calibration

        specs = json.loads((Path(__file__).parents[2] / "evaluation/camera_contract.json").read_text())["cameras"]
        roles = {"head_camera": "main", "HeadRecordCamera": "main",
                 "wrist_left_camera": "wrist_left", "wrist_right_camera": "wrist_right",
                 "LeftWristRecordCamera": "wrist_left", "RightWristRecordCamera": "wrist_right"}

        def camera_init(camera, cfg, *camera_args, **camera_kwargs):
            role = roles.get(cfg.prim_path.rsplit("/", 1)[-1])
            if role is not None:
                spec = specs[role]
                cfg = copy.deepcopy(cfg)
                cfg.width, cfg.height = spec["native_width_height"]
                assert cfg.spawn is not None
                vertical = float(cfg.spawn.vertical_aperture or 20.955)
                cfg.spawn.vertical_aperture = vertical
                cfg.spawn.horizontal_aperture = vertical * cfg.width / cfg.height
                cfg.spawn.focal_length = spec["K_effective"][1][1] * vertical / cfg.height
                cfg.spawn.horizontal_aperture_offset = 0.0
                cfg.spawn.vertical_aperture_offset = 0.0
                if "near_clip_m" in spec:
                    cfg.spawn.clipping_range = (spec["near_clip_m"], cfg.spawn.clipping_range[1])
                if role == "main":
                    cfg.offset.pos = tuple(calibration["position_m"])
                    cfg.offset.rot = quat_wxyz_to_isaaclab(calibration["quaternion_wxyz_opengl"])
                    cfg.offset.convention = "opengl"
                    camera._handson_camera_constructor_applied = True
                camera._handson_camera_role = role
            original(camera, cfg, *camera_args, **camera_kwargs)
            if role is not None:
                from isaaclab.sim import find_matching_prims
                lens = dict(runtime_K=spec["K_effective"], runtime_D=spec["D_effective"],
                            native_width_height=spec["native_width_height"])
                for prim in find_matching_prims(camera.cfg.prim_path):
                    apply_lens(prim, lens)

        Camera.__init__ = camera_init
        original_intrinsics = Camera._update_intrinsic_matrices

        def update_intrinsics(camera, env_ids=None):
            original_intrinsics(camera, env_ids)
            role = getattr(camera, "_handson_camera_role", None)
            if role is not None:
                import torch
                matrices = camera._data.intrinsic_matrices
                matrices = getattr(matrices, "torch", matrices)
                if hasattr(camera, "_resolve_env_ids_np"):
                    env_ids = camera._resolve_env_ids_np(env_ids)
                elif env_ids is None:
                    env_ids = slice(None)
                matrices[env_ids] = torch.tensor(
                    specs[role]["K_effective"], device=camera.device,
                    dtype=matrices.dtype)

        Camera._update_intrinsic_matrices = update_intrinsics
        original_pose = Camera.set_world_poses
        original_view = Camera.set_world_poses_from_view

        def fixed_pose(camera, env_ids=None):
            import torch
            assert camera._view.count == 1, "HandsOn fixed station supports one environment per camera"
            device = camera.device
            positions = torch.tensor([calibration["position_m"]], device=device, dtype=torch.float32)
            orientations = torch.tensor([quat_wxyz_to_isaaclab(calibration["quaternion_wxyz_opengl"])], device=device, dtype=torch.float32)
            return original_pose(camera, positions, orientations, env_ids=env_ids, convention="opengl")

        def set_pose(camera, positions=None, orientations=None, env_ids=None, convention="ros"):
            if (getattr(camera, "_handson_camera_constructor_applied", False)
                    and not getattr(camera, "_handson_sampled_pose", False)):
                return fixed_pose(camera, env_ids)
            return original_pose(camera, positions, orientations, env_ids=env_ids, convention=convention)

        def set_view(camera, eyes, targets, env_ids=None):
            if (getattr(camera, "_handson_camera_constructor_applied", False)
                    and not getattr(camera, "_handson_sampled_pose", False)):
                return fixed_pose(camera, env_ids)
            return original_view(camera, eyes, targets, env_ids=env_ids)

        Camera.set_world_poses = set_pose
        Camera.set_world_poses_from_view = set_view
        Camera._handson_camera_installed = True

    AppLauncher.__init__ = initialize
    AppLauncher._handson_camera_installed = True


def apply_lens(prim, p):
    """Author the native schema exactly as Isaac Sim 5.1's camera API does."""
    from pxr import Gf
    assert prim.ApplyAPI("OmniLensDistortionOpenCvPinholeAPI")
    assert prim.GetAttribute("omni:lensdistortion:model").Set("opencvPinhole")
    k = p["runtime_K"]
    values = dict(fx=k[0][0], fy=k[1][1], cx=k[0][2], cy=k[1][2],
                  imageSize=Gf.Vec2i(*p["native_width_height"]))
    names = ["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6", "s1", "s2", "s3", "s4"]
    d = p["runtime_D"]
    assert len(d) <= len(names)
    values.update(zip(names, d + [0.] * (len(names)-len(d))))
    for name, value in values.items():
        attr = prim.GetAttribute("omni:lensdistortion:opencvPinhole:"+name)
        assert attr and attr.Set(value), name


def metadata(inherited):
    """Keep wrist/visibility/photometry contracts byte-for-byte equivalent."""
    p = profile()
    result = copy.deepcopy(inherited)
    result.update(name=p["profile"], version=p["profile"])
    head = result["head"]
    head.update(
        production_device_serial=p["device_serial"],
        calibration_device_serial=p["device_serial"],
        calibration_is_same_model_proxy=False,
        eye_xyz_m=p["position_m"],
        quat_wxyz_opengl=p["quaternion_wxyz_opengl"],
        extrinsic_source=p.get("extrinsic_source", "tabletop_depth_and_manual_edge_registration"),
        calibration_status=p.get("calibration_status", "estimated camera-to-robot registration; measured tabletop normal and distance"),
        K_calibration=[[p["factory_color"]["fx"], 0., p["factory_color"]["cx"]],
                       [0., p["factory_color"]["fy"], p["factory_color"]["cy"]], [0., 0., 1.]],
        D_calibration=p["factory_color"]["D_opencv"],
        native_width_height=p["native_width_height"],
        K_runtime=p["runtime_K"],
        principal_point_runtime=p.get("lens_model", "centered_omniverse_pinhole"),
        D_runtime=p.get("runtime_D", []),
        factory_stream_is_rectified=p.get("factory_stream_is_rectified", "not established by this camera fit"),
        full_robot_extrinsic_calibration=p.get("full_robot_extrinsic_calibration", False),
        fit_frames=p.get("fit_frames", ["floor7_20260916_rgb_depth"]),
        holdout_frames=p.get("holdout_frames", []),
    )
    import math
    head["vertical_fov_deg"] = math.degrees(2*math.atan(480/(2*p["runtime_K"][1][1])))
    head["horizontal_fov_deg_at_16_9"] = math.degrees(2*math.atan(854/(2*p["runtime_K"][0][0])))
    head["camera_to_world_opengl"] = [
        [*row, p["position_m"][i]] for i, row in enumerate(p["rotation_camera_to_world_opengl"])
    ] + [[0., 0., 0., 1.]]
    result["handson_camera_profile_sha256"] = hashlib.sha256(
        Path(__file__).with_name("camera_calibration.json").read_bytes()).hexdigest()
    result["inherited_photo_extrinsics_sha256"] = result.pop("photo_extrinsics_sha256", None)
    return result


def verify(cameras):
    import numpy as np
    p = profile()
    camera = cameras["head_camera"]
    assert getattr(camera, "_handson_camera_constructor_applied", False), "Uncovered head camera constructor"
    assert list(camera.image_shape) == list(reversed(p["native_width_height"]))
    actual = camera._data.intrinsic_matrices.detach().cpu().numpy()
    assert np.max(np.abs(actual[0]-np.asarray(p["runtime_K"]))) < 1e-3, actual
    receipt = {"native_size_verified": True, "intrinsics_max_error_px": float(
        np.max(np.abs(actual[0]-np.asarray(p["runtime_K"]))))}
    if p.get("lens_model") == "opencvPinhole":
        prim = camera._sensor_prims[0].GetPrim()
        assert prim.GetAttribute("omni:lensdistortion:model").Get() == "opencvPinhole"
        k = p["runtime_K"]
        values = dict(fx=k[0][0], fy=k[1][1], cx=k[0][2], cy=k[1][2])
        values.update(zip(["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"], p["runtime_D"]))
        readback = {name: prim.GetAttribute("omni:lensdistortion:opencvPinhole:"+name).Get() for name in values}
        assert max(abs(values[n]-readback[n]) for n in values) < 1e-3
        assert list(prim.GetAttribute("omni:lensdistortion:opencvPinhole:imageSize").Get()) == p["native_width_height"]
        receipt["native_lens_schema"] = readback
    return receipt
