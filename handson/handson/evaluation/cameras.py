"""Three-view capture from the paper evaluator, using packaged assets."""
from pathlib import Path
import json
import numpy as np
from .contracts import require, write_new
SENSORS = {"main": "head_camera", "wrist_left": "wrist_left_camera", "wrist_right": "wrist_right_camera"}

def install_launcher(physics_backend="physx"):
    """Keep RTX cameras enabled while native step remains physics-only."""
    from isaaclab.app import AppLauncher
    original = AppLauncher.__init__
    def launch(app, launcher_args=None, **kwargs):
        args = dict(vars(launcher_args) if hasattr(launcher_args, "__dict__") else launcher_args or {})
        args["enable_cameras"] = True
        result = original(app, args, **kwargs)
        if physics_backend == "newton":
            from .newton_runtime import pin_warp
            pin_warp()
        return result
    AppLauncher.__init__ = launch

def render_probe(env, job, output, number, seed, phase, capture, *, save_png=True, image_sink=None):
    """Capture the original six-pass per-view schedule without physics steps."""
    import torch
    import carb.settings
    from PIL import Image
    from scipy.spatial.transform import Rotation
    from ..eef52 import measured_palm_pose
    from ..sim.isaaclab.compat import quat_wxyz_to_isaaclab, quat_isaaclab_to_wxyz
    from .camera_randomization import sample
    from ..sim.isaaclab.default_appearance import verify_environment
    from ..sim.isaaclab.wrist_cable_visuals import update_wrist_cables

    contract = json.loads(Path(__file__).with_name("camera_contract.json").read_text())
    cameras = env._cameras
    require(set(SENSORS.values()) <= set(cameras), "Native three-view cameras missing")
    before = capture(env)
    if env._physics_backend == "newton":
        from .newton_runtime import prepare_render
        prepare_render()
    update_wrist_cables(env)
    jitter = sample(job["task"], "independent-camera-probe", seed)
    expected_poses = {}
    for role, sensor in SENSORS.items():
        spec = contract["cameras"][role]
        camera = cameras[sensor]
        if role == "main":
            nominal = np.asarray(spec["nominal_world_pose_opengl"])
        else:
            side = role.removeprefix("wrist_")
            state, _ = measured_palm_pose(env, side)
            nominal = pose_matrix(state[:3], state[3:]) @ np.asarray(spec["nominal_camera_to_palm_opengl"])
        draw = jitter["cameras"][role]
        target = nominal @ pose_matrix(draw["translation_m"], draw["quaternion_wxyz"])
        quaternion = Rotation.from_matrix(target[:3, :3]).as_quat()[[3, 0, 1, 2]]
        # The paper capture owns the sampled pose at this boundary. The ordinary
        # preview keeps its nominal calibrated head mount outside this call.
        camera._handson_sampled_pose = True
        try:
            camera.set_world_poses(
                torch.tensor([target[:3, 3]], device=camera.device, dtype=torch.float32),
                torch.tensor([quat_wxyz_to_isaaclab(quaternion)], device=camera.device, dtype=torch.float32), convention="opengl")
        finally:
            camera._handson_sampled_pose = False
        if env._physics_backend == "newton":
            from .newton_runtime import publish_camera_pose
            publish_camera_pose(camera, target)
        expected_poses[role] = target

    textures = role_render_products(cameras)
    repairs = prepare_toggle_safe_annotators(cameras, textures)
    def capture_one(role):
        camera = cameras[SENSORS[role]]
        for other, texture in textures.items():
            texture.set_updates_enabled(other == role)
        ready = False
        for _ in range(6):
            env._sim.render()
            if hasattr(camera, "_render_data"):
                from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update
                ensure_isaac_rtx_render_update()
            ready = has_frame(camera)
            if ready:
                camera.update(0.0, force_recompute=True)
        require(ready, "No fresh camera frame: " + role)
        return camera.data.output["rgb"][0].detach().cpu().numpy()[..., :3]
    try:
        images, passes = capture_split_views(capture_one, env._set_wrist_obstructions_hidden,
                                             carb.settings.get_settings(), contract["photometry"])
    finally:
        for texture in textures.values():
            texture.set_updates_enabled(True)
    require(before == capture(env), "Rendering changed measured physics")
    verify_environment(env)
    camera_readback = {}
    for role, rgb in images.items():
        require(rgb.dtype == np.uint8 and list(rgb.shape) == contract["cameras"][role]["image_hwc"],
                "Invalid camera buffer: " + role)
        camera = cameras[SENSORS[role]]
        spec = contract["cameras"][role]
        positions, quaternions = camera._view.get_world_poses()
        if hasattr(positions, "detach"):
            positions, quaternions = positions.detach().cpu().numpy(), quaternions.detach().cpu().numpy()
        actual = pose_matrix(positions[0], quat_isaaclab_to_wxyz(quaternions[0]))
        if env._physics_backend == "newton":
            from .newton_runtime import rendered_camera_pose
            usd_pose, fabric_pose = rendered_camera_pose(camera)
            require(np.max(np.abs(usd_pose - expected_poses[role])) < 2e-5
                    and np.max(np.abs(fabric_pose - expected_poses[role])) < 2e-5,
                    "RTX USD/Fabric camera pose differs from requested pose: " + role)
        intrinsic = camera.data.intrinsic_matrices.detach().cpu().numpy()[0]
        pose_error = float(np.max(np.abs(actual - expected_poses[role])))
        intrinsic_error = float(np.max(np.abs(intrinsic - np.asarray(spec["K_effective"]))))
        prim = camera._sensor_prims[0].GetPrim()
        lens = dict(model=prim.GetAttribute("omni:lensdistortion:model").Get(),
                    image_size=list(prim.GetAttribute("omni:lensdistortion:opencvPinhole:imageSize").Get()),
                    values={name: prim.GetAttribute("omni:lensdistortion:opencvPinhole:" + name).Get()
                            for name in ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6", "s1", "s2", "s3", "s4"]})
        lens_errors = check_lens(spec, lens)
        camera_readback[role] = dict(expected_opengl=expected_poses[role].tolist(), actual_opengl=actual.tolist(),
                                    max_pose_error=pose_error, intrinsic_max_error_px=intrinsic_error,
                                    intrinsic=intrinsic.tolist(), lens=lens, lens_errors=lens_errors)
        # Preserve the original evaluator's strict main and warning-only wrist checks.
        if role == "main":
            require(pose_error < 2e-5 and intrinsic_error < 1e-3 and not lens_errors,
                    "Main camera projection or sampled pose differs from the paper contract")
        elif pose_error >= 2e-5 or intrinsic_error >= 1e-3 or lens_errors:
            print("WRIST_CAMERA_WARNING", role, camera_readback[role], flush=True)
        if save_png:
            Image.fromarray(rgb).save(output / f"{phase}_{number:03d}_{role}.png")
    if save_png:
        write_new(output / f"camera_{phase}_{number:03d}.json",
                  dict(jitter=jitter, passes=passes, repairs=repairs, camera_readback=camera_readback,
                       physics_unchanged=True, renders_per_view=6))
    if image_sink is not None:
        image_sink(images)
    return images


def capture_split_views(capture_one, set_hidden, settings, photometry):
    """Capture/copy each role under its own exposure and visibility state."""
    prefix = '/rtx/post/tonemap/'
    common = {prefix+'cameraShutter': photometry['camera_shutter'],
              prefix+'fNumber': photometry['f_number']}
    key = prefix+'filmIso'
    previous = {k: settings.get(k) for k in (key, *common)}
    images, receipt = {}, []
    try:
        for k,v in common.items():
            settings.set(k, float(v))
        for role,sensor in SENSORS.items():
            hidden = role != 'main'
            set_hidden(hidden)
            iso = float(photometry['film_iso_by_sensor'][sensor])
            settings.set(key, iso)
            # Copy now: a later camera pass can update every render product.
            images[role] = capture_one(role).copy()
            receipt.append(dict(role=role, wrist_obstructions_hidden=hidden,
                                requested_iso=iso, actual_iso=settings.get(key)))
    finally:
        try:
            set_hidden(False)
        finally:
            for k,v in previous.items():
                settings.set(k, v)
    return images, receipt

def role_render_products(cameras):
    """Each role's own HydraTexture, looked up in replicator's registry by the camera's render product."""
    from omni.replicator.core.scripts.utils.viewport_manager import ViewportManager
    manager = ViewportManager()
    textures = {}
    for role, sensor in SENSORS.items():
        paths = render_product_paths(cameras[sensor])
        require(len(paths) == 1, 'Expected one render product per camera: '+role)
        texture = manager._hydra_textures.get(manager._context, paths[0]).hydra_texture
        require(texture.get_render_product_path() == paths[0], 'Render product lookup differs: '+role)
        textures[role] = texture
    require(len({id(t) for t in textures.values()}) == len(SENSORS), 'Cameras share a render product')
    return textures

def has_frame(camera):
    """Whether every annotator of the camera holds data. A render product re-enabled after
    skipped frames has none until it renders again, and Camera.update fails on it."""
    for annotator in camera_annotators(camera):
        output = annotator.get_data()
        data = output['data'] if isinstance(output, dict) else output
        if data is None or int(getattr(data, 'size', 0)) == 0:
            return False
    return True


def render_product_paths(camera):
    """Lab 2 owns render products on Camera; Lab 3 owns them on render data."""
    if hasattr(camera, "render_product_paths"):
        return camera.render_product_paths
    return camera._render_data.render_product_paths


def camera_annotators(camera):
    if hasattr(camera, "_rep_registry"):
        return tuple(a for group in camera._rep_registry.values() for a in group)
    return tuple(camera._render_data.annotators.values())

def prepare_toggle_safe_annotators(cameras, textures):
    """Bind annotators while their HydraTexture is disabled, once per binding.

    Replicator 1.12.27 loses the LdrColorSD render var after a texture toggle
    when IsaacLab originally attached its annotators while the texture was on.
    Attaching while off preserves that render var across later toggles. Do not
    read annotators here: their pointers are stale until a subsequent render.
    No render or physics step is added to the camera capture schedule.
    """
    import time
    repaired = []
    for role, sensor in SENSORS.items():
        camera, texture = cameras[sensor], textures[role]
        paths = tuple(render_product_paths(camera))
        require(len(paths) == 1, 'Toggle repair requires one render product: '+role)
        require(texture.get_render_product_path() == paths[0], 'Toggle repair product differs: '+role)
        annotators = camera_annotators(camera)
        require(annotators, 'Toggle repair has no annotators: '+role)
        binding = (id(texture), paths, tuple(id(a) for a in annotators))
        if getattr(camera, '_toggle_safe_binding', None) == binding:
            continue
        before = time.monotonic()
        enabled = bool(texture.get_updates_enabled())
        try:
            texture.set_updates_enabled(False)
            for annotator in annotators:
                annotator.detach(list(paths))
                annotator.attach(paths[0])
        finally:
            texture.set_updates_enabled(enabled)
        # Mark only a fully successful binding and restoration.
        camera._toggle_safe_binding = binding
        repaired.append(dict(role=role, render_product=paths[0], annotators=len(annotators),
                             attached_while_disabled=True, seconds=time.monotonic()-before))
    return repaired

def pose_matrix(position, quaternion):
    import numpy as np
    from scipy.spatial.transform import Rotation
    t = np.eye(4)
    t[:3, :3] = Rotation.from_quat(np.asarray(quaternion)[[1,2,3,0]]).as_matrix()
    t[:3, 3] = position
    return t


def install_native_camera_creation(environment_class):
    """Enable only the camera factory, not render-in-step or different physics.

    The historical ABC scene has an independent _cameras mapping and no
    InteractiveScene. Camera sensors must be constructed before native reset
    initializes their timeline subscriptions.
    """
    original = environment_class._init_recording_cameras

    def initialize(env):
        env._evaluation_rtx_cameras = True
        image_obs = env.image_obs
        had_deferred = hasattr(env, '_deferred_camera_render')
        deferred = getattr(env, '_deferred_camera_render', None)
        try:
            env.image_obs = True
            env._deferred_camera_render = False
            return original(env)
        finally:
            env.image_obs = image_obs
            if had_deferred:
                env._deferred_camera_render = deferred
            else:
                del env._deferred_camera_render

    environment_class._init_recording_cameras = initialize
    original_contacts = getattr(environment_class, '_configure_ground_truth_contact_reports', None)
    if original_contacts is not None:
        def contacts(env, *args, **kwargs):
            result = original_contacts(env, *args, **kwargs)
            if not env._cameras:
                env._init_recording_cameras()
            return result
        environment_class._configure_ground_truth_contact_reports = contacts


def install_native_scene_camera_creation(environment_class):
    """Enable the InteractiveScene sensor factory without render-in-step."""
    original = environment_class._make_native_scene_cfg

    def make_scene(env, *args, **kwargs):
        image_obs, deferred = env.image_obs, env._deferred_camera_render
        try:
            env.image_obs = True
            env._deferred_camera_render = False
            return original(env, *args, **kwargs)
        finally:
            env.image_obs = image_obs
            env._deferred_camera_render = deferred

    environment_class._make_native_scene_cfg = make_scene


def check_lens(spec, readback):
    """Check authored native lens, not the cached Camera K we populate."""
    k = spec['K_effective']
    expected = dict(fx=k[0][0], fy=k[1][1], cx=k[0][2], cy=k[1][2])
    names = ['k1','k2','p1','p2','k3','k4','k5','k6','s1','s2','s3','s4']
    d = spec['D_effective']
    expected.update(zip(names, d+[0.]*(len(names)-len(d))))
    errors = {n: readback['values'].get(n) for n, v in expected.items()
              if readback['values'].get(n) is None or
              abs(float(readback['values'][n])-v) > (1e-3 if n in ('fx','fy','cx','cy') else 1e-7)}
    if readback['model'] != 'opencvPinhole':
        errors['model'] = readback['model']
    if readback['image_size'] != spec['native_width_height']:
        errors['image_size'] = readback['image_size']
    return errors

