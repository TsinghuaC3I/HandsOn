#!/usr/bin/env python3
"""Preview any HandsOn ROKAE + Wuji task in native Isaac Sim/PhysX."""

from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

# Isaac Sim bootstraps its CUDA/Kit extensions while AppLauncher is imported.
# Set the process-wide allocator workaround before that import; setting it
# below AppLauncher is too late for direct preview launches and can leave the
# first PhysX reset in the CUDA-700/illegal-memory failure mode.
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "handson"))

from isaaclab.app import AppLauncher

from handson.sim.isaaclab.compat import create_app_launcher
from handson.sim.isaaclab.physics_config import (
    bounded_single_env_gpu_physx_kwargs,
)
from handson.sim.isaaclab.runtime_display import enforce_isaac_viewport_only, isaac_kit_args


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task_name", required=True, help="HandsOn task id to preview.")
parser.add_argument(
    "--hand_type",
    choices=("wuji1", "wuji2"),
    default="wuji1",
    help="Wuji articulation to import into the Isaac task preview.",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=0,
    help="Stop after this many physics steps; 0 keeps the interactive preview open.",
)
parser.add_argument(
    "--status_interval",
    type=int,
    default=240,
    help="Print physical scene state every N steps; 0 disables periodic output.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if not any(argument == "--device" or argument.startswith("--device=") for argument in sys.argv[1:]):
    args_cli.device = "cuda:0"
# The pizza operator-test scene is intentionally CPU PhysX. On this host the
# generic CUDA bootstrap can be killed by the PhysX allocator before the first
# frame (returncode=-9), which makes a healthy browser look unresponsive.
if args_cli.task_name == "user_make_pizza_long_horizon":
    args_cli.device = "cpu"
    # Keep the operator viewport focused on the scene.  Isaac's camera/light
    # gizmos are authoring aids, not task assets, and otherwise resemble
    # unexplained transparent objects in a collection preview.
    os.environ.setdefault("HANDSON_ISAACSIM_HIDE_VIEWPORT_HELPERS", "1")
if args_cli.task_name in {
    "e01_fold_garment",
    "e02_tie_rope_knot",
    "e03_untie_random_rope_knot",
    "e04_cats_cradle_cycle",
    "e05_fold_cloth_wipe_table",
    "eval_flex02_fold_towel_twice",
}:
    args_cli.kit_args = " ".join(
        filter(
            None,
            (
                str(args_cli.kit_args or ""),
                "--/persistent/physics/enableDeformableBeta=true",
            ),
        )
    )
elif args_cli.task_name in {
    "g01_deformable_asset_inspection",
    "g01_2_deformable_asset_inspection",
    "g01_4_soft_pillow_to_carton",
    "eval_flex03_pack_pillow",
    "eval_flex01_compress_sponge_into_mug",
}:
    args_cli.kit_args = " ".join(
        filter(
            None,
            (
                str(args_cli.kit_args or ""),
                "--/persistent/physics/enableDeformableBeta=true",
            ),
        )
    )
args_cli.kit_args = isaac_kit_args(args_cli.kit_args, headless=args_cli.headless)

app_launcher = create_app_launcher(AppLauncher, args_cli)
simulation_app = app_launcher.app
enforce_isaac_viewport_only(simulation_app, headless=args_cli.headless)


def _abort_uncaught_startup_error(exc_type, exc_value, exc_traceback) -> None:
    """Make failures during post-Kit imports visible to the UI supervisor."""

    traceback.print_exception(exc_type, exc_value, exc_traceback)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)


# Most Isaac imports must happen after AppLauncher. They therefore execute
# before the normal ``main`` guard below; without this hook a failed extension
# import can be swallowed by Kit teardown and incorrectly reported as exit 0.
sys.excepthook = _abort_uncaught_startup_error


from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

enable_extension("isaacsim.asset.importer.mjcf")
enable_extension("omni.replicator.core")

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from omni.physx import get_physx_interface  # noqa: E402
from omni.physx.bindings._physx import ErrorEvent  # noqa: E402
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

from handson.sim.isaaclab.task_preview import (  # noqa: E402
    TASK_SPECS,
    create_success_condition_overlay,
    spawn_task_scene,
    validate_task_assets,
)
from handson.sim.isaaclab.user_make_pizza_long_horizon_scene import (  # noqa: E402
    hide_default_assembly_clutter,
    spawn_assets as spawn_pizza_assets,
)
from handson.sim.isaaclab.e_collection_scene import (  # noqa: E402
    E01_TASK_NAME,
    spawn_e01_garmentlab_surface_deformable,
)
from handson.sim.isaaclab.e02_rope_knot_scene import (  # noqa: E402
    E02_TASK_NAME,
    spawn_e02_surface_deformable_rope,
)
from handson.sim.isaaclab.e03_untie_rope_scene import (  # noqa: E402
    E03_TASK_NAME,
    spawn_e03_surface_deformable_rope,
)
from handson.sim.isaaclab.e04_cats_cradle_scene import (  # noqa: E402
    E04_TASK_NAME,
    spawn_e04_surface_deformable_string,
)
from handson.sim.isaaclab.e05_fold_wipe_scene import (  # noqa: E402
    E05_TASK_NAME,
    spawn_e05_surface_deformable_cloth,
)
from handson.sim.isaaclab.g01_deformable_asset_scene import (  # noqa: E402
    G01_2_PILLOW_SIM_POINT_COUNT,
    G01_2_TASK_NAME,
    G01_CARTON_TASK_NAMES,
    G01_COMPARISON_NODE_COUNTS,
    G01_COMPARISON_PROFILES,
    G01_PILLOW_BODY_PATH,
    G01_TASK_NAMES,
    G01_TOWEL_NODE_COUNT,
    G01_TOWEL_SURFACE_ROOT_PATH,
    spawn_g01_deformable_assets,
)
from handson.sim.isaaclab.ycb11_scene import (  # noqa: E402
    YCB11_SPONGE_ROOT_PATH,
    YCB11_TASK_NAME,
    spawn_ycb11_volume_deformable_sponge,
)
from handson.sim.isaaclab.t06_fit_device_cover_scene import (  # noqa: E402
    G02_TASK_NAME,
    spawn_g02_repair_assets,
)
from handson.sim.isaaclab.flexible_towel_double_fold_scene import (  # noqa: E402
    FLEXIBLE_TOWEL_TASK_NAME,
    spawn_flexible_towel_surface,
)


HEAD_CAMERA_EYE = (-0.675, 0.0, 1.5885)
HEAD_CAMERA_TARGET = (0.4957, 0.0, 0.8208)
HEAD_CAMERA_FOVY_DEG = 80.0
LIVE_PREVIEW_RESOLUTION = (1280, 960)
LIVE_PREVIEW_INTERVAL_STEPS = 12
# A file-backed remote preview is viewed from outside the center-head mesh,
# so give the free viewport a small -Y offset.  The straight-behind operator
# pose puts the center enclosure exactly between the camera and tabletop.
REMOTE_PREVIEW_CAMERA_EYE = (-1.30, -0.95, 1.68)
REMOTE_PREVIEW_CAMERA_TARGET = (-0.20, 0.0, 1.00)
REMOTE_D08_CAMERA_EYE = (-1.30, 0.90, 1.68)
REMOTE_D08_CAMERA_TARGET = (-0.35, 0.10, 1.00)


_FATAL_PHYSX_ERROR_TYPES = frozenset(
    {
        int(ErrorEvent.USD_LOAD_ERROR),
        int(ErrorEvent.PHYSX_ERROR),
        int(ErrorEvent.PHYSX_CUDA_ERROR),
        int(ErrorEvent.PHYSX_TOO_MANY_ERRORS),
    }
)


def _physx_error_detail(event: object) -> str:
    """Extract a bounded diagnostic from an Isaac PhysX error event."""

    payload = getattr(event, "payload", None)
    getter = getattr(payload, "get", None)
    for key in ("errorString", "error", "message", "description"):
        value = None
        if callable(getter):
            try:
                value = getter(key)
            except Exception:
                value = None
        if value is None:
            try:
                value = payload[key]
            except Exception:
                value = None
        if value:
            return str(value)[:512]
    return str(event)[:512]


def _install_physx_error_guard() -> tuple[dict[str, object], object]:
    """Subscribe to PhysX errors so preview cannot report a false pass."""

    state: dict[str, object] = {"fatal": False, "details": []}
    stream = get_physx_interface().get_error_event_stream()

    def _on_error(event) -> None:
        try:
            event_type = int(getattr(event, "type", -1))
        except (TypeError, ValueError):
            return
        if event_type not in _FATAL_PHYSX_ERROR_TYPES:
            return
        details = state["details"]
        if not isinstance(details, list):
            details = []
            state["details"] = details
        if len(details) < 8:
            details.append(_physx_error_detail(event))
        state["fatal"] = True

    subscription = stream.create_subscription_to_pop(_on_error)
    return state, subscription


def _raise_if_physx_error(state: dict[str, object], phase: str) -> None:
    """Turn an asynchronous PhysX error into a failed preview immediately."""

    if not bool(state.get("fatal", False)):
        return
    details = state.get("details")
    if isinstance(details, list) and details:
        detail_text = "; ".join(str(item) for item in details[:8])
    else:
        detail_text = "unspecified PhysX error"
    raise RuntimeError(f"PhysX error during preview {phase}: {detail_text}")


def live_preview_path() -> Path | None:
    raw = str(os.environ.get("HANDSON_LIVE_PREVIEW_PATH") or "").strip()
    return Path(raw).expanduser() if raw else None


def write_live_preview(path: Path, image_data: object) -> bool:
    """Atomically publish one Replicator RGB frame for the remote browser."""

    if isinstance(image_data, dict):
        image_data = image_data.get("data", image_data.get("rgb"))
    if image_data is None:
        return False
    array = np.asarray(image_data)
    if array.ndim != 3 or array.shape[2] < 3 or array.size == 0:
        return False
    array = array[:, :, :3]
    if np.issubdtype(array.dtype, np.floating):
        maximum = float(np.nanmax(array)) if array.size else 0.0
        if maximum <= 1.0:
            array = array * 255.0
        array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
    array = np.clip(array, 0, 255).astype(np.uint8, copy=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    Image.fromarray(array).save(temporary, format="PNG")
    os.replace(temporary, path)
    return True


def scene_stats(stage, root_path: str = "/World/Task") -> dict[str, object]:
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        raise RuntimeError(f"Imported task root is missing: {root_path}")

    rigid_bodies = []
    articulation_roots = []
    joints = 0
    render_meshes = 0
    surface_deformables = []
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) and not prim.IsInstanceProxy():
            rigid_bodies.append(prim)
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI) and not prim.IsInstanceProxy():
            articulation_roots.append(prim)
        if prim.IsA(UsdPhysics.Joint):
            joints += 1
        if prim.IsA(UsdGeom.Gprim) and "/visuals/" in str(prim.GetPath()):
            render_meshes += 1
    for prim in stage.Traverse():
        if prim.GetAttribute("physxDeformableBody:selfCollision").IsValid():
            surface_deformables.append(prim)
    return {
        "rigid_bodies": rigid_bodies,
        "articulation_roots": articulation_roots,
        "joint_count": joints,
        "render_mesh_count": render_meshes,
        "surface_deformables": surface_deformables,
    }


def validate_world_transforms(stats: dict[str, object]) -> tuple[bool, float]:
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    min_height = math.inf
    stable = True
    for prim in stats["rigid_bodies"]:
        transform = xform_cache.GetLocalToWorldTransform(prim)
        translation = transform.ExtractTranslation()
        values = (float(translation[0]), float(translation[1]), float(translation[2]))
        stable = stable and all(math.isfinite(value) for value in values)
        min_height = min(min_height, values[2])
    if min_height == math.inf:
        min_height = 0.0
    return stable and min_height > -2.0, min_height


def _preview_tensor_numpy(value: object) -> np.ndarray:
    if hasattr(value, "torch"):
        value = value.torch
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def _preview_soft_state(view: object, quantity: str) -> np.ndarray:
    names = (
        ("get_sim_nodal_positions", "get_simulation_nodal_positions")
        if quantity == "positions"
        else ("get_sim_nodal_velocities", "get_simulation_nodal_velocities")
    )
    for name in names:
        method = getattr(view, name, None)
        if callable(method):
            return _preview_tensor_numpy(method())
    raise RuntimeError(f"G01 preview {quantity} readback is unavailable")


def _create_g01_deformable_preview_views(
    sim: SimulationContext,
) -> tuple[tuple[str, object, int | None], ...]:
    if args_cli.task_name not in G01_TASK_NAMES:
        return ()
    physics_sim_view = sim.physics_sim_view
    if physics_sim_view is None:
        raise RuntimeError("G01 preview deformable simulation view is unavailable")
    pillow = physics_sim_view.create_volume_deformable_body_view(
        G01_PILLOW_BODY_PATH
    )
    views = [("Pillow024", pillow)]
    if args_cli.task_name not in G01_CARTON_TASK_NAMES:
        towel = physics_sim_view.create_surface_deformable_body_view(
            G01_TOWEL_SURFACE_ROOT_PATH
        )
        views.append(("Towl011", towel))
    for label, view in views:
        if view is None or int(view.count) != 1:
            raise RuntimeError(
                f"G01 preview {label} view count is "
                f"{0 if view is None else int(view.count)}, expected 1"
            )
    expected_pillow_nodes = (
        G01_COMPARISON_NODE_COUNTS[args_cli.task_name]
        if args_cli.task_name in G01_COMPARISON_PROFILES
        else G01_2_PILLOW_SIM_POINT_COUNT
        if args_cli.task_name == G01_2_TASK_NAME
        else None
    )
    output = [("Pillow024", pillow, expected_pillow_nodes)]
    if args_cli.task_name not in G01_CARTON_TASK_NAMES:
        output.append(("Towl011", towel, G01_TOWEL_NODE_COUNT))
    return tuple(output)


def validate_g01_deformable_states(
    views: tuple[tuple[str, object, int | None], ...],
) -> tuple[bool, float, float]:
    if not views:
        return True, math.inf, 0.0
    minimum_height = math.inf
    maximum_speed = 0.0
    for label, view, expected_nodes in views:
        positions = _preview_soft_state(view, "positions")
        velocities = _preview_soft_state(view, "velocities")
        if (
            positions.ndim != 3
            or positions.shape[0] != 1
            or positions.shape[2] != 3
        ):
            raise RuntimeError(
                f"G01 preview {label} nodal shape is {positions.shape}"
            )
        if expected_nodes is not None and positions.shape[1] != expected_nodes:
            raise RuntimeError(
                f"G01 preview {label} has {positions.shape[1]} nodes, "
                f"expected {expected_nodes}"
            )
        if velocities.shape != positions.shape:
            raise RuntimeError(
                f"G01 preview {label} velocity shape is {velocities.shape}"
            )
        if not np.all(np.isfinite(positions)) or not np.all(
            np.isfinite(velocities)
        ):
            raise RuntimeError(f"G01 preview {label} nodal state is not finite")
        nodes = positions[0]
        lower = np.min(nodes, axis=0)
        upper = np.max(nodes, axis=0)
        speeds = np.linalg.norm(velocities[0], axis=1)
        node_speed = float(np.max(speeds)) if speeds.size else math.inf
        if (
            np.any(lower < np.asarray([-2.0, -2.0, 0.50]))
            or np.any(upper > np.asarray([2.0, 2.0, 3.00]))
            or np.any(upper - lower > 2.0)
            or node_speed > 20.0
        ):
            raise RuntimeError(
                f"G01 preview {label} left its physical envelope: "
                f"bounds={lower.tolist()}..{upper.tolist()}, "
                f"max_speed={node_speed:.6f}m/s"
            )
        minimum_height = min(minimum_height, float(lower[2]))
        maximum_speed = max(maximum_speed, node_speed)
    return True, minimum_height, maximum_speed


def _create_ycb11_deformable_preview_view(
    sim: SimulationContext,
) -> object | None:
    """Create the native PhysX volume view for the YCB11 sponge.

    The MJCF scene intentionally contains only the rigid YCB mug.  YCB11's
    tracked sponge is authored after import as a volume deformable, so a
    standalone preview must create and validate its view explicitly instead of
    reporting a healthy scene with the target silently absent.
    """

    if args_cli.task_name != YCB11_TASK_NAME:
        return None
    physics_sim_view = sim.physics_sim_view
    if physics_sim_view is None:
        raise RuntimeError("YCB11 preview deformable simulation view is unavailable")
    try:
        view = physics_sim_view.create_volume_deformable_body_view(
            YCB11_SPONGE_ROOT_PATH
        )
    except Exception as exc:
        raise RuntimeError("YCB11 preview could not create the sponge volume view") from exc
    if view is None or int(getattr(view, "count", 0)) != 1:
        raise RuntimeError(
            "YCB11 preview sponge volume view count is "
            f"{0 if view is None else int(getattr(view, 'count', 0))}, expected 1"
        )
    return view


def validate_ycb11_deformable_state(
    view: object | None,
) -> tuple[bool, float, float]:
    """Fail closed when the native YCB11 sponge nodal state is missing/bad."""

    if view is None:
        return True, math.inf, 0.0
    positions = _preview_soft_state(view, "positions")
    velocities = _preview_soft_state(view, "velocities")
    if (
        positions.ndim != 3
        or positions.shape[0] != 1
        or positions.shape[2] != 3
        or positions.shape[1] < 32
    ):
        raise RuntimeError(
            f"YCB11 preview sponge nodal shape is {positions.shape}"
        )
    if velocities.shape != positions.shape:
        raise RuntimeError(
            f"YCB11 preview sponge velocity shape is {velocities.shape}"
        )
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocities)):
        raise RuntimeError("YCB11 preview sponge nodal state is not finite")
    nodes = positions[0]
    lower = np.min(nodes, axis=0)
    upper = np.max(nodes, axis=0)
    speeds = np.linalg.norm(velocities[0], axis=1)
    node_speed = float(np.max(speeds)) if speeds.size else math.inf
    # Keep the preview envelope broad enough for a seeded reset while still
    # catching an uninitialised/escaped PhysX volume before declaring PASSED.
    if (
        np.any(lower < np.asarray([-2.0, -2.0, -2.0]))
        or np.any(upper > np.asarray([2.0, 2.0, 3.0]))
        or np.any(upper - lower > 2.0)
        or node_speed > 20.0
    ):
        raise RuntimeError(
            "YCB11 preview sponge left its physical envelope: "
            f"bounds={lower.tolist()}..{upper.tolist()}, "
            f"max_speed={node_speed:.6f}m/s"
        )
    return True, float(lower[2]), node_speed


def validate_ycb11_deformable_states(
    view: object | None,
) -> tuple[bool, float, float]:
    """Plural compatibility alias matching the G01 preview validator API."""

    return validate_ycb11_deformable_state(view)


def main() -> int:
    spec = validate_task_assets(
        args_cli.task_name,
        hand_type=args_cli.hand_type,
    )
    preview_path = live_preview_path()
    # The generic preview also runs one scene. Apply all memory bounds to
    # every task, preserving its former solver partition count. Otherwise
    # tasks outside the list below can exhaust host RAM before sim.reset().
    physx_buffer_kwargs = bounded_single_env_gpu_physx_kwargs()
    physx_buffer_kwargs["gpu_max_num_partitions"] = 8
    if args_cli.task_name in {
        "g01_2_deformable_asset_inspection",
        "g01_4_soft_pillow_to_carton",
        "eval_flex03_pack_pillow",
        YCB11_TASK_NAME,
        "p004_hold_spring_clamp_open_insert_card",
        "p005_key_insert_turn_unlock_open_shackle",
        "p006_side_release_buckle_align_press",
    }:
        physx_buffer_kwargs.update(bounded_single_env_gpu_physx_kwargs())
    sim_cfg = sim_utils.SimulationCfg(
        device=args_cli.device,
        dt=1.0 / 120.0,
        render_interval=2,
        gravity=(0.0, 0.0, -9.81),
        physx=sim_utils.PhysxCfg(
            solver_type=1,
            max_position_iteration_count=64,
            max_velocity_iteration_count=4,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
            **physx_buffer_kwargs,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.8),
    )
    sim = SimulationContext(sim_cfg)
    # PhysX reports CUDA/narrowphase and USD-load failures through an event
    # stream instead of raising from ``step``.  Keep the subscription alive
    # for the whole preview so an asynchronous error cannot be followed by a
    # misleading ``preview PASSED`` line.
    physx_error_state, physx_error_subscription = _install_physx_error_guard()
    # The physical head optical origin sits inside the imported center-head
    # mesh.  Using it for the free viewport hides the whole task behind that
    # shell (a gray wall/black bar in remote previews).  Use a slight -Y
    # offset for the generic file-backed preview; D09 supplies its own
    # calibrated front-left override through the task spec.
    if preview_path is not None and args_cli.task_name == "d08_toast_bread":
        sim.set_camera_view(
            eye=REMOTE_D08_CAMERA_EYE,
            target=REMOTE_D08_CAMERA_TARGET,
        )
    elif preview_path is not None and args_cli.task_name != "t31_prepare_coffee":
        sim.set_camera_view(
            eye=REMOTE_PREVIEW_CAMERA_EYE,
            target=REMOTE_PREVIEW_CAMERA_TARGET,
        )
    else:
        sim.set_camera_view(eye=spec.camera_eye, target=spec.camera_target)

    print(f"[HandsOn/IsaacLab] Importing task {args_cli.task_name}...", flush=True)
    task_mjcf, (rebound, hidden) = spawn_task_scene(
        args_cli.task_name,
        hand_type=args_cli.hand_type,
        # This standalone preview imports the robot pair from the MJCF and
        # does not construct the native collection environment's replacement
        # arms.  The task-only placeholder is therefore unsafe here: it
        # leaves a long-horizon task with no robot articulation to drive and
        # can trip the GPU tensors/narrowphase on reset.  Collection envs set
        # ``first_physx_safe=True`` themselves after creating their native
        # arms; keep this diagnostic preview on the complete scene path.
        first_physx_safe=False,
    )
    _raise_if_physx_error(physx_error_state, "scene import")
    stage = omni.usd.get_context().get_stage()
    pizza_assets = None
    pizza_hidden_bootstrap_props: tuple[str, ...] = ()
    if args_cli.task_name == "user_make_pizza_long_horizon":
        # Standalone preview normally imports only the MJCF. Mount the exact
        # downloaded pizza assets here too, and remove generic assembly props
        # inherited from the shared bootstrap scene.
        pizza_hidden_bootstrap_props = hide_default_assembly_clutter(stage)
        pizza_assets = spawn_pizza_assets(stage, sim_utils)
        sim_utils.update_stage()
        print(
            "[HandsOn/IsaacLab] Pizza preview assets mounted: "
            f"{len(pizza_assets['assets'])} source assets; "
            f"hidden bootstrap props={len(pizza_hidden_bootstrap_props)}",
            flush=True,
        )
    if preview_path is not None:
        camera = UsdGeom.Camera(stage.GetPrimAtPath("/OmniverseKit_Persp"))
        vertical_aperture = float(
            camera.GetVerticalApertureAttr().Get() or 15.2908
        )
        focal_length = vertical_aperture / (
            2.0 * math.tan(math.radians(HEAD_CAMERA_FOVY_DEG) / 2.0)
        )
        camera.GetFocalLengthAttr().Set(focal_length)
    garment_asset = None
    rope_asset = None
    g01_assets = None
    g02_assets = None
    flexible_towel_asset = None
    ycb11_asset = None
    if args_cli.task_name == E01_TASK_NAME:
        garment_asset = spawn_e01_garmentlab_surface_deformable(stage)
        sim_utils.update_stage()
    elif args_cli.task_name == E02_TASK_NAME:
        rope_asset = spawn_e02_surface_deformable_rope(stage)
        sim_utils.update_stage()
    elif args_cli.task_name == E03_TASK_NAME:
        rope_asset = spawn_e03_surface_deformable_rope(stage)
        sim_utils.update_stage()
    elif args_cli.task_name == E04_TASK_NAME:
        rope_asset = spawn_e04_surface_deformable_string(stage)
        sim_utils.update_stage()
    elif args_cli.task_name == E05_TASK_NAME:
        garment_asset = spawn_e05_surface_deformable_cloth(stage)
        sim_utils.update_stage()
    elif args_cli.task_name in G01_TASK_NAMES:
        g01_assets = spawn_g01_deformable_assets(
            stage,
            task_name=args_cli.task_name,
        )
        sim_utils.update_stage()
    elif args_cli.task_name == G02_TASK_NAME:
        g02_assets = spawn_g02_repair_assets(stage, sim_utils)
        sim_utils.update_stage()
    elif args_cli.task_name == FLEXIBLE_TOWEL_TASK_NAME:
        flexible_towel_asset = spawn_flexible_towel_surface(stage, sim_utils)
        sim_utils.update_stage()
    elif args_cli.task_name == YCB11_TASK_NAME:
        # The YCB11 MJCF carries the rigid mug; mount the exact source-scale
        # sponge separately so the preview cannot silently omit the target.
        ycb11_asset = spawn_ycb11_volume_deformable_sponge(stage, sim_utils)
        sim_utils.update_stage()
    print(
        f"[HandsOn/IsaacLab] Source migration ready: {rebound} visual bindings repaired, "
        f"{hidden} collider meshes hidden\n  cached MJCF: {task_mjcf}",
        flush=True,
    )

    from handson.sim.isaaclab.default_appearance import update_preview, verify_preview

    sim.reset()
    _raise_if_physx_error(physx_error_state, "reset")
    g01_deformable_views = _create_g01_deformable_preview_views(sim)
    validate_g01_deformable_states(g01_deformable_views)
    ycb11_deformable_view = _create_ycb11_deformable_preview_view(sim)
    validate_ycb11_deformable_states(ycb11_deformable_view)
    preview_annotator = None
    preview_render_product = None
    if preview_path is not None:
        preview_render_product = rep.create.render_product(
            "/OmniverseKit_Persp",
            LIVE_PREVIEW_RESOLUTION,
        )
        preview_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        preview_annotator.attach([preview_render_product])
        print(
            f"[HandsOn/IsaacLab] Publishing head-camera frames to {preview_path}",
            flush=True,
        )
    stats = scene_stats(stage)
    print(
        f"[HandsOn/IsaacLab] Task {args_cli.task_name} ready: "
        f"{len(stats['articulation_roots'])} articulations, "
        f"{len(stats['rigid_bodies'])} rigid bodies, {stats['joint_count']} joints, "
        f"{stats['render_mesh_count']} render meshes, "
        f"{len(stats['surface_deformables'])} surface deformables"
        + (f", garment_asset={garment_asset}" if garment_asset is not None else "")
        + (f", rope_asset={rope_asset}" if rope_asset is not None else "")
        + (f", g01_assets={g01_assets}" if g01_assets is not None else "")
        + (f", g02_assets={g02_assets}" if g02_assets is not None else "")
        + (
            f", flexible_towel_asset={flexible_towel_asset}"
            if flexible_towel_asset is not None
            else ""
        )
        + (f", ycb11_asset={ycb11_asset}" if ycb11_asset is not None else ""),
        flush=True,
    )
    condition_overlay = None
    if not args_cli.headless:
        condition_overlay = create_success_condition_overlay(args_cli.task_name)

    step = 0
    while simulation_app.is_running() and (args_cli.max_steps <= 0 or step < args_cli.max_steps):
        if condition_overlay is not None:
            condition_overlay.visible = True
        update_preview(stage)
        sim.step()
        _raise_if_physx_error(physx_error_state, f"step {step + 1}")
        deformables_stable, deformable_min_height, deformable_max_speed = (
            validate_g01_deformable_states(g01_deformable_views)
        )
        ycb11_stable, ycb11_min_height, ycb11_max_speed = (
            validate_ycb11_deformable_states(ycb11_deformable_view)
        )
        step += 1
        if (
            preview_path is not None
            and preview_annotator is not None
            and (step == 1 or step % LIVE_PREVIEW_INTERVAL_STEPS == 0)
        ):
            write_live_preview(preview_path, preview_annotator.get_data())
        if args_cli.status_interval > 0 and step % args_cli.status_interval == 0:
            stable, min_height = validate_world_transforms(stats)
            deformable_detail = (
                f" min_deformable_z={deformable_min_height:.4f}m "
                f"max_deformable_speed={deformable_max_speed:.4f}m/s"
                if g01_deformable_views
                else ""
            )
            ycb11_detail = (
                f" min_ycb11_sponge_z={ycb11_min_height:.4f}m "
                f"max_ycb11_sponge_speed={ycb11_max_speed:.4f}m/s"
                if ycb11_deformable_view is not None
                else ""
            )
            print(
                f"[step {step:06d}] finite={stable and deformables_stable and ycb11_stable} "
                f"min_rigid_body_z={min_height:.4f}m{deformable_detail}{ycb11_detail}"
            )

    receipt = verify_preview(stage)
    if preview_path is not None:
        import json
        preview_path.with_suffix(".render.json").write_text(json.dumps(receipt, indent=2))
    _raise_if_physx_error(physx_error_state, "finalization")
    stable, min_height = validate_world_transforms(stats)
    deformables_stable, deformable_min_height, deformable_max_speed = (
        validate_g01_deformable_states(g01_deformable_views)
    )
    ycb11_stable, ycb11_min_height, ycb11_max_speed = (
        validate_ycb11_deformable_states(ycb11_deformable_view)
    )
    stable = bool(stable and deformables_stable and ycb11_stable)
    deformable_detail = (
        f", min_deformable_z={deformable_min_height:.4f}m, "
        f"max_deformable_speed={deformable_max_speed:.4f}m/s"
        if g01_deformable_views
        else ""
    )
    ycb11_detail = (
        f", min_ycb11_sponge_z={ycb11_min_height:.4f}m, "
        f"max_ycb11_sponge_speed={ycb11_max_speed:.4f}m/s"
        if ycb11_deformable_view is not None
        else ""
    )
    print(
        f"[HandsOn/IsaacLab] Task {args_cli.task_name} preview "
        f"{'PASSED' if stable else 'FAILED'} after {step} steps "
        f"(min_rigid_body_z={min_height:.4f}m{deformable_detail}{ycb11_detail})"
    )
    # Release the callback before Kit teardown.  On an exception the process
    # exits through the outer fail-closed handler, so retaining the callback
    # until shutdown is harmless and preserves the original traceback.
    try:
        physx_error_subscription.unsubscribe()
    except Exception:
        pass
    physx_error_subscription = None
    return 0 if stable else 1


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    except BaseException:
        traceback.print_exc()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        if exit_code != 0:
            # Kit's normal close path may terminate the interpreter with a
            # success status even after ``main`` raised (notably when PhysX
            # has poisoned the CUDA context).  Do not let teardown turn a
            # failed preview into a supervisor-visible false pass; the
            # process is already in a fail-closed state, so skip destructors
            # and preserve the non-zero status explicitly.
            os._exit(int(exit_code))
        shutdown_guard = threading.Timer(5.0, lambda: os._exit(exit_code))
        shutdown_guard.daemon = True
        shutdown_guard.start()
        try:
            simulation_app.close(wait_for_replicator=False)
        finally:
            shutdown_guard.cancel()
    raise SystemExit(exit_code)
