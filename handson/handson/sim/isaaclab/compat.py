"""Import-safe compatibility helpers for Isaac Lab 2 and 3.

Isaac Sim 5 uses Isaac Lab 2 with a built-in PhysX configuration, while
Isaac Sim 6 uses Isaac Lab 3 and selects the physics manager explicitly.
Keep those API differences out of the task runtime so the existing PhysX
backend and the A01 Newton backend can share one scene implementation.
"""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
import ctypes
import importlib
import importlib.metadata
import importlib.util
import os
import re
import sys
from pathlib import Path
from typing import Any, Literal, TypeVar


IsaacPhysicsBackend = Literal["physx", "newton"]
_T = TypeVar("_T")
_ISAAC_CUDA_PRIME_TENSOR: Any = None
_ISAAC_CUDA_PRIME_BYTES = 16 * 1024 * 1024


def is_isaaclab_v3() -> bool:
    """Return whether the split Isaac Lab 3 PhysX package is installed."""

    if "isaaclab_physx" in sys.modules:
        return True
    try:
        return importlib.util.find_spec("isaaclab_physx") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def is_lab3() -> bool:
    return is_isaaclab_v3()


def normalize_physics_backend(value: str | None) -> IsaacPhysicsBackend:
    backend = str(value or "physx").strip().lower().replace("-", "_")
    if backend in {"physx", "isaacsim", "isaac"}:
        return "physx"
    if backend in {
        "newton",
        "nvidia_newton",
        "isaacsim_newton",
        "isaac_newton",
    }:
        return "newton"
    raise ValueError(f"Unsupported Isaac physics backend: {value!r}")


def configure_app_launcher(args: Namespace) -> Namespace:
    """Select the Kit visualizer required by Isaac Lab 3."""

    if not hasattr(args, "visualizer"):
        return args
    if bool(getattr(args, "visualizer_explicit", False)) or args.visualizer is not None:
        return args
    args.visualizer = ["none"] if _host_requests_headless(args) else ["kit"]
    args.visualizer_explicit = True
    return args


def create_app_launcher(
    app_launcher_type: type[_T],
    launcher_args: Any,
    *,
    environ: Mapping[str, str] | None = None,
    cuda_primer: Callable[[str, int], Any] | None = None,
) -> _T:
    """Keep CUDA-logical physics and Vulkan-physical renderer IDs distinct."""

    if app_launcher_type.__module__.startswith("isaaclab."):
        from .calibrated_cameras import install
        install(app_launcher_type)
    source = os.environ if environ is None else environ
    raw_renderer_gpu = str(
        source.get("HANDSON_ISAACSIM_RENDERER_GPU", "")
    ).strip()
    if not raw_renderer_gpu:
        return app_launcher_type(launcher_args)
    if not raw_renderer_gpu.isdigit():
        raise ValueError(
            "HANDSON_ISAACSIM_RENDERER_GPU must be a non-negative integer, "
            f"got {raw_renderer_gpu!r}"
        )
    renderer_gpu = int(raw_renderer_gpu)

    if isinstance(launcher_args, Mapping):
        launch_device = str(launcher_args.get("device", "cuda:0"))
    else:
        launch_device = str(getattr(launcher_args, "device", "cuda:0"))
    if launch_device.startswith("cuda"):
        primer = _prime_isaac_cuda_device if cuda_primer is None else cuda_primer
        global _ISAAC_CUDA_PRIME_TENSOR
        _ISAAC_CUDA_PRIME_TENSOR = primer(
            launch_device,
            _ISAAC_CUDA_PRIME_BYTES,
        )

    class _PhysicalRendererAppLauncher(app_launcher_type):
        def _resolve_device_settings(self, resolved_args: dict[str, Any]) -> None:
            super()._resolve_device_settings(resolved_args)
            resolved_args["active_gpu"] = renderer_gpu

    return _PhysicalRendererAppLauncher(launcher_args)


def _prime_isaac_cuda_device(device: str, byte_count: int) -> Any:
    """Hold selected CUDA memory before Kit creates its small GPU 0 context."""

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("cannot prime the selected Isaac CUDA device: CUDA unavailable")
    tensor = torch.empty(int(byte_count), dtype=torch.uint8, device=device)
    torch.cuda.synchronize(device)
    print(
        f"[HandsOn/IsaacLab] Primed {byte_count} bytes on {device} before Kit startup",
        flush=True,
    )
    return tensor


def _enabled_environment_flag(name: str) -> bool:
    return os.environ.get(name, "0") in {"1", "2"}


def _host_requests_headless(args: Namespace) -> bool:
    return (
        bool(getattr(args, "headless", False))
        or os.environ.get("HEADLESS", "0") == "1"
        or int(getattr(args, "livestream", 0) or 0) in {1, 2}
        or _enabled_environment_flag("LIVESTREAM")
    )


def app_launcher_is_headless(args: Namespace) -> bool:
    if _host_requests_headless(args):
        return True
    if not hasattr(args, "visualizer"):
        return bool(getattr(args, "headless", False))
    if bool(getattr(args, "visualizer_explicit", False)):
        return "kit" not in (getattr(args, "visualizer", None) or [])
    return bool(getattr(args, "headless", False))


def _load_lab3_physx_cfg() -> type[Any] | None:
    try:
        module = importlib.import_module("isaaclab_physx.physics")
    except ModuleNotFoundError as exc:
        if exc.name not in {"isaaclab_physx", "isaaclab_physx.physics"}:
            raise
        return None
    return module.PhysxCfg


def _preload_newton_cuda_runtime() -> bool:
    """Make Isaac Sim's bundled CUDA 12 runtime visible to graph capture."""

    for soname in ("libcudart.so.12", "libcudart.so"):
        try:
            ctypes.CDLL(soname, mode=ctypes.RTLD_GLOBAL)
            return True
        except OSError:
            pass

    prefix = Path(sys.prefix)
    patterns = (
        "lib/python*/site-packages/isaacsim/extscache/"
        "omni.gpu_foundation-*/bin/deps/libcudart.so.12",
        "lib/python*/site-packages/isaacsim/extscache/"
        "omni.hydra.rtx-*/bin/deps/libcudart.so.12",
    )
    for pattern in patterns:
        for candidate in sorted(prefix.glob(pattern)):
            try:
                ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
                # Newton imports the runtime by SONAME.  Loading the absolute
                # Kit path first registers that SONAME with the dynamic loader.
                ctypes.CDLL("libcudart.so.12", mode=ctypes.RTLD_GLOBAL)
                return True
            except OSError:
                continue
    return False


def make_simulation_cfg(
    sim_utils: Any,
    *,
    physics_backend: str = "physx",
    physx_kwargs: Mapping[str, Any] | None = None,
    newton_solver_kwargs: Mapping[str, Any] | None = None,
    newton_deterministic_contacts: bool = False,
    newton_gpu_to_gpu: bool = False,
    newton_use_cuda_graph: bool = False,
    **simulation_kwargs: Any,
) -> Any:
    """Build a Lab 2/3 simulation config for PhysX or NVIDIA Newton."""

    if "physics" in simulation_kwargs or "physx" in simulation_kwargs:
        raise TypeError(
            "Pass backend settings through physx_kwargs/newton_solver_kwargs"
        )
    backend = normalize_physics_backend(physics_backend)
    lab3_physx_cfg = _load_lab3_physx_cfg()
    if backend == "newton":
        if lab3_physx_cfg is None:
            raise RuntimeError(
                "NVIDIA Newton requires Isaac Sim 6 / Isaac Lab 3 with "
                "isaaclab_newton installed."
            )
        try:
            mujoco_version = importlib.metadata.version("mujoco")
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                "NVIDIA Newton/MuJoCo-Warp requires mujoco>=3.8.0."
            ) from exc
        numeric_version = tuple(
            int(part) for part in re.findall(r"\d+", mujoco_version)[:3]
        )
        if numeric_version < (3, 8, 0):
            raise RuntimeError(
                "NVIDIA Newton/MuJoCo-Warp requires mujoco>=3.8.0; "
                f"the selected Isaac Sim 6 toolchain has {mujoco_version}. "
                "Upgrade that toolchain before launching the backend."
            )
        if newton_use_cuda_graph:
            _preload_newton_cuda_runtime()
        module = importlib.import_module("isaaclab_newton.physics")
        solver_settings = {
            # Newton 1.4 derives both capacities from the finalized model and
            # then derives the GPU_TO_GPU scatter-record bound from that live
            # allocation.  The previous single-environment values (8192 / 4096)
            # multiplied the deterministic record buffer past Warp's int32
            # address limit before B03's first step.
            "njmax": None,
            "nconmax": None,
            "iterations": 100,
            "ls_iterations": 50,
            "ccd_iterations": 50,
            "solver": "newton",
            "integrator": "implicitfast",
            "cone": "pyramidal",
            "tolerance": 1.0e-8,
            "use_mujoco_cpu": False,
            # MuJoCo-Warp's internal collision path forces per-geom gaps to
            # zero for NATIVECCD compatibility.  That defeats the Newton
            # contact envelope authored on A01's hand and task shapes and can
            # miss a fast finger/object crossing.  Let Newton generate the
            # contacts, then feed them to the same MJ-Warp constraint solver.
            "use_mujoco_contacts": False,
        }
        solver_settings.update(dict(newton_solver_kwargs or {}))
        collision_cfg = None
        if newton_deterministic_contacts:
            collision_cfg_type = getattr(
                module,
                "NewtonCollisionPipelineCfg",
                None,
            )
            if collision_cfg_type is None:
                raise RuntimeError(
                    "Deterministic Newton replay requires "
                    "NewtonCollisionPipelineCfg."
                )
            collision_cfg = collision_cfg_type(broad_phase="explicit")
            # Newton 1.2 already supports deterministic contact ordering, but
            # the Isaac Lab 3 beta config has not exposed that constructor
            # argument yet.  Configclass.to_dict() includes instance fields,
            # so this forwards the upstream option without patching the
            # installed Isaac Lab checkout.
            setattr(collision_cfg, "deterministic", True)
            pipeline_args = collision_cfg.to_pipeline_args()
            if pipeline_args.get("deterministic") is not True:
                raise RuntimeError(
                    "Newton collision configuration did not preserve "
                    "deterministic=True."
                )
        solver_cfg_type = module.MJWarpSolverCfg
        canonical_solver_class_type = getattr(
            solver_cfg_type(),
            "class_type",
            None,
        )
        gpu_to_gpu = None
        if newton_gpu_to_gpu:
            warp = importlib.import_module("warp")
            enum_type = getattr(warp, "DeterministicMode", None)
            gpu_to_gpu = getattr(enum_type, "GPU_TO_GPU", None)
            if gpu_to_gpu is None:
                raise RuntimeError(
                    "Newton replay requires "
                    "warp.DeterministicMode.GPU_TO_GPU."
                )
            if getattr(getattr(warp, "config", None), "deterministic", None) != gpu_to_gpu:
                raise RuntimeError(
                    "Warp GPU_TO_GPU must be configured before constructing "
                    "the Newton/MuJoCo-Warp solver."
                )
            # Isaac Lab beta's generic class_to_dict() recursively converts an
            # IntEnum into its private __dict__, so merely adding this field to
            # MJWarpSolverCfg would pass a dict to SolverMuJoCo.  Preserve the
            # actual Warp enum at this one serialization boundary.
            class GPUToGPUMJWarpSolverCfg(solver_cfg_type):
                def to_dict(self) -> dict[str, Any]:
                    value = super().to_dict()
                    value["deterministic"] = gpu_to_gpu
                    return value

            solver_cfg_type = GPUToGPUMJWarpSolverCfg
        solver_cfg = solver_cfg_type(**solver_settings)
        if gpu_to_gpu is not None:
            # ResolvableString expands ``{DIR}`` from the concrete config
            # class.  Our local serialization adapter must still resolve the
            # manager from isaaclab_newton, not from this compat module.
            if canonical_solver_class_type is not None:
                solver_cfg.class_type = canonical_solver_class_type
            setattr(solver_cfg, "deterministic", gpu_to_gpu)
            solver_dict = solver_cfg.to_dict()
            if solver_dict.get("deterministic") is not gpu_to_gpu:
                raise RuntimeError(
                    "MJWarpSolverCfg did not preserve deterministic=GPU_TO_GPU."
                )
        newton_cfg_kwargs = {
            "solver_cfg": solver_cfg,
            "num_substeps": 1,
            "debug_mode": False,
            # Keep the caller's execution choice explicit so dataset metadata
            # and fresh replay can bind the same graph/eager contract.
            "use_cuda_graph": bool(newton_use_cuda_graph),
        }
        if collision_cfg is not None:
            newton_cfg_kwargs["collision_cfg"] = collision_cfg
        physics = module.NewtonCfg(**newton_cfg_kwargs)
        return sim_utils.SimulationCfg(
            physics=physics,
            # Let Newton execute explicit actuator models directly inside the
            # solver.  A01 uses IdealPDActuatorCfg here; no per-substep
            # joint-state restoration is needed (or allowed).
            use_newton_actuators=True,
            **simulation_kwargs,
        )

    settings = dict(physx_kwargs or {})
    if lab3_physx_cfg is not None:
        return sim_utils.SimulationCfg(
            physics=lab3_physx_cfg(**settings),
            **simulation_kwargs,
        )
    return sim_utils.SimulationCfg(
        physx=sim_utils.PhysxCfg(**settings),
        **simulation_kwargs,
    )


def configure_newton_single_environment() -> None:
    """Register the non-cloned A01 stage as one Newton environment.

    Isaac Lab 3 normally fills ``NewtonManager._num_envs`` through its scene
    cloner.  HandsOn authors one interactive world directly under ``/World``,
    so that callback is intentionally absent.  The solver did not need the
    count for implicit drives, but the native explicit-actuator adapter does.
    The cloner also registers MuJoCo-Warp custom attributes before importing
    USD; without that step, authored ``mjc:gravcomp`` and
    ``mjc:actuatorgravcomp`` values are silently absent from the finalized
    model.  Build the same registered flat-stage builder before the first
    simulation reset.
    """

    module = importlib.import_module("isaaclab_newton.physics.newton_manager")
    manager = module.NewtonManager
    num_envs = manager.get_num_envs()
    if num_envs not in {None, 1}:
        raise RuntimeError(
            "The A01 Newton runtime requires exactly one environment; "
            f"NewtonManager reports {num_envs}."
        )
    manager._num_envs = 1

    builder = getattr(manager, "_builder", None)
    body_count = int(getattr(builder, "body_count", 0) or 0)
    custom_attributes = getattr(builder, "custom_attributes", {})
    has_mujoco_attributes = any(
        str(name).startswith("mujoco:") for name in custom_attributes
    )
    if body_count > 0:
        if not has_mujoco_attributes:
            raise RuntimeError(
                "The direct A01 Newton builder imported USD before MuJoCo-Warp "
                "custom attributes were registered."
            )
        return

    newton = importlib.import_module("newton")
    if builder is None:
        builder = manager.create_builder()
    if not has_mujoco_attributes:
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

    sim_utils = importlib.import_module("isaaclab.sim")
    stage = sim_utils.get_current_stage()
    if stage is None:
        raise RuntimeError("Cannot initialize the A01 Newton builder without a USD stage")
    schema_module = importlib.import_module("newton._src.usd.schemas")
    builder.add_usd(
        stage,
        schema_resolvers=[
            schema_module.SchemaResolverNewton(),
            schema_module.SchemaResolverPhysx(),
        ],
    )
    manager.set_builder(builder)
    warp = importlib.import_module("warp")
    manager._world_xforms = [warp.transform()]


def qualify_newton_mujoco_gpu_to_gpu_solver() -> dict[str, int]:
    """Replace Newton's overly dense actuator record bound with an exact one.

    Newton 1.4 conservatively treats every actuator moment row as dense.  A01
    and B03 contain only joint transmissions, whose row width is exactly the
    target joint's DOF count.  The dense fallback makes Warp request more than
    2^31 scatter records for the bimanual model and cannot execute.  Validate
    every transmission before installing the smaller, model-derived bound;
    unknown/state-dependent transmissions fail closed.
    """

    manager_module = importlib.import_module(
        "isaaclab_newton.physics.newton_manager"
    )
    manager = manager_module.NewtonManager
    solver = getattr(manager, "_solver", None)
    if solver is None:
        raise RuntimeError("Newton SolverMuJoCo is not initialized.")
    model = getattr(solver, "mj_model", None)
    data = getattr(solver, "mjw_data", None)
    if model is None or data is None:
        raise RuntimeError("Newton SolverMuJoCo has no finalized MuJoCo model.")

    mujoco = importlib.import_module("mujoco")
    joint_transmissions = {
        int(mujoco.mjtTrn.mjTRN_JOINT),
        int(mujoco.mjtTrn.mjTRN_JOINTINPARENT),
    }
    joint_width = {
        int(mujoco.mjtJoint.mjJNT_FREE): 6,
        int(mujoco.mjtJoint.mjJNT_BALL): 3,
        int(mujoco.mjtJoint.mjJNT_SLIDE): 1,
        int(mujoco.mjtJoint.mjJNT_HINGE): 1,
    }
    actuator_width = 0
    unsupported_transmissions: set[int] = set()
    for actuator_id in range(int(model.nu)):
        transmission = int(model.actuator_trntype[actuator_id])
        if transmission not in joint_transmissions:
            unsupported_transmissions.add(transmission)
            continue
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        if not 0 <= joint_id < int(model.njnt):
            raise RuntimeError(
                f"Newton actuator {actuator_id} targets invalid joint {joint_id}."
            )
        actuator_width = max(
            actuator_width,
            joint_width[int(model.jnt_type[joint_id])],
        )
    if unsupported_transmissions:
        raise RuntimeError(
            "GPU_TO_GPU's sparse actuator bound supports only fixed joint "
            "transmissions; found MuJoCo transmission types "
            f"{sorted(unsupported_transmissions)}."
        )

    # The imported robot MJCF carries legacy joint/wrist force sensors, but
    # HandsOn observes Newton articulation state and CollisionPipeline contact
    # buffers directly.  MuJoCo-Warp 3.10's tactile/force sensor kernel mixes
    # max and add reductions on one array, which Warp intentionally rejects in
    # deterministic mode.  Sensors are read-only and do not affect dynamics,
    # so disable their evaluation in both host and device model options.
    sensor_disable_bit = int(mujoco.mjtDisableBit.mjDSBL_SENSOR)
    model.opt.disableflags = int(model.opt.disableflags) | sensor_disable_bit
    warp_model = getattr(solver, "mjw_model", None)
    if warp_model is None:
        raise RuntimeError("Newton SolverMuJoCo has no MuJoCo-Warp model.")
    warp_model.opt.disableflags = (
        int(warp_model.opt.disableflags) | sensor_disable_bit
    )

    chain_widths: list[int] = []
    seen_welded_bodies: set[int] = set()
    for body_id in range(int(model.nbody)):
        welded_body_id = int(model.body_weldid[body_id])
        if welded_body_id in seen_welded_bodies:
            continue
        seen_welded_bodies.add(welded_body_id)
        dof_id = (
            int(model.body_dofadr[welded_body_id])
            + int(model.body_dofnum[welded_body_id])
            - 1
        )
        width = 0
        while dof_id >= 0:
            width += 1
            dof_id = int(model.dof_parentid[dof_id])
        chain_widths.append(width)
    chain_widths.sort(reverse=True)
    body_pair_width = sum(chain_widths[:2])
    tendon_pair_width = 2 * max(
        (int(width) for width in model.ten_J_rownnz),
        default=0,
    )
    flex_width = max(
        (int(width) for width in model.flexedge_J_rownnz),
        default=0,
    )
    row_width = min(
        int(model.nv),
        max(
            1,
            body_pair_width,
            tendon_pair_width,
            flex_width,
            actuator_width,
        ),
    )
    hessian_records = row_width * (row_width + 1) // 2
    tendon_records = 0
    for path_size, width in zip(
        model.tendon_num,
        model.ten_J_rownnz,
        strict=True,
    ):
        segment_count = max(int(path_size) - 1, 1)
        tendon_records = max(
            tendon_records,
            2 * segment_count * int(width),
        )
    max_records = max(
        1,
        int(data.njmax),
        hessian_records,
        tendon_records,
    )
    solver._deterministic_max_records = max_records
    solver._set_mujoco_warp_module_options()
    solver._prepare_generated_kernels()
    return {
        "solver_deterministic_max_records": max_records,
        "solver_constraint_capacity": int(data.njmax),
        "solver_sparse_row_width": row_width,
        "solver_actuator_row_width": actuator_width,
        "solver_disabled_read_only_sensors": int(model.nsensor),
    }


def reset_newton_mujoco_solver_history() -> dict[str, Any]:
    """Clear MJWarp warm-start/input buffers without changing authored state."""

    module = importlib.import_module("isaaclab_newton.physics")
    manager = getattr(module, "NewtonManager", None)
    solver = getattr(manager, "_solver", None)
    reset = getattr(solver, "reset", None)
    state = getattr(manager, "_state_0", None)
    if manager is None or not callable(reset) or state is None:
        raise RuntimeError(
            "Newton/MuJoCo-Warp solver history reset is unavailable."
        )
    # flags=0 deliberately preserves the just-authored joint_q/joint_qd while
    # SolverMuJoCo clears qacc_warmstart, applied forces, ctrl and act.
    reset(state, world_mask=None, flags=0)
    return {
        "strategy": "solver_mujoco_internal_buffers_flags_zero",
        "flags": 0,
        "preserves_authored_joint_state": True,
        "performed_on_last_reset": True,
    }


def _config_declares_field(config_type: type[Any], field_name: str) -> bool:
    for base in config_type.__mro__:
        if field_name in base.__dict__.get("__dataclass_fields__", {}):
            return True
        if field_name in base.__dict__.get("__annotations__", {}):
            return True
    return False


def make_mjcf_file_cfg(
    sim_utils: Any,
    *,
    import_sites: bool = True,
    **kwargs: Any,
) -> Any:
    """Build ``MjcfFileCfg`` without Lab 3's removed ``import_sites`` field."""

    config_type = sim_utils.MjcfFileCfg
    if _config_declares_field(config_type, "import_sites"):
        kwargs["import_sites"] = import_sites
    return config_type(**kwargs)


def isaacsim_cache_namespace() -> str:
    try:
        sim_version = importlib.metadata.version("isaacsim")
    except importlib.metadata.PackageNotFoundError:
        sim_version = "unknown"
    safe_version = re.sub(r"[^A-Za-z0-9_.-]+", "-", sim_version).strip("-")
    return f"isaacsim-{safe_version or 'unknown'}-{'lab3' if is_lab3() else 'lab2'}-mjcf-v1"


def mjcf_robot_articulation_root_prim_path(side: str) -> str:
    if side not in {"left", "right"}:
        raise ValueError(f"Unsupported robot side: {side!r}")
    return "/Geometry" if is_lab3() else f"/base_{side}"


def mjcf_robot_root_link_path(asset_prim_path: str, side: str) -> str:
    if side not in {"left", "right"}:
        raise ValueError(f"Unsupported robot side: {side!r}")
    suffix = f"Geometry/base_{side}" if is_lab3() else f"base_{side}/base_{side}"
    return f"{asset_prim_path.rstrip('/')}/{suffix}"


def mjcf_full_scene_fix_base() -> bool | None:
    return None if is_lab3() else False


def initialize_interactive_scene(sim: Any, scene: Any) -> None:
    initialize_renderers = getattr(scene, "initialize_renderers", None)
    if callable(initialize_renderers):
        initialize_renderers()
    register_scene = getattr(sim, "register_interactive_scene", None)
    if callable(register_scene):
        register_scene(scene)


def isaaclab_uses_xyzw() -> bool:
    return is_isaaclab_v3()


def _quaternion(value: Sequence[_T]) -> tuple[_T, _T, _T, _T]:
    if len(value) != 4:
        raise ValueError(f"Expected a quaternion with 4 values, got {len(value)}")
    return value[0], value[1], value[2], value[3]


def quat_wxyz_to_isaaclab(value: Sequence[_T]) -> tuple[_T, _T, _T, _T]:
    w, x, y, z = _quaternion(value)
    return (x, y, z, w) if isaaclab_uses_xyzw() else (w, x, y, z)


def quat_isaaclab_to_wxyz(value: Sequence[_T]) -> tuple[_T, _T, _T, _T]:
    first, second, third, fourth = _quaternion(value)
    if isaaclab_uses_xyzw():
        return fourth, first, second, third
    return first, second, third, fourth


__all__ = [
    "IsaacPhysicsBackend",
    "app_launcher_is_headless",
    "configure_app_launcher",
    "qualify_newton_mujoco_gpu_to_gpu_solver",
    "reset_newton_mujoco_solver_history",
    "configure_newton_single_environment",
    "initialize_interactive_scene",
    "isaaclab_uses_xyzw",
    "isaacsim_cache_namespace",
    "is_lab3",
    "is_isaaclab_v3",
    "make_mjcf_file_cfg",
    "make_simulation_cfg",
    "mjcf_full_scene_fix_base",
    "mjcf_robot_articulation_root_prim_path",
    "mjcf_robot_root_link_path",
    "normalize_physics_backend",
    "quat_isaaclab_to_wxyz",
    "quat_wxyz_to_isaaclab",
]
