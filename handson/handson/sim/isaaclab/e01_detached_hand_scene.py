"""Isaac-ready canonical Wuji hand assets for the E01 live runtime.

This module deliberately has no top-level Isaac imports.  Interface tests can
therefore validate the exact MJCF that will be imported without starting Kit.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Any

from ..genesis.mjcf_assets import prepare_detached_rokae_wuji_hand_mjcf
from .compat import is_lab3, make_mjcf_file_cfg


_XML_DIR = Path(__file__).resolve().parents[1] / "envs" / "xmls"
_DEFAULT_CACHE_DIR = (
    Path.home() / ".cache" / "handson" / "isaaclab" / "e01_detached_hands"
)

E01_DETACHED_HAND_HOME_POSES = {
    "right": (
        -0.3715,
        -0.2395,
        0.9975,
        0.70710678,
        0.0,
        0.70710678,
        0.0,
    ),
    "left": (
        -0.3715,
        0.2395,
        0.9975,
        0.70710678,
        0.0,
        0.70710678,
        0.0,
    ),
}
# Matches the canonical A-D tracker/TCP home in
# ``bimanual_assembly_env.py`` without importing that runtime module here.
DIRECT_WUJI_HAND_HOME_POSES = {
    "right": (-0.10, -0.16, 1.07, 0.0, 1.0, 0.0, 0.0),
    "left": (-0.10, 0.16, 1.07, 0.0, 1.0, 0.0, 0.0),
}
E01_WUJI_HAND_DIM = 20
E01_WUJI_HAND_HOME = (0.10,) + (0.0,) * (E01_WUJI_HAND_DIM - 1)
# Tracker motion is limited to 0.06 m/s and about 0.52 rad/s.  These broader
# PhysX caps leave contact-recovery headroom without allowing a collision
# impulse to launch the floating hand at the importer defaults (8/20).
E01_HAND_MAX_LINEAR_VELOCITY = 0.5
E01_HAND_MAX_ANGULAR_VELOCITY = 5.0
E01_HAND_CONTACT_OFFSET = 0.002
E01_HAND_REST_OFFSET = 0.0005
E01_HAND_MAX_DEPENETRATION_VELOCITY = 0.20
# The simplified post-IK collector writes a floating dynamic hand state every
# physics substep.  The old importer defaults (100 m/s and 100 rad/s) allowed a
# contact impulse with a tray wall to launch the hand out of the scene.  These
# caps still exceed normal Tracker motion while keeping contact recovery tame.
E01_DIRECT_HAND_MAX_LINEAR_VELOCITY = 2.0
E01_DIRECT_HAND_MAX_ANGULAR_VELOCITY = 10.0
E01_HAND_SOLVER_POSITION_ITERATIONS = 32
E01_HAND_SOLVER_VELOCITY_ITERATIONS = 4


def detached_wuji_articulation_root_path(side: str) -> str:
    """Return the root suffix authored by Isaac's MJCF converter."""

    if side not in {"right", "left"}:
        raise ValueError(f"Unsupported Wuji side: {side!r}")
    palm = f"{side}_palm_link"
    if is_lab3():
        return f"/Geometry/{palm}"
    return f"/{palm}/{palm}"


def wxyz_to_xyzw(quaternion: tuple[float, ...]) -> tuple[float, ...]:
    """Convert the repository's quaternion convention to IsaacLab 3.x."""

    if len(quaternion) != 4:
        raise ValueError(f"Expected a four-value quaternion, got {quaternion!r}")
    w, x, y, z = (float(value) for value in quaternion)
    return (x, y, z, w)


def isaaclab_uses_xyzw_quaternions(initial_state_cfg_type: Any) -> bool:
    """Detect IsaacLab 2.x (WXYZ) versus 3.x (XYZW) from its identity.

    IsaacLab 3.0 deliberately changed both asset configuration and runtime
    articulation tensors to XYZW.  Inspecting the public InitialStateCfg
    identity keeps this E01 runtime usable with the installed 2.3 toolchain
    while making the selected 3.0 toolchain explicit and safe.
    """

    identity = tuple(float(value) for value in initial_state_cfg_type().rot)
    if identity == (0.0, 0.0, 0.0, 1.0):
        return True
    if identity == (1.0, 0.0, 0.0, 0.0):
        return False
    raise RuntimeError(
        "Unrecognized IsaacLab InitialStateCfg quaternion convention: "
        f"identity={identity!r}"
    )


def wxyz_to_isaaclab_quaternion(
    quaternion: tuple[float, ...],
    *,
    uses_xyzw: bool,
) -> tuple[float, ...]:
    values = tuple(float(value) for value in quaternion)
    if len(values) != 4:
        raise ValueError(f"Expected a four-value quaternion, got {quaternion!r}")
    return wxyz_to_xyzw(values) if uses_xyzw else values


def _cache_dir(cache_dir: Path | None = None) -> Path:
    configured = os.environ.get("HANDSON_ISAACLAB_CACHE", "").strip()
    if cache_dir is not None:
        return cache_dir.expanduser().resolve()
    if configured:
        return Path(configured).expanduser().resolve() / "e01_detached_hands"
    return _DEFAULT_CACHE_DIR.resolve()


def _relative_mesh_paths(root: ET.Element, output_dir: Path) -> None:
    """Rewrite absolute helper output for Isaac's MJCF importer.

    Isaac Sim 5.1 treats a leading slash in an MJCF mesh filename as relative.
    Real relative paths are consequently required even when the source helper
    intentionally emitted absolute paths for Genesis.
    """

    compiler = root.find("compiler")
    if compiler is not None:
        compiler.attrib["meshdir"] = "."
    for mesh in root.iter("mesh"):
        filename = mesh.attrib.get("file", "").strip()
        if not filename:
            continue
        path = Path(filename)
        if not path.is_absolute():
            path = (output_dir / path).resolve()
        mesh.attrib["file"] = os.path.relpath(path.resolve(), output_dir)


def prepare_e01_detached_wuji_hand_mjcf(
    side: str,
    *,
    cache_dir: Path | None = None,
    ensure_proximal_finger_collisions: bool = False,
) -> Path:
    """Return an Isaac-compatible, arm-free copy of the original black hand."""

    if side not in {"right", "left"}:
        raise ValueError(f"Unsupported Wuji side: {side!r}")
    output_dir = _cache_dir(cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source = _XML_DIR / f"rokae_wuji1_{side}.xml"
    primitive = _XML_DIR / "wuji_hand1" / "mjcf" / f"{side}_primitive.xml"
    generated = prepare_detached_rokae_wuji_hand_mjcf(
        source,
        primitive,
        side=side,
        collision_profile="full",
        cache_dir=output_dir / "source",
        ensure_proximal_finger_collisions=ensure_proximal_finger_collisions,
    )

    tree = ET.parse(generated)
    root = tree.getroot()
    _relative_mesh_paths(root, output_dir)
    content = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    digest = hashlib.sha256(
        content
        + b"\0isaac-detached-wuji-v1"
        + (
            b"\0proximal-finger-collisions-v1"
            if ensure_proximal_finger_collisions
            else b""
        )
    ).hexdigest()[:12]
    output = output_dir / f"rokae_wuji1_{side}_detached_hand_isaac_{digest}.xml"
    if not output.exists() or output.read_bytes() != content:
        output.write_bytes(content)
    return output


def canonical_wuji_hand_drive_maps(
    side: str,
    *,
    hand_mjcf: Path | None = None,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Read the exact per-joint position drives from the canonical MJCF.

    Keeping these values tied to the source asset avoids replacing the soft
    Wuji finger drives with a single high-gain PhysX drive.  In particular,
    the authored force limits are only about 0.29--1.30 Nm.
    """

    if side not in {"right", "left"}:
        raise ValueError(f"Unsupported Wuji side: {side!r}")
    path = hand_mjcf or prepare_e01_detached_wuji_hand_mjcf(side)
    root = ET.parse(path).getroot()
    stiffness: dict[str, float] = {}
    damping: dict[str, float] = {}
    effort: dict[str, float] = {}
    actuator_root = root.find("actuator")
    if actuator_root is not None:
        for actuator in actuator_root:
            joint_name = actuator.attrib.get("joint", "")
            if not joint_name.startswith(f"{side}_finger"):
                continue
            try:
                force_range = tuple(
                    float(value)
                    for value in actuator.attrib["forcerange"].split()
                )
                stiffness[joint_name] = float(actuator.attrib["kp"])
                damping[joint_name] = float(actuator.attrib["kv"])
            except (KeyError, ValueError) as exc:
                raise ValueError(
                    f"Incomplete canonical drive for {joint_name!r} in {path}"
                ) from exc
            if len(force_range) != 2:
                raise ValueError(
                    f"Expected a two-value force range for {joint_name!r}, "
                    f"got {force_range!r}"
                )
            effort[joint_name] = max(abs(force_range[0]), abs(force_range[1]))

    expected = {
        f"{side}_finger{finger}_joint{joint}"
        for finger in range(1, 6)
        for joint in range(1, 5)
    }
    if set(stiffness) != expected or set(damping) != expected or set(effort) != expected:
        missing = sorted(expected - set(stiffness))
        unexpected = sorted(set(stiffness) - expected)
        raise ValueError(
            f"Canonical {side} Wuji drive map is incomplete: "
            f"missing={missing!r}, unexpected={unexpected!r}"
        )
    return stiffness, damping, effort


def detached_wuji_hand_cfg(side: str, *, prim_path: str) -> Any:
    """Build the floating-palm Isaac Lab articulation used only by E01."""

    if side not in {"right", "left"}:
        raise ValueError(f"Unsupported Wuji side: {side!r}")

    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg

    hand_mjcf = prepare_e01_detached_wuji_hand_mjcf(side)
    stiffness, damping, effort = canonical_wuji_hand_drive_maps(
        side,
        hand_mjcf=hand_mjcf,
    )
    joint_pos = {
        f"{side}_finger[2-5]_joint[1-4]": 0.0,
        f"{side}_finger1_joint[2-4]": 0.0,
        f"{side}_finger1_joint1": 0.10,
    }
    uses_xyzw = isaaclab_uses_xyzw_quaternions(
        ArticulationCfg.InitialStateCfg
    )
    return ArticulationCfg(
        prim_path=prim_path,
        # Isaac's MJCF converter nests the model default prim and the MJCF
        # root body even when both share the palm name.  Point the tensor
        # articulation at the actual authored body, not the outer Xform.
        articulation_root_prim_path=detached_wuji_articulation_root_path(side),
        spawn=make_mjcf_file_cfg(
            sim_utils,
            asset_path=str(hand_mjcf),
            usd_dir=str(_cache_dir() / "usd" / hand_mjcf.stem),
            fix_base=False,
            import_sites=True,
            self_collision=False,
            make_instanceable=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=E01_HAND_MAX_LINEAR_VELOCITY,
                max_angular_velocity=E01_HAND_MAX_ANGULAR_VELOCITY,
                max_depenetration_velocity=(
                    E01_HAND_MAX_DEPENETRATION_VELOCITY
                ),
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=(
                    E01_HAND_SOLVER_POSITION_ITERATIONS
                ),
                solver_velocity_iteration_count=(
                    E01_HAND_SOLVER_VELOCITY_ITERATIONS
                ),
                sleep_threshold=0.0,
                stabilization_threshold=0.0001,
            ),
            joint_drive_props=sim_utils.JointDrivePropertiesCfg(
                drive_type="force"
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=E01_DETACHED_HAND_HOME_POSES[side][:3],
            rot=wxyz_to_isaaclab_quaternion(
                E01_DETACHED_HAND_HOME_POSES[side][3:],
                uses_xyzw=uses_xyzw,
            ),
            joint_pos=joint_pos,
            joint_vel={".*": 0.0},
        ),
        actuators={
            "hand": ImplicitActuatorCfg(
                joint_names_expr=[f"{side}_finger[1-5]_joint[1-4]"],
                effort_limit_sim=effort,
                stiffness=stiffness,
                damping=damping,
                friction=0.0,
            )
        },
        soft_joint_pos_limit_factor=1.0,
    )


def direct_kinematic_wuji_hand_cfg(
    side: str,
    *,
    prim_path: str,
    ensure_proximal_finger_collisions: bool = False,
    state_driven_dynamic_articulation: bool = False,
    initial_pose_wxyz: tuple[float, ...] | None = None,
) -> Any:
    """Build the arm-free Wuji used by Newton's direct-hand collector.

    Unlike :func:`detached_wuji_hand_cfg`, this articulation has no actuator
    model.  Its floating palm and all finger coordinates are written from the
    Tracker/MANUS command immediately before every Newton substep.  Marking the
    imported rigid bodies kinematic lets them push task objects without being
    pushed away by the solver in Newton.  PhysX cannot build finger joints
    between kinematic bodies, so the A-D simplified profile keeps a floating,
    gravity-free articulation and deterministically overwrites its root and
    finger state before every substep instead.  ``initial_pose_wxyz`` is an
    optional physical palm-root override for guidance modes that must share a
    formal collector's frame-0 hand pose.
    """

    if side not in {"right", "left"}:
        raise ValueError(f"Unsupported Wuji side: {side!r}")

    import isaaclab.sim as sim_utils
    from isaaclab.assets import ArticulationCfg

    from .compat import make_mjcf_file_cfg

    home_pose = tuple(
        float(value)
        for value in (
            DIRECT_WUJI_HAND_HOME_POSES[side]
            if initial_pose_wxyz is None
            else initial_pose_wxyz
        )
    )
    if len(home_pose) != 7:
        raise ValueError(
            f"{side} direct-hand initial_pose_wxyz must have seven values"
        )

    hand_mjcf = prepare_e01_detached_wuji_hand_mjcf(
        side,
        ensure_proximal_finger_collisions=(
            ensure_proximal_finger_collisions
        ),
    )
    uses_xyzw = isaaclab_uses_xyzw_quaternions(
        ArticulationCfg.InitialStateCfg
    )
    joint_pos = {
        f"{side}_finger[2-5]_joint[1-4]": 0.0,
        f"{side}_finger1_joint[2-4]": 0.0,
        # The canonical Wuji thumb-abduction joint has a positive lower
        # limit (about 0.047 rad); use the asset's established safe home.
        f"{side}_finger1_joint1": 0.10,
    }
    return ArticulationCfg(
        prim_path=prim_path,
        # Isaac Sim 6's USD converter authors the articulation at
        # ``Geometry/<side>_palm_link`` while the older converter used the
        # doubly nested palm suffix consumed by ``detached_wuji_hand_cfg``.
        # Let IsaacLab discover the sole ArticulationRootAPI below prim_path
        # so the Newton-only direct hand follows the USD actually generated
        # by the selected toolchain.
        articulation_root_prim_path=(
            detached_wuji_articulation_root_path(side)
            if state_driven_dynamic_articulation
            else None
        ),
        spawn=make_mjcf_file_cfg(
            sim_utils,
            asset_path=str(hand_mjcf),
            usd_dir=str(_cache_dir() / "usd" / hand_mjcf.stem),
            fix_base=False,
            self_collision=False,
            make_instanceable=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=not state_driven_dynamic_articulation,
                disable_gravity=True,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=(
                    E01_DIRECT_HAND_MAX_LINEAR_VELOCITY
                    if state_driven_dynamic_articulation
                    else 100.0
                ),
                max_angular_velocity=(
                    E01_DIRECT_HAND_MAX_ANGULAR_VELOCITY
                    if state_driven_dynamic_articulation
                    else 100.0
                ),
                max_depenetration_velocity=E01_HAND_MAX_DEPENETRATION_VELOCITY,
                enable_gyroscopic_forces=False,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=1,
                solver_velocity_iteration_count=1,
                sleep_threshold=0.0,
                stabilization_threshold=0.0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=home_pose[:3],
            rot=wxyz_to_isaaclab_quaternion(
                home_pose[3:],
                uses_xyzw=uses_xyzw,
            ),
            joint_pos=joint_pos,
            joint_vel={".*": 0.0},
        ),
        actuators={},
        soft_joint_pos_limit_factor=1.0,
    )


__all__ = [
    "DIRECT_WUJI_HAND_HOME_POSES",
    "E01_DETACHED_HAND_HOME_POSES",
    "E01_HAND_CONTACT_OFFSET",
    "E01_HAND_MAX_ANGULAR_VELOCITY",
    "E01_HAND_MAX_DEPENETRATION_VELOCITY",
    "E01_HAND_MAX_LINEAR_VELOCITY",
    "E01_DIRECT_HAND_MAX_ANGULAR_VELOCITY",
    "E01_DIRECT_HAND_MAX_LINEAR_VELOCITY",
    "E01_HAND_REST_OFFSET",
    "E01_HAND_SOLVER_POSITION_ITERATIONS",
    "E01_HAND_SOLVER_VELOCITY_ITERATIONS",
    "E01_WUJI_HAND_DIM",
    "E01_WUJI_HAND_HOME",
    "canonical_wuji_hand_drive_maps",
    "detached_wuji_articulation_root_path",
    "detached_wuji_hand_cfg",
    "direct_kinematic_wuji_hand_cfg",
    "isaaclab_uses_xyzw_quaternions",
    "prepare_e01_detached_wuji_hand_mjcf",
    "wxyz_to_isaaclab_quaternion",
    "wxyz_to_xyzw",
]
