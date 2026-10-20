"""Scene contract for user-provided LW native-USD articulation tasks.

The source USDs remain byte-identical to the supplied archive.  This module
authors only the integration layer: world placement, stable table/fixtures,
visible grasp sleeves, contact reporting, and fixed-root joints where the task
requires a mounted center body.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .lw_native_articulation_contract import G08_LID_HARD_STOP_DEG
from .gen3_hinged_trash_bin_scene import (
    GEN3_TASK_NAME, GEN3_SUPPORT_GAP_M, GEN3_SUCCESS_LABEL,
    validate_gen3_asset_files, configure_gen3_dynamics,
)


LW_SCENE_XML = "arena_arm_hand_lw_native_articulation_rokae_wuji1.xml"
LW_TASK_ROOT = "/World/Task/LWNativeArticulation"
LW_TABLE_PATH = f"{LW_TASK_ROOT}/table"
LW_TABLE_COLLIDER_PATH = f"{LW_TABLE_PATH}/geometry/mesh"
LW_TABLE_TOP_Z = 0.890
LW_PHYSICS_HZ = 120
LW_CONTROL_HZ = 30
LW_ASSET_DIRECTORY = (
    Path(__file__).resolve().parent.parent
    / "envs"
    / "usd"
    / "lw_native_articulation"
)
LW_MANIFEST_PATH = LW_ASSET_DIRECTORY / "manifest.json"

G05_TASK_NAME = "g05_folding_fan_bimanual_open"
G08_TASK_NAME = "t18_open_micropipette_box"
G10_TASK_NAME = "t49_unfold_ruler"
LW_TASK_NAMES = (G05_TASK_NAME, G08_TASK_NAME, G10_TASK_NAME, GEN3_TASK_NAME)


@dataclass(frozen=True)
class LWAssetTaskSpec:
    task_id: str
    task_name: str
    relative_usd: str
    asset_prim_name: str
    root_body_name: str
    left_body_name: str
    right_body_name: str
    dof_names: tuple[str, ...]
    translation: tuple[float, float, float]
    orientation_wxyz: tuple[float, float, float, float]
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    fixed_root: bool = False
    ccd: bool = False

    @property
    def source_path(self) -> Path:
        return LW_ASSET_DIRECTORY / self.relative_usd

    @property
    def asset_path(self) -> str:
        return f"{LW_TASK_ROOT}/{self.asset_prim_name}"

    def body_path(self, name: str) -> str:
        return f"{self.asset_path}/{name}"

    @property
    def root_body_path(self) -> str:
        return self.body_path(self.root_body_name)

    @property
    def left_body_path(self) -> str:
        return self.body_path(self.left_body_name)

    @property
    def right_body_path(self) -> str:
        return self.body_path(self.right_body_name)


# IsaacLab's ``UsdFileCfg`` takes quaternions in WXYZ order.  Keep the box in
# the source frame: its local +Y hinge then sits on the +Y (left-hand) side of
# the calibrated head-camera view, while the lid's free edge sits on -Y
# (right-hand side).  The source joint's negative X rotation consequently
# opens the free edge from right to left, matching the physical hand roles.
_IDENTITY_ORIENTATION = (1.0, 0.0, 0.0, 0.0)
_FAN_TABLETOP_ORIENTATION = (0.5, -0.5, -0.5, 0.5)
_G08_CONTACT_OFFSET_M = 0.002
# These seven authored convex hulls are rejected by the Isaac Sim 5.1 GPU
# convex cooker.  The four lid entries include the broad top panel and two
# long rims, so accepting PhysX's CPU fallback leaves most of the visible lid
# outside the active GPU contact path.  Their local bounding boxes remain
# tightly fitted to the authored thin wall/panel meshes.
_G08_GPU_INCOMPATIBLE_COLLIDER_NAMES = frozenset(
    {
        "MicropipetteBox001_Collider14",
        "MicropipetteBox001_Collider22",
        "MicropipetteBox001_Collider23",
        "MicropipetteBox001_Lid001_Collider4",
        "MicropipetteBox001_Lid001_Collider6",
        "MicropipetteBox001_Lid001_Collider7",
        "MicropipetteBox001_Lid001_Collider9",
    }
)
_G08_SOLVER_POSITION_ITERATIONS = 64
_G08_SOLVER_VELOCITY_ITERATIONS = 8
_G08_MAX_DEPENETRATION_VELOCITY_MPS = 0.40
_G08_ASSET_UNIFORM_SCALE = 0.40
_G08_SOURCE_BOTTOM_Z_M = -0.1364005804061890
_G08_AUTHORED_SUPPORT_GAP_M = 0.00025
_G08_SPAWN_Z_M = (
    LW_TABLE_TOP_Z
    + _G08_AUTHORED_SUPPORT_GAP_M
    - _G08_SOURCE_BOTTOM_Z_M * _G08_ASSET_UNIFORM_SCALE
)
# The source USD already assigns realistic masses (700 g base, 300 g lid), but
# its lid COM is ``(-inf, -inf, -inf)`` and its diagonal inertia is zero.  The
# old finite-value repair consequently put the COM on the hinge.  That removes
# the real gravity lever arm and leaves only the small centroid inertia, so a
# light hand contact can fling the lid through its full travel.
_G08_BASE_MASS_KG = 0.700
_G08_LID_MASS_KG = 0.300
# Measured from the source lid's composed visual envelope, expressed in the
# lid rigid body's unscaled local frame.  The asset-root scale turns this into
# the physical hinge-to-COM lever arm in the simulation.
_G08_LID_COM_SOURCE_LOCAL_M = (
    0.0,
    -0.2411680296,
    0.0275114477,
)
# Measured source-local lid envelope.  MassAPI inertia is authored in final
# kg*m^2 units, whereas its body-local COM is transformed by the ancestor
# scale.  Derive both base/lid physical sizes from the one scene scale so a
# future visual-size change cannot silently retain stale inertia.
_G08_LID_SOURCE_SIZE_M = (
    0.846138,
    0.492128,
    0.063034,
)
_G08_LID_COMPOSED_SIZE_M = tuple(
    dimension * _G08_ASSET_UNIFORM_SCALE
    for dimension in _G08_LID_SOURCE_SIZE_M
)
_G08_BASE_COMPOSED_SIZE_M = tuple(
    dimension * _G08_ASSET_UNIFORM_SCALE
    for dimension in (0.850, 0.450, 0.140)
)
_G08_LID_ANGULAR_DAMPING = 2.0
# Preserve the source author's intended joint-friction value, but apply the
# PhysxJointAPI explicitly so PhysX 5.1 actually consumes the opinion.
_G08_LID_JOINT_FRICTION = 0.10
_G08_LID_DYNAMICS_POLICY_VERSION = "g08_measured_lid_dynamics_v3_scale040"
_G10_CONTACT_OFFSET_M = 0.002
_G10_ROOT_SOLVER_POSITION_ITERATIONS = 64
_G10_ROOT_SOLVER_VELOCITY_ITERATIONS = 8
_G10_ROOT_MAX_DEPENETRATION_VELOCITY_MPS = 0.40
# G10's root deliberately remains free.  The source USD does not author a
# usable passive hinge-damping policy, so a tiny tabletop tilt can keep one
# light outer segment moving for the full recording-origin timeout.  Link
# damping plus a zero-stiffness, zero-velocity damping drive dissipate that
# motion without a position target, scripted state write, or world constraint.
# The force cap and joint friction match the independently audited P003 folding
# ruler policy and remain readily back-drivable by a Wuji pinch.
_G10_LINK_LINEAR_DAMPING = 2.0
_G10_LINK_ANGULAR_DAMPING = 8.0
_G10_JOINT_FRICTION = 0.03
_G10_DRIVE_STIFFNESS = 0.0
_G10_DRIVE_DAMPING = 0.05
_G10_DRIVE_MAX_FORCE_NM = 0.15
_G10_DYNAMICS_POLICY_VERSION = "g10_free_root_passive_hinge_damping_v2"
# The proxy cuboids are authored in each outer-link local frame.  At z=.004 m
# their lower face (and the 2 mm contact shell) crossed the z=.890 m table at
# the audited reset pose; z=.010 m leaves several millimetres of clearance.
_G10_GRIP_PROXY_LOCAL_Z_M = 0.010

# The supplied LW USDs carry zero diagonal inertias and non-finite centers of
# mass on their rigid-body prims.  PhysX 5.1 accepts the stage long enough to
# build an articulation, then the first randomized contact can enter an
# invalid CUDA narrowphase state.  Keep the source bytes immutable and author
# finite integration-layer values before the first SimulationContext.reset().
_LW_MASS_PROPERTY_POLICY_VERSION = "lw_native_finite_mass_properties_v1"
# PhysX 5.1 GPU does not safely consume an authored ``density=0`` opinion on
# a referenced articulation once an explicit MassAPI override is added.  The
# explicit mass below remains authoritative; this positive density is only a
# finite fallback for the USD schema/scene builder.
_LW_DENSITY_FALLBACK_KG_M3 = 1000.0


def _box_diagonal_inertia(
    mass_kg: float,
    size_m: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Return the principal inertia of a conservative integration box."""

    mass = max(float(mass_kg), 1.0e-6)
    length, width, height = (max(float(value), 1.0e-4) for value in size_m)
    values = (
        mass * (width * width + height * height) / 12.0,
        mass * (length * length + height * height) / 12.0,
        mass * (length * length + width * width) / 12.0,
    )
    return tuple(max(float(value), 1.0e-8) for value in values)


def _lw_mass_fallbacks(
    spec: LWAssetTaskSpec,
    prim_name: str,
) -> tuple[float, tuple[float, float, float]]:
    """Return finite mass/inertia for one source rigid-body link.

    G10 and the G08 lid use measured source-link envelopes.  G08/G05 values
    are expressed in the already-composed (scaled) frame; they are only an
    integration fallback for invalid source attributes and do not change
    visual or collision geometry.
    """

    if spec.task_name == G10_TASK_NAME:
        masses = {
            "Ruler035": 0.050,
            "Ruler035_Prop001": 0.010,
            "Ruler035_Prop002": 0.010,
        }
        sizes = {
            "Ruler035": (0.24960, 0.01498, 0.00400),
            "Ruler035_Prop001": (0.24961, 0.01498, 0.00400),
            "Ruler035_Prop002": (0.22465, 0.01498, 0.00400),
        }
        mass = masses.get(prim_name, 0.010)
        size = sizes.get(prim_name, (0.225, 0.015, 0.004))
    elif spec.task_name == G08_TASK_NAME:
        # Preserve the source masses.  Use the measured physical lid envelope
        # so the centroid inertia and the separately authored hinge-to-COM
        # lever arm reproduce a realistic resistance to opening.
        mass = (
            _G08_BASE_MASS_KG
            if prim_name == "MicropipetteBox001"
            else _G08_LID_MASS_KG
        )
        size = (
            _G08_BASE_COMPOSED_SIZE_M
            if prim_name == "MicropipetteBox001"
            else _G08_LID_COMPOSED_SIZE_M
        )
    else:
        # G05's source fan links are very light and narrow.  The fallback is
        # deliberately small but non-singular; authored positive values are
        # retained whenever the source is already finite.
        mass = 0.015 if prim_name == "Foldingfan001" else 0.005
        size = (0.030, 0.180, 0.004)
    return float(mass), _box_diagonal_inertia(mass, size)


def _lw_mass_fallback_center(
    spec: LWAssetTaskSpec,
    prim_name: str,
) -> tuple[float, float, float]:
    """Return the measured body-local COM used when source COM is invalid.

    G08 and G10 write ``(-inf, -inf, -inf)`` for their source bodies.  A zero
    COM is numerically finite, but it is not physically neutral for a body
    whose geometry is offset from its joint frame.  Keep measured body-local
    offsets in the integration layer so the repair cannot silently remove
    gravity torque or create an unrealistically small joint inertia.
    """

    if (
        spec.task_name == G08_TASK_NAME
        and prim_name == "MicropipetteBox001_Lid001"
    ):
        return _G08_LID_COM_SOURCE_LOCAL_M
    if spec.task_name == G10_TASK_NAME:
        return {
            "Ruler035": (-0.0000045514, 0.0000023065, -0.0020844589),
            "Ruler035_Prop001": (0.0950109794, 0.0000119934, 0.0001900034),
            "Ruler035_Prop002": (-0.0885262708, 0.0000072944, 0.0001878154),
        }.get(prim_name, (0.0, 0.0, 0.0))
    return (0.0, 0.0, 0.0)


def _g08_lid_effective_hinge_inertia_kg_m2(
    spec: LWAssetTaskSpec,
) -> float:
    """Return the repaired lid inertia about its local-X hinge axis."""

    mass, centroid_inertia = _lw_mass_fallbacks(spec, spec.right_body_name)
    center = _lw_mass_fallback_center(spec, spec.right_body_name)
    offset_y = float(center[1]) * abs(float(spec.scale[1]))
    offset_z = float(center[2]) * abs(float(spec.scale[2]))
    return float(
        centroid_inertia[0]
        + mass * (offset_y * offset_y + offset_z * offset_z)
    )


def _sanitize_lw_mass_properties(
    stage: Any,
    spec: LWAssetTaskSpec,
    root: Any,
) -> int:
    """Author finite mass properties on every composed LW rigid body.

    This is intentionally a pre-PhysX composition step.  It does not write
    any source USD file and it never clamps velocities or teleports a body;
    it only replaces invalid mass/inertia/COM sentinels that otherwise make
    the GPU articulation undefined at first contact.
    """

    from pxr import Gf, Usd, UsdPhysics

    # G05 has a separate, already-audited fan mass/drive path.  Keep this
    # repair scoped to the two tasks being repaired so an unrelated fan
    # dynamics contract cannot change as a side effect.
    if spec.task_name not in (G08_TASK_NAME, G10_TASK_NAME):
        return 0

    tuned = 0
    for prim in Usd.PrimRange(root):
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        mass_api = (
            UsdPhysics.MassAPI(prim)
            if prim.HasAPI(UsdPhysics.MassAPI)
            else UsdPhysics.MassAPI.Apply(prim)
        )
        if not mass_api:
            raise RuntimeError(
                f"LW finite-mass policy could not apply MassAPI: {prim.GetPath()}"
            )
        fallback_mass, fallback_inertia = _lw_mass_fallbacks(
            spec,
            str(prim.GetName()),
        )

        mass_attr = mass_api.GetMassAttr()
        authored_mass = mass_attr.Get() if mass_attr and mass_attr.IsValid() else None
        try:
            mass = float(authored_mass)
        except (TypeError, ValueError, OverflowError):
            mass = fallback_mass
        if not math.isfinite(mass) or mass <= 0.0:
            mass = fallback_mass
        mass_api.CreateMassAttr().Set(float(mass))
        # Explicit mass owns the body.  Keep density finite and positive as
        # well: a zero density is a common MJCF-to-USD sentinel, but on Isaac
        # Sim 5.1 GPU it can make the articulation view's mass upload fail
        # after an integration-layer MassAPI override.  Because Mass is
        # authored, this fallback does not replace the task's measured mass.
        density_attr = mass_api.CreateDensityAttr()
        authored_density = density_attr.Get()
        try:
            density = float(authored_density)
        except (TypeError, ValueError, OverflowError):
            density = _LW_DENSITY_FALLBACK_KG_M3
        if not math.isfinite(density) or density <= 0.0:
            density = _LW_DENSITY_FALLBACK_KG_M3
        density_attr.Set(float(density))

        center_attr = mass_api.CreateCenterOfMassAttr()
        center_value = center_attr.Get()
        center_valid = False
        try:
            center_values = tuple(float(value) for value in center_value)
            center_valid = len(center_values) == 3 and all(
                math.isfinite(value) for value in center_values
            )
        except (TypeError, ValueError, OverflowError):
            center_valid = False
        if not center_valid:
            center_attr.Set(Gf.Vec3f(*_lw_mass_fallback_center(spec, str(prim.GetName()))))

        inertia_attr = mass_api.CreateDiagonalInertiaAttr()
        inertia_value = inertia_attr.Get()
        inertia_valid = False
        try:
            inertia_values = tuple(float(value) for value in inertia_value)
            inertia_valid = len(inertia_values) == 3 and all(
                math.isfinite(value) and value > 1.0e-8
                for value in inertia_values
            )
        except (TypeError, ValueError, OverflowError):
            inertia_valid = False
        if not inertia_valid:
            # Scale the known fallback inertia by the authored/source mass
            # ratio when a finite mass was present on the source prim.  This
            # preserves relative dynamics while repairing only the singular
            # matrix.
            inertia_attr.Set(
                Gf.Vec3f(
                    *tuple(
                        max(
                            1.0e-8,
                            value * (mass / max(fallback_mass, 1.0e-8)),
                        )
                        for value in fallback_inertia
                    )
                )
            )

        principal_attr = mass_api.CreatePrincipalAxesAttr()
        principal_value = principal_attr.Get()
        principal_valid = False
        try:
            real = float(principal_value.GetReal())
            imaginary = tuple(float(value) for value in principal_value.GetImaginary())
            principal_valid = (
                math.isfinite(real)
                and len(imaginary) == 3
                and all(math.isfinite(value) for value in imaginary)
                and math.sqrt(real * real + sum(value * value for value in imaginary))
                > 1.0e-8
            )
        except (AttributeError, TypeError, ValueError, OverflowError):
            principal_valid = False
        if not principal_valid:
            principal_attr.Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))

        # Fail closed if a binding silently rejected an authored integration
        # opinion.  This check runs before any PhysX reset and is intentionally
        # independent of a later tensor readback.
        try:
            checked_mass = float(mass_api.GetMassAttr().Get())
            checked_inertia = tuple(
                float(value)
                for value in mass_api.GetDiagonalInertiaAttr().Get()
            )
            checked_com = tuple(
                float(value) for value in mass_api.GetCenterOfMassAttr().Get()
            )
        except (AttributeError, TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(
                f"LW finite-mass readback failed: {prim.GetPath()}"
            ) from exc
        if not (
            math.isfinite(checked_mass)
            and checked_mass > 0.0
            and len(checked_inertia) == 3
            and all(math.isfinite(value) and value > 1.0e-8 for value in checked_inertia)
            and len(checked_com) == 3
            and all(math.isfinite(value) for value in checked_com)
        ):
            raise RuntimeError(
                "LW finite-mass policy produced a non-finite/singular readback: "
                f"{prim.GetPath()} mass={checked_mass!r} "
                f"inertia={checked_inertia!r} com={checked_com!r}"
            )

        prim.SetCustomDataByKey(
            "handson:finiteMassProperties",
            _LW_MASS_PROPERTY_POLICY_VERSION,
        )
        tuned += 1

    root.SetCustomDataByKey(
        "handson:finiteMassPropertiesBodyCount",
        int(tuned),
    )
    root.SetCustomDataByKey(
        "handson:finiteMassPropertiesPolicy",
        _LW_MASS_PROPERTY_POLICY_VERSION,
    )
    return tuned


def _configure_g08_lid_dynamics(
    stage: Any,
    spec: LWAssetTaskSpec,
) -> dict[str, float | str]:
    """Author and verify the physical G08 lid damping/friction policy.

    The immutable source USD contains a raw ``physxJoint:jointFriction``
    attribute but does not apply ``PhysxJointAPI`` to the revolute joint, so
    PhysX can ignore it.  Apply the schema in the composed stage and add
    modest rigid-body angular damping.  Neither term drives the joint or
    clamps its velocity; hand contact and gravity remain the only sources of
    lid motion.
    """

    if spec.task_name != G08_TASK_NAME:
        return {}

    from pxr import PhysxSchema, UsdPhysics

    lid_prim = stage.GetPrimAtPath(spec.right_body_path)
    if not lid_prim.IsValid() or not lid_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError(
            f"G08 lid rigid body is missing before dynamics repair: {spec.right_body_path}"
        )
    rigid_api = (
        PhysxSchema.PhysxRigidBodyAPI(lid_prim)
        if lid_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI)
        else PhysxSchema.PhysxRigidBodyAPI.Apply(lid_prim)
    )
    rigid_api.CreateAngularDampingAttr().Set(float(_G08_LID_ANGULAR_DAMPING))

    joint_path = f"{spec.right_body_path}/{spec.dof_names[0]}"
    joint_prim = stage.GetPrimAtPath(joint_path)
    if not joint_prim.IsValid() or not joint_prim.IsA(UsdPhysics.RevoluteJoint):
        raise RuntimeError(
            f"G08 lid revolute joint is missing before dynamics repair: {joint_path}"
        )
    joint_api = (
        PhysxSchema.PhysxJointAPI(joint_prim)
        if joint_prim.HasAPI(PhysxSchema.PhysxJointAPI)
        else PhysxSchema.PhysxJointAPI.Apply(joint_prim)
    )
    joint_api.CreateJointFrictionAttr().Set(float(_G08_LID_JOINT_FRICTION))

    angular_damping = float(rigid_api.GetAngularDampingAttr().Get())
    joint_friction = float(joint_api.GetJointFrictionAttr().Get())
    if not (
        math.isclose(
            angular_damping,
            float(_G08_LID_ANGULAR_DAMPING),
            rel_tol=0.0,
            abs_tol=1.0e-6,
        )
        and math.isclose(
            joint_friction,
            float(_G08_LID_JOINT_FRICTION),
            rel_tol=0.0,
            abs_tol=1.0e-6,
        )
    ):
        raise RuntimeError(
            "G08 composed lid dynamics opinion did not stick: "
            f"angular_damping={angular_damping!r}, "
            f"joint_friction={joint_friction!r}"
        )

    effective_hinge_inertia = _g08_lid_effective_hinge_inertia_kg_m2(spec)
    lid_prim.SetCustomDataByKey(
        "handson:g08LidDynamicsPolicy",
        _G08_LID_DYNAMICS_POLICY_VERSION,
    )
    lid_prim.SetCustomDataByKey(
        "handson:g08EffectiveHingeInertiaKgM2",
        float(effective_hinge_inertia),
    )
    joint_prim.SetCustomDataByKey(
        "handson:g08LidDynamicsPolicy",
        _G08_LID_DYNAMICS_POLICY_VERSION,
    )
    return {
        "policy": _G08_LID_DYNAMICS_POLICY_VERSION,
        "angular_damping": angular_damping,
        "joint_friction": joint_friction,
        "effective_hinge_inertia_kg_m2": effective_hinge_inertia,
    }


def _configure_g10_ruler_dynamics(
    stage: Any,
    spec: LWAssetTaskSpec,
) -> dict[str, Any]:
    """Author and verify passive settling for the free-root G10 ruler.

    No position target or fixed joint is added.  Linear/angular link damping,
    joint friction, and a force-limited zero-stiffness velocity damper let the
    randomly positioned articulation reach an honest recording origin while
    remaining freely movable by the operator.
    """

    if spec.task_name != G10_TASK_NAME:
        return {}

    from pxr import PhysxSchema, UsdPhysics

    body_readback: dict[str, dict[str, float]] = {}
    for body_name in (
        spec.left_body_name,
        spec.root_body_name,
        spec.right_body_name,
    ):
        body_path = f"{spec.asset_path}/{body_name}"
        body_prim = stage.GetPrimAtPath(body_path)
        if not body_prim.IsValid() or not body_prim.HasAPI(
            UsdPhysics.RigidBodyAPI
        ):
            raise RuntimeError(
                f"G10 rigid link is missing before passive dynamics repair: {body_path}"
            )
        rigid_api = (
            PhysxSchema.PhysxRigidBodyAPI(body_prim)
            if body_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI)
            else PhysxSchema.PhysxRigidBodyAPI.Apply(body_prim)
        )
        rigid_api.CreateLinearDampingAttr().Set(
            float(_G10_LINK_LINEAR_DAMPING)
        )
        rigid_api.CreateAngularDampingAttr().Set(
            float(_G10_LINK_ANGULAR_DAMPING)
        )
        linear = float(rigid_api.GetLinearDampingAttr().Get())
        angular = float(rigid_api.GetAngularDampingAttr().Get())
        if not (
            math.isclose(
                linear,
                float(_G10_LINK_LINEAR_DAMPING),
                rel_tol=0.0,
                abs_tol=1.0e-6,
            )
            and math.isclose(
                angular,
                float(_G10_LINK_ANGULAR_DAMPING),
                rel_tol=0.0,
                abs_tol=1.0e-6,
            )
        ):
            raise RuntimeError(
                "G10 passive link damping opinion did not stick: "
                f"{body_path} linear={linear!r} angular={angular!r}"
            )
        body_prim.SetCustomDataByKey(
            "handson:g10DynamicsPolicy",
            _G10_DYNAMICS_POLICY_VERSION,
        )
        body_readback[body_name] = {
            "linear_damping": linear,
            "angular_damping": angular,
        }

    joint_paths = {
        "Prop001_joint": (
            f"{spec.asset_path}/Ruler035_Prop001/Prop001_joint"
        ),
        "Prop002_joint": (
            f"{spec.asset_path}/Ruler035_Prop002/Prop002_joint"
        ),
    }
    joint_readback: dict[str, dict[str, Any]] = {}
    for dof_name in spec.dof_names:
        joint_path = joint_paths[dof_name]
        joint_prim = stage.GetPrimAtPath(joint_path)
        if not joint_prim.IsValid() or not joint_prim.IsA(
            UsdPhysics.RevoluteJoint
        ):
            raise RuntimeError(
                f"G10 revolute joint is missing before passive dynamics repair: {joint_path}"
            )
        drive = UsdPhysics.DriveAPI.Get(joint_prim, "angular")
        if not drive:
            drive = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
        drive.CreateTypeAttr().Set("force")
        drive.CreateStiffnessAttr().Set(float(_G10_DRIVE_STIFFNESS))
        drive.CreateDampingAttr().Set(float(_G10_DRIVE_DAMPING))
        drive.CreateMaxForceAttr().Set(float(_G10_DRIVE_MAX_FORCE_NM))
        drive.CreateTargetVelocityAttr().Set(0.0)
        drive_readback = {
            "type": str(drive.GetTypeAttr().Get()),
            "stiffness": float(drive.GetStiffnessAttr().Get()),
            "damping": float(drive.GetDampingAttr().Get()),
            "max_force_nm": float(drive.GetMaxForceAttr().Get()),
            "target_velocity_deg_s": float(
                drive.GetTargetVelocityAttr().Get()
            ),
        }
        joint_api = (
            PhysxSchema.PhysxJointAPI(joint_prim)
            if joint_prim.HasAPI(PhysxSchema.PhysxJointAPI)
            else PhysxSchema.PhysxJointAPI.Apply(joint_prim)
        )
        joint_api.CreateJointFrictionAttr().Set(float(_G10_JOINT_FRICTION))
        friction = float(joint_api.GetJointFrictionAttr().Get())
        if not (
            math.isclose(
                friction,
                float(_G10_JOINT_FRICTION),
                rel_tol=0.0,
                abs_tol=1.0e-6,
            )
            and drive_readback["type"] == "force"
            and math.isclose(
                drive_readback["stiffness"],
                _G10_DRIVE_STIFFNESS,
                abs_tol=1.0e-6,
            )
            and math.isclose(
                drive_readback["damping"],
                _G10_DRIVE_DAMPING,
                abs_tol=1.0e-6,
            )
            and math.isclose(
                drive_readback["max_force_nm"],
                _G10_DRIVE_MAX_FORCE_NM,
                abs_tol=1.0e-6,
            )
            and math.isclose(
                drive_readback["target_velocity_deg_s"],
                0.0,
                abs_tol=1.0e-6,
            )
        ):
            raise RuntimeError(
                "G10 passive hinge dynamics opinion did not stick: "
                f"{joint_path} friction={friction!r} drive={drive_readback!r}"
            )
        joint_prim.SetCustomDataByKey(
            "handson:g10DynamicsPolicy",
            _G10_DYNAMICS_POLICY_VERSION,
        )
        joint_readback[dof_name] = {
            "friction": friction,
            "drive": drive_readback,
        }

    return {
        "policy": _G10_DYNAMICS_POLICY_VERSION,
        "body": body_readback,
        "joint": joint_readback,
    }


def _repair_g10_reflected_collision_scales(
    stage: Any,
    spec: LWAssetTaskSpec,
) -> int:
    """Remove the two reflected source hull scales in the G10 integration layer.

    The supplied ruler USD uses a negative uniform scale on each small
    ``Collider2`` mesh.  Its determinant is negative, so PhysX sees a
    reflected, right-handed hull with reversed winding.  That is legal in a
    USD viewport but is a known CUDA narrowphase hazard for thin meshes.  Keep
    the source asset immutable and author the same-magnitude positive scale on
    the composed prims.  The AABB and visual geometry are unchanged, while the
    cooked hull has a positive determinant.  Fail closed if an override does
    not stick before the first PhysX reset.
    """

    if spec.task_name != G10_TASK_NAME:
        return 0
    from pxr import Gf, Usd

    repaired = 0
    root = stage.GetPrimAtPath(spec.asset_path)
    for prim in Usd.PrimRange(root):
        if prim.GetName() not in {
            "Ruler035_prop001_Collider2",
            "Ruler035_prop002_Collider2",
        }:
            continue
        scale_attr = prim.GetAttribute("xformOp:scale")
        if not scale_attr or not scale_attr.IsValid():
            raise RuntimeError(
                f"G10 reflected collider has no scale attribute: {prim.GetPath()}"
            )
        value = scale_attr.Get()
        try:
            values = tuple(float(component) for component in value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(
                f"G10 reflected collider scale is unreadable: {prim.GetPath()}"
            ) from exc
        if len(values) != 3 or not all(math.isfinite(component) for component in values):
            raise RuntimeError(
                f"G10 reflected collider scale is non-finite: {prim.GetPath()}"
            )
        if all(component >= 0.0 for component in values):
            continue
        positive = tuple(abs(component) for component in values)
        scale_attr.Set(Gf.Vec3f(*positive))
        checked = tuple(float(component) for component in scale_attr.Get())
        if len(checked) != 3 or not all(
            math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-9)
            for actual, expected in zip(checked, positive)
        ):
            raise RuntimeError(
                f"G10 reflected collider scale override did not stick: "
                f"{prim.GetPath()} {checked!r} != {positive!r}"
            )
        prim.SetCustomDataByKey("handson:negativeScaleRepaired", True)
        prim.SetCustomDataByKey(
            "handson:negativeScaleOriginal",
            json.dumps(values, separators=(",", ":")),
        )
        repaired += 1
    if repaired not in (0, 2):
        raise RuntimeError(
            f"G10 expected both Collider2 reflections to be repaired, got {repaired}"
        )
    root.SetCustomDataByKey("handson:g10NegativeScaleRepairCount", int(repaired))
    return repaired

LW_TASK_SPECS: dict[str, LWAssetTaskSpec] = {
    G05_TASK_NAME: LWAssetTaskSpec(
        task_id="G05",
        task_name=G05_TASK_NAME,
        relative_usd="assets/Foldingfan001/Foldingfan001.usd",
        asset_prim_name="folding_fan",
        root_body_name="Foldingfan001",
        left_body_name="Foldingfan001_Prop001",
        right_body_name="Foldingfan001_Prop019",
        dof_names=tuple(f"Prop{index:03d}_joint" for index in range(1, 20)),
        translation=(-0.300, 0.0, 0.960),
        # Every source hinge rotates about local Y.  Map that axis to world -Z
        # and lay the long local-Z ribs toward world -X: gravity then produces
        # no opening torque, while positive Prop001 motion spreads toward the
        # +Y left robot and negative Prop019 motion toward the -Y right robot.
        # The first live upright attempt correctly failed because gravity
        # opened all 18 moving joints to their target hard stops unaided.
        orientation_wxyz=_FAN_TABLETOP_ORIENTATION,
        fixed_root=True,
        ccd=True,
    ),
    G08_TASK_NAME: LWAssetTaskSpec(
        task_id="18",
        task_name=G08_TASK_NAME,
        relative_usd="assets/MicropipetteBox001/MicropipetteBox001.usd",
        asset_prim_name="micropipette_box",
        root_body_name="MicropipetteBox001",
        left_body_name="MicropipetteBox001",
        right_body_name="MicropipetteBox001_Lid001",
        dof_names=("Lid001_joint",),
        # Recompute Z from the exact source collision bottom whenever the
        # uniform scale changes.  The composed box keeps a 0.250 mm gap above
        # the table instead of floating or starting in penetration.  Reset
        # randomization changes only XY and yaw.
        translation=(-0.280, 0.0, _G08_SPAWN_Z_M),
        orientation_wxyz=_IDENTITY_ORIENTATION,
        scale=(_G08_ASSET_UNIFORM_SCALE,) * 3,
        fixed_root=True,
        ccd=True,
    ),
    GEN3_TASK_NAME: LWAssetTaskSpec(
        task_id="EVAL-Gen02",
        task_name=GEN3_TASK_NAME,
        relative_usd="../eval_gen02_open_trash_bin/runtime/trash_bin.usda",
        asset_prim_name="trash_bin",
        root_body_name="trash_bin_base",
        left_body_name="trash_bin_base",
        right_body_name="trash_bin_lid",
        dof_names=("lid_hinge",),
        translation=(-0.280, 0.0, LW_TABLE_TOP_Z + GEN3_SUPPORT_GAP_M),
        orientation_wxyz=_IDENTITY_ORIENTATION,
        fixed_root=True,
        ccd=True,
    ),
    G10_TASK_NAME: LWAssetTaskSpec(
        task_id="49",
        task_name=G10_TASK_NAME,
        relative_usd="assets/Ruler035/Ruler035.usd",
        asset_prim_name="folding_ruler",
        # The source is a serial chain. Keep the authored middle link as part
        # of the free articulation and manipulate both outer segments through
        # their real hinges.
        root_body_name="Ruler035_Prop001",
        left_body_name="Ruler035",
        right_body_name="Ruler035_Prop002",
        dof_names=("Prop001_joint", "Prop002_joint"),
        # The source collision envelope reaches about 2.833 mm below the
        # composed root frame.  The former z=.8933 m pose entered the 2 mm
        # contact shell before all three links were level and left a sustained
        # hinge torque.  Start at z=.8958 m so the damped free articulation
        # lands coherently on the .890 m table while remaining inside the
        # 2 mm physical-origin settling budget; no body is fixed or suspended.
        translation=(-0.300, 0.0, 0.8958),
        orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
        fixed_root=False,
        ccd=True,
    ),
}

LW_SUCCESS_CONDITION_LABELS = {
    GEN3_TASK_NAME: (GEN3_SUCCESS_LABEL,),
    G05_TASK_NAME: (
        "Both assigned hands physically manipulated opposite outer fan ribs",
        "Both fan halves opened with synchronized signed joint progress",
        "At least sixteen moving ribs reached their open windows",
        "Both hands released and the opened fan remained stable",
    ),
    G08_TASK_NAME: (
        "盒盖开启角度达到 75° 即成功",
    ),
    G10_TASK_NAME: (
        "两根外段均到达直线窗：Prop001 −180°~−175°、Prop002 175°~180°",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_lw_asset_files(task_name: str | None = None) -> tuple[Path, ...]:
    """Validate every manifest byte before a Kit process is started."""

    if task_name == GEN3_TASK_NAME:
        return validate_gen3_asset_files()

    if not LW_MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Missing LW asset manifest: {LW_MANIFEST_PATH}")
    manifest = json.loads(LW_MANIFEST_PATH.read_text(encoding="utf-8"))
    validated: list[Path] = []
    for entry in manifest.get("files", ()):
        path = LW_ASSET_DIRECTORY / str(entry["path"])
        if not path.is_file():
            raise FileNotFoundError(f"LW asset dependency is missing: {path}")
        actual_size = path.stat().st_size
        if actual_size != int(entry["size_bytes"]):
            raise ValueError(
                f"LW asset size mismatch for {path}: {actual_size} != "
                f"{entry['size_bytes']}"
            )
        actual_hash = _sha256(path)
        if actual_hash != str(entry["sha256"]):
            raise ValueError(
                f"LW asset SHA-256 mismatch for {path}: {actual_hash} != "
                f"{entry['sha256']}"
            )
        validated.append(path)
    required = {spec.source_path for spec in LW_TASK_SPECS.values()
                if spec.task_name != GEN3_TASK_NAME}
    if not required.issubset(validated):
        missing = sorted(str(path) for path in required - set(validated))
        raise ValueError("LW manifest omits task USDs: " + ", ".join(missing))
    return tuple(validated)


def _quatf(wxyz: tuple[float, float, float, float]) -> Any:
    from pxr import Gf

    return Gf.Quatf(float(wxyz[0]), *(float(value) for value in wxyz[1:]))


def _author_fixed_root(stage: Any, spec: LWAssetTaskSpec) -> str:
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics

    asset_root = stage.GetPrimAtPath(spec.asset_path)
    root_body = stage.GetPrimAtPath(spec.root_body_path)
    if not asset_root.IsValid() or not asset_root.HasAPI(
        UsdPhysics.ArticulationRootAPI
    ):
        raise RuntimeError(
            "LW fixed task requires the source articulation root at "
            f"{spec.asset_path}"
        )
    if not root_body.IsValid() or not root_body.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError(
            f"LW fixed task root body is missing: {spec.root_body_path}"
        )

    # PhysX classifies a mechanism as fixed-base only when the articulation API
    # is applied to the inbound world joint.  Merely constraining an already
    # floating articulation leaves a writable floating root and can inject a
    # large correction on the first simulation frame.
    asset_root.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    if asset_root.HasAPI(PhysxSchema.PhysxArticulationAPI):
        asset_root.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
    UsdPhysics.RigidBodyAPI(root_body).CreateKinematicEnabledAttr().Set(False)
    physx_body = (
        PhysxSchema.PhysxRigidBodyAPI(root_body)
        if root_body.HasAPI(PhysxSchema.PhysxRigidBodyAPI)
        else PhysxSchema.PhysxRigidBodyAPI.Apply(root_body)
    )
    physx_body.CreateDisableGravityAttr().Set(True)

    path = f"{spec.asset_path}/world_fixed_joint"
    joint = UsdPhysics.FixedJoint.Define(stage, path)
    joint.CreateBody1Rel().SetTargets([Sdf.Path(spec.root_body_path)])
    joint.CreateJointEnabledAttr().Set(True)
    # The fixed body is not necessarily the reference root (G10 fixes its
    # authored middle link).  Anchor the world side to the body's composed
    # frame before the first PhysX reset, avoiding a frame-zero correction.
    world_matrix = UsdGeom.XformCache(
        Usd.TimeCode.Default()
    ).GetLocalToWorldTransform(root_body)
    world_translation = world_matrix.ExtractTranslation()
    world_rotation = Gf.Transform(world_matrix).GetRotation().GetQuat()
    world_imaginary = world_rotation.GetImaginary()
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*world_translation))
    joint.CreateLocalRot0Attr().Set(
        Gf.Quatf(
            float(world_rotation.GetReal()),
            Gf.Vec3f(*(float(value) for value in world_imaginary)),
        )
    )
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
    fixed_root = joint.GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(fixed_root)
    PhysxSchema.PhysxArticulationAPI.Apply(
        fixed_root
    ).CreateEnabledSelfCollisionsAttr().Set(spec.task_name == GEN3_TASK_NAME)
    return path


def _configure_task_joint_limits(
    stage: Any,
    spec: LWAssetTaskSpec,
) -> dict[str, tuple[float, float]]:
    """Apply and verify task-scoped limits on composed native joints.

    The user-provided USDs remain byte-identical. G08 narrows its authored
    authored −90-degree lid stop for the task. G10 preserves the
    symmetric ±180-degree task envelope for both ruler hinges.
    """

    if spec.task_name not in (G08_TASK_NAME, G10_TASK_NAME, GEN3_TASK_NAME):
        return {}

    from pxr import UsdPhysics

    if spec.task_name in (G08_TASK_NAME, GEN3_TASK_NAME):
        if len(spec.dof_names) != 1:
            raise RuntimeError(
                f"G08 task-scoped limit expects one DOF, got {spec.dof_names}"
            )
        limits = {spec.dof_names[0]: (-float(G08_LID_HARD_STOP_DEG), 0.0)}
        joint_paths = {
            spec.dof_names[0]: (
                f"{spec.asset_path}/{spec.right_body_name}/{spec.dof_names[0]}"
            )
        }
    else:
        if tuple(spec.dof_names) != ("Prop001_joint", "Prop002_joint"):
            raise RuntimeError(
                "G10 task-scoped limits expect Prop001_joint and Prop002_joint, "
                f"got {spec.dof_names}"
            )
        limits = {
            "Prop001_joint": (-180.0, 180.0),
            "Prop002_joint": (-180.0, 180.0),
        }
        joint_paths = {
            "Prop001_joint": f"{spec.asset_path}/Ruler035_Prop001/Prop001_joint",
            "Prop002_joint": f"{spec.asset_path}/Ruler035_Prop002/Prop002_joint",
        }

    measured: dict[str, tuple[float, float]] = {}
    for dof_name in spec.dof_names:
        joint_path = joint_paths[dof_name]
        joint_prim = stage.GetPrimAtPath(joint_path)
        if not joint_prim.IsValid() or not joint_prim.IsA(
            UsdPhysics.RevoluteJoint
        ):
            label = "G08 composed lid" if spec.task_name in (G08_TASK_NAME, GEN3_TASK_NAME) else "G10 composed ruler"
            raise RuntimeError(
                f"{label} revolute joint is missing or has the wrong schema: "
                f"{joint_path}"
            )
        joint = UsdPhysics.RevoluteJoint(joint_prim)
        # USD physics revolute limits are degrees (not radians).
        lower_expected, upper_expected = limits[dof_name]
        joint.CreateLowerLimitAttr().Set(float(lower_expected))
        joint.CreateUpperLimitAttr().Set(float(upper_expected))
        lower = float(joint.GetLowerLimitAttr().Get())
        upper = float(joint.GetUpperLimitAttr().Get())
        if not (
            math.isclose(lower, float(lower_expected), abs_tol=1.0e-6)
            and math.isclose(upper, float(upper_expected), abs_tol=1.0e-6)
        ):
            raise RuntimeError(
                f"{spec.task_id} composed joint limit override did not stick: "
                f"{joint_path} lower={lower} upper={upper}"
            )
        if spec.task_name in (G08_TASK_NAME, GEN3_TASK_NAME):
            joint_prim.SetCustomDataByKey(
                "handson:taskHardStopDeg",
                float(G08_LID_HARD_STOP_DEG),
            )
            joint_prim.SetCustomDataByKey(
                "handson:sourceAuthoredLowerLimitDeg",
                -90.0,
            )
        else:
            # USD customData values are restricted to scalar/registered Vt
            # types in the Isaac Sim Python bindings.  Keep the source
            # interval auditable as two scalar fields (a Python list here is
            # rejected during stage composition before PhysX starts).
            joint_prim.SetCustomDataByKey(
                "handson:sourceAuthoredLowerLimitDeg",
                float(limits[dof_name][0]),
            )
            joint_prim.SetCustomDataByKey(
                "handson:sourceAuthoredUpperLimitDeg",
                float(limits[dof_name][1]),
            )
        measured[dof_name] = (lower, upper)
    return measured


def _configure_native_physics(stage: Any, spec: LWAssetTaskSpec) -> tuple[int, int]:
    from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics

    root = stage.GetPrimAtPath(spec.asset_path)
    if not root.IsValid():
        raise RuntimeError(f"LW native physics root is missing: {spec.asset_path}")
    reflected_scales_repaired = _repair_g10_reflected_collision_scales(stage, spec)
    mass_properties_tuned = _sanitize_lw_mass_properties(stage, spec, root)
    g08_lid_dynamics = _configure_g08_lid_dynamics(stage, spec)
    g10_ruler_dynamics = _configure_g10_ruler_dynamics(stage, spec)
    bodies = 0
    colliders = 0
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            body_api = (
                PhysxSchema.PhysxRigidBodyAPI(prim)
                if prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI)
                else PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            )
            body_api.CreateSolverPositionIterationCountAttr().Set(32)
            body_api.CreateSolverVelocityIterationCountAttr().Set(4)
            body_api.CreateMaxDepenetrationVelocityAttr().Set(1.0)
            body_api.CreateSleepThresholdAttr().Set(0.005)
            body_api.CreateStabilizationThresholdAttr().Set(0.0025)
            if spec.task_name == G10_TASK_NAME:
                # All three ruler links are thin contact bodies.  Solve each
                # link at the same high iteration budget as the hand shells,
                # and cap depenetration impulses before they can propagate
                # through the serial hinge chain.  The authored Prop001 body
                # is the free articulation root and receives this treatment
                # before any PhysX reset.
                body_api.CreateSolverPositionIterationCountAttr().Set(
                    _G10_ROOT_SOLVER_POSITION_ITERATIONS
                )
                body_api.CreateSolverVelocityIterationCountAttr().Set(
                    _G10_ROOT_SOLVER_VELOCITY_ITERATIONS
                )
                body_api.CreateMaxDepenetrationVelocityAttr().Set(
                    _G10_ROOT_MAX_DEPENETRATION_VELOCITY_MPS
                )
            elif spec.task_name == G08_TASK_NAME:
                # The closed box base is fixed to the world and the lid is a
                # thin revolute link.  A normal hand pinch can otherwise
                # inject a large one-frame depenetration correction into the
                # fixed joint, which presents as visible box/lid interpenetration
                # and destabilizes the release sequence.
                body_api.CreateSolverPositionIterationCountAttr().Set(
                    _G08_SOLVER_POSITION_ITERATIONS
                )
                body_api.CreateSolverVelocityIterationCountAttr().Set(
                    _G08_SOLVER_VELOCITY_ITERATIONS
                )
                body_api.CreateMaxDepenetrationVelocityAttr().Set(
                    _G08_MAX_DEPENETRATION_VELOCITY_MPS
                )
            if spec.ccd:
                if spec.task_name == G08_TASK_NAME:
                    # Retain the G08 swept-CCD path that was stable in Isaac
                    # Sim 5.1. Applying speculative CCD to every thin link and
                    # Wuji body triggered CUDA 700 in an earlier layout.
                    body_api.CreateEnableCCDAttr().Set(True)
                elif spec.task_name == G05_TASK_NAME:
                    body_api.CreateEnableSpeculativeCCDAttr().Set(True)
                elif spec.task_name == G10_TASK_NAME:
                    # Use swept CCD for the free thin ruler links; speculative
                    # CCD can over-expand contacts on GPU PhysX.
                    body_api.CreateEnableCCDAttr().Set(True)
                else:
                    body_api.CreateEnableCCDAttr().Set(True)
            bodies += 1
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            prim_path = str(prim.GetPath())
            if "/Collisions/" in prim_path:
                # The source packages author dedicated convex collision
                # geometry under a Collisions scope.  ``purpose=guide`` alone
                # is not a visibility guarantee in the human viewport, where
                # it exposed convex polyhedra and z-fighting.  Explicit
                # invisibility leaves CollisionAPI fully active while making
                # the final composed render tree unambiguous.
                imageable = UsdGeom.Imageable(prim)
                if imageable:
                    imageable.MakeInvisible()
                prim.SetCustomDataByKey(
                    "handson:renderClassification",
                    "collision_only_source",
                )
            if (
                spec.task_name == G08_TASK_NAME
                and prim.GetName() == "MicropipetteBox001_Collider1"
            ):
                # Keep the validated GPU approximation only on the thin bottom
                # support collider. Applying it to every box/lid mesh exhausts
                # the Isaac Sim CUDA contact allocator.
                mesh_collision = UsdPhysics.MeshCollisionAPI(prim)
                if not mesh_collision:
                    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
                mesh_collision.CreateApproximationAttr().Set(
                    UsdPhysics.Tokens.boundingCube
                )
                prim.SetCustomDataByKey("handson:g08CollisionApproximation", "boundingCube")
            if (
                spec.task_name == G08_TASK_NAME
                and prim.GetName() in _G08_GPU_INCOMPATIBLE_COLLIDER_NAMES
                and prim.IsA(UsdGeom.Mesh)
            ):
                # The latest live G08 session proved that these seven source
                # convexes are not GPU compatible.  In particular, lid
                # Collider6 is the wide top panel: CPU fallback made the hand
                # appear to pass straight through the rendered cover.  Keep
                # the source meshes and visuals immutable, but author their
                # tight local AABBs as GPU-native collision shapes on the
                # composed stage.
                mesh_collision = UsdPhysics.MeshCollisionAPI(prim)
                if not mesh_collision:
                    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
                mesh_collision.CreateApproximationAttr().Set(
                    UsdPhysics.Tokens.boundingCube
                )
            if (
                spec.task_name == G10_TASK_NAME
                and prim.GetName() in {
                    "Ruler035_Collider1",
                    "Ruler035_prop001_Collider1",
                    "Ruler035_prop002_Collider1",
                }
                and prim.IsA(UsdGeom.Mesh)
            ):
                # The three long ruler collision meshes are source-authored
                # thin convex hulls.  Isaac Sim 5.1 GPU PhysX cannot cook
                # these hulls for the CUDA narrowphase (it falls back to CPU
                # and can fail the first GPU reset with error 700).  Keep the
                # immutable source/render mesh and author a GPU-compatible
                # bounding cube only in the composed integration layer.  The
                # small hinge-pin Collider2 meshes remain source-authored.
                mesh_collision = UsdPhysics.MeshCollisionAPI(prim)
                if not mesh_collision:
                    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
                mesh_collision.CreateApproximationAttr().Set(
                    UsdPhysics.Tokens.boundingCube
                )
                prim.SetCustomDataByKey(
                    "handson:g10CollisionApproximation",
                    "boundingCube",
                )
            collision_api = (
                PhysxSchema.PhysxCollisionAPI(prim)
                if prim.HasAPI(PhysxSchema.PhysxCollisionAPI)
                else PhysxSchema.PhysxCollisionAPI.Apply(prim)
            )
            # G05's source ribs and G10's ruler links are only a few
            # millimetres thick.  Use a wider detection shell for robot
            # contact without changing render geometry.  G05 still keeps
            # rib-to-rib self collision disabled because adjacent source
            # layers are only about 0.22 mm apart in the closed stack.
            contact_offset = (
                0.003
                if spec.task_name == G05_TASK_NAME
                else _G10_CONTACT_OFFSET_M
                if spec.task_name == G10_TASK_NAME
                else _G08_CONTACT_OFFSET_M
                if spec.task_name == G08_TASK_NAME
                else 0.001
            )
            collision_api.CreateContactOffsetAttr().Set(contact_offset)
            collision_api.CreateRestOffsetAttr().Set(0.0)
            colliders += 1
    if spec.task_name == GEN3_TASK_NAME:
        configure_gen3_dynamics(stage, spec)
    for path in (spec.left_body_path, spec.right_body_path):
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid() or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(f"LW assigned contact body is missing: {path}")
        report = (
            PhysxSchema.PhysxContactReportAPI(prim)
            if prim.HasAPI(PhysxSchema.PhysxContactReportAPI)
            else PhysxSchema.PhysxContactReportAPI.Apply(prim)
        )
        report.CreateThresholdAttr().Set(0.0)
    root.SetCustomDataByKey(
        "handson:finiteMassPropertiesApplied",
        int(mass_properties_tuned),
    )
    root.SetCustomDataByKey(
        "handson:g10NegativeScaleRepairCount",
        int(reflected_scales_repaired),
    )
    if g08_lid_dynamics:
        root.SetCustomDataByKey(
            "handson:g08LidDynamicsPolicy",
            str(g08_lid_dynamics["policy"]),
        )
        root.SetCustomDataByKey(
            "handson:g08EffectiveHingeInertiaKgM2",
            float(g08_lid_dynamics["effective_hinge_inertia_kg_m2"]),
        )
    if g10_ruler_dynamics:
        root.SetCustomDataByKey(
            "handson:g10DynamicsPolicy",
            str(g10_ruler_dynamics["policy"]),
        )
    return bodies, colliders


def _count_asset_topology(stage: Any, spec: LWAssetTaskSpec) -> tuple[int, int]:
    """Count the final composed articulation, including integration shapes."""

    from pxr import Usd, UsdPhysics

    root = stage.GetPrimAtPath(spec.asset_path)
    if not root.IsValid():
        raise RuntimeError(f"LW composed asset root is missing: {spec.asset_path}")
    bodies = 0
    colliders = 0
    for prim in Usd.PrimRange(root):
        bodies += int(prim.HasAPI(UsdPhysics.RigidBodyAPI))
        colliders += int(prim.HasAPI(UsdPhysics.CollisionAPI))
    return bodies, colliders


def _spawn_fixture(stage: Any, sim_utils: Any, spec: LWAssetTaskSpec) -> tuple[str, ...]:
    fixture_paths: list[str] = []
    if spec.task_name == G05_TASK_NAME:
        fixtures = (
            # The fixed pivot is presented horizontally so gravity is parallel
            # to its hinge axes.  This visible pedestal reaches from the table
            # to z=.930 m; the lowest audited rib collider stays at z=.9376 m,
            # leaving more than the combined 2 mm contact-offset shell.
            ("fan_pedestal", (0.075, 0.085, 0.040), (-0.300, 0.0, 0.910)),
        )
    elif spec.task_name == G10_TASK_NAME:
        # G10 is intentionally free on the tabletop; do not add a fixed
        # middle cradle or other fixture that would constrain its root pose.
        fixtures = ()
    else:
        fixtures = ()
    for name, size, translation in fixtures:
        path = f"{LW_TASK_ROOT}/{name}"
        cfg = sim_utils.CuboidCfg(
            size=size,
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.001,
                rest_offset=0.0,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.2,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.12, 0.24, 0.34),
                roughness=0.75,
            ),
        )
        cfg.func(path, cfg, translation=translation)
        stage.GetPrimAtPath(path).SetCustomDataByKey(
            "handson:fixture", spec.task_name
        )
        fixture_paths.append(path)
    return tuple(fixture_paths)


def _spawn_grip_sleeves(stage: Any, sim_utils: Any, spec: LWAssetTaskSpec) -> tuple[str, ...]:
    from pxr import UsdGeom

    if spec.task_name == G05_TASK_NAME:
        # Source-USD body-local bounds put Prop001 at
        # y=[-18.859, -17.061] mm and Prop019 at
        # y=[+17.046, +18.843] mm.  Place each proxy outside the rib stack with
        # its inner face at +/-18 mm.  The former visible 12 mm-thick sleeve is
        # replaced by a 4 mm collision-only proxy, which embeds about 0.85 mm
        # into only the assigned outer rib, retains >1.1 mm from the neighboring
        # rib, and limits the outward proxy expansion to about 3.1 mm.  These
        # are collision-only grasp proxies, never render geometry.  Extend the
        # proxy along nearly the complete usable outer rib so an operator does
        # not have to hit one 50 mm patch to establish physical contact.
        sleeves = (
            (
                spec.left_body_path,
                "left_grip",
                (0.0, -0.0200, 0.105),
                (0.030, 0.004, 0.180),
            ),
            (
                spec.right_body_path,
                "right_grip",
                (0.0, 0.0200, 0.105),
                (0.030, 0.004, 0.180),
            ),
        )
    elif spec.task_name == G10_TASK_NAME:
        sleeves = (
            # With q=(-90,+90), these lie at approximately
            # (-0.405,+0.185) and (-0.189,-0.170), giving uncrossed left/right
            # targets.  Both remain on their source outer-link colliders.  The
            # local z lift is intentional: at .004 m the hidden cuboids
            # penetrated the tabletop by up to 1.7 mm at reset (including the
            # 2 mm contact shell), creating a first-frame support impulse.
            (spec.left_body_path, "left_grip", (0.090, 0.010, _G10_GRIP_PROXY_LOCAL_Z_M), (0.050, 0.030, 0.018)),
            (spec.right_body_path, "right_grip", (-0.170, -0.010, _G10_GRIP_PROXY_LOCAL_Z_M), (0.050, 0.030, 0.018)),
        )
    else:
        sleeves = ()
    paths: list[str] = []
    for parent, name, translation, size in sleeves:
        path = f"{parent}/{name}"
        cfg = sim_utils.CuboidCfg(
            size=size,
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=(
                    0.003
                    if spec.task_name == G05_TASK_NAME
                    else _G10_CONTACT_OFFSET_M
                    if spec.task_name == G10_TASK_NAME
                    else _G08_CONTACT_OFFSET_M
                    if spec.task_name == G08_TASK_NAME
                    else 0.001
                ),
                rest_offset=0.0,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.6,
                dynamic_friction=1.35,
                restitution=0.0,
            ),
        )
        cfg.func(path, cfg, translation=translation)
        proxy_prim = stage.GetPrimAtPath(path)
        proxy_prim.SetCustomDataByKey("handson:gripRole", name)
        proxy_prim.SetCustomDataByKey(
            "handson:renderClassification",
            "collision_only_grasp_proxy",
        )
        imageable = UsdGeom.Imageable(proxy_prim)
        if not imageable:
            raise RuntimeError(f"LW grip proxy is not imageable: {path}")
        imageable.MakeInvisible()
        if imageable.ComputeVisibility() != UsdGeom.Tokens.invisible:
            raise RuntimeError(f"LW grip proxy remained renderable: {path}")
        paths.append(path)
    return tuple(paths)


def spawn_lw_task_assets(stage: Any, sim_utils: Any, task_name: str) -> dict[str, Any]:
    """Compose one source USD and its stable, visible task integration layer."""

    spec = LW_TASK_SPECS[task_name]
    validate_lw_asset_files(task_name)
    from pxr import UsdGeom

    root = UsdGeom.Xform.Define(stage, LW_TASK_ROOT)
    root.GetPrim().SetCustomDataByKey("handson:task", task_name)
    root.GetPrim().SetCustomDataByKey(
        "handson:provenance",
        "CC-BY-4.0; AmongModels; Sketchfab/Objaverse"
        if task_name == GEN3_TASK_NAME else "user_provided_unresolved",
    )

    table_cfg = sim_utils.CuboidCfg(
        size=(0.720, 1.000, 0.060),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.001,
            rest_offset=0.0,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.90,
            dynamic_friction=0.75,
            restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.70, 0.70, 0.66),
            roughness=0.72,
        ),
    )
    table_cfg.func(LW_TABLE_PATH, table_cfg, translation=(-0.280, 0.0, 0.860))
    stage.GetPrimAtPath(LW_TABLE_PATH).SetCustomDataByKey("handson:fixture", "table")

    # IsaacLab applies file scale from ``UsdFileCfg.scale``.  Passing a
    # ``scale=`` keyword to ``spawn_from_usd`` is accepted through **kwargs but
    # ignored, which would silently leave the G08 box at its 0.85 m source
    # width and invalidate the authored support/reach envelope.
    asset_cfg = sim_utils.UsdFileCfg(
        usd_path=str(spec.source_path.resolve()),
        scale=spec.scale,
    )
    asset_cfg.func(
        spec.asset_path,
        asset_cfg,
        translation=spec.translation,
        orientation=spec.orientation_wxyz,
    )
    asset_prim = stage.GetPrimAtPath(spec.asset_path)
    asset_prim.SetCustomDataByKey("handson:sourceAsset", spec.relative_usd)
    asset_prim.SetCustomDataByKey("handson:sourceSha256", _sha256(spec.source_path))
    asset_prim.SetCustomDataByKey("handson:formalTruth", True)
    composed_joint_limits_deg = _configure_task_joint_limits(stage, spec)
    source_body_count, source_collider_count = _configure_native_physics(
        stage,
        spec,
    )
    fixed_joint = _author_fixed_root(stage, spec) if spec.fixed_root else ""
    fixture_paths = _spawn_fixture(stage, sim_utils, spec)
    grip_paths = _spawn_grip_sleeves(stage, sim_utils, spec)
    body_count, collider_count = _count_asset_topology(stage, spec)
    expected_collider_count = source_collider_count + len(grip_paths)
    if body_count != source_body_count or collider_count != expected_collider_count:
        raise RuntimeError(
            "LW integration changed the composed articulation topology: "
            f"source=({source_body_count} bodies, {source_collider_count} "
            f"colliders), grips={len(grip_paths)}, final=({body_count} bodies, "
            f"{collider_count} colliders)"
        )
    return {
        "task_name": task_name,
        "asset_path": spec.asset_path,
        "root_body_path": spec.root_body_path,
        "left_body_path": spec.left_body_path,
        "right_body_path": spec.right_body_path,
        "dof_names": spec.dof_names,
        "fixed_joint_path": fixed_joint,
        "fixed_root": bool(spec.fixed_root),
        "composed_joint_limits_deg": composed_joint_limits_deg,
        "fixture_paths": fixture_paths,
        "grip_paths": grip_paths,
        "source_body_count": source_body_count,
        "source_collider_count": source_collider_count,
        "body_count": body_count,
        "collider_count": collider_count,
    }


__all__ = [
    "G05_TASK_NAME",
    "G08_TASK_NAME",
    "G08_LID_HARD_STOP_DEG",
    "G10_TASK_NAME",
    "LW_ASSET_DIRECTORY",
    "LW_CONTROL_HZ",
    "LW_PHYSICS_HZ",
    "LW_SCENE_XML",
    "LW_SUCCESS_CONDITION_LABELS",
    "LW_TABLE_PATH",
    "LW_TABLE_COLLIDER_PATH",
    "LW_TABLE_TOP_Z",
    "LW_TASK_NAMES",
    "LW_TASK_ROOT",
    "LW_TASK_SPECS",
    "LWAssetTaskSpec",
    "_configure_g08_lid_dynamics",
    "_configure_task_joint_limits",
    "_count_asset_topology",
    "_g08_lid_effective_hinge_inertia_kg_m2",
    "_lw_mass_fallback_center",
    "_lw_mass_fallbacks",
    "spawn_lw_task_assets",
    "validate_lw_asset_files",
]
