"""Versioned tabletop-physics overlay for the exact P0-03 Ruler035 USD.

The benchmark-owned ``Ruler035.usd`` is an immutable input.  This module only
authors opinions on the already-composed runtime stage.  Its pure policy
objects intentionally import neither Isaac Sim nor USD so reset geometry and
mass/inertia values can be unit-tested with the ordinary Python interpreter.

The live entry point is :func:`apply_p003_physics_overlay`.  It validates the
known three-link/two-joint topology before replacing the five thin source
colliders with versioned, invisible analytic shapes.  If that collider mapping
cannot be proved, it leaves every source collider enabled and reports the
blocked reason instead of guessing at articulation paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence


P003_PHYSICS_POLICY_VERSION = "p003_tabletop_physics_v1"
P003_SOURCE_ASSET_SHA256 = (
    "55f3823f9ae46ba34f5bfcc12332aaa512cf99eb344e2db8bfa92a8825ca0369"
)
DEFAULT_P003_ROOT_PATH = "/World/Task/P003Ruler"


def _require_positive_finite(name: str, value: float) -> None:
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value!r}")


def _require_nonnegative_finite(name: str, value: float) -> None:
    if not math.isfinite(float(value)) or float(value) < 0.0:
        raise ValueError(f"{name} must be non-negative and finite, got {value!r}")


@dataclass(frozen=True)
class P003SegmentPhysics:
    """Analytic physical properties for one articulation link.

    ``root_translation_m`` is the source link's authored translation relative
    to the articulation root.  All joints rotate about local Z, so its Z value
    plus the local proxy bounds gives the exact pre-PhysX support bottom for
    every reset joint angle and root yaw.
    """

    label: str
    body_suffix: str
    source_collision_suffixes: tuple[str, ...]
    nominal_mass_kg: float
    root_translation_m: tuple[float, float, float]
    box_center_m: tuple[float, float, float]
    box_size_m: tuple[float, float, float]
    hinge_center_xy_m: tuple[float, float] | None = None
    hinge_radius_m: float = 0.0

    def scaled_mass_kg(self, mass_scale: float) -> float:
        _require_positive_finite("mass_scale", mass_scale)
        return self.nominal_mass_kg * float(mass_scale)

    def diagonal_inertia_kg_m2(
        self, mass_scale: float = 1.0
    ) -> tuple[float, float, float]:
        """Return the principal inertia of the conservative analytic box."""

        mass = self.scaled_mass_kg(mass_scale)
        length, width, thickness = self.box_size_m
        return (
            mass * (width * width + thickness * thickness) / 12.0,
            mass * (length * length + thickness * thickness) / 12.0,
            mass * (length * length + width * width) / 12.0,
        )

    def local_collision_bounds_m(
        self,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        half = tuple(value / 2.0 for value in self.box_size_m)
        lower = tuple(center - radius for center, radius in zip(self.box_center_m, half))
        upper = tuple(center + radius for center, radius in zip(self.box_center_m, half))
        if self.hinge_center_xy_m is None:
            return lower, upper

        hinge_center = (
            self.hinge_center_xy_m[0],
            self.hinge_center_xy_m[1],
            self.box_center_m[2],
        )
        hinge_half = (
            self.hinge_radius_m,
            self.hinge_radius_m,
            self.box_size_m[2] / 2.0,
        )
        hinge_lower = tuple(
            center - radius for center, radius in zip(hinge_center, hinge_half)
        )
        hinge_upper = tuple(
            center + radius for center, radius in zip(hinge_center, hinge_half)
        )
        return (
            tuple(min(a, b) for a, b in zip(lower, hinge_lower)),
            tuple(max(a, b) for a, b in zip(upper, hinge_upper)),
        )

    def root_relative_bottom_m(self) -> float:
        lower, _ = self.local_collision_bounds_m()
        return self.root_translation_m[2] + lower[2]


@dataclass(frozen=True)
class P003JointPhysics:
    label: str
    joint_suffix: str
    body0_suffix: str
    body1_suffix: str
    lower_limit_deg: float
    upper_limit_deg: float


@dataclass(frozen=True)
class P003PhysicsPolicy:
    version: str
    source_asset_sha256: str
    segments: tuple[P003SegmentPhysics, ...]
    joints: tuple[P003JointPhysics, ...]
    drive_stiffness: float = 0.0
    # Zero stiffness keeps the hinges passive.  A tightly bounded viscous
    # drive absorbs the
    # reset drop instead of letting a millimetre-scale tabletop impact fling a
    # leaf through tens of degrees before the operator touches it.
    drive_damping: float = 0.05
    drive_max_force: float = 0.15
    joint_friction: float = 0.03
    rest_offset_m: float = 0.0
    contact_offset_m: float = 0.001
    enable_ccd: bool = True
    proxy_scope_name: str = "P003PhysicsOverlay_v1"

    def __post_init__(self) -> None:
        if not self.version:
            raise ValueError("P003 physics policy version must be non-empty")
        if len(self.segments) != 3 or {item.label for item in self.segments} != {
            "s0",
            "s1",
            "s2",
        }:
            raise ValueError("P003 physics policy must define exactly s0, s1, and s2")
        if len(self.joints) != 2 or {item.label for item in self.joints} != {
            "j1",
            "j2",
        }:
            raise ValueError("P003 physics policy must define exactly j1 and j2")
        _require_nonnegative_finite("drive_stiffness", self.drive_stiffness)
        _require_nonnegative_finite("drive_damping", self.drive_damping)
        _require_nonnegative_finite("drive_max_force", self.drive_max_force)
        _require_nonnegative_finite("joint_friction", self.joint_friction)
        _require_nonnegative_finite("rest_offset_m", self.rest_offset_m)
        _require_positive_finite("contact_offset_m", self.contact_offset_m)
        if self.contact_offset_m <= self.rest_offset_m:
            raise ValueError("contact_offset_m must be greater than rest_offset_m")
        for segment in self.segments:
            _require_positive_finite(
                f"{segment.label}.nominal_mass_kg", segment.nominal_mass_kg
            )
            for axis, value in zip("xyz", segment.box_size_m):
                _require_positive_finite(f"{segment.label}.box_size_{axis}", value)
            if segment.hinge_center_xy_m is not None:
                _require_positive_finite(
                    f"{segment.label}.hinge_radius_m", segment.hinge_radius_m
                )

    @property
    def nominal_total_mass_kg(self) -> float:
        return sum(segment.nominal_mass_kg for segment in self.segments)

    def segment(self, label: str) -> P003SegmentPhysics:
        return next(item for item in self.segments if item.label == label)

    def joint(self, label: str) -> P003JointPhysics:
        return next(item for item in self.joints if item.label == label)

    def scaled_masses_kg(self, mass_scale: float = 1.0) -> dict[str, float]:
        return {
            segment.label: segment.scaled_mass_kg(mass_scale)
            for segment in self.segments
        }

    def local_collision_bounds_m(
        self,
    ) -> dict[
        str,
        tuple[tuple[float, float, float], tuple[float, float, float]],
    ]:
        return {
            segment.label: segment.local_collision_bounds_m()
            for segment in self.segments
        }

    def minimum_root_relative_bottom_m(self) -> float:
        return min(segment.root_relative_bottom_m() for segment in self.segments)


# The X/Y extents and centers below are the read-only USD visual/physical
# bounds in each rigid-body frame.  Only Z thickness is deliberately expanded
# to 4 mm for the tabletop pilot.  The center is preserved, making support
# height deterministic and avoiding an unreviewed lateral grasp envelope.
DEFAULT_P003_PHYSICS_POLICY = P003PhysicsPolicy(
    version=P003_PHYSICS_POLICY_VERSION,
    source_asset_sha256=P003_SOURCE_ASSET_SHA256,
    segments=(
        P003SegmentPhysics(
            label="s0",
            body_suffix="Ruler035",
            source_collision_suffixes=(
                "Ruler035/Collisions/Ruler035_Collider1",
            ),
            nominal_mass_kg=0.055,
            root_translation_m=(0.0, 0.0, 0.0),
            box_center_m=(-0.000004551410711, 0.000002306459010, -0.002084458945319),
            box_size_m=(0.24959762, 0.01497697, 0.004),
        ),
        P003SegmentPhysics(
            label="s1",
            body_suffix="Ruler035_Prop001",
            source_collision_suffixes=(
                "Ruler035_Prop001/Collisions/Ruler035_prop001_Collider1",
                "Ruler035_Prop001/Collisions/Ruler035_prop001_Collider2",
            ),
            nominal_mass_kg=0.055,
            root_translation_m=(-0.095010802149773, -0.000009653610505, -0.000471499050036),
            box_center_m=(0.0950109794, 0.000011993354, 0.000190003361),
            box_size_m=(0.24960740, 0.01497702, 0.004),
            hinge_center_xy_m=(-0.000007025898, -0.000005187852),
            hinge_radius_m=0.004579695873,
        ),
        P003SegmentPhysics(
            label="s2",
            body_suffix="Ruler035_Prop002",
            source_collision_suffixes=(
                "Ruler035_Prop002/Collisions/Ruler035_prop002_Collider1",
                "Ruler035_Prop002/Collisions/Ruler035_prop002_Collider2",
            ),
            nominal_mass_kg=0.050,
            root_translation_m=(0.101002775132656, -0.000009654690984, 0.001627802150324),
            box_center_m=(-0.0885262708, 0.000007294386, 0.000187815423),
            box_size_m=(0.22465437, 0.01497696, 0.004),
            hinge_center_xy_m=(-0.000000648201, 0.000015392590),
            hinge_radius_m=0.004579695873,
        ),
    ),
    joints=(
        P003JointPhysics(
            label="j1",
            joint_suffix="Ruler035_Prop001/Prop001_joint",
            body0_suffix="Ruler035",
            body1_suffix="Ruler035_Prop001",
            lower_limit_deg=-180.0,
            upper_limit_deg=0.0,
        ),
        P003JointPhysics(
            label="j2",
            joint_suffix="Ruler035_Prop002/Prop002_joint",
            body0_suffix="Ruler035_Prop001",
            body1_suffix="Ruler035_Prop002",
            lower_limit_deg=0.0,
            upper_limit_deg=180.0,
        ),
    ),
)


@dataclass
class P003PhysicsOverlayResult:
    """Runtime handles plus serializable evidence from one overlay application."""

    policy_version: str
    mass_attrs_by_label: dict[str, Any] = field(default_factory=dict)
    inertia_attrs_by_label: dict[str, Any] = field(default_factory=dict)
    metadata_attrs_by_name: dict[str, Any] = field(default_factory=dict)
    source_collision_paths: tuple[str, ...] = ()
    proxy_collision_paths: tuple[str, ...] = ()
    collision_local_bounds_by_label: dict[
        str,
        tuple[tuple[float, float, float], tuple[float, float, float]],
    ] = field(default_factory=dict)
    reset_root_support_offset_m: float = 0.0
    proxy_topology_status: str = "not_checked"
    proxy_blocked_reason: str | None = None
    joint_friction_authored: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def collision_proxies_applied(self) -> bool:
        return self.proxy_topology_status == "applied"

    def report_dict(self) -> dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "source_collision_paths": list(self.source_collision_paths),
            "proxy_collision_paths": list(self.proxy_collision_paths),
            "collision_local_bounds_by_label": {
                label: [list(lower), list(upper)]
                for label, (lower, upper) in self.collision_local_bounds_by_label.items()
            },
            "reset_root_support_offset_m": self.reset_root_support_offset_m,
            "proxy_topology_status": self.proxy_topology_status,
            "proxy_blocked_reason": self.proxy_blocked_reason,
            "joint_friction_authored": self.joint_friction_authored,
            "metadata": self.metadata,
        }


def _root_join(root_path: str, suffix: str) -> str:
    return f"{root_path.rstrip('/')}/{suffix.lstrip('/')}"


def reset_root_support_offset_m(
    clearance_m: float = 0.0005,
    policy: P003PhysicsPolicy = DEFAULT_P003_PHYSICS_POLICY,
) -> float:
    """Return the root Z offset that starts every proxy above its support.

    The default 4 mm S0 box is centered at ``-0.002084458945319`` m, so its
    bottom is ``-0.004084458945319`` m.  A 0.5 mm bounded pre-drop clearance
    therefore produces ``0.004584458945319`` m.
    """

    _require_nonnegative_finite("clearance_m", clearance_m)
    return -policy.minimum_root_relative_bottom_m() + float(clearance_m)


def p003_physics_policy_metadata(
    *,
    mass_scale: float = 1.0,
    clearance_m: float = 0.0005,
    policy: P003PhysicsPolicy = DEFAULT_P003_PHYSICS_POLICY,
) -> dict[str, Any]:
    """Return deterministic JSON-safe policy metadata for logs/replay state."""

    _require_positive_finite("mass_scale", mass_scale)
    return {
        "format": "handson.p003.physics_overlay.v1",
        "policy_version": policy.version,
        "source_asset_sha256": policy.source_asset_sha256,
        "nominal_masses_kg": {
            segment.label: segment.nominal_mass_kg for segment in policy.segments
        },
        "scaled_masses_kg": policy.scaled_masses_kg(mass_scale),
        "nominal_total_mass_kg": policy.nominal_total_mass_kg,
        "scaled_total_mass_kg": policy.nominal_total_mass_kg * float(mass_scale),
        "mass_scale": float(mass_scale),
        "drive": {
            "stiffness": policy.drive_stiffness,
            "damping": policy.drive_damping,
            "max_force": policy.drive_max_force,
            "joint_friction": policy.joint_friction,
        },
        "collision": {
            "proxy_thickness_m": policy.segment("s0").box_size_m[2],
            "rest_offset_m": policy.rest_offset_m,
            "contact_offset_m": policy.contact_offset_m,
            "ccd": policy.enable_ccd,
            "computed_visibility": "invisible",
        },
        "reset": {
            "clearance_m": float(clearance_m),
            "minimum_root_relative_bottom_m": policy.minimum_root_relative_bottom_m(),
            "root_support_offset_m": reset_root_support_offset_m(clearance_m, policy),
        },
    }


def classify_p003_physics_path(
    path: str,
    *,
    root_path: str = DEFAULT_P003_ROOT_PATH,
    policy: P003PhysicsPolicy = DEFAULT_P003_PHYSICS_POLICY,
) -> str:
    """Classify a composed P003 path without importing USD."""

    normalized = str(path).rstrip("/") or "/"
    root = root_path.rstrip("/")
    if normalized == root:
        return "articulation_root"

    body_paths = {_root_join(root, segment.body_suffix) for segment in policy.segments}
    if normalized in body_paths:
        return "rigid_body"

    joint_paths = {_root_join(root, joint.joint_suffix) for joint in policy.joints}
    if normalized in joint_paths:
        return "revolute_joint"

    source_paths = {
        _root_join(root, suffix)
        for segment in policy.segments
        for suffix in segment.source_collision_suffixes
    }
    if normalized in source_paths:
        return "source_collision"

    for segment in policy.segments:
        scope = _root_join(
            root, f"{segment.body_suffix}/{policy.proxy_scope_name}"
        )
        if normalized == scope:
            return "proxy_scope"
        if normalized in {
            f"{scope}/segment_box",
            f"{scope}/hinge_cylinder",
        }:
            return "proxy_collision"
        if normalized.startswith(f"{scope}/"):
            return "proxy_other"

    if normalized.startswith(f"{root}/"):
        return "other_under_articulation"
    return "outside_articulation"


def _coerce_named_prims(
    supplied: Mapping[str, Any] | Sequence[Any] | None,
    labels: Sequence[str],
    expected_paths: Mapping[str, str],
    stage: Any,
    kind: str,
) -> dict[str, Any]:
    if supplied is None:
        result = {
            label: stage.GetPrimAtPath(expected_paths[label]) for label in labels
        }
    elif isinstance(supplied, Mapping):
        missing = [label for label in labels if label not in supplied]
        if missing:
            raise ValueError(f"P003 {kind} mapping is missing labels {missing}")
        result = {label: supplied[label] for label in labels}
    else:
        values = tuple(supplied)
        if len(values) != len(labels):
            raise ValueError(
                f"P003 expected {len(labels)} {kind} prims, got {len(values)}"
            )
        result = dict(zip(labels, values))

    for label, prim in result.items():
        if prim is None or not prim.IsValid():
            raise RuntimeError(f"P003 {kind} {label} is missing at {expected_paths[label]}")
        actual = str(prim.GetPath())
        if actual != expected_paths[label]:
            raise RuntimeError(
                f"P003 {kind} {label} path mismatch: expected "
                f"{expected_paths[label]}, got {actual}"
            )
    return result


def _topology_block_reason(
    *,
    stage: Any,
    root: Any,
    bodies: Mapping[str, Any],
    joints: Mapping[str, Any],
    root_path: str,
    policy: P003PhysicsPolicy,
    Usd: Any,
    UsdGeom: Any,
    UsdPhysics: Any,
) -> tuple[str | None, tuple[str, ...]]:
    if root.IsInstance() or root.IsInstanceProxy():
        raise RuntimeError(
            "P003 articulation root is instance-backed and cannot accept "
            "runtime link physics opinions"
        )
    if not root.HasAPI(UsdPhysics.ArticulationRootAPI):
        raise RuntimeError("P003 articulation root API is missing")

    for segment in policy.segments:
        body = bodies[segment.label]
        if not body.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(f"P003 {segment.label} is not a USD rigid body")

    for item in policy.joints:
        prim = joints[item.label]
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            raise RuntimeError(f"P003 {item.label} is not a revolute joint")
        joint = UsdPhysics.RevoluteJoint(prim)
        if str(joint.GetAxisAttr().Get()) != "Z":
            raise RuntimeError(f"P003 {item.label} axis is not Z")
        lower = float(joint.GetLowerLimitAttr().Get())
        upper = float(joint.GetUpperLimitAttr().Get())
        if not math.isclose(lower, item.lower_limit_deg, abs_tol=1e-6):
            raise RuntimeError(
                f"P003 {item.label} lower limit changed from the pinned topology"
            )
        if not math.isclose(upper, item.upper_limit_deg, abs_tol=1e-6):
            raise RuntimeError(
                f"P003 {item.label} upper limit changed from the pinned topology"
            )
        body0 = [str(value) for value in joint.GetBody0Rel().GetTargets()]
        body1 = [str(value) for value in joint.GetBody1Rel().GetTargets()]
        if body0 != [_root_join(root_path, item.body0_suffix)]:
            raise RuntimeError(f"P003 {item.label} body0 relation changed: {body0}")
        if body1 != [_root_join(root_path, item.body1_suffix)]:
            raise RuntimeError(f"P003 {item.label} body1 relation changed: {body1}")

    expected_source = tuple(
        _root_join(root_path, suffix)
        for segment in policy.segments
        for suffix in segment.source_collision_suffixes
    )
    for path in expected_source:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            return f"pinned source collider is missing: {path}", expected_source
        if not prim.IsA(UsdGeom.Mesh):
            return f"pinned source collider is no longer a mesh: {path}", expected_source
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            return f"pinned source collider lost CollisionAPI: {path}", expected_source

    observed_collision_paths = {
        str(prim.GetPath())
        for prim in Usd.PrimRange(root)
        if prim.IsActive() and prim.HasAPI(UsdPhysics.CollisionAPI)
    }
    allowed = set(expected_source)
    for segment in policy.segments:
        scope = _root_join(
            root_path, f"{segment.body_suffix}/{policy.proxy_scope_name}"
        )
        allowed.add(f"{scope}/segment_box")
        if segment.hinge_center_xy_m is not None:
            allowed.add(f"{scope}/hinge_cylinder")
    unexpected = sorted(observed_collision_paths - allowed)
    if unexpected:
        return f"unexpected collision prims under articulation: {unexpected}", expected_source
    return None, expected_source


def _set_xform_translation_and_scale(
    prim: Any,
    translation: tuple[float, float, float],
    scale: tuple[float, float, float] | None,
    *,
    Gf: Any,
    UsdGeom: Any,
) -> None:
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(*translation)
    )
    if scale is not None:
        xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*scale))


def _make_collision_invisible(
    prim: Any,
    *,
    PhysxSchema: Any,
    UsdGeom: Any,
    UsdPhysics: Any,
    contact_offset_m: float,
    rest_offset_m: float,
) -> None:
    imageable = UsdGeom.Imageable(prim)
    imageable.CreatePurposeAttr().Set(UsdGeom.Tokens.guide)
    imageable.MakeInvisible()
    collision = (
        UsdPhysics.CollisionAPI(prim)
        if prim.HasAPI(UsdPhysics.CollisionAPI)
        else UsdPhysics.CollisionAPI.Apply(prim)
    )
    collision.CreateCollisionEnabledAttr().Set(True)
    physx_collision = (
        PhysxSchema.PhysxCollisionAPI(prim)
        if prim.HasAPI(PhysxSchema.PhysxCollisionAPI)
        else PhysxSchema.PhysxCollisionAPI.Apply(prim)
    )
    physx_collision.CreateContactOffsetAttr().Set(float(contact_offset_m))
    physx_collision.CreateRestOffsetAttr().Set(float(rest_offset_m))


def _author_collision_proxies(
    *,
    stage: Any,
    bodies: Mapping[str, Any],
    root_path: str,
    policy: P003PhysicsPolicy,
    Gf: Any,
    PhysxSchema: Any,
    UsdGeom: Any,
    UsdPhysics: Any,
) -> tuple[str, ...]:
    proxy_paths: list[str] = []
    proxy_prims: list[Any] = []
    try:
        for segment in policy.segments:
            body_path = str(bodies[segment.label].GetPath())
            scope_path = f"{body_path}/{policy.proxy_scope_name}"
            scope = UsdGeom.Xform.Define(stage, scope_path).GetPrim()
            scope_xform = UsdGeom.Xformable(scope)
            scope_xform.ClearXformOpOrder()
            scope_imageable = UsdGeom.Imageable(scope)
            scope_imageable.CreatePurposeAttr().Set(UsdGeom.Tokens.guide)
            scope_imageable.MakeInvisible()

            box_path = f"{scope_path}/segment_box"
            box = UsdGeom.Cube.Define(stage, box_path)
            box.CreateSizeAttr().Set(1.0)
            _set_xform_translation_and_scale(
                box.GetPrim(),
                segment.box_center_m,
                segment.box_size_m,
                Gf=Gf,
                UsdGeom=UsdGeom,
            )
            _make_collision_invisible(
                box.GetPrim(),
                PhysxSchema=PhysxSchema,
                UsdGeom=UsdGeom,
                UsdPhysics=UsdPhysics,
                contact_offset_m=policy.contact_offset_m,
                rest_offset_m=policy.rest_offset_m,
            )
            proxy_paths.append(box_path)
            proxy_prims.append(box.GetPrim())

            if segment.hinge_center_xy_m is not None:
                hinge_path = f"{scope_path}/hinge_cylinder"
                hinge = UsdGeom.Cylinder.Define(stage, hinge_path)
                hinge.CreateAxisAttr().Set(UsdGeom.Tokens.z)
                hinge.CreateRadiusAttr().Set(float(segment.hinge_radius_m))
                hinge.CreateHeightAttr().Set(float(segment.box_size_m[2]))
                hinge_center = (
                    segment.hinge_center_xy_m[0],
                    segment.hinge_center_xy_m[1],
                    segment.box_center_m[2],
                )
                _set_xform_translation_and_scale(
                    hinge.GetPrim(),
                    hinge_center,
                    None,
                    Gf=Gf,
                    UsdGeom=UsdGeom,
                )
                _make_collision_invisible(
                    hinge.GetPrim(),
                    PhysxSchema=PhysxSchema,
                    UsdGeom=UsdGeom,
                    UsdPhysics=UsdPhysics,
                    contact_offset_m=policy.contact_offset_m,
                    rest_offset_m=policy.rest_offset_m,
                )
                proxy_paths.append(hinge_path)
                proxy_prims.append(hinge.GetPrim())

        for prim in proxy_prims:
            if not prim.IsValid() or not prim.HasAPI(UsdPhysics.CollisionAPI):
                raise RuntimeError(f"proxy collision API missing at {prim.GetPath()}")
            computed = UsdGeom.Imageable(prim).ComputeVisibility()
            if computed != UsdGeom.Tokens.invisible:
                raise RuntimeError(
                    f"proxy computed visibility is not invisible at {prim.GetPath()}: "
                    f"{computed}"
                )
        return tuple(proxy_paths)
    except Exception:
        # Source colliders have not yet been disabled.  Fail safe by turning
        # off every known partially-authored proxy before propagating the
        # authoring error to the caller's blocked report.
        for prim in proxy_prims:
            if prim.IsValid() and prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(False)
        raise


def _set_known_collision_mode(
    *,
    stage: Any,
    source_paths: Sequence[str],
    root_path: str,
    policy: P003PhysicsPolicy,
    proxies_enabled: bool,
    PhysxSchema: Any,
    UsdPhysics: Any,
) -> None:
    """Select exactly one known collider representation, including reapply."""

    for path in source_paths:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid() or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        source_collision = UsdPhysics.CollisionAPI(prim)
        source_collision.CreateCollisionEnabledAttr().Set(not proxies_enabled)
        if not proxies_enabled:
            physx_collision = (
                PhysxSchema.PhysxCollisionAPI(prim)
                if prim.HasAPI(PhysxSchema.PhysxCollisionAPI)
                else PhysxSchema.PhysxCollisionAPI.Apply(prim)
            )
            physx_collision.CreateContactOffsetAttr().Set(policy.contact_offset_m)
            physx_collision.CreateRestOffsetAttr().Set(policy.rest_offset_m)

    for segment in policy.segments:
        scope = _root_join(
            root_path, f"{segment.body_suffix}/{policy.proxy_scope_name}"
        )
        paths = [f"{scope}/segment_box"]
        if segment.hinge_center_xy_m is not None:
            paths.append(f"{scope}/hinge_cylinder")
        for path in paths:
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid() and prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(
                    proxies_enabled
                )


def _author_mass_inertia_and_ccd(
    *,
    bodies: Mapping[str, Any],
    mass_scale: float,
    policy: P003PhysicsPolicy,
    Gf: Any,
    PhysxSchema: Any,
    UsdPhysics: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    mass_attrs: dict[str, Any] = {}
    inertia_attrs: dict[str, Any] = {}
    for segment in policy.segments:
        body = bodies[segment.label]
        mass_api = (
            UsdPhysics.MassAPI(body)
            if body.HasAPI(UsdPhysics.MassAPI)
            else UsdPhysics.MassAPI.Apply(body)
        )
        mass = segment.scaled_mass_kg(mass_scale)
        inertia = segment.diagonal_inertia_kg_m2(mass_scale)
        mass_attr = mass_api.CreateMassAttr()
        mass_attr.Set(float(mass))
        mass_api.CreateDensityAttr().Set(0.0)
        mass_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(*segment.box_center_m))
        inertia_attr = mass_api.CreateDiagonalInertiaAttr()
        inertia_attr.Set(Gf.Vec3f(*inertia))
        mass_api.CreatePrincipalAxesAttr().Set(
            Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0))
        )
        mass_attrs[segment.label] = mass_attr
        inertia_attrs[segment.label] = inertia_attr

        rigid = (
            PhysxSchema.PhysxRigidBodyAPI(body)
            if body.HasAPI(PhysxSchema.PhysxRigidBodyAPI)
            else PhysxSchema.PhysxRigidBodyAPI.Apply(body)
        )
        rigid.CreateDisableGravityAttr().Set(False)
        rigid.CreateEnableCCDAttr().Set(bool(policy.enable_ccd))
    return mass_attrs, inertia_attrs


def _author_passive_joints(
    *,
    joints: Mapping[str, Any],
    policy: P003PhysicsPolicy,
    PhysxSchema: Any,
    UsdPhysics: Any,
) -> bool:
    friction_authored_for_all = True
    for item in policy.joints:
        prim = joints[item.label]
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        if not drive:
            drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
        drive.CreateStiffnessAttr().Set(float(policy.drive_stiffness))
        drive.CreateDampingAttr().Set(float(policy.drive_damping))
        drive.CreateMaxForceAttr().Set(float(policy.drive_max_force))
        drive.CreateTargetVelocityAttr().Set(0.0)

        joint_api = (
            PhysxSchema.PhysxJointAPI(prim)
            if prim.HasAPI(PhysxSchema.PhysxJointAPI)
            else PhysxSchema.PhysxJointAPI.Apply(prim)
        )
        create_friction = getattr(joint_api, "CreateJointFrictionAttr", None)
        if callable(create_friction):
            create_friction().Set(float(policy.joint_friction))
        else:
            friction_authored_for_all = False
    return friction_authored_for_all


def _author_root_metadata(
    *,
    root: Any,
    metadata: Mapping[str, Any],
    collider_mode: str,
    Sdf: Any,
) -> dict[str, Any]:
    attributes = {
        "handson:p003PhysicsPolicyVersion": (
            Sdf.ValueTypeNames.String,
            str(metadata["policy_version"]),
        ),
        "handson:p003PhysicsSourceAssetSha256": (
            Sdf.ValueTypeNames.String,
            str(metadata["source_asset_sha256"]),
        ),
        "handson:p003PhysicsColliderMode": (
            Sdf.ValueTypeNames.String,
            collider_mode,
        ),
        "handson:p003PhysicsMassScale": (
            Sdf.ValueTypeNames.Double,
            float(metadata["mass_scale"]),
        ),
        "handson:p003PhysicsResetSupportOffsetM": (
            Sdf.ValueTypeNames.Double,
            float(metadata["reset"]["root_support_offset_m"]),
        ),
    }
    authored: dict[str, Any] = {}
    for name, (value_type, value) in attributes.items():
        attribute = root.CreateAttribute(name, value_type, custom=True)
        attribute.Set(value)
        authored[name] = attribute
    return authored


def set_p003_mass_scale(
    result: P003PhysicsOverlayResult,
    mass_scale: float,
    *,
    policy: P003PhysicsPolicy = DEFAULT_P003_PHYSICS_POLICY,
) -> dict[str, float]:
    """Update both mass and explicit box inertia for a later seeded reset."""

    _require_positive_finite("mass_scale", mass_scale)
    expected = {segment.label for segment in policy.segments}
    if set(result.mass_attrs_by_label) != expected:
        raise RuntimeError("P003 overlay result has incomplete mass attributes")
    if set(result.inertia_attrs_by_label) != expected:
        raise RuntimeError("P003 overlay result has incomplete inertia attributes")

    from pxr import Gf

    masses = policy.scaled_masses_kg(mass_scale)
    for segment in policy.segments:
        result.mass_attrs_by_label[segment.label].Set(masses[segment.label])
        result.inertia_attrs_by_label[segment.label].Set(
            Gf.Vec3f(*segment.diagonal_inertia_kg_m2(mass_scale))
        )
    result.metadata["mass_scale"] = float(mass_scale)
    result.metadata["scaled_masses_kg"] = masses
    result.metadata["scaled_total_mass_kg"] = (
        policy.nominal_total_mass_kg * float(mass_scale)
    )
    scale_attr = result.metadata_attrs_by_name.get("handson:p003PhysicsMassScale")
    if scale_attr is not None and scale_attr.IsValid():
        scale_attr.Set(float(mass_scale))
    return masses


def apply_p003_physics_overlay(
    stage: Any,
    bodies: Mapping[str, Any] | Sequence[Any] | None = None,
    joints: Mapping[str, Any] | Sequence[Any] | None = None,
    mass_scale: float = 1.0,
    *,
    root_path: str = DEFAULT_P003_ROOT_PATH,
    clearance_m: float = 0.0005,
    enable_collision_proxies: bool = True,
    policy: P003PhysicsPolicy = DEFAULT_P003_PHYSICS_POLICY,
) -> P003PhysicsOverlayResult:
    """Apply the versioned tabletop policy to a composed Ruler035 stage.

    ``bodies`` and ``joints`` may be the exact prim sequences already resolved
    by the runtime, or may be omitted and resolved from ``root_path``.  Their
    paths are always checked against the pinned topology before any opinions
    are authored.  The source USD file is never opened for writing or saved.

    Collider replacement is fail-safe: an unprovable topology returns a result
    with ``proxy_topology_status == 'blocked'`` and leaves source collision
    enabled.  Mass, finite inertia, CCD, and passive joint settings are still
    authored because the rigid-body/joint paths have already been validated.
    """

    _require_positive_finite("mass_scale", mass_scale)
    _require_nonnegative_finite("clearance_m", clearance_m)
    if stage is None:
        raise ValueError("P003 physics overlay requires a valid USD stage")

    # Runtime-only dependency boundary.  Ordinary unit tests can import and
    # exercise every policy/reset function without an Isaac installation.
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics

    normalized_root = root_path.rstrip("/")
    root = stage.GetPrimAtPath(normalized_root)
    if not root.IsValid():
        raise RuntimeError(f"P003 articulation root is missing: {normalized_root}")

    body_expected = {
        segment.label: _root_join(normalized_root, segment.body_suffix)
        for segment in policy.segments
    }
    joint_expected = {
        item.label: _root_join(normalized_root, item.joint_suffix)
        for item in policy.joints
    }
    body_prims = _coerce_named_prims(
        bodies,
        tuple(segment.label for segment in policy.segments),
        body_expected,
        stage,
        "body",
    )
    joint_prims = _coerce_named_prims(
        joints,
        tuple(item.label for item in policy.joints),
        joint_expected,
        stage,
        "joint",
    )

    blocked_reason, source_paths = _topology_block_reason(
        stage=stage,
        root=root,
        bodies=body_prims,
        joints=joint_prims,
        root_path=normalized_root,
        policy=policy,
        Usd=Usd,
        UsdGeom=UsdGeom,
        UsdPhysics=UsdPhysics,
    )
    mass_attrs, inertia_attrs = _author_mass_inertia_and_ccd(
        bodies=body_prims,
        mass_scale=mass_scale,
        policy=policy,
        Gf=Gf,
        PhysxSchema=PhysxSchema,
        UsdPhysics=UsdPhysics,
    )
    friction_authored = _author_passive_joints(
        joints=joint_prims,
        policy=policy,
        PhysxSchema=PhysxSchema,
        UsdPhysics=UsdPhysics,
    )

    proxy_paths: tuple[str, ...] = ()
    status = "disabled"
    if enable_collision_proxies and blocked_reason is None:
        try:
            proxy_paths = _author_collision_proxies(
                stage=stage,
                bodies=body_prims,
                root_path=normalized_root,
                policy=policy,
                Gf=Gf,
                PhysxSchema=PhysxSchema,
                UsdGeom=UsdGeom,
                UsdPhysics=UsdPhysics,
            )
        except Exception as exc:
            blocked_reason = f"analytic proxy authoring failed: {type(exc).__name__}: {exc}"
            status = "blocked"
            _set_known_collision_mode(
                stage=stage,
                source_paths=source_paths,
                root_path=normalized_root,
                policy=policy,
                proxies_enabled=False,
                PhysxSchema=PhysxSchema,
                UsdPhysics=UsdPhysics,
            )
        else:
            # Disable source collision only after every replacement proxy has
            # passed its API and computed-visibility checks.
            _set_known_collision_mode(
                stage=stage,
                source_paths=source_paths,
                root_path=normalized_root,
                policy=policy,
                proxies_enabled=True,
                PhysxSchema=PhysxSchema,
                UsdPhysics=UsdPhysics,
            )
            for path in source_paths:
                prim = stage.GetPrimAtPath(path)
                imageable = UsdGeom.Imageable(prim)
                imageable.CreatePurposeAttr().Set(UsdGeom.Tokens.guide)
                imageable.MakeInvisible()
            status = "applied"
    elif enable_collision_proxies:
        status = "blocked"
        _set_known_collision_mode(
            stage=stage,
            source_paths=source_paths,
            root_path=normalized_root,
            policy=policy,
            proxies_enabled=False,
            PhysxSchema=PhysxSchema,
            UsdPhysics=UsdPhysics,
        )
    else:
        # Explicit source-collider mode also disables any previously-authored
        # overlay shapes, so reapplication cannot create double manifolds.
        _set_known_collision_mode(
            stage=stage,
            source_paths=source_paths,
            root_path=normalized_root,
            policy=policy,
            proxies_enabled=False,
            PhysxSchema=PhysxSchema,
            UsdPhysics=UsdPhysics,
        )

    metadata = p003_physics_policy_metadata(
        mass_scale=mass_scale,
        clearance_m=clearance_m,
        policy=policy,
    )
    metadata["collider_mode"] = status
    metadata["proxy_blocked_reason"] = blocked_reason
    metadata_attrs = _author_root_metadata(
        root=root,
        metadata=metadata,
        collider_mode=status,
        Sdf=Sdf,
    )
    bounds = policy.local_collision_bounds_m() if status == "applied" else {}
    return P003PhysicsOverlayResult(
        policy_version=policy.version,
        mass_attrs_by_label=mass_attrs,
        inertia_attrs_by_label=inertia_attrs,
        metadata_attrs_by_name=metadata_attrs,
        source_collision_paths=source_paths,
        proxy_collision_paths=proxy_paths,
        collision_local_bounds_by_label=bounds,
        reset_root_support_offset_m=reset_root_support_offset_m(clearance_m, policy),
        proxy_topology_status=status,
        proxy_blocked_reason=blocked_reason,
        joint_friction_authored=friction_authored,
        metadata=metadata,
    )


__all__ = [
    "DEFAULT_P003_PHYSICS_POLICY",
    "DEFAULT_P003_ROOT_PATH",
    "P003JointPhysics",
    "P003PhysicsOverlayResult",
    "P003PhysicsPolicy",
    "P003SegmentPhysics",
    "P003_PHYSICS_POLICY_VERSION",
    "P003_SOURCE_ASSET_SHA256",
    "apply_p003_physics_overlay",
    "classify_p003_physics_path",
    "p003_physics_policy_metadata",
    "reset_root_support_offset_m",
    "set_p003_mass_scale",
]
