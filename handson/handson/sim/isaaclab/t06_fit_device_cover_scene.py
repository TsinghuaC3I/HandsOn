"""Scene metadata and authored-USD placement for G02 device closure."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


G02_TASK_NAME = "t06_fit_device_cover"
G02_SCENE_XML = "arena_arm_hand_t06_fit_device_cover_rokae_wuji1.xml"
G02_ASSET_ROOT_PATH = "/World/Task/G02RepairCloseDevice"
G02_SUPPORT_PATH = f"{G02_ASSET_ROOT_PATH}/workbench_support"
G02_COVER_RISER_PATH = f"{G02_ASSET_ROOT_PATH}/cover_pickup_riser"
G02_PHYSICS_HZ = 120
G02_TABLE_TOP_Z = 0.85
G02_POSITION_TOLERANCE_M = 0.012
G02_ROTATION_TOLERANCE_DEG = 8.0
G02_BODY_HEIGHT_M = 0.050
G02_BODY_LINEAR_SPEED_TOLERANCE_M_S = 0.015
G02_BODY_ANGULAR_SPEED_TOLERANCE_RAD_S = 0.17453292519943295
G02_BODY_COVER_MAX_PENETRATION_M = 0.003
G02_DYNAMIC_CONTACT_OFFSET_M = 0.006
G02_BODY_RESET_XY_BOUNDS_M = ((-0.44, -0.04), (0.08, 0.30))
G02_COVER_RESET_XY_BOUNDS_M = ((-0.44, -0.04), (-0.30, -0.08))
G02_RESET_YAW_BOUNDS_DEG = (-35.0, 35.0)
G02_COVER_SIDE_MIN_OFFSET_M = 0.025
G02_SINGLE_HAND_GRACE_CONTROL_FRAMES = 60
G02_DUAL_CONTACT_MIN_CONTROL_FRAMES = 6
G02_DUAL_CONTACT_DROPOUT_GRACE_CONTROL_FRAMES = 6
G02_DUAL_TRANSPORT_MIN_DISTANCE_M = 0.100
G02_DUAL_TRANSPORT_TARGET_XY_TOLERANCE_M = 0.060
G02_STABLE_CONTROL_FRAMES = 8

# The source SceneSmith layout faces +Y.  HandsOn's operator station faces +X,
# so all authored USDs receive Rz(-90 degrees).  The two manipulated objects
# are deliberately re-laid out in separate source/target zones rather than
# preserving the source preview's one-sided +Y row.  The cover starts at
# y=-0.18 m with enough overhang for the hands to take either pair of opposite
# cover edges before carrying it together to the body at y=+0.18 m.
G02_SCENE_TO_HANDSON_QUAT_WXYZ = (
    0.7071067811865476,
    0.0,
    0.0,
    -0.7071067811865475,
)
G02_DEVICE_BODY_HOME = (-0.10, 0.18, 0.852)
G02_DEVICE_COVER_HOME = (-0.10, -0.18, 0.877)

G02_SUCCESS_CONDITION_LABELS = (
    "Both hands held opposite cover sides for at least 0.2 seconds",
    "device_cover moved at least 100 mm during continuous two-hand contact and reached above device_body",
    "Current cover contact is either both opposite sides or neither hand",
    "device_body is dynamically stable at its current pose",
    "device_cover is horizontally aligned within 12 mm",
    "device_cover is seated at the body interface within 12 mm",
    "device_cover rotation matches device_body within 8 degrees",
    "device_body and device_cover have physical contact",
    "device_body/device_cover penetration is at most 3 mm",
    "All conditions remain valid for eight consecutive control frames",
)


@dataclass(frozen=True)
class G02AssetSpec:
    name: str
    relative_usd: str
    translation: tuple[float, float, float]
    dynamic: bool
    kinematic: bool = False
    mass_kg: float = 0.0
    static_friction: float = 1.6
    dynamic_friction: float = 1.3
    restitution: float = 0.0
    collision_approximation: str = "convexHull"
    dimensions_m: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def source_path(self) -> Path:
        return G02_ASSET_DIRECTORY / self.relative_usd

    @property
    def prim_path(self) -> str:
        return f"{G02_ASSET_ROOT_PATH}/{self.name}"


G02_ASSET_DIRECTORY = (
    Path(__file__).resolve().parent.parent
    / "envs"
    / "usd"
    / "t06_fit_device_cover"
)

G02_ASSET_SPECS = (
    G02AssetSpec(
        name="workbench",
        relative_usd="workbench/workbench.usd",
        translation=(-0.27, 0.0, 0.0),
        dynamic=False,
        static_friction=0.85,
        dynamic_friction=0.75,
        collision_approximation="meshSimplification",
        dimensions_m=(1.75, 0.78, 0.85),
    ),
    G02AssetSpec(
        name="device_body",
        relative_usd="device_body/device_body.usd",
        translation=G02_DEVICE_BODY_HOME,
        dynamic=True,
        # The body is a free dynamic object: reset randomizes its pose, then
        # both hands and the cover may move it through ordinary PhysX contact.
        kinematic=False,
        mass_kg=0.450,
        dimensions_m=(0.182, 0.250, 0.050),
    ),
    G02AssetSpec(
        name="device_cover",
        relative_usd="device_cover/device_cover.usd",
        translation=G02_DEVICE_COVER_HOME,
        dynamic=True,
        # The broad panel is now shared by both hands, so 160 g gives it more
        # useful inertia without approaching the overly heavy 432.4 g source.
        mass_kg=0.160,
        dimensions_m=(0.180, 0.260, 0.022),
    ),
)
G02_DYNAMIC_ASSET_SPECS = tuple(spec for spec in G02_ASSET_SPECS if spec.dynamic)


def validate_g02_asset_files() -> tuple[Path, ...]:
    """Return the self-contained source USDs after basic provenance checks."""

    missing = [spec.source_path for spec in G02_ASSET_SPECS if not spec.source_path.is_file()]
    if missing:
        detail = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"G02 asset bundle is incomplete: {detail}")
    empty = [spec.source_path for spec in G02_ASSET_SPECS if spec.source_path.stat().st_size < 1024]
    if empty:
        detail = ", ".join(str(path) for path in empty)
        raise ValueError(f"G02 asset files are unexpectedly small: {detail}")
    return tuple(spec.source_path for spec in G02_ASSET_SPECS)


def spawn_g02_repair_assets(stage: Any, sim_utils: Any) -> tuple[str, ...]:
    """Spawn the workbench and the two formal rigid task bodies."""

    from isaaclab.sim import schemas
    from pxr import Usd, UsdGeom, UsdPhysics

    validate_g02_asset_files()
    root = UsdGeom.Xform.Define(stage, G02_ASSET_ROOT_PATH)
    root.GetPrim().SetCustomDataByKey("handson:task", G02_TASK_NAME)
    root.GetPrim().SetCustomDataByKey(
        "handson:sourceScene",
        "scene_06_electronics_repair",
    )

    spawned: list[str] = []
    for spec in G02_ASSET_SPECS:
        collision_props = sim_utils.CollisionPropertiesCfg(
            contact_offset=(
                G02_DYNAMIC_CONTACT_OFFSET_M if spec.dynamic else 0.002
            ),
            rest_offset=0.0,
        )
        spawn_cfg = sim_utils.UsdFileCfg(
            usd_path=str(spec.source_path.resolve()),
            semantic_tags=[
                ("class", "workbench" if not spec.dynamic else spec.name),
                ("instance", spec.name),
            ],
            collision_props=collision_props,
        )
        spawn_cfg.func(
            spec.prim_path,
            spawn_cfg,
            translation=spec.translation,
            orientation=G02_SCENE_TO_HANDSON_QUAT_WXYZ,
        )
        prim = stage.GetPrimAtPath(spec.prim_path)
        prim.SetCustomDataByKey("handson:sourceAsset", spec.relative_usd)
        prim.SetCustomDataByKey("handson:formalTruth", bool(spec.dynamic))

        if spec.name == "workbench":
            # The supplier workbench is a single concave triangle mesh whose
            # top faces do not generate reliable GPU-PhysX support contacts.
            # Keep it as the exact visual asset and replace only its collision
            # with the axis-aligned top/support volume authored below.
            for descendant in Usd.PrimRange(prim):
                if descendant.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(descendant).CreateCollisionEnabledAttr(
                        False
                    )

        material_path = f"{spec.prim_path}/physicsMaterial"
        material_cfg = sim_utils.RigidBodyMaterialCfg(
            static_friction=spec.static_friction,
            dynamic_friction=spec.dynamic_friction,
            restitution=spec.restitution,
            friction_combine_mode="max" if spec.dynamic else "average",
            restitution_combine_mode="min",
        )
        material_cfg.func(material_path, material_cfg)
        sim_utils.bind_physics_material(
            spec.prim_path,
            material_path,
            stage=stage,
            stronger_than_descendants=True,
        )

        if spec.dynamic:
            schemas.define_rigid_body_properties(
                prim_path=spec.prim_path,
                cfg=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=spec.kinematic,
                    disable_gravity=spec.kinematic,
                    retain_accelerations=False,
                    linear_damping=0.12,
                    angular_damping=0.80,
                    max_depenetration_velocity=1.0,
                    solver_position_iteration_count=32,
                    solver_velocity_iteration_count=4,
                    sleep_threshold=0.02,
                    stabilization_threshold=0.005,
                ),
            )
            schemas.define_mass_properties(
                prim_path=spec.prim_path,
                cfg=sim_utils.MassPropertiesCfg(mass=spec.mass_kg),
            )
        spawned.append(spec.prim_path)

    # Rz(-90 degrees) swaps the source workbench's 1.75 x 0.78 m footprint.
    # The 40 mm slab ends exactly at z=0.85, so the supplier geometry remains
    # visually authoritative while dynamic parts get a simple, stable support.
    support_cfg = sim_utils.CuboidCfg(
        size=(0.78, 1.75, 0.040),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.002,
            rest_offset=0.0,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.85,
            dynamic_friction=0.75,
            restitution=0.0,
            friction_combine_mode="average",
            restitution_combine_mode="min",
        ),
    )
    support_cfg.func(
        G02_SUPPORT_PATH,
        support_cfg,
        translation=(-0.27, 0.0, 0.830),
    )
    support_prim = stage.GetPrimAtPath(G02_SUPPORT_PATH)
    support_prim.SetCustomDataByKey("handson:collisionProxy", "workbench_top")
    UsdGeom.Imageable(support_prim).MakeInvisible()
    spawned.append(G02_SUPPORT_PATH)

    # A thin, broad cover lying flush on the tabletop gives Wuji no opposing
    # contact below its edge.  This compact center riser leaves 60+ mm of free
    # overhang on every side, turning the pickup into a genuine pinch instead
    # of a friction-only attempt to scrape the panel off the table.
    riser_cfg = sim_utils.CuboidCfg(
        size=(0.060, 0.100, 0.025),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.002,
            rest_offset=0.0,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=0.9,
            restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.18, 0.20, 0.22),
            roughness=0.85,
        ),
    )
    riser_cfg.func(
        G02_COVER_RISER_PATH,
        riser_cfg,
        translation=(
            G02_DEVICE_COVER_HOME[0],
            G02_DEVICE_COVER_HOME[1],
            0.8625,
        ),
    )
    riser_prim = stage.GetPrimAtPath(G02_COVER_RISER_PATH)
    # Keep the compact support fixed during an episode, but make its reset
    # transform writable so it can remain centered beneath the broadly
    # randomized dynamic cover instead of constraining the cover to +/-20 mm.
    schemas.define_rigid_body_properties(
        prim_path=G02_COVER_RISER_PATH,
        cfg=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True,
            disable_gravity=True,
            retain_accelerations=False,
        ),
    )
    riser_prim.SetCustomDataByKey("handson:fixture", "cover_pickup_riser")
    spawned.append(G02_COVER_RISER_PATH)
    return tuple(spawned)


__all__ = [
    "G02_ASSET_DIRECTORY",
    "G02_ASSET_ROOT_PATH",
    "G02_ASSET_SPECS",
    "G02_BODY_ANGULAR_SPEED_TOLERANCE_RAD_S",
    "G02_BODY_COVER_MAX_PENETRATION_M",
    "G02_BODY_HEIGHT_M",
    "G02_BODY_LINEAR_SPEED_TOLERANCE_M_S",
    "G02_BODY_RESET_XY_BOUNDS_M",
    "G02_COVER_SIDE_MIN_OFFSET_M",
    "G02_COVER_RESET_XY_BOUNDS_M",
    "G02_COVER_RISER_PATH",
    "G02_DEVICE_BODY_HOME",
    "G02_DEVICE_COVER_HOME",
    "G02_DYNAMIC_CONTACT_OFFSET_M",
    "G02_DYNAMIC_ASSET_SPECS",
    "G02_DUAL_CONTACT_MIN_CONTROL_FRAMES",
    "G02_DUAL_CONTACT_DROPOUT_GRACE_CONTROL_FRAMES",
    "G02_DUAL_TRANSPORT_MIN_DISTANCE_M",
    "G02_DUAL_TRANSPORT_TARGET_XY_TOLERANCE_M",
    "G02_PHYSICS_HZ",
    "G02_POSITION_TOLERANCE_M",
    "G02_RESET_YAW_BOUNDS_DEG",
    "G02_ROTATION_TOLERANCE_DEG",
    "G02_SCENE_XML",
    "G02_SINGLE_HAND_GRACE_CONTROL_FRAMES",
    "G02_STABLE_CONTROL_FRAMES",
    "G02_SUPPORT_PATH",
    "G02_SUCCESS_CONDITION_LABELS",
    "G02_TABLE_TOP_Z",
    "G02_TASK_NAME",
    "spawn_g02_repair_assets",
    "validate_g02_asset_files",
]
