"""Approved D03 physical profiles; current scene and controller are shared."""

from typing import Any

friction_4_4_D03_B04_PHYSICS_MATERIAL_PATH = '/World/PhysicsMaterials/HandsOnD03B04SmallBlockStability'

friction_4_4_STACK_ANGULAR_DAMPING = 1.0

friction_4_4_STACK_CONTACT_OFFSET = 0.001

friction_4_4_STACK_LINEAR_DAMPING = 0.25

friction_4_4_STACK_REST_OFFSET = 0.0

friction_4_4_STACK_SLEEP_THRESHOLD = 0.01

friction_4_4_STACK_SMALL_DYNAMIC_FRICTION = 4.0

friction_4_4_STACK_SMALL_STATIC_FRICTION = 4.0

def configure_friction_4_4(self, sim_utils: Any) -> bool:
    """Apply B04's proven small-block contact contract to D03 unchanged.

        D03 uses the exact same BuildingBlock017 visual and convex collider as
        B04.  Reuse B04's PhysX friction, contact offsets, damping, and sleep
        threshold too; do not replace its source 20 g mass or invent a second
        solver profile for the handoff task.
        """
    if self._stage is None:
        return False
    target_prim = self._find_rigid_body_prim('d03_target_root')
    if target_prim is None:
        return False
    target_path = str(target_prim.GetPath())
    sim_utils.make_uninstanceable(target_path, stage=self._stage)
    sim_utils.update_stage()
    target_prim = self._find_rigid_body_prim('d03_target_root')
    if target_prim is None:
        return False
    material_cfg = sim_utils.RigidBodyMaterialCfg(static_friction=friction_4_4_STACK_SMALL_STATIC_FRICTION, dynamic_friction=friction_4_4_STACK_SMALL_DYNAMIC_FRICTION, restitution=0.0, friction_combine_mode='max', restitution_combine_mode='min')
    material_cfg.func(friction_4_4_D03_B04_PHYSICS_MATERIAL_PATH, material_cfg)
    sim_utils.bind_physics_material(target_path, friction_4_4_D03_B04_PHYSICS_MATERIAL_PATH, stage=self._stage, stronger_than_descendants=True)
    from pxr import PhysxSchema, Usd, UsdPhysics
    collider_count = 0
    for descendant in Usd.PrimRange(target_prim):
        if not descendant.HasAPI(UsdPhysics.CollisionAPI):
            continue
        collision_api = PhysxSchema.PhysxCollisionAPI(descendant)
        if not collision_api:
            collision_api = PhysxSchema.PhysxCollisionAPI.Apply(descendant)
        contact_attr = collision_api.GetContactOffsetAttr()
        if not contact_attr or not contact_attr.IsValid():
            contact_attr = collision_api.CreateContactOffsetAttr()
        contact_attr.Set(friction_4_4_STACK_CONTACT_OFFSET)
        rest_attr = collision_api.GetRestOffsetAttr()
        if not rest_attr or not rest_attr.IsValid():
            rest_attr = collision_api.CreateRestOffsetAttr()
        rest_attr.Set(friction_4_4_STACK_REST_OFFSET)
        collider_count += 1
    rigid_api = PhysxSchema.PhysxRigidBodyAPI(target_prim)
    if not rigid_api:
        rigid_api = PhysxSchema.PhysxRigidBodyAPI.Apply(target_prim)
    for getter_name, creator_name, value in (('GetLinearDampingAttr', 'CreateLinearDampingAttr', friction_4_4_STACK_LINEAR_DAMPING), ('GetAngularDampingAttr', 'CreateAngularDampingAttr', friction_4_4_STACK_ANGULAR_DAMPING), ('GetSleepThresholdAttr', 'CreateSleepThresholdAttr', friction_4_4_STACK_SLEEP_THRESHOLD)):
        attribute = getattr(rigid_api, getter_name)()
        if not attribute or not attribute.IsValid():
            attribute = getattr(rigid_api, creator_name)()
        attribute.Set(value)
    return collider_count > 0

friction_4_2p5_D03_TARGET_DYNAMIC_FRICTION = 2.5

friction_4_2p5_D03_TARGET_ENABLE_CCD = False

friction_4_2p5_D03_TARGET_ENABLE_SPECULATIVE_CCD = True

friction_4_2p5_D03_TARGET_MASS = 0.02

friction_4_2p5_D03_TARGET_MAX_ANGULAR_VELOCITY = 30.0

friction_4_2p5_D03_TARGET_MAX_DEPENETRATION_VELOCITY = 0.5

friction_4_2p5_D03_TARGET_MAX_LINEAR_VELOCITY = 1.0

friction_4_2p5_D03_TARGET_PHYSICS_MATERIAL_PATH = '/World/PhysicsMaterials/HandsOnD03TargetContactSafety'

friction_4_2p5_D03_TARGET_SOLVER_POSITION_ITERATIONS = 64

friction_4_2p5_D03_TARGET_SOLVER_VELOCITY_ITERATIONS = 8

friction_4_2p5_D03_TARGET_STATIC_FRICTION = 4.0

friction_4_2p5_STACK_ANGULAR_DAMPING = 1.0

friction_4_2p5_STACK_CONTACT_OFFSET = 0.001

friction_4_2p5_STACK_LINEAR_DAMPING = 0.25

friction_4_2p5_STACK_REST_OFFSET = 0.0

friction_4_2p5_STACK_SLEEP_THRESHOLD = 0.01

def configure_friction_4_2p5(self, sim_utils: Any) -> bool:
    """Keep D03's 20 g handoff block inside a bounded contact envelope.

        D03 uses B04's visual/collider but needs stricter velocity and mass
        bounds for two-hand squeezing in GPU PhysX.
        """
    if self._stage is None:
        return False
    target_prim = self._find_rigid_body_prim('d03_target_root')
    if target_prim is None:
        return False
    target_path = str(target_prim.GetPath())
    sim_utils.make_uninstanceable(target_path, stage=self._stage)
    sim_utils.update_stage()
    target_prim = self._find_rigid_body_prim('d03_target_root')
    if target_prim is None:
        return False
    material_cfg = sim_utils.RigidBodyMaterialCfg(static_friction=friction_4_2p5_D03_TARGET_STATIC_FRICTION, dynamic_friction=friction_4_2p5_D03_TARGET_DYNAMIC_FRICTION, restitution=0.0, friction_combine_mode='max', restitution_combine_mode='min')
    material_cfg.func(friction_4_2p5_D03_TARGET_PHYSICS_MATERIAL_PATH, material_cfg)
    sim_utils.bind_physics_material(target_path, friction_4_2p5_D03_TARGET_PHYSICS_MATERIAL_PATH, stage=self._stage, stronger_than_descendants=True)
    from pxr import PhysxSchema, Usd, UsdPhysics
    mass_api_factory = getattr(UsdPhysics, 'MassAPI', None)
    if mass_api_factory is not None:
        mass_api = mass_api_factory(target_prim)
        if not mass_api:
            mass_api = mass_api_factory.Apply(target_prim)
        mass_attr = mass_api.GetMassAttr()
        if not mass_attr or not mass_attr.IsValid():
            mass_attr = mass_api.CreateMassAttr()
        mass_attr.Set(friction_4_2p5_D03_TARGET_MASS)
    collider_count = 0
    for descendant in Usd.PrimRange(target_prim):
        if not descendant.HasAPI(UsdPhysics.CollisionAPI):
            continue
        collision_api = PhysxSchema.PhysxCollisionAPI(descendant)
        if not collision_api:
            collision_api = PhysxSchema.PhysxCollisionAPI.Apply(descendant)
        contact_attr = collision_api.GetContactOffsetAttr()
        if not contact_attr or not contact_attr.IsValid():
            contact_attr = collision_api.CreateContactOffsetAttr()
        contact_attr.Set(friction_4_2p5_STACK_CONTACT_OFFSET)
        rest_attr = collision_api.GetRestOffsetAttr()
        if not rest_attr or not rest_attr.IsValid():
            rest_attr = collision_api.CreateRestOffsetAttr()
        rest_attr.Set(friction_4_2p5_STACK_REST_OFFSET)
        collider_count += 1
    rigid_api = PhysxSchema.PhysxRigidBodyAPI(target_prim)
    if not rigid_api:
        rigid_api = PhysxSchema.PhysxRigidBodyAPI.Apply(target_prim)
    for getter_name, creator_name, value in (('GetEnableCCDAttr', 'CreateEnableCCDAttr', friction_4_2p5_D03_TARGET_ENABLE_CCD), ('GetEnableSpeculativeCCDAttr', 'CreateEnableSpeculativeCCDAttr', friction_4_2p5_D03_TARGET_ENABLE_SPECULATIVE_CCD), ('GetSolverPositionIterationCountAttr', 'CreateSolverPositionIterationCountAttr', friction_4_2p5_D03_TARGET_SOLVER_POSITION_ITERATIONS), ('GetSolverVelocityIterationCountAttr', 'CreateSolverVelocityIterationCountAttr', friction_4_2p5_D03_TARGET_SOLVER_VELOCITY_ITERATIONS), ('GetMaxDepenetrationVelocityAttr', 'CreateMaxDepenetrationVelocityAttr', friction_4_2p5_D03_TARGET_MAX_DEPENETRATION_VELOCITY), ('GetMaxLinearVelocityAttr', 'CreateMaxLinearVelocityAttr', friction_4_2p5_D03_TARGET_MAX_LINEAR_VELOCITY), ('GetMaxAngularVelocityAttr', 'CreateMaxAngularVelocityAttr', friction_4_2p5_D03_TARGET_MAX_ANGULAR_VELOCITY), ('GetLinearDampingAttr', 'CreateLinearDampingAttr', friction_4_2p5_STACK_LINEAR_DAMPING), ('GetAngularDampingAttr', 'CreateAngularDampingAttr', friction_4_2p5_STACK_ANGULAR_DAMPING), ('GetSleepThresholdAttr', 'CreateSleepThresholdAttr', friction_4_2p5_STACK_SLEEP_THRESHOLD)):
        getter = getattr(rigid_api, getter_name, None)
        creator = getattr(rigid_api, creator_name, None)
        if getter is None:
            continue
        attribute = getter()
        if not attribute or not attribute.IsValid():
            if creator is None:
                continue
            attribute = creator()
        attribute.Set(value)
    return collider_count > 0
