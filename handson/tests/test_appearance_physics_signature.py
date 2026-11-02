"""Deinstancing must preserve collision evidence, rather than hiding it."""
import pytest

pytest.importorskip('pxr')
from pxr import Usd, UsdGeom, UsdPhysics

from handson.sim.isaaclab.blender_visual_layer import physics_signature


def test_instance_collision_is_checked_before_and_after_material_graft():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, '/Template')
    collider = UsdGeom.Cube.Define(stage, '/Template/Collider')
    collider.CreateSizeAttr(0.2)
    UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
    instance = UsdGeom.Xform.Define(stage, '/World/Object').GetPrim()
    instance.GetReferences().AddInternalReference('/Template')
    instance.SetInstanceable(True)
    before = physics_signature(stage)
    instance.SetInstanceable(False)
    assert physics_signature(stage) == before
    UsdGeom.Cube(stage.GetPrimAtPath('/World/Object/Collider')).GetSizeAttr().Set(0.3)
    assert physics_signature(stage) != before
