"""Paper Core scrub scene; Transfer keeps its separate collision settings."""
from pathlib import Path
from typing import Any

from .abc_collection_env import (
    HandsOnABCIsaacLabEnv, _HS005_TASK_NAME, _TASK_ROOT,
    _HS005_ASSET_LAYER_BY_SCENE as _SHARED_SCENE_LAYERS,
    HS005_MODERN_APARTMENT_SINK_SCENE, selected_simulation_scene,
)

_HS005_ASSET_LAYER = (Path(__file__).resolve().parents[1]
    / "envs/usd/hs005_sink/scene/core_faucet_assets.usda")
_HS005_ASSET_LAYER_BY_SCENE = {
    **_SHARED_SCENE_LAYERS,
    HS005_MODERN_APARTMENT_SINK_SCENE: _HS005_ASSET_LAYER,
}


class CoreScrubBottleEnv(HandsOnABCIsaacLabEnv):
    def __init__(self, task_name="t37_scrub_bottle", **kwargs):
        if task_name != _HS005_TASK_NAME:
            raise ValueError("CoreScrubBottleEnv supports only t37_scrub_bottle")
        super().__init__(task_name=task_name, **kwargs)

    def _configure_task_collision_safety(self, sim_utils):
        # In source 17d1dd817997 this task reaches the (0, 0) return.
        # Later hand shells and bottle/brush tuning belong to the shared scene.
        return 0, 0

    def _spawn_hs005_task_specific_assets(self, sim_utils: Any) -> None:
        if self.task_name != _HS005_TASK_NAME:
            return None
        if self._stage is None:
            raise RuntimeError("HS005 requires a live USD stage")
        requested_scene = selected_simulation_scene()
        asset_layer = _HS005_ASSET_LAYER_BY_SCENE.get(
            requested_scene,
            _HS005_ASSET_LAYER,
        )
        if not asset_layer.is_file():
            raise FileNotFoundError(
                f"HS005 authoritative asset layer is missing: {asset_layer}"
            )

        from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

        root_path = f"{_TASK_ROOT}/HS005Assets"
        root = UsdGeom.Xform.Define(self._stage, root_path).GetPrim()
        root.GetReferences().ClearReferences()
        root.GetReferences().AddReference(str(asset_layer))
        root.SetCustomDataByKey("runtime_scope", "collection_vertical_slice")
        root.SetCustomDataByKey("benchmark_task_id", "B320-HS005")
        root.SetCustomDataByKey("simulation_scene", requested_scene)

        # Every HS005 scene provides its own countertop and sink.  The generic
        # bootstrap table must be absent both visually and physically: its
        # 0.89 m collision top crosses the deeper downloaded basins and can
        # eject correctly placed props at the first physics step.
        table_prim = self._find_prim("table")
        disabled_table_colliders = 0
        if table_prim is not None and table_prim.IsValid():
            table_path = str(table_prim.GetPath())
            # MJCF task bodies are normally instanceable.  Expand this one
            # before first PhysX so collisionEnabled can be authored on its
            # actual collision descendants instead of silently skipping the
            # instance proxies.
            sim_utils.make_uninstanceable(table_path, stage=self._stage)
            table_prim = self._stage.GetPrimAtPath(table_path)
            table_imageable = UsdGeom.Imageable(table_prim)
            if table_imageable:
                table_imageable.MakeInvisible()
            for prim in Usd.PrimRange(table_prim):
                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                collision = UsdPhysics.CollisionAPI(prim)
                collision.CreateCollisionEnabledAttr().Set(False)
                disabled_table_colliders += 1

        material = UsdShade.Material.Define(
            self._stage,
            f"{root_path}/Looks/HS005PhysicsMaterial",
        )
        physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physics_material.CreateStaticFrictionAttr(self._hs005_friction)
        physics_material.CreateDynamicFrictionAttr(self._hs005_friction)
        physics_material.CreateRestitutionAttr(0.0)

        bound_colliders = 0
        for prim in Usd.PrimRange(root):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            binding = UsdShade.MaterialBindingAPI.Apply(prim)
            binding.Bind(
                material,
                UsdShade.Tokens.weakerThanDescendants,
                "physics",
            )
            bound_colliders += 1
        if bound_colliders < 2:
            raise RuntimeError(
                "HS005 asset layer did not expose the expected physical colliders"
            )
        print(
            "[HandsOn/IsaacLab] Composed B320-HS005 asset layer "
            f"({bound_colliders} physics-material bindings, "
            f"{disabled_table_colliders} generic table colliders disabled): "
            f"{asset_layer}",
            flush=True,
        )
        return None
