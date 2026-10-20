"""Source-backed towel assets adapted to the fixed Isaac Sim 5.1 runtime.

Source snapshots/licenses: third_party/towel_sources. Port limitations and
parameter provenance: docs/open_source_towel_comparison.md. No particle picker
or grasp assistance is used by either collection environment.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from . import flexible_cotton_towel as cotton


@dataclass(frozen=True)
class TowelSource:
    task_name: str
    task_id: str
    label: str
    repository: str
    revision: str
    license: str
    size_xy: tuple[float, float]
    mass_kg: float
    solver: str
    source_files: tuple[str, ...]

    def metadata(self):
        data = asdict(self)
        data.update({
            "adaptation": True,
            "runtime": "Isaac Sim 5.1 / IsaacLab 2.3.2",
            "grasp_assist": False,
            "material_calibrated": False,
            "grid": [cotton.ROWS, cotton.COLUMNS],
        })
        if self.solver == "physx_particle_cloth":
            data["upstream_flex_stretch_bend_shear"] = [2.0, 0.5, 1.0]
            data["port_spring_stretch_shear_bend"] = [9000.0, 4500.0, 0.06]
            data["port_parameter_note"] = "Native dimensional coefficients; not equivalent to FleX stiffness"
        else:
            data["physics_note"] = "Source geometry/material; HandsOn native 600 gsm surface physics"
        return data

    def topology(self):
        # Both upstream flat templates are rectangular sheets. Retain their
        # dimensions and corner order; the task's material coordinates require
        # a regular 41 x 41 tessellation instead of Blender's arbitrary indices.
        if self.task_name == "flexible-towel-synthetic-cloth":
            import numpy as np
            path = Path(__file__).resolve().parent / "usd/open_source_towels/synthetic_cloth.npz"
            with np.load(path, allow_pickle=False) as asset:
                return asset["points"].tolist(), asset["triangles"].tolist()
        return cotton.towel_topology(self.size_xy)

    def author(self, stage, root_path, mesh_path, material_path):
        texture = (
            Path(__file__).resolve().parent / "usd/open_source_towels/synthetic_grid.png"
            if self.solver != "physx_particle_cloth" else None
        )
        cotton.author_towel(
            stage, root_path, mesh_path, material_path,
            size_xy=self.size_xy, mass=self.mass_kg, color_texture=texture,
            color=(0.82, 0.32, 0.16) if texture is None else None,
            mesh_data=self.topology(),
        )
        if self.solver == "physx_particle_cloth":
            self._author_particle_physics(stage, root_path, mesh_path, material_path)
        root = stage.GetPrimAtPath(root_path)
        root.SetCustomDataByKey("handson:materialRevision", self.task_name + "-v1")
        root.SetCustomDataByKey("handson:sourceRepository", self.repository)
        root.SetCustomDataByKey("handson:sourceRevision", self.revision)
        root.SetCustomDataByKey("handson:sourceLicense", self.license)
        root.SetCustomDataByKey("handson:clothSolver", self.solver)
        root.SetCustomDataByKey("handson:assetType", self.solver + "_with_render_shell")
        root.SetCustomDataByKey("handson:sourceAdaptation", True)
        return root_path

    def _author_particle_physics(self, stage, root_path, mesh_path, material_path):
        from pxr import UsdPhysics
        from omni.physx.scripts import particleUtils, physicsUtils

        root = stage.GetPrimAtPath(root_path)
        mesh = stage.GetPrimAtPath(mesh_path)
        # Replace the common author's FEM schemas before first physics reset.
        # There is one solver actor and one invisible simulation mesh.
        for api in ("OmniPhysicsDeformableBodyAPI", "PhysxSurfaceDeformableBodyAPI"):
            root.RemoveAPI(api)
        for api in ("OmniPhysicsSurfaceDeformableSimAPI", "PhysicsCollisionAPI", "PhysxCollisionAPI"):
            mesh.RemoveAPI(api)
        scene = next(p.GetPath() for p in stage.Traverse() if p.IsA(UsdPhysics.Scene))
        system = root_path + "_particles"
        material = material_path + "_particles"
        particleUtils.add_physx_particle_system(
            stage, system, contact_offset=0.006, rest_offset=0.0015,
            particle_contact_offset=0.0035, solid_rest_offset=0.0015,
            fluid_rest_offset=0.0, solver_position_iterations=64,
            simulation_owner=scene, enable_ccd=True,
        )
        particleUtils.add_pbd_particle_material(stage, material, friction=0.75, drag=0.0, lift=0.0)
        physicsUtils.add_physics_material_to_prim(stage, stage.GetPrimAtPath(system), material)
        # FleX's dimensionless coefficients cannot be passed unchanged to
        # PhysX. These are explicit, independently testable adapter settings.
        particleUtils.add_physx_particle_cloth(
            stage, mesh_path, None, system,
            spring_stretch_stiffness=9000.0, spring_shear_stiffness=4500.0,
            spring_bend_stiffness=0.06, spring_damping=0.2,
            self_collision=True, self_collision_filter=True,
        )
        UsdPhysics.MassAPI.Apply(mesh).CreateMassAttr(self.mass_kg)


SOURCES = {
    "flexible-towel-softgym": TowelSource(
        "flexible-towel-softgym", "FLEXIBLE-003", "SoftGym 毛巾 · 粒子布料适配",
        "https://github.com/halid1020/softgym",
        "8a2196648c4ef6c4463ca862324bae442bcba22d", "BSD-3-Clause",
        (0.30, 0.30), 0.0054, "physx_particle_cloth",
        ("benchmarks/fabric_domain_builder.py", "PyFlex/bindings/softgym_scenes/softgym_cloth.h"),
    ),
    "flexible-towel-synthetic-cloth": TowelSource(
        "flexible-towel-synthetic-cloth", "FLEXIBLE-004", "Synthetic Cloth 毛巾 · 网格资产适配",
        "https://github.com/tlpss/synthetic-cloth-data",
        "9cc1a04f3d7f3c9b2abf1642acb1f7a2a4de51b5", "MIT",
        (0.30, 0.50), 0.090, "physx_surface_deformable",
        ("synthetic-cloth-data/synthetic_cloth_data/meshes/flat_meshes/geometric_templates.py",
         "synthetic-cloth-data/synthetic_cloth_data/materials/towels.py"),
    ),
}


class ParticleClothViewAdapter:
    """Expose native particle tensors through the shared cloth lifecycle API.

    No global patching, node pinning or trajectory correction. Positions and
    velocities remain native solver state and use the existing recorder fields.
    """
    def __init__(self, view):
        self.view = view

    @property
    def count(self):
        return self.view.count

    @property
    def max_simulation_elements_per_body(self):
        return len(cotton.towel_topology()[1])

    def check(self):
        return self.view.check()

    def get_simulation_nodal_positions(self):
        return self.view.get_positions().reshape(self.count, -1, 3)

    def get_simulation_nodal_velocities(self):
        return self.view.get_velocities().reshape(self.count, -1, 3)

    def set_simulation_nodal_positions(self, positions, indices):
        return self.view.set_positions(positions.reshape(self.count, -1), indices)

    def set_simulation_nodal_velocities(self, velocities, indices):
        return self.view.set_velocities(velocities.reshape(self.count, -1), indices)
