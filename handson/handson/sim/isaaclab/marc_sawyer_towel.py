"""CC-BY towel geometry with native PhysX open-source PBD cloth constraints.

See docs/marc_sawyer_towel.md for upstream values and dimensional adaptations.
The downloaded asset has no physics. Graphics follow, but never drive, physics.
"""
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

ASSET_DIR = Path(__file__).resolve().parent.parent / 'envs/usd/marc_sawyer_towel'
PHYSX_REVISION = '09ff24f3279b735e672ff27b155cbf49f6296f4d'
STRETCH_STIFFNESS = 10000.0
SHEAR_STIFFNESS = 4500.0
# A real towel crease does not act like a restoring leaf spring.  Keep the
# in-plane rest-length network so the sheet stays connected and inextensible,
# but do not author bending springs that can store and return fold energy.
BEND_STIFFNESS = 0.0
SPRING_DAMPING = 0.55
# Preserve the sheet's rest-length constraints, but damp the return motion so
# the towel does not rebound like rubber after a fold is released.
PARTICLE_DAMPING = 0.50
# The source cloth was nearly frictionless in contact.  A modest increase
# keeps the terry pile from skating under the Wuji fingers without adding a
# grasp assist or changing the recorded action contract.
PARTICLE_FRICTION = 1.60
# Nominal 500 gsm terry towel; the upstream large-sheet demo is only 64.5 gsm.
AREAL_DENSITY = 0.5
# Restore the source shell and add 2.5 mm of nominal loft on each face.
# This is an effective cloth shell, not a compressible pile simulation.
SHELL_THICKNESS_SCALE = 1.0
PILE_LOFT_M = 0.0025
MATERIAL_REVISION = 'marc-sawyer-physx-pbd-v7-non-rebounding'


@lru_cache(maxsize=1)
def asset():
    with np.load(ASSET_DIR / 'prepared.npz', allow_pickle=False) as source:
        data = {name: source[name] for name in source.files}
    for value in data.values():
        value.flags.writeable = False
    return data


@lru_cache(maxsize=1)
def provenance():
    return json.loads((ASSET_DIR / 'provenance.json').read_text())


def spring_constraints():
    """Structural/shear rest-length edges without elastic bend springs."""
    edges, stiffness = [], []
    for r in range(41):
        for c in range(41):
            i = r * 41 + c
            if r:
                edges.append((i - 41, i))
                stiffness.append(STRETCH_STIFFNESS)
            if c:
                edges.append((i - 1, i))
                stiffness.append(STRETCH_STIFFNESS)
            if r and c:
                edges.extend(((i - 42, i), (i - 41, i - 1)))
                stiffness.extend((SHEAR_STIFFNESS, SHEAR_STIFFNESS))
            if BEND_STIFFNESS > 0.0:
                if r >= 2:
                    edges.append((i - 82, i))
                    stiffness.append(BEND_STIFFNESS)
                if c >= 2:
                    edges.append((i - 2, i))
                    stiffness.append(BEND_STIFFNESS)
    edges = np.asarray(edges, dtype=np.int32)
    rest = asset()['points']
    lengths = np.linalg.norm(rest[edges[:, 1]] - rest[edges[:, 0]], axis=1)
    return edges, np.asarray(stiffness, np.float32), lengths.astype(np.float32)


def visual_points(nodes, index):
    data = asset()
    cells = np.asarray(nodes)[data[f'mesh{index}_bind_ids']]
    anchor = np.einsum('ni,nij->nj', data[f'mesh{index}_bind_weights'], cells)
    u = cells[:, 1] - cells[:, 0]
    u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
    n = np.cross(u, cells[:, 2] - cells[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    v = np.cross(n, u)
    offset = data[f'mesh{index}_bind_offset']
    return (anchor + offset[:, :1]*u + offset[:, 1:2]*v
            + shell_normal_offsets(index)[:, None]*n).astype(np.float32)


def visual_normals(points, faces):
    """Area-weighted normals, recomputed after deformation for the original mesh."""
    face_normals = np.cross(points[faces[:, 1]] - points[faces[:, 0]],
                            points[faces[:, 2]] - points[faces[:, 0]])
    normals = np.stack([
        np.bincount(faces.ravel(), weights=np.repeat(face_normals[:, axis], 3),
                    minlength=len(points)) for axis in range(3)
    ], axis=1)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    return normals.astype(np.float32)


@lru_cache(maxsize=5)
def shell_normal_offsets(index):
    """Inflate along the cloth normal with continuous source-facing weights.

    Front and back get opposite loft, while the original side walls transition
    smoothly. Unlike sign(offset), the source normals cannot flip sides where
    interpolation puts a front vertex slightly behind the coarse midsurface.
    """
    data = asset()
    ids = data[f'mesh{index}_bind_ids']
    rest = data['points'][ids]
    normal = np.cross(rest[:, 1]-rest[:, 0], rest[:, 2]-rest[:, 0])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    source_normals = visual_normals(data[f'mesh{index}_points'], data[f'mesh{index}_faces'])
    facing = np.clip(np.sum(source_normals*normal, axis=1), -1.0, 1.0)
    return (SHELL_THICKNESS_SCALE*data[f'mesh{index}_bind_offset'][:, 2]
            + PILE_LOFT_M*facing).astype(np.float32)


class MarcSawyerTowel:
    task_name = 'eval_flex02_fold_towel_twice'
    task_id = 'FLEXIBLE-001'
    label = 'Marc Sawyer white towel / PhysX SnippetPBDCloth adaptation'
    solver = 'physx_particle_cloth'

    @property
    def mass_kg(self):
        return AREAL_DENSITY * provenance()['area_m2']

    @property
    def thickness(self):
        return 2 * max(float(np.max(np.abs(shell_normal_offsets(i)))) for i in range(5))

    @property
    def spawn_lift(self):
        # Existing authored placement is 4 mm above the table. Cover the full
        # shell and collision envelope before the first PhysX reset.
        return max(0.0, self.thickness/2 + 0.0005 - 0.004)

    def topology(self):
        return asset()['points'].tolist(), asset()['triangles'].tolist()

    def metadata(self):
        return {
            **provenance(), 'solver': self.solver, 'mass_kg': self.mass_kg,
            'physics_repository': 'https://github.com/NVIDIA-Omniverse/PhysX',
            'physics_revision': PHYSX_REVISION, 'physics_license': 'BSD-3-Clause',
            'physics_source': 'physx/snippets/snippetpbdcloth/SnippetPBDCloth.cpp',
            'stretch_stiffness': STRETCH_STIFFNESS, 'shear_stiffness': SHEAR_STIFFNESS,
            'spring_damping': SPRING_DAMPING, 'bend_springs': False,
            'bend_stiffness': BEND_STIFFNESS, 'bend_model': 'plastic crease / no restoring bend spring',
            'areal_density_kg_m2': AREAL_DENSITY,
            'shell_thickness_scale': SHELL_THICKNESS_SCALE,
            'pile_loft_per_face_m': PILE_LOFT_M,
            'pile_compression_simulated': False,
            'spawn_lift_m': self.spawn_lift,
            'material_revision': MATERIAL_REVISION,
            'upstream_spring_damping': 0.001,
            'particle_rest_offset_m': self.thickness / 2,
            'particle_contact_offset_m': 1.4 * self.thickness / 2,
            'particle_friction': PARTICLE_FRICTION, 'particle_damping': PARTICLE_DAMPING,
            'elastic_rebound_profile': 'non_rebounding_plastic_fold',
            'stretch_rest_lengths_preserved': True,
            'particle_adhesion': 0.0, 'solver_position_iterations': 64,
            'self_collision': True, 'ccd': True,
            'adaptation': True, 'material_calibrated': False, 'grasp_assist': False,
            'runtime': 'Isaac Sim 5.1 / IsaacLab 2.3.2',
        }

    def surface_metadata(self):
        meta = provenance()
        return dict(label=self.label, solver_label=self.solver, nominal_mass_kg=self.mass_kg,
                    visual_vertex_count=sum(m['vertices'] for m in meta['meshes']),
                    visual_triangle_count=sum(m['triangles'] for m in meta['meshes']),
                    sim_vertex_count=1681, sim_triangle_count=3200)

    def author(self, stage, root_path, mesh_path, material_path):
        from pxr import UsdGeom, UsdPhysics, Vt
        from omni.physx.scripts import particleUtils, physicsUtils

        root = UsdGeom.Xform.Define(stage, root_path).GetPrim()
        for key, value in {'assetType': self.solver, 'sourceLicense': 'CC-BY-4.0',
                           'sourceUrl': provenance()['url'], 'clothSolver': self.solver,
                           'materialRevision': MATERIAL_REVISION}.items():
            root.SetCustomDataByKey('handson:' + key, value)
        scene = next(p.GetPath() for p in stage.Traverse() if p.IsA(UsdPhysics.Scene))
        system, material = root_path + '_particles', material_path + '_particles'
        radius = self.thickness / 2
        particleUtils.add_physx_particle_system(
            stage, system, contact_offset=1.4*radius, rest_offset=radius,
            particle_contact_offset=1.4*radius, solid_rest_offset=radius,
            fluid_rest_offset=0.0, solver_position_iterations=64,
            simulation_owner=scene, enable_ccd=True,
        )
        particleUtils.add_pbd_particle_material(
            stage, material, friction=PARTICLE_FRICTION, damping=PARTICLE_DAMPING, adhesion=0.0,
            viscosity=0.0, vorticity_confinement=0.0, surface_tension=0.0,
            cohesion=0.0, lift=0.0, drag=0.0,
        )
        physicsUtils.add_physics_material_to_prim(stage, stage.GetPrimAtPath(system), material)
        points, faces = asset()['points'], asset()['triangles']
        edges, stiffness, lengths = spring_constraints()
        particleUtils.add_physx_particle_cloth_with_constraints(
            stage, mesh_path, system,
            positions=Vt.Vec3fArray.FromNumpy(points), normals=[],
            rest_positions=Vt.Vec3fArray.FromNumpy(points),
            velocities=Vt.Vec3fArray.FromNumpy(np.zeros_like(points)),
            triangle_indices=Vt.IntArray.FromNumpy(faces.ravel()),
            spring_indices=Vt.Vec2iArray.FromNumpy(edges),
            spring_stiffnesses=Vt.FloatArray.FromNumpy(stiffness),
            spring_dampings=Vt.FloatArray.FromNumpy(np.full(len(edges), SPRING_DAMPING, np.float32)),
            spring_rest_lengths=Vt.FloatArray.FromNumpy(lengths),
            self_collision=True, self_collision_filter=True,
        )
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath(mesh_path))
        mesh.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
        UsdPhysics.MassAPI.Apply(mesh.GetPrim()).CreateMassAttr(self.mass_kg)
        self._author_visuals(stage, root_path)
        return root_path

    def _author_visuals(self, stage, root_path):
        from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt
        gltf = json.loads((ASSET_DIR / 'scene.gltf').read_text())
        materials = []
        for i, source in enumerate(gltf['materials']):
            path = root_path + f'/Looks/Material{i}'
            material = UsdShade.Material.Define(stage, path)
            shader = UsdShade.Shader.Define(stage, path + '/Surface')
            shader.CreateIdAttr('UsdPreviewSurface')
            pbr = source['pbrMetallicRoughness']
            shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(pbr['roughnessFactor'])
            shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(pbr['metallicFactor'])
            reader = UsdShade.Shader.Define(stage, path + '/UV')
            reader.CreateIdAttr('UsdPrimvarReader_float2')
            reader.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('st')
            reader.CreateOutput('result', Sdf.ValueTypeNames.Float2)
            for field, spec in [('diffuseColor', pbr['baseColorTexture']), ('normal', source.get('normalTexture'))]:
                if spec is None:
                    continue
                texture = UsdShade.Shader.Define(stage, path + '/' + field)
                texture.CreateIdAttr('UsdUVTexture')
                uri = gltf['images'][gltf['textures'][spec['index']]['source']]['uri']
                texture.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(ASSET_DIR / uri)))
                texture.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), 'result')
                for wrap in ('wrapS', 'wrapT'):
                    texture.CreateInput(wrap, Sdf.ValueTypeNames.Token).Set('repeat')
                texture.CreateInput('sourceColorSpace', Sdf.ValueTypeNames.Token).Set('raw' if field == 'normal' else 'sRGB')
                if field == 'normal':
                    texture.CreateInput('scale', Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2, 2, 2, 1))
                    texture.CreateInput('bias', Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1, -1, -1, 0))
                texture.CreateOutput('rgb', Sdf.ValueTypeNames.Float3)
                shader.CreateInput(field, Sdf.ValueTypeNames.Normal3f if field == 'normal' else Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(), 'rgb')
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
            materials.append(material)
        for i, spec in enumerate(provenance()['meshes']):
            mesh = UsdGeom.Mesh.Define(stage, root_path + f'/Visual/Part{i}')
            p, f = visual_points(asset()['points'], i), asset()[f'mesh{i}_faces']
            mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(p))
            mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(visual_normals(p, f)))
            mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
            mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(f), 3, np.int32)))
            mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(f.ravel()))
            mesh.CreateSubdivisionSchemeAttr('none')
            mesh.CreateDoubleSidedAttr(True)
            mesh.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.array([p.min(0), p.max(0)])))
            # trimesh exposes OpenGL UVs, already converted from glTF's V-down convention.
            UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set(Vt.Vec2fArray.FromNumpy(asset()[f'mesh{i}_uv']))
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(materials[spec['material']])

    def update_visual(self, stage, root_path, world_points):
        from pxr import Usd, UsdGeom, Vt
        prim = stage.GetPrimAtPath(root_path)
        inverse = np.asarray(UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim).GetInverse())
        local = (np.c_[world_points, np.ones(len(world_points))] @ inverse)[:, :3]
        for i in range(5):
            mesh = UsdGeom.Mesh(stage.GetPrimAtPath(root_path + f'/Visual/Part{i}'))
            points = visual_points(local, i)
            mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))
            mesh.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(
                visual_normals(points, asset()[f'mesh{i}_faces'])))
            mesh.GetExtentAttr().Set(Vt.Vec3fArray.FromNumpy(np.array([points.min(0), points.max(0)])))


SOURCE = MarcSawyerTowel()
