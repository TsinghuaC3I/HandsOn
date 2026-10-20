"""Normalize the licensed GLB and derive GEN1 visuals/collision on CPU.

Run from repository root with trimesh, scipy, matplotlib and rtree installed.
No visible geometry is modeled. Source nodes are transformed and uniformly scaled.
"""
from pathlib import Path
import hashlib
import json
import math
import os
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[3]
TASK = Path(__file__).resolve().parent
ASSET = ROOT / 'handson/handson/sim/envs/xmls/assets/eval_gen03_orient_bottle_upright'
SOURCE = TASK / 'source/soap_dispenser.glb'
EXPECTED_SHA256 = '97aa710e4dc0377da973a65155f4ca788ba092aa62b480f2a26696c5d15327dd'
WIDTH_M = 0.065
MASS_KG = 0.100

def numbers(a):
    return ' '.join(f'{v:.12g}' for v in np.asarray(a).ravel())

def source_meshes():
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == EXPECTED_SHA256
    scene = trimesh.load(SOURCE, force='scene', process=False)
    parts = {}
    for node in scene.graph.nodes_geometry:
        if node == 'Object_4':  # Authored photography floor/backdrop (Plane_0).
            continue
        transform, key = scene.graph[node]
        mesh = scene.geometry[key].copy()
        mesh.apply_transform(transform)
        parts[node] = mesh
    body = parts['Object_6']
    axis = body.bounds.mean(axis=0)
    lo = min(m.bounds[0, 1] for m in parts.values())
    hi = max(m.bounds[1, 1] for m in parts.values())
    pivot = np.array([axis[0], (lo + hi)/2, axis[2]])
    scale = WIDTH_M / body.extents[0]
    # Source world +Y is bottle-up. Pump spout becomes canonical +X.
    rotation = Rotation.from_euler('z', 55.3795, degrees=True).as_matrix() @ Rotation.from_euler('x', 90, degrees=True).as_matrix()
    for mesh in parts.values():
        mesh.vertices = (mesh.vertices - pivot) @ rotation.T * scale
    return parts, {'uniform_scale': scale, 'source_pivot': pivot.tolist(), 'source_to_canonical_rotation': rotation.tolist(), 'source_up_axis': '+Y', 'canonical_up_axis': '+Z', 'source_units': 'glTF nominal metres, authored at nonphysical scene scale; normalized uniformly to a 65 mm bottle body', 'excluded_nodes': ['Plane_0/Object_4 (photography background)']}

def support_hull(mesh):
    # Vertices are actual source support points: no expanded grasp proxy.
    n = 120
    z = 1 - 2 * (np.arange(n) + .5) / n
    theta = np.arange(n) * math.pi * (3 - math.sqrt(5))
    dirs = np.column_stack((np.sqrt(1-z*z)*np.cos(theta), np.sqrt(1-z*z)*np.sin(theta), z))
    dirs = np.concatenate((dirs, np.eye(3), -np.eye(3)))
    selected = np.unique(np.argmax(mesh.vertices @ dirs.T, axis=0))
    return trimesh.Trimesh(vertices=mesh.vertices[selected], process=False).convex_hull

def plot_meshes(parts, hulls, filename):
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.gen1-cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(figsize=(12,6))
    for i, elev, azim in [(1,15,-65),(2,20,130),(3,15,-65)]:
        ax = fig.add_subplot(1,3,i,projection='3d')
        for name, mesh in (parts.items() if i < 3 else hulls.items()):
            color = '#d5d8dd' if name in ('Object_6','Object_7','body') else '#8795a5'
            if i == 3: color = '#e2a44a'
            ax.add_collection3d(Poly3DCollection(mesh.triangles * 1000, facecolors=color, linewidths=0, shade=True, lightsource=matplotlib.colors.LightSource(azdeg=220,altdeg=45)))
        ax.set(xlim=(-40,40),ylim=(-40,40),zlim=(-95,95))
        ax.set_box_aspect((80,80,190)); ax.view_init(elev,azim); ax.set_axis_off()
        ax.set_title('Source visuals' if i < 3 else 'Derived collision only')
    fig.suptitle('GEN1 Soap Dispenser — 65 mm body width, uniform source scale')
    fig.savefig(filename,dpi=160,bbox_inches='tight'); plt.close(fig)

def main():
    ASSET.mkdir(parents=True, exist_ok=True)
    parts, normalization = source_meshes()
    groups = {'body': trimesh.util.concatenate([parts['Object_6'], parts['Object_7']])}
    groups.update({n:m for n,m in parts.items() if n not in ('Object_6','Object_7')})
    hulls = {n:support_hull(m) for n,m in groups.items() if n not in ('body','Object_21','Object_17')}
    # Preserve paired barrel rings so the side support planes do not acquire
    # an artificial slope from unconstrained point reduction.
    barrel = groups['body'].vertices
    levels = np.unique(np.round(barrel[:,2],7))
    rings = []
    for z in levels:
        ring = np.unique(np.round(barrel[np.round(barrel[:,2],7)==z],10),axis=0)
        ring = ring[np.argsort(np.arctan2(ring[:,1],ring[:,0]))]
        rings.append(ring)
    selections = [np.concatenate((rings[10],rings[11])),
                  np.concatenate([rings[i][::2] for i in (0,3,5,8,10)]),
                  np.concatenate([rings[i][::2] for i in (11,13,16,18,21)])]
    for i, vertices in enumerate(selections):
        hulls[f'body_part_{i:02d}'] = trimesh.Trimesh(vertices=vertices,process=False).convex_hull
    decomposition = json.loads((ROOT/'.gen1-cache/gen1_pump_decomposition.json').read_text())
    for name in ('Object_21','Object_17'):
        for i, piece in enumerate(decomposition[name]):
            hulls[f'{name}_part_{i:02d}'] = trimesh.Trimesh(**piece,process=False).convex_hull
    full_source = trimesh.util.concatenate(list(parts.values()))
    audit = {}
    for name, hull in hulls.items():
        original = groups[name.split('_part_')[0]]
        eq = ConvexHull(hull.vertices).equations
        uncovered = np.maximum((original.vertices @ eq[:,:3].T + eq[:,3]).max(axis=1), 0)
        sampled, sampled_faces = trimesh.sample.sample_surface(hull,4000,seed=7)
        points = np.concatenate((hull.vertices, hull.triangles_center, sampled))
        normals = np.concatenate((hull.vertex_normals, hull.face_normals, hull.face_normals[sampled_faces]))
        exterior_probes = points + normals * 2e-6
        _, distances, _ = trimesh.proximity.closest_point(original, points)
        exposed = np.ones(len(points),dtype=bool)
        for other, other_hull in hulls.items():
            if other == name: continue
            other_eq = ConvexHull(other_hull.vertices).equations
            exposed &= ~np.all(exterior_probes @ other_eq[:,:3].T + other_eq[:,3] < -1e-9,axis=1)
        exterior_max = 0.0
        if exposed.any():
            _, exterior_distance, _ = trimesh.proximity.closest_point(full_source,points[exposed])
            exterior_max = float(exterior_distance.max())
        audit[name] = {'exterior_test':'2 micrometre outward normal probes; shared internal interfaces excluded','exposed_collider_to_full_source_sample_max_m':exterior_max,'source_faces':len(original.faces),'convex_vertices':len(hull.vertices),'convex_faces':len(hull.faces),'volume_ratio':float(original.volume/hull.volume),'source_outside_collider_plane_max_m':float(uncovered.max()),'collider_to_source_surface_sample_max_m':float(distances.max())}
    all_vertices = np.concatenate([h.vertices for h in hulls.values()])
    lower, upper = all_vertices.min(axis=0), all_vertices.max(axis=0)
    # Union-volume inertia avoids double-counting the intersecting source parts.
    pitch = .001
    axes = [np.arange(a+pitch/2,b,pitch) for a,b in zip(lower,upper)]
    grid = np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,3)
    inside = np.zeros(len(grid),dtype=bool)
    for h in hulls.values():
        bounds = h.bounds
        ix = np.flatnonzero(np.all((grid>=bounds[0]) & (grid<=bounds[1]),axis=1) & ~inside)
        eq = ConvexHull(h.vertices).equations
        for chunk in np.array_split(ix,max(1,math.ceil(len(ix)/20000))):
            inside[chunk] = np.all(grid[chunk] @ eq[:,:3].T + eq[:,3] <= 1e-10,axis=1)
    points = grid[inside]
    com = points.mean(axis=0)
    d = points - com
    inertia = MASS_KG / len(d) * (np.eye(3) * (d*d).sum() - d.T @ d)
    inertia += np.eye(3) * MASS_KG * pitch*pitch/6
    values, vectors = np.linalg.eigh(inertia)
    if np.linalg.det(vectors) < 0: vectors[:,0] *= -1
    quat = Rotation.from_matrix(vectors).as_quat(scalar_first=True)
    whole = trimesh.Trimesh(vertices=all_vertices,process=False).convex_hull
    poses, probs = whole.compute_stable_poses(center_mass=com,threshold=.001)
    candidates = []
    for T, p in zip(poses,probs):
        support_z = (all_vertices @ T[:3,:3].T)[:,2].min()
        spout_clearance = (parts['Object_17'].vertices @ T[:3,:3].T)[:,2].min() - support_z
        if abs(T[2,2]) < 1e-5 and spout_clearance >= .025:
            candidates.append((T,p))
    if not candidates: raise RuntimeError('No source-derived stable side-rest orientation found')
    T, prob = max(candidates,key=lambda x:x[1])
    # Point the bottle length along world Y, maintaining the computed support normal.
    direction = T[:3,:3] @ np.array([0.,0.,1.])
    heading = math.pi/2 - math.atan2(direction[1],direction[0])
    rest = Rotation.from_rotvec([0,0,heading]).as_matrix() @ T[:3,:3]
    source_side_rest_quat = Rotation.from_matrix(rest).as_quat(scalar_first=True)
    # Native compound contact settles about four degrees around the barrel.
    # Author the observed resting orientation before physics instead of asking
    # the ordinary origin gate to tolerate that roll. Five native layouts show
    # <= 0.431 degree drift with this half-facet (3.6 degree) refinement.
    # This changes the reset pose only; source meshes, COM and dynamics stay fixed.
    rest = rest @ Rotation.from_euler('z',-3.6,degrees=True).as_matrix()
    rest_quat = Rotation.from_matrix(rest).as_quat(scalar_first=True)
    rotated = all_vertices @ rest.T
    pose = np.r_[[-.30,-.06,.89-rotated[:,2].min()+.0005],rest_quat]
    bottom = all_vertices[all_vertices[:,2] <= lower[2]+.0002]
    geometry = {
        **normalization, 'mass_kg':MASS_KG, 'bounds_m':[lower.tolist(),upper.tolist()],
        'collision_vertices_m':all_vertices.tolist(), 'bottom_vertices_m':bottom.tolist(),
        'center_of_mass_m':com.tolist(),'diagonal_inertia_kg_m2':values.tolist(),'principal_axes_wxyz':quat.tolist(),
        'side_rest_quat_wxyz':rest_quat.tolist(),'side_rest_geometric_probability':float(prob),
        'source_facet_quat_wxyz':source_side_rest_quat.tolist(),
        'native_side_rest_roll_refinement_deg':-3.6,
        'inertia_method':'Uniform density over source-derived collider union, 1 mm voxel quadrature; fixed pump, no COM bias',
        'inertia_voxel_pitch_m':pitch, 'union_volume_m3':len(points)*pitch**3,
    }
    (ASSET/'geometry.json').write_text(json.dumps(geometry,indent=2)+'\n')
    root=ET.Element('mujoco',model='GEN1SourceSoapDispenser'); assets=ET.SubElement(root,'asset')
    world=ET.SubElement(root,'worldbody'); body=ET.SubElement(world,'body',name='gen1_soap_bottle',pos=numbers(pose[:3]),quat=numbers(pose[3:]))
    ET.SubElement(body,'freejoint',name='gen1_soap_bottle_joint')
    ET.SubElement(body,'inertial',mass=str(MASS_KG),pos=numbers(com),quat=numbers(quat),diaginertia=numbers(values))
    for name, mesh in sorted(parts.items()):
        filename=f'visual_{name}.obj'; mesh.export(ASSET/filename,include_color=False,include_texture=False)
        material=mesh.visual.material
        color=np.ones(4) if material.baseColorFactor is None else material.baseColorFactor/255
        matname=f'gen1_mat_{name}'
        ET.SubElement(assets,'material',name=matname,rgba=numbers(color),shininess=str(1-float(material.roughnessFactor or .5)),specular=str(material.metallicFactor if material.metallicFactor is not None else 1.0))
        meshname=f'gen1_visual_{name}'
        ET.SubElement(assets,'mesh',name=meshname,file=filename)
        ET.SubElement(body,'geom',name=meshname,type='mesh',mesh=meshname,material=matname,contype='0',conaffinity='0',group='1',mass='0')
    for name, mesh in sorted(hulls.items()):
        filename=f'collision_{name}.obj'; mesh.export(ASSET/filename,include_color=False,include_texture=False)
        meshname=f'gen1_collision_{name}'
        ET.SubElement(assets,'mesh',name=meshname,file=filename)
        ET.SubElement(body,'geom',name=meshname,type='mesh',mesh=meshname,group='3',rgba='0 0 0 0',friction='.8 .005 .0001',mass='0')
    ET.indent(root,space='  '); ET.ElementTree(root).write(ASSET/'bottle.xml',encoding='utf-8',xml_declaration=True)
    # The shared native importer resolves mesh paths per include, while MuJoCo
    # resolves them from the main XML. Inline this one asset into the arena so
    # both loaders use the same repository-relative closure.
    import copy
    arena_path = ROOT/'handson/handson/sim/envs/xmls/arena_arm_hand_eval_gen03_orient_bottle_upright_rokae_wuji1.xml'
    arena = ET.parse(arena_path).getroot()
    for include in list(arena.findall('include')):
        if include.get('file') == 'assets/eval_gen03_orient_bottle_upright/bottle.xml': arena.remove(include)
    for section in list(arena):
        if section.tag in ('asset','worldbody'):
            for element in list(section):
                if element.get('name','').startswith('gen1_'): section.remove(element)
    for element in assets:
        cloned = copy.deepcopy(element)
        if cloned.get('file'): cloned.set('file','assets/eval_gen03_orient_bottle_upright/'+cloned.get('file'))
        arena.find('asset').append(cloned)
    arena.find('worldbody').append(copy.deepcopy(body))
    ET.indent(arena,space='  '); ET.ElementTree(arena).write(arena_path,encoding='utf-8',xml_declaration=True)
    for obsolete in ('collision_Object_21.obj','collision_Object_17.obj','collision_body.obj'):
        (ASSET/obsolete).unlink(missing_ok=True)

    report={'source_sha256':EXPECTED_SHA256,'normalization':normalization,'dimensions_m':(upper-lower).tolist(),'mass_kg':MASS_KG,'center_of_mass_m':com.tolist(),'diagonal_inertia_kg_m2':values.tolist(),'reset_pose_wxyz':pose.tolist(),'colliders':audit,'max_exposed_surface_deviation_m':max(a['exposed_collider_to_full_source_sample_max_m'] for a in audit.values()),'actual_isaac_validation':'pending'}
    (TASK/'evidence/source_geometry_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    plot_meshes(parts,hulls,TASK/'evidence/source_geometry_views.png')
    print(json.dumps(report,indent=2))

if __name__ == '__main__': main()
