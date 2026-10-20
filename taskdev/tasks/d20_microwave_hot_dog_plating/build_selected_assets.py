#!/usr/bin/env python3
"""Rebuild D20 from the eight user-downloaded archives; never alter originals.

CPU conversion requires numpy, trimesh, Pillow and coacd. Runtime consumes the
checked-in OBJ/PNG/JSON files and has no coacd dependency. Source glTF UVs and
PBR textures are retained. Convex decomposition is performed per rigid link.
"""
from pathlib import Path
import argparse, hashlib, json, math, zipfile
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[3]
OUT = REPO/'handson/handson/sim/envs/xmls/assets/d20_selected'
ARCHIVES = {
 'microwave':'microwave_oven.zip', 'hot_dog':'cc0_-_hot_dog.zip',
 'tongs':'metal_tongs.zip', 'plate':'ceramic_plate_set.zip',
 'fork':'knife_and_fork_set_-_free_blender_model.zip',
 'mustard':'squeeze_bottle_ketchup_mustard_bbq.zip',
 'mug':'coffee_mug__cup_game-ready.zip', 'tray':'serving_tray_model__realistic.zip',
}
SCALE_OVEN = 0.208425864577
TIMER_MINUTES = [0,1,2,3,4,5,6,7,8,10,15,20,25,30,35]
TIMER_DEGREES = [0,22,43,65,86,108,130,151,173,197,222,244,265,286,307]


def transform(mesh, matrix, offset=(0,0,0)):
    m=mesh.copy(); t=np.eye(4);t[:3,:3]=matrix;t[:3,3]=offset
    m.apply_transform(t);return m


def components(mesh):
    welded=mesh.copy();welded.merge_vertices(merge_tex=True,merge_norm=True)
    groups=trimesh.graph.connected_components(welded.face_adjacency,
        nodes=np.arange(len(welded.faces)),min_len=1)
    return sorted(groups,key=len,reverse=True)


def part(mesh, faces):
    return mesh.submesh([np.asarray(faces)],append=True,repair=False)


def clip_mesh(mesh, normal, origin, **_kwargs):
    """Half-space clip with bounded edge interpolation on near-coplanar faces.

    The bundled trimesh plane slicer extrapolates some nearly coplanar plate
    triangles below their original support plane. Clamp segment parameters;
    every generated point is then a convex combination of source vertices.
    """
    triangles=mesh.triangles;normal=np.asarray(normal);origin=np.asarray(origin)
    distances=(triangles-origin)@normal
    wholly=np.all(distances>=0,axis=1);crossing=(~wholly)&np.any(distances>=0,axis=1)
    output=list(triangles[wholly])
    for tri,dist in zip(triangles[crossing],distances[crossing]):
        polygon=[]
        for i in range(3):
            j=(i+1)%3;inside=dist[i]>=0;following=dist[j]>=0
            if inside:polygon.append(tri[i])
            if inside != following:
                t=float(np.clip(dist[i]/(dist[i]-dist[j]),0,1))
                polygon.append(tri[i]+t*(tri[j]-tri[i]))
        for i in range(1,len(polygon)-1):
            output.append(np.array([polygon[0],polygon[i],polygon[i+1]]))
    if not output:return trimesh.Trimesh()
    vertices=np.asarray(output).reshape(-1,3)
    return trimesh.Trimesh(vertices,np.arange(len(vertices)).reshape(-1,3),process=False)


def base_center(mesh):
    offset=-mesh.bounds.mean(axis=0);offset[2]=-mesh.bounds[0,2]
    return transform(mesh,np.eye(3),offset)


def compact_source_hull(mesh, max_vertices=60, tolerance=.0002):
    """GPU-sized inscribed hull; retain support planes and bound surface error."""
    from scipy.spatial import ConvexHull
    points=mesh.convex_hull.vertices
    selected=set()
    for axis in range(3):
        selected.update([int(np.argmin(points[:,axis])),int(np.argmax(points[:,axis]))])
    # Preserve planar support footprints before spending the vertex budget
    # on curved sides. Otherwise a decimated bottle can balance on one low
    # vertex and rock, despite its source having a broad flat base.
    for axis in range(3):
        other=[i for i in range(3) if i != axis]
        for edge in (points[:,axis].min(), points[:,axis].max()):
            on_plane=np.flatnonzero(np.abs(points[:,axis]-edge)<1e-7)
            if len(on_plane)<3:continue
            for angle in np.linspace(0,2*np.pi,8,endpoint=False):
                direction=np.array([np.cos(angle),np.sin(angle)])
                selected.add(int(on_plane[np.argmax(points[on_plane][:,other]@direction)]))
    for _ in range(max_vertices):
        hull=ConvexHull(points[sorted(selected)])
        error=np.max(points@hull.equations[:,:3].T+hull.equations[:,3],axis=1)
        if float(error.max())<1e-7 or len(selected)>=max_vertices:break
        selected.add(int(np.argmax(error)))
    if float(error.max())>=tolerance:
        axis=int(np.argmax(mesh.extents));point=mesh.bounds.mean(0);normal=np.eye(3)[axis]
        return (compact_source_hull(clip_mesh(mesh,normal,point).convex_hull,max_vertices,tolerance)
                +compact_source_hull(clip_mesh(mesh,-normal,point).convex_hull,max_vertices,tolerance))
    return [trimesh.convex.convex_hull(points[sorted(selected)])]


def convexes(mesh, key):
    """Deterministic source-derived collision, cached by exact input geometry."""
    import coacd
    coacd.set_log_level('warn')
    m=mesh.copy();m.merge_vertices(merge_tex=True,merge_norm=True)
    version=b'd20-mustard-v11' if key=='mustard' else b'd20-plate-v9' if key=='plate' else b'd20-collision-v9' if key in ('microwave_shell','control_panel','microwave_door','mug','tray','hot_dog','power_knob','timer_knob') else b'd20-coacd-v3'
    digest=hashlib.sha256(m.vertices.tobytes()+m.faces.tobytes()+version).hexdigest()
    cache=OUT/'collision'/key;cache.mkdir(parents=True,exist_ok=True)
    stamp=cache/'digest.txt'
    if stamp.exists() and stamp.read_text()==digest:
        return [trimesh.load(p,force='mesh',process=False) for p in sorted(cache.glob('*.obj'))]
    # Small convex knobs/solid food need no decomposition. Containers and the
    # fork/tongs are decomposed, so a hull cannot bridge their openings.
    if key == 'microwave_shell':
        groups=components(m)
        main_group=next(g for g in groups if len(g)==340)
        shell=part(m,main_group)
        # Five housing walls are clipped at measured inner-cavity surfaces.
        # Each hull is confined to that wall; no hull spans the open cavity.
        planes=[([-1,0,0],[-.83917*SCALE_OVEN,0,0]),
                ([1,0,0],[.47208*SCALE_OVEN,0,0]),
                ([0,1,0],[0,.59874*SCALE_OVEN,0]),
                ([0,0,-1],[0,0,-.46313*SCALE_OVEN]),
                ([0,0,1],[0,0,.48803*SCALE_OVEN])]
        result=[clip_mesh(shell,n,p,cap=False).convex_hull for n,p in planes]
        # Keep the four original feet, hinge fittings and glass turntable
        # supports separate so their contact surfaces match the source.
        for g in groups:
            if g is main_group or len(g)<4:continue
            c=part(m,g)
            if np.min(c.extents)>1e-5:
                result.append(c.convex_hull)
    elif key == 'tray':
        # Preserve the measured, perfectly flat wooden support surface.
        # Voxel hulls introduced sub-millimetre bumps underneath tableware.
        result=[clip_mesh(m,[0,0,-1],[0,0,.01358238]).convex_hull]
        for normal,point in (([-1,0,0],[-.16155801,0,0]),
                             ([1,0,0],[.16155801,0,0]),
                             ([0,-1,0],[0,-.26093262,0]),
                             ([0,1,0],[0,.26093262,0])):
            result.append(clip_mesh(m,normal,point).convex_hull)
    elif key == 'microwave_door':
        result=[part(m,g).convex_hull for g in components(m) if len(g)>=4 and np.min(part(m,g).extents)>1e-5]
    elif key in ('plate','mug'):
        # Radial pieces retain the actual inner surfaces of these containers.
        # Voxel preprocessing of thin ceramic walls can seal a cup or bridge
        # its foot ring; use exact clipped source vertices instead.
        result=[]
        if key=='mug':
            floor=clip_mesh(m,[0,0,-1],[0,0,.010])
            result.append(floor.convex_hull)
            wall=clip_mesh(m,[0,0,1],[0,0,.010])
            wall=clip_mesh(wall,[-1,0,0],[.0415,0,0])
        else:
            upward=[i for i,n in enumerate(m.facets_normal) if n[2]>.99]
            facet=max(upward,key=lambda i:m.facets_area[i])
            z=float(m.vertices[m.faces[m.facets[facet]].ravel(),2].max())
            result.append(clip_mesh(m,[0,0,-1],[0,0,z]).convex_hull)
            # Separate the flat food support from the raised rim. A convex
            # pie slice of the entire dish bridges its concavity with a cone.
            wall=clip_mesh(m,[0,0,1],[0,0,z+1e-6])
        sectors=16
        for i in range(sectors):
            a,b=2*math.pi*i/sectors,2*math.pi*(i+1)/sectors
            wedge=clip_mesh(wall,[-math.sin(a),math.cos(a),0],[0,0,0])
            wedge=clip_mesh(wedge,[math.sin(b),-math.cos(b),0],[0,0,0])
            result.append(wedge.convex_hull)
        if key=='mug':
            handle=clip_mesh(m,[1,0,0],[.0415,0,0])
            center=handle.bounds.mean(axis=0)
            for i in range(12):
                a,b=2*math.pi*i/12,2*math.pi*(i+1)/12
                wedge=clip_mesh(handle,[-math.sin(a),0,math.cos(a)],center)
                wedge=clip_mesh(wedge,[math.sin(b),0,-math.cos(b)],center)
                if len(wedge.faces)>3 and min(wedge.extents)>1e-6:
                    result.append(wedge.convex_hull)
    elif key == 'mustard':
        from scipy.spatial import ConvexHull
        # Keep the bottle's curved foot in one hull. Splitting through its
        # flat bottom duplicates the support manifolds; allowing PhysX to
        # simplify a tall 60-vertex hull instead destroys that flat bottom.
        foot=clip_mesh(m,[0,0,-1],[0,0,.006]).convex_hull
        selected=[]
        for z in (0.,.002,.006):
            vertices=clip_mesh(foot,[0,0,1],[0,0,z]).vertices if z else foot.vertices
            ring=vertices[np.abs(vertices[:,2]-z)<1e-7]
            for angle in np.linspace(0,2*np.pi,14,endpoint=False):
                direction=np.array([np.cos(angle),np.sin(angle)])
                selected.append(ring[np.argmax(ring[:,:2]@direction)])
        support=trimesh.convex.convex_hull(np.asarray(selected))
        hull=ConvexHull(support.vertices)
        error=float(np.max(foot.vertices@hull.equations[:,:3].T+hull.equations[:,3]))
        assert error < .001, ('mustard foot surface error',error)
        # Three 14-point rings have 42 vertices and at most 58 polygons once
        # the planar end caps are merged, below both native cooking budgets.
        result=[support,clip_mesh(m,[0,0,1],[0,0,.006]).convex_hull]
    elif key in ('power_knob','timer_knob','hot_dog','control_panel'):
        result=[m.convex_hull]
    else:
        values=coacd.run_coacd(coacd.Mesh(m.vertices,m.faces),threshold=0.018,
            max_convex_hull=48,preprocess_mode='auto',preprocess_resolution=80,
            resolution=1800,mcts_nodes=8,mcts_iterations=45,mcts_max_depth=3,
            merge=True,decimate=True,max_ch_vertex=64,seed=20)
        result=[trimesh.Trimesh(v,f,process=False) for v,f in values]
    if version in (b'd20-collision-v9',b'd20-plate-v9',b'd20-coacd-v3',b'd20-mustard-v11'):
        tolerance=.001 if key in ('hot_dog','mustard') else .0001 if key=='tray' else .0002
        try:
            if key=='mustard':
                result=[result[0],*compact_source_hull(result[1],max_vertices=32,tolerance=tolerance)]
            else:
                result=[piece for c in result for piece in compact_source_hull(c,max_vertices=32 if key.startswith('tongs') else 60,tolerance=tolerance)]
        except AssertionError as exc:
            raise AssertionError(f'{key}: {exc}') from exc
    for p in cache.glob('*.obj'):p.unlink()
    for i,c in enumerate(result):
        assert np.isfinite(c.vertices).all() and abs(c.volume)>1e-12,(key,i,c.volume)
        (cache/f'{i:03d}.obj').write_text(trimesh.exchange.obj.export_obj(c,include_texture=False))
    stamp.write_text(digest)
    return result


def filled_bottle_mass_properties(pieces):
    """Task model: 25 g container plus 225 g contents, not hidden ballast.

    The opaque source provides geometry but no measured mass or fill level.
    Use an explicit 1100 kg/m^3 fill-density assumption and integrate the
    occupied lower volume; approximate the light container by its envelope.
    """
    container=trimesh.util.concatenate(pieces)
    container_mass,fill_mass,density=.025,.225,1100.
    total_mass=container_mass+fill_mass
    target_volume=fill_mass/density
    assert 0 < target_volume < container.volume
    def contents(level):
        parts=[]
        for piece in pieces:
            clipped=clip_mesh(piece,[0,0,-1],[0,0,level])
            if len(clipped.vertices)>3 and min(clipped.extents)>1e-8:
                parts.append(clipped.convex_hull)
        return trimesh.util.concatenate(parts)
    lower=float(container.bounds[0,2])+1e-6
    upper=float(container.bounds[1,2])
    for _ in range(30):
        middle=(lower+upper)/2
        if contents(middle).volume < target_volume:lower=middle
        else:upper=middle
    level=(lower+upper)/2
    fill=contents(level)
    center=(container_mass*container.center_mass+fill_mass*fill.center_mass)/total_mass
    tensor=np.zeros((3,3))
    for part,mass in ((container,container_mass),(fill,fill_mass)):
        offset=part.center_mass-center
        tensor += part.moment_inertia*(mass/part.volume)+mass*(np.dot(offset,offset)*np.eye(3)-np.outer(offset,offset))
    return {'model':'lightweight_container_and_partial_fill',
            'container_mass_kg':container_mass,'fill_mass_kg':fill_mass,
            'fill_density_kg_m3':density,'fill_level_m':level,
            'volume_m3':float(container.volume),
            'center_of_mass_m':center.tolist(),
            'inertia_per_kg_m2':(tensor/total_mass).tolist()}


def make_panel():
    """Author a new enamel-panel texture; photo is a calibration reference."""
    dest=OUT/'textures';dest.mkdir(exist_ok=True)
    width,height=800,2560
    img=Image.new('RGB',(width,height),(221,222,211));d=ImageDraw.Draw(img)
    fontfile='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font=ImageFont.truetype(fontfile,45);small=ImageFont.truetype(fontfile,33)
    def label(x,y,t,f=small):d.text((x,y),t,font=f,fill=(49,53,48),anchor='mm')
    # Panel source spans x=-1..-.61262, y=-.6065...631407 before normalization.
    def xy(z,y):return ((-z-0.61262)/0.38738*width,(0.631407-y)/1.237912*height)
    cx,cy=xy(-.819235,.41868);r=250
    label(width/2,120,'火力',font)
    for angle,text in zip([0,45,90,135,180],['低火','解冻','中火','中高火','高火']):
        a=math.radians(angle);x=cx-r*math.cos(a);y=cy-r*math.sin(a)
        d.line((cx-(r-28)*math.cos(a),cy-(r-28)*math.sin(a),x,y),fill=(49,53,48),width=5)
        label(cx-(r+55)*math.cos(a),cy-(r+55)*math.sin(a),text)
    cx,cy=xy(-.819235,.09005)
    label(width/2,cy-320,'时间 · 分',font)
    for minutes,angle in zip(TIMER_MINUTES,TIMER_DEGREES):
        a=math.radians(angle);sx,sy=math.sin(a),-math.cos(a)
        d.line((cx+(r-27)*sx,cy+(r-27)*sy,cx+r*sx,cy+r*sy),fill=(49,53,48),width=5)
        label(cx+(r+46)*sx,cy+(r+46)*sy,'关 0' if minutes==0 else str(minutes))
    img.save(dest/'controls.png')
    # Direction marker is printed directly on each original knob face.
    knob=Image.new('RGB',(512,512),(206,208,196));kd=ImageDraw.Draw(knob)
    kd.line((256,58,256,140),fill=(47,50,44),width=15)
    knob.save(dest/'knob.png')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--downloads',type=Path,required=True)
    ap.add_argument('--only',nargs='*');args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    source={};raw={};meta={};provenance={}
    for key,filename in ARCHIVES.items():
        dest=OUT/'source'/key;dest.mkdir(parents=True,exist_ok=True)
        archive=args.downloads/filename
        with zipfile.ZipFile(archive) as z:
            for item in z.infolist():
                if not (dest/item.filename).resolve().is_relative_to(dest.resolve()):
                    raise ValueError('Unsafe archive path')
            z.extractall(dest)
        source[key]=dest;meta[key]=json.loads((dest/'scene.gltf').read_text())
        raw[key]=list(trimesh.load(dest/'scene.gltf',process=False).geometry.values())
        provenance[key]={'archive':filename,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
            'license_text':(dest/'license.txt').read_text()}
    (OUT/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
    make_panel()
    render={};materials={};positions={};notes={}
    def add(key, mesh, src, material_index=0):
        render[key]=mesh;materials[key]=(src,material_index)
    # Raw oven Y is up, X front, Z left. Root scene scale is uniform and its
    # parent rotations cancel. Normalize once into x=right,y=back,z=up.
    orient=np.array([[0,0,-1],[-1,0,0],[0,1,0]])*SCALE_OVEN
    oven=raw['microwave'][0];groups=components(oven)
    knob_groups=[g for g in groups if len(g)==96]
    assert len(knob_groups)==2
    knob_groups.sort(key=lambda g:oven.triangles_center[g,1].mean(),reverse=True)
    panel_group=next(g for g in groups if len(g)==64)
    excluded=set(np.concatenate([*knob_groups,panel_group]).tolist())
    shell=part(oven,[i for i in range(len(oven.faces)) if i not in excluded])
    # The turntable and its original support mesh belong to the fixed cavity.
    add('microwave_shell',trimesh.util.concatenate([transform(shell,orient),transform(raw['microwave'][1],orient)]),'microwave')
    panel=transform(part(oven,panel_group),orient)
    panel.visual=trimesh.visual.TextureVisuals(uv=np.column_stack([
        (panel.vertices[:,0]/SCALE_OVEN-.61262)/.38738,
        (panel.vertices[:,2]/SCALE_OVEN+.606505)/1.237912]))
    add('control_panel',panel,'generated')
    for key,g in zip(('power_knob','timer_knob'),knob_groups):
        m=transform(part(oven,g),orient);center=m.bounds.mean(axis=0);positions[key]=center.tolist()
        m=transform(m,np.eye(3),-center)
        uv=np.column_stack([m.vertices[:,0]/(.19646*SCALE_OVEN)+.5,m.vertices[:,2]/(.19646*SCALE_OVEN)+.5])
        m.visual=trimesh.visual.TextureVisuals(uv=uv)
        if key=='power_knob':
            # Joint zero points left. +axisY rotates clockwise in the front view.
            m=transform(m,trimesh.transformations.rotation_matrix(-math.pi/2,[0,1,0])[:3,:3])
        add(key,m,'generated')
    door=transform(raw['microwave'][2],orient)
    # Door panel's long principal axis gives its baked opening, independent of
    # labels/handles. The closed hinge sits at the source's left front corner.
    core=part(door,components(door)[0]);welded=door.copy();welded.merge_vertices(merge_tex=True,merge_norm=True)
    normal=welded.facets_normal[np.argmax(welded.facets_area)]
    angle=-math.atan2(abs(normal[0]),abs(normal[1]));hinge=np.array([-.96335,-.584,0.0])*SCALE_OVEN
    rot=trimesh.transformations.rotation_matrix(-angle,[0,0,1])[:3,:3]
    door=transform(door,rot,hinge-rot@hinge)
    # Seat the door just outside the measured front plane, instead of keeping
    # the DCC display pose's intersection with the cavity frame.
    seat_y=-.54896*SCALE_OVEN-.0008-door.bounds[1,1]
    # The source display pose also overlaps the right control fascia by
    # 1.9 mm. A 4 mm hinge/door translation leaves a real opening seam.
    door=transform(door,np.eye(3),[-.004,seat_y,0]);hinge+=np.array([-.004,seat_y,0])
    notes['door_seating_translation_x']=-.004
    notes['door_seating_translation_y']=float(seat_y)
    positions['door_hinge']=hinge.tolist();notes['source_door_open_degrees']=-math.degrees(angle)
    add('microwave_door',door,'microwave')
    # Food: source long axis Z; preserve proportions with one scale.
    m=raw['hot_dog'][0];scale=.18/np.ptp(m.vertices[:,2])
    # Lay the bun on its broad underside. The DCC display orientation leaves
    # the 79 mm-wide bun standing on its 54 mm side and it rolls after reset.
    m=transform(m,np.array([[0,0,1],[0,1,0],[-1,0,0]])*scale)
    m=transform(m,np.eye(3),-m.bounds.mean(axis=0));add('hot_dog',m,'hot_dog')
    # Use the selected dinner plate at 29 cm, uniformly scaled from 27 cm,
    # so the complete 18 cm hot dog fits before heating and after serving.
    m=raw['plate'][0];plate=base_center(transform(m,np.array([[1,0,0],[0,0,-1],[0,1,0]])*(.29/np.ptp(m.vertices[:,0]))))
    add('plate',plate,'plate')
    welded_plate=plate.copy();welded_plate.merge_vertices()
    upward=[i for i,n in enumerate(welded_plate.facets_normal) if n[2]>.99]
    flat=max(upward,key=lambda i:welded_plate.facets_area[i])
    flat_vertices=welded_plate.vertices[welded_plate.faces[welded_plate.facets[flat]].ravel()]
    notes['plate_food_support_local_z']=float(flat_vertices[:,2].max())
    notes['plate_inner_radius']=float(np.min(np.ptp(flat_vertices[:,:2],axis=0))/2)
    m=raw['fork'][0];s=.21/np.ptp(m.vertices[:,1]);add('fork',base_center(transform(m,np.eye(3)*s)),'fork')
    m=trimesh.util.concatenate(raw['mustard'][4:6]);s=.205/np.ptp(m.vertices[:,2]);add('mustard',base_center(transform(m,np.eye(3)*s)),'mustard',2)
    add('mug',transform(raw['mug'][1],np.eye(3)*.001),'mug',1)
    # Preserve the tray's authored nonuniform DCC proportions, then use one
    # physical scale. Its long side is task/world Y; x is the shorter side.
    m=raw['tray'][0];source_matrix=np.diag([525,337.5,12.5]);s=.60/(np.ptp(m.vertices[:,0])*525)
    m=transform(m,np.array([[0,-1,0],[1,0,0],[0,0,1]])@source_matrix*s)
    add('tray',base_center(m),'tray')
    # Split the continuous spring strip only at its bend, preserving both
    # metal arms. The moving half opens about that bend, never a second root.
    m=raw['tongs'][0];s=.30/np.ptp(m.vertices[:,1])
    m=transform(m,np.array([[0,1,0],[0,0,1],[1,0,0]])*s)
    m=transform(m,np.eye(3),[-m.bounds[:,0].mean(),0,0])
    moving=(m.triangles_center[:,1]>.001)&(m.triangles_center[:,0]>-.144)
    pivot=np.array([-.144,.001,0]);positions['tongs_hinge']=pivot.tolist()
    add('tongs_fixed',part(m,np.where(~moving)[0]),'tongs')
    add('tongs_moving',transform(part(m,np.where(moving)[0]),np.eye(3),-pivot),'tongs')
    manifest={'schema':1,'positions':positions,'notes':notes,'timer_minutes':TIMER_MINUTES,
              'timer_degrees':TIMER_DEGREES,'assets':{},'materials':{}}
    for key,m in render.items():
        visual=OUT/'visual'/f'{key}.obj';visual.parent.mkdir(exist_ok=True)
        visual.write_text(trimesh.exchange.obj.export_obj(m,include_normals=True,include_color=False,include_texture=True,write_texture=False))
        src,idx=materials[key]
        if src=='generated':
            mat={'baseColor':str(Path('textures')/('controls.png' if key=='control_panel' else 'knob.png')),'metallicFactor':0,'roughnessFactor':.32}
        else:
            material=meta[src]['materials'][idx];pbr=material.get('pbrMetallicRoughness',{})
            mat={'baseColorFactor':pbr.get('baseColorFactor',[1,1,1,1]),'metallicFactor':pbr.get('metallicFactor',1),
                 'roughnessFactor':pbr.get('roughnessFactor',1)}
            for field,info in [('baseColor',pbr.get('baseColorTexture')),('metallicRoughness',pbr.get('metallicRoughnessTexture')),('normal',material.get('normalTexture'))]:
                if info:
                    image=meta[src]['images'][meta[src]['textures'][info['index']]['source']]['uri']
                    mat[field]=str(Path('source')/src/image)
            if src=='tongs':mat.update(metallicFactor=1,roughnessFactor=.27)
            if src=='microwave':mat['baseColorFactor']=[1,1,1,1]
        manifest['materials'][key]=mat
        entry={'visual':str(visual.relative_to(OUT)),'bounds':m.bounds.tolist(),
               'vertices':len(m.vertices),'triangles':len(m.faces),'collision':[]}
        if args.only is None or key in args.only:
            hulls=convexes(m,key)
            entry['collision']=[str(Path('collision')/key/f'{i:03d}.obj') for i in range(len(hulls))]
            entry['collision_bounds']=[h.bounds.tolist() for h in hulls]
            if key in ('mustard','mug'):
                # These parts meet at clipping planes without overlapping.
                # Measure the exact exported bytes used by the simulator,
                # rather than treating a bottle or handled mug as a box.
                pieces=[trimesh.load(OUT/p,force='mesh') for p in entry['collision']]
                solid=trimesh.util.concatenate(pieces)
                entry['mass_properties']=(filled_bottle_mass_properties(pieces) if key=='mustard' else {
                    'model':'uniform_density_nonoverlapping_convex_parts',
                    'volume_m3':float(solid.volume),
                    'center_of_mass_m':solid.center_mass.tolist(),
                    'inertia_per_kg_m2':(solid.moment_inertia/solid.volume).tolist()})
            print(key,entry['bounds'],'hulls',len(hulls),flush=True)
        manifest['assets'][key]=entry
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print('Wrote',OUT/'manifest.json',flush=True)


if __name__=='__main__':main()
