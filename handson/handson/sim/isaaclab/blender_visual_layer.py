"""Attach a hash-checked Blender visual bundle to existing articulated links."""
from pathlib import Path
import hashlib
import json


def physics_signature(stage):
    """Fingerprint physics attributes/relationships, collision geometry and schema."""
    from pxr import Usd, UsdPhysics
    rows=[]
    for prim in stage.Traverse():
        schemas=[s for s in prim.GetAppliedSchemas() if 'Physics' in s or 'Physx' in s]
        attrs={a.GetName():str(a.Get()) for a in prim.GetAttributes()
               if a.GetName().startswith(('physics:', 'physx'))}
        rels={r.GetName():[str(p) for p in r.GetTargets()] for r in prim.GetRelationships()
              if r.GetName().startswith(('physics:', 'physx')) or r.GetName() == 'material:binding:physics'}
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            attrs.update({a.GetName():str(a.Get()) for a in prim.GetAttributes()
                          if a.GetName() in ('points','faceVertexCounts','faceVertexIndices','radius','size','height','axis') or a.GetName().startswith('xformOp')})
        if schemas or attrs or rels: rows.append([str(prim.GetPath()),schemas,attrs,rels])
    return hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()


def apply_blender_visual_bundle(stage, bundle_path, robot_root_paths):
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils
    bundle=Path(bundle_path).resolve()
    manifest=json.loads((bundle/'manifest.json').read_text())
    if manifest.get('hardware')!='rokae_ar5_5_08_wuji1' or not manifest.get('visual_only'):
        raise ValueError('Expected audited Wuji1 visual-only bundle')
    if manifest.get('meters_per_unit')!=1 or manifest.get('up_axis')!='Z':
        raise ValueError('Unexpected visual asset units/axis')
    verified = set()
    for dependency in [*manifest['links'], *manifest.get('dependencies', [])]:
        asset = (bundle / dependency['asset']).resolve()
        if bundle not in asset.parents:
            raise ValueError('Asset escapes bundle: ' + str(asset))
        if hashlib.sha256(asset.read_bytes()).hexdigest() != dependency['sha256']:
            raise ValueError('Asset hash mismatch: ' + str(asset))
        verified.add(asset)
    found={}
    wanted={i['link_name'] for i in manifest['links']}
    for root_path in robot_root_paths:
        root=stage.GetPrimAtPath(root_path)
        if not root: raise ValueError(f'Missing robot: {root_path}')
        for prim in Usd.PrimRange(root):
            if prim.GetName() not in wanted or not prim.HasAPI(UsdPhysics.RigidBodyAPI): continue
            if prim.GetName() in found: raise ValueError(f'Duplicate link name: {prim.GetName()}')
            # Mesh names can duplicate across links; index only non-proxy link prims.
            if prim.IsA(UsdGeom.Xform):found[prim.GetName()]=prim
    selected=[]
    for side in ('left', 'right'):
        full={n for n in wanted if n.startswith(side+'_') or n.endswith('_'+side)}
        hand={n for n in full if n == side+'_palm_link' or n.startswith(side+'_finger')}
        present=set(found) & full
        if not present:
            continue
        if present not in (full, hand):
            raise ValueError('Incomplete robot visual target set: '+side+' missing '+str(sorted(full-present)))
        selected.extend(item for item in manifest['links'] if item['link_name'] in present)
    if not selected:
        raise ValueError('No supported robot links found')
    changes=[]
    # Fail before any mutation on missing targets or corrupted dependencies.
    for item in selected:
        link=found.get(item['link_name'])
        asset=bundle/item['asset']
        if link is None: raise ValueError('No runtime link for '+item['link_name'])
        if hashlib.sha256(asset.read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError('Asset hash mismatch: '+str(asset))
        layers, dependencies, unresolved = UsdUtils.ComputeAllDependencies(str(asset))
        if unresolved:
            raise ValueError('Unresolved dependencies: ' + str(unresolved))
        resolved = {Path(layer.realPath).resolve() for layer in layers}
        resolved.update(Path(path).resolve() for path in dependencies)
        if not resolved.issubset(verified):
            raise ValueError('Unverified asset dependencies: ' + str(resolved - verified))
        visual_stage=Usd.Stage.Open(str(asset))
        if not visual_stage or not visual_stage.GetPrimAtPath(item['prim']):
            raise ValueError('Missing exported root: '+str(asset))
        for p in visual_stage.Traverse():
            if any('Physics' in s or 'Physx' in s for s in p.GetAppliedSchemas()):
                raise ValueError('Visual bundle contains physics schema')
        changes.append((link,asset,item))
    before=physics_signature(stage)
    hidden=[]; attached=[]
    for link,asset,item in changes:
        visuals_path=link.GetPath().AppendChild('visuals')
        visuals_prim=stage.GetPrimAtPath(visuals_path)
        if not visuals_prim:
            visuals_prim=UsdGeom.Xform.Define(stage,visuals_path).GetPrim()
        if visuals_prim.IsInstance():
            visuals_prim.SetInstanceable(False)
            visuals_prim=stage.GetPrimAtPath(visuals_path)
        # The importer binds a single material strongly on the visual group.
        # Block that binding so each exported part keeps its own PBR material.
        # Unbind only render purposes; preserve any physics-purpose binding.
        binding = UsdShade.MaterialBindingAPI(visuals_prim)
        for purpose in (UsdShade.Tokens.allPurpose, UsdShade.Tokens.preview, UsdShade.Tokens.full):
            rel = binding.GetDirectBindingRel(purpose)
            if rel:
                rel.SetTargets([])
        for child in visuals_prim.GetChildren():
            if child.GetName()!='BlenderVisuals' and UsdGeom.Imageable(child):
                UsdGeom.Imageable(child).MakeInvisible();hidden.append(str(child.GetPath()))
        accent=link.GetChild('ReferenceAppearance')
        if accent:
            UsdGeom.Imageable(accent).MakeInvisible();hidden.append(str(accent.GetPath()))
        path=visuals_path.AppendChild('BlenderVisuals')
        visual=UsdGeom.Xform.Define(stage,path)
        visual.GetPrim().GetReferences().SetReferences([Sdf.Reference(str(asset), item['prim'])])
        # Assets are in link coordinates; cancel any importer-owned visual offset.
        xf=UsdGeom.Xformable(visuals_prim)
        if xf:
            visual.MakeMatrixXform().Set(xf.GetLocalTransformation().GetInverse())
        # USD instancing shares render data; rigid links and colliders stay live.
        visual.GetPrim().SetInstanceable(True)
        for part in Usd.PrimRange(visual.GetPrim(), Usd.TraverseInstanceProxies()):
            if part.IsA(UsdGeom.Mesh):
                material, _ = UsdShade.MaterialBindingAPI(part).ComputeBoundMaterial()
                if not material or not material.GetPath().HasPrefix(path):
                    raise RuntimeError('Exported material overridden: ' + str(part.GetPath()))
        attached.append(str(path))
    after=physics_signature(stage)
    if before!=after: raise RuntimeError('Physics contract changed while attaching visuals')
    return {'profile':manifest['profile'],'hardware':manifest['hardware'],
            'links':len(attached),'link_names':[i['link_name'] for i in selected],'attached':attached,'hidden':hidden,
            'physics_before':before,'physics_after':after,
            'bundle_manifest_sha256':hashlib.sha256((bundle/'manifest.json').read_bytes()).hexdigest()}
