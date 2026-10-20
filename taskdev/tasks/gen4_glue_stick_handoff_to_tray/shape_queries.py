"""Actual final PhysX shape query shared by GEN4 diagnostic probes."""
from handson.tasks.eval_gen04_pass_glue_stick import contract as c


def query_final_shapes(environment, phase, *, allowed_hand_contacts=()):
    import omni.physx
    from pxr import Usd, UsdGeom, UsdPhysics, PhysicsSchemaTools
    stage = environment._stage
    intended_hands={str(path):side for side in allowed_hand_contacts for path in environment._d_hand_body_paths(side)}
    inventory = {}
    predicate = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
    excluded = set(); fixed_edges=[]; fixed_parent={}
    def fixed_root(path):
        fixed_parent.setdefault(path,path)
        while fixed_parent[path]!=path: path=fixed_parent[path]
        return path
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), predicate):
        if not prim.IsActive(): continue
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            if prim.IsA(UsdPhysics.FixedJoint) and joint.GetJointEnabledAttr().Get() is not False:
                for a in joint.GetBody0Rel().GetTargets():
                    for b in joint.GetBody1Rel().GetTargets():
                        a,b=str(a),str(b)
                        fixed_edges.append({'joint':str(prim.GetPath()),'bodies':[a,b]})
                        fixed_parent[fixed_root(a)]=fixed_root(b)
            if joint.GetCollisionEnabledAttr().Get() is not True:
                for a in joint.GetBody0Rel().GetTargets():
                    for b in joint.GetBody1Rel().GetTargets(): excluded.add(tuple(sorted((str(a),str(b)))))
        if prim.HasAPI(UsdPhysics.FilteredPairsAPI):
            for target in UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel().GetTargets():
                excluded.add(tuple(sorted((str(prim.GetPath()),str(target)))))
        if not prim.HasAPI(UsdPhysics.CollisionAPI): continue
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False: continue
        owner = prim
        while owner and not owner.IsPseudoRoot() and not owner.HasAPI(UsdPhysics.RigidBodyAPI): owner = owner.GetParent()
        inventory[str(prim.GetPath())] = str(owner.GetPath()) if owner and not owner.IsPseudoRoot() else str(prim.GetPath())
    assert all(any('/'+n+'/' in p for p in inventory) for n in (c.GLUE_BODY,c.TRAY_BODY,*c.DISTRACTORS))
    assert any('/HandsOnRobots/' in p for p in inventory), 'Complete robot collision inventory is absent'
    query = omni.physx.get_physx_scene_query_interface()
    pairs = {}; errors = []; hit_count = 0
    query_inputs = []; self_hit_shapes = set()
    def permitted(a,b):
        x,y=inventory.get(a,a),inventory.get(b,b)
        if x==y: return 'same physical owner'
        if '/HandsOnRobots/' in x and '/HandsOnRobots/' in y and fixed_root(x)==fixed_root(y):
            return 'internal fixed robot mounting assembly'
        if tuple(sorted((x,y))) in excluded or tuple(sorted((a,b))) in excluded: return 'authored collision exclusion'
        for target,hand in ((x,y),(y,x)):
            if target.rsplit('/',1)[-1]==c.GLUE_BODY and hand in intended_hands:
                return 'intended '+intended_hands[hand]+'-hand contact with glue'
        names={x.rsplit('/',1)[-1],y.rsplit('/',1)[-1]}
        table=any('table' in p.lower() and '/HandsOnRobots/' not in p for p in (a,b))
        if table and names.intersection((c.GLUE_BODY,c.TRAY_BODY,*c.DISTRACTORS)): return 'declared table support'
        return None
    for path in inventory:
        # Infinite planes are supported query targets, not finite input
        # gprims. Query all other shapes against them instead.
        if stage.GetPrimAtPath(path).GetTypeName() == 'Plane': continue
        query_inputs.append(path)
        encoded=PhysicsSchemaTools.encodeSdfPath(stage.GetPrimAtPath(path).GetPath())
        def on_hit(hit):
            nonlocal hit_count
            hit_count += 1
            other=str(hit.collision)
            if other==path:
                self_hit_shapes.add(path)
                return True
            pair=tuple(sorted((path,other)))
            pairs[pair]={'shapes':list(pair),'owners':[inventory.get(x,x) for x in pair], 'allow_reason':permitted(*pair)}
            return True
        try: query.overlap_shape(encoded[0],encoded[1],on_hit,False)
        except Exception as exc: errors.append({'source':path,'error':str(exc)})
    rows=list(pairs.values()); bad=[p for p in rows if p['allow_reason'] is None]
    report={'phase':phase,'physics_time_s':float(environment._sim.current_time),
        'inventory':inventory,'shape_count':len(inventory),'callback_count':hit_count,'fixed_joint_edges':fixed_edges,
        'pairs':rows,'query_errors':errors,'unapproved_pairs':bad}
    # Every finite input must find its own shape as a known positive. A few
    # callbacks elsewhere do not calibrate all the individual overlap queries.
    report['query_input_shape_count'] = len(query_inputs)
    report['self_hit_shape_count'] = len(self_hit_shapes)
    report['missing_self_hit_shapes'] = sorted(set(query_inputs) - self_hit_shapes)
    report['free_body_properties']={}
    for name in (c.GLUE_BODY,*c.DISTRACTORS):
        body=environment._find_rigid_body_prim(name)
        report['free_body_properties'][name]={
            'articulation_root':body.HasAPI(UsdPhysics.ArticulationRootAPI),
            **{key:body.GetAttribute('physxRigidBody:'+key).Get() for key in
                ('sleepThreshold','linearDamping','angularDamping')},
        }
    if phase in {'post_reset','post_constructor_reset'}:
        import numpy as np
        cooking=omni.physx.get_physx_cooking_interface()
        cache=UsdGeom.XformCache(Usd.TimeCode.Default())
        report['cooked_meshes']={}
        for path,owner_path in inventory.items():
            if '/HandsOnRobots/' in path:continue
            prim=stage.GetPrimAtPath(path)
            if not prim.IsA(UsdGeom.Mesh):continue
            owner=stage.GetPrimAtPath(owner_path)
            matrix=cache.GetLocalToWorldTransform(prim)*cache.GetLocalToWorldTransform(owner).GetInverse()
            from pxr import UsdUtils
            from omni.physx.bindings._physx import PhysxCollisionRepresentationResult
            meshes=[];cooking_result=[]
            def on_cooked(status,convexes):
                cooking_result.append(str(status))
                if status != PhysxCollisionRepresentationResult.RESULT_VALID:return
                for data in convexes:
                    meshes.append({'vertices':[list(v) for v in data.vertices],
                        'num_vertices':len(data.vertices),'indices':list(data.indices),
                        'polygons':[{'plane':list(p.plane),'num_vertices':p.num_vertices,'index_base':p.index_base} for p in data.polygons]})
            cooking.request_convex_collision_representation(
                stage_id=UsdUtils.StageCache.Get().GetId(stage).ToLongInt(),
                collision_prim_id=PhysicsSchemaTools.sdfPathToInt(path),
                run_asynchronously=False,on_result=on_cooked)
            if not meshes:raise RuntimeError(f'Cooked geometry unavailable: {path}: {cooking_result}')
            mass=UsdPhysics.MassAPI(owner)
            report['cooked_meshes'][path]={'owner':owner_path,
                'authored_points':np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get()).tolist(),
                'mesh_to_owner':np.asarray(matrix).tolist(), 'cooked_hulls':meshes,
                'mass_kg':mass.GetMassAttr().Get(),
                'center_of_mass':list(mass.GetCenterOfMassAttr().Get()),
                'diagonal_inertia':list(mass.GetDiagonalInertiaAttr().Get()),
                'rest_offset':prim.GetAttribute('physxCollision:restOffset').Get()}
    return report
