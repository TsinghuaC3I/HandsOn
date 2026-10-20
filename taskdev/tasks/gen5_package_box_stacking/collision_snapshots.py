"""Immutable collision audit snapshots from authored USD and live GPU tensors."""
from __future__ import annotations
import numpy as np


def capture(env,phase):
    from pxr import Usd,UsdPhysics,UsdGeom
    from handson.sim.isaaclab.abc_collection_env import _pose_vector_to_matrix
    stage=env._stage; cache=UsdGeom.XformCache()
    predicate=Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
    bodies={}
    inventory=[]
    for prim in Usd.PrimRange(stage.GetPseudoRoot(),predicate):
        if not prim.IsActive():continue
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            path=str(prim.GetPath())
            bodies[path]={'matrix':np.asarray(cache.GetLocalToWorldTransform(prim)).T.tolist(),
                          'source':'authored_usd','kinematic':bool(UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get())}
        if not prim.HasAPI(UsdPhysics.CollisionAPI) or UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:continue
        body=prim
        while body and not body.IsPseudoRoot() and not body.HasAPI(UsdPhysics.RigidBodyAPI):body=body.GetParent()
        inventory.append({'path':str(prim.GetPath()),'owner':str(body.GetPath()) if body and not body.IsPseudoRoot() else None,
            'visibility':str(UsdGeom.Imageable(prim).ComputeVisibility()),'type':str(prim.GetTypeName())})
    for side,robot in getattr(env,'_robots',{}).items():
        if not robot.is_initialized:continue
        state=robot.data.body_state_w[0].detach().cpu().numpy()
        for i,name in enumerate(robot.body_names):
            matches=[p for p in bodies if f'/{side.capitalize()}Robot/' in p and p.endswith('/'+name)]
            if len(matches)!=1:raise RuntimeError(f'Cannot bind physical robot body {side}/{name}: {matches}')
            bodies[matches[0]].update(matrix=_pose_vector_to_matrix(state[i,:7]).tolist(),source='live_articulation_tensor')
    for name in getattr(env,'_task_body_views',{}):
        pose=env._physics_body_pose(name)
        if pose is None:raise RuntimeError(f'Missing physical pose for {name}')
        prim=env._find_rigid_body_prim(name); path=str(prim.GetPath())
        if path not in bodies:raise RuntimeError(f'Missing composed rigid body {path}')
        bodies[path].update(matrix=_pose_vector_to_matrix(pose).tolist(),source='live_rigid_body_tensor')
    return {'phase':phase,'inventory':inventory,'bodies':bodies,'simulation_time':float(env._sim.current_time),
        'scope':'Snapshot of final composed collision geometry plus actual GPU world transforms; independent PhysX query results are produced separately.'}
