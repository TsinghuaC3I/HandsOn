"""Original Z laboratory visuals and provisional native PhysX composition.

The shared ABC lifecycle owns robot control, reset, recording and replay.
"""
from pathlib import Path

ASSET_ROOT = Path(__file__).resolve().parents[1] / 'envs/usd/z_realistic'
TASK_ASSETS = {'z01_pipette': 'z01_pipette.usdc', 'z02_bottle_cap': 'z02_bottle_cap.usdc'}


def compose_visual_scene(stage, task_name: str, *, table_top_z: float, root_path: str):
    """Place the authored tabletop arrangement above the shared table."""
    import json
    from pxr import Gf, UsdGeom
    asset = ASSET_ROOT / TASK_ASSETS[task_name]
    parameters = json.loads((ASSET_ROOT / 'parameters.json').read_text())
    if stage.GetPrimAtPath(root_path):
        raise ValueError(f'Z scene already exists: {root_path}')
    root = UsdGeom.Xform.Define(stage, root_path)
    root.GetPrim().GetReferences().AddReference(str(asset))
    root.AddTranslateOp(opSuffix='table_height').Set(
        Gf.Vec3d(float(parameters.get('workspace_offset_x_m',0)), 0, float(table_top_z) - float(parameters['table_height']))
    )
    return root.GetPrim()


def compose_physical_scene(stage, task_name: str, *, table_top_z: float, root_path: str):
    """Compose provisional rigid mechanics for the native ABC runtime.

    Fixture meshes are static triangle surfaces so their openings stay open.
    Dynamic shapes are small native cylinders/boxes inside the visual shell.
    All helpers have explicit invisible visibility, independent of purpose.
    """
    import math
    import json
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, PhysxSchema
    params = json.loads((ASSET_ROOT / 'parameters.json').read_text())
    root = compose_visual_scene(stage, task_name, table_top_z=table_top_z, root_path=root_path)
    dynamic = {}
    colliders = []

    def rigid(name, mass):
        prim = stage.GetPrimAtPath(root_path + '/' + name)
        if not prim:
            raise ValueError(f'Missing authored body {name}')
        UsdPhysics.RigidBodyAPI.Apply(prim)
        PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
        rb=PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        rb.CreateSolverPositionIterationCountAttr(16)
        rb.CreateSolverVelocityIterationCountAttr(4)
        rb.CreateLinearDampingAttr(.2)
        rb.CreateAngularDampingAttr(.5)
        rb.CreateMaxDepenetrationVelocityAttr(.1)
        rb.CreateSleepThresholdAttr(0.)
        rb.CreateStabilizationThresholdAttr(.001)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)
        dynamic[name] = str(prim.GetPath())
        return prim

    def shape(body, name, kind, position, size, rotation=None):
        path = str(body.GetPath()) + '/Collision_' + name
        geom = UsdGeom.Cylinder.Define(stage, path) if kind == 'cylinder' else UsdGeom.Cube.Define(stage, path)
        if kind == 'cylinder':
            geom.CreateRadiusAttr(size[0]); geom.CreateHeightAttr(size[1]); geom.CreateAxisAttr('Z')
        else:
            geom.CreateSizeAttr(1)
        geom.AddTranslateOp().Set(Gf.Vec3d(*position))
        if rotation is not None: geom.AddRotateZOp().Set(rotation)
        if kind == 'box': geom.AddScaleOp().Set(Gf.Vec3f(*size))
        geom.MakeInvisible()
        UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
        contact=PhysxSchema.PhysxCollisionAPI.Apply(geom.GetPrim())
        contact.CreateContactOffsetAttr(.0005);contact.CreateRestOffsetAttr(0.)
        colliders.append(path)
        return geom

    def ring(body, name, radius, width, height, z, segments=24):
        # Tangential boxes preserve the neck/cap opening. Boxes stay within
        # the shell's radial envelope rather than adding a grasp sleeve.
        tangential = 2 * radius * math.tan(math.pi / segments)
        for i in range(segments):
            a = 2 * math.pi * i / segments
            shape(body, f'{name}_{i}', 'box', (radius*math.cos(a),radius*math.sin(a),z),
                  (width,tangential,height),i*360/segments)

    if task_name == 'z02_bottle_cap':
        bottle = rigid('bottle_body', params['bottle_mass_kg'])
        cap = rigid('bottle_cap', params['cap_mass_kg'])
        shape(bottle,'body','cylinder',(0,0,.057),(.030,.106))
        shape(bottle,'flat_base','box',(0,0,.002),(.034,.034,.004))
        shape(bottle,'shoulder','cylinder',(0,0,.115),(.026,.010))
        ring(bottle,'neck',.017,.002,.024,.133)
        ring(cap,'skirt',.0215,.0025,.019,.0095)
        shape(cap,'roof','cylinder',(0,0,.022),(.021,.002))
        shape(cap,'roof_support_x','box',(0,0,.021),(.036,.014,.002))
        shape(cap,'roof_support_y','box',(0,0,.021),(.014,.036,.002))
    else:
        body=rigid('pipette_body', params['pipette_mass_kg'])
        plunger=rigid('pipette_plunger', params['plunger_mass_kg'])
        UsdPhysics.ArticulationRootAPI.Apply(root)
        articulation=PhysxSchema.PhysxArticulationAPI.Apply(root)
        articulation.CreateEnabledSelfCollisionsAttr(False)
        articulation.CreateSleepThresholdAttr(0.)
        articulation.CreateStabilizationThresholdAttr(.001)
        articulation.CreateSolverPositionIterationCountAttr(16)
        articulation.CreateSolverVelocityIterationCountAttr(4)
        # The installed tip shares the pipette rigid body. A separate tiny
        # constrained body adds numerical stiffness without a task need.
        old_tip=stage.GetPrimAtPath(root_path+'/Disposable_hollow_tip')
        tip_visual=UsdGeom.Xform.Define(stage,str(body.GetPath())+'/InstalledTip')
        tip_visual.GetPrim().GetReferences().AddReference(str(ASSET_ROOT/TASK_ASSETS[task_name]),'/root/Disposable_hollow_tip')
        tip_visual.ClearXformOpOrder()
        old_tip.SetActive(False)
        shape(body,'barrel','cylinder',(0,0,.134),(.010,.076))
        shape(body,'collar','cylinder',(0,0,.096),(.010,.012))
        shape(body,'collar_support','box',(0,0,.088),(.016,.012,.004))
        shape(body,'shaft','cylinder',(0,0,.040),(.0025,.080))
        shape(body,'finger_rest','box',(0,.013,.169),(.017,.024,.004))
        shape(plunger,'button','cylinder',(0,0,.199),(.0065,.006))
        shape(body,'tip','cylinder',(0,0,-.018),(.0007,.038))
        joint=UsdPhysics.PrismaticJoint.Define(stage,root_path+'/PlungerJoint')
        joint.CreateBody0Rel().SetTargets([body.GetPath()]);joint.CreateBody1Rel().SetTargets([plunger.GetPath()])
        joint.CreateAxisAttr('Z');joint.CreateLowerLimitAttr(-params['plunger_travel']);joint.CreateUpperLimitAttr(0)
        drive=UsdPhysics.DriveAPI.Apply(joint.GetPrim(),'linear');drive.CreateTypeAttr('force');drive.CreateStiffnessAttr(params['plunger_stiffness_n_m']);drive.CreateDampingAttr(params['plunger_damping_n_s_m']);drive.CreateTargetPositionAttr(params['plunger_spring_preload_m']);drive.CreateMaxForceAttr(2)
    # Fixed fixtures retain their real mesh cavities. Dynamic mesh surfaces
    # remain visuals; only the invisible native shapes above collide.
    static=[]
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh): continue
        path=str(prim.GetPath())
        if any(path.startswith(p+'/') for p in dynamic.values()): continue
        if any(word in path for word in ('liquid','Label','label','Tray_ID')):continue
        UsdPhysics.CollisionAPI.Apply(prim)
        contact=PhysxSchema.PhysxCollisionAPI.Apply(prim)
        contact.CreateContactOffsetAttr(.0005);contact.CreateRestOffsetAttr(0.)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('none')
        static.append(path)
    return {'root':str(root.GetPath()),'dynamic':dynamic,'static_colliders':static,'hidden_colliders':colliders,
            'units':'m','parameter_status':'provisional','physics_hz':120}
