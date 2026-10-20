"""Z-series construction on the existing native ABC robot/physics lifecycle.

The task objects, event history and native joints are recorded explicitly.
Dimensions and operating thresholds remain provisional pending real measurements.
"""
from .abc_collection_env import HandsOnABCIsaacLabEnv, _TABLE_TOP_Z, _TASK_ROOT
from .z_realistic_scene import compose_physical_scene, TASK_ASSETS


class ZRealisticIsaacLabEnv(HandsOnABCIsaacLabEnv):
    def __init__(self, task_name, **kwargs):
        if task_name not in TASK_ASSETS:
            raise ValueError(task_name)
        import json
        from .z_realistic_scene import ASSET_ROOT
        from .z_task_progress import ZTaskProgress
        self._z_parameters=json.loads((ASSET_ROOT/'parameters.json').read_text())
        self._z_progress=ZTaskProgress()
        self._z_contacts={}
        self._z_resetting=False
        self._z_fixture_positions={}
        self._z_composition = {}
        self._z_initial_state = {}
        self._z_authored_positions = {}
        # The first integration uses the authored fixture layout. No source
        # parameter or randomization distribution has been provided yet.
        kwargs['randomize'] = False
        kwargs.setdefault('physics_dt', 1/120)
        kwargs.setdefault('control_dt', 1/30)
        kwargs.setdefault('hz', 30)
        kwargs.setdefault('physics_backend', 'physx')
        super().__init__(task_name, **kwargs)

    def _spawn_task_specific_assets(self, sim_utils):
        self._z_composition = compose_physical_scene(
            self._stage, self.task_name,
            table_top_z=_TABLE_TOP_Z, root_path=_TASK_ROOT+'/Z',
        )

        from pxr import UsdGeom
        cache = UsdGeom.XformCache()
        self._z_fixture_positions={p.GetName():tuple(cache.GetLocalToWorldTransform(p).ExtractTranslation())
            for p in self._stage.GetPrimAtPath(_TASK_ROOT+'/Z').GetChildren() if p.IsA(UsdGeom.Xform)}
        self._z_authored_positions = {
            name: tuple(cache.GetLocalToWorldTransform(self._stage.GetPrimAtPath(path)).ExtractTranslation())
            for name, path in self._z_composition['dynamic'].items()
        }

    def _configure_task_reset_stability(self, sim_utils):
        # IsaacLab writes cfg.init_state after its initial PhysX reset. Author
        # the matching native joint state and drive targets before that reset
        # so the first GPU state also has the canonical arm/hand home.
        import math
        import re
        from pxr import Usd, UsdPhysics, PhysxSchema
        count = 0
        for side, robot in self._robots.items():
            root = self._stage.GetPrimAtPath(robot.cfg.prim_path)
            for prim in Usd.PrimRange(root):
                if not prim.IsA(UsdPhysics.RevoluteJoint):
                    continue
                values = [v for pattern, v in robot.cfg.init_state.joint_pos.items()
                          if re.fullmatch(pattern, prim.GetName())]
                if len(values) != 1:
                    raise ValueError(f'No unique canonical home for {prim.GetPath()}')
                degrees = math.degrees(float(values[0]))
                state = PhysxSchema.JointStateAPI.Apply(prim, 'angular')
                state.CreatePositionAttr(degrees)
                state.CreateVelocityAttr(0.)
                drive = UsdPhysics.DriveAPI.Apply(prim, 'angular')
                drive.CreateTargetPositionAttr(degrees)
                drive.CreateTargetVelocityAttr(0.)
                count += 1
        self._z_authored_robot_joint_count = count
        return len(self._robots)

    def _validate_task_initial_state_before_ready(self):
        import numpy as np
        from pxr import UsdGeom
        from .priority_zero_robot_audit import initial_robot_state_report
        robot_ok, robot_state = initial_robot_state_report(self)
        errors = [] if robot_ok else ['robot initial state']
        bodies = {}
        for name in self._z_composition['dynamic']:
            pose = self._physics_body_pose(name)
            velocity = self._physics_body_velocity(name)
            if pose is None or velocity is None:
                errors.append(name + ': missing live body state')
                continue
            pose = np.asarray(pose); velocity = np.asarray(velocity)
            drift = float(np.linalg.norm(pose[:3] - self._z_authored_positions[name]))
            linear = float(np.linalg.norm(velocity[:3])); angular = float(np.linalg.norm(velocity[3:]))
            # Same initial task-body bounds as HS014; robot limits remain
            # those of initial_robot_state_report, which are much stricter.
            okay = bool(np.isfinite(pose).all() and np.isfinite(velocity).all()
                        and drift <= .005 and linear <= .05 and angular <= .75)
            bodies[name] = {'pose_wxyz':pose.tolist(), 'velocity':velocity.tolist(),
                            'authored_drift_m':drift, 'passed':okay}
            if not okay: errors.append(name + ': invalid initial pose or motion')
        for path in self._z_composition['hidden_colliders']:
            if UsdGeom.Imageable(self._stage.GetPrimAtPath(path)).ComputeVisibility() != 'invisible':
                errors.append(path + ': visible collision helper')
        self._z_initial_state = {'passed':not errors, 'robot':robot_state, 'bodies':bodies, 'errors':errors}
        if errors:
            raise RuntimeError('Z initial-state validation failed: ' + '; '.join(errors))
        return self._z_initial_state

    def _task_extra_pose_names(self):
        return {'pipette_body', 'pipette_plunger'} if self.task_name == 'z01_pipette' else {'bottle_body', 'bottle_cap'}

    def _task_reset_write_is_required(self, path, *, state_kind):
        return path.startswith(_TASK_ROOT+'/Z/') or path == _TASK_ROOT+'/Z'


    def _build_observation_space(self):
        import numpy as np
        from gymnasium import spaces
        space = super()._build_observation_space()
        space.spaces['state'].spaces.update({
            'z_task_state':spaces.Box(-np.inf,np.inf,shape=(12,),dtype=np.float64),
            'z_entity_state':spaces.Box(-np.inf,np.inf,shape=(2,13),dtype=np.float64),
            'native_joint_positions':spaces.Box(-np.inf,np.inf,shape=(54,),dtype=np.float64),
            'native_joint_velocities':spaces.Box(-np.inf,np.inf,shape=(54,),dtype=np.float64),
            'z_articulation_root_state':spaces.Box(-np.inf,np.inf,shape=(13,),dtype=np.float64),
            'z_plunger_state':spaces.Box(-np.inf,np.inf,shape=(2,),dtype=np.float64),
        })
        return space

    def _init_task_specific_runtime(self):
        from pxr import Usd, UsdPhysics
        self._z_contacts = {}
        view = self._robots['right']._physics_sim_view
        hands = {}
        for side,robot in self._robots.items():
            hands[side] = [str(p.GetPath()) for p in Usd.PrimRange(self._stage.GetPrimAtPath(robot.cfg.prim_path))
                           if p.HasAPI(UsdPhysics.RigidBodyAPI) and
                           (p.GetName()==f'{side}_palm_link' or p.GetName().startswith(f'{side}_finger'))]
        def contact(name,body,filters):
            if not filters: raise RuntimeError(f'No contact filters for {name}')
            sensor=view.create_rigid_contact_view(body,filter_patterns=filters,max_contact_data_count=256)
            if sensor.sensor_count < 1 or sensor.filter_count < 1:
                raise RuntimeError(f'No physical contact view for {name}')
            self._z_contacts[name]=sensor
        for name,path in self._z_composition['dynamic'].items():
            for side in ('left','right'):contact(side+'_'+name,path,hands[side])
        static=self._z_composition['static_colliders']
        if self.task_name=='z01_pipette':
            contact('support',self._z_composition['dynamic']['pipette_body'],
                    [p for p in static if 'Pipette_stand' in p or 'Pipette_holder' in p])
        else:
            contact('support',self._z_composition['dynamic']['bottle_cap'],[p for p in static if 'receiving_dish' in p])
        self._handson_post_physics_substep_callback=self._z_after_physics
        self._sync_b320_tcp_command_poses_to_physical_state()

    def _z_force(self,name):
        import numpy as np
        value=self._z_contacts[name].get_contact_force_matrix(self._physics_dt)
        if hasattr(value,'detach'):value=value.detach().cpu().numpy()
        return float(np.linalg.norm(value,axis=-1).sum())

    @staticmethod
    def _z_rotation(pose):
        import numpy as np
        w,x,y,z=pose[3:]
        return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                         [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                         [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

    def _z_after_physics(self,_index,_count):
        import numpy as np
        import math
        if self._z_resetting or not self._z_contacts:return
        p=self._z_parameters
        body_name='pipette_body' if self.task_name=='z01_pipette' else 'bottle_cap'
        pose=np.asarray(self._physics_body_pose(body_name))
        velocity=np.asarray(self._physics_body_velocity(body_name))
        left=self._z_force('left_'+body_name);right=self._z_force('right_'+body_name)
        released=left+right<.005
        quiet=np.linalg.norm(velocity[:3])<.02 and np.linalg.norm(velocity[3:])<.10
        supported=self._z_force('support')>.005
        if self.task_name=='z01_pipette':
            rot=self._z_rotation(pose)
            plunger=np.asarray(self._physics_body_pose('pipette_plunger'))
            stroke=-float((plunger[:3]-pose[:3])@rot[:,2])/p['plunger_travel']
            tip=pose[:3]+rot@np.array([0,0,-.040*p['pipette_length']/.242])
            def inside(name):
                delta=tip-np.asarray(self._z_fixture_positions[name])
                return bool(np.linalg.norm(delta[:2])<.007 and .006<delta[2]<.037)
            on_stand=supported and np.linalg.norm(pose[:3]-self._z_authored_positions['pipette_body'])<.015
            self._z_progress.pipette(stroke=stroke,source=inside('Sample_tube'),destination=inside('Sample_tube_001'),
                hand_contact=left+right>.02,on_stand=on_stand,released=released,quiet=quiet,
                dt=self._physics_dt,capacity_ul=p['pipette_capacity_ul'],hold_seconds=p['completion_hold_seconds'])
            self._z_update_liquids()
        else:
            bottle=np.asarray(self._physics_body_pose('bottle_body'));br=self._z_rotation(bottle)
            relative=br.T@self._z_rotation(pose)
            angle=math.atan2(relative[1,0],relative[0,0])
            delta=br.T@(pose[:3]-bottle[:3])
            baseline=np.asarray(self._z_authored_positions['bottle_cap'])-np.asarray(self._z_authored_positions['bottle_body'])
            lift=float(delta[2]-baseline[2])
            role=self._z_force('left_bottle_body')>.02 and right>.02
            center=np.asarray(self._z_fixture_positions['Cap_receiving_dish'])
            self._z_progress.cap(angle=angle,seated=abs(lift)<.005 and np.linalg.norm(delta[:2])<.004,
                axial_lift=lift,role_contacts=role,released=released,in_dish=supported and np.linalg.norm(pose[:2]-center[:2])<.035,
                quiet=quiet,dt=self._physics_dt,required_turn=p['cap_required_turn_rad'],lift_distance=p['cap_lift_distance_m'],
                hold_seconds=p['completion_hold_seconds'])

    def _z_update_liquids(self):
        from pxr import UsdGeom,Gf
        for name,volume in [('Sample_liquid',5630-self._z_progress.aspirated_ul-self._z_progress.dispensed_ul-self._z_progress.spilled_ul),
                            ('Sample_liquid_001',self._z_progress.dispensed_ul)]:
            prim=self._stage.GetPrimAtPath(_TASK_ROOT+'/Z/'+name)
            if not prim:continue
            imageable=UsdGeom.Imageable(prim)
            if volume<=0:imageable.MakeInvisible()
            else:imageable.MakeVisible()
            xform=UsdGeom.Xformable(prim)
            ops=[op for op in xform.GetOrderedXformOps() if op.GetOpName()=='xformOp:scale:liquid_volume']
            op=ops[0] if ops else xform.AddScaleOp(opSuffix='liquid_volume')
            op.Set(Gf.Vec3f(1,1,max(.00001,min(1,volume/5630))))

    def _reset_scene_state(self):
        from .z_task_progress import ZTaskProgress
        self._z_resetting=True
        self._z_progress=ZTaskProgress()
        try:
            super()._reset_scene_state()
            if self.task_name=='z01_pipette' and self._stage is not None:self._z_update_liquids()
        finally:self._z_resetting=False

    def _compute_success_now(self):
        return self._z_progress.succeeded

    def _compute_failure(self):
        # Ordinary contact and incomplete operations remain recoverable.
        # The shared episode time limit truncates an unfinished operation.
        return False

    def _compute_observation(self):
        import numpy as np
        observation=super()._compute_observation()
        entities=[]
        for name in self._task_extra_pose_names_ordered():
            pose=self._physics_body_pose(name) if self._robots else None
            velocity=self._physics_body_velocity(name) if self._robots else None
            entities.append(np.r_[pose if pose is not None else [0,0,0,1,0,0,0],velocity if velocity is not None else np.zeros(6)])
        joints=[]
        joint_velocities=[]
        for side in ('left','right'):
            if side in self._robots:
                values=self._robots[side].data.joint_pos[0]
                ids=list(self._arm_joint_ids[side])+list(self._hand_joint_ids[side])
                joints.extend(values[ids].detach().cpu().numpy().tolist())
                joint_velocities.extend(self._robots[side].data.joint_vel[0,ids].detach().cpu().numpy().tolist())
            else:
                joints.extend([0.]*27)
                joint_velocities.extend([0.]*27)
        plunger=np.zeros(2)
        articulation_root=np.array([0,0,0,1,0,0,0,0,0,0,0,0,0],dtype=np.float64)
        if self.task_name=='z01_pipette' and 'PlungerJoint' in self._task_dof_views:
            view,index,_=self._task_dof_views['PlungerJoint']
            transform=view.get_root_transforms()[0].detach().cpu().numpy()
            velocity=view.get_root_velocities()[0].detach().cpu().numpy()
            articulation_root=np.r_[transform[[0,1,2,6,3,4,5]],velocity]
            plunger=np.array([view.get_dof_positions()[0,index].item(),view.get_dof_velocities()[0,index].item()])
        observation['state'].update(z_task_state=np.array(self._z_progress.values(),dtype=np.float64),
            z_entity_state=np.array(entities,dtype=np.float64),native_joint_positions=np.array(joints,dtype=np.float64),
            native_joint_velocities=np.array(joint_velocities,dtype=np.float64),z_plunger_state=plunger,z_articulation_root_state=articulation_root)
        return observation

    def _task_extra_pose_names_ordered(self):
        return ('pipette_body','pipette_plunger') if self.task_name=='z01_pipette' else ('bottle_body','bottle_cap')

    def _task_info(self):
        from dataclasses import asdict
        return {'z_task_progress':asdict(self._z_progress),'z_parameters_provisional':True,
                'z_initial_state':self._z_initial_state,'z_liquid_model':'plunger_stroke_estimate'}

    def restore_recorded_initial_state(self, state_parts):
        """Restore a snapshot at replay initialization or for offline rendering.

        Live physical replay remains action-driven: the shared write guard
        forbids calling this to overwrite an ongoing physical trajectory.
        """
        import numpy as np
        from .z_task_progress import ZTaskProgress
        if getattr(self,'_trajectory_action_phase_active',False):
            raise RuntimeError('Cannot restore a Z snapshot during a live action')
        self._forbid_physics_only_task_state_write('Z recorded snapshot restoration')
        arrays = {}
        for key,shape in (('z_entity_state',(2,13)),('native_joint_positions',(54,)),
                          ('native_joint_velocities',(54,)),('z_plunger_state',(2,)),
                          ('z_task_state',(12,)),('z_articulation_root_state',(13,))):
            value=np.asarray(state_parts.get(key,[]),dtype=np.float64)
            if value.size!=int(np.prod(shape)) or not np.isfinite(value).all():
                raise ValueError(f'{key} must contain finite state with shape {shape}')
            arrays[key]=value.reshape(shape)
        poses=arrays['z_entity_state']
        if not np.allclose(np.linalg.norm(poses[:,3:7],axis=1),1.,atol=.001):
            raise ValueError('Z entity quaternions must be normalized wxyz')
        if not np.isclose(np.linalg.norm(arrays['z_articulation_root_state'][3:7]),1.,atol=.001):
            raise ValueError('Z articulation root quaternion must be normalized wxyz')
        progress=ZTaskProgress.from_values(arrays['z_task_state'])
        if self.task_name=='z01_pipette':
            q=arrays['z_plunger_state'][0]
            if not -self._z_parameters['plunger_travel']-.001<=q<=.001:
                raise ValueError('Recorded plunger position is outside its joint range')
        self._z_resetting=True
        try:
            super().restore_recorded_initial_state(state_parts)
            self._restore_recorded_robot_command_state(state_parts)
            names=self._task_extra_pose_names_ordered()
            def tensor(values, template):
                return self._torch.as_tensor(values,dtype=template.dtype,device=template.device).reshape(template.shape)
            if self.task_name=='z01_pipette':
                view,indices=self._task_articulation_views['pipette_body']
                root=arrays['z_articulation_root_state']
                view.set_root_transforms(tensor(root[[0,1,2,4,5,6,3]],view.get_root_transforms()),indices)
                view.set_root_velocities(tensor(root[7:],view.get_root_velocities()),indices)
                dof_view,index,_=self._task_dof_views['PlungerJoint']
                q=dof_view.get_dof_positions().clone();qd=dof_view.get_dof_velocities().clone()
                q[:,index]=arrays['z_plunger_state'][0];qd[:,index]=arrays['z_plunger_state'][1]
                dof_view.set_dof_positions(q,indices);dof_view.set_dof_velocities(qd,indices)
            else:
                for name,row in zip(names,poses):
                    view=self._task_body_views[name]
                    if not self._set_task_body_world_pose(view,row[:7]):
                        raise RuntimeError('Could not restore '+name)
                    indices=self._torch.arange(view.count,dtype=self._torch.long,device=self._isaac_device)
                    view.set_velocities(tensor(row[7:],view.get_velocities()),indices)
            for offset,side in ((0,'left'),(27,'right')):
                robot=self._robots[side]
                ids=list(self._arm_joint_ids[side])+list(self._hand_joint_ids[side])
                q=robot.data.joint_pos.clone();qd=robot.data.joint_vel.clone()
                q[:,ids]=tensor(arrays['native_joint_positions'][offset:offset+27],q[:,ids])
                qd[:,ids]=tensor(arrays['native_joint_velocities'][offset:offset+27],qd[:,ids])
                robot.write_joint_state_to_sim(q,qd)
                robot.set_joint_position_target(q);robot.write_data_to_sim()
                self._joint_targets[side]=q.clone()
            self._z_progress=progress
            if self.task_name=='z01_pipette':self._z_update_liquids()
            self._record_trajectory_state_write('restore_recorded_initial_state',subject=self.task_name)
            self._sim.forward()
            for robot in self._robots.values():robot.update(0.)
            self._refresh_object_poses()
            return self._compute_observation()
        finally:
            self._z_resetting=False

    def restore_recorded_visual_state(self, state_parts, *, robot_iterations=8):
        # Exact recorded native joints supersede the legacy TCP/IK visual fit.
        return self.restore_recorded_initial_state(state_parts)

    def _sync_b320_tcp_command_poses_to_physical_state(self):
        """Anchor Z Cartesian control to actual palms at every reset."""
        import numpy as np
        from .bimanual_assembly_env import _matrix_to_quat_wxyz
        if not self._robots:return
        matrices=self.get_end_effector_pose_matrix()
        poses=[]
        for matrix in matrices:
            matrix=np.asarray(matrix,dtype=np.float64).reshape(4,4)
            if not np.isfinite(matrix).all():
                raise ValueError('Z physical palm pose is non-finite')
            poses.append(np.r_[matrix[:3,3],_matrix_to_quat_wxyz(matrix[:3,:3])])
        self._right_tcp_pose,self._left_tcp_pose=poses
