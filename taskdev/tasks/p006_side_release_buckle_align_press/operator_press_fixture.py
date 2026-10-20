#!/usr/bin/env python3
"""Bounded force-only press fixture; NOT a human episode or success certificate.

Fixture poses are set ONCE, before any driven step. During the trial only forces
and torques are applied. No fake hand contacts, success writes or per-step pose /
velocity resets. Uses the production environment, assets, materials and solver.
Run only on an idle workstation with an external watchdog.
"""
from __future__ import annotations

import argparse
import faulthandler
import importlib.metadata as md
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'handson'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert md.version('isaacsim').startswith('5.1.')
    assert md.version('isaaclab').startswith('2.3.2')
    os.environ.update(HANDSON_ENABLE_ISAACSIM_NEWTON='0',
                      HANDSON_ISAACLAB_DEFER_ARM_RENDER='1',
                      HANDSON_RECORD_PHYSX_GROUND_TRUTH='1')
    faulthandler.dump_traceback_later(150, exit=True)
    result = {'scope': 'force-driven aligned fixture; not natural grasp or successful episode',
              'runtime': {'isaacsim': md.version('isaacsim'), 'isaaclab': md.version('isaaclab')},
              'passed': False, 'samples': []}
    env = None
    try:
        from handson.tasks.mappings import CONFIG_MAPPING
        from handson.sim.isaaclab.p006_contract import (
            FEMALE_ROOT_PATH, MALE_ROOT_PATH, MALE_TONGUE_PATHS,
            GRIP_COLLIDER_PATHS, MIN_INSERTION_DEPTH_M, MAX_PENETRATION_M,
        )
        wrapped = CONFIG_MAPPING['p006_side_release_buckle_align_press']().get_environment(
            policy_mode=True, sim_backend='isaacsim', launch_isaac=True,
            headless=True, image_obs=False, randomize=False, render_mode='rgb_array')
        env = wrapped.unwrapped
        env.reset(seed=17)
        from pxr import UsdGeom, UsdPhysics, UsdShade
        from scipy.spatial.transform import Rotation

        # Audit the composed runtime, not just text in the USD source.
        colliders = []
        for p in env._stage.Traverse():
            if not any(str(p.GetPath()).startswith(root + '/') for root in (FEMALE_ROOT_PATH, MALE_ROOT_PATH)):
                continue
            if p.HasAPI(UsdPhysics.CollisionAPI):
                material, _ = UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial('physics')
                physics_mat = UsdPhysics.MaterialAPI(material.GetPrim())
                colliders.append({'path': str(p.GetPath()),
                                  'visibility': str(UsdGeom.Imageable(p).ComputeVisibility()),
                                  'material': str(material.GetPath()),
                                  'static_friction': physics_mat.GetStaticFrictionAttr().Get(),
                                  'dynamic_friction': physics_mat.GetDynamicFrictionAttr().Get()})
        result['colliders'] = colliders
        assert len(colliders) == 15 and all(c['visibility'] == 'invisible' for c in colliders)
        for c in colliders:
            expected = (1.10, .90) if c['path'] in GRIP_COLLIDER_PATHS else (.75, .62)
            assert np.allclose([c['static_friction'], c['dynamic_friction']], expected), c
        result['sim_cfg'] = str(env._sim.cfg)
        result['contact_filters'] = env._p006_contact_paths
        assert env._p006_contact_paths['buckle_body'][1] == (MALE_ROOT_PATH, *MALE_TONGUE_PATHS)

        torch = env._torch
        array = lambda t: t.detach().cpu().numpy().copy()
        def tensor(values, view):
            ref = view.get_transforms()
            return torch.as_tensor(np.asarray(values), dtype=ref.dtype, device=ref.device)
        def indices(view):
            return torch.arange(view.count, dtype=torch.long, device=view.get_transforms().device)
        fpos = np.array([-.18, -.08, 1.03])
        mpos = fpos + [.11, 0., .0036]
        female = env._p006_body_views['female']
        female.set_transforms(tensor([[*fpos, 0, 0, 0, 1]], female), indices(female))
        female.set_velocities(torch.zeros_like(female.get_velocities()), indices(female))
        _, articulation, art_ids, root_pose, root_vel, dof_pos, dof_vel = next(
            s for s in env._task_reset_articulation_states if s[0] == MALE_ROOT_PATH)
        art_pose = root_pose.clone()
        art_pose[0] = torch.as_tensor([*mpos, 0, 0, 0, 1], dtype=art_pose.dtype, device=art_pose.device)
        articulation.set_root_transforms(art_pose, art_ids)
        articulation.set_root_velocities(torch.zeros_like(root_vel), art_ids)
        articulation.set_dof_positions(torch.zeros_like(dof_pos), art_ids)
        articulation.set_dof_velocities(torch.zeros_like(dof_vel), art_ids)
        max_depth = max_penetration = max_contact = 0.
        max_compression = np.zeros(2)
        stable_depth_steps = 0
        stable_motion_steps = 0
        link_paths = list(articulation.link_paths[0])
        result['male_link_paths'] = link_paths
        for step in range(720):
            # 6-second slow press, then stationary force fixture at 22.6 mm.
            progress = min(1., step / 480.)
            targets = {'female': fpos, 'male': fpos + [.11 - .045 * progress, 0., .0036]}
            link_forces, link_torques, link_positions = (np.zeros((1, articulation.max_links, 3)) for _ in range(3))
            for key, mass in [('female', .12), ('male', .075), ('tongue_neg', .0075), ('tongue_pos', .0075)]:
                view = env._p006_body_views[key]
                pose, vel = array(view.get_transforms())[0], array(view.get_velocities())[0]
                rotation = Rotation.from_quat(pose[3:7])
                com = pose[:3] + rotation.apply(array(view.get_coms())[0, :3])
                force, torque = np.array([0., 0., mass * 9.81]), np.zeros(3)
                if key in targets:
                    fixture_force = 400. * (targets[key] - pose[:3]) - 8. * vel[:3]
                    force += np.clip(fixture_force, -3., 3.)
                    torque = np.clip(-.03 * rotation.as_rotvec() - .001 * vel[3:], -.03, .03)
                if key == 'female':
                    view.apply_forces_and_torques_at_position(tensor([force], view), tensor([torque], view),
                                                              tensor([com], view), indices(view), True)
                else:
                    path = {'male': MALE_ROOT_PATH, 'tongue_neg': MALE_TONGUE_PATHS[0],
                            'tongue_pos': MALE_TONGUE_PATHS[1]}[key]
                    i = link_paths.index(path)
                    link_forces[0, i], link_torques[0, i], link_positions[0, i] = force, torque, com
            articulation.apply_forces_and_torques_at_position(
                tensor(link_forces, female), tensor(link_torques, female),
                tensor(link_positions, female), art_ids, True)
            env._step_isaaclab()
            metrics = env._refresh_metrics(update_latch=(step % 4 == 3))
            dofs = array(articulation.get_dof_positions())[0]
            max_compression = np.maximum(max_compression, np.abs(dofs))
            max_depth = max(max_depth, metrics['insertion_depth_m'])
            max_penetration = max(max_penetration, metrics['max_penetration_m'])
            max_contact = max(max_contact, metrics['buckle_body_contact_force_n'])
            if metrics['insertion_depth_m'] >= MIN_INSERTION_DEPTH_M and metrics['aligned']:
                stable_depth_steps += 1
            else:
                stable_depth_steps = 0
            if (stable_depth_steps and metrics['relative_linear_speed_mps'] <= .12
                    and metrics['relative_angular_speed_radps'] <= 1.):
                stable_motion_steps += 1
            else:
                stable_motion_steps = 0
            if step % 12 == 0 or step == 719:
                result['samples'].append({'step': step, 'dofs_m': dofs.tolist(),
                    'female_pose': env._body_state('female')[0].tolist(),
                    'male_pose': env._body_state('male')[0].tolist(),
                    'female_velocity': env._body_state('female')[1].tolist(),
                    'male_velocity': env._body_state('male')[1].tolist(),
                    **{k: metrics[k] for k in ('insertion_depth_m', 'axis_error_deg', 'transverse_offset_m',
                         'max_penetration_m', 'buckle_body_contact_force_n', 'success_now', 'failed',
                         'relative_linear_speed_mps', 'relative_angular_speed_radps')}})
            assert not metrics['success_now'], 'No hands: fixture must never count as task success'
            assert metrics['finite'], 'nonfinite physical readback'
        result.update(max_depth_m=max_depth, max_penetration_m=max_penetration,
                      max_contact_n=max_contact, max_compression_m=max_compression.tolist(),
                      stable_depth_physics_steps=stable_depth_steps,
                      stable_motion_physics_steps=stable_motion_steps)
        result['passed'] = bool(max_depth >= MIN_INSERTION_DEPTH_M and stable_depth_steps >= 48
                                and stable_motion_steps >= 48
                                and min(max_compression) > .001
                                and max_penetration <= MAX_PENETRATION_M
                                and max_contact >= .1)
    except BaseException as exc:
        result['error'] = repr(exc)
        traceback.print_exc()
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        print('P006_PRESS_FIXTURE', result['passed'], result.get('error', ''), flush=True)
        if env is not None:
            faulthandler.dump_traceback_later(20, exit=True)
            env._fast_shutdown_on_close = True
            env.close()
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
