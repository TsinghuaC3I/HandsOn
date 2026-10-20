import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from handson.evaluation import rollout as m


class RolloutTests(unittest.TestCase):
    def test_b_setting_requires_pre_reset_receipt(self):
        task = 't37_scrub_bottle'
        with self.assertRaises(ValueError):
            m.task_setting_contract({'task': task})
        from handson.evaluation import settings as b_task_config
        selection = b_task_config.select_setting(0, task, 0)
        selection_job = {
            'task': task,
            'setting_selection': selection,
            'setting_applied_before_reset': True,
            'setting_runtime_verified': dict(verified=True, task=task,
                                             setting_id=selection['setting_id']),
        }
        self.assertEqual(m.task_setting_contract(selection_job), selection)

    def test_c12_irreversible_latch_is_not_other_tasks_failure(self):
        ret=dict(terminated=False,truncated=False,info={'c12_direct_hand_contact':True})
        self.assertEqual(m.terminal_reason('t11_line_up_blocks',ret),'native_c12_direct_hand_contact')
        self.assertIsNone(m.terminal_reason('t24_place_ball_in_bowl',ret))
        self.assertEqual(m.terminal_reason('t24_place_ball_in_bowl',{**ret,'terminated':True}),'native_terminated')

    def trial(self, task='t24_place_ball_in_bowl', fail_step=None):
        observed=[]; images={k:np.ones((4,4,3),dtype=np.uint8)*100 for k in ('main','wrist_left','wrist_right')}
        env=SimpleNamespace(_control_dt=1/30,_physics_dt=1/120,_time_limit=2.,env_step=0,
                            _n_substeps=4,_sim=SimpleNamespace(current_time_step_index=0))
        def step(command):
            env.env_step+=1;env._sim.current_time_step_index+=4
            env._c12_direct_hand_contact=env.env_step==fail_step
            return None,0,False,env.env_step==60,{'succeed':False}
        env.step=step
        class Client:
            def __init__(self,cfg,folder):pass
            def start(self):return {'checkpoint':'fixture_only'}
            def predict(self,state,frames,index,seed):
                observed.append((env.env_step,index,seed));return np.zeros((30,52))
            def close(self):pass
        class Video:
            def __init__(self,*args,**kw):self.n=0
            def write(self,frames):self.n+=1
            def close(self):return {k:dict(exit_code=0,frames=self.n) for k in images}
        def render(*a,**kw):kw['image_sink'](images)
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td)
            for k,v in images.items():Image.fromarray(v).save(folder/f'initial_000_{k}.png')
            job=dict(task=task,source_execution=None,reference_provenance={'recorded_git':{'commit':'fixture'}},
                benchmark={'executor':'legacy_native_eef','render_stride':30},
                policy_probe={'max_chunks':2,'prompt':'fixture','ffmpeg':'fixture'})
            with patch.object(m,'PolicyClient',Client),patch.object(m,'Videos',Video),\
                 patch.object(m,'live_state',lambda e:(np.zeros(52),{})),\
                 patch.object(m,'command_readback',lambda e,c:{}),\
                 patch('handson.evaluation.cameras.render_probe',render):
                result=m.run(env,job,folder,0,77,lambda e:{})
            self.assertEqual(env._sim.current_time_step_index,result['steps']*4)
            self.assertEqual(result['actions'], result['steps'])
            return observed,result

    def test_policy_freshness_physics_and_video_cadence(self):
        requests,result=self.trial()
        self.assertEqual(requests,[(0,0,77),(30,1,78)])
        self.assertEqual(result['frame_steps'],[0,30,60])
        self.assertEqual(result['steps'],60)
        self.assertFalse(result['counts_as_formal_trial'])

    def test_no_action_after_native_c12_latch(self):
        requests,result=self.trial('t11_line_up_blocks',1)
        self.assertEqual(result['steps'],1)
        self.assertEqual(len(requests),1)
        self.assertEqual(result['frame_steps'],[0,1])


if __name__=='__main__':unittest.main()
