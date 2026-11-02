import ast
import json
from pathlib import Path
import unittest

from handson.sim.isaaclab import physics_config


class PaperGpuBuffersTests(unittest.TestCase):
    def test_native_factory_uses_original_allocations_for_all_approved_tasks(self):
        root = Path(physics_config.__file__).parents[2]
        contracts = json.loads((root / 'tasks/assets/native_contracts.json').read_text())
        tasks = {task for task, row in contracts.items() if row['origin_mode'] == 'approved_b_boundary'}
        self.assertEqual(len(tasks), 16)
        tree = ast.parse(Path(physics_config.__file__).with_name('abc_collection_env.py').read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == '_physx_gpu_buffer_kwargs')
        # Only the real paper-profile branch should execute for these tasks.
        # Later generic allocation dependencies are deliberately not provided.
        namespace = {'paper_gpu_physx_kwargs': physics_config.paper_gpu_physx_kwargs}
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<native buffer factory>', 'exec'), namespace)
        for task in tasks:
            with self.subTest(task=task):
                actual = namespace[function.name](task)
                if task == 't21_pass_block':
                    self.assertEqual(actual['gpu_max_num_partitions'], 1)
                    self.assertEqual(actual['gpu_max_rigid_contact_count'], 2**18)
                    self.assertEqual(actual['gpu_found_lost_pairs_capacity'], 2**16)
                else:
                    self.assertEqual(actual, {'gpu_max_rigid_contact_count': 2**20,
                                              'gpu_max_rigid_patch_count': 5 * 2**15})
                self.assertNotIn('gpu_max_particle_contacts', actual)

    def test_profiles_do_not_change_unrelated_or_transfer_tasks(self):
        for task in ['eval_comb03_pass_block_and_close_box',
                     'eval_flex03_pack_pillow', 'unregistered_task']:
            self.assertIsNone(physics_config.paper_gpu_physx_kwargs(task))

    def test_core_allocations_match_recovered_original_source_cohorts(self):
        fixture = json.loads((Path(__file__).parent / 'fixtures' /
                              'paper_core_gpu_buffers.json').read_text())
        tree = ast.parse(Path(physics_config.__file__).with_name('abc_collection_env.py').read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == '_physx_gpu_buffer_kwargs')
        namespace = {'paper_gpu_physx_kwargs': physics_config.paper_gpu_physx_kwargs}
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<native buffer factory>', 'exec'), namespace)
        for row in fixture['cohorts']:
            with self.subTest(source=row['source_key']):
                self.assertIsNotNone(physics_config.paper_gpu_physx_kwargs(row['task']))
                self.assertEqual(namespace[function.name](row['task']), row['kwargs'])

    def test_native_articulated_tasks_retain_lab_default_pair_and_heap_buffers(self):
        # The paper leaves these capacities at Isaac Lab 2.3.2 defaults. The
        # later blanket low-memory profile also changed particle capacity.
        for task in ['t16_open_laptop', 't17_open_latched_door',
                     't18_open_micropipette_box', 't48_unfold_fan', 't49_unfold_ruler',
                     't28_place_tableware', 't22_pass_screwdriver']:
            with self.subTest(task=task):
                self.assertEqual(physics_config.paper_gpu_physx_kwargs(task), {
                    'gpu_max_rigid_contact_count': 2**20,
                    'gpu_max_rigid_patch_count': 5 * 2**15,
                })

    def test_caller_cannot_mutate_later_episode_allocations(self):
        task = 't15_open_drawers'
        first = physics_config.paper_gpu_physx_kwargs(task)
        first['gpu_max_num_partitions'] = 1
        self.assertNotIn('gpu_max_num_partitions', physics_config.paper_gpu_physx_kwargs(task))

    def test_carton_retains_eight_partitions_and_default_particle_capacity(self):
        actual = physics_config.paper_gpu_physx_kwargs('t10_lift_carton')
        self.assertEqual(actual['gpu_max_num_partitions'], 8)
        self.assertEqual(actual['gpu_max_rigid_contact_count'], 2**18)
        self.assertEqual(actual['gpu_heap_capacity'], 2**22)
        self.assertNotIn('gpu_max_particle_contacts', actual)
        self.assertNotIn('gpu_max_soft_body_contacts', actual)


if __name__ == '__main__':
    unittest.main()
