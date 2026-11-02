import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from handson.evaluation import cli, settings
from handson.evaluation.goals import bind
from types import SimpleNamespace


class EntryTests(unittest.TestCase):
    def test_hold_diagnostics_keep_each_model_execution_profile_without_a_server(self):
        from handson.policies.profiles import PROFILES, execution_settings
        from handson.evaluation.schedule import RenderSchedule
        from handson.evaluation.history import ImageHistory
        for model in PROFILES:
            with self.subTest(model=model):
                argv = ["evaluation", "--task", "t24_place_ball_in_bowl", "--model", model,
                        "--diagnostic-steps", "90", "--worker"]
                with patch("sys.argv", argv), patch("handson.evaluation.native.run_episode", return_value={}) as run:
                    cli.main()
                kwargs = run.call_args.kwargs
                self.assertIsNone(kwargs["policy"])
                self.assertEqual(kwargs["diagnostic_steps"], 90)
                expected = execution_settings(model)
                self.assertEqual(kwargs["render_stride"], expected.pop("render_stride"))
                self.assertEqual(kwargs["execution_settings"], expected)
                schedule = RenderSchedule(kwargs["render_stride"], expected["action_horizon"])
                ImageHistory(expected["history_frames"], expected["history_stride"],
                             expected["action_horizon"], schedule.stride)

    @patch("handson.robots.assets.require_benchmark_assets")
    def test_silent_kit_exit_is_not_a_completed_trial(self, _assets):
        class Worker:
            stdout = io.StringIO("constructor failed before returning an environment\n")
            returncode = 0
            def wait(self):
                return 0
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "episode"
            argv = ["evaluation", "--task", "t12_load_and_start_microwave",
                    "--diagnostic-steps", "90", "--output", str(output)]
            with patch("sys.argv", argv), patch.object(cli.subprocess, "Popen", return_value=Worker()):
                with self.assertRaises(SystemExit) as error:
                    cli.main()
            self.assertEqual(error.exception.code, 1)
            self.assertFalse((output / "RESULT.json").exists())
            self.assertFalse(json.loads((output / "ERROR.json").read_text())["counts_as_model_failure"])
            self.assertIn("constructor failed", (output / "runtime.log").read_text())

    def test_seeded_knob_settings_have_matching_instructions(self):
        task = "t38_set_dual_knobs"
        import math
        for profile in settings.task_spec(task)["profiles"]:
            selection = settings.select_setting(42, task, 0, forced_setting=profile["setting_id"])
            angles = selection["profile"]["target_degrees"]
            env = SimpleNamespace(env_step=0, _ka005_2_reset_sample=SimpleNamespace(
                target_rad=tuple(math.radians(x) for x in angles)))
            receipt = bind(env, task)
            self.assertEqual(receipt["goal"]["left_angle_degrees"], angles[0])
            self.assertEqual(receipt["goal"]["right_angle_degrees"], -angles[1])

    def test_unknown_goal_cannot_fall_back_to_default_instruction(self):
        env = SimpleNamespace(env_step=0, _target_object_name="unknown_target")
        with self.assertRaises(ValueError):
            bind(env, "t24_place_ball_in_bowl")


if __name__ == "__main__":
    unittest.main()
