"""CPU contract tests; these do not certify learned-checkpoint rollouts."""
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from http.server import HTTPServer
from unittest.mock import patch
import urllib.error
import urllib.request

import numpy as np

from handson.policies.common import ROOT, read
from handson.tasks.language import bank as language_bank
from handson.policies.profiles import PROFILES, execution_settings, noise_seed
from handson.policies.server import handler_for, predict_request
from handson.policies.adapters.dp import validate_contract
from handson.policies.native.wam_interfaces import PolicyNormalizer, eef52_to_alpha80, alpha80_to_eef52

TASK = "t26_place_fruit_in_bucket"
PROMPT = language_bank()["tasks"][TASK]["goals"][0]["canonical"]


def packet(model, task=TASK, **updates):
    history = PROFILES[model].history_frames
    shape = (history, 20, 30, 3) if history > 1 else (20, 30, 3)
    values = dict(state=np.zeros(52, np.float32), prompt=np.array(PROMPT), task=np.array(task),
                  seed=np.array(1), request=np.array(0),
                  **{k: np.zeros(shape, np.uint8) for k in ("front", "wrist_left", "wrist_right")})
    values.update(updates)
    stream = io.BytesIO()
    np.savez(stream, **values)
    return stream.getvalue()


class RecordingPolicy:
    def __init__(self, model):
        self.model = model
        self.calls = []

    def predict(self, state, images, prompt, seed):
        self.calls.append((state, images, prompt, seed))
        return np.repeat(np.arange(PROFILES[self.model].model_horizon, dtype=np.float32)[:, None], 52, axis=1)


class PaperPolicyTests(unittest.TestCase):
    def test_all_profiles_execute_only_the_selected_prefix(self):
        for model, profile in PROFILES.items():
            with self.subTest(model=model):
                policy = RecordingPolicy(model)
                result = predict_request(policy, profile, model, packet(model))
                actions = np.asarray(result["actions"])
                self.assertEqual(actions.shape, (profile.action_horizon, 52))
                np.testing.assert_array_equal(actions[:, 0], np.arange(profile.action_horizon))
                self.assertEqual(policy.calls[0][2], PROMPT)
                self.assertEqual(policy.calls[0][3], result["noise_seed"])

    def test_hy_requires_real_six_frame_input(self):
        policy = RecordingPolicy("hy")
        with self.assertRaisesRegex(ValueError, "rank 4"):
            predict_request(policy, PROFILES["hy"], "hy", packet("hy", front=np.zeros((20, 30, 3), np.uint8)))
        self.assertFalse(policy.calls)
        self.assertEqual(execution_settings("hy"), dict(action_horizon=30, render_stride=10,
                                                       history_frames=6, history_stride=10))

    def test_no_wrong_task_or_prompt_fallback(self):
        for payload, bound in [(packet("dp"), "t03_cover_box"),
                               (packet("dp", prompt=np.array("Do an unrelated task.")), TASK)]:
            with self.assertRaises(ValueError):
                predict_request(RecordingPolicy("dp"), PROFILES["dp"], "dp", payload, bound)

    def test_rng_preserves_original_task_names_and_spatial_stream(self):
        def original(text):
            return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")
        self.assertEqual(noise_seed("dp", 1, 2, TASK), original("dp:a01_fruit_to_bucket:1:2") % (2**63 - 1))
        self.assertEqual(noise_seed("openwam", 1, 2, TASK), original("openwam:a01_fruit_to_bucket:1:2") % (2**63 - 1))
        self.assertEqual(noise_seed("spatial", 1, 2), original("spatial-84000:1:2"))

    def test_alpha_boundary_preserves_sides_fingers_and_rotations(self):
        x = np.linspace(-0.5, 0.5, 52, dtype=np.float32)
        alpha = eef52_to_alpha80(x)
        np.testing.assert_array_equal(alpha[10:30], x[32:52])
        np.testing.assert_array_equal(alpha[44:64], x[6:26])
        np.testing.assert_allclose(alpha80_to_eef52(alpha), x, atol=1e-6)
        normalizer = PolicyNormalizer(read(ROOT / "assets/openwam/norm_stats.json"), "openwam")
        self.assertEqual(int(normalizer.valid_mask.sum()), 58)

    def test_motus_uses_action_bounds_for_state(self):
        stats = dict(state=dict(min=[100] * 52, max=[200] * 52),
                     action=dict(min=[-2] * 52, max=[2] * 52))
        normalizer = PolicyNormalizer(stats, "motus")
        np.testing.assert_array_equal(normalizer.normalize(np.zeros(52), "state"), np.full(52, .5))
        np.testing.assert_array_equal(normalizer.denormalize(np.full((48, 52), .5)), np.zeros((48, 52)))

    def test_dp_all_fifty_contracts_and_wrong_checkpoint_rejection(self):
        files = list((ROOT / "assets/dp").glob("*.json"))
        self.assertEqual(len(files), 50)
        for path in files:
            contract = read(path)
            config = dict(task=contract["checkpoint_task"], normalization=contract["normalization"],
                          goal_dim=contract["goal_dim"], goal_contract=contract["goal_contract"],
                          package_provenance=dict(training_manifest_sha256=contract["training_manifest_sha256"]))
            validate_contract(config, contract)
            wrong = dict(config, task="different_task")
            with self.assertRaisesRegex(ValueError, "different task"):
                validate_contract(wrong, contract)

    def test_loopback_http_returns_metadata_and_physical_actions(self):
        policy = RecordingPolicy("openwam")
        metadata = dict(PROFILES["openwam"].metadata(), model="openwam", checkpoint_sha256="test-only")
        server = HTTPServer(("127.0.0.1", 0), handler_for(policy, "openwam", metadata))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}"
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(url) as response:
                self.assertEqual(json.load(response), metadata)
            with opener.open(urllib.request.Request(url, data=packet("openwam"))) as response:
                self.assertEqual(np.asarray(json.load(response)["actions"]).shape, (16, 52))
        finally:
            server.shutdown()
            thread.join()
            server.server_close()

    def test_evaluator_rejects_conflicting_horizon_before_launch(self):
        from handson.evaluation.cli import main
        argv = ["evaluation", "--task", TASK, "--model", "openwam", "--action-horizon", "30"]
        with patch("sys.argv", argv), patch("subprocess.Popen") as launch:
            with self.assertRaises(SystemExit) as failure:
                main()
            self.assertEqual(failure.exception.code, 2)
            launch.assert_not_called()

    def test_evaluator_rejects_wrong_model_metadata_before_launch(self):
        from handson.evaluation.cli import main
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "metadata.json"
            path.write_text(json.dumps(dict(PROFILES["pi05"].metadata(), model="pi05", checkpoint="test")))
            argv = ["evaluation", "--task", TASK, "--model", "hy",
                    "--policy-url", "http://127.0.0.1:8000", "--policy-metadata", str(path)]
            with patch("sys.argv", argv), patch("subprocess.Popen") as launch:
                with self.assertRaises(SystemExit):
                    main()
                launch.assert_not_called()

    def test_dp_language_contract_covers_published_goals(self):
        bank = language_bank()["tasks"]
        for path in (ROOT / "assets/dp").glob("*.json"):
            contract = read(path)
            known = {text for entry in contract["goal_contract"]["entries"]
                     for text in [entry["canonical_prompt"], *entry["train_paraphrases"]]}
            expected = {text for goal in bank[path.stem]["goals"]
                        for text in [goal["canonical"], *goal["train"]]}
            self.assertFalse(expected - known, path.stem)


if __name__ == "__main__":
    unittest.main()
