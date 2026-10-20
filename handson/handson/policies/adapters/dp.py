"""Single-task diffusion policy with checkpoint-owned goals and normalization."""
from ..common import asset, read, require, vendor


def validate_contract(config, contract):
    require(config["task"] == contract["checkpoint_task"], "Checkpoint belongs to a different task")
    require(config["normalization"] == contract["normalization"], "DP normalization mismatch")
    require(config["goal_dim"] == contract["goal_dim"], "DP goal dimension mismatch")
    actual = config["package_provenance"]["training_manifest_sha256"]
    if actual != contract["training_manifest_sha256"]:
        require(not contract["requires_corrected_checkpoint"], "Superseded unconditional DP checkpoint")
        transition = contract["prompt_revision_transition"]
        old = {tuple(e["vector"]) for e in config["goal_contract"]["entries"]}
        new = {tuple(e["vector"]) for e in contract["goal_contract"]["entries"]}
        require(transition["legacy_checkpoint_compatible"] and old == new,
                "Checkpoint conditioning is not equivalent to the release goals")


class Policy:
    def __init__(self, options):
        import torch
        vendor("dp")
        from ..native import dp_engine
        require(options.task and asset("dp", options.task + ".json").is_file(),
                "DP requires one Core-50 --task; no implicit Transfer-16 skill routing")
        contract = read(asset("dp", options.task + ".json"))
        saved = torch.load(options.checkpoint, map_location="cpu", weights_only=False)
        require(saved["step"] == 30000, "Expected selected step-30000 DP checkpoint")
        self.config = saved["config"]
        validate_contract(self.config, contract)
        self.model = dp_engine.GoalDP(self.config["goal_dim"]).cuda().eval()
        self.model.load_state_dict(saved["model"], strict=True)
        self.stats = dp_engine.normalizers({"normalization": self.config["normalization"]}, "cuda")
        self.goals = {}
        for entry in contract["goal_contract"]["entries"]:
            for text in [entry["canonical_prompt"], *entry["train_paraphrases"]]:
                self.goals[text] = entry["vector"]
        self.receipt = dict(task=options.task, strict_checkpoint_load=True)

    def predict(self, state, images, prompt, seed):
        import torch
        from ..native import dp_engine as core
        require(prompt in self.goals, "Unknown DP goal; no fallback or guessed skill composition")
        state = torch.as_tensor(state, dtype=torch.float32).reshape(1, 1, 52)
        rgb = torch.stack([core.resize_image(images[c]) for c in self.config["camera_order"]])[None, None]
        goal = torch.tensor(self.goals[prompt], dtype=torch.float32)[None]
        observation = core.observations({"observation.state": state, "observation.images": rgb,
                                         "observation.goal": goal}, self.stats, "cuda")
        generator = torch.Generator(device="cuda").manual_seed(seed)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            condition = self.model._prepare_global_conditioning(observation)
            actions = self.model.conditional_sample(1, condition, generator=generator)
        return core.physical(actions.float(), self.stats["action"])[0].cpu().numpy()
