import torch
from torch import nn
import numpy as np
from .wam_images import tensor_images
MODELS = None

class Float32Linear(nn.Linear):
    """Keep native timestep math float32 even when ZeRO casts parameter storage."""
    def forward(self, x):
        return torch.nn.functional.linear(x.float(), self.weight.float(),
                                         self.bias.float() if self.bias is not None else None)

class MotusAdapter(nn.Module):
    def train(self, mode=True):
        super().train(mode)
        self.policy.vlm_model.eval()
        return self

    def __init__(self, *args, **kwargs):
        raise RuntimeError("Construct with the selected-checkpoint Motus loader")

    def condition(self, samples):
        device = torch.device("cuda")
        return {"first_frame": torch.stack([tensor_images([s["images"][0]], device)[0] for s in samples]),
                "state": torch.as_tensor(np.stack([s["state"] for s in samples]), device=device, dtype=torch.bfloat16),
                "vlm_inputs": [self.preprocess_vlm(s["prompt"], s["images"][0], self.processor) for s in samples],
                "language_embeddings": [self.text_cache[s["prompt"]].to(device=device, dtype=torch.bfloat16) for s in samples]}

    def forward(self, samples):
        inputs = self.condition(samples)
        device = inputs["state"].device
        inputs.update(video_frames=torch.stack([tensor_images(s["images"][1:], device) for s in samples]),
                      actions=torch.as_tensor(np.stack([s["action"] for s in samples]), device=device, dtype=torch.bfloat16),
                      action_time_mask=torch.as_tensor(np.stack([s["action_time_mask"] for s in samples]), device=device),
                      video_frame_mask=torch.as_tensor(np.stack([s["video_mask"] for s in samples]), device=device))
        result = self.policy.training_step(**inputs)
        self.last_metrics = {"train/flow_loss": result["total_loss"].detach(),
                             "train/action_flow_loss": result["action_loss"].detach(),
                             "train/video_flow_loss": result["video_loss"].detach()}
        return result["total_loss"]

    @torch.no_grad()
    def decode_action(self, condition, steps, seed):
        # This function receives no future images or target actions.
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        _, actions = self.policy.inference_step(**self.condition([condition]), num_inference_steps=steps, decode_video=False)
        return actions.float()

class OpenWAMAdapter(nn.Module):
    def train(self, mode=True):
        super().train(mode)
        for name in self.source_config.model.freeze:
            module = self.policy
            for part in name.split("."):
                module = getattr(module, part, None)
                if module is None:
                    break
            if isinstance(module, nn.Module):
                module.eval()
        return self

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        from openwam.train.utils.ckpt_model_loader import build_architecture_from_ckpt_dir
        _, self.policy, self.source_config = build_architecture_from_ckpt_dir(
            str(MODELS / "OpenWAM-Alpha-Pretrain-Foundation-Model/checkpoint_step_154000.safetensors"),
            weights_required=True)
        if self.policy.action_backbone.action_dim != 80 or self.policy.proprio_dim != 80:
            raise ValueError("Unexpected Foundation checkpoint interface")
        # Keep every pretrained 80D interface weight. All adaptation is in the reader.
        self.policy.freeze_modules(list(self.source_config.model.freeze))
        self.policy.set_dtype_device(torch.bfloat16, torch.device("cuda", torch.cuda.current_device()))
        self.policy.init_training_schedulers(1000)
        self.policy.set_training_runtime(use_gradient_checkpointing=cfg["use_gradient_checkpointing"])
        self.migration = {"source_action_dim": 80, "target_action_dim": 80,
                          "native_robot_action_dim": 52, "fresh_modules": [],
                          "action_horizon": 32, "physical_slots": 58, "padding_slots": 22}

    def forward(self, samples):
        batch = [{"video": s["images"], "prompt": s["prompt"], "action": s["action"],
                  "proprio": s["state"], "proprio_mask": s["model_mask"][None],
                  "action_mask": s["action_time_mask"][:, None] & s["model_mask"][None],
                  "video_mask": s["video_mask"]} for s in samples]
        result = self.policy.compute_loss(**self.policy.prepare_inputs(batch), lambda_video=1.0, lambda_action=1.0)
        self.last_metrics = {"train/flow_loss": result["loss"].detach(),
                             "train/action_flow_loss": result["loss_action"].detach(),
                             "train/video_flow_loss": result["loss_video"].detach()}
        return result["loss"]

    @torch.no_grad()
    def decode_action(self, condition, steps, seed):
        from openwam.deploy.denoise_schedule import schedule_sync
        import copy
        # Native video scheduler is a read-only property backed by _scheduler.
        # Preserve object identity (including holder aliases), snapshot mutable
        # scheduler state, and restore it even if generation fails.
        vb, ab = self.policy.video_backbone, self.policy.action_backbone
        schedulers = vb.scheduler, ab.scheduler
        saved = [copy.deepcopy(vars(scheduler)) for scheduler in schedulers]
        try:
            schedule = schedule_sync(vb.scheduler, ab.scheduler, num_steps=steps, shift=self.cfg["inference"]["scheduler_shift"])
            result = self.policy.generate(schedule, condition["prompt"], first_frame_image=[condition["images"][0]],
                       num_frames=9, action_num_frames=33, height=384, width=320, seed=seed,
                       num_inference_steps=steps, decode_video=False, cfg_scale=1.0,
                       proprio=torch.as_tensor(condition["state"], device=self.policy.device, dtype=self.policy.dtype)[None],
                       active_action_mask=torch.as_tensor(condition["model_mask"], dtype=torch.bool, device=self.policy.device))
            return torch.as_tensor(result["actions"], device=self.policy.device, dtype=torch.float32)[None]
        finally:
            for scheduler, state in zip(schedulers, saved):
                vars(scheduler).clear()
                vars(scheduler).update(state)
