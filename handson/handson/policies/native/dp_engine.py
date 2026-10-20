import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionModel, DiffusionConditionalUnet1d
from lerobot.configs.types import FeatureType, PolicyFeature
CAMERAS = ("front", "wrist_left", "wrist_right")
IMAGE_MEAN = torch.tensor([.485, .456, .406]).reshape(1, 1, 1, 3, 1, 1)
IMAGE_STD = torch.tensor([.229, .224, .225]).reshape(1, 1, 1, 3, 1, 1)

def resize_image(image):
    x = torch.from_numpy(image.copy()).permute(2, 0, 1).float().unsqueeze(0) / 255.
    h, w = image.shape[:2]
    scale = min(224 / h, 224 / w)
    oh, ow = round(h * scale), round(w * scale)
    x = F.interpolate(x, size=(oh, ow), mode="bilinear", align_corners=False, antialias=True)
    x = F.pad(x, ((224-ow)//2, 224-ow-(224-ow)//2, (224-oh)//2, 224-oh-(224-oh)//2))
    return x[0]

class GoalDP(DiffusionModel):
    def __init__(self, goal_dim):
        features = {"observation.state": PolicyFeature(FeatureType.STATE, (52,))}
        features.update({f"observation.images.{camera}": PolicyFeature(FeatureType.VISUAL, (3,224,224))
                         for camera in CAMERAS})
        config = DiffusionConfig(
            n_obs_steps=1, horizon=32, n_action_steps=30, drop_n_last_frames=0,
            input_features=features, output_features={"action": PolicyFeature(FeatureType.ACTION, (52,))},
            vision_backbone="resnet18", crop_shape=None,
            pretrained_backbone_weights=None, use_group_norm=False,
            use_separate_rgb_encoder_per_camera=False, down_dims=(512,1024,2048),
            num_train_timesteps=100, num_inference_steps=10, prediction_type="epsilon",
            clip_sample=True, clip_sample_range=1.0, do_mask_loss_for_padding=True,
            device="cuda", use_amp=True,
        )
        super().__init__(config)
        self.goal_encoder = nn.Sequential(nn.Linear(goal_dim,32), nn.SiLU(), nn.Linear(32,32))
        self.unet = DiffusionConditionalUnet1d(
            config, global_cond_dim=52 + self.rgb_encoder.feature_dim * 3 + 32)

    def _prepare_global_conditioning(self, batch):
        return torch.cat((super()._prepare_global_conditioning(batch),
                          self.goal_encoder(batch["observation.goal"])), dim=-1)

    def loss(self, observation, targets, valid):
        cond = self._prepare_global_conditioning(observation)
        eps = torch.randn_like(targets)
        times = torch.randint(0, 100, (len(targets),), device=targets.device)
        noisy = self.noise_scheduler.add_noise(targets, eps, times)
        pred = self.unet(noisy, times, global_cond=cond)
        mask = valid.unsqueeze(-1)
        return ((pred.float() - eps.float()).square() * mask).sum() / (mask.sum() * 52).clamp_min(1)

def normalizers(metadata, device):
    result = {}
    for key in ("state", "action"):
        norm = metadata["normalization"][key]
        lo = torch.tensor(norm["q01"], device=device, dtype=torch.float32)
        hi = torch.tensor(norm["q99"], device=device, dtype=torch.float32)
        result[key] = (lo, (hi-lo).clamp_min(1e-8))
    return result

def normalize(x, stats):
    lo, scale = stats
    return (x - lo) / scale * 2 - 1

def physical(x, stats):
    lo, scale = stats
    return (x + 1) * .5 * scale + lo

def observations(batch, stats, device):
    # Allowlist: ground-truth actions, future observations and masks never reach sampling.
    image = batch["observation.images"].to(device, non_blocking=True)
    return {
        "observation.state": normalize(batch["observation.state"].to(device), stats["state"]).clamp(-3,3),
        "observation.images": (image - IMAGE_MEAN.to(device)) / IMAGE_STD.to(device),
        "observation.goal": batch["observation.goal"].to(device),
    }
