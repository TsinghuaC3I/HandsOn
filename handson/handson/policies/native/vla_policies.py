"""Pinned upstream policy implementation; paper package-owned EEF52 adaptation."""
import copy
from dataclasses import MISSING, fields
import json
from types import SimpleNamespace, MethodType

import numpy as np
import torch
from torch import nn
from transformers import AutoProcessor

FOUNDATION = {}
NORM_DIR = None
FEATURE_CONFIG = None
VIEWS = ("front", "wrist_left", "wrist_right")

def stack_batch(items):
    return {k: torch.stack([x[k] for x in items]) for k in items[0]}


def no_suffix_cuda_graph(self, initial_noise):
    """FSDP collectives cannot participate in the native single-GPU capture."""
    return False


class DMProcessor:
    def __init__(self, training=False, horizon=50):
        from opendm.data.augmentations import NoAugmentationPipeline, TrainingTransformPipeline
        from opendm.data.transforms import ChatTokenization, Normalize, PixelTransform
        from opendm.data.collator import TrainingCollator
        self.processor = AutoProcessor.from_pretrained(str(FOUNDATION["dm05"]), local_files_only=True)
        self.normalize = Normalize(str(NORM_DIR / "dm05_norm.json"), ["state", "action"], use_quantiles=True)
        self.pixel = PixelTransform(TrainingTransformPipeline() if training else NoAugmentationPipeline())
        # Fail on truncation instead of silently losing the task instruction.
        self.tokens = ChatTokenization(self.processor, max_length=4096,
                                       image_prompts=["Head", "Left wrist", "Right wrist"],
                                       enable_logging=False)
        self.collate = TrainingCollator(self.processor.tokenizer.pad_token_id, max_length=1536)

    def __call__(self, sample, inference=False):
        item = {"state": sample["state"].copy(), "images": sample["images"],
                "prompt": sample["prompt"],
                "meta_data": {"robot_type": "ROKAE_WUJI1", "control_mode": "eef"}}
        # Ground truth and episode-end masks are NOT passed to sampling.
        if not inference:
            item["action"] = sample["action"][None].copy()
            item["action_mask"] = np.broadcast_to(sample["valid"][None, :, None], item["action"].shape).copy()
        item = self.tokens(self.normalize(self.pixel(item)))
        assert item["input_ids"].shape[1] <= 1536, "paper instruction would be truncated"
        if inference:
            result = {k: v for k, v in item.items()
                      if v is not None and k not in ("action", "action_mask")}
            padding = 1536 - result["input_ids"].shape[1]
            for key in ("input_ids", "attention_mask", "token_type_ids", "history_mask"):
                if key in result:
                    result[key] = torch.nn.functional.pad(
                        result[key], (0, padding),
                        value=self.processor.tokenizer.pad_token_id if key == "input_ids" else 0)
            return result
        return item


class LingProcessor:
    def __init__(self, training=False, horizon=50):
        from lingbotvla.data.vla_data.utils import FeatureTransform
        self.processor = AutoProcessor.from_pretrained(str(FOUNDATION["qwen"]), local_files_only=True)
        data = SimpleNamespace(
            joints=["{'eef52': 52}"],
            cameras=["camera_top", "camera_wrist_left", "camera_wrist_right"],
            norm_type="meanstd", max_state_dim=75, max_action_dim=75,
            resize_imgs_with_padding=[224, 224], tokenizer_max_length=192,
        )
        self.feature = FeatureTransform(str(FEATURE_CONFIG), data,
                                        self.processor.tokenizer, self.processor.image_processor,
                                        chunk_size=horizon, norm_stats_path=str(NORM_DIR / "lingbot_norm.json"))
        self.collate = stack_batch
        self.horizon = horizon

    def __call__(self, sample, inference=False):
        item = {"observation.state": torch.tensor(sample["state"]),
                "task": sample["prompt"]}
        # The upstream preprocessing object expects action fields even at
        # inference. Supply independent zeros, never the ground-truth actions;
        # discard these fields before passing inputs into the model.
        item["action"] = torch.zeros(self.horizon, 52) if inference else torch.tensor(sample["action"])
        item["action_is_pad"] = torch.zeros(self.horizon, dtype=torch.bool) if inference else torch.tensor(~sample["valid"])
        for view, image in zip(VIEWS, sample["images"], strict=True):
            item[f"observation.images.{view}"] = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255
        full_prompt = f"<bos>{sample['prompt']}\n"
        assert len(self.processor.tokenizer(full_prompt)["input_ids"]) <= 192, "Prompt truncation"
        result = self.feature.apply(item)
        if inference:
            return {k: v.unsqueeze(0) for k, v in result.items()
                    if k in ("images", "img_masks", "state", "lang_tokens", "lang_masks")}
        return result


def dataclass_defaults(cls):
    result = {}
    for f in fields(cls):
        if f.default is not MISSING:
            result[f.name] = copy.deepcopy(f.default)
        elif f.default_factory is not MISSING:
            result[f.name] = f.default_factory()
    return result


class NativePolicy(nn.Module):
    def __init__(self, kind, horizon=50, denoise_steps=10):
        super().__init__()
        self.kind, self.horizon, self.denoise_steps = kind, horizon, denoise_steps
        self.weight_receipt = {}
        if kind == "dm05":
            from opendm.model.dm05.dm05_arch import DM05Config, DM05ForConditionalGeneration
            config = DM05Config.from_pretrained(str(FOUNDATION["dm05"]), local_files_only=True)
            # The checkpoint metadata requests FA2 during constructor execution.
            # Select supported native SDPA before instantiation. Torch 2.11's
            # default FlexAttention backward tile for Gemma head_dim=256 exceeds
            # RTX 5090's 99 KiB per-block shared-memory limit.
            config._attn_implementation = "eager"
            for sub in (config.vlm_config, config.vlm_config.text_config,
                        config.vlm_config.vision_config, config.action_config):
                sub._attn_implementation = "sdpa"
            config.precision_policy = "fp32_mixed"
            model, loading = DM05ForConditionalGeneration.from_pretrained(
                str(FOUNDATION["dm05"]), dtype=torch.float32, local_files_only=True,
                config=config, output_loading_info=True)
            assert not any(loading.get(k) for k in
                           ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")), loading
            old_dim = model.config.action_dim
            assert old_dim == 32
            # New embodiment: initialize both complete I/O projections. A
            # semantic equivalence between old 32D and paper EEF52 is not assumed.
            inp, out = model.model.action_in_proj, model.model.action_out_proj
            model.model.action_in_proj = nn.Linear(52, inp.out_features, bias=inp.bias is not None)
            model.model.action_out_proj = nn.Linear(out.in_features, 52, bias=out.bias is not None)
            model.config.action_dim = model.model.config.action_dim = 52
            model.config.chunk_size = model.model.config.chunk_size = horizon
            model.set_attention_implementation(llm_attn_implementation="sdpa",
                vision_attn_implementation="sdpa", action_attn_implementation="sdpa", bf16=True)
            model.set_precision_policy("fp32_mixed")
            model.enable_gradient_checkpointing(vlm_gradient_checkpointing=True, ae_gradient_checkpointing=True)
            model.freeze_vlm_embedding()  # Native OpenDM fine-tuning default.
            model.config.use_cache = False
            model.model.vlm.config.use_cache = False
            # Disable only the CUDA-graph optimization. The same native
            # independent-noise, ten-step Euler denoising loop is retained.
            model._can_use_suffix_graph = MethodType(no_suffix_cuda_graph, model)
            self.weight_receipt = {"pretrained": "Dexmal/DM05",
                "loading_info": {k: list(loading.get(k, [])) for k in
                                 ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")},
                "fresh_parameters": ["model.action_in_proj.*", "model.action_out_proj.*"],
                "fresh_action_dim": 52, "pretrained_action_dim": old_dim,
                "suffix_cuda_graph": False,
                "training_scope": "upstream default: VLM embedding frozen; remaining parameters trainable"}
        else:
            from lerobot.configs.policies import PreTrainedConfig
            from lingbotvla.models.vla.pi0.modeling_lingbot_vla import LingbotVlaPolicy
            from lingbotvla.models.module_utils import init_empty_weights
            from lingbotvla.utils.arguments import ModelArguments, TrainingArguments
            from safetensors.torch import load_file
            config = PreTrainedConfig.from_pretrained(str(FOUNDATION["lingbot"]), local_files_only=True)
            options = dataclass_defaults(ModelArguments) | dataclass_defaults(TrainingArguments)
            for key, value in options.items():
                setattr(config, key, value)
            config.max_action_dim = config.max_state_dim = 75
            config.action_dim = 52
            config.n_action_steps = config.chunk_size = horizon
            config.tokenizer_path = str(FOUNDATION["qwen"])
            config.vocab_size = 151936
            config.tokenizer_max_length = 192
            config.resize_imgs_with_padding = [224, 224]
            config.attention_implementation = "flex"
            config.freeze_vision_encoder = False
            config.train_expert_only = False
            config.use_cache = False
            config.num_steps = denoise_steps
            config.align_params = {}
            with init_empty_weights():
                model = LingbotVlaPolicy(config, config.tokenizer_path)
            state = load_file(str(FOUNDATION["lingbot"] / "model.safetensors"), device="cpu")
            model.load_state_dict(state, strict=True, assign=True)
            del state
            model.float()
            self.weight_receipt = {"pretrained": "robbyant/lingbot-vla-4b",
                "fresh_parameters": [], "strict_load": True,
                "max_action_dim": 75, "supervised_action_dim": 52,
                "training_scope": "all parameters trainable; no depth auxiliary model"}
        # Nested checkpoint configs may retain bf16 dtypes even when the
        # top-level from_pretrained requested fp32. FSDP master shards must
        # start with one uniform original dtype; autocast controls compute.
        model.float()
        assert {p.dtype for p in model.parameters()} == {torch.float32}
        self.native = model

    def forward(self, inputs, inference=False):
        if inference:
            assert not {"action", "actions", "action_mask", "action_is_pad"}.intersection(inputs)
            if self.kind == "dm05":
                return self.native.inference_action(**inputs, diffusion_steps=self.denoise_steps)
            old_cache = self.native.model.config.use_cache
            self.native.model.config.use_cache = True
            try:
                return self.native.model.sample_actions(
                    **inputs, vlm_causal=False, num_steps=self.denoise_steps)
            finally:
                self.native.model.config.use_cache = old_cache
        if self.kind == "dm05":
            return self.native(**inputs)["loss"]
        # Preserve upstream flow-matching objective, correcting only the
        # sample-validity reduction: upstream currently ignores action_is_pad.
        losses, depth_loss, _ = self.native.model(
            images=inputs["images"], img_masks=inputs["img_masks"],
            lang_tokens=inputs["lang_tokens"], lang_masks=inputs["lang_masks"],
            state=inputs["state"], actions=inputs["actions"],
            label=None, noise=None, time=None, vlm_causal=False,
            loss_type=self.native.config.loss_type, depth_targets=None, norm_qkv=False)
        assert isinstance(depth_loss, (int, float)) and depth_loss == 0
        valid = inputs["joint_mask"][:, None, :] & ~inputs["action_is_pad"][:, :, None]
        return (losses * valid).sum() / valid.sum().clamp_min(1)


def processor_for(kind, **kwargs):
    return DMProcessor(**kwargs) if kind == "dm05" else LingProcessor(**kwargs)


class NumericSpace:
    def __init__(self, kind):
        self.kind = kind
        self.stats = {k: np.asarray(v, np.float32) for k, v in
                      json.loads((NORM_DIR / "norm_stats.json").read_text())["norm_stats"]["action"].items()}

    def normalize(self, x):
        s = self.stats
        if self.kind == "dm05":
            out = (np.clip(x, s["q01"], s["q99"]) - s["q01"]) / (s["q99"] - s["q01"] + 1e-6) * 2 - 1
            return np.where((s["q01"] == 0) & (s["q99"] == 0), 0, out)
        return (x - s["mean"]) / (s["std"] + 1e-6)

    def denormalize(self, x):
        s = self.stats
        if self.kind == "dm05":
            out = (x + 1) / 2 * (s["q99"] - s["q01"] + 1e-6) + s["q01"]
            return np.where((s["q01"] == 0) & (s["q99"] == 0), 0, out)
        return x * (s["std"] + 1e-6) + s["mean"]
