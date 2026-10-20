"""Throughput helpers qualified on paper without changing the model objective."""
import torch

from .vla_policies import NativePolicy, processor_for
from .vla_attention import CachedFlexAttention


class TrimPaddingCollator:
    def __init__(self, original):
        self.original = original

    def __call__(self, items):
        batch = self.original(items)
        occupied = batch["attention_mask"].bool().any(0)
        last = int(occupied.nonzero()[-1]) + 1
        length = min(batch["input_ids"].shape[1], ((last + 63) // 64) * 64)
        assert not batch["attention_mask"][:, length:].any()
        for key in ("input_ids", "attention_mask", "token_type_ids", "history_mask"):
            if key in batch:
                batch[key] = batch[key][:, :length].contiguous()
        return batch


def optimized_processor(kind, training=False, horizon=50):
    processor = processor_for(kind, training=training, horizon=horizon)
    if kind == "dm05" and training:
        processor.collate = TrimPaddingCollator(processor.collate)
    return processor


class OptimizedPolicy(NativePolicy):
    def __init__(self, kind, horizon=50, denoise_steps=10):
        super().__init__(kind, horizon, denoise_steps)
        if kind == "lingbot":
            self.attention_cache = CachedFlexAttention()
            self.native.model.qwenvl_with_expert.attention_interface = self.attention_cache

    def forward(self, inputs, inference=False):
        if self.kind == "lingbot":
            self.attention_cache.clear()
            shape = tuple(inputs["images"].shape)
            if getattr(self, "_image_input_shape", None) != shape:
                core = self.native.model.qwenvl_with_expert
                for key in ("rotary_pos_emb", "window_index", "cu_window_seqlens", "cu_seqlens"):
                    setattr(core, key, None)
                self._image_input_shape = shape
        if self.kind == "dm05" or inference:
            return super().forward(inputs, inference=inference)
        losses, depth_loss, _ = self.native.model(
            images=inputs["images"], img_masks=inputs["img_masks"],
            lang_tokens=inputs["lang_tokens"], lang_masks=inputs["lang_masks"],
            state=inputs["state"], actions=inputs["actions"],
            label=None, noise=None, time=None, vlm_causal=False,
            loss_type=self.native.config.loss_type, depth_targets=None, norm_qkv=False)
        assert isinstance(depth_loss, (int, float)) and depth_loss == 0
        valid = inputs["joint_mask"][:, None, :] & ~inputs["action_is_pad"][:, :, None]
        # Same objective as averaging four original B=1 microbatches, even when
        # their valid chunk lengths differ near episode ends.
        return ((losses * valid).sum((1, 2)) / valid.sum((1, 2)).clamp_min(1)).mean()
