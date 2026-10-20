"""Native Hy-VLA math regrouped into callable FSDP/checkpoint layer units.

The upstream loop bypasses decoder-layer __call__, so wrapping those layers
alone would NOT activate FSDP hooks. Extract its exact layer body via AST,
preserving all routing/attention/residual math and cache semantics. Joint
layers own the original parameter objects; no weights are copied or frozen.
"""
import ast
import copy
import inspect
import textwrap
import types
from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers.modeling_utils import no_init_weights
from safetensors.torch import load_model

from hy_vla import HyVLA, HyVLAConfig
from hy_vla.configuration_hy_vla import PolicyFeature, FeatureType
import hy_vla.modeling_dual_tower as native
FOUNDATION = None


class JointBlock(nn.Module):
    def __init__(self, layers, configs, tower_config):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.model_configs = configs
        self.config = tower_config
        self.use_checkpoint = False

    eager_attention_forward = native.HyDualTower.eager_attention_forward
    flash_attention_forward = native.HyDualTower.flash_attention_forward
    get_attention_interface = native.HyDualTower.get_attention_interface

    def forward(self, *args):
        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            return checkpoint(self.native_body, *args, use_reentrant=False)
        return self.native_body(*args)


class LayerRefs(ast.NodeTransformer):
    def visit_Subscript(self, node):
        # models[i].layers[layer_idx] -> self.layers[i]
        if (isinstance(node.value, ast.Attribute) and node.value.attr == "layers"
                and isinstance(node.value.value, ast.Subscript)
                and isinstance(node.value.value.value, ast.Name)
                and node.value.value.value.id == "models"):
            index = node.value.value.slice
            return ast.Subscript(ast.Attribute(ast.Name("self", ast.Load()), "layers", ast.Load()),
                                 index, ast.Load())
        return self.generic_visit(node)

    def visit_Attribute(self, node):
        if (node.attr == "config" and isinstance(node.value, ast.Subscript)
                and isinstance(node.value.value, ast.Name) and node.value.value.id == "models"):
            return ast.Subscript(ast.Attribute(ast.Name("self", ast.Load()), "model_configs", ast.Load()),
                                 node.value.slice, ast.Load())
        return self.generic_visit(node)


def install_joint_blocks(tower):
    source = ast.parse(textwrap.dedent(inspect.getsource(native.HyDualTower.forward)))
    method = source.body[0]
    loop = next(n for n in method.body if isinstance(n, ast.For)
                and isinstance(n.target, ast.Name) and n.target.id == "layer_idx")
    args = ("inputs_embeds, modality_masks, attention_mask, position_embeddings, "
            "past_key_values, use_cache, fill_kv_cache, layer_idx, batch_size")
    # The two append-only diagnostics are reconstructed in the outer loop.
    inner = ast.parse(f"def native_body(self, {args}):\n    pass").body[0]
    inner.body = copy.deepcopy(loop.body[:-2])
    # Last statements in upstream: append prefix output; set inputs_embeds.
    assert ast.unparse(loop.body[-2]).startswith("prefix_emb_layer_outputs.append")
    assert ast.unparse(loop.body[-1]) == "inputs_embeds = outputs_embeds"
    inner.body = [n for n in inner.body if not (
        isinstance(n, ast.Expr) and ast.unparse(n).startswith("att_vis_output.append"))]
    inner.body.append(ast.parse("return outputs_embeds, past_key_values, probs").body[0])
    inner = LayerRefs().visit(inner)
    generated = ast.fix_missing_locations(ast.Module(body=[inner], type_ignores=[]))
    scope = dict(vars(native))
    exec(compile(generated, "<native-hy-joint-layer>", "exec"), scope)
    JointBlock.native_body = scope["native_body"]
    loop.body = ast.parse(
        f"inputs_embeds, past_key_values, probs = self.joint_blocks[layer_idx]({args})\n"
        "att_vis_output.append(probs)\n"
        "prefix_emb_layer_outputs.append(inputs_embeds[0])").body
    scope = dict(vars(native))
    exec(compile(ast.fix_missing_locations(source), "<native-hy-joint-forward>", "exec"), scope)
    models = [tower.vlm.language_model.model, tower.expert.model]
    old_params = {id(p) for p in tower.parameters()}
    originals = [list(m.layers) for m in models]
    tower.joint_blocks = nn.ModuleList([
        JointBlock([originals[0][i], originals[1][i]], [m.config for m in models], tower.config)
        for i in range(len(originals[0]))])
    for m in models:
        del m.layers
        # Native outer forward only reads norms/rotary modules after extraction.
    tower.forward = types.MethodType(scope["forward"], tower)
    assert old_params == {id(p) for p in tower.parameters()}, "Parameter coverage changed"
    return {
        "generated_joint_body": ast.unparse(generated),
        "generated_outer_forward": ast.unparse(source),
        "parameter_identity_preserved": True,
    }


def checkpoint_vision_blocks(model):
    """Preserve SpaceTimeBlock isinstance dispatch, unlike a wrapper class."""
    for block in model.model.dual_tower.vlm.visual.vision_tower.blocks:
        original = block.forward

        def forward(self, *args, _original=original, **kwargs):
            if self.training and torch.is_grad_enabled():
                return checkpoint(_original, *args, use_reentrant=False, **kwargs)
            return _original(*args, **kwargs)

        block.forward = types.MethodType(forward, block)


class HunyuanPolicy(nn.Module):
    def __init__(self, history=6, regroup=True):
        super().__init__()
        path = str(FOUNDATION)
        config = HyVLAConfig.from_pretrained(path, local_files_only=True)
        # Skip randomizing soon-to-be-overwritten parameters, not buffers.
        with no_init_weights():
            self.policy = HyVLA(config)
        # no_init_weights also skips HF's post-init tying. Restore the released
        # config's actual shared text embedding BEFORE loading safetensors.
        lm = self.policy.model.dual_tower.vlm.language_model
        assert lm.config.tie_word_embeddings
        lm.tie_weights()
        assert lm.lm_head.weight is lm.model.embed_tokens.weight
        missing, unexpected = load_model(self.policy, str(Path(path) / "model.safetensors"),
                                        strict=False, device="cpu")
        # The official release omits exactly this never-called token LM head.
        # It is NOT the action output projection. Never ignore arbitrary keys
        # or leave uninitialized no_init_weights storage in a saved checkpoint.
        expected_missing = {"model.dual_tower.expert.lm_head.weight"}
        assert set(missing) == expected_missing and not unexpected, (missing, unexpected)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(271828)
            nn.init.normal_(self.policy.model.dual_tower.expert.lm_head.weight, std=0.02)
        self.weight_receipt = {
            "missing_keys_exactly": sorted(expected_missing), "unexpected_keys": [],
            "missing_key_reason": "Official checkpoint omits unused expert token LM head; initialized seed271828.",
            "vlm_embedding_tied_to_released_lm_head": True,
            "all_action_path_pretrained_weights_loaded": True,
        }
        # UMI has 32 padded dims with delta-pose/parallel gripper semantics.
        # New absolute EEF52 + 20 hand joints/side needs new complete I/O maps.
        core = self.policy.model
        config.max_state_dim = config.max_action_dim = 64
        config.action_feature = PolicyFeature(FeatureType.ACTION, (52,))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(314159)
            core.state_proj = nn.Linear(64, config.proj_width)
            core.action_in_proj = nn.Linear(64, config.proj_width)
            core.action_out_proj = nn.Linear(config.proj_width, 64)
        config.tokenizer_max_length = 192
        config.use_video_encoder = history > 1
        self.policy.enable_video_encoder_if_needed()
        self.policy.float()  # FP32 master parameters; BF16 gathered compute.
        self.denoise_steps = config.num_steps
        self.regroup_receipt = install_joint_blocks(core.dual_tower) if regroup else {}
        self.history = history

    def forward(self, batch, inference=False, noise=None, time=None):
        if inference:
            assert not {"action", "action_is_pad", "target_physical"}.intersection(batch)
            return self.policy.forward_evaluate(batch)["pred"]
        p = self.policy
        images, masks = p.prepare_images(batch)
        tokens, token_masks, _ = p.prepare_language(batch)
        losses, ntp = p.model(
            images, masks, tokens, token_masks, p.prepare_state(batch), p.prepare_action(batch),
            noise, time, None)
        assert ntp is None
        valid = (~batch["action_is_pad"]).unsqueeze(-1)
        # Same native flow objective, excluding padded dimensions/tail labels.
        losses = losses[..., :52]
        per_sample = (losses * valid).sum((1, 2)) / (valid.sum((1, 2)) * 52).clamp_min(1)
        return per_sample.mean()
