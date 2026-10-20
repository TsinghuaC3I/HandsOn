"""Single-GPU inference with the exact paper training classes and checkpoints.

Training ran FSDP2 with MixedPrecisionPolicy(param_dtype=bf16): every
parameter was all-gathered as bf16, buffers stayed fp32, each wrapped unit
cast floating forward inputs to bf16 (cast_forward_inputs default) and the
root did not (cast_forward_inputs=False); validate() called the model under
bf16 autocast. Here the same model class is built, the model-only DCP state is
loaded strictly, parameters (not buffers) are cast once to bf16, the same unit
set gets FSDP2's own input-cast function as a pre-hook, and LingBot blocks are
checkpoint-wrapped and torch.compile'd as in the optimized training recipe.
heldout.py measures this against the logged held-out validation.
"""
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

SPECS = {
 "hy": dict(step=125000, horizon=50, history=6, history_stride=10),
 "dm05": dict(step=86000, horizon=50, history=1, history_stride=0,
              units=("PatchedGemma3DecoderLayer", "SiglipEncoderLayer", "DM05ActionExpert")),
 "lingbot": dict(step=151000, horizon=50, history=1, history_stride=0,
              units=("Qwen2DecoderLayer", "Qwen2_5_VLDecoderLayer", "Qwen2_5_VLVisionBlock")),
}
VIEWS = ("front", "wrist_left", "wrist_right")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def _fsdp_cast():
    try:
        from torch.distributed.fsdp._fully_shard._fsdp_common import _cast_fp_tensor
    except ImportError:
        from torch.distributed._composable.fsdp._fsdp_common import _cast_fp_tensor
    from torch.distributed.utils import _apply_to_tensors
    import functools
    cast = functools.partial(_cast_fp_tensor, torch.bfloat16)

    def hook(module, args, kwargs):
        return _apply_to_tensors(cast, args), _apply_to_tensors(cast, kwargs)
    return hook


def build_model(kind):
    """Exact training class on CPU in fp32, plus the modules FSDP wrapped."""
    if kind == "hy":
        from .hy_runtime import HunyuanPolicy, checkpoint_vision_blocks
        model = HunyuanPolicy()
        checkpoint_vision_blocks(model.policy)
        tower = model.policy.model.dual_tower
        units = [f"policy.model.dual_tower.joint_blocks.{i}" for i in range(len(tower.joint_blocks))]
        units += [f"policy.model.dual_tower.vlm.visual.vision_tower.blocks.{i}"
                  for i in range(len(tower.vlm.visual.vision_tower.blocks))]
    else:
        # Both formal runs used --optimized (checked against trainer.json).
        from .vla_optimized import OptimizedPolicy
        model = OptimizedPolicy(kind, SPECS[kind]["horizon"], 10)
        units = [n for n, m in model.named_modules() if type(m).__name__ in SPECS[kind]["units"]]
    assert units
    return model, units


def load_checkpoint(model, checkpoint):
    """Strict model-only DCP load; only tied aliases may be absent from disk."""
    import torch.distributed.checkpoint as dcp
    assert not (torch.distributed.is_available() and torch.distributed.is_initialized())
    reader = dcp.FileSystemReader(str(Path(checkpoint) / "state"))
    saved = {k[len("model."):] for k in reader.read_metadata().state_dict_metadata
             if k.startswith("model.")}
    local = model.state_dict()
    unexpected = sorted(saved - set(local))
    assert not unexpected, unexpected[:10]
    storage = {}
    for name, tensor in local.items():
        storage.setdefault((tensor.untyped_storage().data_ptr(), tensor.storage_offset(),
                            tuple(tensor.shape)), []).append(name)
    aliases = []
    for name in sorted(set(local) - saved):
        group = storage[(local[name].untyped_storage().data_ptr(), local[name].storage_offset(),
                         tuple(local[name].shape))]
        assert any(g in saved for g in group), f"not in checkpoint and not tied: {name}"
        aliases.append(name)
    request = {k: v for k, v in local.items() if k in saved}
    before = {k: request[k].flatten()[:64].clone() for k in list(request)[:: max(1, len(request) // 32)]}
    started = time.monotonic()
    dcp.load({"model": request}, storage_reader=reader)
    changed = sum(not torch.equal(v, request[k].flatten()[:64]) for k, v in before.items())
    return dict(loaded_tensors=len(request), tied_aliases=aliases,
                load_seconds=time.monotonic() - started,
                sampled_tensors_changed=f"{changed}/{len(before)}")


def emulate_fsdp_bf16(model, units, kind, compile_blocks):
    hook = _fsdp_cast()
    for p in model.parameters():  # parameters() dedups tied weights
        p.data = p.data.to(torch.bfloat16)
    for name in units:
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        module = model.get_submodule(name)
        if kind == "lingbot":
            # Same order as training: checkpoint_wrapper -> fully_shard -> compile.
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
            module = checkpoint_wrapper(module)
        module.register_forward_pre_hook(hook, prepend=True, with_kwargs=True)
        if kind == "lingbot" and compile_blocks:
            module = torch.compile(module, fullgraph=False, dynamic=False)
        if kind == "lingbot":
            setattr(parent, child, module)
    buffers = {str(b.dtype) for b in model.buffers() if b.is_floating_point()}
    params = {str(p.dtype) for p in model.parameters()}
    assert params == {"torch.bfloat16"}, params
    return dict(parameter_dtypes=sorted(params), floating_buffer_dtypes=sorted(buffers),
                cast_units=len(units), compile_blocks=bool(kind == "lingbot" and compile_blocks))


class Engine:
    def __init__(self, kind, checkpoint=None, step=None, device=None, compile_blocks=True):
        spec = SPECS[kind]
        self.kind, self.spec = kind, spec
        self.checkpoint = Path(checkpoint)
        self.step = int(step or spec["step"])
        info = json.loads((self.checkpoint / "trainer.json").read_text())
        assert info["complete"] is True and int(info["step"]) == self.step, (info.get("complete"), info.get("step"))
        if kind != "hy":
            assert info["config"].get("optimized") is True
            assert info["config"]["horizon"] == 50
        started = time.monotonic()
        torch.manual_seed(0)  # construction RNG is overwritten by the strict load
        self.model, units = build_model(kind)
        self.receipt = dict(kind=kind, step=self.step, checkpoint=str(self.checkpoint),
                            build_seconds=time.monotonic() - started)
        self.receipt.update(load_checkpoint(self.model, self.checkpoint))
        self.receipt.update(emulate_fsdp_bf16(self.model, units, kind, compile_blocks))
        self.device = device or torch.device("cuda", 0)
        self.model.to(self.device).eval()
        self.denoise_steps = self.model.denoise_steps
        assert self.denoise_steps == 10
        if kind == "hy":
            from . import hy_data
            self.space = hy_data.NumericSpace()
            self.data = hy_data
        else:
            from .vla_policies import NumericSpace, processor_for
            self.space = NumericSpace(kind)
            self.processor = processor_for(kind, training=False, horizon=spec["horizon"])
        self.receipt["ready_seconds"] = time.monotonic() - started

    def inputs(self, state, views, prompt):
        """Raw uint8 views + measured EEF52 state -> the validate() tensors.

        views[v] is (H,W,3) uint8, or for HY (6,H,W,3) oldest..current at
        stride 10, first frame repeated when history is unavailable."""
        state = np.asarray(state, np.float32)
        assert state.shape == (52,) and np.isfinite(state).all()
        if self.kind == "hy":
            batch = {}
            for view, key in zip(VIEWS, ("top_head", "hand_left", "hand_right")):
                frames = np.asarray(views[view])
                assert frames.dtype == np.uint8 and frames.ndim == 4 and frames.shape[0] == 6
                batch["observation.images." + key] = torch.from_numpy(
                    np.ascontiguousarray(frames)).permute(0, 3, 1, 2)[None]
            batch["observation.state"] = torch.from_numpy(self.space.normalize(state, "state"))[None]
            batch["task"] = [prompt]
            return self.data.model_inputs(batch, self.device, inference=True)
        from PIL import Image
        sample = dict(state=state, prompt=prompt,
                      images=[Image.fromarray(np.ascontiguousarray(views[v])) for v in VIEWS])
        return _to_device(self.processor(sample, inference=True), self.device)

    @torch.no_grad()
    def predict_normalized(self, inputs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.model(inputs, inference=True)
        out = out[0] if self.kind == "hy" else out[0, :, :52]
        return out.float().cpu().numpy()

    def predict(self, state, views, prompt, noise_seed, execute=30):
        # Same order as validate(): seed both generators, build inputs, sample.
        torch.manual_seed(noise_seed)
        torch.cuda.manual_seed(noise_seed)
        pred = self.predict_normalized(self.inputs(state, views, prompt))
        assert pred.shape == (self.spec["horizon"], 52) and np.isfinite(pred).all()
        return np.asarray(self.space.denormalize(pred), np.float32)[:execute], pred


def _to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {k: _to_device(v, device) for k, v in value.items()}
    return value


def noise_seed(kind, step, episode_seed, request):
    digest = hashlib.sha256(f"{kind}-{step}:{episode_seed}:{request}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & (2**63 - 1)
