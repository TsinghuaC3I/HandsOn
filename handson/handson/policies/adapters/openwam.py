"""Native OpenWAM model and EEF52/Alpha80 boundary from the paper run."""
from ..common import asset, path_option, read, require, vendor


class Policy:
    def __init__(self, options):
        import os
        os.environ["WAM_ATTENTION_IMPL"] = "sdpa"
        os.environ["DIFFSYNTH_ATTENTION_IMPLEMENTATION"] = "torch"
        vendor("openwam")
        import torch
        from ..native import wam_models
        from ..native.wam_interfaces import PolicyNormalizer
        from openwam.model.video_backbone.wan.models import dit
        from openwam.model.action_backbone import components
        dit.FLASH_ATTN_3_AVAILABLE = dit.FLASH_ATTN_2_AVAILABLE = dit.SAGE_ATTN_AVAILABLE = False
        components._ATTENTION_FN = components._sdpa
        torch.cuda.set_device(0)
        wam_models.MODELS = path_option(options, "foundation")
        self.normalizer = PolicyNormalizer(read(path_option(
            options, "normalization", asset("openwam", "norm_stats.json"))), "openwam")
        self.adapter = wam_models.OpenWAMAdapter(read(asset("openwam", "run_config.json")))
        checkpoint = torch.load(options.checkpoint, map_location="cpu", mmap=True, weights_only=False)
        require(checkpoint["step"] == checkpoint["global_steps"] == 70000, "Wrong OpenWAM checkpoint step")
        self.adapter.load_state_dict(checkpoint["module"], strict=True)
        self.adapter.eval()

    def predict(self, state, images, prompt, seed):
        import torch
        from PIL import Image
        from ..native.wam_images import mosaic
        image = mosaic(*[Image.fromarray(images[k]).convert("RGB")
                         for k in ("front", "wrist_left", "wrist_right")])
        condition = dict(images=[image], state=self.normalizer.normalize(state, "state"),
                         prompt=prompt, model_mask=self.normalizer.valid_mask)
        with torch.inference_mode():
            output = self.adapter.decode_action(condition, steps=10, seed=int(seed))[0].float().cpu().numpy()
        require(output.shape == (32, 80), "Unexpected native OpenWAM action shape")
        return self.normalizer.denormalize(output, "action")
