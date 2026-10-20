"""Spatial Forcing: native full-pytree load, including the alignment head."""
from types import SimpleNamespace
import numpy as np
from ..common import path_option, require, vendor


class Policy:
    def __init__(self, options):
        vendor("spatial")
        import flax.nnx as nnx
        import jax
        from openpi.models import model as model_api
        from openpi.models.pi0_align import Pi0Align
        from openpi.models.pi0_config import Pi0Config
        from openpi.shared import array_typing as at, nnx_utils
        from ..native.spatial_preprocessing import Preprocessor
        config = SimpleNamespace(model=Pi0Config(pi05=True, action_dim=52,
                                 action_horizon=50, max_token_len=512),
                                 vla_layers_align=12, vggt_dim=1024,
                                 vla_adapt_layers_align=None, vggt_adapt_layers_align=None,
                                 iSF_zscore=False, use_vlm_norm=True, align_loss_coeff=0.2)
        require(options.checkpoint.name == "90000", "Spatial Forcing expects the selected 90000 directory")
        model = nnx.eval_shape(lambda: Pi0Align(config, rngs=nnx.Rngs(0)))
        graph, state = nnx.split(model)
        params = model_api.restore_params(options.checkpoint / "params")
        at.check_pytree_equality(expected=state.to_pure_dict(), got=params,
                                check_shapes=True, check_dtypes=False)
        state.replace_by_pure_dict(params)
        model = nnx.merge(graph, state)
        model.eval()
        self.decode = nnx_utils.module_jit(model.sample_actions)
        self.prep = Preprocessor(path_option(options, "normalization"), path_option(options, "tokenizer"))
        self.model_api, self.jax = model_api, jax

    def predict(self, state, images, prompt, seed):
        raw = self.prep.observation(state, prompt, images)
        obs = self.model_api.Observation.from_dict(self.jax.tree.map(lambda x: np.expand_dims(x, 0), raw))
        noise = np.random.default_rng(seed).standard_normal((1, 50, 52)).astype(np.float32)
        result = np.asarray(self.jax.device_get(self.decode(
            self.jax.random.key(seed & 0xffffffff), obs, num_steps=10, noise=noise)))
        require(result.shape == (1, 50, 52) and np.isfinite(result).all(), "Invalid native Spatial prediction")
        return self.prep.denormalize(result[0])
