"""π0.5: exact paper architecture, transforms and checkpoint-owned statistics."""
import numpy as np
from ..common import path_option, read, require, vendor


class Policy:
    def __init__(self, options):
        vendor("pi05")
        import jax.numpy as jnp
        from openpi.models import model as model_api
        from openpi.models.pi0_config import Pi0Config
        from openpi.models.tokenizer import PaligemmaTokenizer
        from openpi.policies.policy import Policy as NativePolicy
        from openpi.policies.dual_arm_policy import DualArmInputs, DualArmOutputs
        from openpi import transforms as t
        from openpi.shared import download, normalize
        norm_path = path_option(options, "normalization")
        tokenizer = path_option(options, "tokenizer")
        # Only the local tokenizer lookup changes; tokenization itself is native.
        original = download.maybe_download
        def local_tokenizer(url, **kwargs):
            if str(url).endswith("paligemma_tokenizer.model"):
                return tokenizer
            return original(url, **kwargs)
        download.maybe_download = local_tokenizer
        try:
            tokens = PaligemmaTokenizer(320)
        finally:
            download.maybe_download = original
        config = Pi0Config(pi05=True, action_dim=52, action_horizon=30,
                           paligemma_variant="gemma_2b", action_expert_variant="gemma_300m",
                           max_token_len=320)
        require(options.checkpoint.name == "120000", "π0.5 expects the selected 120000 directory")
        model = config.load(model_api.restore_params(options.checkpoint / "params", dtype=jnp.bfloat16))
        raw = read(norm_path)["norm_stats"]
        stats = {k: normalize.NormStats(**{n: np.asarray(v) for n, v in values.items()})
                 for k, values in raw.items()}
        require(set(stats) == {"state", "actions"}, "Expected checkpoint-owned state/actions statistics")
        self.policy = NativePolicy(model, transforms=[
            DualArmInputs(model_type=config.model_type, state_dim=52),
            t.Normalize(stats, use_quantiles=True),
            t.ClipNormalized(keys=("state", "actions"), clip_value=3.0),
            t.InjectDefaultPrompt(None), t.ResizeImages(224, 224),
            t.TokenizePrompt(tokens, discrete_state_input=config.discrete_state_input),
            t.PadStatesAndActions(52),
        ], output_transforms=[
            t.ClipNormalized(keys=("actions",), clip_value=3.0),
            t.Unnormalize(stats, use_quantiles=True), DualArmOutputs(action_dim=52),
        ], sample_kwargs={"num_steps": 10})

    def predict(self, state, images, prompt, seed):
        observation = dict(state=state, prompt=prompt, base=images["front"],
                           wrist_left=images["wrist_left"], wrist_right=images["wrist_right"])
        noise = np.random.default_rng(seed).standard_normal((30, 52)).astype(np.float32)
        return self.policy.infer(observation, noise=noise)["actions"]
