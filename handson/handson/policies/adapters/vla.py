"""Original HY/DM/LingBot DCP loader and training-precision emulation."""
from ..common import asset, path_option, vendor


class Policy:
    def __init__(self, options):
        kind = options.model
        vendor(kind)
        foundation = path_option(options, "foundation")
        if kind == "hy":
            from ..native import hy_data, hy_runtime
            hy_runtime.FOUNDATION = foundation
            hy_data.NORM_FILE = path_option(options, "normalization", asset(kind, "norm_stats.json"))
        else:
            from ..native import vla_policies
            vla_policies.FOUNDATION = {kind: foundation}
            if kind == "lingbot":
                vla_policies.FOUNDATION["qwen"] = path_option(options, "tokenizer")
            vla_policies.NORM_DIR = path_option(options, "normalization", asset(kind, "."))
            vla_policies.FEATURE_CONFIG = asset("lingbot", "eef52.yaml")
        from ..native.vla_engine import Engine
        from ..profiles import PROFILES
        self.engine = Engine(kind, checkpoint=options.checkpoint, step=PROFILES[kind].step)
        self.receipt = self.engine.receipt

    def predict(self, state, images, prompt, seed):
        actions, _ = self.engine.predict(state, images, prompt, seed, execute=50)
        return actions
