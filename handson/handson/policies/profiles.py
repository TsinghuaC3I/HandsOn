"""Selected paper checkpoints and their closed-loop execution contracts."""
from dataclasses import dataclass
import hashlib
import importlib

from ..interfaces import Policy, ACTION_DIM, STATE_DIM, CONTROL_HZ


@dataclass(frozen=True)
class Profile:
    name: str
    step: int
    model_horizon: int
    action_horizon: int
    history_frames: int = 1
    history_stride: int = 0
    denoising_steps: int = 10
    adapter: str = ""
    asset_options: frozenset[str] = frozenset()
    task_splits: tuple[str, ...] = ("core", "transfer")
    noise_prefix: str = ""
    noise_task_name: bool = False
    noise_reduction: str = "none"

    def load(self, options) -> Policy:
        module = importlib.import_module(f"handson.policies.adapters.{self.adapter}")
        policy = module.Policy(options)
        if not isinstance(policy, Policy):
            raise TypeError(f"{self.adapter} does not implement the policy interface")
        return policy

    def supports(self, task):
        from ..tasks.registry import TASKS
        return task in TASKS and TASKS[task].split in self.task_splits

    @property
    def render_stride(self):
        return self.history_stride or self.action_horizon

    def metadata(self):
        return dict(name=self.name, step=self.step, model_horizon=self.model_horizon,
                    action_horizon=self.action_horizon, history_frames=self.history_frames,
                    history_stride=self.history_stride, denoising_steps=self.denoising_steps,
                    render_stride=self.render_stride, state_dim=STATE_DIM, action_dim=ACTION_DIM,
                    control_hz=CONTROL_HZ,
                    coordinate_frame="simulation_world", side_order=["right", "left"],
                    state_semantics="current_measured",
                    action_semantics="absolute_controller_command")


PROFILES = {
    "pi05": Profile("π0.5", 120000, 30, 30, adapter="pi05",
                    asset_options=frozenset({"normalization", "tokenizer"}),
                    noise_prefix="A-pi05-120000"),
    "dm05": Profile("DM0.5", 86000, 50, 30, adapter="vla",
                    asset_options=frozenset({"foundation", "normalization"}),
                    noise_prefix="dm05-86000", noise_reduction="mask63"),
    "hy": Profile("HY-Embodied-0.5-VLA", 125000, 50, 30, 6, 10, adapter="vla",
                  asset_options=frozenset({"foundation", "normalization"}),
                  noise_prefix="hy-125000", noise_reduction="mask63"),
    "lingbot": Profile("LingBot-VLA", 151000, 50, 30, adapter="vla",
                       asset_options=frozenset({"foundation", "normalization", "tokenizer"}),
                       noise_prefix="lingbot-151000", noise_reduction="mask63"),
    "openwam": Profile("OpenWAM", 70000, 32, 16, adapter="openwam",
                       asset_options=frozenset({"foundation", "normalization"}),
                       noise_prefix="openwam", noise_task_name=True, noise_reduction="mod63"),
    "fastwam": Profile("FastWAM", 100000, 32, 16, adapter="fastwam",
                       asset_options=frozenset({"normalization", "text_cache", "model_config"}),
                       noise_prefix="fastwam", noise_reduction="mod63"),
    "motus": Profile("Motus", 100000, 48, 16, adapter="motus",
                     asset_options=frozenset({"foundation", "text_cache", "model_config"}),
                     noise_prefix="motus", noise_reduction="mod63"),
    "spatial": Profile("Spatial Forcing", 90000, 50, 16, adapter="spatial",
                       asset_options=frozenset({"normalization", "tokenizer"}),
                       noise_prefix="spatial-84000"),  # Preserve the original RNG stream.
    "dp": Profile("Single-task DP", 30000, 32, 30, adapter="dp", task_splits=("core",),
                  noise_prefix="dp", noise_task_name=True, noise_reduction="mod63"),
}


def execution_settings(model):
    profile = PROFILES[model]
    return dict(action_horizon=profile.action_horizon, render_stride=profile.render_stride,
                history_frames=profile.history_frames,
                history_stride=profile.history_stride or profile.action_horizon)


def noise_seed(model, episode_seed, request, task=None):
    """Preserve the original servers' independent-noise streams."""
    profile = PROFILES[model]
    prefix = profile.noise_prefix
    if profile.noise_task_name:
        from ..tasks.registry import TASKS
        if task not in TASKS:
            raise ValueError("A public task identifier is required for this noise stream")
        prefix += ":" + TASKS[task].checkpoint_name
    value = int.from_bytes(hashlib.sha256(
        f"{prefix}:{episode_seed}:{request}".encode()).digest()[:8], "little")
    if profile.noise_reduction == "mask63":
        return value & (2**63 - 1)
    if profile.noise_reduction == "mod63":
        return value % (2**63 - 1)
    return value
