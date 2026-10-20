"""Explicit render/recording cadence; never change policy or physics cadence."""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True)
class RenderSchedule:
    stride: int = 1
    horizon: int = 30
    control_hz: int = 30

    def __post_init__(self):
        if (isinstance(self.stride, bool) or not isinstance(self.stride, int)
                or self.stride < 1 or self.horizon % self.stride):
            raise ValueError("render_stride must be a positive divisor of action horizon")

    def capture(self, step):
        return step % self.stride == 0

    def receipt(self):
        return dict(protocol="native_rgb_every_step_v1" if self.stride == 1 else
                    "native_rgb_sampled_v1", render_stride=self.stride,
                    policy_horizon=self.horizon, control_hz=self.control_hz,
                    video_fps=self.control_hz / self.stride,
                    video="sampled live frames plus terminal; actual times in frames.jsonl",
                    terminal_frame_max_timestamp_rounding_s=self.stride / self.control_hz,
                    physics_or_action_changes=False, renderer_quality_changes=False)


@contextmanager
def observation_images(env, enabled):
    """Skip only native image observation, restoring the flag even on an error."""
    original = env.image_obs
    if not original:
        raise ValueError("Expected an image-observation environment")
    env.image_obs = enabled
    try:
        yield
    finally:
        env.image_obs = original


def load_schedule(job, code_directory):
    """Resolve only this episode's immutable code snapshot, never a live flag."""
    if "render_stride" in job:
        return RenderSchedule(job["render_stride"]), {"source": "JOB.json"}
    path = Path(code_directory) / "eval_protocol.json"
    if not path.exists():
        return RenderSchedule(), {"source": "legacy_default"}
    content = path.read_bytes()
    config = json.loads(content)
    assert config["version"] == 1
    return RenderSchedule(config["render_stride"]), dict(source=str(path),
        sha256=hashlib.sha256(content).hexdigest(), configuration=config)
