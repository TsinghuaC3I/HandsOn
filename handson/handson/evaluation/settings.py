"""Paper evaluator's approved task settings and independent random streams."""
from fractions import Fraction
import hashlib,json,math,random
from pathlib import Path
from typing import Any
_PROTOCOL = "handson-approved-settings"
_ROOT = Path(__file__).parent
_SETTINGS_SHA256 = "beaf08faf21ccba553d8524c65e97399ab17ec6baa8436a5d1df969e07b2cc88"
def verify_receipt():
    if hashlib.sha256((_ROOT / "settings.json").read_bytes()).hexdigest() != _SETTINGS_SHA256:
        raise ValueError("Packaged task settings changed")
def task_ids():
    return tuple(sorted(json.loads((_ROOT / "settings.json").read_text())))
def task_spec(task):
    verify_receipt()
    return json.loads((_ROOT / "settings.json").read_text())[task]
def _profiles(spec):
    return spec["profiles"]
def derive_seed(master_seed: int, task: str, episode_index: int, stream: str) -> int:
    payload = json.dumps(
        [_PROTOCOL, int(master_seed), task, int(episode_index), stream],
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")

def select_setting(
    master_seed: int,
    task: str,
    episode_index: int,
    *,
    forced_setting: str | None = None,
) -> dict[str, Any]:
    """Select one complete setting before reset; dependent values stay bound."""
    verify_receipt()
    spec = task_spec(task)
    profiles = _profiles(spec)
    seeds = {
        name: derive_seed(master_seed, task, episode_index, name)
        for name in ("setting", "reset", "camera", "appearance", "policy")
    }
    weights = [
        Fraction(
            profile["selection_probability"]["numerator"],
            profile["selection_probability"]["denominator"],
        )
        for profile in profiles
    ]
    if sum(weights, Fraction()) != 1:
        raise ValueError(f"Selection probabilities do not sum to one for {task}")
    denominator = math.lcm(*(weight.denominator for weight in weights))
    counts = [int(weight * denominator) for weight in weights]
    draw = random.Random(seeds["setting"]).randrange(denominator)
    chosen = None
    cumulative = 0
    for profile, count in zip(profiles, counts):
        cumulative += count
        if draw < cumulative:
            chosen = profile
            break
    if forced_setting is not None:
        chosen = next((row for row in profiles if row["setting_id"] == forced_setting), None)
        if chosen is None:
            raise ValueError(f"Unknown B setting {task}/{forced_setting}")
    if chosen is None:  # pragma: no cover - guarded by probability validation
        raise RuntimeError(f"Could not select an approved B setting for {task}")
    return {
        "protocol": _PROTOCOL,
        "master_seed": int(master_seed),
        "task": task,
        "episode_index": int(episode_index),
        "short_id": spec["short_id"],
        "setting_id": chosen["setting_id"],
        "seeds": seeds,
        "setting_draw": draw,
        "setting_draw_denominator": denominator,
        "forced_for_validation": forced_setting is not None,
        "profile": chosen,
        "key_parameters": spec.get("key_parameters", {}),
        "effective_physics_target": spec["effective_physics_target"],
        "physics": spec["effective_physics_target"],
        "visual_target": spec.get("visual_target", {}),
        "camera_profile_id": spec.get("visual_target", {}).get("camera_profile_id"),
        "asset_receipt": spec.get("asset_receipt", {}),
    }

def runtime_contract(task: str) -> dict[str, Any]:
    """Return the narrow interface the common evaluator must honor."""
    spec = task_spec(task)
    physics = spec["effective_physics_target"]
    return {
        "task": task,
        "protocol": _PROTOCOL,
        "control_hz": round(1.0 / physics["control_dt"]),
        "physics_hz": round(1.0 / physics["physics_dt"]),
        "n_substeps": physics["n_substeps"],
        "setting_before_reset": True,
        "executor": "legacy_native_eef",
        "success_rule": "native_task_checker",
        "policy_future_state_input": False,
        "selected_profile_must_bind_all_dependent_values": True,
        "task_card": "settings.json#" + task,
    }
