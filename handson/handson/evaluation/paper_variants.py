"""Explicit source-cohort selection, including its language and reset contract.

No variant is selected from a policy outcome or sampled automatically. Omission
keeps the task's default source; historical cohorts must be requested by name.
"""
import hashlib
import json
from pathlib import Path


def load(task, variant=None, physics_backend="physx"):
    if variant is None:
        return None
    if (task, variant, physics_backend) != ("t06_fit_device_cover", "stabilize_body", "physx"):
        raise ValueError(f"Unsupported paper variant: {task}/{variant}/{physics_backend}")
    path = Path(__file__).with_name("device_cover_stabilize_body.json")
    data = path.read_bytes()
    row = json.loads(data)
    if row["task"] != task or row["variant"] != variant:
        raise ValueError("Paper variant identity mismatch")
    row["configuration_sha256"] = hashlib.sha256(data).hexdigest()
    return row


def sample_language(task, variant, values, seed):
    from ..tasks.language import choose, LANGUAGE_SEED
    row = load(task, variant)
    goal = row["language_goal"]
    if goal["values"] != values:
        raise ValueError("Paper variant and native goal disagree")
    bank = {"tasks": {task: {"goals": [goal]}}}
    return dict(choose(task, values, bank, seed, LANGUAGE_SEED),
                paper_variant=variant, configuration_sha256=row["configuration_sha256"],
                source_language_bank_sha256=row["source_language_bank_sha256"],
                constant_within_episode=True, heldout_in_production=False,
                historical_draw_reproduced=False)
