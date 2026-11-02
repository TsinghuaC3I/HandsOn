"""Bind the task catalog's native goal to its frozen language bank."""
import json
from .contracts import require
from ..tasks.registry import TASKS, ASSETS


def bind(env, task, seed=0, paper_variant=None):
    require(env.env_step == 0, "Bind the instruction at the recording origin")
    require(getattr(env, "paper_variant", None) == paper_variant,
            "Environment and instruction source cohorts disagree")
    entries = json.loads((ASSETS / "goals.json").read_text())[task]
    values, readback = TASKS[task].goal(env)
    if paper_variant is not None:
        from .paper_variants import sample_language
        language = sample_language(task, paper_variant, values, seed)
        return dict(language, goal=values, native_goal_readback=readback,
                    source="source-cohort language bank bound to native seeded reset",
                    native_goals_mutated=False)
    matches = [x for x in entries if x["values"] == values]
    require(len(matches) == 1, f"Unsupported native goal: {task}/{values}")
    from ..tasks.language import sample
    language = sample(task, values, seed)
    require(language['canonical_prompt'] == matches[0]['canonical'],
            'Goal and language bank disagree')
    return dict(language, goal=values, native_goal_readback=readback,
                source="paper language bank bound to native seeded reset",
                native_goals_mutated=False)
