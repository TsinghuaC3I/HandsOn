"""Score one checkpoint's Transfer-16 control-step logs, using the Table 6 rules."""
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from statistics import fmean


def load_tasks():
    tasks = {}
    with Path(__file__).with_name("stages.tsv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            task = tasks.setdefault(row["task"], {
                "ids": {row["task"], row["release_id"], row["historical_id"]},
                "limit": int(row["limit"]), "stages": [],
            })
            task["stages"].append(row["predicate"])
    return tasks


TASKS = load_tasks()
NAMES = {alias: name for name, task in TASKS.items() for alias in task["ids"]}


@lru_cache(maxsize=None)
def parse_predicate(expression):
    """A disjunction of conjunctions. An atom is a flag, negated flag or lower bound."""
    alternatives = []
    for branch in expression.split(" | "):
        terms = []
        for atom in branch.split(" & "):
            key, op, threshold = atom.partition(" >= ")
            terms.append((key.lstrip("!").split("."), key.startswith("!"),
                          float(threshold) if op else None))
        alternatives.append(terms)
    return alternatives


def term_value(info, path, negate, lower):
    value = info
    for key in path:
        value = value[key]
    if lower is None:
        if type(value) is not bool:
            raise ValueError(f"Expected boolean at {'.'.join(path)}")
        return value != negate
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"Expected finite number at {'.'.join(path)}")
    return value >= lower


def evaluate(rule, info):
    # Conditions joined by & or | are evaluated on the same control step.
    return any(all(term_value(info, *term) for term in branch) for branch in rule)


def prefix_length(name, hits):
    if name == "Comb06":
        # Salt and pepper can be achieved in either order.
        hits = [any(hits[:2]), all(hits[:2]), *hits[2:]]
    return next((i for i, hit in enumerate(hits) if not hit), len(hits))


def score_episode(name, folder, seed=None):
    spec = TASKS[name]
    result = json.loads((folder / "RESULT.json").read_text())
    if result["task"] not in spec["ids"] or (seed is not None and result["seed"] != seed):
        raise ValueError("Receipt task or seed disagrees with the manifest")
    if result.get("complete_evaluation_episode") is not True or result.get("error") is not None:
        raise ValueError("Scored episode has an incomplete or errored receipt")
    if type(result["success"]) is not bool:
        raise ValueError("Receipt has no boolean success")
    rules = [parse_predicate(p) for p in spec["stages"]]
    hits = [False] * len(rules)
    count = 0
    last_success = None
    with (folder / "steps.jsonl").open() as trace:
        for count, line in enumerate(trace, 1):
            row = json.loads(line)
            if row["step"] != count:
                raise ValueError(f"Expected control step {count}, got {row['step']}")
            info = row["native"]["info"]
            last_success = info["succeed"]
            if type(last_success) is not bool:
                raise ValueError(f"Expected native boolean succeed at step {count}")
            if count <= spec["limit"]:
                for i, rule in enumerate(rules):
                    if not hits[i]:
                        try:
                            hits[i] = evaluate(rule, info)
                        except (KeyError, TypeError, ValueError) as exc:
                            raise ValueError(f"Step {count}, predicate {spec['stages'][i]}: {exc}") from exc
    if not count or count != result["steps"] or last_success != result["success"]:
        raise ValueError("Trace length or final native success disagrees with receipt")
    level = None if name == "Gen02" else prefix_length(name, hits)
    return {"scored_steps": min(count, spec["limit"]), "success": hits[-1],
            "L": level, "K": None if level is None else len(hits),
            "progress": None if level is None else level / len(hits)}


def aggregate(episodes, skipped):
    per_task = {}
    for name in TASKS:
        scored = episodes[name]
        graded = [e["progress"] for e in scored if e["progress"] is not None]
        per_task[name] = {"valid": len(scored), "graded": len(graded), **dict(skipped[name]),
                          "SR_pct": 100 * fmean(e["success"] for e in scored) if scored else None,
                          "Prog_pct": 100 * fmean(graded) if graded else None}
    groups = {}
    for group in ("Gen", "Comb", "Flex", "All"):
        tasks = [k for k in TASKS if group == "All" or k.startswith(group)]
        scores = {}
        for metric in ("SR_pct", "Prog_pct"):
            values = [per_task[t][metric] for t in tasks if metric == "SR_pct" or t != "Gen02"]
            scores[metric] = fmean(values) if None not in values else None
        groups[group] = scores
    return per_task, groups


def score_manifest(path):
    by_task = defaultdict(list)
    skipped = defaultdict(Counter)
    seen = set()
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            name = NAMES[row["task"]]
            trial = row["trial"]
            if type(trial) is not int or trial < 0:
                raise ValueError("Expected a nonnegative integer trial index")
            key = (name, trial)
            if key in seen:
                raise ValueError(f"Duplicate planned slot {key}")
            seen.add(key)
            status = row["status"]
            if status == "scored":
                result = score_episode(name, path.parent / row["episode_dir"], row.get("seed"))
                by_task[name].append({"trial": trial, **result})
            elif status in ("invalid", "missing"):
                skipped[name][status] += 1
            else:
                raise ValueError(f"Unknown slot status {status!r}")
        except (KeyError, OSError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}, line {number}: {exc}") from exc
    per_task, groups = aggregate(by_task, skipped)
    return {"episodes": {k: v for k, v in by_task.items() if v},
            "per_task": per_task, "groups": groups}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(score_manifest(args.manifest), indent=2, allow_nan=False))
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
