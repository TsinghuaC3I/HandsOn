# Contributing

Thanks for your interest in HandsOn.  This document covers the three things
people usually want to do.

## Reporting a problem

Open a GitHub issue.  Please include:

- the task name (`t12_load_and_start_microwave`, …) and the config you used
- your Isaac Sim / Isaac Lab version and CUDA driver version
- the full traceback, or the success-condition panel output for a
  behavioural mismatch
- the seed, if the problem is not reproducible

## Adding or changing a task

Tasks live in `handson/handson/tasks/<task_name>/`.  A task package must
provide a `config.py` defining a class named `TaskConfig`; the registry in
`handson/handson/tasks/mappings.py` imports it by name and expects:

- `task_name` — the directory name, used as the environment discriminator
- `task_id` — the manuscript identifier
- `task_prompt` — the natural-language instruction
- `proprio_keys` — the ordered proprioception keys for the policy
- `get_environment(...)` — builds the environment
- `process_demos(demo)` — demo post-processing

New tasks additionally need a `configs/multi_task/<task_name>.yaml` with the
prompt, camera mapping and robot type, and an entry in `mappings.py`.

## Task acceptance

`taskdev/` holds the fail-closed acceptance contract.  A task is accepted only
when its report passes all nine checks — `schema`, `assets`, `reachability`,
`contact`, `reset`, `success`, `camera`, `stability`, `collection` — with
measured evidence.  Start from `taskdev/templates/` and validate against
`taskdev/schema/task_manifest_v1.schema.yaml`.

## Development

From the repository root, install the CPU test dependencies and run the
maintained release suite, matching the CPU regression workflow:

```bash
python -m pip install -r requirements/cpu-test-lock.txt
python -m pip install --no-deps -e './handson[test]'
python -m pip check
mkdir -p outputs/test-tmp
TMPDIR="$PWD/outputs/test-tmp" python -m pytest handson/tests -q
```

Most tests construct environments without launching Isaac, so they run on CPU.
Tests that require Isaac are skipped without it.

`scripts/tests/` also contains historical collection-tool tests. Some reference
unreleased teleoperation, collection-UI and arm-planning modules, so that
directory is not a standalone release test suite. The public CPU suite does
not certify those private collection workflows or GPU rollout success.
