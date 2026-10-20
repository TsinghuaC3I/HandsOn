# Developing and accepting benchmark tasks

The task-development pipeline separates authoring from acceptance. CPU-only
commands create isolated worktrees and validate YAML. Isaac Sim 5.1 runners,
implemented separately, produce evidence. The final gate accepts measured
results only.

## 1. Create exactly one task worktree

Run from a clean checkout containing this pipeline:

```bash
python3 scripts/taskdev.py scaffold \
  --task-id example_task \
  --title "Example task" \
  --base-ref remote-task-dev-latest \
  --worktree-root "$TASKDEV_WORKTREE_ROOT"
```

The command refuses invalid IDs, existing branches, existing destinations,
unresolved base refs, and a filesystem-root destination. It creates branch
`taskdev/example_task`, then adds:

```text
taskdev/tasks/example_task/
├── WORKLOG.md
├── assets/
├── evidence/
├── task.yaml
└── validation_report.yaml
```

Use `--dry-run` to inspect the resolved branch, worktree, and base commit without
writing. Never point multiple task agents at the same branch/worktree.

If `git worktree add` returned non-zero only because a `post-checkout` hook was
unavailable, do not delete the worktree or branch. Resume it explicitly with the
exact commit already checked out:

```bash
python3 scripts/taskdev.py scaffold \
  --task-id example_task \
  --title "Example task" \
  --base-ref "$(git -C "$TASKDEV_WORKTREE_ROOT/example_task" rev-parse HEAD)" \
  --worktree-root "$TASKDEV_WORKTREE_ROOT" \
  --resume-existing
```

Resume is fail-closed: before writing templates it verifies Git registration,
the exact HEAD and symbolic branch, a clean worktree, and all three tracked
template files against the base commit.

## 2. Make the manifest truthful

Replace every `REPLACE_WITH_*` path. Asset paths and `validation.report` must be
repository-relative. Record task provenance: `adapted` and `handson` origins
require a source reference; `handson` also requires license provenance.

The robot contract is deliberately strict: real ROKAE kinematics and Wuji gen1
hands, one declared base pose per arm, explicit workspace targets, monitored and
forbidden contact pairs, reset invariants, temporal success hold, the robot head
camera, stability bounds, and mandatory collection streams.

Validate without launching the simulator:

```bash
python3 scripts/taskdev.py validate-manifest taskdev/tasks/example_task/task.yaml
```

Exit `0` means structurally valid; exit `1` is a validation rejection; exit `2`
is an input or dependency error. Output is one JSON object for automation.

## 3. Produce evidence in Isaac Sim 5.1

Before each GPU launch, the caller must apply the Jinshan cluster policy and
freshly inspect GPU processes, memory, utilization, and host memory. This
scaffold never performs placement and must not be used to bypass that check.

Each category needs a small immutable evidence file under the task's `evidence/`
directory plus measured metrics in `validation_report.yaml`:

| Check | Minimum meaning |
|---|---|
| schema | zero manifest/runtime contract errors |
| assets | no missing files; scale and collisions visually/programmatically verified |
| reachability | every target solved by ROKAE + Wuji gen1 within joint/error margins |
| contact | required contacts observed; penetration and forbidden events bounded |
| reset | enough deterministic trials with bounded failure fraction |
| success | positive and adversarial-negative trials; temporal hold verified |
| camera | robot **head** RGB frames, resolution/energy/black-frame metrics |
| stability | settled simulation duration with no NaNs or exploding bodies |
| collection | successful episodes with all required streams and no schema errors |

Populate each evidence entry with `path` and the current file `sha256`. Reports
from Isaac Sim 6.0/Kit 110 or a Python-server workflow are rejected: this branch
targets the repository's Isaac Sim 5.1 runtime.

## 4. Run the single release gate

```bash
python3 scripts/taskdev.py gate-report taskdev/tasks/example_task/task.yaml
```

The gate first revalidates the manifest, verifies its exact SHA-256 in the
report, hashes every evidence file, then recomputes all nine threshold decisions.
Every check must be `pass`; the gate has no warning-only or partial-pass mode.

Commit a task only after both commands exit `0`, then integrate that atomic task
commit into `remote-task-dev-latest` through the parent integration workflow.

## CPU tests

Only Python's standard library and PyYAML are required:

```bash
python3 -m unittest -v scripts.tests.test_taskdev_pipeline
```

Tests create temporary local Git repositories only; they never start Isaac Sim,
inspect or claim a GPU, or mutate an existing task implementation.
