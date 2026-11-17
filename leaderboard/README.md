# Submitting to the HandsOn leaderboard

The [HandsOn leaderboard](../docs/leaderboard.md) accepts results from any team.
Each submission follows the frozen evaluation protocol **`handson-eval-v1`** and
is sent as a pull request. Each entry carries one of three statuses:

| Status | Meaning |
| --- | --- |
| **HandsOn team** | Evaluated by the maintainers (paper models, GPT-6, Fable 5.2). |
| **Verified** | Submitted by its authors on the public trials. The maintainers then reran the released system on hidden seeds and the two results agree. |
| **Self-reported** | Submitted by its authors and checked by CI, but not yet verified. Shown separately from ranked entries. |

## Tracks

| Track | System | Planned trials | Score |
| --- | --- | --- | --- |
| `multi_task` | One checkpoint and one inference procedure for all 66 tasks | Core-50 × 100, Transfer-16 × 50 (5,800) | Core-50, Transfer-16, Overall |
| `single_task` | One checkpoint per Core-50 task | Core-50 × 100 (5,000) | Core-50 only |
| `zero_shot` | No HandsOn demonstrations or finetuning (e.g. prompted foundation-model agents) | 66 tasks × 10 (660) | Core-50, Transfer-16, Overall |

Core-50 and Transfer-16 are task-macro success rates. Overall is the 66-task
macro mean, `(50 × Core + 16 × Transfer) / 66`. Trial counts match the
paper protocol, so `multi_task` entries can be compared with the paper models.
Planned trials without a scored outcome count as failures.

## Protocol `handson-eval-v1`

- **Evaluator.** `python -m handson.evaluation`, run from a clean git checkout
  of this repository. It uses PhysX, 30 Hz actions, the packaged 3×P75 time
  budgets, the native task checkers and the three-camera EEF52 interface from
  [the policy contract](../docs/custom_policy_integration.md).
  - The suite records the git commit and the hashes of the files that define
    seeds, settings, language, budgets and cameras.
  - A protocol change gets a new version; results are never mixed across versions.
- **Public trials.** Every `(task, trial)` has a fixed seed derived from
  `sha256(["handson-eval-v1", "public", task, trial])`.
  - The seed selects the task setting, reset layout, camera jitter, goal and instruction.
  - `python -m handson.evaluation.suite manifest --track <track>` prints the list.
  - Track lists are prefixes of one frozen list of 100 seeds per task.
- **Hidden trials.** The maintainers hold a key that generates further seeds
  for the same tasks. They are used only for verification.
- **Scored attempt.** Each trial is scored on its first completed attempt.
  - Only infrastructure errors (an evaluator crash, a lost policy connection)
    are retried, at most three attempts by default.
  - Every attempt stays in the run directory and must be included in the
    uploaded artifacts.

## Rules

1. **One fixed system.** The same checkpoint, code and inference settings run
   every trial of a track. The exception is `single_task`, where each task has
   its own checkpoint. Do not route between policies by task.
2. **No task identity or seed in the policy.** The websocket transport sends
   only images, measured state and the instruction. The HTTP transport also
   sends the task identifier and seed, for the paper models' language binding.
   Submissions must not use either to choose checkpoints, code, prompts or actions.
3. **Transfer-16 is evaluation-only.** No Transfer-16 demonstrations, rollouts
   or evaluation feedback may be used for training, tuning or model selection.
   This applies to every track.
4. **No evaluation feedback.** Do not train, tune, select checkpoints or edit
   prompts using outcomes of the public or hidden trials. Do not rerun a
   completed trial to get a better outcome.
5. **Policy inputs only.** Use the three camera images, the measured 52-D state
   and the instruction. Do not read simulator ground truth, checker state or
   files written by the evaluator. Do not modify the evaluator, budgets or
   success checks.
6. **`zero_shot` track.** This track excludes HandsOn training data.
   - Generic system prompts and harnesses must be identical for all tasks.
     Task-specific prompts, scripts, waypoints or skill code are not allowed.
   - Each episode starts from the same frozen system with empty episode memory.
   - Report the model-call budget in `interface.max_model_calls_per_trial`, and
     include the call logs in the artifacts.
7. **Disclosure.** Declare training data, external robot data and the
   pretrained initialization in `metadata.yaml`. A paper or technical report
   is expected. Verification requires the evaluated weights (or a callable API
   model version) and inference code to be public.

The maintainers may mark, move or remove entries that do not follow these
rules, and will state the reason in the pull request.

## How to submit

1. **Install the evaluator.** Follow [installation](../docs/installation.md) in
   a git checkout of this repository. Do not modify anything under `handson/`.
2. **Serve your policy.** Run an OpenPI websocket or HTTP policy server that
   returns `(horizon, 52)` physical EEF52 actions; see
   [policy integration](../docs/custom_policy_integration.md). Put a checkpoint
   identity in the server metadata and pass it as `--policy-metadata`.
3. **Run the track.** The run is resumable: rerun the same command after an
   interruption. Use `--shard i/n` with one output directory per machine, and
   `--dry-run` to print the evaluator commands.

   ```bash
   python -m handson.evaluation.suite run --track multi_task --output runs/my-model \
       --gpus 0,1,2,3 --per-gpu 1 \
       -- --host 127.0.0.1 --port 8000 --policy-metadata expected.json
   ```

   Arguments after `--` go to `python -m handson.evaluation`, for example
   `--action-horizon 16 --render-stride 16` or `--policy-url ...`. The suite
   sets `--task`, `--seed` and `--output` itself.
4. **Summarize.** Pick a submission id `YYYYMMDD_<team>_<model>` (lowercase).
   This command writes `results.jsonl` (one row per planned trial) and
   `summary.json` (scores with 95% intervals):

   ```bash
   python -m handson.evaluation.suite summarize runs/my-model \
       --out leaderboard/submissions/20261201_mylab_my-model
   ```

5. **Upload the raw evidence.** Upload the run directories to a public
   location such as a Hugging Face dataset, and put the URL in
   `links.artifacts`.
   - Include for every attempt: `SUITE.json`, `MANIFEST.jsonl`, and each
     `attempt_NN/` with its `console.txt`, `ATTEMPT.json` and `episode/`
     (`RESULT.json`, `JOB.json`, `PROMPT.json`, `steps.jsonl`, the three `.mp4`
     videos, …).
   - `input_*.npz` request dumps and `scene-cache/` may be left out.
6. **Describe the system.** Copy [`templates/metadata.yaml`](templates/metadata.yaml)
   into the submission directory and fill it in.
7. **Check and open a pull request.** Run the check below, then open a pull
   request that adds only `leaderboard/submissions/<id>/`. CI runs the same check.

   ```bash
   python -m handson.evaluation.suite check leaderboard/submissions/<id> --git-repo . --git-ref origin/main
   ```

   It confirms all of the following:
   - every planned trial has a scored outcome on the public seeds;
   - the scores recompute from the rows;
   - one execution setting was used throughout;
   - the evaluator commit is on `main` and the protocol files are unmodified;
   - the metadata is complete.

## Verification

When `submission.request_verification` is `true` and the weights or API are
available, the maintainers:

1. rerun the system on **10 hidden seeds per task** (660 episodes, or 500 for
   `single_task`), with the same evaluator commit and execution settings;
2. compare the hidden rerun with the submitted public score;
3. record the result in [`verifications.json`](verifications.json).

The entry becomes **Verified** when the task-macro difference is within 1.96
combined standard errors. Otherwise it stays **Self-reported**, and the
pull request explains why.

Systems that cannot be released stay **Self-reported**. They can be verified
if their authors keep a policy endpoint reachable during the rerun.

## Licence

Submitted `metadata.yaml`, `results.jsonl` and `summary.json` are published in
this repository under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Uploaded artifacts stay under their owners' licence, but must remain publicly
readable while the entry is listed.
