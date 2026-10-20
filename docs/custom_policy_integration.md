# Policy integration

Run every task with `python -m handson.evaluation --task TASK`. The evaluator
uses the paper's measured-state codec, synchronous native rollout, split camera
capture, approved task settings and evaluation budgets. Each invocation starts
one fresh Isaac Sim worker. Use a new output directory for every episode.

## Packaged paper models

For the paper checkpoints, use the native adapters and launch commands in
[paper model evaluation](paper-models.md). Pass `--model` to the evaluator so
execution horizons and HY history are selected and checked automatically.
The generic protocol below remains available for custom policies.

The packaged model server also requires the public `task` identifier as a
scalar string in the NPZ request. The evaluator supplies it automatically;
it binds task language and preserves the original task-specific noise stream.

## Observation and action contract

State is a finite float32 vector of length 52, ordered **right, then left**.
Each 26-value block contains world palm position (xyz, metres), world palm
rotation vector (xyz, radians), and 20 measured Wuji finger joint positions
(radians, finger-major order). State comes from the actual articulation.

Actions have shape `(action_horizon, 52)` with the same order and units.
They are absolute world commands. The evaluator converts rotation vectors
to wxyz quaternions and passes named right/left commands to the native
controller, preserving its limits and contact guards. There is no additional
delta conversion, single-hand mask or action rescaling.

| Camera | Websocket key | HTTP key | RGB uint8 shape |
| --- | --- | --- | --- |
| Main | `base` | `front` | `(480, 854, 3)` |
| Left wrist | `wrist_left` | `wrist_left` | `(480, 848, 3)` |
| Right wrist | `wrist_right` | `wrist_right` | `(480, 848, 3)` |

Both transports also receive `state` and `prompt`. Model-side resizing,
normalization and checkpoint-specific transforms belong to the policy server.
The server returns physical EEF52 commands in `actions`.

## Python environment interface

All 66 task configs inherit the same public factory. Policy mode accepts one
52-value physical action per step and returns measured `state` plus `front`,
`wrist_left` and `wrist_right` when images are enabled:

```python
from handson.tasks import CONFIG_MAPPING

with CONFIG_MAPPING["t12_load_and_start_microwave"]().get_environment(
    policy_mode=True, image_obs=True, headless=True,
) as env:
    observation, info = env.reset(seed=1)
    action = observation["state"].copy()  # diagnostic measured-pose hold
    observation, reward, terminated, truncated, info = env.step(action)
```

The wrapper and formal rollout share `handson.eef52.EEF52Control`. Neither
uses privileged task features or the last command as measured policy state.
This factory retains the native task lifecycle; run `handson.evaluation` for
paper stable-origin initialization, calibrated split-view capture, task budgets
and formal result receipts. `image_obs=False` exposes measured state only.

Collection and trajectory replay use the explicitly named
`get_native_environment(...)` factory with their original recorded action and
observation layouts. `get_environment(policy_mode=False)` retains collection
behavior. Code that previously supplied 54/46-value actions in policy mode must
use the native factory or convert its actions to EEF52.

## Module boundaries and extension points

- `handson.interfaces`: lightweight policy protocol, dimensions, view names
  and input/output validation; no simulator or model framework imports.
- `handson.tasks.registry.TASKS`: all task identifiers, splits, checkpoint
  names, goal readers and native evaluation hooks. Language/goal assets live
  under `handson/tasks/assets` and are shared with model servers.
- `handson.evaluation`: episode lifecycle, rendering, transport, recording
  and result handling. `ImageHistory` belongs to one rollout; HTTP, websocket
  and held-action clients own their transports without changing global methods
  or the global HTTP opener. Task-specific hooks live in `task_adapters`.
- `handson.policies.profiles.PROFILES`: model loader, accepted asset options,
  task splits, history/horizons and original noise-stream settings. Add a model
  here and implement `Policy.predict(state, images, prompt, seed)` in an adapter;
  return the full native horizon. The server selects the executed prefix.
- `handson.policies.native` and `vendor`: original model implementation and
  preprocessing. Run one model per process to isolate upstream import names.

Isaac Sim still runs one episode per worker process: native scene initialization
has process-wide Kit state and camera/trace installation. Instance-owned policy
clients do not imply that multiple Isaac simulations can share a process.

## Transports

For an OpenPI websocket server:

```bash
pip install -e ./openpi/packages/openpi-client
python -m handson.evaluation --task t12_load_and_start_microwave \
  --host 127.0.0.1 --port 8000 --output outputs/policy-microwave
```

`--policy-metadata expected.json` verifies exact keys in the websocket
handshake. Include the checkpoint identity in server metadata.

The paper HTTP transport uses GET for model metadata and POST for a NumPy NPZ
packet with `Content-Type: application/octet-stream`. POST also includes scalar
`seed` and `request`; the response is JSON containing `actions`. Expected
metadata with a checkpoint identity is required:

```bash
python -m handson.evaluation --task t12_load_and_start_microwave \
  --policy-url http://127.0.0.1:8000/tasks/t12_load_and_start_microwave \
  --policy-metadata expected.json --output outputs/http-microwave
```

The default action horizon is 30. A checkpoint that executes 16 actions per
request uses `--action-horizon 16 --render-stride 16`. For HTTP history models,
set the checkpoint's `--history-frames` and `--history-stride`; render stride
must equal history stride and divide the horizon. Images then have a leading
history dimension, oldest first. The initial image fills missing past frames.
No future state or demonstration action is supplied.

## Rendering and timing

All cameras use the packaged detailed appearance and original calibration.
Capture performs six render passes per view with the original exposure and
obstruction visibility. It cannot advance physics or change measured robot
state. Seeded mount jitter stays fixed within the episode. Videos sample live
frames at `30 / render_stride` fps; this is playback cadence, not real-time
performance. The default stride is 30.

Actual action time is checked as `physics_dt * n_substeps = 1/30`. Budgets live
in `handson/evaluation/budgets.json`. Core tasks retain their frozen native
control and initialization parameters, including the approved task-specific
boundaries. Transfer tasks use the original native recording-origin lifecycle.

Instructions bind to the native seeded goal. The original language sampler
chooses canonical text with probability 0.2 and otherwise one of 32 training
paraphrases, independently of environment randomness. It samples once per
episode and never uses the eight held-out paraphrases. Public task identifiers
give these draws a new identity. Historical per-model frozen goal distributions
and exact trial draws are not recreated; compare scores only under matching
seed, goal, language, checkpoint and runtime protocols.

## Results and diagnostics

`RESULT.json` records completion, native success, action count, timing and video
paths. `PROMPT.json`, `SETTING_SELECTION.json` (where applicable), `BUDGET.json`,
`JOB.json`, `STATE_CONTRACT.json` and `RENDERING.json` record the effective
protocol. Infrastructure errors produce `ERROR.json` and a nonzero CLI exit.
A completed result is required even if Kit exits with code zero.

```bash
python -m handson.evaluation --task t12_load_and_start_microwave \
  --diagnostic-steps 90 --output outputs/diagnostic
```

Diagnostics hold measured poses and return `success: null` and
`counts_as_formal_trial: false`. They exercise reset, state/action encoding,
native stepping and camera recording, without measuring policy performance.
`--check` constructs all 66 factories without Isaac Sim; it does not certify
reset, rendering or task completion.

The paper's carton protocol records raw force without failing or terminating
on force magnitude; bilateral support, lift, three-second hold and drop checks
remain active. The instruction's 12 N clause is waived for scoring.
The latched-door task retains its source-alignment waiver: the instruction
specifies left lever >45 degrees/right door, while the native checker uses
right lever ≥25 degrees/left door. These are disclosed protocol exceptions,
not low-force or source-alignment certifications.
