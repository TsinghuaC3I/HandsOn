# Paper model evaluation

The release includes the inference adapters, native model source, numeric
transforms, task conditioning and HTTP server for the paper's eight multi-task
models and the single-task diffusion-policy reference. Model weights and
foundation-model assets are supplied separately. `--model` selects the paper
execution profile; it does not download weights or reproduce the paper scores
by itself.

The paper also evaluates GPT-6 with a prompted `move_eef` palm-and-finger
controller. Its [660-trial results and protocol](gpt6-results.md) are published
separately. The nine checkpoint adapters documented here retain their existing
execution profiles; the prompted API controller is not a `--model` choice in
this release.

| `--model` | Model | Selected step | Predicted / executed actions | Image history |
| --- | --- | ---: | ---: | --- |
| `pi05` | π0.5 | 120000 | 30 / 30 | Current frame |
| `dm05` | DM0.5 | 86000 | 50 / 30 | Current frame |
| `hy` | HY-Embodied-0.5-VLA | 125000 | 50 / 30 | 6 frames, stride 10 actions |
| `lingbot` | LingBot-VLA | 151000 | 50 / 30 | Current frame |
| `openwam` | OpenWAM | 70000 | 32 / 16 | Current frame |
| `fastwam` | FastWAM | 100000 | 32 / 16 | Current frame |
| `motus` | Motus | 100000 | 48 / 16 | Current frame |
| `spatial` | Spatial Forcing | 90000 | 50 / 16 | Current frame |
| `dp` | Single-task DP | 30000 per task | 32 / 30 | Current frame |

All models use ten denoising steps and return absolute physical EEF52 targets:
right arm/hand 26 values, then left arm/hand 26 values. The evaluator executes
at 30 Hz. The server returns only the executed prefix and rejects nonfinite or
incorrectly shaped predictions. HY receives causal oldest-to-current history;
the initial frame is repeated when earlier frames do not exist.

DP has one checkpoint per Core-50 task. Its 50 packaged goal contracts preserve
the checkpoint's numeric condition vectors and cover the corresponding
released canonical instructions and training paraphrases. There is no automatic
DP routing or skill composition for Transfer-16. ACT and GR00T are not paper
evaluation profiles.

The model registry is `handson.policies.profiles.PROFILES`; each entry owns
its loader, input-asset options, task scope, execution settings and original
noise stream. Every adapter implements the shared `handson.interfaces.Policy`
protocol. The old standalone DP reference evaluator has been removed; DP uses
the same server and evaluator as the other paper models.

## Install the model environment

Use a separate Linux CUDA environment for inference. Do not install these
requirements into the Isaac Sim environment. From the repository root:

```bash
# Example: OpenWAM, in a Python 3.10 virtual environment
python -m pip install -r requirements/policies/openwam.txt
python scripts/serve_paper_policy.py --list
```

Use the requirement file matching the model. The source environments used
Python 3.11 for π0.5, Spatial Forcing and DP; Python 3.12 for HY, DM0.5 and
LingBot; and Python 3.10 for OpenWAM, FastWAM and Motus. The requirement files
pin the principal model libraries to those inspected environments. They are
not full environment lockfiles or fresh-install certification.

HY and LingBot additionally use FlashAttention 2.8.3 with PyTorch 2.8 and CUDA
12.8. Install a wheel compatible with that Python/PyTorch/CUDA combination,
or build it in the activated environment:

```bash
python -m pip install flash-attn==2.8.3 --no-build-isolation
```

The launcher imports the packaged source directly and does not import Isaac
Sim. The vendored native packages are isolated by model process; run one model
per server process. Existing unrelated `openpi`, `lerobot`, `wan`, or `models`
packages must not already be imported into that process.

## Required model inputs

`--checkpoint` is always required. The following paths are local paths chosen
by the user; none refers to an internal training machine.

| Model | Checkpoint format | Other arguments |
| --- | --- | --- |
| π0.5 | Selected `120000/` containing Orbax `params/` | `--normalization` checkpoint's `norm_stats.json`; `--tokenizer` local `paligemma_tokenizer.model` |
| DM0.5 | DCP directory with `trainer.json` and `state/` | `--foundation` local Dexmal/DM05 assets, including processor and original foundation weights |
| HY | DCP directory with `trainer.json` and `state/` | `--foundation` local Hunyuan UMI assets, including config, tokenizer and `model.safetensors` |
| LingBot | DCP directory with `trainer.json` and `state/` | `--foundation` LingBot config and `model.safetensors`; `--tokenizer` local Qwen tokenizer/processor directory |
| OpenWAM | Step-70000 `mp_rank_00_model_states.pt` | `--foundation` a directory containing `OpenWAM-Alpha-Pretrain-Foundation-Model/`, with `config.yaml`, `checkpoint_step_154000.safetensors` and its tokenizer |
| FastWAM | Step-100000 `mp_rank_00_model_states.pt` | `--text-cache` native SHA256-prompt-keyed safetensors cache |
| Motus | Step-100000 `mp_rank_00_model_states.pt` | `--foundation` a directory containing `Qwen3-VL-2B-Instruct/` and `Wan2.2-TI2V-5B/`, including `Wan2.2_VAE.pth`; `--text-cache` native prompt-keyed `.pt` cache |
| Spatial Forcing | Selected `90000/` containing Orbax `params/` | `--normalization` checkpoint's `norm_stats.json`; `--tokenizer` local `paligemma_tokenizer.model` |
| DP | Task's step-30000 `.pt` file | `--task` the matching public Core-50 identifier |

HY, DM0.5, LingBot, OpenWAM, FastWAM and Motus use packaged numeric statistics
from the original inference deployment. DP validates the statistics stored in
its checkpoint against its packaged task contract. π0.5 and Spatial Forcing
require the statistics accompanying their selected checkpoint. A normalization
override is a file for HY/OpenWAM/FastWAM, or a directory containing both
`norm_stats.json` and `dm05_norm.json`/`lingbot_norm.json` for DM0.5/LingBot.

No optimizer restore, trainer launch or demonstration dataset is required.
The DCP reader loads model tensors from the original checkpoint shards. Keep
the original DCP shard structure; do not rename or remove individual shards.

Examples for each model, using your own asset locations:

```bash
python scripts/serve_paper_policy.py --model pi05 --checkpoint /models/pi05/120000 \
  --normalization /models/pi05/norm_stats.json --tokenizer /models/paligemma_tokenizer.model
python scripts/serve_paper_policy.py --model dm05 --checkpoint /models/dm05/checkpoint-86000 \
  --foundation /models/DM05
python scripts/serve_paper_policy.py --model hy --checkpoint /models/hy/checkpoint-125000 \
  --foundation /models/hunyuan-umi
python scripts/serve_paper_policy.py --model lingbot --checkpoint /models/lingbot/checkpoint-151000 \
  --foundation /models/lingbot-vla-4b --tokenizer /models/qwen
python scripts/serve_paper_policy.py --model openwam --checkpoint /models/openwam/step-070000/mp_rank_00_model_states.pt \
  --foundation /models/official
python scripts/serve_paper_policy.py --model fastwam --checkpoint /models/fastwam/step-100000/mp_rank_00_model_states.pt \
  --text-cache /models/fastwam/text.safetensors
python scripts/serve_paper_policy.py --model motus --checkpoint /models/motus/step-100000/mp_rank_00_model_states.pt \
  --foundation /models/official --text-cache /models/motus/text.pt
python scripts/serve_paper_policy.py --model spatial --checkpoint /models/spatial/90000 \
  --normalization /models/spatial/norm_stats.json --tokenizer /models/paligemma_tokenizer.model
python scripts/serve_paper_policy.py --model dp --checkpoint /models/dp/t12_load_and_start_microwave.pt \
  --task t12_load_and_start_microwave
```

These are alternative launches. Each defaults to loopback port 8000 and writes
a new `policy-metadata.json` after loading the checkpoint and binding the socket.
For concurrent servers, choose distinct `--port` and `--metadata-output` values.
Expose the intended GPU through `CUDA_VISIBLE_DEVICES` before launch. A supplied
`--checkpoint-sha256` is verified before model loading. The resulting metadata
always records the actual checkpoint digest. Startup does not certify a full
rollout or successful task execution.

## Run an episode

In the simulator environment, use the metadata file produced by that server:

```bash
python -m handson.evaluation --model openwam --task t12_load_and_start_microwave \
  --policy-url http://127.0.0.1:8000 --policy-metadata policy-metadata.json \
  --seed 0 --output outputs/openwam-microwave
```

Use the same `--model` on both sides. The evaluator checks the selected model's
identity, step, dimensions, action prefix and history before starting Isaac
Sim. Explicit incompatible `--action-horizon`, `--history-frames`,
`--history-stride` or `--render-stride` flags are rejected. For example, HY
selects 6 history frames and render stride 10; OpenWAM selects an execution and
render stride of 16. Camera poses, native resets, scoring and budgets continue
to come from the common paper evaluator.

If model and simulator run on different hosts, connect the loopback listener
using an SSH tunnel and copy the metadata file. See
[custom policy integration](custom_policy_integration.md) for the NPZ protocol.

## FastWAM and Motus language caches

Caches are part of the model inputs and are not bundled as model weights.
Use the original cache, or prepare one with the original frozen UMT5 encoder:

```bash
PYTHONPATH="$PWD/handson" python -m handson.policies.encode_text \
  --model fastwam --encoder /models/wan/models_t5_umt5-xxl-enc-bf16.pth \
  --tokenizer /models/wan/google/umt5-xxl --task t12_load_and_start_microwave \
  --output /models/fastwam/microwave.safetensors
```

For Motus, select `--model motus` and a `.pt` output. Repeat `--task` to cover
multiple tasks; omitting it selects all released tasks. Each selected task
includes its canonical instructions and 32 training paraphrases per goal.
FastWAM wraps instructions with its native robot-video prefix and uses a
128-token context with preparation batches of 16. Motus uses raw instructions
and a 512-token context. The builder reports overlength instructions and stops
without creating a cache; it never silently truncates or replaces instructions.
Consequently, a model adapter's presence does not imply every Transfer-16
instruction is admissible under that model's native token limit. Missing cache
entries are explicit errors, not substituted prompts.

## Source and validation

The native loaders preserve the deployed DCP precision hooks, tied-weight
checks, strict state-dictionary loads, model-specific image transforms and
noise streams. OpenWAM retains the explicit EEF52/Alpha80 side and rotation
conversion; Motus retains action min/max bounds for state normalization.
FastWAM's tensor resize and OpenWAM/Motus's letterboxed PIL mosaic remain
separate. Spatial Forcing loads its alignment-head parameters without loading
a teacher at inference. DP initializes its ResNet without downloading ImageNet
weights because the full learned state is immediately loaded strictly.

Source hashes and migrated-file hashes are recorded in
`handson/handson/policies/source_provenance.json`. Third-party notices are in
[policy sources](policy-source-notices.md); their licenses remain in each
vendored directory.

The release checks cover source syntax, model dependency imports, the HTTP
contract, execution prefixes, causal history rejection, deterministic seed
mapping, EEF52/Alpha80 conversion and all 50 DP task contracts. Protocol tests
use explicit test policies. The existing 66-task rendering diagnostics also
use held actions. Neither check is a new nine-model checkpoint rollout or a
reproduction of the paper's success rates.

The [machine-readable validation receipt](validation/paper-models.json) records
23 passed CPU tests, nine model-profile subtests, successful native dependency
imports for all nine models, and complete model-file inclusion in the wheel
builder's file manifest. No learned-checkpoint GPU rollout or fresh package
installation was performed for this adapter migration.
