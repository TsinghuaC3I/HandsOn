# Newton + Wuji Hand 2 trajectories

> **Publication status:** The selection, renderer, and acceptance sample are available. Raw shards are uploading; wait for `newton_wuji2/RELEASE_COMPLETE.json` before downloading the complete dataset.

This release contains **10,000 complete raw trajectories: 200 from each of the
50 Core tasks**, selected from 11,678 recordings. It adds a separate
`newton_wuji2/` directory to
[HandsOn-Datasets](https://huggingface.co/datasets/Cyan666/HandsOn-Datasets/tree/main/newton_wuji2).
The existing PhysX/Wuji1 LeRobot corpus retains its own splits, videos and
normalization statistics.

The new recordings contain **7,077,577 timesteps at 30 Hz** (about 65.53 hours)
and 191,065,403,988 bytes of original files. They are **raw Zarr replay archives
with metadata and episode assets**, not a pre-rendered LeRobot dataset. Render
the three camera streams with the procedure below before using an RGB policy
training pipeline. Raw action/state layouts are specified per episode in
`recording_schema.json`; do not assume the PhysX LeRobot corpus's 52-dimensional
layout. This release does not claim 30,000 completed training videos.

## Selection and versions

Within each task, sort ascending by `collection_meta.json:duration_seconds`,
then lexicographically by the original episode ID, and retain the first 200.
Tasks with exactly 200 recordings retain every recording. No trajectory is
truncated, padded, interpolated or duplicated. The selector checks duration
against `step_count / data_fps` (within 2 microseconds) and checks the action
array header's first dimension against `step_count` for all 11,678 candidates.

All selected source metadata records `task_outcome=success`. This is the
recorded outcome, not a new physical-replay or policy-evaluation certification.
Task 09 (peg insertion) contributes 200 `nvidia_newton_cross_gpu_v2` episodes;
the remaining 49 tasks contribute 9,800 `nvidia_newton_v1` episodes. The
original markers and directory suffixes are preserved.

The release includes `selection.jsonl` (10,000 entries), `candidates.jsonl`
(11,678 entries with selection ranks), `SELECTION.json` (per-task ranges),
and `SHARDS.json` (archive sizes, SHA256s, membership and source-MD5 receipts).
The selection SHA256 is
`a9b2aa504d09fc8a26669a73cc18264d98b8f3e528cff42637c2cc113552d704`.
Short-duration selection changes the duration distribution; it is not a random
sample. No train/validation split is assigned to this addition. Split by
collection session before training, and compute normalization on that training
split only; do not reuse the PhysX corpus's statistics without checking units
and schema.

## Download

Install the Hub client in your download environment:

```bash
python -m pip install huggingface_hub
# Download and verify all 200 episodes of task 03:
python scripts/newton_wuji2/download.py --task 3 --output ./newton-wuji2
# Omit --task for all 10,000 episodes.
```

There are four uncompressed tar shards per task, 50 episodes per shard. The
inner `replay.zarr.zip` files retain their original compression and bytes.
Extraction produces `tasks/01` through `tasks/50`, preserving episode names
and all source files. During publication, every file is read through an MD5
check against the verified transfer manifest; each uploaded tar is checked
against the Hub's LFS SHA256 and byte size before its temporary copy is removed.

## Render the three camera videos

Use Linux, an NVIDIA RTX GPU, **Isaac Sim 5.1** and **Isaac Lab 2.3.2**. The
acceptance sample was rendered on an RTX 5090. Install the simulator following
[installation](installation.md), and use its Python executable below. Accept
NVIDIA's EULA according to your installation. This is offline recorded-state
visual replay: it reads Newton observations/actions and restores recorded
poses to render them. It does not rerun Newton dynamics to certify success.

```bash
export ISAAC_LAB_DIR=/path/to/IsaacLab-v2.3.2
export OMNI_KIT_ACCEPT_EULA=YES
python scripts/newton_wuji2/render_episode.py \
  --episode ./newton-wuji2/tasks/03/EPISODE_DIRECTORY \
  --output ./renders/EPISODE_DIRECTORY \
  --runtime ./newton-wuji2-render-runtime \
  --python /path/to/isaacsim-python \
  --isaac-lab "$ISAAC_LAB_DIR"
```

Replace `EPISODE_DIRECTORY` with a downloaded episode directory. The entry
downloads the SHA256-pinned replay runtime and native task assets from
`newton_wuji2/renderer/replay-runtime.tar`; alternatively pass
`--runtime-archive /path/to/replay-runtime.tar` for an offline installation.
The source archive SHA256 is
`2a7908dcb2c5984ad4c253141545e48ebf49f137a6c17a1f3d9071e3a3dad33b`.
The archive includes the legacy collection task names and their compatible
replay implementation. These names differ from the public evaluator's `t01`
through `t50` names; the entry reads the correct source task from metadata.
GitHub's `scripts/newton_wuji2/visuals/` provides the released appearance code.

The preparation step relocates texture/MDL paths and records the resulting
hashes in `PORTABLE_RUNTIME.json`. It requires the exact validated
`OmniSurface.mdl` hash, changes no geometry or numeric material parameters,
and never edits the input episode. `HANDSON_OMNISURFACE_MDL` may point to the
installed Kit shader when its standard location cannot be discovered.

The renderer uses the acceptance sample's **V16 scene, 128 samples per pixel,
path tracing, OptiX denoising and separate head/wrist visibility groups**.
The main camera produces **640×640** frames; the left and right wrist cameras
each produce **848×480**, all at **30 fps**. These are this replay profile's
dimensions, distinct from the 854×480 front stream in the PhysX paper corpus.
V16 camera calibration, exposures, lighting and native task objects are
preserved. The only appearance replacement is the Wuji2 hand/wrist surface.

Wuji2 v14 uses ten continuous knitted finger covers spanning 30 articulated
joints, closed finger roots, a soft fabric shading response, photo-informed
palm textures, distal wear/metal details and wrist housings. Cloth vertices
follow measured native link poses. This is a visual deformation model, not
a cloth dynamics solver. The adapter rejects changes to physical schemas,
collision geometry or effective contact materials. Folds and wear are
estimated from reference photographs; exact photographic equivalence is not
claimed.

Outputs are `videos/main.mp4`, `videos/wrist_left.mp4`,
`videos/wrist_right.mp4`, camera poses/calibration, appearance receipts,
`deferred_arm_replay_status.json` and atomic `frame_checkpoints/`.
Rerun the same command to resume an interrupted output: source, state/action
arrays, appearance, sampling and checkpoint code hashes must match, and saved
frames must form a contiguous sequence. The writer encodes only after complete
frame coverage. Preserve the checkpoint directory until video validation passes.
Use a separate runtime directory per concurrent worker while preparing it;
never run two workers on the same output directory.

`--max-frames 3` creates a diagnostic only; omit it for the whole trajectory.
Do not publish such a diagnostic as a complete training video. Before training,
verify every stream's decode, FPS, dimensions and exact source-frame count.
Record severe wrist occlusions rather than changing camera poses or fabricating
frames. The 296-frame acceptance trajectory has two very dark right-wrist frames
at source indices 114/115 because of occlusion; they are preserved.

If the container reports missing OptiX denoiser weights, install the weights
matching its NVIDIA driver through the official NVIDIA package. Do not silently
disable denoising and label the output as this validated profile. The cloud
acceptance environment used driver 570.153.02 and its matching `nvoptix.bin`.

## Maintainer commands

```bash
python scripts/newton_wuji2/select_shortest.py \
  --source /path/to/newton_wuji2_50_20261106 \
  --output /path/to/release/package/newton_wuji2
python scripts/newton_wuji2/publish_dataset.py \
  --source /path/to/newton_wuji2_50_20261106 --release /path/to/release
```

The uploader uses the existing Hugging Face login or `HF_TOKEN`, bounded
temporary storage and resumable per-shard receipts. It changes only the new
release prefix. The dataset card/timeline must announce completion only after
all 200 shards and their membership have been independently verified on the Hub.
