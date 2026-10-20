# Wrist Camera Batch Repair

`scripts/rerender_wrist_cameras.py` repairs native A/B/C IsaacLab wrist videos from the
recorded action and state arrays in each episode's `replay.zarr`. It leaves the
main camera, low-dimensional data, review result, and episode identity intact.

The tool is non-destructive by default. A plain invocation only scans for the
old near-uniform gray camera-housing frames:

```bash
/home/user/.micromamba/envs/handson/bin/python \
  scripts/rerender_wrist_cameras.py --from-time 2026-07-17
```

Test rendering a few selected episodes into the staging directory:

```bash
/home/user/.micromamba/envs/handson/bin/python \
  scripts/rerender_wrist_cameras.py \
  --from-time 2026-07-17 --max-episodes 3 --rerender
```

Stop active Isaac collection before applying a batch. Local replacement needs
an explicit confirmation and backs up old videos under
`/home/user/handson_wrist_rerender_backups/<batch-id>/`:

```bash
/home/user/.micromamba/envs/handson/bin/python \
  scripts/rerender_wrist_cameras.py \
  --from-time 2026-07-17 --rerender --apply-local --yes
```

Use the configured KS3 key to update cloud data after local validation:

```bash
/home/user/.micromamba/envs/handson/bin/python \
  scripts/rerender_wrist_cameras.py \
  --from-time 2026-07-17 --rerender --apply-local --upload-cloud \
  --parallel-tasks 2 --yes
```

For an already repaired local episode, cloud upload can be retried without
rendering it again:

```bash
/home/user/.micromamba/envs/handson/bin/python \
  scripts/rerender_wrist_cameras.py \
  --episode t04_cover_pitcher_demo_70_2026-07-20_14-53-41_591390 \
  --upload-cloud --yes
```

For previously verified uploads, the tool reuses `remote_prefix` from
`cloud_upload_meta.json` and PUTs the two repaired videos to the same object
keys. For an unverified or never-uploaded episode it uploads every episode file
to that prefix, so a successful wrist upload cannot incorrectly mark missing
episode files as uploaded. `cloud_upload_meta.json` is uploaded last and the
local marker is changed only after every required PUT returns HTTP 2xx and an
ETag. A partial failure is retryable and is recorded separately in
`wrist_rerender_upload_failure.json`.

Useful filters are repeatable `--task` and `--episode`, plus `--collector`,
`--from-time`, and `--to-time`. Pass `--all` to skip video health scanning and
rerender every selected new-engine episode unconditionally. Use
`--full-cloud-reupload` when a complete cloud refresh is required even for a
trusted old upload marker. `--parallel-tasks 2` runs two task-specific Isaac
workers concurrently; use it only when no collection process is active and GPU
memory has been checked for the selected task set.
