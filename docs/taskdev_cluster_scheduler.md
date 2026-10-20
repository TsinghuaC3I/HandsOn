# Task-development cluster scheduler

`scripts/taskdev_cluster.py` is the fail-closed launcher for one task per
physical GPU on `hp-13` and `hp-14`. It never accepts `hp-17`, does not run a
network service, and uses SSH only for one-shot commands. SSH forwarding is
disabled.

Before every real launch the target atomically acquires the `(host, GPU)`
lease, then re-queries compute processes, `memory.used`, GPU utilization, and
host `MemAvailable` while holding that lease. A compute process, unexplained
memory/utilization, low host memory, missing GPU, query failure, or lease
collision aborts the launch. Small driver-only reservations are reported but
accepted.

## Usage

Run from a shared-PFS checkout so the same absolute script path exists on both
hosts. Audit is read-only:

```bash
python3 scripts/taskdev_cluster.py audit --hosts hp-13 hp-14
```

`launch` is a dry run unless the explicit `--launch` flag is present. The
scheduler leases `{physical_gpu}`, then creates a private mount namespace in
which that one NVIDIA device is the only visible GPU. CUDA, Vulkan, PhysX, and
USD-RT therefore all address it as `{logical_gpu}` (`0`). This is required for
Isaac Sim 5.1 camera workloads on multi-GPU hosts.

```bash
python3 scripts/taskdev_cluster.py launch \
  --host hp-13 --gpu 2 --task-id t03_cover_box \
  --repo /absolute/shared/path/to/handson \
  --base-ref remote-task-dev-latest \
  --max-runtime-seconds 900 \
  -- python3 scripts/preview_isaaclab_task.py \
     --task_name t03_cover_box --device cuda:{logical_gpu} --headless
```

Review that JSON plan, repeat it with `--launch`, and retain the returned
`run_dir`:

```bash
python3 scripts/taskdev_cluster.py status \
  --host hp-13 --run-dir /absolute/path/from/run_dir

python3 scripts/taskdev_cluster.py stop \
  --host hp-13 --run-dir /absolute/path/from/run_dir
```

The scheduler does not rely on `CUDA_VISIBLE_DEVICES`, because that variable
cannot filter Vulkan devices. Instead it bind-masks non-selected `/dev/nvidiaN`
nodes only inside the worker's private mount namespace; the host `/dev` is not
modified. The mapping is recorded in `gpu_namespace.json`. Use `{gpu}` or
`{physical_gpu}` only for provenance/metadata and `{logical_gpu}` for runtime
device arguments.

Every run gets a unique branch, worktree, run directory, `run.json`, heartbeat,
worker PID and isolated process group. `stop` validates the lease token, PID
start time, command identity, and PGID, then signals only that exact owned
process group. Worktrees and run artifacts are intentionally preserved for
inspection and commits. GNU `timeout --foreground` keeps the deadline wrapper
and task inside the scheduler-owned PGID. Before releasing the GPU lease the
worker drains and revalidates every remaining live member of its owned PGID;
the recovery `stop` path performs the same drain if the worker already exited.
The configured wall-clock bound is 30 minutes by default and an overrun is
recorded as `timed_out`. PID namespaces are intentionally not used because they
prevent Isaac Sim 5.1 from initializing CUDA on the current cluster runtime.

Defaults can be tightened with `--driver-reservation-mib`,
`--max-idle-utilization`, and `--min-host-available-gib`. A contradictory
snapshot is never overridden automatically; investigate it and rerun a fresh
audit.

GPU isolation also fails closed if `unshare`, the private mount namespace,
`CAP_SYS_ADMIN`, or validated NVIDIA character devices are unavailable.
