"""Run, summarize and check the HandsOn leaderboard protocol (handson-eval-v1).

    python -m handson.evaluation.suite run --track multi_task --output runs/my-model \\
        --gpus 0,1,2,3 -- --host 127.0.0.1 --port 8000 --policy-metadata expected.json
    python -m handson.evaluation.suite summarize runs/my-model --out leaderboard/submissions/<id>
    python -m handson.evaluation.suite check leaderboard/submissions/<id>

Every trial runs ``python -m handson.evaluation`` once, in a fresh Isaac Sim worker,
with the trial's public seed. Arguments after ``--`` go to that evaluator
unchanged (policy transport, metadata and execution settings).
"""
import argparse
import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from . import leaderboard as lb

RESERVED = ("--task", "--config", "--seed", "--output", "--paper-trial", "--inspect-paper-trial",
            "--paper-variant", "--diagnostic-steps", "--max-policy-steps", "--worker", "--physics-backend",
            "--list", "--check")


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write(path, data):
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _passthrough(args):
    for value in args:
        flag = value.split("=", 1)[0]
        if flag in RESERVED:
            raise SystemExit(f"{flag} is set by the suite and cannot be passed to the evaluator")
    return list(args)


def _load_manifest(args):
    if args.manifest is None:
        rows = lb.manifest(args.track)
    else:
        rows = [json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line.strip()]
        if not rows or any(row.get("protocol") != lb.PROTOCOL or row.get("track") != args.track for row in rows):
            raise SystemExit(f"{args.manifest} is not a {lb.PROTOCOL} {args.track} manifest")
        if rows[0]["set"] == "public" and rows != lb.manifest(args.track):
            raise SystemExit("A public manifest file must equal the packaged public trials")
    return rows


def _select(rows, args):
    tasks = set(args.tasks.split(",")) if args.tasks else None
    if tasks:
        unknown = tasks - {row["task"] for row in rows}
        if unknown:
            raise SystemExit(f"Tasks not in this manifest: {sorted(unknown)}")
    index, count = (int(x) for x in args.shard.split("/")) if args.shard else (0, 1)
    if not 0 <= index < count:
        raise SystemExit("--shard must be i/n with 0 <= i < n")
    return [row for i, row in enumerate(rows)
            if i % count == index and (tasks is None or row["task"] in tasks)]


def _prepare_run(output, rows, args, passthrough):
    output.mkdir(parents=True, exist_ok=True)
    identity = lb.evaluator_identity()
    if identity["git_commit"] is None:
        raise SystemExit("Run the suite from a git checkout of HandsOn, with git on PATH, "
                         "so the evaluator commit is recorded")
    if identity["git_dirty"] and not args.allow_dirty:
        raise SystemExit("The checkout has local changes under handson/; commit or stash them "
                         "(--allow-dirty records a run that cannot be submitted)")
    if identity["protocol_files"] != lb.PROTOCOL_FILE_SHA256:
        raise SystemExit(f"Protocol files differ from {lb.PROTOCOL}; use an unmodified release")
    record = dict(schema="handson_suite_run_v1", protocol=lb.PROTOCOL, track=args.track, set=rows[0]["set"],
                  manifest_sha256=lb.manifest_sha256(rows), evaluator=identity, evaluator_args=passthrough)
    path = output / "SUITE.json"
    if path.exists():
        previous = json.loads(path.read_text())
        same = {k: previous.get(k) for k in record} == record
        if not same:
            raise SystemExit(f"{output} holds a run with a different protocol, manifest, evaluator commit "
                             "or evaluator arguments; use a new output directory")
    else:
        _write(path, dict(record, created_utc=_now()))
        (output / "MANIFEST.jsonl").write_text("".join(lb._canonical(row) + "\n" for row in rows))
    return record


class Runner:
    def __init__(self, output, passthrough, args):
        self.output, self.passthrough, self.args = output, passthrough, args
        self.lock = threading.Lock()
        self.running = {}
        self.stopping = False
        self.done = self.failed = self.successes = 0

    def attempts(self, row):
        folder = lb.trial_dir(self.output, row["task"], row["trial"])
        return sorted(p for p in folder.glob("attempt_[0-9][0-9]")) if folder.is_dir() else []

    def pending(self, row):
        attempts = self.attempts(row)
        if any(lb.attempt_complete(p) for p in attempts):
            return False
        # Attempts stopped by Ctrl-C (or a killed runner) do not use the retry budget.
        used = 0
        for attempt in attempts:
            record = attempt / "ATTEMPT.json"
            used += record.is_file() and json.loads(record.read_text()).get("exit_code") != "interrupted"
        return used < self.args.max_attempts

    def command(self, row, attempt):
        return [sys.executable, "-m", "handson.evaluation", "--task", row["task"], "--seed", str(row["seed"]),
                "--output", str(attempt / "episode"), *self.passthrough]

    def run_one(self, row, gpu):
        folder = lb.trial_dir(self.output, row["task"], row["trial"])
        folder.mkdir(parents=True, exist_ok=True)
        attempt = folder / f"attempt_{len(self.attempts(row)):02d}"
        attempt.mkdir()
        env = dict(os.environ)
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = gpu
        started = time.monotonic()
        with open(attempt / "console.txt", "w") as log:
            proc = subprocess.Popen(self.command(row, attempt), stdout=log, stderr=subprocess.STDOUT, env=env,
                                    start_new_session=True)
            with self.lock:
                self.running[proc.pid] = proc
            try:
                code = proc.wait(timeout=self.args.timeout or None)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                code = "timeout"
            finally:
                with self.lock:
                    self.running.pop(proc.pid, None)
        if self.stopping:
            code = "interrupted"
        cache = attempt / "episode" / "scene-cache"
        if not self.args.keep_scene_cache and lb.attempt_complete(attempt) and cache.is_dir():
            shutil.rmtree(cache, ignore_errors=True)  # per-episode asset conversion workspace
        _write(attempt / "ATTEMPT.json", dict(task=row["task"], trial=row["trial"], seed=row["seed"], gpu=gpu,
               exit_code=code, wall_seconds=round(time.monotonic() - started, 1), finished_utc=_now()))
        return lb.attempt_complete(attempt), attempt

    def worker(self, jobs, gpu, total):
        while not self.stopping:
            try:
                row = jobs.get_nowait()
            except queue.Empty:
                return
            complete, attempt = self.run_one(row, gpu)
            outcome = "error"
            if complete:
                success = json.loads((attempt / "episode" / "RESULT.json").read_text()).get("success")
                outcome = "success" if success else "failure"
            retry = not complete and not self.stopping and self.pending(row)
            with self.lock:
                if complete:
                    self.done += 1
                    self.successes += outcome == "success"
                elif not retry:
                    self.failed += 1
                print(f"[{self.done + self.failed}/{total}] {row['task']} trial {row['trial']:03d} "
                      f"seed {row['seed']} -> {outcome} ({attempt.name}, gpu {gpu})"
                      + (" retrying" if retry else ""), flush=True)
            if retry:
                jobs.put(row)

    def stop(self):
        self.stopping = True
        with self.lock:
            for proc in list(self.running.values()):
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


def cmd_run(args, passthrough):
    output = Path(args.output)
    rows = _load_manifest(args)
    selected = _select(rows, args)
    _prepare_run(output, rows, args, passthrough)
    runner = Runner(output, passthrough, args)
    todo = [row for row in selected if runner.pending(row)]
    print(f"{lb.PROTOCOL} {args.track}: {len(selected)} selected trials, {len(todo)} to run", flush=True)
    if args.dry_run:
        for row in todo:
            folder = lb.trial_dir(output, row["task"], row["trial"])
            print(" ".join(runner.command(row, folder / f"attempt_{len(runner.attempts(row)):02d}")))
        return
    gpus = args.gpus.split(",") if args.gpus else [None]
    slots = [gpu for gpu in gpus for _ in range(args.per_gpu)]
    jobs = queue.Queue()
    for row in todo:
        jobs.put(row)
    threads = [threading.Thread(target=runner.worker, args=(jobs, gpu, len(todo)), daemon=True) for gpu in slots]
    for thread in threads:
        thread.start()
    try:
        while any(thread.is_alive() for thread in threads):
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping: killing running episodes; rerun the same command to resume", flush=True)
        runner.stop()
        raise SystemExit(130)
    print(f"Completed {runner.done}/{len(todo)} trials ({runner.successes} successes); "
          f"{runner.failed} exhausted {args.max_attempts} attempts", flush=True)
    if runner.failed:
        raise SystemExit(1)


def _summarize(dirs):
    records = []
    for folder in dirs:
        path = Path(folder) / "SUITE.json"
        if not path.is_file():
            raise SystemExit(f"{folder} is not a suite run directory")
        records.append(json.loads(path.read_text()))
    keys = ("protocol", "track", "set", "manifest_sha256", "evaluator")
    if any({k: r.get(k) for k in keys} != {k: records[0].get(k) for k in keys} for r in records):
        raise SystemExit("Run directories differ in protocol, track, manifest or evaluator commit")
    track = records[0]["track"]
    rows = [json.loads(line) for line in (Path(dirs[0]) / "MANIFEST.jsonl").read_text().splitlines() if line.strip()]
    if lb.manifest_sha256(rows) != records[0]["manifest_sha256"]:
        raise SystemExit("MANIFEST.jsonl does not match SUITE.json")
    results = []
    for planned in rows:
        owners = [folder for folder in dirs if lb.attempt_dirs(folder, planned["task"], planned["trial"])]
        if len(owners) > 1:
            raise SystemExit(f"Trial {planned['task']}/{planned['trial']} was run in several directories")
        results.append(lb.collect_row(owners[0] if owners else dirs[0], planned))
    summary = lb.build_summary(results, track, records[0]["evaluator"], records[0]["set"],
                               records[0]["manifest_sha256"])
    return results, summary


def _print_summary(summary):
    counts = summary["counts"]
    print(f"{summary['protocol']} {summary['track']} ({summary['set']}): {counts['scored']}/{counts['planned']} "
          f"scored, {counts['invalid']} invalid, {counts['missing']} missing")
    for suite, block in summary["suites"].items():
        print(f"  {lb.SUITE_LABELS[suite]}: {block['percent']:.2f}% (95% CI {block['ci95_percent'][0]:.2f}–"
              f"{block['ci95_percent'][1]:.2f}), {block['successes']}/{block['planned']}")
    if summary["overall"]:
        block = summary["overall"]
        print(f"  Overall (66-task macro): {block['percent']:.2f}% (95% CI {block['ci95_percent'][0]:.2f}–"
              f"{block['ci95_percent'][1]:.2f})")


def cmd_summarize(args):
    results, summary = _summarize(args.runs)
    out = Path(args.out or args.runs[0])
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.jsonl").write_text(lb.results_text(results))
    _write(out / "summary.json", summary)
    _print_summary(summary)
    if summary["counts"]["scored"] != summary["counts"]["planned"]:
        print("Incomplete: rerun `suite run` with the same arguments to fill missing or invalid trials",
              file=sys.stderr)
        raise SystemExit(1)


def cmd_check(args):
    status = 0
    for folder in args.submissions:
        report = lb.validate_submission(folder, git_repo=args.git_repo, git_ref=args.git_ref)
        for message in report["errors"]:
            print(f"ERROR {folder}: {message}")
        for message in report["warnings"]:
            print(f"WARNING {folder}: {message}")
        if not report["errors"]:
            print(f"OK {folder}")
        status |= bool(report["errors"])
    raise SystemExit(status)


def cmd_manifest(args):
    key = Path(args.hidden_key).read_bytes().strip() if args.hidden_key else None
    rows = lb.manifest(args.track, hidden_key=key, trials=args.trials)
    text = "".join(lb._canonical(row) + "\n" for row in rows)
    if args.output:
        Path(args.output).write_text(text)
        print(f"{len(rows)} trials, sha256 {lb.manifest_sha256(rows)}")
    else:
        sys.stdout.write(text)


def cmd_verify(args):
    public = json.loads((Path(args.submission) / "summary.json").read_text())
    results, hidden = _summarize(args.hidden_runs)
    if hidden["set"] != "hidden" or hidden["track"] != public["track"]:
        raise SystemExit("Hidden runs must use a hidden manifest of the submission's track")
    if hidden["counts"]["scored"] != hidden["counts"]["planned"]:
        raise SystemExit(f"Hidden rerun is incomplete ({hidden['counts']})")
    decision = lb.consistency(public, hidden)
    decision.update(submission=Path(args.submission).name, hidden_trials=hidden["counts"]["planned"],
                    hidden_successes=hidden["counts"]["successes"], hidden_summary_sha256=lb.sha256_text(
                        lb._canonical(hidden)))
    print(json.dumps(decision, indent=2))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    passthrough = []
    if "--" in argv:
        split = argv.index("--")
        argv, passthrough = argv[:split], argv[split + 1:]
    parser = argparse.ArgumentParser(prog="python -m handson.evaluation.suite", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run every planned trial of a track (resumable)")
    run.add_argument("--track", choices=sorted(lb.TRACKS), required=True)
    run.add_argument("--output", required=True, help="Run directory; rerun the same command to resume")
    run.add_argument("--manifest", help="Trial list file (hidden verification); default: public trials")
    run.add_argument("--tasks", help="Comma-separated subset for debugging; a submission needs all tasks")
    run.add_argument("--shard", help="i/n: run every n-th trial starting at i (several machines, one directory each)")
    run.add_argument("--gpus", help="Comma-separated CUDA device indices; default: inherit the environment")
    run.add_argument("--per-gpu", type=int, default=1, help="Concurrent episodes per GPU")
    run.add_argument("--max-attempts", type=int, default=3,
                     help="Attempts per trial; only infrastructure errors are retried")
    run.add_argument("--timeout", type=float, default=0, help="Seconds before an attempt is killed (0: none)")
    run.add_argument("--keep-scene-cache", action="store_true",
                     help="Keep each completed episode's converted scene cache (deleted by default)")
    run.add_argument("--allow-dirty", action="store_true", help=argparse.SUPPRESS)
    run.add_argument("--dry-run", action="store_true", help="Print evaluator commands and exit")
    summarize = sub.add_parser("summarize", help="Write results.jsonl and summary.json from run directories")
    summarize.add_argument("runs", nargs="+", help="Run directories of one track (shards may be combined)")
    summarize.add_argument("--out", help="Output directory, e.g. leaderboard/submissions/<id>")
    check = sub.add_parser("check", help="Validate submission directories")
    check.add_argument("submissions", nargs="+")
    check.add_argument("--git-repo", help="HandsOn checkout used to verify the evaluator commit")
    check.add_argument("--git-ref", help="Branch the evaluator commit must be on, e.g. origin/main")
    show = sub.add_parser("manifest", help="Print the planned trials of a track")
    show.add_argument("--track", choices=sorted(lb.TRACKS), required=True)
    show.add_argument("--hidden-key", help=argparse.SUPPRESS)
    show.add_argument("--trials", type=int, help="Trials per task (default: the track's plan)")
    show.add_argument("--output")
    verify = sub.add_parser("verify", help=argparse.SUPPRESS)
    verify.add_argument("submission")
    verify.add_argument("hidden_runs", nargs="+")
    args = parser.parse_args(argv)
    if args.command != "run" and passthrough:
        parser.error("evaluator arguments after -- only apply to `run`")
    if args.command == "run":
        if args.per_gpu < 1 or args.max_attempts < 1:
            parser.error("--per-gpu and --max-attempts must be positive")
        cmd_run(args, _passthrough(passthrough))
    else:
        dict(summarize=cmd_summarize, check=cmd_check, manifest=cmd_manifest, verify=cmd_verify)[args.command](args)


if __name__ == "__main__":
    main()
