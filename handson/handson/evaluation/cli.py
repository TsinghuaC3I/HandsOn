"""Command line entry for the packaged native evaluator."""
import argparse
import contextlib
import json
import subprocess
import sys
import tempfile
import os
from pathlib import Path


def main():
    from handson.tasks import CONFIG_MAPPING
    from handson.policies.profiles import PROFILES, execution_settings
    from .backends import BACKENDS, supported_tasks, validate_task
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--task", choices=sorted(CONFIG_MAPPING))
    group.add_argument("--config", type=Path, help="Task YAML (legacy command spelling)")
    parser.add_argument("--list", action="store_true", help="List task identifiers supported by the selected backend")
    parser.add_argument("--check", action="store_true", help="Check native factories without starting Isaac Sim")
    parser.add_argument("--physics-backend", choices=BACKENDS, default="physx",
                        help="Physics engine; --list reports the selected engine's supported tasks")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument("--diagnostic-steps", type=int, default=0,
                        help="Run held measured commands, with no policy score (minimum 90)")
    parser.add_argument("--max-policy-steps", type=int, default=0,
                        help="Bound real policy execution for integration checks (minimum 90; no formal score)")
    parser.add_argument("--model", choices=PROFILES, help="Use the paper model's execution/history contract")
    parser.add_argument("--render-stride", type=int)
    parser.add_argument("--action-horizon", type=int)
    parser.add_argument("--history-frames", type=int)
    parser.add_argument("--history-stride", type=int)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--policy-url", help="Original physical EEF52 HTTP policy endpoint")
    parser.add_argument("--policy-metadata", type=Path, help="Expected HTTP model/checkpoint metadata JSON")
    parser.add_argument("--host", default="127.0.0.1", help="OpenPI websocket policy host")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.list:
        print("\n".join(supported_tasks(args.physics_backend)))
        return
    task = args.task
    if args.config:
        import yaml
        try:
            config = yaml.safe_load(args.config.read_text())
        except (OSError, yaml.YAMLError) as exc:
            parser.error(f"Cannot read task config: {exc}")
        if not isinstance(config, dict) or not isinstance(config.get("env_name"), str):
            parser.error("Task YAML must contain a string env_name")
        task = config["env_name"]
        if task not in CONFIG_MAPPING:
            parser.error(f"Unknown release task {task!r}; use --list for supported tasks")
    if task:
        try:
            validate_task(task, args.physics_backend)
        except ValueError as exc:
            parser.error(str(exc))
    if args.check:
        from .native import check_task
        report = []
        for name in [task] if task else supported_tasks(args.physics_backend):
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    row = check_task(name, physics_backend=args.physics_backend)
                report.append(dict(row, passed=True))
            except Exception as exc:
                report.append(dict(task=name, passed=False, error=repr(exc)))
        print(json.dumps(report, indent=2))
        if not all(row["passed"] for row in report):
            raise SystemExit(1)
        return
    if not task:
        parser.error("--task or --config is required")
    defaults = execution_settings(args.model) if args.model else dict(
        render_stride=30, action_horizon=30, history_frames=1, history_stride=30)
    for key, value in defaults.items():
        explicit = getattr(args, key)
        if args.model and explicit is not None and explicit != value:
            parser.error(f"--{key.replace('_', '-')} conflicts with the {args.model} paper profile")
        if explicit is None:
            setattr(args, key, value)
    if args.model and not args.diagnostic_steps:
        if not args.policy_url or not args.policy_metadata:
            parser.error("--model requires --policy-url and --policy-metadata from the paper model server")
        if not PROFILES[args.model].supports(task):
            parser.error(f"{args.model} has no implicit Transfer-16 routing")
        metadata = json.loads(args.policy_metadata.read_text())
        expected = dict(PROFILES[args.model].metadata(), model=args.model)
        if not all(metadata.get(k) == v for k, v in expected.items()):
            parser.error("Policy metadata does not match the selected model's paper profile")
        if metadata.get("task", task) != task:
            parser.error("Policy checkpoint belongs to a different task")
    if args.diagnostic_steps and args.diagnostic_steps < 90:
        parser.error("--diagnostic-steps must be at least 90")
    if args.max_policy_steps and args.max_policy_steps < 90:
        parser.error("--max-policy-steps must be at least 90")
    if args.max_policy_steps and args.diagnostic_steps:
        parser.error("--max-policy-steps cannot be combined with --diagnostic-steps")
    if args.action_horizon < 1 or args.history_frames < 1:
        parser.error("action horizon and history frames must be positive")
    from .schedule import RenderSchedule
    RenderSchedule(args.render_stride, args.action_horizon)
    if args.history_frames > 1 and (args.history_stride != args.render_stride or not args.policy_url):
        parser.error("history models require --policy-url and history stride equal to render stride")
    if not args.worker:
        from handson.robots.assets import require_benchmark_assets
        try:
            require_benchmark_assets()
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        if args.output.exists():
            parser.error("--output must be a new directory")
        # Kit's fast shutdown can exit(0) while unwinding an exception. Keep the
        # CLI outside Kit and require a completed receipt as well as its exit code.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", dir=args.output.parent, prefix=".evaluation-",
                                         suffix=".log", delete=False) as log:
            log_path = Path(log.name)
            child = subprocess.Popen([sys.executable, "-m", "handson.evaluation", *sys.argv[1:], "--worker"],
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                     errors="replace")
            try:
                for line in child.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
            finally:
                child.stdout.close()
                child.wait()
        args.output.mkdir(parents=True, exist_ok=True)
        os.replace(log_path, args.output / "runtime.log")
        result_file = args.output / "RESULT.json"
        if child.returncode or (args.output / "ERROR.json").exists() or not result_file.exists():
            error_file = args.output / "ERROR.json"
            if not error_file.exists():
                error_file.write_text(json.dumps(dict(error="Worker exited without a completed result",
                    child_exit_code=child.returncode, log="runtime.log", counts_as_model_failure=False), indent=2))
            print(f"Evaluation did not complete; inspect {args.output}", file=sys.stderr)
            # Popen uses negative return codes for signals; shell exit codes
            # use 128 + signal (SIGABRT -> 134, rather than wrapping -6 to 250).
            code = child.returncode
            raise SystemExit(128 - code if code < 0 else code or 1)
        print(result_file.read_text())
        return
    metadata = json.loads(args.policy_metadata.read_text()) if args.policy_metadata else {}
    if args.diagnostic_steps:
        policy = None
    elif args.policy_url:
        if not metadata:
            parser.error("--policy-url requires --policy-metadata with a checkpoint identity")
        policy = dict(transport="http_physical_eef52", endpoint=args.policy_url,
                      required_metadata=metadata, front_wire_key="front")
    else:
        policy = dict(transport="openpi_websocket", host=args.host, port=args.port,
                      required_metadata=metadata, front_wire_key="base")
    if policy is not None:
        policy.update(task=task, action_horizon=args.action_horizon, history_frames=args.history_frames,
                      history_stride=args.history_stride)
    from .native import run_episode
    result = run_episode(task, args.output, seed=args.seed, policy=policy,
                         diagnostic_steps=args.diagnostic_steps, render_stride=args.render_stride,
                         physics_backend=args.physics_backend, max_policy_steps=args.max_policy_steps)
    print(json.dumps(result, indent=2))
