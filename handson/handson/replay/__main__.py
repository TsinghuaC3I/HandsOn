"""Run a bundled episode without interpreting seed reset as historical t0."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from .bundle import BUNDLES, load_episode


def stop_worker(child):
    """Reap only this command's worker process group."""
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()


def interrupted(signum, frame):
    raise KeyboardInterrupt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', default='e015401', choices=sorted(BUNDLES))
    parser.add_argument('--control', choices=['eef52', 'recorded_joint_targets'], default='eef52',
                        help='EEF52 commands, or explicit original substep motor targets when bundled')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check', action='store_true', help='Verify bundle and source identities on CPU, then exit')
    parser.add_argument('--render-stride', type=int, default=10)
    parser.add_argument('--complete-recording', action='store_true',
                        help='Diagnostic: finish remaining recorded commands after native success, never after failure')
    parser.add_argument('--timeout', type=int, default=1200, help='Worker wall-clock limit in seconds')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.render_stride < 1 or args.timeout < 1:
        parser.error('render stride and timeout must be positive')
    try:
        episode = load_episode(args.episode)
        episode.require_control(args.control)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    if args.check:
        print(json.dumps(dict(episode=args.episode, task=episode.manifest['task'],
            bundle_verified=True, runtime_identity_verified=True, action_count=len(episode.actions),
            paper_variant=episode.manifest.get('paper_variant'), control=args.control,
            available_controls=['eef52'] + (['recorded_joint_targets'] if episode.joint_targets is not None else []),
            scope='CPU integrity check; no simulation performed'), indent=2))
        return
    if args.output is None:
        parser.error('--output is required unless --check is used')
    if args.worker:
        from .runtime import run
        run(args.episode, args.output, render_stride=args.render_stride, complete_recording=args.complete_recording,
            control_mode=args.control, isolated_worker=True)
        return
    if args.output.exists():
        parser.error('--output must be a new directory')
    args.output.mkdir(parents=True)
    episode_output = args.output / 'episode'
    command = [sys.executable, '-m', 'handson.replay', '--episode', args.episode,
               '--output', str(episode_output), '--render-stride', str(args.render_stride), '--worker']
    command.extend(['--control', args.control])
    if args.complete_recording:
        command.append('--complete-recording')
    print(f'Replay output: {args.output.resolve()} (runtime.log contains live progress)', flush=True)
    with (args.output / 'runtime.log').open('w') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        previous_handler = signal.signal(signal.SIGTERM, interrupted)
        try:
            code = child.wait(timeout=args.timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            stop_worker(child)
            code = 124
        finally:
            signal.signal(signal.SIGTERM, previous_handler)
    result = episode_output / 'RESULT.json'
    # Kit may exit(0) while unwinding a constructor exception.
    if code or not result.is_file() or (episode_output / 'ERROR.json').exists():
        (args.output / 'ERROR.json').write_text(json.dumps(dict(
            error='Replay worker did not complete', child_exit_code=code,
            log='runtime.log', formal_benchmark_trial=False), indent=2) + '\n')
        raise SystemExit(128 - code if code < 0 else code or 1)
    value = json.loads(result.read_text())
    print(json.dumps({k: v for k, v in value.items() if k != 'terminal'}, indent=2))


if __name__ == '__main__':
    main()
