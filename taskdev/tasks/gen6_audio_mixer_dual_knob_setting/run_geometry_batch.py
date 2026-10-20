"""Run the frozen engineering layouts sequentially, one guarded Kit at a time."""
import argparse
import json
from pathlib import Path
import subprocess
import time

TASK_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-index', required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--batch-label', required=True)
    parser.add_argument('--layouts', nargs='+', type=int, required=True)
    args = parser.parse_args()
    output = TASK_DIR / 'evidence' / args.batch_label
    output.mkdir(exist_ok=False)
    results = []
    for layout in args.layouts:
        label = f'{args.batch_label}_layout_{layout:02d}'
        command = ['python3', '-B', str(TASK_DIR / 'launch_remote_probe.py'),
                   '--gpu-index', args.gpu_index, '--gpu-uuid', args.gpu_uuid,
                   '--run-label', label, '--layout-index', str(layout),
                   '--steps', '60', '--reset-seeds', '1', '--angle-sweep']
        launch = subprocess.run(command, capture_output=True, text=True)
        (output / f'launch_{layout:02d}.log').write_text(launch.stdout + launch.stderr)
        if launch.returncode:
            raise RuntimeError(f'Guarded launch stopped for layout {layout}: {launch.stderr}')
        metadata = json.loads(launch.stdout)
        report_file = Path(metadata['output']) / 'report.json'
        started = time.monotonic()
        while True:
            proc = Path('/proc') / str(metadata['pid'])
            finished = not proc.exists() or (proc / 'stat').read_text().split()[2] == 'Z'
            if finished:
                break
            if time.monotonic() - started > 300:
                # Leave the owned probe inspectable; never launch a second Kit.
                raise TimeoutError(f'Inspect GEN6 PID {metadata["pid"]}; no next layout was launched')
            time.sleep(1.)
        report = json.loads(report_file.read_text())
        row = {'layout_index': layout, 'label': label,
               'probe_completed': report.get('probe_completed', False),
               'geometry_queries_passed': report.get('geometry_queries_passed', False),
               'error': report.get('error'), 'elapsed_s': report.get('elapsed_s')}
        results.append(row)
        (output / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
        print('GEN6_LAYOUT_RESULT', json.dumps(row), flush=True)
        if not row['probe_completed'] or not row['geometry_queries_passed']:
            raise RuntimeError(f'Geometry audit did not pass layout {layout}')
    print('GEN6_GEOMETRY_BATCH_COMPLETED', flush=True)


if __name__ == '__main__':
    main()
