"""Tiny synthetic encoder preflight; this is not task-render evidence."""
import argparse
import json
from pathlib import Path
import traceback

import av
import numpy as np

from handson.data.video_writer import Mp4VideoWriter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--encoder', choices=('auto', 'cpu', 'nvenc'), default='nvenc')
    args = parser.parse_args()
    report = {'scope': 'synthetic encoder dependency preflight only', 'passed': False,
              'requested_encoder': args.encoder}
    writer = Mp4VideoWriter.create_h264(fps=30, encoder=args.encoder)
    av.logging.set_level(av.logging.DEBUG)
    capture = av.logging.Capture(local=False)
    logs = capture.__enter__()
    try:
        if args.encoder == 'nvenc' and writer.codec != 'h264_nvenc':
            raise RuntimeError('The installed PyAV lacks the NVENC codec')
        path = args.output / 'encoder_probe.mp4'
        writer.start(str(path))
        for value in (32, 128, 224):
            writer.write_frame(np.full((256, 256, 3), value, np.uint8))
        report['encoder_implementation'] = writer._stream.codec_context.name
        writer.stop()
        with av.open(str(path)) as container:
            count = sum(1 for _ in container.decode(video=0))
        if count != 3:
            raise RuntimeError(f'Expected three decoded synthetic frames, got {count}')
        report.update(passed=True, codec=writer.codec, decoded_frames=count)
    except Exception:
        writer.abort()
        report['traceback'] = traceback.format_exc()
        raise
    finally:
        capture.__exit__(None, None, None)
        report['codec_log'] = [list(item) for item in logs]
        (args.output / 'encoder_preflight.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
