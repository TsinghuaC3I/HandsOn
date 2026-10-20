"""Passive native-return logging and owned CPU video encoders."""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess


def json_safe(value, path='$', nonfinite=None):
    """Preserve native failure details without invalid JSON NaN or rechecking success."""
    if nonfinite is None:
        nonfinite = []
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    if hasattr(value, 'tolist'):
        value = value.tolist()
    if isinstance(value, float) and not math.isfinite(value):
        nonfinite.append(dict(path=path, value=str(value)))
        return None
    if isinstance(value, dict):
        return {str(k):json_safe(v, path+'.'+str(k), nonfinite) for k,v in value.items()}
    if isinstance(value, (list,tuple)):
        return [json_safe(v, path+f'[{i}]', nonfinite) for i,v in enumerate(value)]
    if value is None or isinstance(value, (str,bool,int,float)):
        return value
    raise TypeError('Unsupported passive log value at '+path+': '+type(value).__name__)


def native_return(result):
    if not isinstance(result, (list,tuple)) or len(result) != 5:
        raise ValueError('Unsupported native step return')
    observation, reward, terminated, truncated, info = result
    markers = []
    # Observation can contain images/trace payload; stored separately as live model inputs.
    value = json_safe(dict(reward=reward, terminated=terminated, truncated=truncated,
                           info=info), nonfinite=markers)
    return dict(**value, nonfinite_fields=markers, observation_type=type(observation).__name__,
                source='values returned by native env.step; no checker called by observer')


def append_event(stream, value):
    markers = []
    clean = json_safe(value, nonfinite=markers)
    if markers:
        clean['log_nonfinite_fields'] = markers
    stream.write(json.dumps(clean, allow_nan=False, separators=(',',':'))+'\n')
    stream.flush()


class Videos:
    def __init__(self, folder, images, fps, ffmpeg):
        self.items = {}
        self.frames = 0
        self.folder = folder
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
        for key in ('LD_LIBRARY_PATH','LD_PRELOAD','PYTHONPATH'):
            env.pop(key, None)
        try:
            for role,image in images.items():
                height,width,channels = image.shape
                if channels != 3 or image.dtype.name != 'uint8':
                    raise ValueError('Invalid native RGB video frame')
                log = (folder/f'ffmpeg_{role}.log').open('x')
                argv = [ffmpeg, '-nostdin', '-n', '-loglevel', 'warning',
                        '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', f'{width}x{height}',
                        '-framerate', str(fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264',
                        '-preset', 'fast', '-crf', '18', '-threads', '1', '-pix_fmt', 'yuv420p',
                        '-movflags', '+faststart', str(folder/f'{role}.mp4')]
                child = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=log, stderr=log, env=env)
                self.items[role] = dict(child=child, log=log, shape=image.shape, sha=hashlib.sha256(), argv=argv)
        except BaseException:
            self.close()
            raise

    def write(self, images):
        if set(images) != set(self.items):
            raise ValueError('Video roles changed')
        for role,row in self.items.items():
            image = images[role]
            if image.shape != row['shape'] or image.dtype.name != 'uint8':
                raise ValueError('Video frame shape/type changed')
            payload = image.tobytes()
            row['child'].stdin.write(payload)
            row['sha'].update(payload)
        self.frames += 1

    def close(self):
        report = {}
        for role,row in self.items.items():
            child = row['child']
            if child.stdin and not child.stdin.closed:
                try:
                    child.stdin.close()
                except BrokenPipeError:
                    pass
            try:
                exit_code = child.wait(timeout=60)
            except subprocess.TimeoutExpired:
                child.terminate()
                try:
                    exit_code = child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    exit_code = child.wait(timeout=10)
            row['log'].close()
            report[role] = dict(exit_code=exit_code, frames=self.frames, shape=list(row['shape']),
                               raw_rgb_sequence_sha256=row['sha'].hexdigest(), argv=row['argv'])
        return report
