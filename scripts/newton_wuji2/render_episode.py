#!/usr/bin/env python3
"""Render a complete raw Newton/Wuji2 episode with the released V16 appearance."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import zipfile

RUNTIME_SHA256 = '2a7908dcb2c5984ad4c253141545e48ebf49f137a6c17a1f3d9071e3a3dad33b'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--runtime-archive', type=Path)
    parser.add_argument('--python', default=sys.executable, help='Isaac Sim 5.1 Python executable')
    parser.add_argument('--isaac-lab', type=Path, default=os.environ.get('ISAAC_LAB_DIR'))
    parser.add_argument('--max-frames', type=int, default=0, help='0 means the complete trajectory; positive values are diagnostic only')
    parser.add_argument('--spp', type=int, default=128)
    args = parser.parse_args()
    if not 16 <= args.spp <= 1024 or args.max_frames < 0:
        parser.error('spp must be 16..1024 and max-frames must be nonnegative')
    root, source, output = args.runtime.resolve(), args.episode.resolve(strict=True), args.output.resolve()
    if source == output or source in output.parents:
        parser.error('Use a separate output directory to preserve raw source data')
    meta = json.loads((source / 'collection_meta.json').read_text())
    if meta['hand_type'] != 'wuji2' or meta['sim_backend'] != 'isaacsim_newton':
        raise RuntimeError('Expected a Newton/Wuji2 source episode')
    if not (root / 'source/scripts/isaac_render_arm_replay.py').is_file():
        archive = args.runtime_archive
        if archive is None:
            from huggingface_hub import hf_hub_download
            archive = Path(hf_hub_download('Cyan666/HandsOn-Datasets',
                'newton_wuji2/renderer/replay-runtime.tar', repo_type='dataset'))
        if sha(archive) != RUNTIME_SHA256:
            raise RuntimeError('Replay runtime archive checksum mismatch')
        root.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                target = (root / member.name).resolve()
                if root not in target.parents or not member.isfile():
                    raise RuntimeError('Unsafe runtime archive member: ' + member.name)
            tar.extractall(root)
    output.mkdir(parents=True, exist_ok=True)
    identity = {'source_archive_sha256': sha(source / 'replay.zarr.zip'),
                'source_episode_id': source.name, 'source_frames': meta['step_count'],
                'fps': meta['data_fps'], 'spp': args.spp, 'max_frames': args.max_frames}
    identity_path = output / 'SOURCE.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise RuntimeError('Output belongs to a different source or rendering configuration')
    for name in [p.name for p in source.glob('DATASET*.json')] + ['replay.zarr.zip', 'collection_meta.json', 'replay_manifest.json', 'recording_schema.json',
                 'DATASET_NEWTON_V1.json', 'DATASET_NEWTON_V2.json', 'manus_raw_schema.json']:
        if (source / name).exists() and not (output / name).exists():
            shutil.copy2(source / name, output / name)
    if (source / 'replay_assets').exists():
        shutil.copytree(source / 'replay_assets', output / 'replay_assets', dirs_exist_ok=True)
    with zipfile.ZipFile(output / 'replay.zarr.zip') as archive:
        for name in archive.namelist():
            if output not in (output / 'replay.zarr' / name).resolve().parents:
                raise RuntimeError('Unsafe source archive member')
        if not (output / 'replay.zarr').exists():
            archive.extractall(output / 'replay.zarr')
    identity_path.write_text(json.dumps(identity, indent=2) + '\n')
    code = Path(__file__).resolve().parent
    env = os.environ.copy()
    pythonpath = [str(root), str(root / 'source'), str(root / 'source/dexjoco')]
    if args.isaac_lab:
        pythonpath.append(str(args.isaac_lab.resolve() / 'source/isaaclab'))
    if env.get('PYTHONPATH'):
        pythonpath.append(env['PYTHONPATH'])
    env.update(PYTHONPATH=os.pathsep.join(pythonpath), HANDSON_WUJI2_RUNTIME=str(root),
        HANDSON_WUJI2_EPISODE=str(output), HANDSON_RENDER_MAX_FRAMES=str(args.max_frames),
        WUJI2_VISUAL_BUNDLE='wuji2_photo_glove_link_v14', WUJI2_CLOTH_MODULE='wuji2_cloth_bridges_v14',
        WUJI2_CLOTH_PROFILE='cloth_joint_profile_v14.json', WUJI2_JOINT_CLOTH='1', WUJI2_CAMERA_TRACE='1',
        DEXJOCO_WUJI2_VISUAL='knit_v1', DEXJOCO_WUJI2_CAPTURE_ACCEL='0',
        DEXJOCO_WUJI2_VISUAL_REPORT=str(output / 'WUJI2_HAND_VISUAL.json'),
        DEXJOCO_V16_RUNTIME=str(root / 'v16'), DEXJOCO_V16_PREVIEW_SPP=str(args.spp),
        DEXJOCO_V16_APPEARANCE_REPORT=str(output / 'V16_APPEARANCE.json'),
        DEXJOCO_ISAACSIM_ULTRA_RENDERING='1', DEXJOCO_ISAACSIM_OPTIX_DENOISER='1',
        DEXJOCO_ISAACSIM_PATH_TRACING_SPP=str(args.spp), DEXJOCO_ISAACSIM_PATH_TRACING_TOTAL_SPP=str(args.spp),
        DEXJOCO_ISAACSIM_ACCURATE_CAMERA_VISIBILITY='1', DEXJOCO_VIDEO_ENCODER='cpu',
        DEXJOCO_ISAACLAB_CACHE=str(root / 'cache'), PYTHONDONTWRITEBYTECODE='1')
    subprocess.run([args.python, str(code / 'prepare_runtime.py'), str(root)], env=env, check=True)
    subprocess.run([args.python, str(code / 'render_checkpointed.py'),
        '--arm_replay_demo_dir=' + str(output), '--exp_name=' + meta['task_name'],
        '--arm_replay_fps=30', '--max_replay_steps=' + str(args.max_frames)],
        env=env, cwd=root / 'source', check=True)
    status = json.loads((output / 'deferred_arm_replay_status.json').read_text())
    if status.get('status') != 'complete' or not status.get('artifact_validation_passed'):
        raise RuntimeError('Canonical video validation did not pass')
    print('Complete canonical videos:', output / 'videos')


if __name__ == '__main__':
    main()
