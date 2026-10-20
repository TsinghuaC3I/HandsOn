"""One-model HTTP/NPZ service; never imports Isaac Sim or launches GPU workers."""
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import io
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
from .common import (ROOT, VIEWS, load_policy, path_option, read, require, sha256,
                     validate_observation)
from .profiles import PROFILES, noise_seed
from ..interfaces import validate_actions
from ..tasks.registry import TASKS
from ..tasks.language import allowed_prompts


def checkpoint_identity(path):
    """Hash the selected file or params/state shards without absolute-path salts."""
    if path.is_file():
        return sha256(path)
    import hashlib
    files = [p for folder in ("params", "state") for p in (path / folder).rglob("*") if p.is_file()]
    require(files, "No checkpoint params/ or state/ files")
    digest = hashlib.sha256()
    for file in sorted(files):
        digest.update(str(file.relative_to(path)).encode() + b"\0")
        digest.update(sha256(file).encode() + b"\n")
    return digest.hexdigest()


def predict_request(policy, profile, model, packet, bound_task=None):
    with np.load(io.BytesIO(packet), allow_pickle=False) as data:
        images = {key: np.array(data[key]) for key in VIEWS}
        prompt = str(data["prompt"].item())
        task = str(data["task"].item()) if "task" in data else bound_task
        state = validate_observation(data["state"], images, prompt, profile.history_frames)
        seed, request = int(data["seed"].item()), int(data["request"].item())
    require(seed >= 0 and request >= 0, "Negative seed/request")
    if bound_task:
        require(task == bound_task, "Task differs from the server's selected checkpoint")
    require(task in TASKS, "Unknown or missing public task identifier")
    require(profile.supports(task), f"{model} has no implicit Transfer-16 routing")
    require(prompt in allowed_prompts(task),
            "Instruction is outside the paper's evaluated language bank")
    noise = noise_seed(model, seed, request, task)
    started = time.monotonic()
    actions = validate_actions(policy.predict(state, images, prompt, noise), profile.model_horizon)
    return dict(actions=actions[:profile.action_horizon].tolist(), noise_seed=noise,
                request=request, task=task, inference_seconds=time.monotonic() - started)


def handler_for(policy, model, metadata, bound_task=None):
    profile = PROFILES[model]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, status, value):
            payload = json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            self.respond(200, metadata)

        def do_POST(self):
            try:
                size = int(self.headers.get("Content-Length", "0"))
                require(0 < size <= 96 * 1024**2, "Invalid observation packet size")
                self.respond(200, predict_request(policy, profile, model, self.rfile.read(size), bound_task))
            except Exception as exc:
                traceback.print_exc()
                self.respond(400 if isinstance(exc, (ValueError, KeyError)) else 500,
                             dict(error=str(exc)))
    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--model", choices=PROFILES)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-sha256", help="Expected model-file or directory digest")
    parser.add_argument("--foundation", type=Path)
    parser.add_argument("--normalization", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--text-cache", type=Path)
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--task", help="Required for single-task DP; optional task binding for other models")
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--metadata-output", type=Path, default=Path("policy-metadata.json"))
    options = parser.parse_args(argv)
    if options.list:
        print(json.dumps({k: p.metadata() for k, p in PROFILES.items()}, indent=2, ensure_ascii=False))
        return
    if not options.model or options.checkpoint is None:
        parser.error("--model and --checkpoint are required")
    for key in {"foundation", "normalization", "tokenizer", "text_cache", "model_config"} - PROFILES[options.model].asset_options:
        if getattr(options, key) is not None:
            parser.error(f"--{key.replace('_', '-')} is not used by {options.model}")
    options.checkpoint = path_option(options, "checkpoint")
    if options.metadata_output.exists():
        parser.error("--metadata-output must be a new file")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("WANDB_MODE", "disabled")
    digest = checkpoint_identity(options.checkpoint)
    if options.checkpoint_sha256:
        require(digest == options.checkpoint_sha256, "Checkpoint digest mismatch")
    policy = load_policy(options)
    metadata = dict(PROFILES[options.model].metadata(), model=options.model,
                    checkpoint=str(options.checkpoint), checkpoint_sha256=digest,
                    adapter_source_manifest_sha256=sha256(ROOT / "source_provenance.json"),
                    packaged_assets_sha256={str(p.relative_to(ROOT / "assets")): sha256(p)
                        for p in sorted((ROOT / "assets" / options.model).glob("*")) if p.is_file()},
                    load_receipt=getattr(policy, "receipt", {}))
    if options.task:
        metadata["task"] = options.task
    for name in ("normalization", "tokenizer", "text_cache", "model_config"):
        path = getattr(options, name)
        if path and path.is_file():
            metadata[name + "_sha256"] = sha256(path)
    server = HTTPServer((options.bind, options.port), handler_for(policy, options.model, metadata, options.task))
    options.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    # Publish only after the checkpoint loads and the listener is bound.
    with options.metadata_output.open("x") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")
    print(json.dumps(metadata), flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
