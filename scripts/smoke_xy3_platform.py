"""Exercise the real branch-backed collection HTTP/recorder lifecycle.

Run inside the authorized GPU namespace. UDP devices are isolated in a private
network namespace, so this checks no-input collection, not human teleoperation.
No launch, physics, origin, or success predicate is mocked.
"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import struct
import sys
import threading
import time
import traceback
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--targets", default="apple,011_banana,pear")
    args = parser.parse_args()
    if os.environ.get("XY3_PRIVATE_NET") != "1":
        os.environ["XY3_PRIVATE_NET"] = "1"
        os.execvp("unshare", ["unshare", "--net", sys.executable, __file__, *sys.argv[1:]])
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        # Bring up only this namespace's loopback; no host interface changes.
        fcntl.ioctl(sock.fileno(), 0x8914, struct.pack("16sH", b"lo", 0x49))
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for key, value in {
        "HANDSON_COLLECTION_CLEAN_STALE_ON_START": "0",
        "HANDSON_COLLECTION_TELEOP_STACK": "external",
        "HANDSON_COLLECTION_RUNTIME_DIR": str(output / "runtime"),
        "HANDSON_GUIDANCE_ROOT": str(output / "guidance"),
        "HANDSON_MODE_A_LOG_DIR": str(output / "mode-a-logs"),
        "HANDSON_REVIEW_AUTO_UPLOAD": "0", "HANDSON_BACKGROUND_V3_UPLOAD": "0",
        "HANDSON_WORK_PLAN_DESKTOP_NOTIFICATIONS": "0",
        "HANDSON_POSTPROCESS_MANUAL_ONLY": "1",
        "PYTHONUNBUFFERED": "1",
    }.items():
        os.environ[key] = value
    from collection_ui.server import CollectionState, CollectionRequestHandler, ThreadingHTTPServer
    state = CollectionState(root, sys.executable, output / "demos", isaacsim_python_bin=sys.executable)
    server = ThreadingHTTPServer(("127.0.0.1", 0), CollectionRequestHandler)
    server.state = state
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    report = {"scope": "ordinary HTTP launch, real recorder, isolated no-input UDP",
              "human_teleoperation_verified": False, "successful_episode_verified": False, "rounds": []}

    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    def request(path, payload=None):
        req = urllib.request.Request(base + path, data=None if payload is None else json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json", "Origin": base})
        with urllib.request.urlopen(req, timeout=180) as response:
            return json.load(response)

    def log_text():
        return state.collection_log_path.read_text(errors="replace") if state.collection_log_path else ""

    def wait_for(predicate, description, timeout=180):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            if state.collection_process is not None and state.collection_process.poll() is not None:
                raise RuntimeError(f"Recorder exited while waiting for {description}: " + log_text()[-6000:])
            time.sleep(.5)
        raise RuntimeError(f"Timeout waiting for {description}: " + log_text()[-6000:])

    try:
        report["catalog"] = next(x for x in request("/api/tasks")["items"]
                                 if x["name"] == "eval_comb04_place_fruit_and_photograph_logo")
        assert report["catalog"]["sim_backends"] == ["isaacsim"]
        for index in range(args.rounds):
            target = args.targets.split(",")[index % len(args.targets.split(","))]
            row = {"index": index, "target": target}
            report["rounds"].append(row)
            row["start"] = request("/api/collection/start", {
                "task_name": "eval_comb04_place_fruit_and_photograph_logo", "sim_backend": "isaacsim",
                "arm_type": "rokae", "hand_type": "wuji1", "target_label": target,
                "render_mode": "rgb_array", "randomize": True,
                "out_dir": str(output / "demos"), "collector_id": f"xy3_auto_{index}",
                "collection_note": "XY3 no-input lifecycle diagnostic; not training data",
            })
            save()
            assert row["start"]["success"], row["start"]
            wait_for(lambda: "Environment reset complete for eval_comb04_place_fruit_and_photograph_logo" in log_text(), "initial reset")
            row["initial_reset_complete"] = True
            time.sleep(3)
            session = json.loads(state.collection_session_path.read_text())
            row["session"] = session
            candidates = []
            for path in Path("/proc").iterdir():
                if not path.name.isdigit():
                    continue
                pid = int(path.name)
                if pid != state.collection_process.pid and not state._pid_descends_from(pid, state.collection_process.pid):
                    continue
                try:
                    cmd = (path / "cmdline").read_bytes().replace(b"\0", b" ")
                except OSError:
                    continue
                if str(root / "scripts/record_demos_zarr.py").encode() in cmd:
                    candidates.append(pid)
            # The normal launcher can use a deferred-arm supervisor; choose
            # its actual leaf recorder rather than assuming a worker env flag.
            candidates = [pid for pid in candidates if not any(
                other != pid and state._pid_descends_from(other, pid) for other in candidates)]
            assert len(candidates) == 1, candidates
            old_count = log_text().count("Hard reset complete: fresh PhysX Scene")
            old_ready = log_text().count("[Recorder] Stable physical frame 0:")
            os.kill(candidates[0], signal.SIGUSR1)
            wait_for(lambda: log_text().count("Hard reset complete: fresh PhysX Scene") > old_count,
                     "ordinary recorder manual reset")
            wait_for(lambda: log_text().count("[Recorder] Stable physical frame 0:") > old_ready,
                     "stable recording origin after manual reset")
            row["manual_reset_complete"] = True
            time.sleep(3)
            row["stop"] = request("/api/collection/stop", {})
            row["log_path"] = str(state.collection_log_path)
            row["log_tail"] = log_text()[-14000:]
            assert row["stop"]["success"], row["stop"]
            deadline = time.monotonic() + 30
            while state._collection_script_pids() and time.monotonic() < deadline:
                time.sleep(.5)
            assert not state._collection_script_pids(), "Own recorder process remained after stop"
            row["stopped"] = True
            save()
            print(f"XY3_PLATFORM_ROUND {index} passed", flush=True)
        report["completed"] = True
    except BaseException:
        report["error"] = traceback.format_exc()
        traceback.print_exc()
    finally:
        if state.collection_process is not None and state.collection_process.poll() is None:
            state.stop_collection()
        # stop_collection is asynchronous. Keep its cleanup threads alive
        # even if a diagnostic assertion fails after the recorder launches.
        deadline = time.monotonic() + 60
        while state._collection_script_pids() and time.monotonic() < deadline:
            time.sleep(.5)
        remaining = state._collection_script_pids()
        if remaining:
            report["forced_cleanup_of_own_pids"] = sorted(remaining)
            for pid, cmd in remaining.items():
                assert str(root / "scripts/record_demos_zarr.py") in cmd
                pgid = os.getpgid(pid)
                assert pgid != os.getpgrp()
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        server.shutdown()
        server.server_close()
        save()
    return 0 if report.get("completed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
