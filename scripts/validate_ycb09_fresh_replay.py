#!/usr/bin/env python3
"""Review an acceptance-only episode with a fresh seeded PhysX process.

The isolated runtime has no Git checkout. This diagnostic verifies the exact
source-file hashes from the live probe and uses the existing platform-review
entry point; it does not issue a portable dataset certificate or publish data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "handson"):
    sys.path.insert(0, str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.probe_report.read_text())
    expected = source["source_sha256"]
    actual = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
              for name in expected}
    assert actual == expected, "runtime changed after the live acceptance probe"
    demo = Path(source["acceptance_recording_demo_dir"])
    assert (demo / "ACCEPTANCE_ONLY.json").is_file()
    os.environ["HANDSON_ISAACLAB_DEFER_CAMERA_RENDER"] = "1"
    os.environ["HANDSON_ISAACLAB_LIVE_MAIN_CAMERA_ONLY"] = "0"
    os.environ["HANDSON_RECORD_PHYSX_GROUND_TRUTH"] = "1"
    from scripts.certify_physx_replay import _json_safe, certify_demo_session

    def completed(path, result):
        # Persist before Isaac shutdown, which may exit the process directly.
        report = {
            "scope": "fresh_process_physical_review_of_acceptance_only_episode",
            "formal_portable_dataset_certification": False,
            "source_identity": "probe_source_files_sha256_match",
            "source_sha256": actual,
            "episode": str(path),
            "renderer_or_reviewer_pid": os.getpid(),
            "pass": result.get("status") == "passed",
            "physical_review": result,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(_json_safe(report), indent=2, allow_nan=False) + "\n")
        print("YCB09_FRESH_PHYSICAL_REVIEW", report["pass"], result.get("reason"), flush=True)

    certify_demo_session([demo], platform_review=True, on_episode_completed=completed)


if __name__ == "__main__":
    main()
