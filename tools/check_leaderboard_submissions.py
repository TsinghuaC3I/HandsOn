"""Check every leaderboard submission and the maintainers' verification records.

Used by the leaderboard CI workflow; runs on CPU without Isaac Sim.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATUSES = {"verified", "rejected", "pending"}


def annotate(level, path, message):
    if os.environ.get("GITHUB_ACTIONS") == "true":
        print(f"::{level} file={path}::{message}")
    print(f"{level.upper()} {path}: {message}")


def main():
    from handson.evaluation import leaderboard as lb
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--git-ref", default="origin/main",
                        help="Branch that evaluator commits must be on (empty: skip)")
    args = parser.parse_args()
    folder = ROOT / "leaderboard" / "submissions"
    submissions = sorted(p for p in folder.iterdir() if p.is_dir()) if folder.is_dir() else []
    failed = False
    for path in submissions:
        report = lb.validate_submission(path, git_repo=ROOT, git_ref=args.git_ref or None)
        rel = path.relative_to(ROOT)
        for message in report["errors"]:
            annotate("error", rel, message)
        for message in report["warnings"]:
            annotate("warning", rel, message)
        failed |= bool(report["errors"])
        if not report["errors"]:
            print(f"OK {rel}")
    record = ROOT / "leaderboard" / "verifications.json"
    data = json.loads(record.read_text())
    if data.get("schema") != "handson_leaderboard_verifications_v1" or not isinstance(data.get("entries"), dict):
        annotate("error", record.relative_to(ROOT), "unexpected verification file schema")
        failed = True
    names = {path.name for path in submissions}
    for name, entry in data.get("entries", {}).items():
        if name not in names:
            annotate("error", record.relative_to(ROOT), f"{name} has no submission directory")
            failed = True
        if not isinstance(entry, dict) or entry.get("status") not in STATUSES:
            annotate("error", record.relative_to(ROOT), f"{name} needs a status in {sorted(STATUSES)}")
            failed = True
    print(f"Checked {len(submissions)} submissions")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
