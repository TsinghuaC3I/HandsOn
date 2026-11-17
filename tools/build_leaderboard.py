"""Publish the unified HandsOn leaderboard and the path-free 100-call Fable snapshot.

    python tools/build_leaderboard.py            # rebuild leaderboard.json and the docs table
    python tools/build_leaderboard.py --check    # fail if the published files are stale
    python tools/build_leaderboard.py --fable-source SNAPSHOT.json   # also refresh the Fable snapshot

Entries come from three sources: the manuscript's trained models, the HandsOn
team's prompted-controller runs (GPT-6, Fable 5.2), and pull-request
submissions under ``leaderboard/submissions`` (see leaderboard/README.md).
"""

import argparse
import collections
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "results"
SUBMISSIONS = ROOT / "leaderboard" / "submissions"
VERIFICATIONS = ROOT / "leaderboard" / "verifications.json"
TABLE = ROOT / "docs" / "leaderboard.md"
BEGIN, END = "<!-- leaderboard-table:begin -->", "<!-- leaderboard-table:end -->"
STATUS_LABEL = {"handson_team": "HandsOn team", "verified": "Verified", "self_reported": "Self-reported"}

# Core and Transfer success rates are the one-decimal values in the manuscript.
PAPER_MODELS = [
    ("OpenWAM", 29.1, 0.4),
    ("FastWAM", 26.4, 0.0),
    ("pi0.5", 24.9, 0.1),
    ("DM0.5", 21.4, 0.9),
    ("LingBot-VLA", 19.7, 0.0),
    ("HY-Embodied", 18.9, 0.0),
    ("Motus", 18.1, 0.1),
    ("Spatial Forcing", 12.4, 0.1),
]


def save(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def refresh_fable(fable_source):
    source_bytes = fable_source.read_bytes()
    source = json.loads(source_bytes)
    gpt = json.loads((RESULTS / "gpt6-results-20261019.json").read_text())

    assert source["schema"] == "fable52_observed_prefix_100_v1"
    assert source["cutoff_completed_model_responses"] == 100
    assert source["original_protocol_max_model_calls"] == 500
    assert len(source["trials"]) == 660
    assert source["audited"] + source["original_queue_status"].get("running", 0) == 660
    assert len(gpt["tasks"]) == 66
    catalog = {task["runtime"]: task for task in gpt["tasks"]}
    assert len(catalog) == 66
    grouped = collections.defaultdict(list)
    public_trials = []
    for trial in source["trials"]:
        task = catalog[trial["task"]]
        suite = "Core-50" if trial["scope"] == "MAIN50" else "Transfer-16"
        assert task["suite"] == suite
        outcome = trial["call100_outcome"]
        assert outcome in {"success_within_100", "native_failure_within_100", "truncated_at_100"}
        assert trial["observed_responses"] > 100 or outcome != "truncated_at_100"
        assert trial["audited"] or (outcome == "truncated_at_100" and trial["observed_responses"] > 100)
        grouped[trial["task"]].append(outcome)
        public_trials.append({
            "id": trial["trial"], "runtime": trial["task"], "suite": suite,
            "outcome_at_100": outcome, "observed_responses": trial["observed_responses"],
            "original_status": trial["original_status"], "audited": trial["audited"],
        })
    assert len({trial["id"] for trial in public_trials}) == 660
    tasks = []
    for task in gpt["tasks"]:
        outcomes = grouped[task["runtime"]]
        assert len(outcomes) == 10
        counts = collections.Counter(outcomes)
        tasks.append({
            "id": task["id"], "name": task["name"], "suite": task["suite"],
            "runtime": task["runtime"], "trials": 10,
            "successes_within_100": counts["success_within_100"],
            "outcomes_at_100": dict(counts),
        })
    counts = collections.Counter(trial["outcome_at_100"] for trial in public_trials)
    assert dict(counts) == source["counts"]
    assert counts["success_within_100"] == 29
    assert sum(task["successes_within_100"] for task in tasks) == 29
    snapshot = datetime.fromtimestamp(source["at"], timezone.utc).isoformat()
    fable = {
        "schema": "handson_fable52_posthoc_call100_v1",
        "snapshot_utc": snapshot,
        "complete_original_audit": source["audited"] == 660,
        "audited_original_trials": source["audited"],
        "original_queue_status": source["original_queue_status"],
        "original_max_model_calls": 500,
        "posthoc_cutoff_completed_model_responses": 100,
        "method": "Each accepted 500-call trajectory is scored at its first 100 completed model responses. A success counts only if native termination occurred by response 100. Trajectories continuing past 100 count as unsuccessful at the cutoff. This is not a 100-call rerun.",
        "counts": dict(counts),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "source_manifest_sha256": source["source"]["manifest_sha256"],
        "source_queue_sha256": source["source"]["queue_sha256"],
        "source_audit_sha256": source["source"]["audit_sha256"],
        "tasks": tasks,
        "trials": sorted(public_trials, key=lambda row: row["id"]),
    }
    save(RESULTS / "fable52-call100.json", fable)
    return fable


def team_entries(fable):
    """Manuscript models, prompted controllers and the single-task DP reference."""
    entries = []
    for name, core, transfer in PAPER_MODELS:
        entries.append({
            "model": name, "core_percent": core, "transfer_percent": transfer,
            "overall_percent": round((50 * core + 16 * transfer) / 66, 1),
            "overall_derived_from_rounded_suite_rates": True,
            "protocol": "Paper trained checkpoint: 100 planned Core trials/task; 50 planned Transfer trials/task.",
            "source": "https://arxiv.org/abs/2610.26765",
            "track": "multi_task", "evaluation": "handson_team", "contributor": "HandsOn team", "ranked": True,
        })
    assert sum(task["successes_within_100"] for task in fable["tasks"]) == 29
    for name, core_n, transfer_n, description, link in [
        ("GPT-6 (move_eef)", 26, 3,
         "Prompted controller; recorded context and call budgets vary across completed trials.",
         "docs/gpt6-results.md"),
        ("Fable 5.2 (move_eef)", 26, 3,
         "Post-hoc 100-response cutoff of original 500-call trajectories; not a 100-call rerun.",
         "docs/fable52-results.md"),
    ]:
        entries.append({
            "model": name, "core_percent": 100 * core_n / 500,
            "transfer_percent": 100 * transfer_n / 160,
            "overall_percent": 100 * (core_n + transfer_n) / 660,
            "core_successes": core_n, "transfer_successes": transfer_n,
            "core_trials": 500, "transfer_trials": 160,
            "overall_derived_from_rounded_suite_rates": False,
            "protocol": description, "source": link,
            "track": "zero_shot", "evaluation": "handson_team", "contributor": "HandsOn team", "ranked": True,
        })
    dp = {
        "model": "DP (single-task)", "core_percent": 9.7,
        "transfer_percent": None, "overall_percent": None,
        "overall_derived_from_rounded_suite_rates": False,
        "protocol": "Core-only single-task reference; 38 tasks use an earlier goal sampler. Not ranked across 66 tasks.",
        "source": "https://arxiv.org/abs/2610.26765",
        "track": "single_task", "evaluation": "handson_team", "contributor": "HandsOn team", "ranked": False,
    }
    return entries, dp


def submission_entries():
    """Pull-request submissions that pass the leaderboard check, with their status."""
    import yaml
    from handson.evaluation import leaderboard as lb
    records = json.loads(VERIFICATIONS.read_text())["entries"]
    entries = []
    folders = sorted(p for p in SUBMISSIONS.iterdir() if p.is_dir()) if SUBMISSIONS.is_dir() else []
    for folder in folders:
        report = lb.validate_submission(folder, git_repo=ROOT)
        if report["errors"]:
            raise SystemExit(f"{folder.name} fails the leaderboard check: {report['errors'][0]}")
        status = records.get(folder.name, {}).get("status")
        if status == "rejected":
            continue
        meta = yaml.safe_load((folder / "metadata.yaml").read_text())
        summary = json.loads((folder / "summary.json").read_text())
        evaluation = "verified" if status == "verified" else "self_reported"
        suites, overall = summary["suites"], summary["overall"]
        execution = summary["execution"]
        entry = {
            "model": f"{meta['model']['name']} {meta['model']['version']}".strip(),
            "core_percent": suites["core"]["percent"],
            "transfer_percent": suites["transfer"]["percent"] if "transfer" in suites else None,
            "overall_percent": overall["percent"] if overall else None,
            "core_successes": suites["core"]["successes"], "core_trials": suites["core"]["planned"],
            "overall_derived_from_rounded_suite_rates": False,
            "protocol": (f"{lb.PROTOCOL} public trials ({lb.TRACKS[summary['track']]['label']}); "
                         f"{execution['transport']}, horizon {execution['action_horizon']}"),
            "source": f"leaderboard/submissions/{folder.name}",
            "track": summary["track"], "evaluation": evaluation,
            "contributor": meta["submission"]["team"],
            "ranked": evaluation == "verified" and overall is not None,
            "submission_id": folder.name, "listed_at": f"{folder.name[:4]}-{folder.name[4:6]}-{folder.name[6:8]}",
            "ci95_percent": (overall or suites["core"])["ci95_percent"],
            "links": {key: meta["links"].get(key) for key in ("report", "code", "checkpoint", "artifacts")},
        }
        if "transfer" in suites:
            entry.update(transfer_successes=suites["transfer"]["successes"],
                         transfer_trials=suites["transfer"]["planned"])
        if status == "verified":
            entry["verification"] = {k: v for k, v in records[folder.name].items() if k != "status"}
        entries.append(entry)
    return entries


def compose(fable):
    team, dp = team_entries(fable)
    submitted = submission_entries()
    ranked = sorted([row for row in team + submitted if row["ranked"]],
                    key=lambda row: (-row["overall_percent"], row["model"]))
    listed = sorted([row for row in submitted if not row["ranked"] and row["overall_percent"] is not None],
                    key=lambda row: (-row["overall_percent"], row["model"]))
    core_only = [dp] + sorted([row for row in submitted if row["overall_percent"] is None],
                              key=lambda row: (-row["core_percent"], row["model"]))
    dates = [fable["snapshot_utc"]] + [f"{row['listed_at']}T00:00:00+00:00" for row in submitted]
    return {
        "schema": "handson_unified_leaderboard_v1",
        "updated_utc": max(dates),
        "metric": "Native task success; Core-50 and Transfer-16 are task-macro rates. Overall is the 66-task macro mean, weighted by 50 and 16 tasks.",
        "caveat": "Models differ in observations, action interface, interaction budgets and trial counts. Trained-model overall rates are derived from manuscript suite rates rounded to one decimal place.",
        "fable_snapshot_complete_original_audit": fable["complete_original_audit"],
        "fable_audited_original_trials": fable["audited_original_trials"],
        "submission_protocol": "handson-eval-v1; see leaderboard/README.md",
        "models": ranked + listed + core_only,
    }


def _percent(value):
    return f"{value:.1f}%"


def table(board):
    """Markdown table: ranked rows (ties share a rank), then self-reported and core-only rows."""
    lines = ["| Rank | Model | Core-50 | Transfer-16 | Overall | Evaluation |",
             "| ---: | --- | ---: | ---: | ---: | --- |"]
    ranked = [row for row in board["models"] if row["ranked"]]
    for index, row in enumerate(ranked):
        same = [i for i, other in enumerate(ranked)
                if round(other["overall_percent"], 1) == round(row["overall_percent"], 1)]
        rank = f"{same[0] + 1}=" if len(same) > 1 else str(index + 1)
        lines.append(_row(rank, row))
    for row in board["models"]:
        if not row["ranked"]:
            lines.append(_row("-", row))
    return "\n".join(lines)


def _row(rank, row):
    name = row["model"].replace("(move_eef)", "(`move_eef`)")
    source = row["source"]
    if source.startswith("docs/"):
        name = f"[{name}]({source[len('docs/'):]})"
    elif source.startswith("leaderboard/"):
        name = f"[{name}](../{source})"
    transfer = "not evaluated" if row["transfer_percent"] is None else _percent(row["transfer_percent"])
    overall = "not ranked" if row["overall_percent"] is None else _percent(row["overall_percent"])
    return (f"| {rank} | {name} | {_percent(row['core_percent'])} | {transfer} | {overall} | "
            f"{STATUS_LABEL[row['evaluation']]} |")


def render_docs(board):
    text = TABLE.read_text()
    pattern = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END), re.S)
    if not pattern.search(text):
        raise SystemExit(f"{TABLE} lacks the {BEGIN} / {END} markers")
    return pattern.sub(lambda _: f"{BEGIN}\n{table(board)}\n{END}", text)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fable-source", type=Path, help="Private Fable snapshot to republish")
    parser.add_argument("--check", action="store_true", help="Fail if the published files are out of date")
    args = parser.parse_args()
    if args.check and args.fable_source:
        parser.error("--check does not refresh the Fable snapshot")
    if args.fable_source:
        fable = refresh_fable(args.fable_source)
    else:
        fable = json.loads((RESULTS / "fable52-call100.json").read_text())
    board = compose(fable)
    board_text = json.dumps(board, indent=2, ensure_ascii=False) + "\n"
    docs_text = render_docs(board)
    if args.check:
        stale = [str(path.relative_to(ROOT)) for path, text in
                 ((RESULTS / "leaderboard.json", board_text), (TABLE, docs_text)) if path.read_text() != text]
        if stale:
            sys.exit(f"Out of date: {', '.join(stale)}; run python tools/build_leaderboard.py")
        print(f"Leaderboard up to date ({len(board['models'])} rows)")
        return
    (RESULTS / "leaderboard.json").write_text(board_text)
    TABLE.write_text(docs_text)
    print(f"Published {len(board['models'])} leaderboard rows")


if __name__ == "__main__":
    main()
