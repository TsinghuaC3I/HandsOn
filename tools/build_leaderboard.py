"""Publish a path-free 100-call Fable snapshot and a unified score table."""

import argparse
import collections
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "results"

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fable-source", type=Path, required=True)
    args = parser.parse_args()
    source_bytes = args.fable_source.read_bytes()
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

    entries = []
    for name, core, transfer in PAPER_MODELS:
        entries.append({
            "model": name, "core_percent": core, "transfer_percent": transfer,
            "overall_percent": round((50 * core + 16 * transfer) / 66, 1),
            "overall_derived_from_rounded_suite_rates": True,
            "protocol": "Paper trained checkpoint: 100 planned Core trials/task; 50 planned Transfer trials/task.",
            "source": "https://arxiv.org/abs/2610.26765",
        })
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
        })
    entries.sort(key=lambda row: (-row["overall_percent"], row["model"]))
    entries.append({
        "model": "DP (single-task)", "core_percent": 9.7,
        "transfer_percent": None, "overall_percent": None,
        "overall_derived_from_rounded_suite_rates": False,
        "protocol": "Core-only single-task reference; 38 tasks use an earlier goal sampler. Not ranked across 66 tasks.",
        "source": "https://arxiv.org/abs/2610.26765",
    })
    leaderboard = {
        "schema": "handson_unified_leaderboard_v1",
        "updated_utc": snapshot,
        "metric": "Native task success; Core-50 and Transfer-16 are task-macro rates. Overall is the 66-task macro mean, weighted by 50 and 16 tasks.",
        "caveat": "Models differ in observations, action interface, interaction budgets and trial counts. Trained-model overall rates are derived from manuscript suite rates rounded to one decimal place.",
        "fable_snapshot_complete_original_audit": fable["complete_original_audit"],
        "fable_audited_original_trials": fable["audited_original_trials"],
        "models": entries,
    }
    save(RESULTS / "leaderboard.json", leaderboard)
    print(f"Published {len(public_trials)} Fable trial rows and {len(entries)} leaderboard rows; audited {source['audited']}/660")


if __name__ == "__main__":
    main()
