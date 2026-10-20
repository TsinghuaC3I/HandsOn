"""Summarize measured GEN6 full-scene geometry reports without inferring reach."""
import hashlib
import json
from pathlib import Path


def main():
    evidence = Path(__file__).resolve().parent / "evidence"
    runs = ["live_probe_010"] + [f"geometry_batch_002_layout_{i:02}" for i in range(1, 9)]
    runs += [f"geometry_batch_003_layout_{i:02}" for i in (9, 10)]
    rows = []
    for run in runs:
        source = evidence / run / "report.json"
        report = json.loads(source.read_text())
        pre = report["pre_first_reset_cases"]
        settled = [case["shapes"] for case in report["angle_settles"]]
        reset = [case["shapes"] for case in report["seed_resets"]]
        checks = pre + settled + reset + [report["post_idle_shapes"]]
        assert report["probe_completed"] and report["geometry_queries_passed"]
        assert len(pre) == len(settled) == 7
        assert all(not c["query_errors"] and not c["unapproved_pairs"] for c in checks)
        assert all(c["sanity_broad_query_shape_count"] >= 250 for c in checks)
        assert all(c["timeline_time_before_after"] == [0.0, 0.0] for c in pre)
        rows.append({
            "run": run, "report_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "layout_dx_dy_yaw_deg": report["engineering_layout_dx_dy_yaw_deg"],
            "pre_first_step_angle_pairs": len(pre), "physically_settled_angle_pairs": len(settled),
            "unapproved_pairs": 0, "query_errors": 0,
            "minimum_loaded_shape_count": min(c["sanity_broad_query_shape_count"] for c in checks),
        })
    seed_source = evidence / "live_probe_008/report.json"
    seeds = json.loads(seed_source.read_text())["seed_resets"]
    assert len(seeds) >= 10 and len({c["seed"] for c in seeds}) >= 10
    assert all(not c["shapes"]["unapproved_pairs"] and not c["shapes"]["query_errors"] for c in seeds)
    result = {
        "scope": "Full final collider queries at authored first-PhysX and settled reset poses; no manipulation or human claim.",
        "layouts": rows, "unique_nominal_reset_seeds": len({c["seed"] for c in seeds}),
        "seed_report_sha256": hashlib.sha256(seed_source.read_bytes()).hexdigest(),
        "pre_first_step_angle_cases": sum(r["pre_first_step_angle_pairs"] for r in rows),
        "settled_angle_cases": sum(r["physically_settled_angle_pairs"] for r in rows),
        "robot_coverage": "35 enabled colliders per arm/hand, all proximal and distal fingers, palms, arms, opposite hand, task and table.",
        "depth_note": "Named source robot joint interfaces and same-body compounds are explicitly justified in each report. Panel/table depth uses final collider vertices and the support plane. The PhysX overlap API does not return other pair depths; those are null, never fabricated.",
        "passed": True,
    }
    (evidence / "live_geometry_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "layouts"}, indent=2))


if __name__ == "__main__":
    main()
