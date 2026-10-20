#!/usr/bin/env python3
"""Classify and validate HandsOn policy datasets/checkpoint assets.

Unmarked data is never assumed to use the canonical hand order.  This script
is intended as a fail-closed preflight for training and evaluation launchers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_SOURCE_ORDER = "handson.physx.articulation_joint_names.v1"
EXPECTED_TARGET_ORDER = "handson.wuji.hand20.control.v1"
KNOWN_POLICY_SCHEMAS = {
    "handson.policy.actual_eef52.v1": {
        "dim": 52,
        "side_order": ["right", "left"],
    },
    "handson.policy.actual_joint54.v1": {
        "dim": 54,
        "side_order": ["left", "right"],
    },
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def find_schema(path: Path) -> tuple[Path, dict[str, Any]]:
    candidates = (
        path / "meta/handson_policy_schema.json",
        path / "handson_policy_schema.json",
        path / "assets/local_repo/handson_policy_schema.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate, load_json(candidate)
    raise ValueError(
        f"UNMARKED: {path} has no handson_policy_schema.json; refusing to "
        "guess whether its hand20 values were converted"
    )


def validate_schema(schema: dict[str, Any], *, source: Path) -> None:
    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(f"INVALID {source}: {message}")

    require(schema.get("format") == "handson.policy_schema", "wrong format")
    require(schema.get("version") == 1, "unsupported marker version")
    schema_id = schema.get("schema_id")
    require(schema_id in KNOWN_POLICY_SCHEMAS, "unknown schema_id")
    expected = KNOWN_POLICY_SCHEMAS[schema_id]
    require(schema.get("state_dim") == expected["dim"], "wrong state dimension")
    require(schema.get("action_dim") == expected["dim"], "wrong action dimension")
    require(
        schema.get("side_order") == expected["side_order"],
        "side order differs from the schema contract",
    )
    order = schema.get("hand_joint_order")
    require(isinstance(order, dict), "hand_joint_order is missing")
    require(order.get("conversion_applied") is True, "conversion is not marked")
    require(
        order.get("source_order_schema") == EXPECTED_SOURCE_ORDER,
        "source joint order differs",
    )
    require(
        order.get("target_order_schema") == EXPECTED_TARGET_ORDER,
        "target hand order differs",
    )
    indices = order.get("source_indices_in_target_order")
    require(isinstance(indices, dict), "source indices are missing")
    for side in ("right", "left"):
        values = indices.get(side)
        require(
            isinstance(values, list)
            and len(values) == 20
            and len(set(values)) == 20,
            f"{side} permutation is not a unique hand20 mapping",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()

    schemas: list[tuple[Path, dict[str, Any]]] = []
    failures: list[str] = []
    for path in args.paths:
        try:
            source, schema = find_schema(path)
            validate_schema(schema, source=source)
            schemas.append((source, schema))
            print(f"CANONICAL {path} marker={source}")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            failures.append(str(exc))
            print(str(exc))

    if schemas:
        reference = schemas[0][1]
        for source, schema in schemas[1:]:
            if schema != reference:
                failures.append(
                    f"SCHEMA_MISMATCH: {source} differs from {schemas[0][0]}"
                )
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
