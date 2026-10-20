from __future__ import annotations

from pathlib import Path

from .asset_contract import ASSETS, USD_COLLISION_MESH_COUNT, audit_assets


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def test_delivered_asset_bundle_is_byte_identical_and_complete():
    report = audit_assets(REPOSITORY_ROOT)
    assert set(report["files"]) == set(ASSETS)
    assert report["modification"] == "none; byte-identical copy"
    assert report["usd_contract"]["collision_mesh_count"] == USD_COLLISION_MESH_COUNT
