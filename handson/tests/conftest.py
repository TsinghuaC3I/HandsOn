"""Integrity checks for tests loading the bundled paper robot."""
import pytest


@pytest.fixture(scope="session")
def rokae_assets():
    from handson.robots.assets import validate_robot_assets
    validate_robot_assets("rokae_ar5")
