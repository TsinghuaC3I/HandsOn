"""Generator import shim for the installed HS005 scene-collision contract."""

from __future__ import annotations

from pathlib import Path
import sys


repository_root = Path(__file__).resolve().parents[3]
package_root = str(repository_root / "handson")
if package_root not in sys.path:
    sys.path.insert(0, package_root)

from handson.sim.isaaclab.hs005_scene_collision import *  # noqa: E402,F403
