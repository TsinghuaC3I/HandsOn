#!/usr/bin/env python3
"""Compatibility entry point for deterministic Isaac physical replay.

The former implementation only densified Cartesian commands and still used
online, frame-local DLS.  That behavior was not global IK and selected
different joint branches during replay.  Keep this filename for existing
automation, but route it to the MuJoCo-style Isaac control contract.
"""

from __future__ import annotations

import sys

from isaac_mujoco_style_replay import main


if __name__ == "__main__":
    sys.exit(main())
