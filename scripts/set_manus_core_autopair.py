#!/usr/bin/env python3
"""Enable MANUS Core auto-pairing in the local CoreLite settings file."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


DEFAULT_SETTINGS = Path.home() / ".config" / "Manus" / "Core 3" / "CoreLite.Settings.3.0.0.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Set deviceSettings.autoPair in MANUS CoreLite settings."
    )
    parser.add_argument(
        "--settings",
        default=os.environ.get("MANUS_CORE_SETTINGS", str(DEFAULT_SETTINGS)),
        help="Path to CoreLite.Settings.3.0.0.json.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print when a change is written.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings_path = Path(args.settings).expanduser()
    if not settings_path.exists():
        if not args.quiet:
            print(f"[WARN] MANUS Core settings not found: {settings_path}")
        return 0

    with settings_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    device_settings = data.setdefault("deviceSettings", {})
    if device_settings.get("autoPair") is True:
        if not args.quiet:
            print(f"MANUS Core autoPair already enabled: {settings_path}")
        return 0

    device_settings["autoPair"] = True
    with settings_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=3)
        handle.write("\n")

    print(f"Enabled MANUS Core autoPair: {settings_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
