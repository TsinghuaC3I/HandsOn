"""Locate and verify the rendering assets shipped with HandsOn (no Kit import)."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ASSET_ROOT = Path(__file__).resolve().parents[2] / "assets" / "rendering"
MANIFEST_SHA256 = "5f70aa593acf49b72cad870e9db36b5b4574666aba3bc7801f2c36bb9846e227"
_verified: dict[tuple[Path, bool], tuple[tuple[str, int, int], ...]] = {}


def validate_assets(root: Path | None = None, *, packaged_only: bool = False) -> Path:
    root = Path(root or os.environ.get("HANDSON_RENDER_ASSETS", "") or ASSET_ROOT).resolve()
    manifest = root / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(
            f"HandsOn rendering assets are missing: {manifest}. "
            "Install the complete release, including handson/assets/rendering."
        )
    payload = manifest.read_bytes()
    if hashlib.sha256(payload).hexdigest() != MANIFEST_SHA256:
        raise ValueError("HandsOn rendering manifest checksum mismatch")
    files = json.loads(payload)["files"]
    stamps = []
    for relative in files:
        path = (root / relative).resolve()
        if root not in path.parents:
            raise ValueError(f"Rendering asset escapes its bundle: {relative}")
        if not path.is_file():
            raise FileNotFoundError(f"Required rendering asset is missing: {path}")
        stat = path.stat()
        stamps.append((relative, stat.st_size, stat.st_mtime_ns))
    stamps = tuple(stamps)
    if _verified.get((root, packaged_only)) != stamps:
        for relative, expected in files.items():
            if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Rendering asset checksum mismatch: {relative}")
        _verified[(root, packaged_only)] = stamps
    return root


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packaged-only", action="store_true",
                        help="compatibility alias; all rendering assets are bundled and always verified")
    args = parser.parse_args()
    try:
        root = validate_assets(packaged_only=args.packaged_only)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({"rendering": "handson", "assets": str(root), "verified": True,
                      "scope": "packaged-only" if args.packaged_only else "complete-render-bundle"}))
