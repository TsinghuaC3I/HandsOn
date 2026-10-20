#!/usr/bin/env python3
"""Convert the downloaded BlenderKit faucet into KA021 visual components.

The BlenderKit download is a source file, not a physics asset.  This helper
exports two small USD components from the authored object boundary:

* ``faucet_static.usd`` contains the fixed body, arch, nozzle and handle base.
* ``faucet_handle.usd`` contains only ``Handle`` and its authored ``Empty``
  pivot parent.

The conversion deliberately disables Blender modifiers while exporting.  The
downloaded asset has a high-resolution subdivision stack which is useful for
offline rendering but needlessly inflates a simulator visual layer.  PhysX
colliders and the revolute joint are authored by the KA021 scene code, so no
physics API is copied from the source file.

Run this file with Blender, for example::

    blender --background --disable-autoexec --python \
      scripts/prepare_ka021_blenderkit_faucet_assets.py -- \
      --source /path/to/kitchen_faucet.blend --output-dir /path/to/bundle

When run with the regular Python interpreter the script prints the command
needed to perform the conversion.  Keeping the conversion recipe in the
repository makes the generated runtime USD provenance reproducible without
shipping the BlenderKit source file in the asset bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Iterable


STATIC_OBJECT_NAMES = (
    "FaucetBase",
    "FaucetMain",
    "FaucetHead",
    "Buttons",
    "BrassTip",
    "SprayNozzle",
    "HandleBase",
)
HANDLE_OBJECT_NAME = "Handle"
PIVOT_NATIVE_M = (0.0, 0.09024405479431152, 0.06231890618801117)
SOURCE_URL = (
    "https://www.blendkit.com/asset-gallery-detail/"
    "b372d45e-2fbe-41cf-bf10-d0fd70dd2574/"
)
SOURCE_UUID = "0390e86f-2ffd-4745-b25c-d48397c67a51"
SOURCE_VERSION_ID = "876c68b4-1341-46ad-8491-3b9762dccd87"
SOURCE_CREATOR = "Cody Flynn"
SOURCE_TITLE = "Kitchen Faucet"
SOURCE_LICENSE = "BlenderKit Royalty Free"
SOURCE_SHA256 = "17faafad74eb833f5cecee9c84b13964bff92492fb1fbe3d795cea6e189183bf"
MODEL_YAW_RAD = math.radians(139.20831019081422)
JOINT_FRAME_YAW_RAD = MODEL_YAW_RAD + math.pi
JOINT_FRAME_ORIENTATION_WXYZ = (
    math.cos(0.5 * JOINT_FRAME_YAW_RAD),
    0.0,
    0.0,
    math.sin(0.5 * JOINT_FRAME_YAW_RAD),
)
JOINT_AXIS_WORLD_W = (
    -math.sin(JOINT_FRAME_YAW_RAD),
    math.cos(JOINT_FRAME_YAW_RAD),
    0.0,
)
INITIAL_OPEN_DIRECTION_WORLD_W = (
    math.cos(MODEL_YAW_RAD),
    math.sin(MODEL_YAW_RAD),
    0.0,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(list(argv) if argv is not None else None)


def _export_with_blender(source: Path, output_dir: Path) -> dict[str, object]:
    """Export the selected object sets using Blender's USD exporter."""

    try:
        import bpy  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised by CLI guidance
        raise RuntimeError(
            "This conversion step must run inside Blender; invoke Blender "
            "with '--python prepare_ka021_blenderkit_faucet_assets.py --'."
        ) from exc

    source = source.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.open_mainfile(filepath=str(source), load_ui=False)

    objects = {obj.name: obj for obj in bpy.data.objects}
    missing = [name for name in (*STATIC_OBJECT_NAMES, HANDLE_OBJECT_NAME) if name not in objects]
    if missing:
        raise RuntimeError(
            "BlenderKit faucet object contract is incomplete: "
            + ", ".join(missing)
        )

    # Keep the source geometry deterministic and avoid exporting the very
    # large subdivision evaluation.  The source .blend remains untouched.
    modifier_states: list[tuple[object, object, bool, bool]] = []
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        for modifier in obj.modifiers:
            modifier_states.append(
                (obj, modifier, bool(modifier.show_viewport), bool(modifier.show_render))
            )
            modifier.show_viewport = False
            modifier.show_render = False

    def select(names: Iterable[str]) -> None:
        bpy.ops.object.select_all(action="DESELECT")
        for name in names:
            objects[name].select_set(True)
        bpy.context.view_layer.objects.active = objects[next(iter(names))]

    def export(path: Path, names: Iterable[str]) -> None:
        select(names)
        # These options are available in Blender 3.6 (the version used for the
        # verified conversion).  Exporting materials keeps the metal/nozzle
        # distinction while avoiding external texture dependencies.
        bpy.ops.wm.usd_export(
            filepath=str(path),
            selected_objects_only=True,
            visible_objects_only=True,
            export_animation=False,
            export_materials=True,
            export_textures=False,
            root_prim_path="/World",
        )

    static_path = output_dir / "faucet_static.usd"
    handle_path = output_dir / "faucet_handle.usd"
    export(static_path, STATIC_OBJECT_NAMES)
    export(handle_path, (HANDLE_OBJECT_NAME,))

    for obj, modifier, viewport, render in modifier_states:
        modifier.show_viewport = viewport  # type: ignore[attr-defined]
        modifier.show_render = render  # type: ignore[attr-defined]

    return {
        "static_path": static_path,
        "handle_path": handle_path,
        "static_object_names": list(STATIC_OBJECT_NAMES),
        "handle_object_names": [HANDLE_OBJECT_NAME],
    }


def _write_manifest(source: Path, output_dir: Path, exported: dict[str, object]) -> Path:
    static_path = Path(exported["static_path"])
    handle_path = Path(exported["handle_path"])
    manifest = {
        "schema": "handson.ka021.blenderkit-kitchen-faucet-asset",
        "schema_version": 1,
        "task_id": "B320-KA021",
        "task_name": "t05_fill_cup",
        "asset_bundle": "blenderkit_kitchen_faucet",
        "provenance": {
            "platform": "BlenderKit",
            "title": SOURCE_TITLE,
            "creator": SOURCE_CREATOR,
            "asset_uuid": SOURCE_UUID,
            "version_id": SOURCE_VERSION_ID,
            "source_url": SOURCE_URL,
            "license": SOURCE_LICENSE,
            "license_url": "https://www.blendkit.com/docs/licenses/",
            "source_sha256": SOURCE_SHA256,
            "source_filename": source.name,
            "source_file_present_in_repo": False,
        },
        "conversion": {
            "tool": "Blender 3.6 USD exporter",
            "modifiers_enabled": False,
            "materials_exported": True,
            "textures_exported": False,
            "split_method": "authored_object_boundary",
            "static_objects": list(STATIC_OBJECT_NAMES),
            "moving_objects": [HANDLE_OBJECT_NAME],
            "moving_parent_pivot": "Empty",
            "pivot_native_m": list(PIVOT_NATIVE_M),
            "physics_authored_by": "handson/handson/sim/isaaclab/b320_bimanual_scene.py",
        },
        "runtime_components": [
            {
                "path": static_path.name,
                "sha256": _sha256(static_path),
                "size_bytes": static_path.stat().st_size,
                "mesh_count": 7,
                "objects": list(STATIC_OBJECT_NAMES),
            },
            {
                "path": handle_path.name,
                "sha256": _sha256(handle_path),
                "size_bytes": handle_path.stat().st_size,
                "mesh_count": 1,
                "objects": [HANDLE_OBJECT_NAME],
            },
        ],
        "coordinate_contract": {
            "source_up_axis": "Z",
            "runtime_up_axis": "Z",
            "pivot_native_m": list(PIVOT_NATIVE_M),
            "static_root_prim": "/World",
            "handle_root_prim": "/World",
        },
        "physics": {
            "visual_components_have_physics_api": False,
            "joint_type": "revolute",
            "joint_name": "RevoluteJoint_faucet_3_right",
            "joint_axis_runtime": "Y",
            "joint_frame": "yawed_source_y_axis_with_sign_flip",
            "joint_frame_orientation_wxyz": list(JOINT_FRAME_ORIENTATION_WXYZ),
            "joint_axis_world_w": list(JOINT_AXIS_WORLD_W),
            "joint_limits_deg": [-45.0, 0.0],
            "open_direction_world": [-1.0, 0.0, 0.0],
            "initial_open_direction_world_w": list(INITIAL_OPEN_DIRECTION_WORLD_W),
        },
    }
    manifest_path = output_dir / "asset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    attribution = output_dir / "ATTRIBUTION.txt"
    attribution.write_text(
        f'"{SOURCE_TITLE}" by {SOURCE_CREATOR} is used under the {SOURCE_LICENSE} '
        f'license.\nSource: {SOURCE_URL}\nLicense: https://www.blendkit.com/docs/licenses/\n\n'
        "The source file is converted into task-specific visual derivatives; the "
        "revolute joint, collision geometry, and task logic are authored by HandsOn.\n",
        encoding="utf-8",
    )
    return manifest_path


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    source = args.source.resolve()
    output_dir = args.output_dir.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if _sha256(source) != SOURCE_SHA256:
        raise ValueError(
            "Downloaded BlenderKit source hash does not match the reviewed "
            f"asset: expected {SOURCE_SHA256}, got {_sha256(source)}"
        )
    exported = _export_with_blender(source, output_dir)
    manifest = _write_manifest(source, output_dir, exported)
    print(f"Wrote {manifest}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    # Blender puts its own arguments before ``--``.  Strip them when this
    # module is executed by Blender while leaving regular Python invocation
    # unchanged.
    if "--" in sys.argv:
        cli_args = sys.argv[sys.argv.index("--") + 1 :]
    else:
        cli_args = sys.argv[1:]
    raise SystemExit(main(cli_args))
