#!/usr/bin/env python3
"""Convert the user-selected BC021 wooden-block glTF with Isaac Sim 5.1.

The source closure is kept beside the generated USD.  This converter is
intentionally small and deterministic: it preserves separate source meshes
and PreviewSurface textures so the BC021 scene can select two rigid blocks and
derive convex collision from those same visible meshes.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
import sys
import traceback


def _clear_empty_texture_references(output: Path) -> None:
    """Remove converter graph inputs whose downloaded files are empty.

    The selected archive legitimately contains zero-byte optional metallic,
    normal, and specular maps.  Keeping those asset paths makes RTX emit
    texture-read errors and can turn the otherwise valid wood base-color
    material into a gray fallback.  The original files remain in the source
    closure; only the derived USD graph drops unusable optional inputs.
    """

    from pxr import Sdf, Usd, UsdShade

    stage = Usd.Stage.Open(str(output), Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"could not reopen converted USD: {output}")
    cleared: list[str] = []
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Shader":
            continue
        shader = UsdShade.Shader(prim)
        for shader_input in shader.GetInputs():
            if shader_input.GetTypeName() != "asset":
                continue
            asset_path = shader_input.Get()
            if not asset_path:
                continue
            raw_path = str(asset_path.path or asset_path)
            resolved = Path(raw_path)
            if not resolved.is_absolute():
                resolved = (output.parent / resolved).resolve()
            if resolved.is_file() and resolved.stat().st_size > 0:
                continue
            shader_input.Set(Sdf.AssetPath(""))
            cleared.append(f"{prim.GetPath()}.{shader_input.GetBaseName()}")
    if cleared:
        stage.GetRootLayer().Save()
    print(
        f"cleared {len(cleared)} empty optional texture references from {output}",
        flush=True,
    )


def _convert(source: Path, output: Path) -> None:
    import omni.kit.asset_converter

    context = omni.kit.asset_converter.AssetConverterContext()
    settings = {
        "ignore_animations": True,
        "ignore_camera": True,
        "ignore_light": True,
        "keep_all_materials": True,
        "export_preview_surface": True,
        "use_meter_as_world_unit": True,
        "create_world_as_default_root_prim": True,
        "convert_stage_up_z": True,
        "embed_textures": True,
        "single_mesh": False,
        "merge_all_meshes": False,
        "disabling_instancing": True,
    }
    for name, value in settings.items():
        if not hasattr(context, name):
            raise RuntimeError(f"AssetConverterContext lacks {name!r}")
        setattr(context, name, value)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.{os.getpid()}.tmp.usd")
    temporary.unlink(missing_ok=True)
    task = omni.kit.asset_converter.get_instance().create_converter_task(
        str(source), str(temporary), None, context
    )
    success = asyncio.get_event_loop().run_until_complete(task.wait_until_finished())
    if not success or not temporary.is_file() or temporary.stat().st_size <= 1024:
        detail = getattr(task, "get_error_message", lambda: "unknown converter error")()
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"BC021 wooden-block glTF conversion failed: {detail}")
    temporary.replace(output)
    _clear_empty_texture_references(output)
    print(f"converted {source} -> {output} bytes={output.stat().st_size}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "enable_cameras": False})
    try:
        app.update()
        from isaacsim.core.utils.extensions import enable_extension

        enable_extension("omni.kit.asset_converter")
        app.update()
        _convert(source, output)
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        app.close(wait_for_replicator=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
