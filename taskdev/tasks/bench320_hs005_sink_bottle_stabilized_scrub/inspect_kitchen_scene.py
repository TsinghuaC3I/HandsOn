#!/usr/bin/env python3
"""Render orthographic-style inspection views of one converted kitchen asset."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaacsim import SimulationApp


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--asset", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--scale", type=float, required=True)
    result.add_argument("--translation", type=float, nargs=3, required=True)
    result.add_argument("--eye", type=float, nargs=3)
    result.add_argument("--target", type=float, nargs=3)
    result.add_argument("--panorama-eye", type=float, nargs=3)
    result.add_argument("--show-only", type=str)
    result.add_argument(
        "--max-mesh-span",
        type=float,
        help="Hide meshes larger than this world-space span and fit the remainder.",
    )
    result.add_argument(
        "--list-meshes",
        action="store_true",
        help="Print memory-safe per-mesh world bounds and exit without rendering.",
    )
    return result


def main() -> int:
    args = parser().parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    app = SimulationApp(
        {
            "headless": True,
            "enable_cameras": True,
            "renderer": "RayTracedLighting",
            "width": 640,
            "height": 480,
        }
    )

    import omni.replicator.core as rep
    import omni.usd
    import numpy as np
    from PIL import Image
    from pxr import Gf, Usd, UsdGeom, UsdLux

    context = omni.usd.get_context()
    context.new_stage()
    stage = context.get_stage()
    root = UsdGeom.Xform.Define(stage, "/World/Asset")
    root.GetPrim().GetReferences().AddReference(str(args.asset.resolve()))
    scale = float(args.scale)
    tx, ty, tz = (float(value) for value in args.translation)
    transform = Gf.Matrix4d(
        -scale, 0.0, 0.0, 0.0,
        0.0, 0.0, scale, 0.0,
        0.0, -scale, 0.0, 0.0,
        tx, ty, tz, 1.0,
    )
    UsdGeom.Xformable(root.GetPrim()).AddTransformOp().Set(transform)
    show_tokens = tuple(
        token.strip() for token in (args.show_only or "").split(",") if token.strip()
    )
    if show_tokens:
        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Mesh):
                continue
            if not any(token in str(prim.GetPath()) for token in show_tokens):
                UsdGeom.Imageable(prim).MakeInvisible()

    dome = UsdLux.DomeLight.Define(stage, "/World/Dome")
    dome.CreateIntensityAttr(900.0)
    dome.CreateColorAttr(Gf.Vec3f(0.92, 0.95, 1.0))
    key = UsdLux.DistantLight.Define(stage, "/World/Key")
    key.CreateIntensityAttr(2200.0)
    key.CreateAngleAttr(0.8)
    key.CreateColorAttr(Gf.Vec3f(1.0, 0.94, 0.86))
    UsdGeom.Xformable(key.GetPrim()).AddRotateXYZOp().Set(
        Gf.Vec3f(-45.0, 25.0, 20.0)
    )

    for _ in range(8):
        app.update()
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=False,
    )
    if show_tokens:
        aligned = Gf.Range3d()
        matched = 0
        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Mesh) or not any(
                token in str(prim.GetPath()) for token in show_tokens
            ):
                continue
            aligned.UnionWith(cache.ComputeWorldBound(prim).ComputeAlignedBox())
            matched += 1
        if matched == 0:
            raise ValueError(f"--show-only matched no mesh: {args.show_only}")
    elif args.max_mesh_span is not None:
        aligned = Gf.Range3d()
        matched = 0
        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Mesh):
                continue
            mesh_range = cache.ComputeWorldBound(prim).ComputeAlignedBox()
            mesh_size = mesh_range.GetSize()
            if max(float(mesh_size[0]), float(mesh_size[1]), float(mesh_size[2])) > args.max_mesh_span:
                UsdGeom.Imageable(prim).MakeInvisible()
                continue
            if not mesh_range.IsEmpty():
                aligned.UnionWith(mesh_range)
                matched += 1
        if matched == 0:
            raise ValueError(f"--max-mesh-span retained no mesh: {args.max_mesh_span}")
    else:
        aligned = cache.ComputeWorldBound(root.GetPrim()).ComputeAlignedBox()
    minimum, maximum = aligned.GetMin(), aligned.GetMax()
    center = (minimum + maximum) * 0.5
    size = maximum - minimum
    if args.list_meshes:
        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Mesh):
                continue
            bounds = cache.ComputeWorldBound(prim).ComputeAlignedBox()
            print(
                f"mesh path={prim.GetPath()} min={tuple(bounds.GetMin())} "
                f"max={tuple(bounds.GetMax())}"
            )
        print(
            f"bounds min={tuple(minimum)} max={tuple(maximum)} "
            f"center={tuple(center)} size={tuple(size)}"
        )
        app.close()
        return 0
    span = max(float(size[0]), float(size[1]), float(size[2]), 1.0)
    height = float(center[2]) + max(0.25 * span, 1.0)
    target = Gf.Vec3d(float(center[0]), float(center[1]), float(center[2]))
    if args.panorama_eye is not None:
        eye = Gf.Vec3d(*(float(value) for value in args.panorama_eye))
        views = {
            "panorama_front": (eye, eye + Gf.Vec3d(0.0, 2.0, -0.25)),
            "panorama_back": (eye, eye + Gf.Vec3d(0.0, -2.0, -0.25)),
            "panorama_left": (eye, eye + Gf.Vec3d(-2.0, 0.0, -0.25)),
            "panorama_right": (eye, eye + Gf.Vec3d(2.0, 0.0, -0.25)),
        }
    elif args.eye is not None or args.target is not None:
        if args.eye is None or args.target is None:
            raise ValueError("--eye and --target must be provided together")
        target = Gf.Vec3d(*(float(value) for value in args.target))
        views = {
            "custom": (
                Gf.Vec3d(*(float(value) for value in args.eye)),
                target,
            ),
        }
    else:
        views = {
            "front": (Gf.Vec3d(float(center[0]), float(center[1]) - 1.15 * span, height), target),
            "back": (Gf.Vec3d(float(center[0]), float(center[1]) + 1.15 * span, height), target),
            "left": (Gf.Vec3d(float(center[0]) - 1.15 * span, float(center[1]), height), target),
            "right": (Gf.Vec3d(float(center[0]) + 1.15 * span, float(center[1]), height), target),
            "top": (Gf.Vec3d(float(center[0]), float(center[1]) - 0.01, float(maximum[2]) + 1.6 * span), target),
        }

    products = []
    annotators = []
    for name, (eye, view_target) in views.items():
        camera = UsdGeom.Camera.Define(stage, f"/World/Cameras/{name}")
        camera.CreateHorizontalApertureAttr(24.0)
        camera.CreateVerticalApertureAttr(18.0)
        camera.CreateFocalLengthAttr(22.0)
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
        view = Gf.Matrix4d(1.0)
        up = Gf.Vec3d(0.0, 1.0, 0.0) if name == "top" else Gf.Vec3d(0.0, 0.0, 1.0)
        view.SetLookAt(eye, view_target, up)
        UsdGeom.Xformable(camera.GetPrim()).AddTransformOp().Set(view.GetInverse())
        product = rep.create.render_product(str(camera.GetPath()), (640, 480))
        annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        annotator.attach([product])
        products.append(product)
        annotators.append((name, annotator))

    for _ in range(20):
        app.update()
    rep.orchestrator.step(rt_subframes=8)
    for name, annotator in annotators:
        frame = np.asarray(annotator.get_data())
        if frame.ndim != 3 or frame.shape[2] < 3:
            raise RuntimeError(f"bad frame for {name}: {frame.shape}")
        Image.fromarray(frame[:, :, :3].astype(np.uint8)).save(
            output_dir / f"{name}.png",
            compress_level=0,
        )

    print(
        f"bounds min={tuple(minimum)} max={tuple(maximum)} "
        f"center={tuple(center)} size={tuple(size)}"
    )
    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
