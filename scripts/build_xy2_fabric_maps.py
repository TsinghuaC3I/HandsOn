#!/usr/bin/env python3
"""Deterministic periodic weave data maps for XY2's photo-referenced black pad.

These are analytic material data, not pixels extracted from the user's photo.
Run with NumPy and Pillow; outputs replace the same existing map paths.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from PIL import Image


def build(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    size, cells, repeats = 1024, 16, 16
    u, v = np.meshgrid((np.arange(size) + .5) / size, (np.arange(size) + .5) / size)
    x, y = u * cells, v * cells
    ix, iy = np.floor(x), np.floor(y)
    fx, fy = x - ix - .5, y - iy - .5
    warp = np.sqrt(np.maximum(0, 1 - (fx / .48) ** 2))
    weft = np.sqrt(np.maximum(0, 1 - (fy / .48) ** 2))
    # Rounded yarns alternate above and below at each crossing.
    h_warp = warp * (.66 + .22 * np.cos(np.pi * (y - .5 + ix)))
    h_weft = weft * (.66 - .22 * np.cos(np.pi * (x - .5 + iy)))
    height = np.maximum(h_warp, h_weft)
    strand = np.sin(2 * np.pi * (x * 5 + y * .5)) * np.sin(2 * np.pi * (y * 5 + x * .5))
    height += .025 * strand * height
    # 1.17 mm yarn spacing on a 0.30 m pad; 90 um relief.
    tile_m = .30 / repeats
    dx = (np.roll(height, -1, 1) - np.roll(height, 1, 1)) * 90e-6 / (2 * tile_m / size)
    dy = (np.roll(height, -1, 0) - np.roll(height, 1, 0)) * 90e-6 / (2 * tile_m / size)
    normal = np.stack([-dx, -dy, np.ones_like(dx)], axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    black = np.clip(24 + height * 20 + strand * 1.2, 0, 255)
    albedo = np.stack([black * .97, black, black * 1.025], axis=-1)
    roughness = np.clip(.83 - .15 * height + .015 * strand, 0, 1)
    maps = {'black_weave_basecolor.png': albedo,
            'black_weave_normal.png': (normal * .5 + .5) * 255,
            'black_weave_roughness.png': np.repeat(roughness[..., None], 3, axis=-1) * 255}
    for name, pixels in maps.items():
        Image.fromarray(np.rint(pixels).astype(np.uint8), 'RGB').save(output / name)
    manifest = {'profile': 'xy2_black_woven_fabric_photo1018', 'source': 'analytic periodic weave informed by IMG_1018 appearance; no photo pixels copied',
                'size_px': size, 'cells_per_tile': cells, 'repeat_across_pad': repeats,
                'nominal_yarn_spacing_m': .30 / (cells * repeats), 'height_relief_m': 90e-6,
                'color_space': {'basecolor': 'sRGB', 'normal': 'raw OpenGL tangent space', 'roughness': 'raw'},
                'sha256': {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in maps}}
    (output / 'fabric_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    build(Path(__file__).resolve().parents[1] / 'assets/xy2_appearance')
