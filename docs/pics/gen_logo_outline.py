"""Extract real glyph outlines from a font so the wordmark needs no font at all,
and so per-letter-pair spacing can be set optically."""
from fontTools.ttLib import TTFont, TTCollection
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.misc.transform import Transform

def load(path, index=0):
    try:
        return TTCollection(path).fonts[index]
    except Exception:
        return TTFont(path, fontNumber=index)

def glyph_paths(font, text, size):
    upem = font["head"].unitsPerEm
    s = size / upem
    cmap = font.getBestCmap()
    gs = font.getGlyphSet()
    hmtx = font["hmtx"]

    out = []
    for ch in text:
        gname = cmap.get(ord(ch))
        if gname is None:
            raise KeyError(f"U+{ord(ch):04X} '{ch}' missing")
        pen = SVGPathPen(gs, ntos=lambda v: f"{v:.2f}")
        # y-flip: font units grow up, SVG grows down
        tp = TransformPen(pen, Transform(s, 0, 0, -s, 0, 0))
        gs[gname].draw(tp)
        d = pen.getCommands()
        adv = hmtx[gname][0] * s
        out.append({"ch": ch, "d": d, "adv": adv})
    return out

def ink_x_bounds(font, ch, size):
    """horizontal ink extent of one glyph, relative to its own origin"""
    upem = font["head"].unitsPerEm
    s = size / upem
    cmap = font.getBestCmap(); gs = font.getGlyphSet()
    gname = cmap[ord(ch)]
    from fontTools.pens.boundsPen import BoundsPen
    bp = BoundsPen(gs)
    gs[gname].draw(bp)
    if bp.bounds is None:
        return 0.0, 0.0
    x0, y0, x1, y1 = bp.bounds
    return x0 * s, x1 * s
