"""HandsOn proposal: navy hand/On, teal object/Hands.

SVG lettering is outlined; exported artwork requires no installed fonts.
Approved brand identity: teal Hands and object, navy On and complete hand.
"""
import argparse
from pathlib import Path
from html import escape
import json

from fontTools.ttLib import TTFont
from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen

parser = argparse.ArgumentParser(description='Generate the approved HandsOn vector logo family.')
parser.add_argument('--font', default='/System/Library/Fonts/Avenir Next.ttc', help='Avenir Next Bold font file or collection')
parser.add_argument('--font-index', type=int, default=0)
parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent)
args = parser.parse_args()
ROOT = args.out
ROOT.mkdir(parents=True, exist_ok=True)
FONT = TTFont(args.font, fontNumber=args.font_index)
INK, TEAL, PAPER, LIGHT = '#183449', '#4198AC', '#FFFFFF', '#7BC0CD'

# Two contours describe an opposed index finger and thumb around a small object.
# The lower opening is the wrist. Both contacts retain a small optical gap so
# the object remains distinct at favicon sizes and in one-colour reproduction.
PALM = 'M29 83V75C19 68 12 58 12 47V41C12 35 19 32 24 36L43 53C48 57 54 59 61 59H78C84 59 88 64 85 69L74 81C69 86 63 89 55 89H43'
INDEX = 'M27 38V30C27 19 36 11 47 11H65C75 11 82 18 82 28'


def mark(ink=INK, accent=TEAL):
    return ('<g transform="rotate(-45 48 50)">'
            f'<g fill="none" stroke-width="8" '
            'stroke-linecap="round" stroke-linejoin="round">'
            f'<path d="{INDEX}" stroke="{ink}"/>'
            f'<path d="{PALM}" stroke="{ink}"/></g>'
            f'<rect x="61" y="33" width="21" height="21" rx="3" fill="{accent}"/></g>')


def wordmark(ink=INK, accent=TEAL):
    size = 200
    scale = size / FONT['head'].unitsPerEm
    cap = FONT['OS/2'].sCapHeight * scale
    gaps = [.068, .063, .066, .073, .061, .067]
    glyphs = FONT.getGlyphSet()
    cmap = FONT.getBestCmap()
    right = 0
    pieces, y_bounds = [], []
    for i, ch in enumerate('HandsOn'):
        glyph = glyphs[cmap[ord(ch)]]
        bp = BoundsPen(glyphs)
        glyph.draw(bp)
        x0, y0, x1, y1 = [v * scale for v in bp.bounds]
        x = (right + gaps[i-1] * cap if i else 0) - x0
        pen = SVGPathPen(glyphs, ntos=lambda v: f'{v:.3f}')
        glyph.draw(TransformPen(pen, (scale, 0, 0, -scale, x, 0)))
        colour = accent if i < 5 else ink
        pieces.append(f'<path d="{pen.getCommands()}" fill="{colour}"/>')
        right = x + x1
        y_bounds.extend([-y1, -y0])
    ymin, ymax = min(y_bounds), max(y_bounds)
    height = ymax-ymin
    outlines = ''.join(pieces)
    return (f'<g transform="translate(0 {-ymin:.3f})">{outlines}</g>', right, height)


def svg(body, w, h, label='HandsOn'):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:.3f}" '
            f'height="{h:.3f}" viewBox="0 0 {w:.3f} {h:.3f}" role="img" '
            f'aria-label="{escape(label)}"><title>{escape(label)}</title>{body}</svg>\n')


def lockup(ink=INK, accent=TEAL):
    letters, w, h = wordmark(ink, accent)
    # Icon ink is 1.13 times the letter ink height; a compact gap keeps the
    # combination readable as a single identity rather than a logo plus title.
    s = h * 1.13 / 86
    pad, gap = 18, h * .24
    icon_w, icon_h = 96*s, 100*s
    total_h = icon_h + 2*pad
    x = pad + icon_w + gap
    body = (f'<g transform="translate({pad} {pad}) scale({s})">{mark(ink, accent)}</g>'
            f'<g transform="translate({x} {(total_h-h)/2})">{letters}</g>')
    return svg(body, x+w+pad, total_h)


def write_assets():
    for name, ink, accent in [('light', INK, TEAL), ('dark', '#E9EFF3', LIGHT)]:
        suffix = '' if name == 'light' else '-'+name
        (ROOT / f'handson-lockup{suffix}.svg').write_text(lockup(ink, accent))
        body, w, h = wordmark(ink, accent)
        (ROOT / f'handson-wordmark{suffix}.svg').write_text(
            svg(f'<g transform="translate(12 12)">{body}</g>', w+24, h+24))
        (ROOT / f'handson-mark{suffix}.svg').write_text(svg(mark(ink, accent), 96, 100))
        icon_scale = w * .31 / 96
        top = 20
        text_y = top + 100 * icon_scale + h * .35
        stacked = (f'<g transform="translate({(w+40-96*icon_scale)/2} {top}) scale({icon_scale})">{mark(ink,accent)}</g>'
                   f'<g transform="translate(20 {text_y})">{body}</g>')
        (ROOT / f'handson-stacked{suffix}.svg').write_text(svg(stacked,w+40,text_y+h+20))
    icon = (f'<rect width="128" height="128" rx="28" fill="{INK}"/>'
            f'<g transform="translate(16 14)">{mark("#E9EFF3", LIGHT)}</g>')
    (ROOT / 'handson-icon.svg').write_text(svg(icon, 128, 128))


if __name__ == '__main__':
    write_assets()
    print(f'Generated approved HandsOn SVGs in {ROOT}')
