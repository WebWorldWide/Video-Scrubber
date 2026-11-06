# SPDX-License-Identifier: GPL-3.0-or-later
"""Builds AppIcon.icon, the Icon Composer document for Video Scrubber.

What the app does, in one picture: a timeline of key frames, with a playhead (the scrubber)
whose handle is a search lens. The lens has a play button in it, and the playhead stands on
the moment it found, which is lit up on the timeline.

Every layer is an SVG on the 1024-point icon canvas; Icon Composer adds the squircle,
Liquid Glass, shadows and specular highlights from icon.json.

    python3 design/icon/build_icon.py      # writes design/icon/AppIcon.icon
"""

import json
import shutil
from pathlib import Path

OUT = Path(__file__).resolve().parent / "AppIcon.icon"
W = 1024

# Timeline: a glass track holding a row of key-frame tiles.
BAR_X, BAR_Y, BAR_W, BAR_H, BAR_R = 104, 618, 816, 206, 58
TILES, TILE_GAP, TILE_PAD = 5, 14, 24
FOUND = 2  # the tile the search found (0-based), under the playhead
# A different little scene in each frame: (horizon height 0–1, hill peak x 0–1, sun x, sun y), so the
# tiles read as pictures rather than buttons.
SCENES = [(0.62, 0.3, 0.72, 0.3), (0.7, 0.75, 0.3, 0.32), (0.64, 0.4, 0.74, 0.3), (0.58, 0.5, 0.78, 0.28),
          (0.6, 0.65, 0.28, 0.3)]

# Playhead: lens on top, stem down through the timeline.
LY, LR, RING = 318, 222, 50
STEM_W = 40


def svg(body: str, defs: str = "") -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{W}" viewBox="0 0 {W} {W}">'
            f"<defs>{defs}</defs>{body}</svg>\n")


def rrect(x, y, w, h, r, **attrs) -> str:
    a = " ".join(f'{k.replace("_", "-")}="{v}"' for k, v in attrs.items())
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{r:.1f}" {a}/>'


def tiles() -> list[tuple[float, float, float, float]]:
    """(x, y, w, h) of each key-frame tile inside the track."""
    inner_w = BAR_W - 2 * TILE_PAD
    w = (inner_w - (TILES - 1) * TILE_GAP) / TILES
    h = BAR_H - 2 * TILE_PAD
    return [(BAR_X + TILE_PAD + i * (w + TILE_GAP), BAR_Y + TILE_PAD, w, h) for i in range(TILES)]


def found_x() -> float:
    x, _, w, _ = tiles()[FOUND]
    return x + w / 2


# ------------------------------------------------------------------ timeline


def track_glass() -> str:
    defs = ('<linearGradient id="g" x1="0" y1="0" x2="0" y2="1">'
            '<stop offset="0" stop-color="#FFFFFF" stop-opacity="0.5"/>'
            '<stop offset="1" stop-color="#DCE8FF" stop-opacity="0.28"/></linearGradient>')
    return svg(rrect(BAR_X, BAR_Y, BAR_W, BAR_H, BAR_R, fill="url(#g)"), defs)


def scene(x, y, w, h, i, hill, sun, alpha) -> str:
    """A hill and a sun inside a tile, clipped to its rounded corners."""
    hz, px, sx, sy = SCENES[i]
    r = 22
    clip = f'<clipPath id="c{i}">{rrect(x, y, w, h, r)}</clipPath>'
    d = (f"M{x - 2:.1f},{y + h * hz:.1f} Q{x + w * px:.1f},{y + h * (hz - 0.34):.1f} {x + w + 2:.1f},{y + h * (hz - 0.04):.1f} "
         f"L{x + w + 2:.1f},{y + h + 2:.1f} L{x - 2:.1f},{y + h + 2:.1f} Z")
    body = (f'<g clip-path="url(#c{i})"><path d="{d}" fill="{hill}" fill-opacity="{alpha}"/>'
            f'<circle cx="{x + w * sx:.1f}" cy="{y + h * sy:.1f}" r="{w * 0.11:.1f}" fill="{sun}" fill-opacity="{alpha}"/></g>')
    return clip, body


def frames() -> str:
    """Key frames: frosted tiles, each with its own little picture."""
    defs, out = [], []
    for i, (x, y, w, h) in enumerate(tiles()):
        if i == FOUND:
            continue
        out.append(rrect(x, y, w, h, 22, fill="#FFFFFF", fill_opacity="0.3"))
        c, b = scene(x, y, w, h, i, "#FFFFFF", "#FFFFFF", 0.45)
        defs.append(c)
        out.append(b)
    return svg("".join(out), "".join(defs))


def found() -> str:
    """The found moment: the frame under the playhead, lit up in amber with its picture in deeper amber."""
    x, y, w, h = tiles()[FOUND]
    defs = ('<linearGradient id="a" x1="0" y1="0" x2="0" y2="1">'
            '<stop offset="0" stop-color="#FFE07A"/><stop offset="1" stop-color="#FFB300"/></linearGradient>')
    c, b = scene(x, y, w, h, FOUND, "#E08A00", "#FFF6D6", 0.9)
    glow = rrect(x - 12, y - 12, w + 24, h + 24, 32, fill="#FFD54A", fill_opacity="0.4")
    return svg(glow + rrect(x, y, w, h, 22, fill="url(#a)") + b, defs + c)


# ------------------------------------------------------------------ playhead


def stem() -> str:
    """The playhead line, from the lens down through the timeline, with a rounded foot."""
    x = found_x()
    top = LY + LR - 6
    bottom = BAR_Y + BAR_H + 34
    return svg(rrect(x - STEM_W / 2, top, STEM_W, bottom - top, STEM_W / 2, fill="#FFFFFF"))


def ring() -> str:
    """The lens rim."""
    x = found_x()
    r = LR - RING / 2
    defs = ('<linearGradient id="r" x1="0" y1="0" x2="0" y2="1">'
            '<stop offset="0" stop-color="#FFFFFF"/><stop offset="1" stop-color="#E4ECFF"/></linearGradient>')
    return svg(f'<circle cx="{x:.1f}" cy="{LY}" r="{r:.1f}" fill="none" stroke="url(#r)" stroke-width="{RING}"/>', defs)


def lens() -> str:
    """The glass inside the rim."""
    return svg(f'<circle cx="{found_x():.1f}" cy="{LY}" r="{LR - RING + 2}" fill="#FFFFFF" fill-opacity="0.3"/>')


def play() -> str:
    """A rounded play triangle, optically centred in the lens."""
    cx, cy = found_x() + 14, LY
    s = 92  # half-height
    pts = [(cx - s * 0.62, cy - s), (cx - s * 0.62, cy + s), (cx + s * 0.95, cy)]
    (ax, ay), (bx, by), (tx, ty) = pts
    d = (f"M{ax:.1f},{ay + 22:.1f} L{bx:.1f},{by - 22:.1f} Q{bx:.1f},{by + 6:.1f} {bx + 24:.1f},{by - 8:.1f} "
         f"L{tx - 18:.1f},{ty + 12:.1f} Q{tx + 6:.1f},{ty:.1f} {tx - 18:.1f},{ty - 12:.1f} "
         f"L{ax + 24:.1f},{ay + 8:.1f} Q{ax:.1f},{ay - 6:.1f} {ax:.1f},{ay + 22:.1f} Z")
    defs = ('<linearGradient id="p" x1="0" y1="0" x2="0" y2="1">'
            '<stop offset="0" stop-color="#FFD54A"/><stop offset="1" stop-color="#FFAA00"/></linearGradient>')
    return svg(f'<path d="{d}" fill="url(#p)"/>', defs)


# ------------------------------------------------------------------ document


def p3(r, g, b, a=1.0) -> str:
    return f"display-p3:{r:.5f},{g:.5f},{b:.5f},{a:.5f}"


def build():
    assets = OUT / "Assets"
    shutil.rmtree(assets, ignore_errors=True)
    assets.mkdir(parents=True)
    files = {"track-glass.svg": track_glass(), "frames.svg": frames(), "found.svg": found(), "stem.svg": stem(),
             "ring.svg": ring(), "lens.svg": lens(), "play.svg": play()}
    for name, body in files.items():
        (assets / name).write_text(body)
    light = {"linear-gradient": [p3(0.24, 0.49, 1.0), p3(0.08, 0.22, 0.80)],
             "orientation": {"start": {"x": 0.5, "y": 0.0}, "stop": {"x": 0.5, "y": 1.0}}}
    dark = {"linear-gradient": [p3(0.13, 0.30, 0.78), p3(0.04, 0.10, 0.42)],
            "orientation": {"start": {"x": 0.5, "y": 0.0}, "stop": {"x": 0.5, "y": 1.0}}}
    doc = {
        # Dark mode keeps the blue, a shade deeper; specialising the fill also stops Icon Composer
        # recolouring the white parts.
        "fill-specializations": [{"value": light}, {"appearance": "dark", "value": dark}],
        # Groups are listed front to back.
        "groups": [
            {"name": "Lens", "layers": [
                {"name": "play", "image-name": "play.svg", "glass": False},
                {"name": "ring", "image-name": "ring.svg", "glass": False},
                {"name": "lens", "image-name": "lens.svg", "glass": True},
            ], "shadow": {"kind": "neutral", "opacity": 0.45}, "translucency": {"enabled": True, "value": 0.3},
             "specular": True, "lighting": "individual"},
            {"name": "Playhead", "layers": [{"name": "stem", "image-name": "stem.svg", "glass": False}],
             "shadow": {"kind": "neutral", "opacity": 0.4}, "translucency": {"enabled": False, "value": 0},
             "specular": True},
            {"name": "Found", "layers": [{"name": "found", "image-name": "found.svg", "glass": False}],
             "shadow": {"kind": "layer-color", "opacity": 0.5}, "translucency": {"enabled": False, "value": 0},
             "specular": True},
            {"name": "Timeline", "layers": [
                {"name": "frames", "image-name": "frames.svg", "glass": False},
                {"name": "track", "image-name": "track-glass.svg", "glass": True},
            ], "shadow": {"kind": "neutral", "opacity": 0.5}, "translucency": {"enabled": True, "value": 0.45},
             "specular": True, "blur-material": 0.25},
        ],
        "supported-platforms": {"circles": ["watchOS"], "squares": "shared"},
    }
    (OUT / "icon.json").write_text(json.dumps(doc, indent=2) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    build()
