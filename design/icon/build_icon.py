# SPDX-License-Identifier: GPL-3.0-or-later
"""Builds AppIcon.icon, the Icon Composer document for Video Scrubber.

A bright yellow sponge with suds slides across a glass timeline, leaving soapy streaks.
Every layer is an SVG on the 1024-point icon canvas; Icon Composer adds the squircle,
Liquid Glass, shadows and specular highlights from icon.json.

    python3 design/icon/build_icon.py      # writes design/icon/AppIcon.icon
"""

import json
import math
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent / "AppIcon.icon"
W = 1024

# Sponge placement: centre, tilt (degrees clockwise), and size of the top face.
SX, SY, TILT = 590, 556, 24
BW, BH, BR = 430, 262, 74
DEPTH = 46  # thickness of the visible side


def svg(body: str, defs: str = "") -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{W}" viewBox="0 0 {W} {W}">'
            f"<defs>{defs}</defs>{body}</svg>\n")


def rrect(x, y, w, h, r, **attrs) -> str:
    a = " ".join(f'{k.replace("_", "-")}="{v}"' for k, v in attrs.items())
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{r:.1f}" {a}/>'


# ------------------------------------------------------------------ timeline

BAR_X, BAR_Y, BAR_W, BAR_H, BAR_R = 112, 512, 800, 180, 60


def timeline_glass() -> str:
    defs = ('<linearGradient id="g" x1="0" y1="0" x2="0" y2="1">'
            '<stop offset="0" stop-color="#FFFFFF" stop-opacity="0.62"/>'
            '<stop offset="1" stop-color="#DCE8FF" stop-opacity="0.38"/></linearGradient>')
    return svg(rrect(BAR_X, BAR_Y, BAR_W, BAR_H, BAR_R, fill="url(#g)"), defs)


def timeline_ticks() -> str:
    """Ruler marks along the bottom of the glass, like the app's timeline."""
    out = []
    x0, x1, base = BAR_X + 44, BAR_X + BAR_W - 44, BAR_Y + BAR_H - 26
    n = 20
    for i in range(n + 1):
        x = x0 + (x1 - x0) * i / n
        h = 42 if i % 5 == 0 else 22
        out.append(rrect(x - 3.5, base - h, 7, h, 3.5, fill="#FFFFFF", fill_opacity="0.85"))
    return svg("".join(out))


def sponge_left_edge(y: float) -> float:
    """Leftmost x of the sponge (top face and side) on the horizontal line y, in canvas points."""
    best = SX
    for k in range(720):
        a = 2 * math.pi * k / 720
        # Superellipse approximates the rounded rectangle outline.
        c, s_ = math.cos(a), math.sin(a)
        lx = BW / 2 * math.copysign(abs(c) ** 0.35, c)
        for ly0 in (0, DEPTH):
            ly = BH / 2 * math.copysign(abs(s_) ** 0.35, s_) + ly0
            x, yy = local_to_canvas(lx, ly)
            if abs(yy - y) < 6:
                best = min(best, x)
    return best


def streaks() -> str:
    """Soapy smears left on the glass behind the sponge, fading out to the left."""
    defs = []
    body = []
    rnd = random.Random(7)
    tail_end = sponge_left_edge(BAR_Y + BAR_H / 2) - 10  # where the sponge's trailing edge meets the bar
    lanes = [(BAR_Y + 44, 22, 1.0), (BAR_Y + 84, 30, 0.95), (BAR_Y + 124, 18, 0.85)]
    for i, (y, thick, alpha) in enumerate(lanes):
        start = BAR_X + 22 + i * 30
        end = sponge_left_edge(y) - thick * 0.6 - 8
        defs.append(f'<linearGradient id="s{i}" x1="0" y1="0" x2="1" y2="0">'
                    f'<stop offset="0" stop-color="#FFFFFF" stop-opacity="0"/>'
                    f'<stop offset="0.45" stop-color="#FFFFFF" stop-opacity="{alpha * 0.7:.2f}"/>'
                    f'<stop offset="1" stop-color="#FFFFFF" stop-opacity="{alpha:.2f}"/></linearGradient>')
        # A tapered, slightly wavy band.
        pts_top, pts_bot = [], []
        steps = 14
        for k in range(steps + 1):
            t = k / steps
            x = start + (end - start) * t
            wave = math.sin(t * math.pi * 2.2 + i) * 4
            half = thick * (0.35 + 0.65 * t) / 2
            pts_top.append((x, y + wave - half))
            pts_bot.append((x, y + wave + half))
        half_end = (pts_bot[-1][1] - pts_top[-1][1]) / 2
        d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts_top)
        d += f" A{half_end:.1f},{half_end:.1f} 0 0 1 {pts_bot[-1][0]:.1f},{pts_bot[-1][1]:.1f}"  # rounded tip
        d += " L" + " L".join(f"{x:.1f},{y:.1f}" for x, y in reversed(pts_bot)) + " Z"
        body.append(f'<path d="{d}" fill="url(#s{i})" stroke-linejoin="round"/>')
    # Little suds left in the trail.
    for _ in range(16):
        t = rnd.random() ** 0.6
        x = BAR_X + 70 + (tail_end - BAR_X - 70) * t
        y = BAR_Y + 30 + rnd.random() * (BAR_H - 70)
        r = 4 + 10 * t * rnd.random()
        body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="#FFFFFF" fill-opacity="{0.35 + 0.6 * t:.2f}"/>')
    return svg("".join(body), "".join(defs))


# ------------------------------------------------------------------ sponge


def pores(rnd: random.Random) -> list[tuple[float, float, float, float]]:
    """Pore positions on the top face (local coords), kept clear of the edge and of each other."""
    out = []
    tries = 0
    while len(out) < 24 and tries < 4000:
        tries += 1
        r = rnd.choice([9, 11, 13, 16, 19, 23, 27])
        x = rnd.uniform(-BW / 2 + BR * 0.45 + r, BW / 2 - BR * 0.45 - r)
        y = rnd.uniform(-BH / 2 + 26 + r, BH / 2 - 22 - r)
        # Inside the rounded corners.
        cx, cy = max(abs(x) - (BW / 2 - BR), 0), max(abs(y) - (BH / 2 - BR), 0)
        if math.hypot(cx, cy) > BR - r - 16:
            continue
        if any(math.hypot(x - a, y - b) < r + c + 14 for a, b, c, _ in out):
            continue
        out.append((x, y, r, rnd.uniform(0.75, 1.0)))
    return out


def sponge() -> str:
    rnd = random.Random(11)
    defs = """
    <linearGradient id="top" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#FFF89A"/><stop offset="0.3" stop-color="#FFEB3B"/>
      <stop offset="0.75" stop-color="#FFD60A"/><stop offset="1" stop-color="#FFC400"/></linearGradient>
    <linearGradient id="side" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#F7BF00"/><stop offset="1" stop-color="#E29C00"/></linearGradient>
    <linearGradient id="sheen" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#FFFFFF" stop-opacity="0.55"/><stop offset="1" stop-color="#FFFFFF" stop-opacity="0"/></linearGradient>
    <radialGradient id="pore" cx="0.42" cy="0.35" r="0.75">
      <stop offset="0" stop-color="#A65A00"/><stop offset="0.7" stop-color="#D08400"/><stop offset="1" stop-color="#EBA800"/></radialGradient>
    <linearGradient id="lip" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#FFF7B0" stop-opacity="0"/><stop offset="1" stop-color="#FFF7B0" stop-opacity="0.95"/></linearGradient>
    """
    x, y = -BW / 2, -BH / 2
    g = []
    # Side (thickness), then the darker under-edge, then the top face.
    g.append(rrect(x + 6, y + DEPTH, BW - 4, BH, BR, fill="url(#side)"))
    for i in range(7):  # pores cut into the side
        px = x + 50 + i * (BW - 100) / 6 + rnd.uniform(-10, 10)
        pr = rnd.choice([7, 9, 11])
        g.append(f'<ellipse cx="{px:.1f}" cy="{y + BH + DEPTH * 0.45:.1f}" rx="{pr}" ry="{pr * 0.7:.1f}" fill="#B86600" fill-opacity="0.75"/>')
    g.append(rrect(x, y, BW, BH, BR, fill="url(#top)"))
    # Pores: dark hollow, a lit lower lip, and a shadowed upper rim.
    for px, py, r, sq in pores(rnd):
        ry = r * sq
        g.append(f'<ellipse cx="{px:.1f}" cy="{py:.1f}" rx="{r:.1f}" ry="{ry:.1f}" fill="url(#pore)"/>')
        g.append(f'<ellipse cx="{px:.1f}" cy="{py + ry * 0.35:.1f}" rx="{r * 0.82:.1f}" ry="{ry * 0.6:.1f}" fill="url(#lip)"/>')
        g.append(f'<ellipse cx="{px - r * 0.1:.1f}" cy="{py - ry * 0.28:.1f}" rx="{r * 0.7:.1f}" ry="{ry * 0.42:.1f}" fill="#6E3600" fill-opacity="0.55"/>')
    # Fine texture.
    for _ in range(36):
        px = rnd.uniform(x + 30, x + BW - 30)
        py = rnd.uniform(y + 24, y + BH - 24)
        g.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{rnd.uniform(1.8, 3.6):.1f}" fill="#D08A00" fill-opacity="0.5"/>')
    # Soft sheen along the upper edge and a crisp rim highlight.
    g.append(rrect(x + 18, y + 10, BW - 36, BH * 0.38, BR * 0.8, fill="url(#sheen)"))
    g.append(rrect(x + 1.5, y + 1.5, BW - 3, BH - 3, BR - 1.5, fill="none", stroke="#FFF9C9", stroke_width="3", stroke_opacity="0.9"))
    body = f'<g transform="translate({SX},{SY}) rotate({TILT})">{"".join(g)}</g>'
    return svg(body, defs)


# ------------------------------------------------------------------ suds


def local_to_canvas(lx, ly):
    a = math.radians(TILT)
    return SX + lx * math.cos(a) - ly * math.sin(a), SY + lx * math.sin(a) + ly * math.cos(a)


def foam() -> str:
    """A rounded heap of lather on the sponge's upper-left, spilling onto the glass behind it."""
    defs = """
    <radialGradient id="f" cx="0.38" cy="0.32" r="0.72">
      <stop offset="0" stop-color="#FFFFFF"/><stop offset="0.65" stop-color="#F3F7FF"/><stop offset="1" stop-color="#C9D7F0"/></radialGradient>
    """
    blobs = [  # (local x, local y, r) on the sponge's top face, back to front
        (-150, -128, 58), (-82, -150, 74), (-8, -138, 62), (52, -116, 48), (-200, -86, 46),
        (-120, -98, 54), (-40, -96, 50), (96, -96, 34), (-178, -40, 32),
    ]
    out = []
    for lx, ly, r in blobs:
        cx, cy = local_to_canvas(lx, ly)
        out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="url(#f)"/>')
        out.append(f'<ellipse cx="{cx - r * 0.3:.1f}" cy="{cy - r * 0.38:.1f}" rx="{r * 0.28:.1f}" ry="{r * 0.18:.1f}" fill="#FFFFFF"/>')
    # Lather dragged off the trailing edge onto the timeline.
    for lx, ly, r in [(-240, 40, 30), (-262, 92, 22), (-292, 70, 16), (-322, 104, 11)]:
        cx, cy = local_to_canvas(lx, ly)
        out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="url(#f)"/>')
    return svg("".join(out), defs)


def bubbles() -> str:
    """Clear bubbles rising off the suds; Icon Composer renders them as Liquid Glass."""
    defs = """
    <radialGradient id="b" cx="0.5" cy="0.5" r="0.5">
      <stop offset="0" stop-color="#FFFFFF" stop-opacity="0.18"/><stop offset="0.82" stop-color="#FFFFFF" stop-opacity="0.32"/>
      <stop offset="1" stop-color="#FFFFFF" stop-opacity="0.9"/></radialGradient>
    """
    out = []
    for cx, cy, r in [(748, 262, 54), (842, 356, 30), (654, 200, 25), (884, 240, 18), (300, 462, 15)]:
        out.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="url(#b)"/>')
        out.append(f'<ellipse cx="{cx - r * 0.34:.1f}" cy="{cy - r * 0.4:.1f}" rx="{r * 0.26:.1f}" ry="{r * 0.16:.1f}" '
                   f'fill="#FFFFFF" transform="rotate(-30 {cx - r * 0.34:.1f} {cy - r * 0.4:.1f})"/>')
    return svg("".join(out), defs)


# ------------------------------------------------------------------ document


def p3(r, g, b, a=1.0) -> str:
    return f"display-p3:{r:.5f},{g:.5f},{b:.5f},{a:.5f}"


def build():
    assets = OUT / "Assets"
    assets.mkdir(parents=True, exist_ok=True)
    files = {
        "timeline-glass.svg": timeline_glass(), "timeline-ticks.svg": timeline_ticks(), "streaks.svg": streaks(),
        "sponge.svg": sponge(), "foam.svg": foam(), "bubbles.svg": bubbles(),
    }
    for name, body in files.items():
        (assets / name).write_text(body)
    light = {"linear-gradient": [p3(0.24, 0.49, 1.0), p3(0.08, 0.22, 0.80)],
             "orientation": {"start": {"x": 0.5, "y": 0.0}, "stop": {"x": 0.5, "y": 1.0}}}
    dark = {"linear-gradient": [p3(0.13, 0.30, 0.78), p3(0.04, 0.10, 0.42)],
            "orientation": {"start": {"x": 0.5, "y": 0.0}, "stop": {"x": 0.5, "y": 1.0}}}
    doc = {
        # Dark mode keeps the blue, a shade deeper; specialising the fill also stops Icon Composer
        # recolouring the white suds.
        "fill-specializations": [{"value": light}, {"appearance": "dark", "value": dark}],
        # Groups are listed front to back.
        "groups": [
            {"name": "Bubbles", "layers": [{"name": "bubbles", "image-name": "bubbles.svg", "glass": True}],
             "shadow": {"kind": "neutral", "opacity": 0.25}, "translucency": {"enabled": True, "value": 0.35},
             "specular": True, "lighting": "individual"},
            {"name": "Suds", "layers": [
                {"name": "foam", "image-name": "foam.svg", "glass": False},
                {"name": "streaks", "image-name": "streaks.svg", "glass": False, "opacity": 0.9},
            ], "shadow": {"kind": "neutral", "opacity": 0.35}, "translucency": {"enabled": False, "value": 0},
             "specular": True},
            {"name": "Sponge", "layers": [{"name": "sponge", "image-name": "sponge.svg", "glass": False}],
             "shadow": {"kind": "layer-color", "opacity": 0.6}, "translucency": {"enabled": False, "value": 0},
             "specular": True},
            {"name": "Timeline", "layers": [
                {"name": "ticks", "image-name": "timeline-ticks.svg", "glass": False, "opacity": 0.85},
                {"name": "glass", "image-name": "timeline-glass.svg", "glass": True},
            ], "shadow": {"kind": "neutral", "opacity": 0.5}, "translucency": {"enabled": True, "value": 0.45},
             "specular": True, "blur-material": 0.25},
        ],
        "supported-platforms": {"circles": ["watchOS"], "squares": "shared"},
    }
    (OUT / "icon.json").write_text(json.dumps(doc, indent=2) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    build()
