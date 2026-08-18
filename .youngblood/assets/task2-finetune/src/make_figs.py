"""Figures for the task2-finetune tour. Emits SVG at 700px wide; rasterize with svg2png.sh at 2x."""

import math
from pathlib import Path

OUT = Path(__file__).resolve().parents[1]

BG, INK, MUTED, RULE = "#f7f8fa", "#1f2328", "#5b6470", "#c9cfd6"
TEAL, LESION, GREEN, AMBER = "#0d7c8b", "#bb3d78", "#1a7f4b", "#9d6a0d"
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"


def txt(x, y, s, size=12, fill=INK, weight="normal", anchor="start", font=FONT):
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" font-weight="{weight}" '
        f'text-anchor="{anchor}" font-family="{font}">{s}</text>'
    )


def box(x, y, w, h, fill="#ffffff", stroke=RULE, width=1.2, rx=5, dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{width}"{d}/>'
    )


def arrow(x1, y1, x2, y2, color=MUTED, width=1.6):
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
        f'stroke-width="{width}" marker-end="url(#ah)"/>'
    )


def svg(w, h, body):
    marker = (
        '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
        f'markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{MUTED}"/>'
        "</marker></defs>"
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="{FONT}">'
        f'<rect width="{w}" height="{h}" rx="10" fill="{BG}"/>{marker}{body}</svg>'
    )


def fig1_system():
    """What the pipeline is, and which two pieces this branch replaces."""
    s = [txt(24, 30, "One subject through task 2, and the two pieces this branch changes", 14, INK, "600")]

    stages = [
        (24, "FLAIR scan", "one patient", False),
        (160, "frozen ViT-L", "8mm patches", False),
        (296, "head", "tokens to voxels", True),
        (432, "threshold", "probability to mask", False),
        (568, "binary mask", "what is scored", False),
    ]
    for x, name, sub, changed in stages:
        fill = "#eef7f8" if changed else "#ffffff"
        stroke = TEAL if changed else RULE
        s.append(box(x, 62, 108, 52, fill, stroke, 2.0 if changed else 1.2))
        s.append(txt(x + 54, 84, name, 12.5, INK, "650", "middle"))
        s.append(txt(x + 54, 101, sub, 10.5, MUTED, "normal", "middle"))
        if x < 568:
            s.append(arrow(x + 112, 88, x + 132, 88))

    s.append(txt(350, 133, "replaced: logistic head to conv decoder", 11, TEAL, "600", "middle"))
    s.append(f'<line x1="24" y1="152" x2="676" y2="152" stroke="{RULE}" stroke-width="1"/>')

    s.append(txt(24, 176, "the protocol, run 23 times, once per held-out subject", 12.5, INK, "600"))
    s.append(box(24, 190, 208, 58, "#ffffff"))
    s.append(txt(128, 212, "fit on the other 22", 12, INK, "normal", "middle"))
    s.append(txt(128, 229, "predict the held-out one", 11, MUTED, "normal", "middle"))
    s.append(arrow(236, 219, 256, 219))

    s.append(box(260, 190, 208, 58, "#eef7f8", TEAL, 2.0))
    s.append(txt(364, 208, "score at 80 thresholds", 12, INK, "normal", "middle"))
    s.append(txt(364, 226, "Dice", 12, MUTED, "600", "middle"))
    s.append(txt(364, 242, "+ NSD, added here", 11, TEAL, "600", "middle"))
    s.append(arrow(472, 219, 492, 219))

    s.append(box(496, 190, 180, 58, "#ffffff"))
    s.append(txt(586, 212, "mean and bootstrap CI", 11.5, INK, "normal", "middle"))
    s.append(txt(586, 229, "metrics.json", 11, MUTED, "normal", "middle", MONO))

    return svg(700, 268, "".join(s))


def blob(cx, cy, r, seed=3):
    """A closed lesion-ish outline, deterministic."""
    pts = []
    for i in range(14):
        a = 2 * math.pi * i / 14
        wob = 1 + 0.22 * math.sin(3 * a + seed) + 0.12 * math.cos(5 * a + 2 * seed)
        pts.append((cx + r * wob * math.cos(a), cy + r * wob * math.sin(a)))
    d = f"M{pts[0][0]:.1f},{pts[0][1]:.1f}"
    for i in range(len(pts)):
        p, q = pts[i], pts[(i + 1) % len(pts)]
        mx, my = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
        d += f" Q{p[0]:.1f},{p[1]:.1f} {mx:.1f},{my:.1f}"
    return d + " Z", pts


def fig2_staircase():
    """The same lesion drawn by a block-constant head and by a per-voxel head."""
    s = [txt(24, 30, "What each head can draw, against the same reference outline", 14, INK, "600")]
    cell = 22

    for panel, (ox, label, sub) in enumerate(
        [(40, "logistic head, before", "one value per 8mm block"),
         (390, "conv decoder, after", "one value per 1mm voxel")]
    ):
        oy = 62
        s.append(box(ox - 10, oy - 10, 8 * cell + 20, 8 * cell + 20, "#ffffff"))
        path, _ = blob(ox + 4 * cell, oy + 4 * cell, 2.6 * cell)

        if panel == 0:
            for gx in range(8):
                for gy in range(8):
                    x0, y0 = ox + gx * cell, oy + gy * cell
                    dx = (x0 + cell / 2) - (ox + 4 * cell)
                    dy = (y0 + cell / 2) - (oy + 4 * cell)
                    a = math.atan2(dy, dx)
                    wob = 1 + 0.22 * math.sin(3 * a + 3) + 0.12 * math.cos(5 * a + 6)
                    if math.hypot(dx, dy) < 2.6 * cell * wob:
                        s.append(box(x0, y0, cell, cell, TEAL, TEAL, 0.5, 0))
            for g in range(9):
                s.append(
                    f'<line x1="{ox + g * cell}" y1="{oy}" x2="{ox + g * cell}" y2="{oy + 8 * cell}" '
                    f'stroke="{TEAL}" stroke-width="0.6" opacity="0.55"/>'
                )
                s.append(
                    f'<line x1="{ox}" y1="{oy + g * cell}" x2="{ox + 8 * cell}" y2="{oy + g * cell}" '
                    f'stroke="{TEAL}" stroke-width="0.6" opacity="0.55"/>'
                )
        else:
            s.append(f'<path d="{path}" fill="{TEAL}" opacity="0.75" transform="translate(1.5,1.5)"/>')
            for g in range(9):
                s.append(
                    f'<line x1="{ox + g * cell}" y1="{oy}" x2="{ox + g * cell}" y2="{oy + 8 * cell}" '
                    f'stroke="{TEAL}" stroke-width="0.6" opacity="0.18"/>'
                )
                s.append(
                    f'<line x1="{ox}" y1="{oy + g * cell}" x2="{ox + 8 * cell}" y2="{oy + g * cell}" '
                    f'stroke="{TEAL}" stroke-width="0.6" opacity="0.18"/>'
                )

        s.append(f'<path d="{path}" fill="none" stroke="{LESION}" stroke-width="2.4"/>')
        s.append(txt(ox + 4 * cell, oy + 8 * cell + 34, label, 12.5, INK, "650", "middle"))
        s.append(txt(ox + 4 * cell, oy + 8 * cell + 50, sub, 11, MUTED, "normal", "middle"))

    s.append(f'<line x1="24" y1="304" x2="676" y2="304" stroke="{RULE}" stroke-width="1"/>')
    s.append(txt(350, 326, "Pink is the reference outline. Teal is what the head can express.", 11.5, MUTED, "normal", "middle"))
    return svg(700, 342, "".join(s))


def fig3_decoder():
    """The shape ladder from token grid to voxel logits."""
    s = [txt(24, 30, "ConvDecoder: three doublings turn one token per patch into one value per voxel", 13.5, INK, "600")]

    rows = [
        ("26 x 30 x 26", "1024", "encoder tokens, one per 8mm patch", 0),
        ("26 x 30 x 26", "32", "1x1x1 conv, then GELU", 1),
        ("52 x 60 x 52", "16", "2x upsample, 3x3x3 conv, GELU", 2),
        ("104 x 120 x 104", "8", "2x upsample, 3x3x3 conv, GELU", 3),
        ("208 x 240 x 208", "4", "2x upsample, 3x3x3 conv", 4),
        ("208 x 240 x 208", "1", "3x3x3 conv, logits", 5),
    ]
    y = 58
    for grid, chans, note, i in rows:
        w = 40 + i * 34
        s.append(box(24, y, w, 26, "#eef7f8" if i else "#ffffff", TEAL if i else RULE, 1.4))
        s.append(txt(400, y + 17, grid, 12, INK, "600", "end", MONO))
        s.append(txt(410, y + 17, "x", 11, MUTED))
        s.append(txt(460, y + 17, chans, 12, TEAL, "700", "end", MONO))
        s.append(txt(478, y + 17, note, 11.5, MUTED))
        if i < 5:
            s.append(arrow(38, y + 26, 38, y + 40, RULE, 1.2))
        y += 40

    s.append(f'<line x1="24" y1="{y + 4}" x2="676" y2="{y + 4}" stroke="{RULE}" stroke-width="1"/>')
    s.append(txt(24, y + 26, "The bar on the left is the spatial grid growing 8x on each side. The number", 11.5, MUTED))
    s.append(txt(24, y + 42, "after the x is channels, shrinking as the grid grows.", 11.5, MUTED))
    return svg(700, y + 58, "".join(s))


def fig4_tolerance():
    """Why a 1mm NSD tolerance is unreachable through-plane at 6mm slices."""
    s = [txt(24, 30, "NSD scores a boundary within 1mm. These scans are 0.7mm across and 6mm deep.", 13.5, INK, "600")]

    ox, scale, truth_mm = 40, 15.0, 21.0
    tx = ox + truth_mm * scale

    def row(oy, label, pitch, note_hit, note_miss):
        s.append(txt(24, oy - 14, label, 12.5, INK, "650"))
        s.append(f'<rect x="{tx - scale:.1f}" y="{oy}" width="{2 * scale:.1f}" height="40" fill="{TEAL}" opacity="0.20"/>')
        n = int(42 / pitch) + 1
        hits = 0
        for i in range(n):
            v = i * pitch
            x = ox + v * scale
            if x > 640:
                break
            inside = abs(v - truth_mm) <= 1.0
            hits += inside
            color, w = (GREEN, 2.6) if inside else (RULE, 1.0)
            s.append(f'<line x1="{x:.1f}" y1="{oy}" x2="{x:.1f}" y2="{oy + 40}" stroke="{color}" stroke-width="{w}"/>')
        s.append(f'<line x1="{tx:.1f}" y1="{oy - 6}" x2="{tx:.1f}" y2="{oy + 46}" stroke="{LESION}" stroke-width="2.6"/>')
        note = note_hit if hits else note_miss
        color = GREEN if hits else AMBER
        s.append(txt(24, oy + 66, note, 12, color, "600"))
        return hits

    row(74, "in plane, a voxel edge every 0.7mm", 0.7,
        "3 places the boundary can land are inside the 1mm band", "")
    row(174, "through plane, a slice edge every 6mm", 6.0,
        "", "0 places inside the band. The nearest is 3mm out, three times the tolerance.")

    s.append(f'<line x1="24" y1="264" x2="676" y2="264" stroke="{RULE}" stroke-width="1"/>')
    s.append(txt(24, 286, "Pink is the true boundary. The teal band is the 1mm tolerance around it. Each", 11.5, MUTED))
    s.append(txt(24, 302, "tick is a place a voxel boundary can actually sit. Green ticks score, grey ones do not.", 11.5, MUTED))
    s.append(txt(24, 324, "Nothing about the head moves the bottom row. That is the acquisition.", 12, INK, "600"))
    return svg(700, 340, "".join(s))


def fig5_results():
    """Paired deltas: Dice clears zero, NSD does not."""
    s = [txt(24, 30, "Conv decoder against the same-machine linear head, paired over 23 subjects", 13.5, INK, "600")]

    x0, x1 = 190, 640
    lo, hi = -0.08, 0.20

    def px(v):
        return x0 + (v - lo) / (hi - lo) * (x1 - x0)

    for v in (-0.05, 0.0, 0.05, 0.10, 0.15, 0.20):
        s.append(f'<line x1="{px(v):.1f}" y1="56" x2="{px(v):.1f}" y2="200" stroke="{RULE}" stroke-width="0.8"/>')
        s.append(txt(px(v), 216, f"{v:+.2f}", 10.5, MUTED, "normal", "middle", MONO))
    s.append(f'<line x1="{px(0):.1f}" y1="56" x2="{px(0):.1f}" y2="200" stroke="{INK}" stroke-width="1.8"/>')
    s.append(txt(px(0), 48, "no change", 10.5, INK, "600", "middle"))

    rows = [
        ("Dice, seed 4466", 0.0779, 0.0099, 0.1589, GREEN, 80),
        ("Dice, seed 1234", 0.0853, 0.0123, 0.1732, GREEN, 110),
        ("NSD, seed 4466", 0.0238, -0.0528, 0.1004, AMBER, 152),
        ("NSD, seed 1234", 0.0431, -0.0349, 0.1274, AMBER, 182),
    ]
    for label, point, low, high, color, y in rows:
        s.append(txt(176, y + 4, label, 12, INK, "normal", "end"))
        s.append(f'<line x1="{px(low):.1f}" y1="{y}" x2="{px(high):.1f}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        for v in (low, high):
            s.append(f'<line x1="{px(v):.1f}" y1="{y - 6}" x2="{px(v):.1f}" y2="{y + 6}" stroke="{color}" stroke-width="2.4"/>')
        s.append(f'<circle cx="{px(point):.1f}" cy="{y}" r="5" fill="{color}"/>')

    s.append(f'<line x1="24" y1="236" x2="676" y2="236" stroke="{RULE}" stroke-width="1"/>')
    s.append(txt(24, 258, "Both Dice intervals sit entirely right of the line. Both NSD intervals cross it.", 12, INK, "600"))
    s.append(txt(24, 275, "Bars are 95% bootstrap intervals on the paired per-subject difference.", 11.5, MUTED))
    return svg(700, 292, "".join(s))


for name, fig in [
    ("fig1-system", fig1_system()),
    ("fig2-staircase", fig2_staircase()),
    ("fig3-decoder", fig3_decoder()),
    ("fig4-tolerance", fig4_tolerance()),
    ("fig5-results", fig5_results()),
]:
    (OUT / "src" / f"{name}.svg").write_text(fig)
    print(name, len(fig))
