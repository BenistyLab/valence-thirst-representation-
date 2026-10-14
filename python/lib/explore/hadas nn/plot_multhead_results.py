#!/usr/bin/env python3
"""Plot multihead decoder accuracy bars and pre-head simplex panels.

Inputs are the CSV exports produced by the direct32 analysis:
  * per_session_metrics.csv: session x source-run x decoder-head metrics
  * direct32_allruns_prehead_oof_predictions.csv: trial-level OOF probabilities
    from the pre-run decoder head, including protocol, mouse, source group,
    true_class, p_water, p_salt, and p_airpuff.

The script uses only the Python standard library and writes SVG figures.
Example:
  python plot_multhead_results.py --metrics per_session_metrics.csv \
      --predictions direct32_allruns_prehead_oof_predictions.csv --out-dir figures
"""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

PROTOCOLS = ("Mock", "1Hz", "20Hz")
SOURCES = ("pre", "slm_water", "slm_airpuff")
HEADS = ("pre", "slm_water", "slm_airpuff")
SOURCE_LABELS = {"pre": "Pre-run", "slm_water": "SLM-water", "slm_airpuff": "SLM-airpuff"}
HEAD_LABELS = {"pre": "Pre-run head", "slm_water": "SLM-water head", "slm_airpuff": "SLM-airpuff head"}
HEAD_COLORS = {"pre": "#3976b9", "slm_water": "#ef8a24", "slm_airpuff": "#c23a46"}
CLASS_COLORS = {"Water": "#3976b9", "NaCl": "#ef8a24", "Airpuff": "#c23a46"}


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def mouse_id(session):
    return session.split("-", 1)[0]


def esc(value):
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def plot_bars(metrics, out_path):
    # Average sessions within mouse first, so mice contribute equally.
    mouse_values = defaultdict(list)
    for r in metrics:
        key = (r["protocol"], mouse_id(r["session"]), r["source_run_group"], r["decoder_head"])
        mouse_values[key].append(float(r["accuracy"]))
    by_mouse = {k: sum(v) / len(v) for k, v in mouse_values.items()}
    grouped = defaultdict(list)
    for (protocol, _mouse, source, head), value in by_mouse.items():
        grouped[(protocol, source, head)].append(value)

    W, H, top, bottom = 1600, 820, 190, 650
    y = lambda v: bottom - max(0, min(1, v)) * (bottom - top)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<text x="800" y="42" text-anchor="middle" font-family="Arial" font-size="27" font-weight="bold">Decoding accuracy across trial sources and decoder heads</text>',
           '<text x="800" y="73" text-anchor="middle" font-family="Arial" font-size="16" fill="#666">Mean ± SEM across mice; sessions averaged within mouse</text>']
    for pi, protocol in enumerate(PROTOCOLS):
        x0, x1 = 115 + pi * 510, 545 + pi * 510
        svg.append(f'<text x="{(x0+x1)/2}" y="130" text-anchor="middle" font-family="Arial" font-size="22" font-weight="bold">{protocol}</text>')
        for tick in (0, .2, .4, .6, .8, 1):
            yy = y(tick)
            svg += [f'<line x1="{x0}" y1="{yy}" x2="{x1}" y2="{yy}" stroke="#dedede"/>',
                    f'<text x="{x0-12}" y="{yy+5}" text-anchor="end" font-family="Arial" font-size="13" fill="#555">{tick:.1f}</text>']
        svg += [f'<line x1="{x0}" y1="{top}" x2="{x0}" y2="{bottom}" stroke="#444"/>',
                f'<line x1="{x0}" y1="{bottom}" x2="{x1}" y2="{bottom}" stroke="#444"/>']
        for si, source in enumerate(SOURCES):
            center = x0 + 78 + si * 137
            for hi, head in enumerate(HEADS):
                vals = grouped.get((protocol, source, head), [])
                if not vals:
                    continue
                mean = sum(vals) / len(vals)
                sem = (math.sqrt(sum((v-mean)**2 for v in vals)/(len(vals)-1))/math.sqrt(len(vals))) if len(vals) > 1 else 0
                xx, bw = center + (hi-1)*29 - 12, 24
                yy = y(mean)
                svg.append(f'<rect x="{xx:.1f}" y="{yy:.1f}" width="{bw}" height="{bottom-yy:.1f}" fill="{HEAD_COLORS[head]}"/>')
                ey1, ey2, cx = y(mean+sem), y(mean-sem), xx+bw/2
                svg += [f'<line x1="{cx}" y1="{ey1}" x2="{cx}" y2="{ey2}" stroke="#222" stroke-width="1.5"/>',
                        f'<line x1="{cx-5}" y1="{ey1}" x2="{cx+5}" y2="{ey1}" stroke="#222"/>',
                        f'<line x1="{cx-5}" y1="{ey2}" x2="{cx+5}" y2="{ey2}" stroke="#222"/>']
            svg.append(f'<text x="{center}" y="{bottom+27}" text-anchor="middle" font-family="Arial" font-size="14">{SOURCE_LABELS[source]}</text>')
        if pi == 0:
            svg.append(f'<text x="35" y="420" text-anchor="middle" transform="rotate(-90 35 420)" font-family="Arial" font-size="16">Accuracy</text>')
    for i, head in enumerate(HEADS):
        x = 410 + i*275
        svg += [f'<rect x="{x}" y="738" width="18" height="18" fill="{HEAD_COLORS[head]}"/>',
                f'<text x="{x+27}" y="753" font-family="Arial" font-size="15">{HEAD_LABELS[head]}</text>']
    svg += ['<text x="800" y="792" text-anchor="middle" font-family="Arial" font-size="13" fill="#666">Bars use accuracy; each source-run group is decoded by all three heads.</text>', '</svg>']
    out_path.write_text("\n".join(svg), encoding="utf-8")


def plot_simplex(predictions, out_path):
    """3x3 grid: rows = source run group, columns = protocol. All panels use pre-run head."""
    SOURCE_ROW_LABELS = {
        "pre": "Pre-run trials",
        "slm_water": "SLM-water trials",
        "slm_airpuff": "SLM-airpuff trials",
    }
    rows = [r for r in predictions if r["source_run_group"] in SOURCES]

    # Layout
    panel_w, panel_h = 460, 390
    margin_left, margin_top = 150, 115
    gap_x, gap_y = 20, 55
    W = margin_left + 3 * panel_w + 2 * gap_x + 40
    H = margin_top + 3 * panel_h + 2 * gap_y + 65
    tri_pad_l, tri_pad_r, tri_pad_top, tri_pad_bot = 58, 18, 30, 38

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{W//2}" y="36" text-anchor="middle" font-family="Arial" font-size="23" font-weight="bold">Pre-run decoder head: trial probabilities</text>',
        f'<text x="{W//2}" y="58" text-anchor="middle" font-family="Arial" font-size="13" fill="#666">Rows = source run group · Columns = protocol · All panels decoded by pre-run head</text>',
        f'<text x="{W//2}" y="76" text-anchor="middle" font-family="Arial" font-size="13" fill="#666">Small circles = individual trials · Diamonds = mouse × class means</text>',
    ]

    # Column headers
    for ci, protocol in enumerate(PROTOCOLS):
        cx = margin_left + ci * (panel_w + gap_x) + panel_w // 2
        svg.append(f'<text x="{cx}" y="{margin_top - 8}" text-anchor="middle" font-family="Arial" font-size="18" font-weight="bold">{protocol}</text>')

    # Row labels (rotated)
    for ri, source in enumerate(SOURCES):
        cy = margin_top + ri * (panel_h + gap_y) + panel_h // 2
        label = SOURCE_ROW_LABELS[source]
        svg.append(f'<text x="20" y="{cy}" text-anchor="middle" transform="rotate(-90 20 {cy})" font-family="Arial" font-size="14" font-weight="bold">{label}</text>')

    def draw_panel(out, panel_rows, ox, oy):
        left = ox + tri_pad_l
        right = ox + panel_w - tri_pad_r
        tri_h = (right - left) * math.sqrt(3) / 2
        base = oy + panel_h - tri_pad_bot
        top_y = base - tri_h
        verts = [(left, base), (right, base), ((left + right) / 2, top_y)]

        def xy(p, v=verts):
            return (sum(p[j] * v[j][0] for j in range(3)),
                    sum(p[j] * v[j][1] for j in range(3)))

        # Gridlines
        for t in (.2, .4, .6, .8):
            for i in range(3):
                a = [0., 0., 0.]; b = [0., 0., 0.]
                a[i] = b[i] = t; a[(i+1)%3] = 1 - t; b[(i+2)%3] = 1 - t
                xa, ya = xy(a); xb, yb = xy(b)
                out.append(f'<line x1="{xa:.1f}" y1="{ya:.1f}" x2="{xb:.1f}" y2="{yb:.1f}" stroke="#ddd" stroke-width="0.8"/>')

        # Triangle border
        pts = " ".join(f"{v[0]},{v[1]}" for v in verts)
        out.append(f'<polygon points="{pts}" fill="none" stroke="#333" stroke-width="1.5"/>')

        # Individual trial dots
        for r in panel_rows:
            p = [float(r["p_water"]), float(r["p_salt"]), float(r["p_airpuff"])]
            x0, y0 = xy(p)
            out.append(f'<circle cx="{x0:.1f}" cy="{y0:.1f}" r="3" fill="{CLASS_COLORS[r["true_class"]]}" fill-opacity="0.38"/>')

        # Mouse x class mean diamonds
        means = defaultdict(list)
        for r in panel_rows:
            means[(r["mouse"], r["true_class"])].append(
                [float(r["p_water"]), float(r["p_salt"]), float(r["p_airpuff"])])
        for (_mouse, cls), ps in means.items():
            p = [sum(v[j] for v in ps) / len(ps) for j in range(3)]
            x0, y0 = xy(p)
            out.append(f'<polygon points="{x0:.1f},{y0-6:.1f} {x0+6:.1f},{y0:.1f} {x0:.1f},{y0+6:.1f} {x0-6:.1f},{y0:.1f}" fill="{CLASS_COLORS[cls]}" stroke="#222" stroke-width="0.7"/>')

        # Vertex labels
        mid_x = (left + right) / 2
        out += [
            f'<text x="{left-4}" y="{base+14}" text-anchor="end" font-family="Arial" font-size="10">Water</text>',
            f'<text x="{right+4}" y="{base+14}" text-anchor="start" font-family="Arial" font-size="10">NaCl</text>',
            f'<text x="{mid_x:.1f}" y="{top_y-6:.1f}" text-anchor="middle" font-family="Arial" font-size="10">Airpuff</text>',
        ]

        # Accuracy + N
        correct = sum(1 for r in panel_rows if r.get("predicted_class", "") == r["true_class"])
        acc = correct / len(panel_rows) if panel_rows else float("nan")
        out.append(f'<text x="{ox + panel_w//2}" y="{oy + tri_pad_top - 4}" text-anchor="middle" font-family="Arial" font-size="11" fill="#444">acc={acc:.1%} | N={len(panel_rows)}</text>')

    # Draw all 9 panels
    for ri, source in enumerate(SOURCES):
        for ci, protocol in enumerate(PROTOCOLS):
            ox = margin_left + ci * (panel_w + gap_x)
            oy = margin_top + ri * (panel_h + gap_y)
            panel_rows = [r for r in rows if r["source_run_group"] == source and r["protocol"] == protocol]
            svg.append(f'<rect x="{ox}" y="{oy}" width="{panel_w}" height="{panel_h}" rx="5" fill="#fafafa" stroke="#ccc" stroke-width="0.8"/>')
            draw_panel(svg, panel_rows, ox, oy)

    # Legend
    legend_y = H - 30
    svg.append(f'<text x="80" y="{legend_y+4}" font-family="Arial" font-size="13" font-weight="bold">Trial label:</text>')
    lx = 185
    for cls in ("Water", "NaCl", "Airpuff"):
        svg += [
            f'<circle cx="{lx}" cy="{legend_y}" r="5" fill="{CLASS_COLORS[cls]}"/>',
            f'<text x="{lx+10}" y="{legend_y+5}" font-family="Arial" font-size="13">{cls}</text>',
        ]
        lx += 110

    svg.append('</svg>')
    out_path.write_text("\n".join(svg), encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True, type=Path, help="per_session_metrics.csv from the training run")
    parser.add_argument("--predictions", required=True, type=Path, help="pre-head OOF trial prediction CSV")
    parser.add_argument("--out-dir", required=True, type=Path, help="directory for the SVG figures")
    args=parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_bars(read_csv(args.metrics), args.out_dir/"accuracy_mean_sem_all_heads.svg")
    plot_simplex(read_csv(args.predictions), args.out_dir/"prehead_oof_simplex_by_condition.svg")
    print(f"Saved figures in {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()