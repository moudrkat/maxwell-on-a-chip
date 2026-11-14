"""The hero picture: one thermometer, three rooms; in each, what the chip wrote, the softmax that wrote it, and the
entropy that came out. All numbers and texts are this model's: charts/measure.json, and the first words of
`python sample.py ... --temps 0.1 0.7 1.9`, verbatim, cut at the line end."""
import json
from html import escape
from pathlib import Path

HERE = Path(__file__).parent
d = json.loads((HERE / "measure.json").read_text())
H = {r["T"]: r["entropy_bits"] for r in d["sweep"]}
ROWS = [  # (degrees C, T, phase, colour, what it wrote at that T, from sample runs)
    (28, "1.9", "GAS", "#e8743b", ["The temperature of a body is therefore", "mechanically equivalent bin doch dann h", "higher Koluch doch geschmacklicken I..."]),
    (22, "0.7", "LIQUID", "#3f9e6b", ["The temperature of a body is therefore at", "the temperature of the body which is", "called a higher temperature."]),
    (19, "0.1", "SOLID", "#4a90d9", ["The temperature of a body is measured by", "the product of the mass of the body into", "the distance between the two points."]),
]
W, Ht = 1200, 628
s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {Ht}" width="{W}" height="{Ht}">',
     '<style>text{font-family:system-ui,-apple-system,Segoe UI,sans-serif}.m{font-family:ui-monospace,Menlo,monospace}</style>',
     f'<rect width="{W}" height="{Ht}" fill="#fbfaf7"/>',
     '<text x="40" y="58" font-size="34" font-weight="700">Maxwell and Boltzmann, on a chip, at room temperature</text>',
     '<text x="40" y="96" font-size="19" fill="#5d5f66">A tiny model of the two people who gave us this equation, sampled with it. The room sets T.</text>',
     # the equation
     '<text x="820" y="150" font-size="30" font-style="italic">p<tspan font-size="18" dy="8">i</tspan><tspan dy="-8"> = </tspan></text>',
     '<text x="905" y="135" font-size="24" font-style="italic">e<tspan font-size="15" dy="-12">z</tspan><tspan font-size="11" dy="5">i</tspan><tspan font-size="15" dy="-5"> / T</tspan></text>',
     '<line x1="900" y1="146" x2="1010" y2="146" stroke="#1b1b1f" stroke-width="2"/>',
     '<text x="902" y="174" font-size="24" font-style="italic">Σ e<tspan font-size="15" dy="-12">z</tspan><tspan font-size="11" dy="5">j</tspan><tspan font-size="15" dy="-5"> / T</tspan></text>',
     '<text x="1020" y="150" font-size="15" fill="#5d5f66">Boltzmann, 1868.</text>',
     '<text x="1020" y="170" font-size="15" fill="#5d5f66">Softmax, today.</text>']
# thermometer
tx, ty0, ty1 = 70, 190, 590
s += [f'<rect x="{tx-12}" y="{ty0}" width="24" height="{ty1-ty0}" rx="12" fill="#fff" stroke="#c9c9cf" stroke-width="2"/>',
      f'<circle cx="{tx}" cy="{ty1+8}" r="22" fill="#e8743b"/>',
      '<defs><linearGradient id="g" x1="0" y1="1" x2="0" y2="0"><stop offset="0" stop-color="#4a90d9"/><stop offset="0.5" stop-color="#3f9e6b"/><stop offset="1" stop-color="#e8743b"/></linearGradient></defs>',
      f'<rect x="{tx-6}" y="{ty0+10}" width="12" height="{ty1-ty0}" rx="6" fill="url(#g)"/>']
row_y = [205, 340, 475]
for (c, T, phase, col, lines), y in zip(ROWS, row_y):
    # tick on the thermometer
    s.append(f'<line x1="{tx+14}" y1="{y+55}" x2="{tx+40}" y2="{y+55}" stroke="{col}" stroke-width="3"/>')
    s.append(f'<text x="{tx+46}" y="{y+62}" font-size="20" font-weight="700" fill="{col}">{c} °C</text>')
    # the chip's screen
    sx = 190
    s.append(f'<rect x="{sx}" y="{y}" width="430" height="112" rx="10" fill="#0d0f12"/>')
    s.append(f'<text x="{sx+16}" y="{y+26}" class="m" font-size="15" fill="#7fd3ff">{c}.0C T={T} {phase}</text>')
    for k, line in enumerate(lines):
        s.append(f'<text x="{sx+16}" y="{y+52+k*21}" class="m" font-size="15" fill="#f1f1f1">{escape(line)}</text>')
    # its softmax: top 8 bars
    bx, bw, bh = 650, 26, 96
    s.append(f'<line x1="{bx-4}" y1="{y+108}" x2="{bx+8*(bw+6)}" y2="{y+108}" stroke="#c9c9cf" stroke-width="1.5"/>')
    for i, (w, p) in enumerate(d["dist"][T][:8]):
        h = max(2, p * bh)
        s.append(f'<rect x="{bx+i*(bw+6)}" y="{y+108-h}" width="{bw}" height="{h}" rx="4" fill="{col}"/>')
    s.append(f'<text x="{bx}" y="{y-6}" font-size="13" fill="#5d5f66">next word, T = {T}</text>')
    # entropy out
    s.append(f'<text x="930" y="{y+50}" font-size="38" font-weight="700" fill="{col}">{H[float(T)]:.1f} bits</text>')
    s.append(f'<text x="930" y="{y+76}" font-size="15" fill="#5d5f66">entropy of each word ({phase.lower()})</text>')
s.append(f'<text x="930" y="{row_y[0]-6}" font-size="13" fill="#5d5f66">temperature in, entropy out</text>')
s.append('</svg>')
(HERE / "hero.svg").write_text("\n".join(s))
print("wrote charts/hero.svg")
