"""Draw the README's charts from charts/measure.json (measure.py). Small, friendly, honest."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
d = json.loads((HERE / "measure.json").read_text())
COLD, ROOM, HOT, INK, MUTE, GRID = "#4a90d9", "#3f9e6b", "#e8743b", "#1b1b1f", "#6b6d75", "#e6e6ea"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK,
                     "xtick.color": MUTE, "ytick.color": MUTE, "axes.spines.top": False, "axes.spines.right": False})

# 1. the same softmax at three temperatures
fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
for ax, (T, col, label) in zip(axes, [("0.1", COLD, "T = 0.1  (3 °C colder)"), ("0.7", ROOM, "T = 0.7  (the room)"),
                                      ("1.9", HOT, "T = 1.9  (6 °C warmer: a hand)")]):
    rows = d["dist"][T][:8]
    names = [w.replace("<|endoftext|>", "(end)").strip() or "␣" for w, _ in rows]
    ax.bar(range(len(rows)), [p for _, p in rows], color=col, width=0.72, edgecolor="white", linewidth=2)
    ax.set_xticks(range(len(rows)), names, rotation=40, ha="right")
    ax.set_title(label, color=INK, fontsize=11.5)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
axes[0].set_ylabel("probability of the next word")
fig.suptitle('"The temperature of a body is ..."   the same softmax, three rooms', color=INK, fontsize=13, y=1.02)
fig.savefig(HERE / "softmax.svg", bbox_inches="tight"); plt.close(fig)

# 2. temperature in, entropy out
sw = d["sweep"]
T = [r["T"] for r in sw]; H = [r["entropy_bits"] for r in sw]
fig, ax = plt.subplots(figsize=(8, 4))
for lo, hi, name, col in [(0, 0.35, "SOLID\nstiff", COLD), (0.35, 1.1, "LIQUID\nMaxwell talks", ROOM), (1.1, 2.05, "GAS\ntwo languages mix", HOT)]:
    ax.axvspan(lo, hi, color=col, alpha=0.10, linewidth=0)
    ax.text((lo + hi) / 2, 10.2, name, ha="center", va="top", color=col, fontsize=10.5, fontweight="bold")
ax.plot(T, H, color=INK, linewidth=2)
ax.scatter(T, H, s=34, color=INK, edgecolor="white", linewidth=1.5, zorder=3)
Dm = [r["demon_bits"] for r in sw]
ax.plot(T, Dm, color="#7d3c98", linewidth=2, linestyle=(0, (5, 3)))
ax.text(1.42, Dm[-1] + 0.45, "with Maxwell's demon", color="#7d3c98", fontsize=10.5, fontweight="bold")
ax.text(1.42, H[14] + 1.5, "without", color=INK, fontsize=10.5, fontweight="bold", ha="right")
ax.axhline(11, color=MUTE, linewidth=1, linestyle=(0, (4, 4)))
ax.text(0.02, 10.75, "11 bits: every word equally likely (2048 of them)", color=MUTE, fontsize=9, va="top")
ax.set_xlabel("sampling temperature T"); ax.set_ylabel("entropy of each word (bits)")
ax.set_xlim(0, 2.05); ax.set_ylim(0, 11.5)
top = ax.secondary_xaxis("top", functions=(lambda t: 22 + (t - 0.7) / 0.2, lambda c: 0.7 + 0.2 * (c - 22)))
top.set_xlabel("room temperature on the chip (°C), if it woke up at 22", color=MUTE); top.tick_params(colors=MUTE)
ax.grid(color=GRID, linewidth=0.8); ax.set_axisbelow(True)
ax.set_title("Temperature in, entropy out", color=INK, fontsize=13, loc="left", pad=34)
fig.savefig(HERE / "entropy.svg", bbox_inches="tight"); plt.close(fig)

# 3. two gases mixing
G = [100 * r["german_share"] for r in sw]
fig, ax = plt.subplots(figsize=(8, 3.4))
ax.axvspan(1.1, 2.05, color=HOT, alpha=0.10, linewidth=0)
ax.plot(T, G, color=HOT, linewidth=1.5, alpha=0.5)
ax.scatter(T, G, s=34, color=HOT, edgecolor="white", linewidth=1.5, zorder=3)
ax.set_xlabel("sampling temperature T"); ax.set_ylabel("German in Maxwell (%)")
ax.set_xlim(0, 2.05); ax.set_ylim(0, 30)
ax.grid(color=GRID, linewidth=0.8); ax.set_axisbelow(True)
ax.set_title("Two gases mixing: Boltzmann's German leaking into Maxwell", color=INK, fontsize=13, loc="left")
ax.text(0.02, 28, "share of German among the small words (the, der, und...) in paragraphs that start in English;\n"
        "48 paragraphs per point", color=MUTE, fontsize=9, va="top")
fig.savefig(HERE / "mixing.svg", bbox_inches="tight"); plt.close(fig)
print("drew softmax.svg, entropy.svg, mixing.svg")
