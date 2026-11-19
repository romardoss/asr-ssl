"""Рисунок 1 для тез: WER і CER моделей на тестових наборах (групова стовпчикова діаграма).

    env/bin/python docs/figures/make_figure1.py

Читає evaluation/results.csv, пише docs/figures/figure1.png (300 dpi).
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_name("figure1.png")

# Times New Roman з Windows (WSL); якщо немає — STIX (схожий serif).
for name in ("times.ttf", "timesbd.ttf"):
    path = Path("/mnt/c/Windows/Fonts") / name
    if path.is_file():
        font_manager.fontManager.addfont(str(path))
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "font.size": 9,
    "mathtext.fontset": "stix",
})

TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3de"

# (run у results.csv, підпис, колір, штрих: результат-орієнтир, не порівнюваний напряму)
MODELS = [
    ("baseline/original-whisper-medium", "Whisper Medium", "#8c8a84", False),
    ("baseline/common-voice-only-baseline", "Вчитель Т0", "#3d3c39", False),
    ("student_mix-m345-cv-m5-cv-s1-pseudo-m345-common_voice-epoch00-01",
     "Студент, ПМ 3–5", "#2a78d6", False),
    ("student_mix-m345-cv-m5-cv-s2-pseudo-m5-common_voice-epoch00-01",
     "Студент, ПМ 5", "#eb6834", False),
    ("whisper-m-fixed-leaks-5-epochs-epoch04-01", "CV-only, 5 епох (орієнтир)", "#1baf7a", True),
]
Y_STEP = {"wer": 20, "cer": 10}  # крок поділок; верхня межа — максимум, округлений угору до кроку
SETS = [
    ("common_voice", "CV27"),
    ("fleurs", "FLEURS"),
    ("ukr_dialects", "Діалекти"),
    ("combined_no_cv10", "Разом"),
]

df = pd.read_csv(ROOT / "evaluation" / "results.csv").set_index("run")
# «Разом» без cv10_test: docs/figures/combined_without_cv10.py
combined = pd.read_csv(Path(__file__).with_name("combined_without_cv10.csv")).set_index("run")
df = df.join(combined[["wer_combined_no_cv10", "cer_combined_no_cv10"]])

fig, axes = plt.subplots(1, 2, figsize=(17 / 2.54, 7.2 / 2.54), sharey=False)
width = 0.16
offsets = [(i - (len(MODELS) - 1) / 2) * width for i in range(len(MODELS))]
plt.rcParams["hatch.linewidth"] = 0.7

for ax, metric in ((axes[0], "wer"), (axes[1], "cer")):
    values = [df.loc[run, f"{metric}_{key}"] for run, *_ in MODELS for key, _ in SETS]
    low = 0.0
    high = math.ceil(max(values) / Y_STEP[metric]) * Y_STEP[metric]
    for (run, _, color, hatched), dx in zip(MODELS, offsets):
        row = df.loc[run]
        for x, (key, _) in enumerate(SETS):
            # вісь Y не від нуля, тож стовпчик росте від нижньої межі осі
            ax.bar(x + dx, row[f"{metric}_{key}"] - low, bottom=low, width=width, color=color,
                   edgecolor="white", linewidth=0.5, hatch="////" if hatched else None, zorder=3)
    ax.set_ylim(low, high)
    ax.set_yticks(range(0, high + 1, Y_STEP[metric]))
    ax.set_ylabel(f"{metric.upper()}, %", color=TEXT)
    ax.set_xticks(range(len(SETS)))
    ax.set_xticklabels([label for _, label in SETS], color=TEXT)
    ax.set_xlim(-0.5, len(SETS) - 0.5)
    ax.axvline(len(SETS) - 1.5, color=MUTED, lw=0.6, ls=(0, (3, 3)))
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.tick_params(colors=MUTED, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.text(0.0, 1.03, f"({'а' if metric == 'wer' else 'б'}) {metric.upper()}",
            transform=ax.transAxes, ha="left", va="bottom", color=TEXT)

handles = [Patch(facecolor=c, edgecolor="white", hatch="////" if h else None, label=l)
           for _, l, c, h in MODELS]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, -0.01), handlelength=1.2, handletextpad=0.4, columnspacing=1.4)
fig.tight_layout(rect=(0, 0.17, 1, 1), w_pad=2.0)
fig.savefig(OUT, dpi=300, facecolor="white")
print(OUT)
