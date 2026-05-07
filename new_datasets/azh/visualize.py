# %% Imports & config
import csv
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

DATA_DIR = Path(__file__).parents[2] / "data" / "azh"
OUT_DIR  = DATA_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_MAP = {
    "V":  "Venous",
    "D":  "Diabetic",
    "S":  "Surgical",
    "P":  "Pressure",
    "BG": "Background",
    "N":  "Normal",
}
CLASS_ORDER  = ["V", "D", "S", "P", "N", "BG"]
CLASS_COLORS = ["#4c8be8", "#e84c4c", "#2ca02c", "#f0a500", "#9467bd", "#aaaaaa"]
SPLIT_COLORS = {"Train": "#4c8be8", "Test": "#f0a500"}


def image_path(name: str) -> Path:
    return DATA_DIR / "IMAGES" / name


def save(fig: plt.Figure, filename: str) -> None:
    path = OUT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path}")


# %% Load metadata
with open(DATA_DIR / "metadata.csv", encoding="utf-8") as f:
    records = list(csv.DictReader(f))

print(f"Loaded {len(records)} images")
print("Wound types:", dict(Counter(r["wound_type"] for r in records).most_common()))
print("Splits:", dict(Counter(r["split"] for r in records).most_common()))

# %% Plot 1 — Class distribution (wound type × split)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("AZH Dataset — Class Distribution", fontsize=14, fontweight="bold")

# Overall counts
labels  = [c for c in CLASS_ORDER if c in Counter(r["wound_type"] for r in records)]
counts  = [sum(1 for r in records if r["wound_type"] == c) for c in labels]
display = [CLASS_MAP.get(c, c) for c in labels]
bars = axes[0].bar(display, counts, color=CLASS_COLORS[:len(labels)], edgecolor="white")
axes[0].set_title("Images per wound type")
axes[0].set_ylabel("Images")
for bar, v in zip(bars, counts):
    pct = v / len(records) * 100
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{v}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)

# Train vs Test per class
x      = np.arange(len(labels))
width  = 0.35
train_counts = [sum(1 for r in records if r["wound_type"] == c and r["split"] == "Train") for c in labels]
test_counts  = [sum(1 for r in records if r["wound_type"] == c and r["split"] == "Test")  for c in labels]
axes[1].bar(x - width/2, train_counts, width, label="Train", color=SPLIT_COLORS["Train"], edgecolor="white")
axes[1].bar(x + width/2, test_counts,  width, label="Test",  color=SPLIT_COLORS["Test"],  edgecolor="white")
axes[1].set_title("Train / Test split per class")
axes[1].set_ylabel("Images")
axes[1].set_xticks(x)
axes[1].set_xticklabels(display)
axes[1].legend()

plt.tight_layout()
save(fig, "class_distribution.png")

# %% Plot 2 — Image size distribution
widths, heights = [], []
for r in records:
    p = image_path(r["image_name"])
    if p.exists():
        with Image.open(p) as img:
            w, h = img.size
        widths.append(w)
        heights.append(h)

print(f"Width:  min={min(widths)}, max={max(widths)}, mean={np.mean(widths):.0f}")
print(f"Height: min={min(heights)}, max={max(heights)}, mean={np.mean(heights):.0f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("AZH Dataset — Image Dimensions", fontsize=14, fontweight="bold")
axes[0].hist(widths,  bins=25, color="#4c8be8", edgecolor="white")
axes[0].set_title("Width distribution"); axes[0].set_xlabel("Width (px)"); axes[0].set_ylabel("Count")
axes[1].hist(heights, bins=25, color="#e84c8b", edgecolor="white")
axes[1].set_title("Height distribution"); axes[1].set_xlabel("Height (px)")

plt.tight_layout()
save(fig, "size_distribution.png")

# %% Plot 3 — Sample grid (one row per wound type)
N_PER_CLASS = 5
rng = np.random.default_rng(42)

rows_to_show = [c for c in CLASS_ORDER if any(r["wound_type"] == c for r in records)]

# Extra left column for row labels
n_cols = N_PER_CLASS + 1
fig, axes = plt.subplots(len(rows_to_show), n_cols,
                         figsize=(n_cols * 2.8, len(rows_to_show) * 2.8),
                         gridspec_kw={"width_ratios": [0.4] + [1] * N_PER_CLASS})
fig.suptitle("AZH Dataset — Sample Images per Wound Type", fontsize=14, fontweight="bold")

for row_idx, code in enumerate(rows_to_show):
    group = [r for r in records if r["wound_type"] == code and image_path(r["image_name"]).exists()]
    chosen = [group[i] for i in rng.choice(len(group), size=min(N_PER_CLASS, len(group)), replace=False)]
    color = CLASS_COLORS[CLASS_ORDER.index(code)]

    # Label cell
    ax_label = axes[row_idx, 0]
    ax_label.axis("off")
    ax_label.text(0.5, 0.5, CLASS_MAP.get(code, code),
                  ha="center", va="center", fontsize=12, fontweight="bold",
                  color=color, transform=ax_label.transAxes, wrap=True)

    # Image cells
    for col_idx in range(N_PER_CLASS):
        ax = axes[row_idx, col_idx + 1]
        if col_idx < len(chosen):
            ax.imshow(Image.open(image_path(chosen[col_idx]["image_name"])).convert("RGB"))
        else:
            ax.set_facecolor("#f0f0f0")
        ax.axis("off")

plt.tight_layout()
save(fig, "sample_grid.png")

# %% Plot 4 — Aspect ratio scatter by wound type
fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle("AZH Dataset — Aspect Ratio by Wound Type", fontsize=14, fontweight="bold")

for code, color in zip(CLASS_ORDER, CLASS_COLORS):
    group = [r for r in records if r["wound_type"] == code]
    ws, hs = [], []
    for r in group:
        p = image_path(r["image_name"])
        if p.exists():
            with Image.open(p) as img:
                ws.append(img.size[0])
                hs.append(img.size[1])
    if ws:
        ax.scatter(ws, hs, label=CLASS_MAP.get(code, code), color=color, alpha=0.6, s=30)

ax.set_xlabel("Width (px)")
ax.set_ylabel("Height (px)")
ax.set_title("Width vs Height per class")
ax.legend()
plt.tight_layout()
save(fig, "aspect_ratio.png")

# %% Plot 5 — Train/test split pie per class
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("AZH Dataset — Train/Test Split per Wound Type", fontsize=14, fontweight="bold")
axes_flat = axes.flatten()

for ax, code in zip(axes_flat, CLASS_ORDER):
    group = [r for r in records if r["wound_type"] == code]
    n_train = sum(1 for r in group if r["split"] == "Train")
    n_test  = sum(1 for r in group if r["split"] == "Test")
    ax.pie([n_train, n_test], labels=["Train", "Test"],
           colors=[SPLIT_COLORS["Train"], SPLIT_COLORS["Test"]],
           autopct="%1.0f%%", startangle=90,
           wedgeprops={"edgecolor": "white"})
    ax.set_title(f"{CLASS_MAP.get(code, code)} (n={len(group)})", fontweight="bold")

plt.tight_layout()
save(fig, "split_per_class.png")
