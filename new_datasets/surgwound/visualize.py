# %% Imports & config
import csv
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

DATA_DIR = Path(__file__).parents[2] / "data" / "surgwound"
OUT_DIR  = DATA_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RISK_ORDER    = ["Low", "Medium", "High", "Uncertain"]
RISK_COLORS   = ["#4c8be8", "#f0a500", "#e84c4c", "#aaaaaa"]
HEAL_ORDER    = ["Healed", "Not Healed", "Uncertain"]
HEAL_COLORS   = ["#4c8be8", "#e84c4c", "#aaaaaa"]
URGENCY_ORDER = [
    "Home Care (Green): Manage with routine care",
    "Clinic Visit (Yellow): Requires professional evaluation within 48 hours",
    "Emergency Care (Red): Seek immediate medical attention",
    "Uncertain",
]
URGENCY_COLORS = ["#2ca02c", "#f0a500", "#e84c4c", "#aaaaaa"]
URGENCY_SHORT  = {
    "Home Care (Green): Manage with routine care":                             "Home Care\n(Green)",
    "Clinic Visit (Yellow): Requires professional evaluation within 48 hours": "Clinic Visit\n(Yellow)",
    "Emergency Care (Red): Seek immediate medical attention":                  "Emergency\n(Red)",
}


def image_path(name: str) -> Path:
    return DATA_DIR / "IMAGES" / Path(name).with_suffix(".png").name


def save(fig: plt.Figure, filename: str) -> None:
    path = OUT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path}")


# %% Load metadata
with open(DATA_DIR / "metadata.csv", encoding="utf-8") as f:
    records = list(csv.DictReader(f))

print(f"Loaded {len(records)} images")

# %% Plot 1 — Key label distributions
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("SurgWound Dataset — Key Label Distributions", fontsize=14, fontweight="bold")

triples = [
    ("healing_status", "Healing Status",  HEAL_ORDER,    HEAL_COLORS),
    ("infection_risk",  "Infection Risk", RISK_ORDER,    RISK_COLORS),
    ("urgency_level",   "Urgency Level",  URGENCY_ORDER, URGENCY_COLORS),
]

for ax, (col, title, order, colors) in zip(axes, triples):
    counts = Counter(r[col] or "Unknown" for r in records)
    full_labels = [v for v in order if v in counts] + [v for v in counts if v not in order]
    vals    = [counts[l] for l in full_labels]
    display = [URGENCY_SHORT.get(l, l) for l in full_labels] if col == "urgency_level" else full_labels
    bars = ax.bar(display, vals, color=colors[:len(full_labels)], edgecolor="white")
    ax.set_title(title)
    ax.set_ylabel("Images")
    ax.tick_params(axis="x", labelrotation=15)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(v), ha="center", va="bottom", fontsize=9)

plt.tight_layout()
save(fig, "label_distributions.png")

# %% Plot 2 — All attribute distributions
attrs = [
    ("location",       "Location"),
    ("closure_method", "Closure Method"),
    ("exudate_type",   "Exudate Type"),
    ("erythema",       "Erythema"),
    ("edema",          "Edema"),
]

fig, axes = plt.subplots(2, 3, figsize=(18, 9))
fig.suptitle("SurgWound Dataset — All Attribute Distributions", fontsize=14, fontweight="bold")
axes_flat = axes.flatten()

for ax, (col, title) in zip(axes_flat, attrs):
    counts = Counter(r[col] or "Unknown" for r in records)
    labels = sorted(counts, key=lambda x: -counts[x])
    vals   = [counts[l] for l in labels]
    ax.barh(labels[::-1], vals[::-1], color="#4c8be8", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel("Images")
    for i, v in enumerate(vals[::-1]):
        ax.text(v + 0.3, i, str(v), va="center", fontsize=8)

axes_flat[-1].axis("off")
plt.tight_layout()
save(fig, "all_attributes.png")

# %% Plot 3 — Image size distribution
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
fig.suptitle("SurgWound Dataset — Image Dimensions", fontsize=14, fontweight="bold")
axes[0].hist(widths,  bins=30, color="#4c8be8", edgecolor="white")
axes[0].set_title("Width distribution"); axes[0].set_xlabel("Width (px)"); axes[0].set_ylabel("Count")
axes[1].hist(heights, bins=30, color="#e84c8b", edgecolor="white")
axes[1].set_title("Height distribution"); axes[1].set_xlabel("Height (px)")

plt.tight_layout()
save(fig, "size_distribution.png")

# %% Plot 4 — Sample grid by infection risk
N_SAMPLES = 9

rng = np.random.default_rng(42)
risk_colors = {"Low": "#4c8be8", "Medium": "#f0a500", "High": "#e84c4c"}
n_each = max(1, N_SAMPLES // 3)

selected = []
for risk, color in risk_colors.items():
    group = [r for r in records if r["infection_risk"] == risk and image_path(r["image_name"]).exists()]
    idx = rng.choice(len(group), size=min(n_each, len(group)), replace=False)
    selected.extend((group[i], risk) for i in idx)

cols = 4
rows = (len(selected) + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
fig.suptitle("SurgWound Dataset — Sample Images by Infection Risk", fontsize=14, fontweight="bold")
axes = np.array(axes).flatten()

for ax, (rec, risk) in zip(axes, selected):
    ax.imshow(Image.open(image_path(rec["image_name"])).convert("RGB"))
    ax.set_title(f"Risk: {risk}\n{rec.get('healing_status', '')}",
                 color=risk_colors[risk], fontsize=8, fontweight="bold")
    ax.axis("off")

for ax in axes[len(selected):]:
    ax.axis("off")

plt.tight_layout()
save(fig, "sample_grid.png")

# %% Plot 5 — Healing status × infection risk cross-tab
heal_vals = [v for v in HEAL_ORDER if v != "Uncertain"]
risk_vals  = [v for v in RISK_ORDER  if v != "Uncertain"]

matrix = np.zeros((len(heal_vals), len(risk_vals)), dtype=int)
for r in records:
    h, ri = r.get("healing_status", ""), r.get("infection_risk", "")
    if h in heal_vals and ri in risk_vals:
        matrix[heal_vals.index(h), risk_vals.index(ri)] += 1

fig, ax = plt.subplots(figsize=(8, 4))
fig.suptitle("SurgWound — Healing Status x Infection Risk", fontsize=14, fontweight="bold")
im = ax.imshow(matrix, cmap="Blues")
ax.set_xticks(range(len(risk_vals)));  ax.set_xticklabels(risk_vals)
ax.set_yticks(range(len(heal_vals))); ax.set_yticklabels(heal_vals)
ax.set_xlabel("Infection Risk"); ax.set_ylabel("Healing Status")
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=12,
                color="white" if matrix[i, j] > matrix.max() * 0.5 else "black")
plt.colorbar(im, ax=ax, label="Images")
plt.tight_layout()
save(fig, "cross_tab.png")
