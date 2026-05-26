"""
Visualize the RedScar dataset.

Usage:
    python new_datasets/redscar/visualize.py
    python new_datasets/redscar/visualize.py --data data/redscar
    python new_datasets/redscar/visualize.py --data data/redscar --samples 8
"""

import argparse
import re
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

DEFAULT_DATA = Path(__file__).parents[2] / "data" / "redscar"
IMAGES_DIR = "IMAGES"
WOUND_MASK_DIR = "GT_WOUND_MASK"
WOUND_COLOR_DIR = "GT_WOUND_COLORMASK"
STAPLES_COLOR_DIR = "GT_STAPLES_COLORMASK"

FILENAME_RE = re.compile(
    r"(?P<id>.+?)_infection=(?P<infection>[01])_capture=(?P<capture>\d+)_resolution=(?P<resolution>[01])\.png"
)


def parse_metadata(path: Path) -> Optional[dict]:
    m = FILENAME_RE.match(path.name)
    if not m:
        return None
    return {
        "path": path,
        "id": m.group("id"),
        "infection": int(m.group("infection")),
        "capture": int(m.group("capture")),
        "resolution": int(m.group("resolution")),
    }


def load_images(data_dir: Path) -> list[dict]:
    images_dir = data_dir / IMAGES_DIR
    if not images_dir.exists():
        raise FileNotFoundError(f"IMAGES folder not found at {images_dir}. Run download.py first.")
    records = []
    for p in sorted(images_dir.glob("*.png")):
        meta = parse_metadata(p)
        if meta:
            records.append(meta)
    return records


# ── Plot 1: class distribution ────────────────────────────────────────────────

def plot_class_distribution(records: list[dict], out_dir: Path) -> None:
    infections = [r["infection"] for r in records]
    resolutions = [r["resolution"] for r in records]

    groups = {
        ("No infection", "Low-res"):  sum(1 for i, r in zip(infections, resolutions) if i == 0 and r == 0),
        ("No infection", "Hi-res"):   sum(1 for i, r in zip(infections, resolutions) if i == 0 and r == 1),
        ("Infection",    "Low-res"):  sum(1 for i, r in zip(infections, resolutions) if i == 1 and r == 0),
        ("Infection",    "Hi-res"):   sum(1 for i, r in zip(infections, resolutions) if i == 1 and r == 1),
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("RedScar Dataset — Class Distribution", fontsize=14, fontweight="bold")

    # Bar chart by group
    labels = [f"{k[0]}\n{k[1]}" for k in groups]
    counts = list(groups.values())
    colors = ["#4c8be8", "#1a4fa0", "#e84c4c", "#a01a1a"]
    bars = axes[0].bar(labels, counts, color=colors, edgecolor="white", linewidth=0.8)
    axes[0].set_title("Images by Infection × Resolution")
    axes[0].set_ylabel("Number of images")
    for bar, count in zip(bars, counts):
        pct = count / len(records) * 100
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f"{count}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)

    # Pie: infection only
    inf_counts = [
        sum(1 for r in records if r["infection"] == 0),
        sum(1 for r in records if r["infection"] == 1),
    ]
    axes[1].pie(inf_counts, labels=["No infection", "Infection"],
                colors=["#4c8be8", "#e84c4c"], autopct="%1.1f%%",
                startangle=90, wedgeprops={"edgecolor": "white"})
    axes[1].set_title(f"Infection split  (n={len(records)})")

    plt.tight_layout()
    _save(fig, out_dir / "class_distribution.png")


# ── Plot 2: image size distribution ──────────────────────────────────────────

def plot_size_distribution(records: list[dict], out_dir: Path) -> None:
    widths, heights = [], []
    for r in records:
        with Image.open(r["path"]) as img:
            w, h = img.size
        widths.append(w)
        heights.append(h)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("RedScar Dataset — Image Dimensions", fontsize=14, fontweight="bold")

    axes[0].hist(widths, bins=30, color="#4c8be8", edgecolor="white")
    axes[0].set_title("Width distribution")
    axes[0].set_xlabel("Width (px)")
    axes[0].set_ylabel("Count")

    axes[1].hist(heights, bins=30, color="#e84c8b", edgecolor="white")
    axes[1].set_title("Height distribution")
    axes[1].set_xlabel("Height (px)")

    print(f"  Width:  min={min(widths)}, max={max(widths)}, mean={np.mean(widths):.0f}")
    print(f"  Height: min={min(heights)}, max={max(heights)}, mean={np.mean(heights):.0f}")

    plt.tight_layout()
    _save(fig, out_dir / "size_distribution.png")


# ── Plot 3: sample grid ───────────────────────────────────────────────────────

def plot_sample_grid(records: list[dict], n_samples: int, out_dir: Path) -> None:
    rng = np.random.default_rng(42)

    infected = [r for r in records if r["infection"] == 1]
    clean = [r for r in records if r["infection"] == 0]

    n_each = n_samples // 2
    chosen_infected = rng.choice(len(infected), size=min(n_each, len(infected)), replace=False)
    chosen_clean = rng.choice(len(clean), size=min(n_each, len(clean)), replace=False)
    selected = (
        [infected[i] for i in chosen_infected] +
        [clean[i] for i in chosen_clean]
    )
    rng.shuffle(selected)

    cols = 4
    rows = (len(selected) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    fig.suptitle("RedScar Dataset — Sample Images", fontsize=14, fontweight="bold")
    axes = np.array(axes).flatten()

    for ax, rec in zip(axes, selected):
        img = Image.open(rec["path"]).convert("RGB")
        ax.imshow(img)
        label = "INFECTED" if rec["infection"] else "CLEAN"
        res = "Hi-res" if rec["resolution"] else "Lo-res"
        color = "#e84c4c" if rec["infection"] else "#4c8be8"
        ax.set_title(f"{label}  |  {res}", color=color, fontsize=9, fontweight="bold")
        ax.axis("off")

    for ax in axes[len(selected):]:
        ax.axis("off")

    plt.tight_layout()
    _save(fig, out_dir / "sample_grid.png")


# ── Plot 4: image triplets (original + masks) ─────────────────────────────────

def plot_triplets(records: list[dict], data_dir: Path, n: int, out_dir: Path) -> None:
    wound_mask_dir = data_dir / WOUND_MASK_DIR
    wound_color_dir = data_dir / WOUND_COLOR_DIR

    available = [
        r for r in records
        if (wound_mask_dir / r["path"].name).exists()
        and (wound_color_dir / r["path"].name).exists()
    ]
    if not available:
        print("  No mask directories found — skipping triplet plot.")
        return

    rng = np.random.default_rng(0)
    chosen = [available[i] for i in rng.choice(len(available), size=min(n, len(available)), replace=False)]

    cols = 3
    rows = len(chosen)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
    if rows == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Original", "Wound mask (binary)", "Wound mask (color)"]
    dirs = [data_dir / IMAGES_DIR, wound_mask_dir, wound_color_dir]

    for row_idx, rec in enumerate(chosen):
        label = "INFECTED" if rec["infection"] else "CLEAN"
        for col_idx, (title, folder) in enumerate(zip(col_titles, dirs)):
            ax = axes[row_idx, col_idx]
            img_path = folder / rec["path"].name
            if img_path.exists():
                img = Image.open(img_path).convert("RGB")
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
            if row_idx == 0:
                ax.set_title(title, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(label, color="#e84c4c" if rec["infection"] else "#4c8be8",
                              fontweight="bold", rotation=0, labelpad=50)
            ax.axis("off")

    fig.suptitle("RedScar Dataset — Original + Wound Masks", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, out_dir / "image_triplets.png")


# ── Plot 5: capture phase distribution ───────────────────────────────────────

def plot_capture_distribution(records: list[dict], out_dir: Path) -> None:
    phases = sorted(set(r["capture"] for r in records))
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle("RedScar Dataset — Capture Phase Distribution", fontsize=14, fontweight="bold")

    x = np.arange(len(phases))
    width = 0.35
    inf_counts = [sum(1 for r in records if r["capture"] == p and r["infection"] == 1) for p in phases]
    clean_counts = [sum(1 for r in records if r["capture"] == p and r["infection"] == 0) for p in phases]

    ax.bar(x - width / 2, clean_counts, width, label="No infection", color="#4c8be8")
    ax.bar(x + width / 2, inf_counts, width, label="Infection", color="#e84c4c")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Phase {p}" for p in phases])
    ax.set_ylabel("Number of images")
    ax.set_title("Images per clinical capture phase")
    ax.legend()

    plt.tight_layout()
    _save(fig, out_dir / "capture_distribution.png")


# ── helpers ───────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize the RedScar wound dataset")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="Path to dataset root")
    parser.add_argument("--samples", type=int, default=8, help="Images to show in sample grid")
    parser.add_argument("--triplets", type=int, default=4, help="Triplet rows (original + masks)")
    parser.add_argument("--out", default=None, help="Output dir for saved plots (default: --data/plots)")
    args = parser.parse_args()

    data_dir = Path(args.data)
    out_dir = Path(args.out) if args.out else data_dir / "plots"

    print(f"Loading metadata from {data_dir} ...")
    records = load_images(data_dir)
    print(f"Found {len(records)} images.")

    print("\nPlot 1/5: class distribution")
    plot_class_distribution(records, out_dir)

    print("Plot 2/5: image size distribution")
    plot_size_distribution(records, out_dir)

    print("Plot 3/5: sample grid")
    plot_sample_grid(records, args.samples, out_dir)

    print("Plot 4/5: original + mask triplets")
    plot_triplets(records, data_dir, args.triplets, out_dir)

    print("Plot 5/5: capture phase distribution")
    plot_capture_distribution(records, out_dir)

    print(f"\nAll plots saved to {out_dir}")


if __name__ == "__main__":
    main()
