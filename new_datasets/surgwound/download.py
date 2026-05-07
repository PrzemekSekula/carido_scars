"""
Download the SurgWound dataset from HuggingFace.

Source: https://huggingface.co/datasets/xuxuxuxuxu/SurgWound
License: CC-BY-SA-4.0
697 surgical wound images annotated for infection risk, healing status, and more.

Usage:
    python new_datasets/surgwound/download.py
    python new_datasets/surgwound/download.py --output data/surgwound
"""

import argparse
import base64
import csv
import io
import json
from collections import defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download
from PIL import Image

DEFAULT_OUTPUT = Path(__file__).parents[2] / "data" / "surgwound"

REPO_ID = "xuxuxuxuxu/SurgWound"
# Only question files carry per-attribute labels (multi_choice answers)
JSON_FILES = [
    "train_question.json",
    "val_question.json",
    "test_question.json",
    "train_report.json",
    "val_report.json",
    "test_report.json",
]

FIELD_TO_COL = {
    "Healing Status":            "healing_status",
    "Infection Risk Assessment": "infection_risk",
    "Urgency Level":             "urgency_level",
    "Location":                  "location",
    "Closure Method":            "closure_method",
    "Exudate Type":              "exudate_type",
    "Erythema":                  "erythema",
    "Edema":                     "edema",
}

METADATA_COLS = [
    "image_name",
    "healing_status",
    "infection_risk",
    "urgency_level",
    "location",
    "closure_method",
    "exudate_type",
    "erythema",
    "edema",
]


def decode_image(raw) -> Image.Image:
    if isinstance(raw, Image.Image):
        return raw.convert("RGB")
    if isinstance(raw, bytes):
        return Image.open(io.BytesIO(raw)).convert("RGB")
    if isinstance(raw, str):
        return Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGB")
    raise ValueError(f"Unknown image type: {type(raw)}")


def download(output_dir: Path) -> None:
    images_dir = output_dir / "IMAGES"
    images_dir.mkdir(parents=True, exist_ok=True)

    meta: dict[str, dict] = defaultdict(lambda: {c: "" for c in METADATA_COLS})
    saved_images: set[str] = set()

    for filename in JSON_FILES:
        print(f"Downloading {filename} ...")
        local_path = hf_hub_download(repo_id=REPO_ID, filename=filename, repo_type="dataset")
        with open(local_path, encoding="utf-8") as f:
            rows = json.load(f)

        print(f"  {len(rows)} rows")
        for row in rows:
            name = row["image_name"]
            meta[name]["image_name"] = name

            col = FIELD_TO_COL.get(row.get("field", ""))
            if col and row.get("task_type") == "multi_choice":
                meta[name][col] = row.get("answer", "")

            if name not in saved_images:
                try:
                    img = decode_image(row["image"])
                    stem = Path(name).stem
                    out_path = images_dir / f"{stem}.png"
                    img.save(out_path, "PNG")
                    saved_images.add(name)
                except Exception as e:
                    print(f"  Warning: could not save {name}: {e}")

        print(f"  Unique images so far: {len(saved_images)}")

    csv_path = output_dir / "metadata.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_COLS)
        writer.writeheader()
        for name in sorted(meta):
            writer.writerow(meta[name])

    print(f"\nDone.")
    print(f"  Images saved : {len(saved_images)}  ->  {images_dir}")
    print(f"  Metadata CSV : {csv_path}")
    _print_summary(meta)


def _print_summary(meta: dict) -> None:
    records = list(meta.values())
    print(f"\nDataset summary ({len(records)} images):")
    for col, label in [
        ("healing_status", "Healing status"),
        ("infection_risk",  "Infection risk"),
        ("urgency_level",   "Urgency level"),
    ]:
        counts: dict[str, int] = defaultdict(int)
        for r in records:
            counts[r[col] or "Unknown"] += 1
        print(f"\n  {label}:")
        for val, n in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"    {val:<30} {n}")


def main():
    parser = argparse.ArgumentParser(description="Download the SurgWound dataset")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help=f"Output directory (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()
    download(Path(args.output))


if __name__ == "__main__":
    main()
