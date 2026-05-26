"""
Download the AZH wound classification dataset from GitHub.

Source: https://github.com/uwm-bigdata/wound-classification-using-images-and-locations
License: see repository
730 wound images across 4 types: Venous, Diabetic, Pressure, Surgical.

Usage:
    python new_datasets/azh/download.py
    python new_datasets/azh/download.py --output data/azh
"""

import argparse
import csv
import zipfile
from pathlib import Path

import requests

DEFAULT_OUTPUT = Path(__file__).parents[2] / "data" / "azh"

SPLITS = {
    "Train": "https://raw.githubusercontent.com/uwm-bigdata/wound-classification-using-images-and-locations/main/dataset/Train.zip",
    "Test":  "https://raw.githubusercontent.com/uwm-bigdata/wound-classification-using-images-and-locations/main/dataset/Test.zip",
}

METADATA_COLS = ["image_name", "split", "wound_type"]


def download_zip(url: str, dest: Path) -> None:
    print(f"  Downloading {dest.name} ...")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(f"    {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB ({downloaded/total*100:.0f}%)", end="\r")
    print()


def download(output_dir: Path) -> None:
    images_dir = output_dir / "IMAGES"
    images_dir.mkdir(parents=True, exist_ok=True)

    records = []

    for split, url in SPLITS.items():
        zip_path = output_dir / f"{split}.zip"
        download_zip(url, zip_path)

        print(f"  Extracting {split}.zip ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = zf.namelist()
            for member in members:
                info = zf.getinfo(member)
                if info.is_dir():
                    continue
                member_path = Path(member)
                # Expect structure: <anything>/<WoundType>/<image.jpg>
                # or flat <WoundType>/<image.jpg>
                parts = member_path.parts
                if len(parts) < 2:
                    continue
                wound_type = parts[-2]
                fname = parts[-1]
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                if fname.startswith("._"):  # macOS resource fork files
                    continue

                dest_name = f"{split}_{wound_type}_{fname}"
                dest_path = images_dir / dest_name
                dest_path.write_bytes(zf.read(member))
                records.append({
                    "image_name": dest_name,
                    "split":      split,
                    "wound_type": wound_type,
                })

        zip_path.unlink()
        print(f"  {split}: {sum(1 for r in records if r['split'] == split)} images extracted")

    csv_path = output_dir / "metadata.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_COLS)
        writer.writeheader()
        writer.writerows(records)

    print(f"\nDone.")
    print(f"  Images saved : {len(records)}  ->  {images_dir}")
    print(f"  Metadata CSV : {csv_path}")
    _print_summary(records)


def _print_summary(records: list) -> None:
    from collections import Counter
    print(f"\nDataset summary ({len(records)} images):")
    print("\n  By wound type:")
    for wt, n in Counter(r["wound_type"] for r in records).most_common():
        print(f"    {wt:<15} {n}")
    print("\n  By split:")
    for sp, n in Counter(r["split"] for r in records).most_common():
        print(f"    {sp:<10} {n}")


def main():
    parser = argparse.ArgumentParser(description="Download the AZH wound dataset")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help=f"Output directory (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()
    download(Path(args.output))


if __name__ == "__main__":
    main()
