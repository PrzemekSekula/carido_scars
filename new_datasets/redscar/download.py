"""
Download the RedScar dataset from https://redscar.uib.es

Access requires a download code obtained by emailing redscar@uib.es
with subject: "Request Redscar© database access"

Usage:
    python new_datasets/redscar/download.py --code YOUR_CODE
    python new_datasets/redscar/download.py --code YOUR_CODE --output data/redscar
"""

import argparse
import sys
import zipfile
from pathlib import Path
from typing import Optional

import requests

BASE_URL = "https://redscar.uib.es"
CHECK_URL = f"{BASE_URL}/check_database_download/"
DOWNLOAD_URL = f"{BASE_URL}/download_redscar_database"

DEFAULT_OUTPUT = Path(__file__).parents[2] / "data" / "redscar"


def check_code(code: str, session: requests.Session) -> dict:
    resp = session.get(CHECK_URL, params={"downloadCode": code}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def download_dataset(code: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / "redscar.zip"

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0", "Referer": f"{BASE_URL}/dataset.html"})

    print(f"Validating code '{code}'...")
    result = check_code(code, session)
    status = result.get("STATUS")

    if status == "MAXIMUM_DOWNLOADS_REACHED":
        sys.exit("Error: this code has already reached its download limit. Request a new code.")
    elif status == "DISABLED_DOWNLOAD_CODE":
        sys.exit("Error: this download code is disabled.")
    elif status != "DOWNLOAD_AVAILABLE":
        sys.exit(f"Error: unexpected status '{status}'. Check your code and try again.")

    data = result.get("DATA", {})
    limit = data.get("DOWNLOADS_LIMIT")
    used = data.get("ACTIVE_DOWNLOADS", 0)
    remaining_str = str(limit - used - 1) if isinstance(limit, int) else "?"
    print(f"Code valid. Downloads remaining after this one: {remaining_str}")

    print(f"Downloading dataset to {zip_path} ...")
    with session.get(DOWNLOAD_URL, params={"download_code": code}, stream=True, timeout=300) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"  {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB  ({pct:.0f}%)", end="\r")
    print(f"\nDownload complete: {zip_path}")

    print(f"Extracting to {output_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)
    print("Extraction complete.")

    zip_path.unlink()
    print(f"Done. Dataset at: {output_dir}")
    _print_summary(output_dir)


def _print_summary(output_dir: Path) -> None:
    folders = sorted(p for p in output_dir.iterdir() if p.is_dir())
    print("\nDataset structure:")
    for folder in folders:
        n = len(list(folder.glob("*.png")))
        print(f"  {folder.name}/  ({n} images)")

    images_dir = output_dir / "IMAGES"
    if images_dir.exists():
        imgs = list(images_dir.glob("*.png"))
        infected = sum(1 for p in imgs if "_infection=1_" in p.name)
        clean = len(imgs) - infected
        hi_res = sum(1 for p in imgs if "_resolution=1_" in p.name)
        print(f"\nTotal images:  {len(imgs)}")
        print(f"  Infected:    {infected}")
        print(f"  Clean:       {clean}")
        print(f"  Hi-res:      {hi_res}")


def main():
    parser = argparse.ArgumentParser(description="Download the RedScar wound dataset")
    parser.add_argument(
        "--code", required=True,
        help="Download code received from redscar@uib.es"
    )
    parser.add_argument(
        "--output", default=str(DEFAULT_OUTPUT),
        help=f"Output directory (default: {DEFAULT_OUTPUT})"
    )
    args = parser.parse_args()
    download_dataset(args.code, Path(args.output))


if __name__ == "__main__":
    main()
