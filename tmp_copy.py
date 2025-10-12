#!/usr/bin/env python3
"""
Flatten-copy files from a source tree into a single destination folder.

Inputs:
  1) source_path: path to the root folder to scan (recursively)
  2) dest_path:   path to the folder where all files will be copied (no subfolders created)
  3) prefix:      string to prefix each output filename

Destination filename format:
  prefix_original_subfolder_name_original_name

- "original_subfolder_name" is the relative directory path (from source_path) of
  the file's containing folder, with path separators replaced by underscores.
  For files that live directly in source_path (no subfolder), "root" is used.

Examples:
  file: /src/a/b/report.csv         -> prefix_a_b_report.csv
  file: /src/readme.txt             -> prefix_root_readme.txt

Collisions:
  If a destination filename already exists, a numeric suffix is added before the
  extension: filename.ext -> filename (1).ext, filename (2).ext, etc.

Usage:
  python flatten_copy.py --source_path /path/to/source --dest_path /path/to/dest --prefix myprefix
  python flatten_copy.py  # Uses default values
"""

import argparse
import os
import re
import shutil
from pathlib import Path
from typing import Tuple

def sanitize_component(name: str) -> str:
    """
    Make a safe path component: replace path separators and trim whitespace.
    Keeps most characters but collapses consecutive whitespace to a single underscore.
    """
    # Replace os-specific separators with underscores (safety if name came from relpath)
    name = name.replace(os.sep, "_")
    # Normalize whitespace
    name = re.sub(r"\s+", "_", name.strip())
    # Avoid empty component
    return name or "root"

def build_flat_name(prefix: str, rel_dir: str, original_name: str) -> str:
    """
    Build destination filename 'prefix_original_subfolder_name_original_name'.
    rel_dir is the *relative* directory of the file's parent to source_path.
    """
    sub = "root" if rel_dir in ("", ".", None) else sanitize_component(rel_dir)
    # Avoid duplicate underscores if prefix or sub already contain underscores
    parts = [p for p in [prefix, sub, original_name] if p != ""]
    return "_".join(parts)

def with_unique_suffix(dest_dir: Path, filename: str) -> Path:
    """
    If dest_dir/filename exists, add ' (n)' before the extension until unique.
    """
    candidate = dest_dir / filename
    if not candidate.exists():
        return candidate

    stem, ext = os.path.splitext(filename)
    n = 1
    while True:
        alt = dest_dir / f"{stem} ({n}){ext}"
        if not alt.exists():
            return alt
        n += 1

def copy_file(src_file: Path, dest_dir: Path, flat_name: str) -> Tuple[Path, Path]:
    """
    Copy src_file to dest_dir/flat_name (or unique variant) with metadata.
    Returns (src_file, dest_file).
    """
    dest_file = with_unique_suffix(dest_dir, flat_name)
    # Ensure parent directory exists (dest_dir is ensured in main, but keep safe)
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_file, dest_file)  # preserves metadata where possible
    return src_file, dest_file

def flatten_copy(source_path: Path, dest_path: Path, prefix: str) -> None:
    """
    Walk source_path recursively and copy every regular file into dest_path
    with flattened names.
    """
    if not source_path.exists() or not source_path.is_dir():
        raise NotADirectoryError(f"source_path does not exist or is not a directory: {source_path}")

    dest_path.mkdir(parents=True, exist_ok=True)

    total = 0
    copied = 0
    skipped = 0
    for root, dirs, files in os.walk(source_path, followlinks=False):
        # We only copy files; directories are traversed but not created in dest
        rel_dir = os.path.relpath(root, source_path)
        # Normalize rel_dir for the top level
        if rel_dir == ".":
            rel_dir = ""
        for fname in files:
            total += 1
            src = Path(root) / fname
            # Skip non-regular files (e.g., device files) and broken symlinks
            try:
                if not src.is_file():
                    skipped += 1
                    continue
            except OSError:
                # In case of permission errors or similar
                skipped += 1
                continue

            flat_name = build_flat_name(prefix, rel_dir, fname)
            try:
                copy_file(src, dest_path, flat_name)
                copied += 1
            except Exception as e:
                skipped += 1
                print(f"[WARN] Failed to copy '{src}': {e}")

    print(f"Done. Files scanned: {total}, copied: {copied}, skipped: {skipped}")
    print(f"Destination: {dest_path.resolve()}")

def main():
    for class_name in ['Good', 'Inflamed']:
        for prefix in ['Leg', 'Stereotomy']:
            source_path = Path(f"../Data/{prefix}/{class_name}")
            dest_path = Path(f"../Data/All/{class_name}/{prefix}")
            flatten_copy(source_path, dest_path, prefix)

if __name__ == "__main__":
    main()
