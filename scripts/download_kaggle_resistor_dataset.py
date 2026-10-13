"""Download and inspect the public Kaggle resistor-segmentation dataset.

Kaggle API credentials must be configured by the user. This script does not
embed or request passwords/tokens in source code.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DATASET_SLUG = "alptugaydin/resistor-images-dataset-for-segmentation"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
ANNOTATION_SUFFIXES = {".json", ".xml", ".txt", ".csv", ".mat", ".npy", ".npz"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the Kaggle resistor images dataset and summarize its annotation files."
    )
    parser.add_argument("--dataset", default=DATASET_SLUG, help="Kaggle dataset slug.")
    parser.add_argument("--output", type=Path, default=Path("data/resistor_kaggle_raw"))
    parser.add_argument("--force", action="store_true", help="Overwrite an existing download.")
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "kaggle",
        "datasets",
        "download",
        "-d",
        args.dataset,
        "-p",
        str(output),
        "--unzip",
    ]
    if args.force:
        command.append("--force")

    print(f"Kaggle dataset: https://www.kaggle.com/datasets/{args.dataset}")
    print(f"Output directory: {output}")
    print("Kaggle credentials are required; this script never reads or prints your token.")
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as error:
        raise SystemExit(
            "The Kaggle Python package is not installed. Run: "
            "python -m pip install -r requirements-training.txt"
        ) from error
    except subprocess.CalledProcessError as error:
        raise SystemExit(
            "Kaggle download failed. Check your internet connection, Kaggle API credentials, "
            "dataset access, and current Kaggle terms/licensing, then retry."
        ) from error

    images = sorted(
        path for path in output.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    annotations = sorted(
        path for path in output.rglob("*")
        if path.is_file() and path.suffix.lower() in ANNOTATION_SUFFIXES
        and path.name.lower() not in {"readme.txt", "license.txt"}
    )
    print(f"Image/mask-like files: {len(images)}")
    print(f"Potential annotation/metadata files: {len(annotations)}")
    for path in annotations[:30]:
        print(f"  {path.relative_to(output)}")
    if len(annotations) > 30:
        print(f"  ... and {len(annotations) - 30} more file(s)")
    print("")
    print("Next: inspect the annotation formats and license before training or redistribution.")
    print("Do not use unlabelled photos as segmentation masks; every mask must pair with its source image.")


if __name__ == "__main__":
    main()
