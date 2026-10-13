"""Download a public Roboflow Universe dataset preset for VisionLab.

The script keeps the downloaded dataset local and prints the output location.
Never commit API keys. Review each dataset's license and source attribution
before sharing derived assets or model weights.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

DATASETS = {
    "iot": {
        "workspace": "design-project-zeico",
        "project": "iot_dataset",
        "version": 2,
        "format": "yolov8",
        "output": Path("data/iot_roboflow"),
        "task": "object detection",
        "source": "https://universe.roboflow.com/design-project-zeico/iot_dataset",
    },
    "resistor-segmentation": {
        "workspace": "resistor-color-code",
        "project": "resistor-value-determination",
        "version": 6,
        "format": "png-mask-semantic",
        "output": Path("data/resistor_value_segmentation"),
        "task": "semantic segmentation",
        "source": "https://universe.roboflow.com/resistor-color-code/resistor-value-determination",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download one of the public Roboflow Universe datasets used by VisionLab."
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        required=True,
        help="Dataset preset: iot or resistor-segmentation.",
    )
    parser.add_argument("--version", type=int, help="Override the dataset version number.")
    parser.add_argument("--format", dest="model_format", help="Override the Roboflow export format.")
    parser.add_argument("--output", type=Path, help="Override the local output directory.")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("ROBOFLOW_API_KEY", ""),
        help="Roboflow API key. Prefer the ROBOFLOW_API_KEY environment variable.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace the existing export if present.")
    args = parser.parse_args()

    config = DATASETS[args.dataset]
    workspace = config["workspace"]
    project_slug = config["project"]
    version_number = args.version or config["version"]
    model_format = args.model_format or config["format"]
    output = (args.output or config["output"]).expanduser().resolve()

    if not args.api_key.strip():
        parser.error(
            "A Roboflow API key is required. Set ROBOFLOW_API_KEY or pass --api-key. "
            "Get the key from https://app.roboflow.com/settings/api."
        )
    if version_number < 1:
        parser.error("--version must be a positive integer.")

    try:
        from roboflow import Roboflow
    except ImportError as error:
        raise SystemExit(
            "Roboflow's optional downloader is not installed. Run: "
            "python -m pip install roboflow"
        ) from error

    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Source: {config['source']}")
    print(f"Task: {config['task']}")
    print(f"Project version: {workspace}/{project_slug}/{version_number}")
    print(f"Export format: {model_format}")
    print(f"Download location: {output}")
    print("Keep the dataset's attribution and license details with the downloaded data.")

    client = Roboflow(api_key=args.api_key.strip())
    project = client.workspace(workspace).project(project_slug)
    version = project.version(version_number)
    dataset = version.download(
        model_format=model_format,
        location=str(output),
        overwrite=args.overwrite,
    )

    print("Dataset download complete.")
    print(f"Dataset folder: {dataset.location}")
    yaml_candidates = list(Path(dataset.location).rglob("data.yaml")) + list(Path(dataset.location).rglob("data.yml"))
    if yaml_candidates:
        print(f"Dataset configuration: {yaml_candidates[0]}")
    classes_csv = list(Path(dataset.location).rglob("_classes.csv"))
    if classes_csv:
        print(f"Class mapping: {classes_csv[0]}")
    if args.dataset == "iot":
        print("Next: train an object detector with scripts/train_components_yolo.py and the exported data.yaml.")
    else:
        print(
            "This export contains semantic masks, not YOLO bounding-box labels. "
            "Inspect data.yaml, _classes.csv, and paired mask PNGs before training a segmentation model."
        )


if __name__ == "__main__":
    main()
