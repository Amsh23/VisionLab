# region 01: Imports and Configuration
from __future__ import annotations

import argparse
import os
from pathlib import Path

# endregion


# region 02: Dataset Download
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the public Roboflow electronics-components dataset in YOLOv8 format."
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("ROBOFLOW_API_KEY", ""),
        help="Roboflow API key. Prefer the ROBOFLOW_API_KEY environment variable.",
    )
    parser.add_argument("--workspace", default="vishnupriyas-workspace")
    parser.add_argument("--project", default="electronics-components-sveck")
    parser.add_argument("--version", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("data/component_dataset"))
    args = parser.parse_args()

    if not args.api_key.strip():
        parser.error(
            "A Roboflow API key is required. Set ROBOFLOW_API_KEY or pass --api-key. "
            "Get your key from https://app.roboflow.com/settings/api."
        )
    if args.version < 1:
        parser.error("--version must be a positive integer.")

    try:
        from roboflow import Roboflow
    except ImportError as error:
        raise SystemExit(
            "The optional Roboflow downloader is not installed. Run: "
            "python -m pip install roboflow"
        ) from error

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {args.workspace}/{args.project}, version {args.version}")
    print(f"Export format: YOLOv8. Download location: {output_path}")
    print("Review the dataset license and attribution instructions before redistribution.")

    roboflow = Roboflow(api_key=args.api_key.strip())
    project = roboflow.workspace(args.workspace).project(args.project)
    version = project.version(args.version)
    dataset = version.download("yolov8", location=str(output_path))

    data_yaml = Path(dataset.location) / "data.yaml"
    if not data_yaml.exists():
        raise SystemExit(
            f"Download finished, but data.yaml was not found at {data_yaml}. "
            "Check the export layout and update --output if needed."
        )

    print("Dataset downloaded successfully.")
    print(f"YOLO dataset configuration: {data_yaml}")
    print("Next step: run scripts/train_components_yolo.py with that data.yaml path.")


if __name__ == "__main__":
    main()
# endregion
