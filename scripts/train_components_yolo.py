# region 01: Imports and Configuration
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

# endregion


# region 02: YOLO Training
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune a small Ultralytics YOLO detector on an exported electronics-components dataset."
    )
    parser.add_argument("--data", type=Path, default=Path("data/component_dataset/data.yaml"))
    parser.add_argument("--model", default="yolo11n.pt", help="Initial Ultralytics model, e.g. yolo11n.pt or yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None, help="Optional device, e.g. cpu, 0, or 0,1")
    parser.add_argument("--project", default="runs/component_detection")
    parser.add_argument("--name", default="electronics_components")
    parser.add_argument(
        "--weights-out",
        type=Path,
        default=Path("models/components.pt"),
        help="Optional destination for the trained best.pt weights. Set to models/iot_boards.pt for the IoTKITs dataset.",
    )
    args = parser.parse_args()

    if args.epochs < 1 or args.imgsz < 32 or args.batch < 1:
        parser.error("--epochs and --batch must be positive; --imgsz must be at least 32.")
    data_path = args.data.expanduser().resolve()
    if not data_path.is_file():
        parser.error(
            f"Dataset YAML not found: {data_path}. First run "
            "scripts/download_component_dataset.py or set --data to your exported data.yaml."
        )

    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit(
            "Ultralytics is an optional training dependency. Install it with: "
            "python -m pip install ultralytics"
        ) from error

    print(f"Dataset YAML: {data_path}")
    print(f"Initial model: {args.model}")
    print(f"Training for {args.epochs} epochs at image size {args.imgsz}.")
    print("Training may download the small initial YOLO checkpoint if it is not cached locally.")
    model = YOLO(args.model)
    train_options = {
        "data": str(data_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": args.project,
        "name": args.name,
        "patience": 10,
        "plots": True,
    }
    if args.device:
        train_options["device"] = args.device

    model.train(**train_options)
    trainer = getattr(model, "trainer", None)
    save_dir = Path(getattr(trainer, "save_dir", Path(args.project) / args.name))
    best_weights = save_dir / "weights" / "best.pt"

    print(f"Training finished. Expected best weights: {best_weights}")
    if best_weights.is_file():
        destination = args.weights_out.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        if best_weights.resolve() != destination:
            shutil.copy2(best_weights, destination)
        print(f"Best weights copied to: {destination}")
        print("Select this weights path in Electronic Component Analyzer -> Local YOLO weights.")
    else:
        print("Training returned, but best.pt was not found at the expected path. Check the training log.")


if __name__ == "__main__":
    main()
# endregion
