# region 01: Imports and Configuration
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

# endregion


# region 02: Dataset Label Cleanup
def sanitize_detection_labels(data_file: Path) -> tuple[int, int]:
    """Drop malformed or segmentation-polygon rows before detection-only training.

    YOLO detection rows must contain exactly five numeric fields:
    class_id, x_center, y_center, width, height. Mixed polygon/detection
    annotation files are normalized by keeping valid detection rows only.
    """
    import math

    scanned = 0
    removed = 0
    dataset_root = data_file.parent
    for label_file in dataset_root.rglob("*.txt"):
        if "labels" not in {part.casefold() for part in label_file.parts}:
            continue
        scanned += 1
        original = label_file.read_text(encoding="utf-8", errors="replace").splitlines()
        kept = []
        for line in original:
            parts = line.split()
            if len(parts) != 5:
                if line.strip():
                    removed += 1
                continue
            try:
                values = [float(value) for value in parts]
            except ValueError:
                removed += 1
                continue
            class_id, x_center, y_center, width, height = values
            valid = (
                all(math.isfinite(value) for value in values)
                and class_id >= 0
                and class_id.is_integer()
                and 0.0 <= x_center <= 1.0
                and 0.0 <= y_center <= 1.0
                and 0.0 < width <= 1.0
                and 0.0 < height <= 1.0
            )
            if valid:
                kept.append(" ".join(parts))
            else:
                removed += 1
        normalized = "\n".join(kept)
        if normalized:
            normalized += "\n"
        if normalized != label_file.read_text(encoding="utf-8", errors="replace"):
            label_file.write_text(normalized, encoding="utf-8")
    print(
        f"Detection-label cleanup: scanned {scanned} label files; "
        f"removed {removed} malformed/polygon rows.",
        flush=True,
    )
    if scanned == 0:
        raise RuntimeError(
            f"No YOLO label files were found beneath {dataset_root}. "
            "Check the dataset export layout."
        )
    return scanned, removed


# endregion


# region 03: YOLO Training
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune a small Ultralytics YOLO detector on an exported electronics-components dataset."
    )
    parser.add_argument("--data", type=Path, default=Path("data/component_dataset/data.yaml"))
    parser.add_argument("--model", default="yolo11n.pt", help="Initial Ultralytics model, e.g. yolo11n.pt or yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--fraction", type=float, default=0.25, help="Fraction of the dataset used for faster lightweight training.")
    parser.add_argument("--workers", type=int, default=2, help="Data-loader workers.")
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

    if args.epochs < 1 or args.imgsz < 32 or args.batch < 1 or not 0 < args.fraction <= 1 or args.workers < 0:
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

    sanitize_detection_labels(data_path)

    print(f"Dataset YAML: {data_path}")
    print(f"Initial model: {args.model}")
    print(f"Training for {args.epochs} epochs at image size {args.imgsz}.")

    # Keep pretrained starter checkpoints in models/base so subsequent local runs
    # can reuse them without fetching them again from Ultralytics.
    base_dir = Path("models/base")
    base_dir.mkdir(parents=True, exist_ok=True)
    requested_model = Path(args.model)
    cached_model = base_dir / requested_model.name
    if requested_model.is_file():
        model = YOLO(str(requested_model.resolve()))
        source_checkpoint = requested_model.resolve()
    elif cached_model.is_file():
        model = YOLO(str(cached_model.resolve()))
        source_checkpoint = cached_model.resolve()
        print(f"Using cached pretrained checkpoint: {source_checkpoint}")
    else:
        model = YOLO(args.model)
        checkpoint_value = getattr(model, "ckpt_path", None)
        source_checkpoint = Path(checkpoint_value).expanduser() if checkpoint_value else None
        if source_checkpoint and source_checkpoint.is_file():
            import shutil
            shutil.copy2(source_checkpoint, cached_model)
            print(f"Cached pretrained checkpoint: {cached_model}")
        else:
            print("Warning: Ultralytics did not expose the downloaded checkpoint path; it remains in its own cache.")

    train_options = {
        "data": str(data_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": args.project,
        "name": args.name,
        "patience": 4,
        "plots": False,
        "fraction": args.fraction,
        "workers": args.workers,
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
