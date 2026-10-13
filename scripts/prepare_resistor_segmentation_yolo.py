"""Convert semantic PNG masks into YOLO segmentation polygon labels.

The converter supports common Roboflow PNG-mask exports (flat files or split
folders) and paired image/mask directories. It maps mask pixel values to class
names from a *_classes.csv file when available. Pixel value 0 is treated as
background by default; change --background-value if the exported mask uses a
different convention.

This conversion traces each connected region of a class into a polygon. It is
useful for training an Ultralytics segmentation model from semantic masks, but
it is not a bit-exact semantic-segmentation trainer.
"""
from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MASK_DIR_NAMES = {"mask", "masks", "labels", "annotations", "semantic", "segmentation"}
MASK_SUFFIXES = ("_mask", "-mask", "_label", "-label", "_seg", "-seg", "_segmentation")


def is_mask_path(path: Path) -> bool:
    parts = {part.casefold() for part in path.parts}
    stem = path.stem.casefold()
    return (
        bool(parts & MASK_DIR_NAMES)
        or any(stem.endswith(suffix) for suffix in MASK_SUFFIXES)
        or stem.startswith("mask_")
    )


def list_images(root: Path) -> list[Path]:
    candidates = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if is_mask_path(path):
            continue
        # In flat Roboflow semantic exports, a JPEG image and a PNG mask can
        # share the same stem. Do not treat that PNG mask as a second source image.
        if path.suffix.lower() == ".png" and any(
            path.with_suffix(suffix).is_file()
            for suffix in (".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
        ):
            continue
        candidates.append(path)
    return sorted(candidates)


def list_masks(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*.png")
        if path.is_file()
        and (
            is_mask_path(path)
            or (path.with_suffix(".jpg").is_file() and path.with_suffix(".jpg") != path)
            or (path.with_suffix(".jpeg").is_file() and path.with_suffix(".jpeg") != path)
        )
    )


def relative_path_under(path: Path, root: Path) -> Path | None:
    try:
        return path.relative_to(root)
    except ValueError:
        return None


def pair_image_mask(image_path: Path, root: Path, mask_files: list[Path]) -> Path | None:
    """Find a mask with the same stem in typical flat or mirrored folder layouts."""
    stem = image_path.stem.casefold()
    image_rel = relative_path_under(image_path, root)
    mask_by_stem: dict[str, list[Path]] = {}
    for path in mask_files:
        mask_by_stem.setdefault(path.stem.casefold(), []).append(path)

    preferred = []
    if image_rel is not None:
        relative_parent = image_rel.parent
        candidate_relatives = [
            relative_parent / f"{image_path.stem}.png",
            relative_parent / "masks" / f"{image_path.stem}.png",
            relative_parent / "mask" / f"{image_path.stem}.png",
            relative_parent / "labels" / f"{image_path.stem}.png",
            relative_parent / "annotations" / f"{image_path.stem}.png",
        ]
        current_parts = list(relative_parent.parts)
        for index, part in enumerate(current_parts):
            if part.casefold() in {"images", "image", "imgs"}:
                altered = current_parts[:]
                altered[index] = "masks"
                candidate_relatives.append(Path(*altered) / f"{image_path.stem}.png")
                altered = current_parts[:]
                altered[index] = "mask"
                candidate_relatives.append(Path(*altered) / f"{image_path.stem}.png")
        for relative in candidate_relatives:
            candidate = root / relative
            if candidate.is_file() and candidate != image_path and candidate.suffix.lower() == ".png":
                preferred.append(candidate)

    # Common same-directory naming variants.
    parent = image_path.parent
    for filename in (
        f"{image_path.stem}.png",
        f"{image_path.stem}_mask.png",
        f"{image_path.stem}-mask.png",
        f"{image_path.stem}_label.png",
        f"{image_path.stem}_seg.png",
        f"mask_{image_path.stem}.png",
    ):
        candidate = parent / filename
        if candidate.is_file() and candidate != image_path:
            preferred.append(candidate)

    # Same-stem recursive fallback, prioritizing mask-looking names/directories.
    preferred.extend(
        sorted(
            mask_by_stem.get(stem, []),
            key=lambda candidate: (
                0 if is_mask_path(candidate) else 1,
                len(candidate.parts),
                str(candidate),
            ),
        )
    )
    seen = set()
    for candidate in preferred:
        if candidate == image_path or candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return candidate
    return None


def read_class_mapping(root: Path) -> dict[int, str]:
    """Read common Roboflow _classes.csv layouts into pixel-value to name."""
    mappings: dict[int, str] = {}
    csv_files = sorted(root.rglob("_classes.csv"))
    if not csv_files:
        return mappings
    with csv_files[0].open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 2:
                continue
            numeric_index = None
            for item in row:
                try:
                    numeric_index = int(item.strip())
                    break
                except (ValueError, AttributeError):
                    continue
            if numeric_index is None:
                continue
            text_fields = [item.strip() for item in row if item.strip()]
            nonnumeric = []
            for item in text_fields:
                try:
                    int(item)
                except ValueError:
                    nonnumeric.append(item)
            if nonnumeric:
                mappings[numeric_index] = nonnumeric[-1]
    return mappings


def read_mask(path: Path) -> np.ndarray:
    """Read a grayscale or indexed PNG mask without losing integer class IDs."""
    with Image.open(path) as image:
        mask = np.array(image)
    if mask.ndim == 2:
        return mask.astype(np.int32, copy=False)
    if mask.ndim == 3 and mask.shape[2] == 1:
        return mask[:, :, 0].astype(np.int32, copy=False)
    if mask.ndim == 3 and mask.shape[2] >= 3:
        first = mask[:, :, 0]
        if np.array_equal(first, mask[:, :, 1]) and np.array_equal(first, mask[:, :, 2]):
            return first.astype(np.int32, copy=False)
        raise ValueError(
            f"Mask {path} is RGB/palette-colored rather than a single-channel class-ID mask. "
            "Convert it to class-index pixels using its palette/metadata before training."
        )
    raise ValueError(f"Unsupported mask shape {mask.shape}: {path}")


def contours_for_class(mask: np.ndarray, class_value: int, min_area: float) -> list[np.ndarray]:
    binary = np.where(mask == class_value, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if abs(cv2.contourArea(contour)) < min_area:
            continue
        simplified = cv2.approxPolyDP(contour, 0.003 * cv2.arcLength(contour, True), True)
        points = simplified.reshape(-1, 2)
        if len(points) >= 3:
            polygons.append(points)
    return polygons


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert paired semantic PNG masks into YOLO segmentation labels."
    )
    parser.add_argument("--input", type=Path, default=Path("data/resistor_value_segmentation"))
    parser.add_argument("--output", type=Path, default=Path("data/resistor_seg_yolo"))
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--background-value", type=int, default=0)
    parser.add_argument("--min-area", type=float, default=6.0)
    args = parser.parse_args()

    if args.val_ratio < 0 or args.test_ratio < 0 or args.val_ratio + args.test_ratio >= 0.8:
        parser.error("Validation/test ratios must be non-negative and sum to less than 0.8.")
    if args.min_area < 0:
        parser.error("--min-area must be non-negative.")

    root = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"Dataset folder does not exist: {root}. Download the Roboflow semantic dataset first.")

    mask_files = list_masks(root)
    images = list_images(root)
    pairs = []
    missing = []
    for image_path in images:
        mask_path = pair_image_mask(image_path, root, mask_files)
        if mask_path is None:
            missing.append(image_path)
            continue
        try:
            mask = read_mask(mask_path)
            with Image.open(image_path) as source_image:
                width, height = source_image.size
            if mask.shape[:2] != (height, width):
                print(f"Skip size mismatch: {image_path.name} ({width}x{height}) vs {mask_path.name} ({mask.shape[1]}x{mask.shape[0]})")
                continue
            values = set(int(value) for value in np.unique(mask))
            if args.background_value in values:
                values.remove(args.background_value)
            if 255 in values:
                values.remove(255)
            if not values:
                continue
            pairs.append((image_path, mask_path, values))
        except (OSError, ValueError) as error:
            print(f"Skip invalid mask pair {image_path.name}: {error}")

    if not pairs:
        raise SystemExit(
            "No valid image/mask pairs were found. Check the downloaded folder layout, "
            "mask filenames, image sizes, and class-ID encoding. Do not train with unpaired images."
        )

    observed_values = sorted({value for _, _, values in pairs for value in values})
    class_mapping = read_class_mapping(root)
    class_names = {
        value: class_mapping.get(value, f"class_{value}")
        for value in observed_values
    }
    contiguous_ids = {value: index for index, value in enumerate(observed_values)}

    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    test_count = int(round(len(pairs) * args.test_ratio))
    val_count = int(round(len(pairs) * args.val_ratio))
    if len(pairs) >= 3 and args.val_ratio > 0:
        val_count = max(1, val_count)
    if len(pairs) >= 3 and args.test_ratio > 0:
        test_count = max(1, test_count)
    while val_count + test_count >= len(pairs):
        if test_count > 0:
            test_count -= 1
        elif val_count > 0:
            val_count -= 1
        else:
            break
    train_count = len(pairs) - val_count - test_count
    if train_count < 1:
        raise SystemExit("Not enough valid pairs for the requested train/validation/test split.")

    split_data = {
        "train": pairs[:train_count],
        "val": pairs[train_count:train_count + val_count],
        "test": pairs[train_count + val_count:],
    }

    for split, split_pairs in split_data.items():
        image_dir = output / "images" / split
        label_dir = output / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        for index, (source_image, mask_path, _) in enumerate(split_pairs):
            with Image.open(source_image) as opened:
                image_width, image_height = opened.size
            mask = read_mask(mask_path)
            file_stem = f"{index:05d}_{source_image.stem}"
            destination_image = image_dir / f"{file_stem}{source_image.suffix.lower()}"
            shutil.copy2(source_image, destination_image)

            label_lines = []
            for pixel_value in sorted(contiguous_ids):
                polygons = contours_for_class(mask, pixel_value, args.min_area)
                class_id = contiguous_ids[pixel_value]
                for polygon in polygons:
                    normalized = []
                    for x, y in polygon:
                        normalized.extend([
                            min(1.0, max(0.0, float(x) / max(1, image_width))),
                            min(1.0, max(0.0, float(y) / max(1, image_height))),
                        ])
                    if len(normalized) >= 6:
                        label_lines.append(
                            str(class_id) + " " + " ".join(f"{value:.6f}" for value in normalized)
                        )
            (label_dir / f"{file_stem}.txt").write_text(
                "\n".join(label_lines) + ("\n" if label_lines else ""),
                encoding="utf-8",
            )

    yaml_lines = [
        f"path: {output.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    for pixel_value in sorted(contiguous_ids):
        class_id = contiguous_ids[pixel_value]
        name = class_names[pixel_value].replace("\\", "\\\\").replace('"', '\\"')
        yaml_lines.append(f'  {class_id}: "{name}"')
    (output / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    print(f"Valid image/mask pairs: {len(pairs)}")
    print(f"Images without a paired mask: {len(missing)}")
    print(f"Class pixel values: {observed_values}")
    print(f"Classes: {', '.join(class_names[value] for value in observed_values)}")
    print(f"Splits: train={len(split_data['train'])}, val={len(split_data['val'])}, test={len(split_data['test'])}")
    print(f"YOLO segmentation dataset: {output}")
    print(f"Dataset YAML: {output / 'data.yaml'}")
    print("The masks were polygonized by connected class region; inspect sample labels before training.")


if __name__ == "__main__":
    main()
