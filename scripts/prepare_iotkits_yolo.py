"""Convert common polygon/box annotation JSON files into a YOLO detection dataset.

Supported inputs:
- COCO JSON with images, categories, and annotations.
- Per-image Roboflow JSON containing objects/classTitle/points.
- LabelMe JSON containing shapes/label/points.
- Per-image annotations stored beside, or in a parallel folder to, each image.

If the download already contains a YOLO data.yaml, use that file directly with
scripts/train_components_yolo.py instead of converting it.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def build_image_index(root: Path) -> tuple[dict[str, list[Path]], dict[str, Path]]:
    by_name: dict[str, list[Path]] = defaultdict(list)
    by_stem: dict[str, list[Path]] = defaultdict(list)
    for image_path in list_images(root):
        by_name[image_path.name.casefold()].append(image_path)
        by_stem[image_path.stem.casefold()].append(image_path)
    return by_name, by_stem


def resolve_image(value: str | None, annotation_path: Path, root: Path,
                  by_name: dict[str, list[Path]], by_stem: dict[str, list[Path]]) -> Path | None:
    """Resolve a file name in JSON, then fall back to the paired image stem."""
    if value:
        candidate = (root / value).resolve()
        try:
            candidate.relative_to(root.resolve())
            if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES:
                return candidate
        except ValueError:
            pass

        basename = Path(value).name.casefold()
        matches = by_name.get(basename, [])
        if len(matches) == 1:
            return matches[0]
        stem_matches = by_stem.get(Path(value).stem.casefold(), [])
        if stem_matches:
            parent_matches = [item for item in stem_matches if item.parent == annotation_path.parent]
            return (parent_matches or stem_matches)[0]

    sibling_matches = by_stem.get(annotation_path.stem.casefold(), [])
    if sibling_matches:
        parent_matches = [item for item in sibling_matches if item.parent == annotation_path.parent]
        return (parent_matches or sibling_matches)[0]
    return None


def image_size(path: Path, metadata: dict | None = None) -> tuple[int, int]:
    metadata = metadata or {}
    width = metadata.get("width") or metadata.get("imageWidth")
    height = metadata.get("height") or metadata.get("imageHeight")
    if width and height:
        return int(width), int(height)
    with Image.open(path) as image:
        return image.size


def parse_points(points) -> list[tuple[float, float]]:
    """Normalize point lists used by COCO, Roboflow, and LabelMe JSON."""
    if isinstance(points, dict):
        points = points.get("exterior") or points.get("points") or []
    parsed: list[tuple[float, float]] = []
    if not isinstance(points, list):
        return parsed

    for point in points:
        if isinstance(point, dict):
            x = point.get("x", point.get("X"))
            y = point.get("y", point.get("Y"))
            if x is not None and y is not None:
                parsed.append((float(x), float(y)))
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            parsed.append((float(point[0]), float(point[1])))
    return parsed


def parse_box(annotation: dict, width: int, height: int) -> tuple[float, float, float, float] | None:
    """Return pixel xyxy coordinates from a bounding box or polygon."""
    points = parse_points(
        annotation.get("points")
        or annotation.get("segmentation")
        or annotation.get("polygon")
    )
    if points:
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        if max(map(abs, xs), default=0) <= 1.0 and max(map(abs, ys), default=0) <= 1.0:
            x1, x2 = x1 * width, x2 * width
            y1, y2 = y1 * height, y2 * height
    else:
        raw_box = annotation.get("bbox") or annotation.get("bounding_box") or annotation.get("box")
        if isinstance(raw_box, dict):
            if all(key in raw_box for key in ("x1", "y1", "x2", "y2")):
                x1, y1, x2, y2 = (float(raw_box[key]) for key in ("x1", "y1", "x2", "y2"))
            else:
                x = raw_box.get("x", raw_box.get("left"))
                y = raw_box.get("y", raw_box.get("top"))
                box_width = raw_box.get("width", raw_box.get("w"))
                box_height = raw_box.get("height", raw_box.get("h"))
                if None in (x, y, box_width, box_height):
                    return None
                x1, y1, x2, y2 = float(x), float(y), float(x) + float(box_width), float(y) + float(box_height)
        elif isinstance(raw_box, list) and len(raw_box) >= 4:
            x, y, box_width, box_height = map(float, raw_box[:4])
            x1, y1, x2, y2 = x, y, x + box_width, y + box_height
        else:
            return None

    # Some annotation tools store normalized values. Convert only when every
    # coordinate is within [0, 1], which cannot describe a useful pixel-sized box.
    if max(abs(x1), abs(x2)) <= 1.0 and max(abs(y1), abs(y2)) <= 1.0:
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height

    x1 = max(0.0, min(float(width), x1))
    x2 = max(0.0, min(float(width), x2))
    y1 = max(0.0, min(float(height), y1))
    y2 = max(0.0, min(float(height), y2))
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return x1, y1, x2, y2


def class_name_for(annotation: dict, category_names: dict[int, str] | None = None) -> str | None:
    for key in ("classTitle", "label", "class_name", "name", "category_name"):
        value = annotation.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    value = annotation.get("class", annotation.get("category"))
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, dict):
        for key in ("name", "title", "classTitle"):
            if isinstance(value.get(key), str):
                return value[key].strip()
    if isinstance(value, (int, float)) and category_names:
        return category_names.get(int(value))
    class_id = annotation.get("category_id", annotation.get("class_id"))
    if isinstance(class_id, (int, float)) and category_names:
        return category_names.get(int(class_id))
    return None


def parse_coco_json(path: Path, root: Path, by_name, by_stem) -> list[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not all(isinstance(data.get(key), list) for key in ("images", "annotations", "categories")):
        return []

    categories = {int(item["id"]): str(item["name"]) for item in data["categories"] if "id" in item and "name" in item}
    images_by_id = {int(item["id"]): item for item in data["images"] if "id" in item}
    grouped: dict[int, list[dict]] = defaultdict(list)
    for annotation in data["annotations"]:
        if annotation.get("iscrowd", 0):
            continue
        grouped[int(annotation.get("image_id", -1))].append(annotation)

    records = []
    for image_id, image_meta in images_by_id.items():
        image_path = resolve_image(image_meta.get("file_name") or image_meta.get("path"),
                                   path, root, by_name, by_stem)
        if image_path is None:
            continue
        width, height = image_size(image_path, image_meta)
        objects = []
        for annotation in grouped.get(image_id, []):
            name = class_name_for(annotation, categories)
            box = parse_box(annotation, width, height)
            if name and box:
                objects.append((name, *box))
        if objects:
            records.append({"image": image_path, "width": width, "height": height, "objects": objects})
    return records


def parse_per_image_json(path: Path, root: Path, by_name, by_stem) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None

    objects = data.get("objects")
    if not isinstance(objects, list):
        objects = data.get("shapes")
    if not isinstance(objects, list):
        objects = data.get("annotations")
    if not isinstance(objects, list):
        return None

    image_meta = data.get("image") if isinstance(data.get("image"), dict) else {}
    image_value = (
        data.get("imagePath") or data.get("image_path") or data.get("file_name")
        or data.get("filename") or image_meta.get("file_name") or image_meta.get("path")
    )
    image_path = resolve_image(image_value, path, root, by_name, by_stem)
    if image_path is None:
        return None
    width, height = image_size(image_path, {
        "width": data.get("imageWidth") or data.get("width") or image_meta.get("width"),
        "height": data.get("imageHeight") or data.get("height") or image_meta.get("height"),
    })

    parsed_objects = []
    for item in objects:
        if not isinstance(item, dict):
            continue
        name = class_name_for(item)
        box = parse_box(item, width, height)
        if name and box:
            parsed_objects.append((name, *box))
    if not parsed_objects:
        return None
    return {"image": image_path, "width": width, "height": height, "objects": parsed_objects}


def write_yolo_dataset(records: list[dict], output: Path, val_ratio: float, test_ratio: float, seed: int) -> None:
    class_names = sorted({obj[0] for record in records for obj in record["objects"]}, key=str.casefold)
    class_ids = {name: index for index, name in enumerate(class_names)}

    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)
    test_count = int(round(len(shuffled) * test_ratio))
    val_count = int(round(len(shuffled) * val_ratio))
    if len(shuffled) >= 3:
        test_count = max(1, test_count)
        val_count = max(1, val_count)
    if val_count + test_count >= len(shuffled):
        val_count, test_count = 1, 1
    train_count = len(shuffled) - val_count - test_count
    split_records = {
        "train": shuffled[:train_count],
        "val": shuffled[train_count:train_count + val_count],
        "test": shuffled[train_count + val_count:],
    }

    for split, split_items in split_records.items():
        image_dir = output / "images" / split
        label_dir = output / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        for index, record in enumerate(split_items):
            source_path = record["image"]
            file_stem = f"{index:05d}_{source_path.stem}"
            extension = source_path.suffix.lower()
            destination_image = image_dir / f"{file_stem}{extension}"
            shutil.copy2(source_path, destination_image)
            width, height = record["width"], record["height"]
            lines = []
            for object_record in record["objects"]:
                name, x1, y1, x2, y2 = object_record
                center_x = ((x1 + x2) / 2) / width
                center_y = ((y1 + y2) / 2) / height
                box_width = (x2 - x1) / width
                box_height = (y2 - y1) / height
                lines.append(
                    f"{class_ids[name]} {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}"
                )
            (label_dir / f"{file_stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    yaml_lines = [
        f"path: {json.dumps(str(output.resolve()))}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    for index, name in enumerate(class_names):
        yaml_lines.append(f"  {index}: {json.dumps(name)}")
    (output / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    print(f"Converted {len(shuffled)} annotated images into {output}")
    print(f"Classes ({len(class_names)}): {', '.join(class_names)}")
    print(f"Splits: train={len(split_records['train'])}, val={len(split_records['val'])}, test={len(split_records['test'])}")
    print(f"YOLO config: {output / 'data.yaml'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert IoTKITs polygon/box JSON annotations into a YOLO object-detection dataset."
    )
    parser.add_argument("--input", type=Path, default=Path("data/iotkits_raw"))
    parser.add_argument("--output", type=Path, default=Path("data/iotkits_yolo"))
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.val_ratio <= 0 or args.test_ratio < 0 or args.val_ratio + args.test_ratio >= 0.8:
        parser.error("Use a positive validation ratio, a non-negative test ratio, and val + test below 0.8.")

    input_root = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not input_root.is_dir():
        parser.error(f"Input directory does not exist: {input_root}. Download IoTKITs first.")

    existing_yaml = list(input_root.rglob("data.yaml")) + list(input_root.rglob("data.yml"))
    if existing_yaml:
        print(f"An existing YOLO config was found: {existing_yaml[0]}")
        print("Use it directly with scripts/train_components_yolo.py; no conversion is necessary.")
        return

    images = list_images(input_root)
    if not images:
        parser.error(f"No image files were found under: {input_root}")
    by_name, by_stem = build_image_index(input_root)

    records_by_image: dict[Path, dict] = {}
    json_paths = sorted(input_root.rglob("*.json"))
    for json_path in json_paths:
        for record in parse_coco_json(json_path, input_root, by_name, by_stem):
            records_by_image[record["image"]] = record

    for json_path in json_paths:
        record = parse_per_image_json(json_path, input_root, by_name, by_stem)
        if record:
            records_by_image[record["image"]] = record

    records = list(records_by_image.values())
    if not records:
        raise SystemExit(
            "No supported annotations were found. The archive may use another JSON structure. "
            "Open the downloaded folder and check whether it contains COCO annotations, per-image "
            "Roboflow/LabelMe JSON, or an exported YOLO data.yaml. If needed, import the dataset into "
            "Roboflow and export it in YOLOv8 format, then use that data.yaml with the training script."
        )

    output.mkdir(parents=True, exist_ok=True)
    write_yolo_dataset(records, output, args.val_ratio, args.test_ratio, args.seed)


if __name__ == "__main__":
    main()
