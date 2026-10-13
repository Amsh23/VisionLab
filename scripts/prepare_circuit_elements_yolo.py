"""Normalize the public Circuit Elements YOLO dataset for VisionLab.

Downloads are performed by the GitHub Actions workflow. This script merges
obvious class-name variants (for example, Resistor/Ressistor/Resistort) and
writes a fresh YOLO dataset without altering the downloaded source files.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml

SPLIT_CANDIDATES = {
    "train": ("train",),
    "val": ("valid", "val", "validation"),
    "test": ("test",),
}
ALIASES = {
    "resistor": "Resistor",
    "resistance": "Resistor",
    "resistort": "Resistor",
    "ressistor": "Resistor",
    "capacitor": "Capacitor",
    "electrolytic capacitor": "Capacitor",
    "capacitor network": "Capacitor Network",
    "capacitor jumper": "Capacitor",
    "diode": "Diode",
    "zener diode": "Diode",
    "led": "LED",
    "inductor": "Inductor",
    "coil": "Inductor",
    "ic": "IC",
    "button": "Switch/Button",
    "switch": "Switch/Button",
    "switch/button": "Switch/Button",
    "jumper": "Jumper Wire",
    "conductor": "Jumper Wire",
    "test point": "Test Point",
    "debug/tp": "Test Point",
    "clock": "Clock / Oscillator",
    "crystal": "Clock / Oscillator",
    "unknown unlabeled": None,
}


def canonical_name(value: str) -> str | None:
    name = " ".join(str(value).strip().replace("_", " ").split())
    if not name:
        return None
    key = name.casefold()
    if key in ALIASES:
        return ALIASES[key]
    if key in {"em", "rc", "rp", "ps", "sk"}:
        return key.upper()
    return name[0].upper() + name[1:]


def read_names(value) -> dict[int, str]:
    if isinstance(value, list):
        return {index: str(name) for index, name in enumerate(value)}
    if isinstance(value, dict):
        result = {}
        for key, name in value.items():
            try:
                result[int(key)] = str(name)
            except (TypeError, ValueError):
                continue
        return result
    raise ValueError("data.yaml must define class names as a list or an ID-to-name mapping.")


def find_split(root: Path, names: tuple[str, ...]) -> tuple[Path, Path] | None:
    for split_name in names:
        base = root / split_name
        for images_name in ("images", "imgs"):
            images = base / images_name
            labels = base / "labels"
            if images.is_dir() and labels.is_dir():
                return images, labels
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize Circuit Elements YOLO class labels.")
    parser.add_argument("--input", type=Path, default=Path("data/circuit-elements"))
    parser.add_argument("--output", type=Path, default=Path("data/circuit-elements-normalized"))
    args = parser.parse_args()

    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    yaml_path = source / "data.yaml"
    if not yaml_path.is_file():
        yaml_candidates = list(source.rglob("data.yaml"))
        if len(yaml_candidates) != 1:
            raise SystemExit(f"Expected one data.yaml in {source}, found {len(yaml_candidates)}.")
        yaml_path = yaml_candidates[0]
        source = yaml_path.parent

    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    source_names = read_names(config.get("names"))
    mapped_names = {old_id: canonical_name(name) for old_id, name in source_names.items()}
    class_names = sorted({name for name in mapped_names.values() if name}, key=str.casefold)
    new_ids = {name: index for index, name in enumerate(class_names)}
    old_to_new = {
        old_id: new_ids[name]
        for old_id, name in mapped_names.items()
        if name is not None
    }
    if not class_names:
        raise SystemExit("No usable class names were found in data.yaml.")

    counts = {"train": 0, "val": 0, "test": 0}
    for split, candidates in SPLIT_CANDIDATES.items():
        split_dirs = find_split(source, candidates)
        if split_dirs is None:
            if split == "test":
                continue
            raise SystemExit(f"Missing {split} images/labels in {source}.")
        source_images, source_labels = split_dirs
        target_images = output / split / "images"
        target_labels = output / split / "labels"
        target_images.mkdir(parents=True, exist_ok=True)
        target_labels.mkdir(parents=True, exist_ok=True)

        for image_path in sorted(p for p in source_images.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}):
            relative = image_path.relative_to(source_images)
            label_path = source_labels / relative.with_suffix(".txt")
            if not label_path.is_file():
                continue
            converted_lines = []
            for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    old_id = int(parts[0])
                except ValueError:
                    continue
                if old_id not in old_to_new:
                    continue
                if len(parts) < 5:
                    continue
                converted_lines.append(str(old_to_new[old_id]) + " " + " ".join(parts[1:]))
            if not converted_lines:
                continue
            destination = target_images / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, destination)
            destination_label = target_labels / relative.with_suffix(".txt")
            destination_label.parent.mkdir(parents=True, exist_ok=True)
            destination_label.write_text("\n".join(converted_lines) + "\n", encoding="utf-8")
            counts[split] += 1

    if counts["train"] < 10 or counts["val"] < 2:
        raise SystemExit(f"Too few annotated images after normalization: {counts}")

    yaml_lines = [
        f"path: {json.dumps(str(output))}",
        "train: train/images",
        "val: val/images",
        "test: test/images" if counts["test"] else "val: val/images",
        "names:",
    ]
    for index, name in enumerate(class_names):
        yaml_lines.append(f"  {index}: {json.dumps(name)}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    (output / "DATASET_SOURCE.txt").write_text(
        "Dataset: LibreYOLO/Circuit Elements (Roboflow 100)\n"
        "Source: https://huggingface.co/datasets/LibreYOLO/circuit-elements\n"
        "License shown on dataset card: CC BY 4.0\n"
        "This derived local training copy merges obvious spelling/class aliases.\n",
        encoding="utf-8",
    )
    print(f"Normalized annotated images: {counts}")
    print(f"Classes ({len(class_names)}): {', '.join(class_names)}")
    print(f"YOLO config: {output / 'data.yaml'}")


if __name__ == "__main__":
    main()
