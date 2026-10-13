# region 01: Imports and Dataset Constants
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

DIGIT_COLORS = ["Black", "Brown", "Red", "Orange", "Yellow", "Green", "Blue", "Violet", "Gray", "White"]
DIGIT_VALUES = {color: index for index, color in enumerate(DIGIT_COLORS)}
MULTIPLIERS = {
    "Silver": 0.01, "Gold": 0.1, "Black": 1, "Brown": 10, "Red": 100,
    "Orange": 1_000, "Yellow": 10_000, "Green": 100_000, "Blue": 1_000_000,
    "Violet": 10_000_000,
}
TOLERANCES = {
    "Brown": 1, "Red": 2, "Green": 0.5, "Blue": 0.25, "Violet": 0.1,
    "Gray": 0.05, "Gold": 5, "Silver": 10,
}
RGB_COLORS = {
    "Black": (35, 35, 35), "Brown": (125, 75, 45), "Red": (215, 45, 45),
    "Orange": (235, 125, 30), "Yellow": (245, 215, 45), "Green": (35, 145, 70),
    "Blue": (40, 95, 210), "Violet": (135, 65, 170), "Gray": (135, 135, 135),
    "White": (242, 240, 225), "Gold": (190, 155, 70), "Silver": (190, 195, 200),
}
# endregion


# region 02: Synthetic Resistor Rendering
def create_band_sequence(rng: random.Random, band_count: int) -> tuple[list[str], float, float]:
    """Create a standards-based random 4-band or 5-band resistor label."""
    significant_count = band_count - 2
    first_digit = rng.choice(DIGIT_COLORS[1:])
    remaining_digits = [rng.choice(DIGIT_COLORS) for _ in range(significant_count - 1)]
    significant_colors = [first_digit, *remaining_digits]
    multiplier_color = rng.choice(list(MULTIPLIERS.keys()))
    tolerance_color = rng.choice(list(TOLERANCES.keys()))
    bands = [*significant_colors, multiplier_color, tolerance_color]
    digits = int("".join(str(DIGIT_VALUES[color]) for color in significant_colors))
    resistance_ohms = digits * MULTIPLIERS[multiplier_color]
    return bands, resistance_ohms, TOLERANCES[tolerance_color]


def render_synthetic_resistor(bands: list[str], rng: random.Random, np_rng: np.random.Generator) -> np.ndarray:
    """Draw a small synthetic RGB resistor crop with mild lighting variation."""
    height, width = 96, 320
    base = rng.randint(224, 249)
    canvas = np.full((height, width, 3), base, dtype=np.uint8)

    # A subtle vertical lighting gradient gives each background a little variation.
    gradient = np.linspace(-8, 8, height, dtype=np.float32).reshape(height, 1, 1)
    canvas = np.clip(canvas.astype(np.float32) + gradient, 0, 255).astype(np.uint8)

    center_y = height // 2
    body_left, body_right = 42, 278
    body_top, body_bottom = rng.randint(31, 35), rng.randint(61, 65)
    body_color = (
        rng.randint(185, 226),
        rng.randint(158, 202),
        rng.randint(115, 166),
    )

    # Draw the metal leads behind the resistor body.
    lead_y = center_y + rng.randint(-2, 2)
    cv2.line(canvas, (3, lead_y), (width - 4, lead_y), (150, 156, 161), 5, cv2.LINE_AA)
    cv2.line(canvas, (3, lead_y - 1), (width - 4, lead_y - 1), (210, 213, 215), 1, cv2.LINE_AA)

    # Rounded cylindrical body.
    body_height = (body_bottom - body_top) // 2
    cv2.rectangle(canvas, (body_left + body_height, body_top),
                  (body_right - body_height, body_bottom), body_color, -1)
    cv2.ellipse(canvas, (body_left + body_height, center_y), (body_height, body_height),
                0, 90, 270, body_color, -1, cv2.LINE_AA)
    cv2.ellipse(canvas, (body_right - body_height, center_y), (body_height, body_height),
                0, -90, 90, body_color, -1, cv2.LINE_AA)
    highlight = tuple(min(255, component + 18) for component in body_color)
    cv2.line(canvas, (body_left + 16, body_top + 4),
             (body_right - 16, body_top + 4), highlight, 2, cv2.LINE_AA)

    band_width = rng.randint(7, 12)
    centers = np.linspace(body_left + 31, body_right - 31, len(bands))
    for band_center, band_name in zip(centers, bands):
        x = int(round(band_center + rng.randint(-3, 3)))
        color = RGB_COLORS[band_name]
        left = x - band_width // 2
        right = left + band_width
        edge = tuple(max(0, channel - 22) for channel in color)
        cv2.rectangle(canvas, (left - 1, body_top + 1), (right + 1, body_bottom - 1), edge, -1)
        cv2.rectangle(canvas, (left, body_top + 1), (right, body_bottom - 1), color, -1)

    # Add slight image noise and a small rotation, but keep labels synthetic and exact.
    noise = np_rng.normal(0, rng.uniform(0.5, 3.0), canvas.shape)
    canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    angle = rng.uniform(-5.0, 5.0)
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    canvas = cv2.warpAffine(canvas, matrix, (width, height), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(245, 245, 245))
    return canvas
# endregion


# region 03: Dataset Generation
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic resistor color-code images and labels.")
    parser.add_argument("--count", type=int, default=300, help="Number of synthetic images to generate (default: 300).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible labels and images.")
    parser.add_argument("--output", type=Path, default=Path("data/generated_resistor_dataset"),
                        help="Output directory for images/ and labels.csv.")
    parser.add_argument("--five-band-share", type=float, default=0.35,
                        help="Fraction of images generated as five-band resistors (0.0 to 1.0).")
    args = parser.parse_args()

    if args.count < 1:
        parser.error("--count must be at least 1.")
    if not 0.0 <= args.five_band_share <= 1.0:
        parser.error("--five-band-share must be between 0 and 1.")

    random_generator = random.Random(args.seed)
    numpy_generator = np.random.default_rng(args.seed)
    images_directory = args.output / "images"
    images_directory.mkdir(parents=True, exist_ok=True)

    indexes = list(range(args.count))
    random_generator.shuffle(indexes)
    split_by_index = {}
    train_cutoff = int(args.count * 0.8)
    validation_cutoff = int(args.count * 0.9)
    for position, sample_index in enumerate(indexes):
        if position < train_cutoff:
            split_by_index[sample_index] = "train"
        elif position < validation_cutoff:
            split_by_index[sample_index] = "val"
        else:
            split_by_index[sample_index] = "test"

    metadata_rows = []
    for index in range(args.count):
        band_count = 5 if random_generator.random() < args.five_band_share else 4
        bands, resistance_ohms, tolerance = create_band_sequence(random_generator, band_count)
        rgb_image = render_synthetic_resistor(bands, random_generator, numpy_generator)
        filename = f"resistor_{index + 1:05d}.jpg"
        Image.fromarray(rgb_image).save(images_directory / filename, format="JPEG", quality=94)

        row = {
            "image": f"images/{filename}",
            "split": split_by_index[index],
            "band_count": band_count,
            "band_1": bands[0],
            "band_2": bands[1],
            "band_3": bands[2] if len(bands) > 2 else "",
            "band_4": bands[3] if len(bands) > 3 else "",
            "band_5": bands[4] if len(bands) > 4 else "",
            "resistance_ohms": f"{resistance_ohms:.6g}",
            "tolerance_percent": f"{tolerance:g}",
            "source": "synthetic",
        }
        metadata_rows.append(row)

    labels_path = args.output / "labels.csv"
    fieldnames = [
        "image", "split", "band_count", "band_1", "band_2", "band_3",
        "band_4", "band_5", "resistance_ohms", "tolerance_percent", "source",
    ]
    with labels_path.open("w", encoding="utf-8", newline="") as labels_file:
        writer = csv.DictWriter(labels_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_rows)

    print(f"Generated {args.count} synthetic images in: {images_directory}")
    print(f"Labels written to: {labels_path}")
    print("Important: synthetic images are for demos and pipeline tests, not a replacement for real photos.")

if __name__ == "__main__":
    main()
# endregion
