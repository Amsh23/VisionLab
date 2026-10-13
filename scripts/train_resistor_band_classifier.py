"""Train a lightweight resistor-band color classifier using augmented labeled color patches.

Synthetic patches are a fallback training source when a real annotated band-color
dataset cannot be downloaded. They are useful for initialization, but do not
replace validation on real photographs.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from resistor_band_classifier import (
    BAND_COLOR_NAMES,
    BAND_COLOR_RGB,
    FEATURE_VERSION,
    extract_color_patch_features,
)


def make_augmented_patch(color_name: str, rng: np.random.Generator) -> np.ndarray:
    """Render a band-like patch with randomized illumination, camera response, and noise."""
    height = int(rng.integers(20, 49))
    width = int(rng.integers(8, 31))
    base = np.asarray(BAND_COLOR_RGB[color_name], dtype=np.float32)
    body = np.asarray([205, 177, 137], dtype=np.float32)

    # Include a variable amount of resistor body at each side to mimic small band crops.
    coverage = float(rng.uniform(0.64, 1.0))
    patch = np.broadcast_to(body, (height, width, 3)).copy()
    band_width = max(3, min(width, int(round(width * coverage))))
    start = int(rng.integers(0, max(1, width - band_width + 1)))
    patch[:, start:start + band_width] = base

    # Apply gradients, soft specular streaks, channel-wise white balance, and gamma.
    yy = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None, None]
    xx = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :, None]
    light = rng.uniform(0.72, 1.24) + yy * rng.uniform(-0.16, 0.16) + xx * rng.uniform(-0.10, 0.10)
    channel_gain = rng.uniform(0.82, 1.18, size=(1, 1, 3))
    patch = patch * light * channel_gain
    if rng.random() < 0.35:
        highlight_center = float(rng.uniform(-0.6, 0.6))
        highlight_width = float(rng.uniform(0.08, 0.35))
        highlight = np.exp(-((xx - highlight_center) ** 2) / (2 * highlight_width ** 2)) * rng.uniform(5, 30)
        patch += highlight
    patch = np.clip(patch, 0, 255)

    gamma = float(rng.uniform(0.78, 1.24))
    patch = 255.0 * np.power(patch / 255.0, gamma)
    patch += rng.normal(0.0, rng.uniform(0.4, 8.0), patch.shape)
    patch = np.clip(patch, 0, 255).astype(np.uint8)

    # A small hue and saturation shift mimics camera white balance and color response.
    if rng.random() < 0.60:
        hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
        hue_delta = int(rng.integers(-5, 6))
        hsv[:, :, 0] = ((hsv[:, :, 0].astype(np.int16) + hue_delta) % 180).astype(np.uint8)
        sat_scale = float(rng.uniform(0.78, 1.20))
        hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(np.float32) * sat_scale, 0, 255).astype(np.uint8)
        patch = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    if rng.random() < 0.20:
        small_w = max(4, width // 2)
        small_h = max(8, height // 2)
        patch = cv2.resize(patch, (small_w, small_h), interpolation=cv2.INTER_AREA)
        patch = cv2.resize(patch, (width, height), interpolation=cv2.INTER_LINEAR)
    return patch


def build_samples(samples_per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    features = []
    labels = []
    for color_name in BAND_COLOR_NAMES:
        for _ in range(samples_per_class):
            patch = make_augmented_patch(color_name, rng)
            features.append(extract_color_patch_features(patch))
            labels.append(color_name)
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=object)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the resistor color-band classifier.")
    parser.add_argument("--samples-per-class", type=int, default=1400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("models/resistor_band_classifier.joblib"))
    args = parser.parse_args()
    if args.samples_per_class < 100:
        parser.error("--samples-per-class must be at least 100.")

    X, y = build_samples(args.samples_per_class, args.seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=args.seed, stratify=y
    )
    model = ExtraTreesClassifier(
        n_estimators=220,
        max_depth=22,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=args.seed,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, predictions))
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "classes": list(model.classes_),
        "feature_version": FEATURE_VERSION,
        "test_accuracy_synthetic": accuracy,
        "samples_per_class": args.samples_per_class,
        "source": "synthetic augmented color-band patches",
    }
    joblib.dump(payload, output, compress=3)
    report_path = output.with_suffix(".metrics.json")
    report_path.write_text(
        json.dumps({
            "source": payload["source"],
            "samples_per_class": args.samples_per_class,
            "training_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "test_accuracy_synthetic": accuracy,
            "classification_report": report,
            "warning": "Synthetic test accuracy is not a real-photo accuracy estimate.",
        }, indent=2),
        encoding="utf-8",
    )
    print(f"Model saved: {output}")
    print(f"Synthetic held-out accuracy: {accuracy:.4f}")
    print(f"Metrics: {report_path}")
    print("This score measures synthetic-patch classification only, not performance on real resistor photos.")


if __name__ == "__main__":
    main()
