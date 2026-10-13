"""Lightweight feature extraction and loading for the resistor band-color classifier."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import joblib
import numpy as np

BAND_COLOR_NAMES = [
    "Black", "Brown", "Red", "Orange", "Yellow", "Green",
    "Blue", "Violet", "Gray", "White", "Gold", "Silver",
]
BAND_COLOR_RGB = {
    "Black": (35, 35, 35),
    "Brown": (125, 75, 45),
    "Red": (215, 45, 45),
    "Orange": (235, 125, 30),
    "Yellow": (245, 215, 45),
    "Green": (35, 145, 70),
    "Blue": (40, 95, 210),
    "Violet": (135, 65, 170),
    "Gray": (135, 135, 135),
    "White": (242, 240, 225),
    "Gold": (190, 155, 70),
    "Silver": (190, 195, 200),
}
FEATURE_VERSION = 1


def extract_color_patch_features(patch_rgb: np.ndarray) -> np.ndarray:
    """Compute color statistics from an RGB crop that contains one resistor band."""
    patch = np.asarray(patch_rgb)
    if patch.ndim == 1 and patch.shape[0] == 3:
        patch = patch.reshape(1, 1, 3)
    elif patch.ndim == 2 and patch.shape[-1] == 3:
        patch = patch.reshape(1, patch.shape[0], 3)
    if patch.ndim != 3 or patch.shape[2] < 3:
        raise ValueError("Expected an RGB patch with shape (height, width, 3).")
    patch = np.clip(patch[..., :3], 0, 255).astype(np.uint8, copy=False)
    if patch.size == 0:
        raise ValueError("The band-color patch is empty.")

    pixels = patch.reshape(-1, 3).astype(np.float32)
    hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(np.float32)
    lab = cv2.cvtColor(patch, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)

    # Keep a fixed feature order for both training and application inference.
    features = np.concatenate([
        np.mean(pixels, axis=0),
        np.median(pixels, axis=0),
        np.std(pixels, axis=0),
        np.mean(hsv, axis=0),
        np.median(hsv, axis=0),
        np.mean(lab, axis=0),
    ])
    return features.astype(np.float32)


@lru_cache(maxsize=1)
def load_trained_band_classifier():
    """Load the small trained classifier once; return None if no checkpoint exists."""
    model_path = Path(__file__).resolve().parent / "models" / "resistor_band_classifier.joblib"
    if not model_path.is_file():
        return None
    payload = joblib.load(model_path)
    if not isinstance(payload, dict) or "model" not in payload or "classes" not in payload:
        raise ValueError("The resistor band classifier checkpoint has an unsupported format.")
    if payload.get("feature_version") != FEATURE_VERSION:
        raise ValueError("The resistor band classifier uses an incompatible feature version.")
    return payload


def predict_band_color(patch_rgb: np.ndarray, minimum_probability: float = 0.38) -> tuple[str, float] | None:
    """Predict the color of a candidate band crop, or defer to the heuristic."""
    payload = load_trained_band_classifier()
    if payload is None:
        return None
    features = extract_color_patch_features(patch_rgb).reshape(1, -1)
    model = payload["model"]
    probabilities = model.predict_proba(features)[0]
    best_index = int(np.argmax(probabilities))
    confidence = float(probabilities[best_index])
    if confidence < float(minimum_probability):
        return None
    return str(payload["classes"][best_index]), confidence
