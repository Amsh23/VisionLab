"""Roboflow hosted-inference response normalization and safe error summaries."""
from __future__ import annotations

import math
import re
from typing import Any, Mapping
from urllib.parse import quote, quote_plus

import requests


class HostedInferenceResponseError(ValueError):
    """Raised when a hosted detector returned an unrecognized response schema."""


_ENVELOPE_KEYS = ("result", "results", "output", "outputs", "data", "response")
_LABEL_KEYS = ("class", "class_name", "label", "name")
_CONFIDENCE_KEYS = ("confidence", "score", "probability")


def _number(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise HostedInferenceResponseError(
            f"Detection field '{field}' is not numeric."
        ) from error
    if not math.isfinite(number):
        raise HostedInferenceResponseError(
            f"Detection field '{field}' is not a finite number."
        )
    return number


def _box_from_mapping(box: Mapping[str, Any], *, center_xywh: bool) -> tuple[float, float, float, float] | None:
    """Read either center-based xywh coordinates or corner-based xyxy coordinates."""
    if all(key in box for key in ("x", "y", "width", "height")) and center_xywh:
        return (
            _number(box["x"], "x"),
            _number(box["y"], "y"),
            _number(box["width"], "width"),
            _number(box["height"], "height"),
        )

    coordinate_sets = (
        ("x1", "y1", "x2", "y2"),
        ("xmin", "ymin", "xmax", "ymax"),
        ("left", "top", "right", "bottom"),
    )
    for left_key, top_key, right_key, bottom_key in coordinate_sets:
        if all(key in box for key in (left_key, top_key, right_key, bottom_key)):
            left = _number(box[left_key], left_key)
            top = _number(box[top_key], top_key)
            right = _number(box[right_key], right_key)
            bottom = _number(box[bottom_key], bottom_key)
            return ((left + right) / 2, (top + bottom) / 2, right - left, bottom - top)

    if all(key in box for key in ("x", "y", "width", "height")):
        # Nested boxes from some libraries use a top-left origin for xywh.
        left = _number(box["x"], "x")
        top = _number(box["y"], "y")
        width = _number(box["width"], "width")
        height = _number(box["height"], "height")
        return (left + width / 2, top + height / 2, width, height)
    return None


def _normalize_detection(item: Any) -> dict[str, Any]:
    """Convert common detector record variants to Roboflow's center-based xywh."""
    if not isinstance(item, dict):
        raise HostedInferenceResponseError("A prediction entry is not an object.")

    record = dict(item)
    for nested_key in ("detection", "prediction"):
        nested = record.get(nested_key)
        if isinstance(nested, dict):
            record = {**record, **nested}

    label = next(
        (str(record[key]).strip() for key in _LABEL_KEYS
         if record.get(key) is not None and str(record[key]).strip()),
        "",
    )
    if not label and record.get("class_id") is not None:
        label = str(record["class_id"])
    if not label:
        raise HostedInferenceResponseError(
            "A prediction is missing its class label."
        )

    raw_confidence = next(
        (record[key] for key in _CONFIDENCE_KEYS if record.get(key) is not None),
        None,
    )
    if raw_confidence is None:
        raise HostedInferenceResponseError(
            f"Prediction '{label}' is missing its confidence score."
        )
    confidence = _number(raw_confidence, "confidence")
    # Some wrappers serialize percentages rather than probabilities.
    if 1 < confidence <= 100:
        confidence /= 100
    if not 0 <= confidence <= 1:
        raise HostedInferenceResponseError(
            f"Prediction '{label}' has confidence outside the valid 0–1 range."
        )

    box = None
    if all(key in record for key in ("x", "y", "width", "height")):
        box = _box_from_mapping(record, center_xywh=True)
    else:
        for key in ("bounding_box", "bbox", "box"):
            candidate = record.get(key)
            if isinstance(candidate, dict):
                box = _box_from_mapping(candidate, center_xywh=True)
            elif isinstance(candidate, (list, tuple)) and len(candidate) == 4:
                # Array-form boxes are conventionally corner coordinates (xyxy).
                left, top, right, bottom = (
                    _number(value, key) for value in candidate
                )
                box = ((left + right) / 2, (top + bottom) / 2, right - left, bottom - top)
            if box is not None:
                break
        if box is None:
            box = _box_from_mapping(record, center_xywh=False)

    if box is None:
        raise HostedInferenceResponseError(
            f"Prediction '{label}' is missing supported bounding-box coordinates."
        )
    center_x, center_y, width, height = box
    if width <= 0 or height <= 0:
        raise HostedInferenceResponseError(
            f"Prediction '{label}' has a zero or negative bounding-box size."
        )

    return {
        "class": label,
        "confidence": confidence,
        "x": center_x,
        "y": center_y,
        "width": width,
        "height": height,
    }


def normalize_detection_predictions(payload: Any) -> list[dict]:
    """Return canonical detections from common Roboflow/API response envelopes.

    The app consumes class, confidence, and center-based x/y/width/height. In
    addition to Roboflow's standard response, this normalizes nested wrappers,
    common alternative label/score names, corner-form boxes, and percent scores.
    Empty prediction arrays are valid. Malformed prediction entries are not
    silently treated as a successful result containing zero detections.
    """
    if isinstance(payload, dict):
        if payload.get("error"):
            raise HostedInferenceResponseError(
                "The inference service returned an error payload."
            )

        if "predictions" in payload:
            predictions = payload["predictions"]
            if predictions is None:
                return []
            if isinstance(predictions, list):
                return [_normalize_detection(item) for item in predictions]
            if isinstance(predictions, dict):
                return normalize_detection_predictions(predictions)
            raise HostedInferenceResponseError(
                f"The response 'predictions' field has type {type(predictions).__name__}, not a list or object."
            )

        nested_errors = []
        for key in _ENVELOPE_KEYS:
            nested = payload.get(key)
            if isinstance(nested, (dict, list)):
                try:
                    return normalize_detection_predictions(nested)
                except HostedInferenceResponseError as error:
                    nested_errors.append(f"{key}: {error}")

        if any(key in payload for key in (*_LABEL_KEYS, "class_id")):
            return [_normalize_detection(payload)]

        keys = ", ".join(sorted(str(key) for key in payload.keys())[:12]) or "<none>"
        detail = f" (nested response details: {'; '.join(nested_errors[:2])})" if nested_errors else ""
        raise HostedInferenceResponseError(
            f"No object-detection predictions field was found (response keys: {keys}){detail}."
        )

    if isinstance(payload, list):
        if not payload:
            return []
        if len(payload) == 1 and isinstance(payload[0], (dict, list)):
            return normalize_detection_predictions(payload[0])

        if all(isinstance(item, dict) for item in payload):
            try:
                return [_normalize_detection(item) for item in payload]
            except HostedInferenceResponseError as direct_error:
                # Batch clients sometimes return one response envelope per image.
                flattened = []
                try:
                    for item in payload:
                        flattened.extend(normalize_detection_predictions(item))
                    return flattened
                except HostedInferenceResponseError:
                    raise direct_error from None

        raise HostedInferenceResponseError(
            f"The inference client returned a list with an unsupported shape ({len(payload)} item(s))."
        )

    # Support Pydantic/model response objects without tying the app to one SDK version.
    for converter_name in ("model_dump", "dict", "to_dict"):
        converter = getattr(payload, converter_name, None)
        if callable(converter):
            try:
                converted = converter()
            except Exception:
                continue
            if converted is not payload:
                return normalize_detection_predictions(converted)

    raise HostedInferenceResponseError(
        f"The inference client returned {type(payload).__name__}, not a JSON response."
    )


def safe_error_summary(error: Exception, api_key: str = "") -> str:
    """Produce a useful, bounded error without accidentally printing API credentials."""
    status_code = getattr(getattr(error, "response", None), "status_code", None)
    status_code = status_code or getattr(error, "status_code", None)
    message = str(error).strip()

    if api_key:
        for candidate in {api_key, quote(api_key, safe=""), quote_plus(api_key)}:
            if candidate:
                message = message.replace(candidate, "[redacted]")
    message = re.sub(
        r"(?i)(api[_-]?key\s*[=:]\s*)[^&\s,;]+",
        r"\1[redacted]",
        message,
    )
    message = re.sub(
        r"(?i)([?&]api_key=)[^&\s]+",
        r"\1[redacted]",
        message,
    )
    message = re.sub(r"(?i)(authorization:\s*bearer\s+)[^\s,;]+", r"\1[redacted]", message)
    message = " ".join(message.split())
    if len(message) > 240:
        message = message[:237] + "..."

    if status_code:
        return f"HTTP {status_code}" + (f": {message}" if message else "")
    return message or type(error).__name__



def request_roboflow_detections(
    image_bytes: bytes,
    model_id: str,
    api_key: str,
    confidence_threshold: float,
    *,
    timeout: float = 60.0,
) -> list[dict]:
    """Request object detections from Roboflow's hosted HTTP API.

    The runtime intentionally uses the lightweight HTTP API rather than the
    inference SDK, whose dependency tree can install GUI OpenCV alongside the
    headless wheel needed by Streamlit Cloud.
    """
    if not isinstance(image_bytes, (bytes, bytearray, memoryview)) or not image_bytes:
        raise ValueError("The inference image must be non-empty JPEG bytes.")
    if not isinstance(model_id, str) or not model_id.strip() or "/" not in model_id:
        raise ValueError("The Roboflow model ID must include a model slug and version.")
    api_key = str(api_key).strip()
    if not api_key:
        raise ValueError("ROBOFLOW_API_KEY is missing.")

    threshold = _number(confidence_threshold, "confidence_threshold")
    if not 0 <= threshold <= 1:
        raise ValueError("Confidence threshold must be between 0 and 1.")

    response = requests.post(
        f"https://detect.roboflow.com/{model_id}",
        params={
            "api_key": api_key,
            "confidence": max(10, min(100, int(round(threshold * 100)))),
            "overlap": 35,
        },
        files={"file": ("image.jpg", bytes(image_bytes), "image/jpeg")},
        timeout=timeout,
    )
    response.raise_for_status()
    try:
        payload = response.json()
    except ValueError as error:
        raise HostedInferenceResponseError(
            "Roboflow returned a non-JSON inference response."
        ) from error
    return normalize_detection_predictions(payload)
