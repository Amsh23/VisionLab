"""Roboflow hosted-inference response normalization and safe error summaries."""
from __future__ import annotations

import re
from typing import Any


class HostedInferenceResponseError(ValueError):
    """Raised when a hosted detector returned an unrecognized response schema."""


_DETECTION_KEYS = {"class", "class_name", "confidence", "x", "y", "width", "height"}
_ENVELOPE_KEYS = ("result", "results", "output", "outputs", "data", "response")


def normalize_detection_predictions(payload: Any) -> list[dict]:
    """Return a flat list of object detections from standard Roboflow response envelopes.

    Standard object-detection responses look like {"predictions": [...]}.
    Some clients wrap one response in a single-item list or a "result"/"outputs"
    envelope. Empty predictions are valid; error payloads and unknown schemas are
    surfaced instead of being misreported as a successful zero-detection result.
    """
    if isinstance(payload, dict):
        if "error" in payload and payload.get("error"):
            raise HostedInferenceResponseError("The inference service returned an error payload.")

        if "predictions" in payload:
            predictions = payload["predictions"]
            if predictions is None:
                return []
            if isinstance(predictions, list):
                if all(isinstance(item, dict) for item in predictions):
                    return predictions
                raise HostedInferenceResponseError(
                    "The response 'predictions' field contains non-object entries."
                )
            if isinstance(predictions, dict):
                # Some client wrappers nest the standard prediction envelope.
                try:
                    return normalize_detection_predictions(predictions)
                except HostedInferenceResponseError:
                    raise HostedInferenceResponseError(
                        "The response 'predictions' field is an unsupported object."
                    ) from None
            raise HostedInferenceResponseError(
                f"The response 'predictions' field has type {type(predictions).__name__}, not a list."
            )

        for key in _ENVELOPE_KEYS:
            if key in payload and isinstance(payload[key], (dict, list)):
                try:
                    return normalize_detection_predictions(payload[key])
                except HostedInferenceResponseError:
                    continue

        keys = ", ".join(sorted(str(key) for key in payload.keys())[:12]) or "<none>"
        raise HostedInferenceResponseError(
            f"No object-detection predictions field was found (response keys: {keys})."
        )

    if isinstance(payload, list):
        if not payload:
            return []
        if len(payload) == 1 and isinstance(payload[0], (dict, list)):
            try:
                return normalize_detection_predictions(payload[0])
            except HostedInferenceResponseError:
                # A one-item list may itself be a single detection record.
                pass
        if all(isinstance(item, dict) for item in payload):
            if all(_DETECTION_KEYS.intersection(item.keys()) for item in payload):
                return payload
        raise HostedInferenceResponseError(
            f"The inference client returned a list with an unsupported shape ({len(payload)} item(s))."
        )

    # Support Pydantic/model response objects without binding the application to one SDK version.
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
        message = message.replace(api_key, "[redacted]")
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
    message = " ".join(message.split())
    if len(message) > 200:
        message = message[:197] + "..."

    if status_code:
        return f"HTTP {status_code}" + (f": {message}" if message else "")
    return message or type(error).__name__
