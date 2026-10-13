"""Unit tests for normalizing Roboflow inference response envelopes."""
import unittest

from hosted_inference import (
    HostedInferenceResponseError,
    normalize_detection_predictions,
    safe_error_summary,
)


class DetectionResponseTests(unittest.TestCase):
    def test_standard_detection_response(self):
        detections = [{"class": "Resistor", "confidence": 0.88, "x": 30, "y": 20, "width": 10, "height": 8}]
        self.assertEqual(normalize_detection_predictions({"predictions": detections}), detections)

    def test_wrapped_single_result(self):
        detections = [{"class": "ESP32", "confidence": 0.91, "x": 30, "y": 20, "width": 10, "height": 8}]
        self.assertEqual(normalize_detection_predictions([{"result": {"predictions": detections}}]), detections)

    def test_empty_predictions_are_valid(self):
        self.assertEqual(normalize_detection_predictions({"predictions": []}), [])
        self.assertEqual(normalize_detection_predictions({"predictions": None}), [])

    def test_error_payload_is_not_silently_reported_as_no_detections(self):
        with self.assertRaises(HostedInferenceResponseError):
            normalize_detection_predictions({"error": "model unavailable"})

    def test_unknown_payload_reports_available_keys(self):
        with self.assertRaisesRegex(HostedInferenceResponseError, "response keys: image, time"):
            normalize_detection_predictions({"image": {}, "time": 0.1})

    def test_api_key_is_redacted_from_error(self):
        message = safe_error_summary(ValueError("request failed api_key=secret123"), "secret123")
        self.assertNotIn("secret123", message)
        self.assertIn("[redacted]", message)


if __name__ == "__main__":
    unittest.main()
