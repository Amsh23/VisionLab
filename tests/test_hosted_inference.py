"""Unit tests for Roboflow hosted-inference response normalization."""
import unittest

from hosted_inference import (
    HostedInferenceResponseError,
    normalize_detection_predictions,
    safe_error_summary,
)


class DetectionResponseTests(unittest.TestCase):
    def test_standard_detection_response(self):
        detections = [{
            "class": "Resistor", "confidence": 0.88,
            "x": 30, "y": 20, "width": 10, "height": 8,
        }]
        self.assertEqual(normalize_detection_predictions({"predictions": detections}), detections)

    def test_wrapped_single_result(self):
        detections = [{
            "class": "ESP32", "confidence": 0.91,
            "x": 30, "y": 20, "width": 10, "height": 8,
        }]
        self.assertEqual(
            normalize_detection_predictions([{"result": {"predictions": detections}}]),
            detections,
        )

    def test_empty_predictions_are_valid(self):
        self.assertEqual(normalize_detection_predictions({"predictions": []}), [])
        self.assertEqual(normalize_detection_predictions({"predictions": None}), [])

    def test_corner_coordinates_are_normalized(self):
        payload = {"predictions": [{
            "label": "ESP32", "score": 91,
            "x1": 10, "y1": 20, "x2": 50, "y2": 60,
        }]}
        self.assertEqual(normalize_detection_predictions(payload), [{
            "class": "ESP32", "confidence": 0.91,
            "x": 30.0, "y": 40.0, "width": 40.0, "height": 40.0,
        }])

    def test_nested_bbox_object_is_normalized(self):
        payload = {"predictions": [{
            "class_name": "Resistor", "probability": 0.8,
            "bbox": {"xmin": 4, "ymin": 6, "xmax": 20, "ymax": 26},
        }]}
        self.assertEqual(normalize_detection_predictions(payload), [{
            "class": "Resistor", "confidence": 0.8,
            "x": 12.0, "y": 16.0, "width": 16.0, "height": 20.0,
        }])

    def test_malformed_predictions_are_not_reported_as_empty(self):
        with self.assertRaisesRegex(HostedInferenceResponseError, "bounding-box coordinates"):
            normalize_detection_predictions({
                "predictions": [{"class": "Resistor", "confidence": 0.8}]
            })

    def test_error_payload_is_not_silently_reported_as_no_detections(self):
        with self.assertRaises(HostedInferenceResponseError):
            normalize_detection_predictions({"error": "model unavailable"})

    def test_unknown_payload_reports_available_keys(self):
        with self.assertRaisesRegex(HostedInferenceResponseError, "response keys: image, time"):
            normalize_detection_predictions({"image": {}, "time": 0.1})

    def test_api_key_is_redacted_from_error(self):
        cases = [
            ("request failed api_key=secret123", "secret123"),
            ("request failed https://service.invalid/?api_key=secret123", "secret123"),
            ("request failed with token secret+/123", "secret+/123"),
        ]
        for message, api_key in cases:
            with self.subTest(message=message):
                safe = safe_error_summary(ValueError(message), api_key)
                self.assertNotIn(api_key, safe)
                self.assertIn("[redacted]", safe)


if __name__ == "__main__":
    unittest.main()
