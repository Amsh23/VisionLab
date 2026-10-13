"""Unit tests for Roboflow hosted-inference response normalization."""
import unittest
from unittest.mock import Mock, patch

import requests

from hosted_inference import (
    HostedInferenceResponseError,
    normalize_detection_predictions,
    request_roboflow_detections,
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


class HostedHttpRequestTests(unittest.TestCase):
    def test_request_sends_expected_parameters_and_normalizes_result(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "predictions": [{
                "class": "ESP32", "confidence": 0.9,
                "x": 50, "y": 40, "width": 30, "height": 20,
            }]
        }
        jpeg = b"fake-jpeg-bytes"
        with patch("hosted_inference.requests.post", return_value=response) as post:
            detections = request_roboflow_detections(
                jpeg,
                "iot-components-learning-ogliq/1",
                "not-a-real-key",
                0.25,
            )

        self.assertEqual(detections[0]["class"], "ESP32")
        self.assertEqual(post.call_count, 1)
        args, kwargs = post.call_args
        self.assertEqual(
            args[0],
            "https://detect.roboflow.com/iot-components-learning-ogliq/1",
        )
        self.assertEqual(kwargs["params"]["api_key"], "not-a-real-key")
        self.assertEqual(kwargs["params"]["confidence"], 25)
        self.assertEqual(kwargs["params"]["overlap"], 35)
        self.assertEqual(kwargs["files"]["file"], ("image.jpg", jpeg, "image/jpeg"))
        self.assertEqual(kwargs["timeout"], 60.0)

    def test_http_errors_are_propagated_for_safe_ui_reporting(self):
        response = Mock()
        response.raise_for_status.side_effect = requests.HTTPError(
            "401 Client Error", response=Mock(status_code=401)
        )
        with patch("hosted_inference.requests.post", return_value=response):
            with self.assertRaises(requests.HTTPError):
                request_roboflow_detections(
                    b"jpeg", "electronics-components-sveck/7", "secret", 0.25
                )

    def test_non_json_http_response_is_rejected(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.side_effect = ValueError("not JSON")
        with patch("hosted_inference.requests.post", return_value=response):
            with self.assertRaisesRegex(HostedInferenceResponseError, "non-JSON"):
                request_roboflow_detections(
                    b"jpeg", "electronics-components-sveck/7", "secret", 0.25
                )

    def test_invalid_confidence_does_not_send_request(self):
        with patch("hosted_inference.requests.post") as post:
            with self.assertRaisesRegex(ValueError, "between 0 and 1"):
                request_roboflow_detections(
                    b"jpeg", "electronics-components-sveck/7", "secret", 1.5
                )
        post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
