"""Live webcam detection for local VisionLab YOLO checkpoints.

Local checkpoints are preferred for low-latency inference. If none are
available, the module can use the configured Roboflow hosted detectors.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import av
import cv2
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

from hosted_inference import request_roboflow_detections

HOSTED_MODELS = (
    ("electronics-components-sveck/7", "Electronics"),
    ("iot-components-learning-ogliq/1", "IoT Components"),
)


def render_live_webcam_detection(
    model_root: Path,
    load_local_yolo_model: Callable[[str], object],
    api_key: str = "",
) -> None:
    """Render a real-time webcam with local model inference and hosted fallback."""
    model_root = Path(model_root)
    # Keep live CPU inference responsive by using one detector per session.
    # Prefer the general electronics model, then fall back to an IoT checkpoint.
    preferred_checkpoints = [
        (model_root / "components.pt", "Electronics"),
        (model_root / "iot_components.pt", "IoT Components"),
        (model_root / "iot_boards.pt", "IoT Boards"),
    ]
    checkpoint_paths = [
        (path, label) for path, label in preferred_checkpoints if path.is_file()
    ][:1]

    if not checkpoint_paths and not api_key:
        st.error(
            "No local detector checkpoints were found and ROBOFLOW_API_KEY is not configured. "
            "Run the model-training workflow or configure the Streamlit secret."
        )
        return

    if checkpoint_paths:
        try:
            from ultralytics import YOLO  # noqa: F401
        except ImportError:
            # Hosted Streamlit deployments may intentionally omit Ultralytics to
            # keep the OpenCV runtime light. Prefer hosted inference in that case.
            checkpoint_paths = []
            if not api_key:
                st.error(
                    "Local model files exist, but Ultralytics is not installed and "
                    "ROBOFLOW_API_KEY is unavailable. Install Ultralytics locally or configure the Streamlit secret."
                )
                return

    confidence = st.slider(
        "Detection confidence",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05,
        key="visionlab_live_confidence",
    )
    model_signature = ", ".join(label for _, label in checkpoint_paths)
    if checkpoint_paths:
        st.caption(f"Live local inference enabled: {model_signature}. Models load once per webcam session.")
    else:
        st.caption(
            "No local checkpoints are available. Live detections will use Roboflow hosted inference, "
            "which may have higher latency and requires an internet connection."
        )

    class LiveWebcamProcessor(VideoProcessorBase):
        def __init__(self) -> None:
            self.error = ""
            self.models = []
            for path, label in checkpoint_paths:
                self.models.append((load_local_yolo_model(str(path)), label))
            # Run local YOLO on alternate frames and reuse boxes in between.
            # This nearly halves CPU inference work while keeping the preview live.
            self.frame_index = 0
            self.cached_local_boxes = None
            self.last_hosted_request = 0.0
            self.hosted_detections = []
            self.hosted_error = ""

        @staticmethod
        def _draw_box(image, x1, y1, x2, y2, label, confidence_value, color):
            height, width = image.shape[:2]
            x1 = max(0, min(width - 1, int(x1)))
            x2 = max(x1 + 1, min(width - 1, int(x2)))
            y1 = max(0, min(height - 1, int(y1)))
            y2 = max(y1 + 1, min(height - 1, int(y2)))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            text = f"{label} {confidence_value:.0%}"
            text_y = max(20, y1 - 7)
            cv2.putText(
                image, text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, color, 2, cv2.LINE_AA,
            )

        def recv(self, frame):
            try:
                image = frame.to_ndarray(format="bgr24")
                annotated = image.copy()
                palette = [(30, 200, 80), (230, 140, 35), (210, 70, 210)]

                if self.models:
                    self.frame_index += 1
                    # Infer on the first frame, then every other frame.
                    if self.cached_local_boxes is None or self.frame_index % 2 == 1:
                        cached = []
                        for model_index, (model, model_label) in enumerate(self.models):
                            result = model.predict(
                                source=image,
                                conf=float(confidence),
                                imgsz=320,
                                max_det=30,
                                iou=0.5,
                                verbose=False,
                            )[0]
                            boxes = result.boxes
                            if boxes is None:
                                continue
                            names = result.names
                            color = palette[model_index % len(palette)]
                            for box in boxes:
                                class_id = int(box.cls[0].item())
                                class_name = (
                                    names.get(class_id, str(class_id))
                                    if isinstance(names, dict)
                                    else names[class_id]
                                )
                                score = float(box.conf[0].item())
                                x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy().tolist()
                                cached.append((
                                    x1, y1, x2, y2,
                                    f"{model_label}: {class_name}", score, color,
                                ))
                        self.cached_local_boxes = cached

                    for x1, y1, x2, y2, label, score, color in self.cached_local_boxes:
                        self._draw_box(annotated, x1, y1, x2, y2, label, score, color)
                else:
                    now = time.monotonic()
                    if now - self.last_hosted_request >= 1.5:
                        self.last_hosted_request = now
                        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                        if ok:
                            detections = []
                            errors = []
                            for model_id, model_label in HOSTED_MODELS:
                                try:
                                    predictions = request_roboflow_detections(
                                        encoded.tobytes(), model_id, api_key, confidence,
                                        timeout=8.0,
                                    )
                                    for item in predictions:
                                        item["model_label"] = model_label
                                        detections.append(item)
                                except Exception as error:
                                    errors.append(f"{model_label}: {type(error).__name__}")
                            self.hosted_detections = detections
                            self.hosted_error = "; ".join(errors)

                    for item in self.hosted_detections:
                        score = float(item["confidence"])
                        if score < float(confidence):
                            continue
                        center_x = float(item["x"])
                        center_y = float(item["y"])
                        box_width = float(item["width"])
                        box_height = float(item["height"])
                        x1 = center_x - box_width / 2
                        y1 = center_y - box_height / 2
                        x2 = center_x + box_width / 2
                        y2 = center_y + box_height / 2
                        self._draw_box(
                            annotated, x1, y1, x2, y2,
                            f"{item.get('model_label', 'Component')}: {item['class']}",
                            score, palette[0 if item.get("model_label") == "Electronics" else 1],
                        )
                return av.VideoFrame.from_ndarray(annotated, format="bgr24")
            except Exception as error:
                self.error = f"{type(error).__name__}: {error}"
                return frame

    camera_ctx = webrtc_streamer(
        key="visionlab_live_component_detection",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=LiveWebcamProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    processor = camera_ctx.video_processor
    if processor is not None and getattr(processor, "error", ""):
        st.warning(f"Live detection issue: {processor.error}")
    if processor is not None and getattr(processor, "hosted_error", ""):
        st.caption(f"Hosted detector warning: {processor.hosted_error}")
