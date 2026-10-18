# region 01: Imports
from __future__ import annotations

from collections import Counter
from io import BytesIO
from pathlib import Path
import csv
import hashlib
import importlib.util

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

from hosted_inference import request_roboflow_detections, safe_error_summary

try:
    from resistor_band_classifier import predict_band_color, load_trained_band_classifier
except Exception:
    predict_band_color = None
    load_trained_band_classifier = None
# endregion


# region 02: Resistor Color-Code Reference
DIGIT_COLORS = [
    "Black", "Brown", "Red", "Orange", "Yellow",
    "Green", "Blue", "Violet", "Gray", "White",
]
DIGIT_VALUES = {color: value for value, color in enumerate(DIGIT_COLORS)}

MULTIPLIER_OHMS = {
    "Silver": 0.01, "Gold": 0.1, "Black": 1, "Brown": 10,
    "Red": 100, "Orange": 1_000, "Yellow": 10_000,
    "Green": 100_000, "Blue": 1_000_000, "Violet": 10_000_000,
    "Gray": 100_000_000, "White": 1_000_000_000,
}
TOLERANCE_PERCENT = {
    "Brown": 1, "Red": 2, "Green": 0.5, "Blue": 0.25,
    "Violet": 0.1, "Gray": 0.05, "Gold": 5, "Silver": 10,
    "No band (20%)": 20,
}
TOLERANCE_OPTIONS = [
    "Brown", "Red", "Green", "Blue", "Violet", "Gray", "Gold", "Silver",
    "No band (20%)",
]
BAND_COLORS_RGB = {
    "Black": (35, 35, 35), "Brown": (125, 75, 45), "Red": (215, 45, 45),
    "Orange": (235, 125, 30), "Yellow": (245, 215, 45), "Green": (35, 145, 70),
    "Blue": (40, 95, 210), "Violet": (135, 65, 170), "Gray": (135, 135, 135),
    "White": (242, 240, 225), "Gold": (190, 155, 70), "Silver": (190, 195, 200),
}
# endregion


# region 03: Resistor-Reading Helpers
def format_resistance(ohms: float) -> str:
    """Format resistance using a readable SI unit."""
    if abs(ohms) < 1:
        return f"{ohms:.3g} Ω"
    for unit, factor in (("GΩ", 1e9), ("MΩ", 1e6), ("kΩ", 1e3)):
        if abs(ohms) >= factor:
            return f"{ohms / factor:.3g} {unit}"
    return f"{ohms:.3g} Ω"


def decode_band_sequence(bands: list[str], band_count: int) -> dict:
    """Decode standard 4-band or 5-band resistor color codes."""
    if len(bands) != band_count:
        raise ValueError(f"Expected {band_count} bands, but received {len(bands)}.")

    significant_count = band_count - 2
    significant_colors = bands[:significant_count]
    invalid_digits = [color for color in significant_colors if color not in DIGIT_VALUES]
    if invalid_digits:
        raise ValueError(
            "The significant-digit bands cannot be Gold or Silver. "
            "Choose Black through White for the first "
            f"{significant_count} bands. Invalid: {', '.join(invalid_digits)}."
        )

    multiplier_color = bands[significant_count]
    if multiplier_color not in MULTIPLIER_OHMS:
        raise ValueError(f"{multiplier_color} is not a valid multiplier-band color.")

    tolerance_color = bands[-1]
    if tolerance_color not in TOLERANCE_PERCENT:
        raise ValueError(f"{tolerance_color} is not a supported tolerance-band color.")

    digits = "".join(str(DIGIT_VALUES[color]) for color in significant_colors)
    significant_value = int(digits)
    resistance = significant_value * MULTIPLIER_OHMS[multiplier_color]
    tolerance = TOLERANCE_PERCENT[tolerance_color]
    return {
        "digits": digits,
        "multiplier": MULTIPLIER_OHMS[multiplier_color],
        "ohms": resistance,
        "tolerance_percent": tolerance,
        "minimum_ohms": resistance * (1 - tolerance / 100),
        "maximum_ohms": resistance * (1 + tolerance / 100),
    }


def classify_hsv_pixels(hsv: np.ndarray) -> np.ndarray:
    """Classify HSV pixels into likely resistor-band colors.

    This is a heuristic, not a trained color classifier. Lighting and camera
    white balance can make neighboring colors overlap.
    """
    hue = hsv[:, :, 0].astype(np.int16)
    saturation = hsv[:, :, 1].astype(np.int16)
    value = hsv[:, :, 2].astype(np.int16)
    labels = np.full(hue.shape, "", dtype=object)

    labels[value < 42] = "Black"
    neutral = saturation < 42
    labels[neutral & (value < 85)] = "Black"
    labels[neutral & (value >= 85) & (value < 145)] = "Gray"
    labels[neutral & (value >= 145) & (value < 210)] = "Silver"
    labels[neutral & (value >= 210)] = "White"

    active = (saturation >= 42) & (value >= 42)
    labels[active & ((hue <= 7) | (hue >= 172)) & (value >= 45)] = "Red"
    labels[active & (hue >= 8) & (hue <= 20) & (value < 135)] = "Brown"
    labels[active & (hue >= 8) & (hue <= 20) & (value >= 135)] = "Orange"

    yellow = active & (hue > 20) & (hue <= 37) & (saturation >= 115) & (value >= 155)
    labels[yellow] = "Yellow"
    gold = active & (hue > 20) & (hue <= 39) & ~yellow & (value >= 82)
    labels[gold] = "Gold"

    labels[active & (hue >= 36) & (hue <= 84)] = "Green"
    labels[active & (hue >= 85) & (hue <= 130)] = "Blue"
    labels[active & (hue >= 131) & (hue <= 171)] = "Violet"
    return labels


def classify_hsv_color(hsv_values: np.ndarray, baseline_value: float = 0.0) -> str:
    """Classify the center pixels of one band, using local brightness as context.

    Hue alone is not enough for brown/orange/red or gold/yellow. Relative
    brightness against the nearby resistor body helps separate those colors.
    """
    hue, saturation, value = (float(part) for part in hsv_values)
    relative_value = value - float(baseline_value)

    if value < 48:
        return "Black"

    if saturation < 38:
        if value < 85:
            return "Black"
        if value < 145:
            return "Gray"
        if value < 218:
            return "Silver"
        return "White"

    if hue <= 4 or hue >= 176:
        return "Red"

    if hue <= 18:
        if value < 145 or relative_value < -22:
            return "Brown"
        return "Orange"

    if hue <= 39:
        if saturation >= 105 and (
            (value >= 205 and relative_value >= -10) or relative_value >= 28
        ):
            return "Yellow"
        return "Gold"

    if hue <= 85:
        return "Green"
    if hue <= 132:
        return "Blue"
    return "Violet"


def detect_band_runs(
    roi_rgb: np.ndarray,
    expected_count: int,
    *,
    use_trained_classifier: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Detect color bands from a resistor-body profile, not from band edges.

    The synthetic-trained patch classifier is disabled by default because its
    held-out score was measured on synthetic patches, not real camera photos.

    A narrow-band edge detector can mistake the two edges of one stripe for
    two separate bands and often misses gold/silver bands when their colors are
    close to the resistor body. This method estimates the body span, classifies
    each column from its median HSV color with respect to local body brightness,
    smooths single-column noise, and groups consecutive color labels into bands.
    """
    height, width = roi_rgb.shape[:2]
    if width < 25 or height < 8 or expected_count not in (4, 5):
        return [], []

    scan_top = max(0, int(round(height * 0.36)))
    scan_bottom = min(height, max(scan_top + 1, int(round(height * 0.64))))
    if scan_bottom - scan_top < 5:
        scan_top, scan_bottom = int(height * 0.25), int(height * 0.75)
    scan_rgb = roi_rgb[scan_top:scan_bottom]

    profile_rgb = np.median(scan_rgb.astype(np.float32), axis=0).astype(np.uint8)
    profile_lab = cv2.cvtColor(profile_rgb.reshape(1, width, 3), cv2.COLOR_RGB2LAB)[0].astype(np.float32)
    profile_hsv = cv2.cvtColor(profile_rgb.reshape(1, width, 3), cv2.COLOR_RGB2HSV)[0].astype(np.float32)

    # Detect the vertical resistor body while excluding thin metal leads. When
    # the crop already fills the frame with the body, fall back to the full crop.
    scan_lab = cv2.cvtColor(scan_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    edge_width = max(3, int(round(width * 0.035)))
    edge_pixels = np.concatenate(
        [
            scan_rgb[:, :edge_width, :].reshape(-1, 3),
            scan_rgb[:, -edge_width:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    background_rgb = np.median(edge_pixels, axis=0).astype(np.uint8)
    background_lab = cv2.cvtColor(background_rgb.reshape(1, 1, 3), cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)
    foreground_ratio = (
        np.linalg.norm(scan_lab - background_lab.reshape(1, 1, 3), axis=2) > 16.0
    ).mean(axis=0)
    body_mask = foreground_ratio >= 0.52

    body_gap = max(2, int(round(width * 0.02)))
    index = 0
    while index < width:
        if body_mask[index]:
            index += 1
            continue
        gap_start = index
        while index < width and not body_mask[index]:
            index += 1
        if gap_start > 0 and index < width and index - gap_start <= body_gap:
            body_mask[gap_start:index] = True

    body_segments = []
    index = 0
    while index < width:
        if not body_mask[index]:
            index += 1
            continue
        body_start = index
        while index < width and body_mask[index]:
            index += 1
        body_segments.append((body_start, index))

    if body_segments:
        body_start, body_end = max(body_segments, key=lambda item: item[1] - item[0])
    else:
        body_start, body_end = int(width * 0.03), int(width * 0.97)
    body_width = body_end - body_start

    if body_width < max(20, int(width * 0.30)):
        body_start, body_end = int(width * 0.03), int(width * 0.97)
        body_width = body_end - body_start

    # Exclude rounded body ends, where changing geometry/lighting may resemble a
    # color band. The margin is small enough to preserve the outer tolerance band.
    end_margin = max(3, int(round(body_width * 0.05)))
    left = min(body_end - 1, body_start + end_margin)
    right = max(left + 1, body_end - end_margin)
    if right - left < max(20, int(width * 0.25)):
        left, right = int(width * 0.03), int(width * 0.97)
        body_width = right - left

    body_lab = np.median(profile_lab[left:right], axis=0)
    body_value = float(np.median(profile_hsv[left:right, 2]))

    # Classify the full width profile first, then group contiguous labels. This
    # detects complete band interiors and avoids counting both band edges.
    column_labels = [
        classify_hsv_color(profile_hsv[x], baseline_value=body_value)
        for x in range(width)
    ]
    smoothed_labels = column_labels.copy()
    for index in range(1, width - 1):
        if (
            column_labels[index - 1] == column_labels[index + 1]
            and column_labels[index] != column_labels[index - 1]
        ):
            smoothed_labels[index] = column_labels[index - 1]

    minimum_width = max(3, int(round(body_width * 0.018)))
    maximum_width = max(minimum_width + 1, int(round(body_width * 0.13)))
    raw_runs: list[dict] = []
    index = left

    while index < right:
        color = smoothed_labels[index]
        run_start = index
        index += 1
        while index < right and smoothed_labels[index] == color:
            index += 1
        run_end = index
        run_width = run_end - run_start
        if not (minimum_width <= run_width <= maximum_width):
            continue

        # Re-evaluate a band from its central color sample to reduce sensitivity
        # to antialiased boundary pixels and dark outline strokes.
        trim = int(round(run_width * 0.12))
        core_start = min(run_end - 1, run_start + trim)
        core_end = max(core_start + 1, run_end - trim)
        band_hsv = np.median(profile_hsv[core_start:core_end], axis=0)
        band_lab = np.median(profile_lab[core_start:core_end], axis=0)
        band_color = classify_hsv_color(band_hsv, baseline_value=body_value)
        classifier_confidence = None
        if use_trained_classifier and predict_band_color is not None:
            try:
                band_patch = scan_rgb[:, core_start:core_end, :]
                model_prediction = predict_band_color(band_patch)
                if model_prediction is not None:
                    band_color, classifier_confidence = model_prediction
            except Exception:
                # Keep the HSV estimate as a fallback if the optional checkpoint is unavailable.
                classifier_confidence = None
        distance = float(np.linalg.norm(band_lab - body_lab))
        if distance < 9.0:
            continue

        score = distance * np.sqrt(run_width)
        raw_runs.append(
            {
                "color": band_color,
                "start": run_start,
                "end": run_end,
                "center": (run_start + run_end) // 2,
                "width": run_width,
                "contrast": distance,
                "score": score,
                "classifier_confidence": classifier_confidence,
            }
        )

    # Keep the strongest candidates, then return them in physical left-to-right
    # order. If a band is not visible enough, the UI asks for manual correction
    # instead of fabricating a complete resistor code.
    selected = sorted(raw_runs, key=lambda run: run["score"], reverse=True)[:expected_count]
    selected.sort(key=lambda run: run["center"])
    return selected, raw_runs

YOLO_BAND_COLOR_ALIASES = {
    **{color.casefold(): color for color in BAND_COLORS_RGB},
    "broan": "Brown",
    "grey": "Gray",
    "purple": "Violet",
    "side-gold": "Gold",
    "side-silver": "Silver",
    "bk": "Black",
    "br": "Brown",
    "rd": "Red",
    "og": "Orange",
    "ye": "Yellow",
    "gr": "Green",
    "bu": "Blue",
    "vt": "Violet",
    "gy": "Gray",
    "wh": "White",
}


def detect_band_runs_yolo(
    roi_rgb: np.ndarray,
    model_path: str | Path,
    expected_count: int,
    confidence_threshold: float = 0.15,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Run a locally trained YOLO segmentation/detection model on one resistor crop.

    Only predictions whose class names match resistor band colors are used as
    bands. Other model classes (such as whole-resistor value labels) are kept
    as extra predictions for display, not silently decoded as colors.
    """
    if importlib.util.find_spec("ultralytics") is None:
        raise RuntimeError(
            "Ultralytics is not installed in this environment. Install the local training "
            "environment with: python -m pip install -r requirements-training.txt"
        )

    model_file = Path(model_path).expanduser()
    if not model_file.is_file():
        raise FileNotFoundError(
            f"Segmentation weights were not found at {model_file}. Train a model first or select another path."
        )

    from ultralytics import YOLO

    model = YOLO(str(model_file))
    image_bgr = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR)
    predictions = model.predict(source=image_bgr, conf=float(confidence_threshold), verbose=False)
    if not predictions:
        return [], [], []

    prediction = predictions[0]
    names = prediction.names
    boxes = prediction.boxes
    color_candidates: list[dict] = []
    other_predictions: list[dict] = []

    if boxes is not None:
        for box in boxes:
            class_id = int(box.cls[0].item())
            class_name = str(
                names.get(class_id, class_id) if isinstance(names, dict) else names[class_id]
            ).strip()
            confidence = float(box.conf[0].item())
            coords = box.xyxy[0].cpu().numpy().astype(float).tolist()
            left, top, right, bottom = coords
            record = {
                "class": class_name,
                "confidence": confidence,
                "box": coords,
            }
            normalized_name = (
                class_name.casefold()
                .replace("band:", "")
                .replace(" color", "")
                .replace(" band", "")
                .strip()
            )
            canonical_color = YOLO_BAND_COLOR_ALIASES.get(normalized_name)
            if canonical_color is not None:
                color_candidates.append({
                    "color": canonical_color,
                    "start": max(0, int(round(left))),
                    "end": min(roi_rgb.shape[1], max(int(round(left)) + 1, int(round(right)))),
                    "center": (left + right) / 2,
                    "width": max(1, right - left),
                    "confidence": confidence,
                    "score": confidence * max(1.0, bottom - top),
                })
            else:
                other_predictions.append(record)

    # Sort by confidence first to suppress accidental duplicate/low-quality
    # color detections, then restore the physical left-to-right band sequence.
    chosen = sorted(color_candidates, key=lambda item: item["score"], reverse=True)[:expected_count]
    chosen.sort(key=lambda item: item["center"])
    return chosen, color_candidates, other_predictions


def annotate_bands(roi_rgb: np.ndarray, runs: list[dict], labels: list[str]) -> np.ndarray:
    """Draw band boxes and readable labels on a copy of the cropped image."""
    output = roi_rgb.copy()
    height, width = output.shape[:2]
    for index, run in enumerate(runs):
        color_name = labels[index] if index < len(labels) else run.get("color", "?")
        color = BAND_COLORS_RGB.get(color_name, (255, 255, 255))
        cv2.rectangle(output, (run["start"], 0), (max(run["start"], run["end"] - 1), height - 1), color, 2)
        text_y = 18 if index % 2 == 0 else min(height - 5, 38)
        cv2.putText(output, f"{index + 1}: {color_name}", (max(0, run["start"] - 5), text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(output, f"{index + 1}: {color_name}", (max(0, run["start"] - 5), text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def png_bytes(image_rgb: np.ndarray) -> bytes:
    buffer = BytesIO()
    Image.fromarray(np.asarray(image_rgb, dtype=np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def get_reference_rows() -> list[dict]:
    """Read the human-readable color-code dataset committed with the project."""
    reference_path = Path(__file__).resolve().parent / "data" / "resistor_color_reference.csv"
    if not reference_path.exists():
        return []
    with reference_path.open("r", encoding="utf-8", newline="") as reference_file:
        return list(csv.DictReader(reference_file))
# endregion


# region 04: Streamlit Resistor Analyzer
@st.cache_resource(show_spinner=False)
def load_local_yolo_model(model_path: str):
    """Load a local model checkpoint once per Streamlit worker."""
    from ultralytics import YOLO
    return YOLO(model_path)


def render_resistor_analyzer(image_rgb: np.ndarray | None, source_token: str = "image") -> None:
    """Render the resistor color-code reader inside VisionLab."""
    st.header("Electronic Component Analyzer")
    st.write(
        "Estimate the value of a single through-hole resistor from its color bands. "
        "A trained color classifier can refine candidate band colors, while OpenCV locates likely band positions "
        "and the standards-based lookup table calculates resistance and tolerance."
    )
    st.info(
        "Use the shared uploader or start the live webcam. The preview runs continuously; "
        "Capture & Detect Components analyzes the current camera frame. "
        "For resistor values, use the close-up crop and verify important readings with a multimeter."
    )

    image_source = st.radio(
        "Image source",
        ["Use shared image", "Webcam"],
        horizontal=True,
        key="electronics_image_source",
    )
    selected_token = source_token

    if image_source == "Webcam":
        class LiveComponentCamera(VideoProcessorBase):
            def __init__(self):
                self.latest_frame = None

            def recv(self, frame):
                frame_rgb = frame.to_ndarray(format="rgb24")
                self.latest_frame = frame_rgb.copy()
                return frame

        st.caption(
            "Live camera preview. Choose CatCam in the browser camera picker if prompted. "
            "Click Capture & Detect Components to analyze the current live frame."
        )
        camera_ctx = webrtc_streamer(
            key="visionlab_live_catcam",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=LiveComponentCamera,
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration={
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]},
                    {"urls": ["stun:stun1.l.google.com:19302"]},
                ]
            },
            async_processing=True,
        )
        capture_col, clear_col = st.columns(2)
        capture_clicked = capture_col.button(
            "Capture & Detect Components",
            type="primary",
            key="capture_live_component_frame",
            use_container_width=True,
        )
        clear_clicked = clear_col.button(
            "Clear Captured Frame",
            key="clear_live_component_frame",
            use_container_width=True,
        )
        if clear_clicked:
            st.session_state.pop("visionlab_live_captured_frame", None)
            st.session_state.pop("visionlab_live_captured_token", None)
        if capture_clicked:
            processor = camera_ctx.video_processor
            latest_frame = getattr(processor, "latest_frame", None) if processor else None
            if latest_frame is None:
                st.warning("The live camera has not delivered a frame yet. Wait for the preview, then try again.")
                return
            captured = np.asarray(latest_frame, dtype=np.uint8).copy()
            st.session_state["visionlab_live_captured_frame"] = captured
            st.session_state["visionlab_live_captured_token"] = hashlib.sha256(captured.tobytes()).hexdigest()[:20]
            st.session_state["visionlab_run_detection_after_capture"] = True

        image_rgb = st.session_state.get("visionlab_live_captured_frame")
        if image_rgb is None:
            st.info("Start the webcam preview, then click Capture & Detect Components. The selected frame will be analyzed automatically.")
            return
        selected_token = str(st.session_state.get("visionlab_live_captured_token", "live-frame"))

    if image_rgb is None:
        st.warning("Upload an image in the sidebar, or select Webcam and capture a frame from the live feed.")
        return
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        st.error("The selected image must decode to a three-channel RGB image.")
        return

    image_height, image_width = image_rgb.shape[:2]
    if image_width < 2 or image_height < 2:
        st.error("The image is too small to crop. Upload an image at least 2 × 2 pixels.")
        return

    # region 04A: Full-Scene Component Detection
    st.subheader("1. Find Electronic Components on the Desk")
    st.write(
        "This detector analyzes the whole selected image and draws a box around each predicted component. "
        "It does not automatically infer a resistor's ohmic value; use the crop-and-read tool below for that."
    )

    confidence_threshold = st.slider(
        "Minimum detection confidence",
        0.10, 0.90, 0.25, step=0.05,
        key="components_detector_confidence",
    )

    # Local checkpoints are populated by the repository's training workflow.
    model_root = Path(__file__).resolve().parent / "models"
    candidate_local_models = [
        {"path": model_root / "components.pt", "label": "Trained Electronics Model"},
        {"path": model_root / "iot_components.pt", "label": "Trained IoT Model"},
        {"path": model_root / "iot_boards.pt", "label": "Trained IoT Board Model"},
    ]
    local_runtime_available = importlib.util.find_spec("ultralytics") is not None
    if local_runtime_available:
        use_experimental_local_models = st.checkbox(
            "Include experimental local checkpoints",
            value=False,
            help=(
                "Off by default: the early CI-trained local checkpoint has not yet passed "
                "real-image validation. The app uses the published hosted detectors instead."
            ),
            key="include_experimental_local_checkpoints",
        )
    else:
        use_experimental_local_models = False
        if any(item["path"].is_file() for item in candidate_local_models):
            st.caption(
                "Local YOLO checkpoint files are present, but this lightweight Streamlit "
                "deployment does not install Ultralytics. Hosted detectors are used instead."
            )

    available_local_models = (
        [
            item for item in candidate_local_models
            if item["path"].is_file()
        ]
        if use_experimental_local_models
        else []
    )
    local_model_signature = tuple(
        (item["label"], item["path"].name, item["path"].stat().st_size, item["path"].stat().st_mtime_ns)
        for item in available_local_models
    )

    # Hosted inference uses the pre-configured Streamlit secret; no key input is exposed.
    try:
        api_key = str(st.secrets["ROBOFLOW_API_KEY"]).strip()
    except Exception:
        api_key = ""

    hosted_models = [
        {
            "id": "electronics-components-sveck/7",
            "label": "Hosted Electronics Components",
            "source": "https://universe.roboflow.com/vishnupriyas-workspace/electronics-components-sveck",
        },
        {
            # The supplied IoT_Dataset is a dataset-only project (Roboflow shows Models: 0),
            # so requesting /iot_dataset/2 from detect.roboflow.com returns HTTP 404.
            # Use this existing trained 58-class IoT/electronics model instead.
            "id": "iot-components-learning-ogliq/1",
            "label": "Hosted IoT Components",
            "source": "https://universe.roboflow.com/robertos-workspace-zpauf/iot-components-learning-ogliq",
        },
    ]

    st.markdown(
        "The analyzer automatically combines locally trained electronics/IoT models when available "
        "with hosted Roboflow predictions when the configured secret is present."
    )
    st.caption(
        "Hosted inference sends the selected image to Roboflow. Local inference runs inside this app. "
        "Avoid uploading private or sensitive images."
    )

    api_key_fingerprint = (
        hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:10] if api_key else "missing"
    )
    model_ids = tuple(model["id"] for model in hosted_models) if api_key else ()
    signature = (
        "automatic_ensemble",
        selected_token,
        model_ids,
        local_model_signature,
        float(confidence_threshold),
        api_key_fingerprint,
    )
    result_key = "components_full_scene_result"
    can_run_local = bool(available_local_models) and local_runtime_available
    can_run_detection = bool(api_key) or can_run_local

    if not can_run_detection:
        st.error(
            "No hosted detector is configured. Add ROBOFLOW_API_KEY under Streamlit app settings, "
            "or run locally with Ultralytics installed and a trained checkpoint enabled."
        )
    if image_source == "Webcam":
        run_detection = bool(st.session_state.pop("visionlab_run_detection_after_capture", False))
    else:
        run_detection = st.button(
            "Detect Electronic and IoT Components",
            type="primary",
            key="components_run_hosted_detection",
            disabled=not can_run_detection,
            width="stretch",
        )

    if run_detection and not can_run_detection:
        st.error("Component detection is unavailable. Configure ROBOFLOW_API_KEY in Streamlit app settings.")
    elif run_detection:
        candidates = []
        model_errors = []
        image_h, image_w = image_rgb.shape[:2]
        source_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        def add_prediction(class_name, confidence, coords, model_label):
            try:
                bx1, by1, bx2, by2 = [int(round(float(value))) for value in coords]
                bx1 = max(0, min(image_w - 1, bx1))
                by1 = max(0, min(image_h - 1, by1))
                bx2 = max(bx1 + 1, min(image_w, bx2))
                by2 = max(by1 + 1, min(image_h, by2))
                if bx2 <= bx1 or by2 <= by1:
                    return
                score = float(confidence)
                if score < float(confidence_threshold):
                    return
                candidates.append({
                    "Component": str(class_name).strip() or "component",
                    "Confidence": score,
                    "Bounding box (x1, y1, x2, y2)": f"{bx1}, {by1}, {bx2}, {by2}",
                    "_box": [bx1, by1, bx2, by2],
                    "Model": model_label,
                })
            except (TypeError, ValueError, OverflowError):
                return

        with st.spinner("Running the available electronics and IoT detectors..."):
            # Locally trained checkpoints are always attempted first.
            if can_run_local:
                for local_config in available_local_models:
                    try:
                        model = load_local_yolo_model(str(local_config["path"]))
                        predictions = model.predict(
                            source=source_bgr,
                            conf=float(confidence_threshold),
                            verbose=False,
                        )
                        if not predictions:
                            continue
                        prediction = predictions[0]
                        boxes = prediction.boxes
                        if boxes is None:
                            continue
                        for box in boxes:
                            class_id = int(box.cls[0].item())
                            names = prediction.names
                            class_name = (
                                names.get(class_id, class_id)
                                if isinstance(names, dict)
                                else names[class_id]
                            )
                            confidence = float(box.conf[0].item())
                            coords = box.xyxy[0].detach().cpu().numpy().astype(float).tolist()
                            add_prediction(class_name, confidence, coords, local_config["label"])
                    except Exception as error:
                        # Show a safe, useful failure reason without exposing secrets.
                        model_errors.append(
                            f"{local_config['label']}: {type(error).__name__}"
                        )

            # Hosted inference is optional and requires the configured Streamlit secret.
            if api_key:
                image_buffer = BytesIO()
                Image.fromarray(image_rgb).save(image_buffer, format="JPEG", quality=92)
                jpeg_bytes = image_buffer.getvalue()
                for model_config in hosted_models:
                    try:
                        predictions = request_roboflow_detections(
                            jpeg_bytes,
                            model_id=model_config["id"],
                            api_key=api_key,
                            confidence_threshold=float(confidence_threshold),
                        )
                        for prediction in predictions:
                            try:
                                class_name = str(
                                    prediction.get("class", prediction.get("class_name", "component"))
                                ).strip() or "component"
                                confidence = float(prediction.get("confidence", 0.0))
                                center_x = float(prediction.get("x", 0))
                                center_y = float(prediction.get("y", 0))
                                box_width = float(prediction.get("width", 0))
                                box_height = float(prediction.get("height", 0))
                                if box_width <= 0 or box_height <= 0:
                                    continue
                                coords = [
                                    center_x - box_width / 2,
                                    center_y - box_height / 2,
                                    center_x + box_width / 2,
                                    center_y + box_height / 2,
                                ]
                                add_prediction(class_name, confidence, coords, model_config["label"])
                            except (TypeError, ValueError, OverflowError):
                                continue
                    except Exception as error:
                        # Show actionable response diagnostics without leaking the configured key.
                        model_errors.append(
                            f"{model_config['label']}: {safe_error_summary(error, api_key)}"
                        )

        # Merge duplicate predictions only when their class names and boxes overlap.
        def box_iou(first, second):
            ax1, ay1, ax2, ay2 = first
            bx1, by1, bx2, by2 = second
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
            area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
            union = area_a + area_b - intersection
            return intersection / union if union else 0.0

        merged_records = []
        for candidate in sorted(candidates, key=lambda row: row["Confidence"], reverse=True):
            normalized_class = (
                candidate["Component"].casefold().replace(" ", "").replace("-", "").replace("_", "")
            )
            duplicate = False
            for kept in merged_records:
                kept_class = (
                    kept["Component"].casefold().replace(" ", "").replace("-", "").replace("_", "")
                )
                if normalized_class == kept_class and box_iou(candidate["_box"], kept["_box"]) >= 0.55:
                    duplicate = True
                    break
            if not duplicate:
                merged_records.append(candidate)

        annotated = image_rgb.copy()
        for record in merged_records:
            bx1, by1, bx2, by2 = record["_box"]
            class_name = record["Component"]
            confidence = float(record["Confidence"])
            color = (45, 210, 75)
            cv2.rectangle(annotated, (bx1, by1), (bx2, by2), color, 2)
            label = f"{class_name} {confidence:.0%}"
            label_y = max(20, by1 - 7)
            cv2.putText(annotated, label, (bx1, label_y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.52, (20, 20, 20), 3, cv2.LINE_AA)
            cv2.putText(annotated, label, (bx1, label_y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.52, color, 1, cv2.LINE_AA)

        st.session_state[result_key] = {
            "signature": repr(signature),
            "source_token": selected_token,
            "image": annotated,
            "records": merged_records,
            "boxes": merged_records,
            "model": " + ".join([item["label"] for item in available_local_models] + list(model_ids)),
            "warnings": model_errors,
            "error": None,
        }

    saved_detection = st.session_state.get("components_full_scene_result")
    if saved_detection and saved_detection.get("signature") == repr(signature):
        st.subheader("Detected Components")
        for warning in saved_detection.get("warnings", []):
            st.warning(f"Some predictions may be missing: {warning}")
        st.image(saved_detection["image"], width="stretch")
        st.metric("Components detected", len(saved_detection["records"]))
        if not saved_detection["records"]:
            if saved_detection.get("warnings"):
                st.info(
                    "No detections were returned because at least one hosted inference request failed. "
                    "Read the model-specific error above; lowering the confidence threshold will not fix "
                    "authentication or response-format errors."
                )
            else:
                st.info(
                    "No components were found above the selected confidence threshold. "
                    "Try a sharper, brighter image with components separated from the background, "
                    "or lower the confidence threshold."
                )
        if saved_detection["records"]:
            visible_records = [
                {key: value for key, value in row.items() if not key.startswith("_")}
                for row in saved_detection["records"]
            ]
            st.dataframe(visible_records, width="stretch", hide_index=True)
            st.markdown("**What each detected part usually does**")
            for number, record in enumerate(saved_detection["records"][:12], start=1):
                component_name = str(record.get("Component", "component"))
                confidence_percent = float(record.get("Confidence", 0.0)) * 100
                st.markdown(f"**{number}. {component_name} — {confidence_percent:.0f}% confidence**")
                lowered_name = component_name.casefold()
                if "resistor" in lowered_name or "resister" in lowered_name:
                    purpose = "Limits current, divides voltage, and sets circuit bias; the resistance value is decoded from its color bands."
                elif "capacitor" in lowered_name or "condens" in lowered_name:
                    purpose = "Temporarily stores electric charge; commonly used for voltage smoothing, noise filtering, and signal coupling."
                elif "diode" in lowered_name:
                    purpose = "Allows current to flow mainly in one direction; used for rectification and reverse-polarity protection."
                elif "inductor" in lowered_name or "coil" in lowered_name:
                    purpose = "Stores energy in a magnetic field; used in filters and power converters."
                elif "transistor" in lowered_name:
                    purpose = "Used to switch or amplify signals and control load current."
                elif "led" in lowered_name:
                    purpose = "Converts electrical energy into light; commonly used as an indicator or light source."
                elif "potentiometer" in lowered_name or "variable resistor" in lowered_name:
                    purpose = "An adjustable resistance used to change voltage or signal level."
                elif "crystal" in lowered_name or "oscillator" in lowered_name:
                    purpose = "Provides a frequency reference for circuit clocks and timing."
                elif "cable" in lowered_name or "wire" in lowered_name:
                    purpose = "Provides an electrical connection for power or signals between circuit sections."
                elif "ic" in lowered_name or "integrated circuit" in lowered_name:
                    purpose = "An integrated circuit; its function depends on the part and may include logic, control, amplification, or processing."
                elif "esp32" in lowered_name:
                    purpose = "A Wi-Fi/Bluetooth-capable microcontroller board commonly used for connected sensors, automation, and embedded IoT projects."
                elif "arduino" in lowered_name:
                    purpose = "A microcontroller development board commonly used to prototype sensor inputs, actuators, and embedded control."
                elif "raspberry pi" in lowered_name or "raspberrypi" in lowered_name:
                    purpose = "A small single-board computer commonly used for Linux-based IoT gateways, edge processing, and hardware prototyping."
                elif "breadboard" in lowered_name:
                    purpose = "A reusable, solderless prototyping board used to connect and test circuits before soldering."
                elif "sensor" in lowered_name:
                    purpose = "Measures a physical quantity such as temperature, humidity, motion, distance, or light; the exact role depends on the sensor model."
                elif "development board" in lowered_name or "microcontroller" in lowered_name or "iot" in lowered_name or "board" in lowered_name:
                    purpose = "An embedded development platform used to connect sensors and actuators, run firmware, and prototype IoT systems."
                else:
                    purpose = "The model suggests a visual label only. Check the part number and circuit to identify its exact function."
                st.caption(f"Typical use: {purpose}")
            if len(saved_detection["records"]) > 12:
                st.caption(f"To keep the page readable, descriptions for {len(saved_detection['records']) - 12} additional detections are hidden.")
        else:
            st.info(
                "The detector did not return any components above this confidence threshold. "
                "Try lowering the threshold or taking a clearer photo with the components spread apart."
            )
        st.download_button(
            "Download Component Detection Image",
            data=png_bytes(saved_detection["image"]),
            file_name="electronic_components_detected.png",
            mime="image/png",
            key="components_detection_download",
            width="stretch",
        )
        st.caption(f"Detector: {saved_detection['model']}. Detections depend on the model's training domain, object size, lighting, and camera angle.")

    with st.expander("Datasets, training, and model limitations"):
        st.markdown(
            "- **General components:** [Electronics Components](https://universe.roboflow.com/vishnupriyas-workspace/electronics-components-sveck), approximately 8,248 images and 10 classes (CC BY 4.0 on the source page).\n"
            "- **IoT parts:** [IoT_Dataset](https://universe.roboflow.com/design-project-zeico/iot_dataset), 209 images and seven classes, including Arduino, Resistor, Transistor, and Ultrasonic Sensor (MIT on the source page).\n"
            "- **Resistor segmentation:** [Resistor Value Determination](https://universe.roboflow.com/resistor-color-code/resistor-value-determination), semantic segmentation with color/value labels (CC BY 4.0). Its model page reports 5.9% mIoU, so do not treat the hosted model as accurate ground truth.\n"
            "- **Kaggle resistor masks:** [Resistor Images Dataset for Segmentation](https://www.kaggle.com/datasets/alptugaydin/resistor-images-dataset-for-segmentation). Configure Kaggle access and review its current license before reuse or redistribution.\n"
            "- **Additional IoT boards:** [IoTKITs on Mendeley Data](https://data.mendeley.com/datasets/x5thzmkxhy/1), 3,200 annotated images across 32 educational kit/board classes (CC BY 4.0 on the source page).\n"
            "- See [DATASETS.md](https://github.com/Amsh23/VisionLab/blob/main/DATASETS.md) for source attribution and complete instructions. Large datasets and model weights are stored locally, not in normal Git history.\n"
            "- Keep object detection and semantic segmentation as separate tasks. A generic YOLO detector does not automatically decode resistor color bands or calculate resistance."
        )
        st.code(
            "python -m pip install -r requirements-training.txt\n"
            "$env:ROBOFLOW_API_KEY=\"YOUR_ROBOFLOW_API_KEY\"\n"
            "python scripts/download_roboflow_dataset.py --dataset iot\n"
            "python scripts/train_components_yolo.py --data data/iot_roboflow/data.yaml --name iot_parts --weights-out models/iot_components.pt\n"
            "python scripts/download_roboflow_dataset.py --dataset resistor-segmentation\n"
            "python scripts/prepare_resistor_segmentation_yolo.py --input data/resistor_value_segmentation --output data/resistor_seg_yolo --background-value 0\n"
            "python scripts/train_components_yolo.py --data data/resistor_seg_yolo/data.yaml --model yolo11n-seg.pt --epochs 80 --batch 4 --name resistor_bands --weights-out models/resistor_seg.pt\n"
            "python scripts/download_kaggle_resistor_dataset.py",
            language="powershell",
        )
        st.caption(
            "Use the training environment on a PC, not inside Streamlit Cloud. "
            "The resistor semantic-segmentation export needs a compatible segmentation trainer; do not train it with the default detection model."
        )
    # endregion

    st.divider()
    st.subheader("2. Select the Resistor Region")
    st.caption("Left and top coordinates are inclusive. Right and bottom coordinates are exclusive.")
    current_scene = st.session_state.get("components_full_scene_result", {})
    scene_boxes = (
        current_scene.get("boxes", [])
        if current_scene.get("source_token") == selected_token
        else []
    )
    resistor_box_indices = [
        index for index, item in enumerate(scene_boxes)
        if "resistor" in str(item.get("Component", "")).lower()
        or "resister" in str(item.get("Component", "")).lower()
    ]
    crop_choices = ["Manual crop coordinates"] + [
        f"Detected resistor #{number} ({scene_boxes[index]['Component']}, {scene_boxes[index]['Confidence']:.0%})"
        for number, index in enumerate(resistor_box_indices, start=1)
    ]
    crop_choice = st.selectbox(
        "Crop source",
        crop_choices,
        key=f"res_crop_source_{selected_token}",
    )
    default_box = [0, 0, image_width, image_height]
    if crop_choice != "Manual crop coordinates":
        selected_box_position = crop_choices.index(crop_choice) - 1
        selected_box_index = resistor_box_indices[selected_box_position]
        default_box = scene_boxes[selected_box_index]["_box"]

    roi_key = f"{selected_token}_{crop_choice.replace(' ', '_').replace('#', '')}"
    step_x = max(1, image_width // 100)
    step_y = max(1, image_height // 100)

    c1, c2, c3, c4 = st.columns(4)
    x1 = c1.number_input("Left (X1)", 0, image_width - 1, int(default_box[0]), step=step_x, key=f"res_roi_x1_{roi_key}")
    y1 = c2.number_input("Top (Y1)", 0, image_height - 1, int(default_box[1]), step=step_y, key=f"res_roi_y1_{roi_key}")
    x2 = c3.number_input("Right (X2)", 1, image_width, int(default_box[2]), step=step_x, key=f"res_roi_x2_{roi_key}")
    y2 = c4.number_input("Bottom (Y2)", 1, image_height, int(default_box[3]), step=step_y, key=f"res_roi_y2_{roi_key}")

    if int(x2) <= int(x1) or int(y2) <= int(y1):
        st.error("Right must be greater than left, and bottom must be greater than top.")
        return

    roi = image_rgb[int(y1):int(y2), int(x1):int(x2)].copy()
    transform = st.selectbox(
        "Rotate the crop if the resistor is not horizontal",
        ["Already horizontal", "Rotate 90° clockwise", "Rotate 90° counter-clockwise", "Rotate 180°"],
        key=f"res_roi_rotation_{roi_key}",
    )
    if transform == "Rotate 90° clockwise":
        roi = cv2.rotate(roi, cv2.ROTATE_90_CLOCKWISE)
    elif transform == "Rotate 90° counter-clockwise":
        roi = cv2.rotate(roi, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif transform == "Rotate 180°":
        roi = cv2.rotate(roi, cv2.ROTATE_180)

    preview_left, preview_right = st.columns(2)
    with preview_left:
        st.image(image_rgb, caption="Original image", width="stretch")
    with preview_right:
        st.image(roi, caption="Selected resistor crop", width="stretch")
    st.caption(f"Crop size: {roi.shape[1]} × {roi.shape[0]} pixels.")
    st.caption(
        "For best results, crop a single resistor and make sure the color bands are clear and vertical. "
        "Avoid direct flash glare on the body and keep the resistor horizontal."
    )

    st.subheader("2. Configure Band Reading")
    p1, p2, p3 = st.columns(3)
    band_count = p1.selectbox("Resistor band count", [4, 5], index=0, format_func=lambda value: f"{value}-band resistor", key=f"res_band_count_{roi_key}")
    model_root = Path(__file__).resolve().parent / "models"
    band_segmentation_path = model_root / "resistor_seg.pt"
    band_classifier_path = model_root / "resistor_band_classifier.joblib"
    # The standard path uses the adaptive HSV/Lab heuristic. The trained patch
    # classifier was evaluated only on synthetic color patches, so keep it opt-in.
    automatic_mode_label = "Automatic color detection (HSV heuristic)"
    experimental_classifier_label = "Experimental synthetic-trained color classifier"
    default_band_mode = automatic_mode_label

    mode_options = [automatic_mode_label]
    if band_classifier_path.is_file():
        mode_options.append(experimental_classifier_label)
    mode_options.extend([
        "Local YOLO segmentation model",
        "Manual color selection",
    ])
    mode = p2.selectbox(
        "Band reading method",
        mode_options,
        index=mode_options.index(default_band_mode),
        key=f"res_read_mode_{roi_key}",
    )
    if mode == experimental_classifier_label:
        st.warning(
            "This classifier was trained and tested on synthetic color patches only. "
            "Its 92.6% synthetic test accuracy is not real-photo accuracy. For physical "
            "resistors, start with the HSV heuristic or correct the colors manually."
        )
    elif mode == automatic_mode_label:
        st.caption(
            "Default for real photos: adaptive HSV/Lab band detection. Lighting, glare, "
            "cropping, and camera white balance may still require manual color correction."
        )
    band_order = p3.selectbox("Read bands in this direction", ["Left to right", "Right to left"], key=f"res_band_order_{roi_key}")

    band_model_path = "models/resistor_seg.pt"
    band_model_confidence = 0.15
    if mode == "Local YOLO segmentation model":
        model_col, confidence_col = st.columns([2, 1])
        band_model_path = model_col.text_input(
            "Local segmentation weights",
            value="models/resistor_seg.pt",
            key=f"res_band_model_path_{roi_key}",
            help="Use weights trained with band-color classes. A generic component detector may not predict individual band colors.",
        )
        band_model_confidence = confidence_col.slider(
            "Model confidence",
            min_value=0.05,
            max_value=0.90,
            value=0.15,
            step=0.05,
            key=f"res_band_model_confidence_{roi_key}",
        )
        st.caption(
            "This mode uses locally trained band-color weights. The model is trained separately from the general component detector."
        )

    manual_bands: list[str] = []
    if mode == "Manual color selection":
        st.caption("Choose each band's visible color in the same order as the selected reading direction.")
        band_columns = st.columns(band_count)
        defaults_4 = ["Yellow", "Violet", "Red", "Gold"]
        defaults_5 = ["Yellow", "Violet", "Black", "Brown", "Brown"]
        defaults = defaults_4 if band_count == 4 else defaults_5

        for index in range(band_count):
            if index < band_count - 2:
                choices = DIGIT_COLORS
            elif index == band_count - 2:
                choices = list(MULTIPLIER_OHMS.keys())
            else:
                choices = TOLERANCE_OPTIONS
            default_color = defaults[index]
            initial_index = choices.index(default_color) if default_color in choices else 0
            manual_bands.append(
                band_columns[index].selectbox(
                    f"Band {index + 1}",
                    choices,
                    index=initial_index,
                    key=f"res_band_{roi_key}_{band_count}_{index}",
                )
            )

    st.caption(
        "The four-band code uses two significant digits, a multiplier, and tolerance. "
        "The five-band code uses three significant digits, a multiplier, and tolerance."
    )

    signature = (
        str(selected_token), crop_choice, int(x1), int(y1), int(x2), int(y2), transform,
        int(band_count), mode, band_order, tuple(manual_bands),
        band_model_path if mode == "Local YOLO segmentation model" else None,
        float(band_model_confidence) if mode == "Local YOLO segmentation model" else None,
    )
    signature_text = repr(signature)
    result_state_key = "resistor_analyzer_last_result"

    if st.button("Analyze Resistor and Estimate Ohms", type="primary", key=f"res_analyze_button_{roi_key}", width="stretch"):
        result: dict = {"bands": [], "runs": [], "decoded": None, "error": None}
        try:
            if mode == "Manual color selection":
                bands = list(manual_bands)
                if band_order == "Right to left":
                    bands.reverse()
                result["bands"] = bands
                result["annotated"] = roi.copy()
                result["decoded"] = decode_band_sequence(bands, int(band_count))
            elif mode == "Local YOLO segmentation model":
                runs, all_runs, model_predictions = detect_band_runs_yolo(
                    roi,
                    band_model_path,
                    int(band_count),
                    float(band_model_confidence),
                )
                physical_runs = sorted(runs, key=lambda run: run["center"])
                read_runs = physical_runs if band_order == "Left to right" else list(reversed(physical_runs))
                result["runs"] = physical_runs
                result["all_candidate_count"] = len(all_runs)
                result["model_predictions"] = model_predictions
                result["bands"] = [run["color"] for run in read_runs]
                result["annotated"] = annotate_bands(
                    roi,
                    physical_runs,
                    [run["color"] for run in physical_runs],
                )
                if len(read_runs) != int(band_count):
                    result["error"] = (
                        f"The local model detected {len(read_runs)} color band(s), but {band_count} are required. "
                        "Check that this model was trained on individual resistor-band colors, adjust confidence, "
                        "or switch to Manual color selection."
                    )
                else:
                    result["decoded"] = decode_band_sequence(result["bands"], int(band_count))
            else:
                runs, all_runs = detect_band_runs(
                    roi,
                    int(band_count),
                    use_trained_classifier=(mode == experimental_classifier_label),
                )
                physical_runs = sorted(runs, key=lambda run: run["center"])
                read_runs = physical_runs if band_order == "Left to right" else list(reversed(physical_runs))
                result["runs"] = physical_runs
                result["all_candidate_count"] = len(all_runs)
                result["bands"] = [run["color"] for run in read_runs]
                physical_labels = [run["color"] for run in physical_runs]
                result["annotated"] = annotate_bands(roi, physical_runs, physical_labels)
                if len(read_runs) != int(band_count):
                    result["error"] = (
                        f"Detected {len(runs)} likely band(s), but {band_count} are required. "
                        "Try cropping closer around the resistor body, improving the lighting, "
                        "or switching to Manual color selection."
                    )
                else:
                    result["decoded"] = decode_band_sequence(result["bands"], int(band_count))
        except Exception as error:
            # Model loading can raise dependency- or checkpoint-specific exceptions.
            # Surface these in the Streamlit page instead of crashing the whole app.
            result["error"] = str(error)
            result.setdefault("annotated", roi.copy())

        st.session_state[result_state_key] = {
            "signature": signature_text,
            "result": result,
        }

    saved = st.session_state.get(result_state_key)
    current_result = None
    if saved and saved.get("signature") == signature_text:
        current_result = saved.get("result")
    elif saved:
        st.caption("The image crop or settings changed. Run the analysis again to refresh the result.")

    if current_result is not None:
        st.subheader("3. Analysis Result")
        suggested_bands = list(current_result.get("bands", []))
        display_bands = suggested_bands.copy()
        corrected = False

        if mode in (automatic_mode_label, experimental_classifier_label, "Local YOLO segmentation model") and len(suggested_bands) == int(band_count):
            st.markdown("**Review or correct the detected colors**")
            st.caption(
                "Predicted band colors can still be wrong because of lighting, image quality, or model limitations. "
                "Review each band color; the resistance value and formula below will be recalculated using your selections."
            )
            correction_hash = hashlib.sha256(signature_text.encode("utf-8")).hexdigest()[:10]
            correction_columns = st.columns(int(band_count))
            corrected_bands: list[str] = []
            for index, suggested_color in enumerate(suggested_bands):
                if index < int(band_count) - 2:
                    choices = DIGIT_COLORS
                elif index == int(band_count) - 2:
                    choices = list(MULTIPLIER_OHMS.keys())
                else:
                    choices = TOLERANCE_OPTIONS
                initial_index = choices.index(suggested_color) if suggested_color in choices else 0
                corrected_bands.append(
                    correction_columns[index].selectbox(
                        f"Band {index + 1}",
                        choices,
                        index=initial_index,
                        key=f"res_auto_color_{roi_key}_{correction_hash}_{index}",
                    )
                )
            display_bands = corrected_bands
            corrected = display_bands != suggested_bands
        elif current_result.get("error"):
            st.caption(
                "If the selected model does not predict individual band colors, train a band-color segmentation model or use Manual color selection."
            )

        annotated = current_result.get("annotated", roi)
        if mode in (automatic_mode_label, experimental_classifier_label, "Local YOLO segmentation model") and len(display_bands) == int(band_count):
            physical_runs = sorted(current_result.get("runs", []), key=lambda run: run["center"])
            if len(physical_runs) == int(band_count):
                physical_labels = display_bands if band_order == "Left to right" else list(reversed(display_bands))
                annotated = annotate_bands(roi, physical_runs, physical_labels)

        output_col1, output_col2 = st.columns([1, 1])
        with output_col1:
            st.image(annotated, caption="Color-band analysis (labels can be corrected below)", width="stretch")
            st.download_button(
                "Download Analysis Image",
                data=png_bytes(annotated),
                file_name="resistor_band_analysis.png",
                mime="image/png",
                key=f"res_download_annotated_{roi_key}",
                width="stretch",
            )
        with output_col2:
            if display_bands:
                st.write("**Band sequence:**")
                st.write(" → ".join(display_bands))
                band_rows = []
                digit_roles = ["First significant digit", "Second significant digit", "Third significant digit"]
                for index, color in enumerate(display_bands):
                    if index < int(band_count) - 2:
                        code = DIGIT_VALUES.get(color)
                        role = digit_roles[index]
                        meaning = (
                            f"Digit {code} is part of the significant resistance value."
                            if code is not None
                            else "This color is not valid for a significant-digit band. Correct the selection."
                        )
                        encoded = str(code) if code is not None else "Invalid"
                    elif index == int(band_count) - 2:
                        multiplier = MULTIPLIER_OHMS.get(color)
                        role = "Multiplier"
                        encoded = f"×{multiplier:g}" if multiplier is not None else "Invalid"
                        meaning = f"Multiply the previous digits by {multiplier:g}." if multiplier is not None else "This is not a valid multiplier color."
                    else:
                        tolerance = TOLERANCE_PERCENT.get(color)
                        role = "Tolerance"
                        encoded = f"±{tolerance:g}%" if tolerance is not None else "Invalid"
                        meaning = (
                            f"The actual resistance may differ from the nominal value by up to {tolerance:g}%."
                            if tolerance is not None
                            else "This is not a valid tolerance color."
                        )
                    band_rows.append({
                        "Band": index + 1,
                        "Color": color,
                        "Code": encoded,
                        "Role": role,
                        "Explanation": meaning,
                    })
                st.dataframe(band_rows, width="stretch", hide_index=True)

            if corrected:
                st.info("The corrected band colors were used to calculate the result.")
            if current_result.get("model_predictions"):
                st.caption("Additional model classes in this crop:")
                st.dataframe(
                    [
                        {
                            "Predicted class": item["class"],
                            "Confidence": f"{item['confidence']:.0%}",
                        }
                        for item in current_result["model_predictions"]
                    ],
                    width="stretch",
                    hide_index=True,
                )

            decoded = None
            decode_error = None
            if len(display_bands) == int(band_count):
                try:
                    decoded = decode_band_sequence(display_bands, int(band_count))
                except ValueError as error:
                    decode_error = str(error)

            if decode_error:
                st.warning(f"The selected colors do not form a valid resistor code: {decode_error}")
            elif current_result.get("error") and not decoded:
                st.warning(current_result["error"])
            elif current_result.get("error") and decoded and corrected:
                st.info("The initial automatic prediction was invalid, but the corrected color sequence forms a valid resistor code.")

            if decoded:
                st.metric("Estimated resistance", format_resistance(decoded["ohms"]))
                st.metric("Tolerance", f"±{decoded['tolerance_percent']:g}%")
                low = format_resistance(decoded["minimum_ohms"])
                high = format_resistance(decoded["maximum_ohms"])
                st.write(f"**Expected range:** {low} to {high}")
                formula = (
                    f"{decoded['digits']} × {decoded['multiplier']:g} Ω "
                    f"= {format_resistance(decoded['ohms'])}"
                )
                st.markdown(f"**Calculation:** {formula}")
                st.caption(
                    "Common resistor uses include limiting current (for example, to protect an LED), dividing voltage, "
                    "and setting circuit bias. This value is estimated from the color code, not measured directly. "
                    "Verify important values with a multimeter."
                )
                st.caption(
                    f"Tolerance is the allowed variation around the nominal value; the expected range is {low} to {high}, "
                    "assuming the component is undamaged and the bands are read correctly."
                )

    st.divider()
    st.subheader("Color-Code Reference Dataset")
    reference_rows = get_reference_rows()
    if reference_rows:
        st.dataframe(reference_rows, width="stretch", hide_index=True)
    else:
        st.info("The reference CSV is missing. Restore data/resistor_color_reference.csv from the repository.")

    with st.expander("How this module works and example Python code"):
        st.write(
            "Automatic mode scans a narrow center strip, compares a Lab-color profile with a smoothed "
            "local baseline to locate narrow bands, and then estimates each band's color from HSV hue, "
            "saturation, and brightness relative to the resistor body. The UI lets you correct every "
            "suggested color before recalculating the code. Strong glare, a loose crop, unusual body "
            "colors, or similar-looking brown/red/orange and gold/yellow bands can still confuse a "
            "classical color heuristic; use manual selection or a multimeter to verify important values."
        )
        st.code(
            "digits = int(''.join(str(DIGIT_VALUES[c]) for c in digit_bands))\n"
            "resistance_ohms = digits * MULTIPLIER_OHMS[multiplier_color]\n"
            "tolerance_percent = TOLERANCE_PERCENT[tolerance_color]\n"
            "lower = resistance_ohms * (1 - tolerance_percent / 100)\n"
            "upper = resistance_ohms * (1 + tolerance_percent / 100)",
            language="python",
        )

# endregion
