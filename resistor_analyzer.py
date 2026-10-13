# region 01: Imports
from __future__ import annotations

from collections import Counter
from io import BytesIO
from pathlib import Path
import csv
import hashlib
import importlib.util

import cv2
import requests
import numpy as np
import streamlit as st
from PIL import Image
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


def detect_band_runs(roi_rgb: np.ndarray, expected_count: int) -> tuple[list[dict], list[dict]]:
    """Find candidate resistor bands using an adaptive color-contrast profile.

    Only the central part of the crop is scanned, avoiding background and the
    specular highlight along the top of the resistor body. A smoothed Lab-color
    profile models the body color/lighting; narrow changes from that local
    baseline are candidate bands. Gold and silver bands may blend into the body,
    so short gaps between their high-contrast edges are bridged before grouping.
    """
    height, width = roi_rgb.shape[:2]
    if width < 25 or height < 8:
        return [], []

    top = max(0, int(round(height * 0.36)))
    bottom = min(height, max(top + 1, int(round(height * 0.64))))
    if bottom - top < 5:
        top, bottom = int(height * 0.25), int(height * 0.75)
    scan_rgb = roi_rgb[top:bottom]

    profile_rgb = np.median(scan_rgb.astype(np.float32), axis=0).astype(np.uint8)
    profile_lab = cv2.cvtColor(profile_rgb.reshape(1, width, 3), cv2.COLOR_RGB2LAB)[0].astype(np.float32)
    profile_hsv = cv2.cvtColor(profile_rgb.reshape(1, width, 3), cv2.COLOR_RGB2HSV)[0].astype(np.float32)

    kernel_width = max(7, min(51, int(round(width * 0.045)) | 1))
    smooth_lab = cv2.GaussianBlur(
        profile_lab.reshape(1, width, 3), (kernel_width, 1), 0
    ).reshape(width, 3)
    contrast = np.linalg.norm(profile_lab - smooth_lab, axis=1)

    adjacent_delta = np.linalg.norm(np.diff(profile_lab, axis=0), axis=1)
    median_delta = float(np.median(adjacent_delta)) if len(adjacent_delta) else 0.0
    delta_mad = float(np.median(np.abs(adjacent_delta - median_delta))) if len(adjacent_delta) else 0.0
    noise_floor = median_delta + 4.5 * 1.4826 * delta_mad
    contrast_threshold = max(3.0, min(6.0, noise_floor))
    mask = contrast >= contrast_threshold

    edge_width = max(3, int(round(width * 0.035)))
    edge_pixels = np.concatenate([profile_rgb[:edge_width], profile_rgb[-edge_width:]], axis=0)
    edge_rgb = np.median(edge_pixels, axis=0).astype(np.uint8)
    edge_lab = cv2.cvtColor(edge_rgb.reshape(1, 1, 3), cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)
    middle_lab = np.median(
        profile_lab[int(width * 0.20):max(int(width * 0.20) + 1, int(width * 0.80))],
        axis=0,
    )
    edge_to_body_distance = float(np.linalg.norm(edge_lab - middle_lab))

    if edge_to_body_distance < 12.0:
        body_start, body_end = int(width * 0.02), int(width * 0.98)
    else:
        body_distance = np.linalg.norm(profile_lab - edge_lab, axis=1)
        body_columns = np.flatnonzero(body_distance > 12.0)
        if len(body_columns) >= max(12, int(width * 0.15)):
            body_start, body_end = int(body_columns[0]), int(body_columns[-1]) + 1
        else:
            body_start, body_end = int(width * 0.02), int(width * 0.98)

    edge_margin = max(3, int(round(width * 0.025)))
    allowed_left = max(int(width * 0.035), body_start + edge_margin)
    allowed_right = min(int(width * 0.965), body_end - edge_margin)
    if allowed_right <= allowed_left:
        allowed_left, allowed_right = int(width * 0.035), int(width * 0.965)

    merge_gap = max(3, int(round(width * 0.035)))
    index = 0
    while index < width:
        if mask[index]:
            index += 1
            continue
        end = index
        while end < width and not mask[end]:
            end += 1
        if index > 0 and end < width and end - index <= merge_gap:
            mask[index:end] = True
        index = end

    mask[:max(1, allowed_left)] = False
    mask[max(0, allowed_right):] = False

    minimum_width = max(2, int(round(width * 0.008)))
    maximum_width = max(minimum_width + 1, int(round(width * 0.10)))
    raw_runs: list[dict] = []
    index = 0

    while index < width:
        if not mask[index]:
            index += 1
            continue

        start = index
        while index < width and mask[index]:
            index += 1
        end = index
        run_width = end - start
        if not (minimum_width <= run_width <= maximum_width):
            continue

        center = (start + end) // 2
        if not (allowed_left <= center <= allowed_right):
            continue

        half_window = max(2, int(round(width * 0.01)))
        core_start = max(start, center - half_window)
        core_end = min(end, center + half_window + 1)
        core = scan_rgb[:, core_start:core_end, :]
        median_rgb = np.median(core.reshape(-1, 3), axis=0).astype(np.uint8)
        median_hsv = cv2.cvtColor(median_rgb.reshape(1, 1, 3), cv2.COLOR_RGB2HSV)[0, 0]

        baseline_left = max(0, center - kernel_width // 2)
        baseline_right = min(width, center + kernel_width // 2 + 1)
        baseline_value = float(np.median(profile_hsv[baseline_left:baseline_right, 2]))
        color = classify_hsv_color(median_hsv, baseline_value=baseline_value)

        strength = float(np.mean(contrast[start:end]))
        raw_runs.append({
            "color": color,
            "start": start,
            "end": end,
            "center": center,
            "width": run_width,
            "contrast": strength,
            "score": strength * np.sqrt(run_width),
        })

    selected = sorted(raw_runs, key=lambda run: run["score"], reverse=True)[:expected_count]
    selected.sort(key=lambda run: run["center"])
    return selected, raw_runs

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
def render_resistor_analyzer(image_rgb: np.ndarray | None, source_token: str = "image") -> None:
    """Render the resistor color-code reader inside VisionLab."""
    st.header("Electronic Component Analyzer")
    st.write(
        "Estimate the value of a single through-hole resistor from its color bands. "
        "This first version uses classical OpenCV color analysis and a standards-based lookup table; "
        "it does not pretend to be a trained object detector."
    )
    st.info(
        "Use the shared uploader or capture a photo of your workbench with your webcam. "
        "The full-scene detector can locate electronic components when a model is available; "
        "the resistor reader estimates ohms from a close crop. A camera capture is a still photo, not a continuous live stream."
    )

    image_source = st.radio(
        "Image source",
        ["Use shared image", "Take a webcam photo"],
        horizontal=True,
        key="electronics_image_source",
    )
    selected_token = source_token

    if image_source == "Take a webcam photo":
        camera_photo = st.camera_input(
            "Open your camera, point it at the components on your desk, and take a picture",
            key="electronics_camera_capture",
            help="Allow camera access in your browser. On Streamlit Community Cloud, open the app using HTTPS.",
        )
        if camera_photo is None:
            st.info("Take a photo to begin component detection.")
            return
        try:
            camera_bytes = camera_photo.getvalue()
            with Image.open(BytesIO(camera_bytes)) as captured:
                image_rgb = np.array(captured.convert("RGB"), dtype=np.uint8, copy=True)
            selected_token = hashlib.sha256(camera_bytes).hexdigest()[:20]
        except Exception as error:
            st.error(f"Could not read the camera image: {error}")
            return

    if image_rgb is None:
        st.warning("Upload an image in the sidebar, or select the webcam option and take a photo.")
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

    detector_backend = st.selectbox(
        "Detection backend",
        ["Hosted Roboflow model (recommended)", "Local YOLO weights"],
        key="components_detector_backend",
    )
    confidence_threshold = st.slider(
        "Minimum detection confidence",
        0.05, 0.95, 0.35, step=0.05,
        key="components_detector_confidence",
    )

    if detector_backend == "Hosted Roboflow model (recommended)":
        model_id = "electronics-components-sveck/7"
        st.markdown(
            "Model: [Electronics Components YOLO model](https://universe.roboflow.com/vishnupriyas-workspace/electronics-components-sveck/model/7) "
            "— advertised classes include resistors, capacitors, diodes, inductors, transistors, LEDs, ICs, and potentiometers."
        )
        st.caption(
            "Hosted inference sends the selected photo to Roboflow for processing. "
            "Check the provider's current terms and quota before using it for anything beyond a class project."
        )
        try:
            configured_api_key = str(st.secrets["ROBOFLOW_API_KEY"]).strip()
        except Exception:
            configured_api_key = ""

        if not configured_api_key:
            st.caption(
                "For deployment, add ROBOFLOW_API_KEY to the app's Streamlit secrets, or paste your key below for this session."
            )
        session_api_key = st.text_input(
            "Roboflow API key",
            value="",
            type="password",
            key="components_roboflow_key_input",
            help="Stored only in this Streamlit session when entered here. Never commit API keys to GitHub.",
        )
        api_key = configured_api_key or session_api_key.strip()
        can_run_hosted = bool(api_key)
        if not api_key:
            st.info("Create a Roboflow account and API key at https://app.roboflow.com/settings/api to use the hosted model.")
        run_detection = st.button(
            "Detect Components in the Whole Image",
            type="primary",
            key="components_run_hosted_detection",
            disabled=not can_run_hosted,
            width="stretch",
        )

        signature = (
            "hosted", selected_token, model_id, float(confidence_threshold),
            hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:10],
        )
        result_key = "components_full_scene_result"

        if run_detection:
            try:
                with st.spinner("Sending the selected image for component detection..."):
                    # Use Roboflow's hosted REST endpoint directly. This avoids installing
                    # inference-sdk, which pulls opencv-python alongside opencv-python-headless.
                    image_buffer = BytesIO()
                    Image.fromarray(image_rgb).save(image_buffer, format="JPEG", quality=92)
                    response = requests.post(
                        f"https://detect.roboflow.com/{model_id}",
                        params={
                            "api_key": api_key,
                            "confidence": int(round(float(confidence_threshold) * 100)),
                            "overlap": 30,
                        },
                        files={"file": ("image.jpg", image_buffer.getvalue(), "image/jpeg")},
                        timeout=60,
                    )
                    response.raise_for_status()
                    prediction_payload = response.json()
                predictions = prediction_payload.get("predictions", [])
                annotated = image_rgb.copy()
                records = []
                image_h, image_w = image_rgb.shape[:2]

                for prediction in predictions:
                    class_name = str(prediction.get("class", prediction.get("class_name", "component")))
                    confidence = float(prediction.get("confidence", 0.0))
                    if confidence < float(confidence_threshold):
                        continue

                    center_x = float(prediction.get("x", 0))
                    center_y = float(prediction.get("y", 0))
                    box_width = float(prediction.get("width", 0))
                    box_height = float(prediction.get("height", 0))
                    bx1 = max(0, min(image_w - 1, int(round(center_x - box_width / 2))))
                    by1 = max(0, min(image_h - 1, int(round(center_y - box_height / 2))))
                    bx2 = max(bx1 + 1, min(image_w, int(round(center_x + box_width / 2))))
                    by2 = max(by1 + 1, min(image_h, int(round(center_y + box_height / 2))))

                    color = (45, 210, 75)
                    cv2.rectangle(annotated, (bx1, by1), (bx2, by2), color, 2)
                    label = f"{class_name} {confidence:.0%}"
                    label_y = max(20, by1 - 7)
                    cv2.putText(annotated, label, (bx1, label_y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.52, (20, 20, 20), 3, cv2.LINE_AA)
                    cv2.putText(annotated, label, (bx1, label_y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.52, color, 1, cv2.LINE_AA)
                    records.append({
                        "Component": class_name,
                        "Confidence": round(confidence, 3),
                        "Bounding box (x1, y1, x2, y2)": f"{bx1}, {by1}, {bx2}, {by2}",
                        "_box": [bx1, by1, bx2, by2],
                    })

                st.session_state[result_key] = {
                    "signature": repr(signature),
                    "source_token": selected_token,
                    "image": annotated,
                    "records": records,
                    "boxes": records,
                    "model": model_id,
                    "error": None,
                }
            except Exception as error:
                st.error(
                    "Hosted inference could not run. Check your API key, model endpoint, network connection, "
                    f"and provider quota. Details: {error}"
                )

    else:
        model_path_text = st.text_input(
            "Local YOLO weights path",
            value="models/components.pt",
            key="components_local_model_path",
            help="Point this at a trained component-detection .pt file. The app never downloads weights automatically.",
        )
        local_confidence = float(confidence_threshold)
        model_path = Path(model_path_text).expanduser()
        if not model_path.is_absolute():
            model_path = Path(__file__).resolve().parent / model_path
        if importlib.util.find_spec("ultralytics") is None:
            st.warning("Local YOLO requires Ultralytics. Install it with: pip install ultralytics")
        elif not model_path.is_file():
            st.info(
                f"No weights file found at {model_path}. Download the dataset and train a model with "
                "scripts/train_components_yolo.py, or choose the hosted Roboflow backend."
            )
        run_detection = st.button(
            "Detect Components in the Whole Image",
            type="primary",
            key="components_run_local_detection",
            disabled=(importlib.util.find_spec("ultralytics") is None or not model_path.is_file()),
            width="stretch",
        )
        signature = (
            "local", selected_token, str(model_path), local_confidence,
        )
        result_key = "components_full_scene_result"

        if run_detection:
            try:
                from ultralytics import YOLO

                with st.spinner("Running local YOLO inference..."):
                    model = YOLO(str(model_path))
                    source_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                    predictions = model.predict(
                        source=source_bgr,
                        conf=local_confidence,
                        verbose=False,
                    )
                prediction = predictions[0]
                annotated = cv2.cvtColor(prediction.plot(), cv2.COLOR_BGR2RGB)
                records = []
                boxes = prediction.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0].item())
                        names = prediction.names
                        class_name = str(names.get(class_id, class_id) if isinstance(names, dict) else names[class_id])
                        confidence = float(box.conf[0].item())
                        coords = box.xyxy[0].cpu().numpy().astype(int).tolist()
                        records.append({
                            "Component": class_name,
                            "Confidence": round(confidence, 3),
                            "Bounding box (x1, y1, x2, y2)": ", ".join(map(str, coords)),
                            "_box": coords,
                        })
                st.session_state[result_key] = {
                    "signature": repr(signature),
                    "source_token": selected_token,
                    "image": annotated,
                    "records": records,
                    "boxes": records,
                    "model": str(model_path),
                    "error": None,
                }
            except Exception as error:
                st.error(f"Local YOLO detection failed: {error}")

    saved_detection = st.session_state.get("components_full_scene_result")
    if saved_detection and saved_detection.get("signature") == repr(signature):
        st.subheader("Detected Components")
        st.image(saved_detection["image"], width="stretch")
        st.metric("Components detected", len(saved_detection["records"]))
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
                    purpose = "محدود کردن جریان، تقسیم ولتاژ و تنظیم بایاس؛ مقدار اهم از روی نوارهای رنگی خوانده می‌شود."
                elif "capacitor" in lowered_name or "condens" in lowered_name:
                    purpose = "ذخیرهٔ موقت بار الکتریکی؛ برای صاف‌کردن ولتاژ، فیلتر نویز و کوپلاژ سیگنال کاربرد دارد."
                elif "diode" in lowered_name:
                    purpose = "عبور جریان عمدتاً در یک جهت؛ در یکسوسازی و محافظت از قطبیت استفاده می‌شود."
                elif "inductor" in lowered_name or "coil" in lowered_name:
                    purpose = "ذخیرهٔ انرژی در میدان مغناطیسی؛ در فیلترها و مبدل‌های تغذیه کاربرد دارد."
                elif "transistor" in lowered_name:
                    purpose = "برای سوئیچ‌کردن یا تقویت سیگنال و کنترل جریان بار استفاده می‌شود."
                elif "led" in lowered_name:
                    purpose = "تبدیل انرژی الکتریکی به نور؛ معمولاً نشانگر یا منبع روشنایی است."
                elif "potentiometer" in lowered_name or "variable resistor" in lowered_name:
                    purpose = "مقاومت قابل تنظیم برای تغییر ولتاژ یا سطح سیگنال."
                elif "crystal" in lowered_name or "oscillator" in lowered_name:
                    purpose = "تولید مرجع فرکانسی برای ساعت و زمان‌بندی مدار."
                elif "cable" in lowered_name or "wire" in lowered_name:
                    purpose = "اتصال الکتریکی برای عبور تغذیه یا سیگنال بین بخش‌های مدار."
                elif "ic" in lowered_name or "integrated circuit" in lowered_name:
                    purpose = "مدار مجتمع؛ بسته به نوعش عملیات منطق، کنترل، تقویت یا پردازش را انجام می‌دهد."
                else:
                    purpose = "مدل فقط برچسب تصویری را پیشنهاد می‌دهد؛ برای تعیین نقش دقیق، شماره قطعه و مدار باید بررسی شود."
                st.caption(f"کاربرد رایج: {purpose}")
            if len(saved_detection["records"]) > 12:
                st.caption(f"برای جلوگیری از شلوغی صفحه، توضیح {len(saved_detection['records']) - 12} تشخیص دیگر مخفی مانده است.")
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

    with st.expander("Dataset, model limitations, and training"):
        st.markdown(
            "- **Recommended public dataset:** [Electronics Components](https://universe.roboflow.com/vishnupriyas-workspace/electronics-components-sveck), published under CC BY 4.0 on its project page. The current model page lists 10 component categories.\n"
            "- The full dataset is hosted externally and should not be copied wholesale into this GitHub repository: image datasets are large, and the license requires attribution when redistributing.\n"
            "- Download/export it through the official provider, then train a small local model with scripts/train_components_yolo.py if you want inference without sending photos to a hosted service.\n"
            "- A detector identifies object classes only. Reading resistor values still requires a close crop and correct band-color reading below."
        )
        st.code(
            "python -m pip install -r requirements.txt\n"
            "python scripts/download_component_dataset.py --api-key YOUR_ROBOFLOW_KEY\n"
            "python scripts/train_components_yolo.py --data data/component_dataset/data.yaml --epochs 40\n"
            "streamlit run app.py",
            language="bash",
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

    st.subheader("2. Configure Band Reading")
    p1, p2, p3 = st.columns(3)
    band_count = p1.selectbox("Resistor band count", [4, 5], index=0, format_func=lambda value: f"{value}-band resistor", key=f"res_band_count_{roi_key}")
    mode = p2.selectbox("Band reading method", ["Automatic color detection (experimental)", "Manual color selection"], key=f"res_read_mode_{roi_key}")
    band_order = p3.selectbox("Read bands in this direction", ["Left to right", "Right to left"], key=f"res_band_order_{roi_key}")

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
            else:
                runs, all_runs = detect_band_runs(roi, int(band_count))
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
        except (ValueError, cv2.error, IndexError, TypeError) as error:
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

        if mode == "Automatic color detection (experimental)" and len(suggested_bands) == int(band_count):
            st.markdown("**Review or correct the detected colors**")
            st.caption(
                "تشخیص رنگ با نور و دوربین تغییر می‌کند. رنگ هر نوار را بررسی کن؛ "
                "مقدار اهم و فرمول پایین با همین انتخاب‌ها دوباره محاسبه می‌شود."
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
                "برای تشخیص ناقص، از Manual color selection استفاده کن و رنگ هر نوار را انتخاب کن."
            )

        annotated = current_result.get("annotated", roi)
        if mode == "Automatic color detection (experimental)" and len(display_bands) == int(band_count):
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
                digit_roles = ["رقم معنادار اول", "رقم معنادار دوم", "رقم معنادار سوم"]
                for index, color in enumerate(display_bands):
                    if index < int(band_count) - 2:
                        code = DIGIT_VALUES.get(color)
                        role = digit_roles[index]
                        meaning = (
                            f"رقم {code} وارد عدد اصلی مقاومت می‌شود."
                            if code is not None
                            else "این رنگ برای نوار رقم معتبر نیست؛ رنگ را اصلاح کن."
                        )
                        encoded = str(code) if code is not None else "نامعتبر"
                    elif index == int(band_count) - 2:
                        multiplier = MULTIPLIER_OHMS.get(color)
                        role = "ضریب / Multiplier"
                        encoded = f"×{multiplier:g}" if multiplier is not None else "نامعتبر"
                        meaning = f"عدد قبلی در {multiplier:g} ضرب می‌شود." if multiplier is not None else "رنگ ضریب معتبر نیست."
                    else:
                        tolerance = TOLERANCE_PERCENT.get(color)
                        role = "تلرانس / Tolerance"
                        encoded = f"±{tolerance:g}%" if tolerance is not None else "نامعتبر"
                        meaning = (
                            f"مقدار واقعی معمولاً تا {tolerance:g}% با مقدار اسمی اختلاف دارد."
                            if tolerance is not None
                            else "رنگ تلرانس معتبر نیست."
                        )
                    band_rows.append({
                        "نوار": index + 1,
                        "رنگ": color,
                        "کد": encoded,
                        "نقش": role,
                        "توضیح": meaning,
                    })
                st.dataframe(band_rows, width="stretch", hide_index=True)

            if corrected:
                st.info("از رنگ‌هایی که اصلاح کردی برای محاسبهٔ نتیجه استفاده شد.")
            if current_result.get("error"):
                st.warning(current_result["error"])

            decoded = None
            if len(display_bands) == int(band_count):
                try:
                    decoded = decode_band_sequence(display_bands, int(band_count))
                except ValueError as error:
                    st.warning(f"رنگ‌های انتخاب‌شده کد معتبری نمی‌سازند: {error}")

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
                st.markdown(f"**نحوهٔ محاسبه:** {formula}")
                st.caption(
                    "کاربرد مقاومت: محدود کردن جریان (مثلاً برای محافظت از LED)، تقسیم ولتاژ، "
                    "و تنظیم بایاس مدار. این مقدار از روی کد رنگ تخمین زده شده، نه اندازه‌گیری مستقیم؛ "
                    "برای مدار حساس با مولتی‌متر بررسی کن."
                )
                st.caption(
                    f"تلرانس یعنی مقدار واقعی معمولاً در بازهٔ {low} تا {high} قرار می‌گیرد، "
                    "با فرض سالم بودن قطعه و درست خواندن رنگ‌ها."
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
            "The automatic mode converts the crop from RGB to HSV, classifies likely color pixels, "
            "looks for narrow vertical color runs, and maps the resulting band sequence to standard "
            "resistor-code values. Camera white balance and lighting can cause incorrect band labels, "
            "so manual selection is included as a fallback."
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
