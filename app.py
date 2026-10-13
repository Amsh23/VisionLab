# VisionLab — Interactive Computer Vision Learning Studio
# Run locally with: streamlit run app.py

# region 01: Imports and Configuration
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import hashlib
import importlib.util
import os
import tempfile

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

try:
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

st.set_page_config(
    page_title="VisionLab | Computer Vision Studio",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)
# endregion


# region 02: Styling
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.7rem; padding-bottom: 3rem; max-width: 1500px; }
    .hero-title { font-size: clamp(2.2rem, 5vw, 3.5rem); font-weight: 800; letter-spacing: -1.5px; margin: 0; }
    .hero-subtitle { color: #8b95a5; font-size: 1.08rem; margin: 0.35rem 0 1.4rem 0; }
    div[data-testid="stMetric"] {
        background: rgba(128,128,128,.07); border: 1px solid rgba(128,128,128,.16);
        padding: 12px 14px; border-radius: 12px;
    }
    div[data-testid="stFileUploader"] { border-radius: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)
# endregion


# region 03: Shared Helpers
def image_to_png_bytes(image: np.ndarray) -> bytes:
    """Encode an RGB, RGBA, or grayscale uint8 image as PNG bytes."""
    prepared = display_ready_image(image)
    if prepared.dtype != np.uint8:
        prepared = np.clip(prepared, 0, 255).astype(np.uint8)
    pil_image = Image.fromarray(prepared)
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    return buffer.getvalue()


def display_ready_image(image: np.ndarray) -> np.ndarray:
    """Normalize supported arrays for st.image without changing the source."""
    array = np.asarray(image)
    if array.ndim == 2:
        return array
    if array.ndim == 3 and array.shape[2] in (3, 4):
        return array
    if array.ndim == 3 and array.shape[2] == 1:
        return array[:, :, 0]
    raise ValueError("Unsupported image shape. Expected grayscale, RGB, or RGBA data.")


def decode_uploaded_image(raw_bytes: bytes):
    """Decode image bytes safely; return RGB, RGBA, format, and source mode."""
    with Image.open(BytesIO(raw_bytes)) as opened_image:
        image_format = opened_image.format or "Unknown"
        source_mode = opened_image.mode
        has_alpha = "A" in opened_image.getbands() or "transparency" in opened_image.info
        oriented = ImageOps.exif_transpose(opened_image)
        rgba = np.array(oriented.convert("RGBA"), dtype=np.uint8, copy=True)
        rgb = np.array(oriented.convert("RGB"), dtype=np.uint8, copy=True)
    return rgb, (rgba if has_alpha else rgb.copy()), image_format, source_mode


def to_gray(image: np.ndarray) -> np.ndarray:
    """Convert RGB/RGBA/grayscale data to an 8-bit grayscale image."""
    if image.ndim == 2:
        return image.copy()
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def show_image_information(image: np.ndarray, image_format: str = "Unknown") -> None:
    """Display the most useful properties of an image array."""
    height, width = image.shape[:2]
    channels = 1 if image.ndim == 2 else image.shape[2]
    cols = st.columns(5)
    cols[0].metric("Width", f"{width:,} px")
    cols[1].metric("Height", f"{height:,} px")
    cols[2].metric("Channels", channels)
    cols[3].metric("Data type", str(image.dtype))
    cols[4].metric("Format", image_format)


def show_comparison(original: np.ndarray, result: np.ndarray, label: str, key: str) -> None:
    """Show source and output side by side, with a downloadable PNG."""
    left, right = st.columns(2)
    with left:
        st.subheader("Original Image")
        st.image(display_ready_image(original), width="stretch")
    with right:
        st.subheader(label)
        st.image(display_ready_image(result), clamp=True, width="stretch")
    st.download_button(
        "Download Processed Image",
        data=image_to_png_bytes(result),
        file_name="visionlab_result.png",
        mime="image/png",
        key=f"download_{key}",
        width="stretch",
    )


def show_code(code: str, explanation: str, title: str = "How it works and Python code") -> None:
    """Put a concise explanation and beginner-friendly source example in an expander."""
    with st.expander(title, expanded=False):
        st.write(explanation)
        st.code(code.strip(), language="python")


def section_intro(title: str, description: str, methods: str = "") -> None:
    st.header(title)
    st.write(description)
    if methods:
        st.caption(f"Topics: {methods}")


def cached_operation(
    label: str,
    storage_key: str,
    signature,
    operation,
):
    """Run only on button click, then retain results while inputs stay unchanged."""
    current_signature = repr(signature)
    if st.button(label, type="primary", key=f"button_{storage_key}", width="stretch"):
        try:
            with st.spinner("Processing image..."):
                result = operation()
            st.session_state[f"result::{storage_key}"] = {
                "signature": current_signature,
                "value": result,
            }
        except (cv2.error, ValueError, TypeError, RuntimeError, MemoryError) as error:
            st.error(f"Could not complete this operation: {error}")
            return None
        except Exception as error:
            st.error(f"Unexpected processing error: {error}")
            return None

    saved = st.session_state.get(f"result::{storage_key}")
    if saved and saved.get("signature") == current_signature:
        return saved.get("value")
    if saved:
        st.caption("Settings have changed. Run the operation again to update the result.")
    return None


def clear_shared_image() -> None:
    """Callback: clear the shared image without assigning to a file-uploader widget."""
    st.session_state["global_image_uploader_version"] = st.session_state.get("global_image_uploader_version", 0) + 1
    for key in ("original_image_bytes", "source_rgb", "source_rgba", "image_format",
                "source_mode", "source_name", "source_token"):
        st.session_state.pop(key, None)
    for key in list(st.session_state.keys()):
        if key.startswith("result::"):
            st.session_state.pop(key, None)


def image_token() -> str:
    return st.session_state.get("source_token", "no-image")


def current_source_image():
    return st.session_state.get("source_rgb")


def safe_odd_size(value: int, maximum: int) -> int:
    """Return an odd local-window size no larger than the available image edge."""
    candidate = min(int(value), int(maximum))
    if candidate % 2 == 0:
        candidate -= 1
    return candidate


def resize_image(image: np.ndarray, width: int, height: int, keep_ratio: bool) -> np.ndarray:
    if keep_ratio:
        height = max(1, round(image.shape[0] * width / image.shape[1]))
    interpolation = cv2.INTER_AREA if width < image.shape[1] else cv2.INTER_CUBIC
    return cv2.resize(image, (int(width), int(height)), interpolation=interpolation)


def draw_label(image: np.ndarray, text: str, x: int, y: int) -> None:
    """Draw readable text on an RGB image."""
    y = max(20, y)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 4, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 240, 70), 2, cv2.LINE_AA)


# endregion


# region 04: Sidebar Navigation and the Shared Image Uploader
PAGES = [
    "Home",
    "Image Fundamentals",
    "Image Processing & Filters",
    "Edge & Shape Detection",
    "Color Detection",
    "Feature Detection",
    "Image Matching",
    "Image Segmentation",
    "Object Detection (YOLO)",
    "Video Processing",
    "Machine Learning Demo",
]

st.sidebar.markdown("# 🔬 VisionLab")
st.sidebar.caption("Interactive Computer Vision Learning Studio")
st.sidebar.divider()
page = st.sidebar.radio("Navigation", PAGES, key="navigation_page")
st.sidebar.divider()
st.sidebar.markdown("### Shared Image")
st.sidebar.caption("Upload once. The same image is available in every image module.")

uploader_version = st.session_state.get("global_image_uploader_version", 0)
uploaded_image = st.sidebar.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg", "bmp", "webp", "tif", "tiff"],
    key=f"global_image_uploader_{uploader_version}",
    help="The first image is shared across all image modules. Image Matching has one additional uploader for the second image.",
)

if uploaded_image is not None:
    try:
        raw_bytes = uploaded_image.getvalue()
        existing_bytes = st.session_state.get("original_image_bytes")
        if raw_bytes != existing_bytes:
            rgb, rgba, image_format, source_mode = decode_uploaded_image(raw_bytes)
            st.session_state["original_image_bytes"] = raw_bytes
            st.session_state["source_rgb"] = rgb
            st.session_state["source_rgba"] = rgba
            st.session_state["image_format"] = image_format
            st.session_state["source_mode"] = source_mode
            st.session_state["source_name"] = uploaded_image.name
            st.session_state["source_token"] = hashlib.sha1(raw_bytes).hexdigest()[:16]
            for key in list(st.session_state.keys()):
                if key.startswith("result::"):
                    st.session_state.pop(key, None)
    except Exception as error:
        st.session_state.pop("source_rgb", None)
        st.session_state.pop("source_rgba", None)
        st.session_state.pop("original_image_bytes", None)
        st.sidebar.error(f"Unable to decode this file: {error}")

if st.sidebar.button("Clear / Reset Image", on_click=clear_shared_image, width="stretch"):
    pass

image = current_source_image()
image_format = st.session_state.get("image_format", "Unknown")
if image is not None:
    st.sidebar.success(f"Loaded: {st.session_state.get('source_name', 'image')}")
    st.sidebar.caption(f"{image.shape[1]} × {image.shape[0]} px · {image_format}")
else:
    st.sidebar.info("No image loaded yet. The other modules will guide you.")

st.sidebar.divider()
st.sidebar.caption("Core: Streamlit · OpenCV · NumPy · Pillow")
# endregion


# region 05: Home and Learning Roadmap
if page == "Home":
    st.markdown('<div class="hero-title">VisionLab</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">A hands-on lab for learning Computer Vision with Python, OpenCV, NumPy, and Streamlit.</div>',
        unsafe_allow_html=True,
    )
    st.info("Start by loading an image in the sidebar. Explore one algorithm at a time, compare the output, inspect the example code, and download your result.")
    cards = st.columns(3)
    with cards[0]:
        st.subheader("01 · Understand Images")
        st.write("Learn pixels, dimensions, color channels, grayscale conversion, resize, crop, and geometric transformations.")
    with cards[1]:
        st.subheader("02 · Process and Detect")
        st.write("Experiment with blur filters, thresholding, morphology, edges, contours, and basic shape detection.")
    with cards[2]:
        st.subheader("03 · Analyze and Segment")
        st.write("Explore HSV color masks, local features, image matching, segmentation, and optional object detection.")
    st.divider()
    st.subheader("Learning Roadmap")
    roadmap = [
        ("Image Fundamentals", "Pixels, RGB channels, grayscale, image geometry, brightness, and contrast."),
        ("Image Processing & Filters", "Gaussian, median and bilateral filtering, sharpening, thresholding, histograms and morphology."),
        ("Edge & Shape Detection", "Canny, Sobel, Laplacian, contours, polygons, Hough lines and circles."),
        ("Color Detection", "HSV ranges, binary masks, and color-based segmentation."),
        ("Feature Detection and Matching", "Harris, Shi-Tomasi, ORB, optional SIFT, BFMatcher and homography."),
        ("Image Segmentation", "Threshold segmentation, HSV masks, GrabCut foreground extraction, and K-means clustering."),
        ("Optional Extensions", "YOLO object detection, video frames, and a small scikit-learn digit-classification example."),
    ]
    for title, description in roadmap:
        st.markdown(f"**{title}** — {description}")
    st.divider()
    if image is not None:
        st.subheader("Your Shared Image")
        st.image(st.session_state.get("source_rgba", image), caption=st.session_state.get("source_name", "Uploaded image"), width="stretch")
        show_image_information(image, image_format)
        st.caption(f"Decoded from {st.session_state.get('source_mode', 'an image')} mode. The uploaded bytes are retained unchanged in the session; algorithms use a separate RGB array.")
    else:
        st.subheader("Get started")
        st.write("Use the shared image uploader in the sidebar. The original image is stored separately and is never overwritten by processing operations.")
    st.subheader("Run locally")
    st.code("pip install streamlit opencv-python numpy pillow scikit-learn matplotlib\nstreamlit run app.py", language="bash")

# endregion


# region 06: Image Fundamentals
elif page == "Image Fundamentals":
    section_intro(
        "Image Fundamentals",
        "Explore the building blocks of digital images and common geometric or intensity transformations.",
        "Grayscale, channels, resize, crop, rotate, flip, brightness, and contrast",
    )
    if image is None:
        st.info("Upload an image using the shared uploader in the sidebar to begin.")
    else:
        method = st.selectbox(
            "Choose a tool",
            ["Grayscale Conversion", "RGB Channel Viewer", "Resize Image", "Crop Image",
             "Rotate or Flip", "Brightness and Contrast"],
            key="fundamentals_method",
        )
        h, w = image.shape[:2]
        params = {}
        if method == "Resize Image":
            c1, c2, c3 = st.columns(3)
            params["width"] = c1.number_input("Output width (pixels)", 1, 10000, int(w), key="resize_width")
            params["height"] = c2.number_input("Output height (pixels)", 1, 10000, int(h), key="resize_height")
            params["keep_ratio"] = c3.checkbox("Keep aspect ratio", True, key="resize_ratio")
        elif method == "Crop Image":
            c1, c2, c3, c4 = st.columns(4)
            params["x1"] = c1.number_input("Left (x1)", 0, w - 1, 0, key="crop_x1")
            params["y1"] = c2.number_input("Top (y1)", 0, h - 1, 0, key="crop_y1")
            params["x2"] = c3.number_input("Right (x2)", 1, w, w, key="crop_x2")
            params["y2"] = c4.number_input("Bottom (y2)", 1, h, h, key="crop_y2")
            st.caption("Coordinates use a top-left origin. The right and bottom edges are exclusive.")
        elif method == "Rotate or Flip":
            params["transform"] = st.selectbox("Transformation", ["Rotate", "Flip horizontally", "Flip vertically", "Flip both ways"], key="transform_type")
            if params["transform"] == "Rotate":
                params["angle"] = st.slider("Rotation angle", -180, 180, 15, key="rotation_angle")
        elif method == "Brightness and Contrast":
            c1, c2 = st.columns(2)
            params["brightness"] = c1.slider("Brightness", -100, 100, 0, key="brightness_value")
            params["contrast"] = c2.slider("Contrast", 0.2, 3.0, 1.0, step=0.1, key="contrast_value")

        def fundamentals_operation():
            if method == "Grayscale Conversion":
                return {"image": to_gray(image)}
            if method == "RGB Channel Viewer":
                red, green, blue = cv2.split(image)
                return {"image": red, "channels": [red, green, blue], "channel_names": ["Red channel", "Green channel", "Blue channel"]}
            if method == "Resize Image":
                return {"image": resize_image(image, int(params["width"]), int(params["height"]), params["keep_ratio"])}
            if method == "Crop Image":
                x1, y1, x2, y2 = (int(params[k]) for k in ("x1", "y1", "x2", "y2"))
                if x2 <= x1 or y2 <= y1:
                    raise ValueError("Right must be greater than left, and bottom must be greater than top.")
                return {"image": image[y1:y2, x1:x2].copy()}
            if method == "Rotate or Flip":
                if params["transform"] == "Flip horizontally":
                    result = cv2.flip(image, 1)
                elif params["transform"] == "Flip vertically":
                    result = cv2.flip(image, 0)
                elif params["transform"] == "Flip both ways":
                    result = cv2.flip(image, -1)
                else:
                    center = (w / 2, h / 2)
                    matrix = cv2.getRotationMatrix2D(center, float(params["angle"]), 1.0)
                    result = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
                return {"image": result}
            if method == "Brightness and Contrast":
                adjusted = image.astype(np.float32) * float(params["contrast"]) + float(params["brightness"])
                return {"image": np.clip(adjusted, 0, 255).astype(np.uint8)}
            raise ValueError("Unknown image fundamentals operation.")

        signature = (image_token(), method, tuple(sorted(params.items())))
        result = cached_operation(
            {
                "Grayscale Conversion": "Convert to Grayscale",
                "RGB Channel Viewer": "Display RGB Channels",
                "Resize Image": "Resize Image",
                "Crop Image": "Crop Image",
                "Rotate or Flip": "Apply Rotation or Flip",
                "Brightness and Contrast": "Apply Brightness and Contrast",
            }[method],
            "fundamentals",
            signature,
            fundamentals_operation,
        )
        code_examples = {
            "Grayscale Conversion": ("gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)", "A grayscale image stores one intensity value per pixel instead of three RGB values. This is useful for many edge and threshold algorithms."),
            "RGB Channel Viewer": ("red, green, blue = cv2.split(image_rgb)", "Each channel is a 2D array of intensity values. A pixel in RGB has red, green, and blue components, usually ranging from 0 to 255."),
            "Resize Image": ("resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)", "Resizing changes the number of pixels. INTER_AREA is often a good choice when shrinking; the keep-ratio option preserves the image proportions."),
            "Crop Image": ("cropped = image[y1:y2, x1:x2]", "NumPy slices use row first and column second, so image[y1:y2, x1:x2] corresponds to (y, x) coordinates."),
            "Rotate or Flip": ("matrix = cv2.getRotationMatrix2D(center, angle, 1.0)\nrotated = cv2.warpAffine(image, matrix, (width, height))", "A geometric transform remaps pixel locations. This example rotates around the image center; the output canvas remains the original size."),
            "Brightness and Contrast": ("adjusted = np.clip(image.astype(np.float32) * contrast + brightness, 0, 255).astype(np.uint8)", "Contrast scales pixel values; brightness adds an offset. Clipping keeps the result in the valid 8-bit range."),
        }
        if result is not None:
            show_comparison(image, result["image"], method, "fundamentals_output")
            if "channels" in result:
                st.subheader("Individual RGB Channels")
                channel_columns = st.columns(3)
                for column, channel, channel_name in zip(channel_columns, result["channels"], result["channel_names"]):
                    column.image(channel, caption=channel_name, clamp=True, width="stretch")
        st.info({
            "Grayscale Conversion": "Grayscale stores brightness rather than separate red, green, and blue values.",
            "RGB Channel Viewer": "RGB is an additive color model. Each channel is a 2D intensity map for one color component.",
            "Resize Image": "Width and height define the output resolution. Keeping the aspect ratio prevents stretching.",
            "Crop Image": "Cropping selects a rectangular region from the source image. Coordinates outside the image are not accepted.",
            "Rotate or Flip": "Rotation and reflection change pixel positions without changing the original array.",
            "Brightness and Contrast": "Brightness adds an offset and contrast multiplies the pixel values. The original image stays unchanged.",
        }[method])
        code, explanation = code_examples[method]
        show_code(code, explanation)
# endregion


# region 07: Image Processing and Filters
elif page == "Image Processing & Filters":
    section_intro(
        "Image Processing & Filters",
        "Reduce noise, enhance details, separate light and dark regions, and refine binary shapes.",
        "Gaussian, median, bilateral, sharpening, thresholding, histograms, and morphology",
    )
    if image is None:
        st.info("Upload an image using the shared uploader in the sidebar to begin.")
    else:
        method = st.selectbox(
            "Choose a processing operation",
            ["Gaussian Blur", "Median Blur", "Bilateral Filter", "Sharpening",
             "Thresholding", "Histogram", "Erosion", "Dilation", "Opening", "Closing"],
            key="processing_method",
        )
        params = {}
        if method in ("Gaussian Blur", "Median Blur", "Bilateral Filter"):
            params["kernel"] = st.slider("Kernel size", 3, 31, 7 if method != "Median Blur" else 5, step=2, key="filter_kernel")
        if method == "Bilateral Filter":
            c1, c2 = st.columns(2)
            params["sigma_color"] = c1.slider("Color sigma", 10, 150, 75, key="bilateral_color")
            params["sigma_space"] = c2.slider("Space sigma", 10, 150, 75, key="bilateral_space")
        if method == "Thresholding":
            params["type"] = st.selectbox("Threshold type", ["Binary", "Binary inverse", "Adaptive", "Otsu"], key="threshold_type")
            if params["type"] in ("Binary", "Binary inverse"):
                params["threshold"] = st.slider("Threshold value", 0, 255, 127, key="threshold_value")
            if params["type"] == "Adaptive":
                params["block"] = st.slider("Neighborhood size", 3, 51, 11, step=2, key="adaptive_block")
                params["constant"] = st.slider("Constant C", -10, 20, 2, key="adaptive_constant")
        if method in ("Erosion", "Dilation", "Opening", "Closing"):
            c1, c2, c3 = st.columns(3)
            params["kernel"] = c1.slider("Kernel size", 3, 21, 5, step=2, key="morph_kernel")
            params["iterations"] = c2.slider("Iterations", 1, 5, 1, key="morph_iterations")
            params["shape"] = c3.selectbox("Kernel shape", ["Rectangle", "Ellipse", "Cross"], key="morph_shape")

        def processing_operation():
            gray = to_gray(image)
            if method == "Gaussian Blur":
                k = int(params["kernel"])
                return {"image": cv2.GaussianBlur(image, (k, k), 0)}
            if method == "Median Blur":
                return {"image": cv2.medianBlur(image, int(params["kernel"]))}
            if method == "Bilateral Filter":
                k = int(params["kernel"])
                return {"image": cv2.bilateralFilter(image, k, float(params["sigma_color"]), float(params["sigma_space"]))}
            if method == "Sharpening":
                kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
                return {"image": cv2.filter2D(image, -1, kernel)}
            if method == "Thresholding":
                if params["type"] == "Binary":
                    threshold = cv2.threshold(gray, int(params["threshold"]), 255, cv2.THRESH_BINARY)[1]
                elif params["type"] == "Binary inverse":
                    threshold = cv2.threshold(gray, int(params["threshold"]), 255, cv2.THRESH_BINARY_INV)[1]
                elif params["type"] == "Otsu":
                    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                else:
                    limit = safe_odd_size(int(params["block"]), min(gray.shape))
                    if limit < 3:
                        _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                    else:
                        threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                          cv2.THRESH_BINARY, limit, int(params["constant"]))
                return {"image": threshold}
            if method == "Histogram":
                if image.ndim == 2:
                    values = {"Grayscale": cv2.calcHist([image], [0], None, [256], [0, 256]).ravel()}
                else:
                    values = {
                        "Red": cv2.calcHist([image], [0], None, [256], [0, 256]).ravel(),
                        "Green": cv2.calcHist([image], [1], None, [256], [0, 256]).ravel(),
                        "Blue": cv2.calcHist([image], [2], None, [256], [0, 256]).ravel(),
                    }
                return {"image": image.copy(), "histogram": values}
            if method in ("Erosion", "Dilation", "Opening", "Closing"):
                shape_map = {"Rectangle": cv2.MORPH_RECT, "Ellipse": cv2.MORPH_ELLIPSE, "Cross": cv2.MORPH_CROSS}
                k = int(params["kernel"])
                kernel = cv2.getStructuringElement(shape_map[params["shape"]], (k, k))
                iterations = int(params["iterations"])
                if method == "Erosion":
                    output = cv2.erode(gray, kernel, iterations=iterations)
                elif method == "Dilation":
                    output = cv2.dilate(gray, kernel, iterations=iterations)
                elif method == "Opening":
                    output = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=iterations)
                else:
                    output = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=iterations)
                return {"image": output}
            raise ValueError("Unsupported processing operation.")

        signature = (image_token(), method, tuple(sorted(params.items())))
        result = cached_operation({
            "Gaussian Blur": "Apply Gaussian Blur",
            "Median Blur": "Apply Median Blur",
            "Bilateral Filter": "Apply Bilateral Filter",
            "Sharpening": "Apply Sharpening",
            "Thresholding": "Run Thresholding",
            "Histogram": "Calculate Image Histogram",
            "Erosion": "Apply Erosion",
            "Dilation": "Apply Dilation",
            "Opening": "Apply Morphological Opening",
            "Closing": "Apply Morphological Closing",
        }[method], "processing", signature, processing_operation)
        if result is not None:
            show_comparison(image, result["image"], "Processed Output", "processing_output")
            if "histogram" in result:
                st.subheader("Pixel Intensity Distribution")
                st.line_chart(result["histogram"], x_label="Intensity (0–255)", y_label="Pixel count")
                st.caption("A histogram counts how many pixels have each intensity. Separate curves show each RGB channel.")
        descriptions = {
            "Gaussian Blur": "A weighted Gaussian neighborhood smooths small variations and noise.",
            "Median Blur": "Each pixel is replaced by the median of its neighborhood. It is especially useful for salt-and-pepper noise.",
            "Bilateral Filter": "Smooths nearby pixels while considering color similarity, helping preserve strong edges.",
            "Sharpening": "A high-pass kernel emphasizes local intensity changes and makes details appear crisper.",
            "Thresholding": "Thresholding turns grayscale intensity into a binary mask. Otsu selects a global threshold automatically; adaptive thresholding uses local neighborhoods.",
            "Histogram": "A histogram summarizes the distribution of intensity values and can help diagnose low contrast or clipped highlights.",
            "Erosion": "Erosion shrinks bright regions and can remove small bright details.",
            "Dilation": "Dilation expands bright regions and can close small breaks.",
            "Opening": "Opening is erosion followed by dilation; it can remove small bright noise while restoring larger objects.",
            "Closing": "Closing is dilation followed by erosion; it can fill small dark holes and gaps.",
        }
        st.info(descriptions[method])
        code_examples = {
            "Gaussian Blur": "blurred = cv2.GaussianBlur(image_rgb, (7, 7), 0)",
            "Median Blur": "denoised = cv2.medianBlur(image_rgb, 5)",
            "Bilateral Filter": "smooth = cv2.bilateralFilter(image_rgb, 7, 75, 75)",
            "Sharpening": "kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])\nsharp = cv2.filter2D(image_rgb, -1, kernel)",
            "Thresholding": "gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)\n_, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)",
            "Histogram": "hist = cv2.calcHist([gray], [0], None, [256], [0, 256])",
            "Erosion": "result = cv2.erode(gray, kernel, iterations=1)",
            "Dilation": "result = cv2.dilate(gray, kernel, iterations=1)",
            "Opening": "result = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)",
            "Closing": "result = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)",
        }
        show_code(code_examples[method], descriptions[method])
# endregion


# region 08: Edge and Shape Detection
elif page == "Edge & Shape Detection":
    section_intro(
        "Edge & Shape Detection",
        "Find intensity boundaries, line segments, circular objects, and polygon-like shapes.",
        "Canny, Sobel, Laplacian, contour approximation, Hough transform",
    )
    if image is None:
        st.info("Upload an image using the shared uploader in the sidebar to begin.")
    else:
        method = st.selectbox("Detection method", ["Canny Edges", "Sobel Gradients", "Laplacian Edges", "Contours and Shapes", "Hough Lines", "Hough Circles"], key="edge_method")
        params = {}
        if method == "Canny Edges":
            c1, c2 = st.columns(2)
            params["low"] = c1.slider("Lower threshold", 0, 254, 80, key="canny_low")
            params["high"] = c2.slider("Upper threshold", 1, 255, 160, key="canny_high")
        elif method == "Sobel Gradients":
            params["kernel"] = st.select_slider("Sobel kernel size", options=[1, 3, 5, 7], value=3, key="sobel_kernel")
        elif method == "Contours and Shapes":
            c1, c2 = st.columns(2)
            params["threshold"] = c1.slider("Binary threshold", 0, 255, 120, key="contour_threshold")
            area_limit = max(1, image.shape[0] * image.shape[1] // 2)
            area_default = min(area_limit, max(10, min(500, image.shape[0] * image.shape[1] // 100)))
            params["min_area"] = c2.slider("Minimum contour area (pixels)", 0, area_limit, area_default, key="contour_min_area")
        elif method == "Hough Lines":
            c1, c2, c3 = st.columns(3)
            params["threshold"] = c1.slider("Hough votes", 10, 200, 80, key="hough_line_votes")
            params["min_length"] = c2.slider("Minimum line length", 5, max(10, image.shape[1]), min(40, max(10, image.shape[1])), key="hough_min_length")
            params["gap"] = c3.slider("Maximum line gap", 0, 50, 10, key="hough_line_gap")
        elif method == "Hough Circles":
            params["sensitivity"] = st.slider("Circle detection sensitivity", 10, 100, 35, key="hough_circle_sensitivity")
            params["min_distance"] = st.slider("Minimum distance between centers", 5, max(10, min(image.shape[:2])), max(10, min(25, min(image.shape[:2]))), key="hough_circle_distance")

        def edge_operation():
            gray = to_gray(image)
            if method == "Canny Edges":
                low, high = int(params["low"]), int(params["high"])
                if high <= low:
                    high = min(255, low + 1)
                return {"image": cv2.Canny(gray, low, high)}
            if method == "Sobel Gradients":
                k = int(params["kernel"])
                dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=k)
                dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=k)
                return {"image": cv2.convertScaleAbs(cv2.magnitude(dx, dy))}
            if method == "Laplacian Edges":
                return {"image": cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_64F))}
            if method == "Contours and Shapes":
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                binary = cv2.threshold(blurred, int(params["threshold"]), 255, cv2.THRESH_BINARY)[1]
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                output = image.copy()
                kept = 0
                shape_counts = {"Triangle": 0, "Rectangle / Quadrilateral": 0, "Circle-like": 0, "Other polygon": 0}
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area < int(params["min_area"]):
                        continue
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter <= 0:
                        continue
                    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
                    vertices = len(approx)
                    if vertices == 3:
                        shape = "Triangle"
                    elif vertices == 4:
                        shape = "Rectangle / Quadrilateral"
                    elif vertices >= 7:
                        shape = "Circle-like"
                    else:
                        shape = "Other polygon"
                    shape_counts[shape] += 1
                    kept += 1
                    cv2.drawContours(output, [approx], -1, (40, 230, 70), 2)
                    moments = cv2.moments(contour)
                    if moments["m00"]:
                        cx = int(moments["m10"] / moments["m00"])
                        cy = int(moments["m01"] / moments["m00"])
                        draw_label(output, shape, cx, cy)
                return {"image": output, "contour_count": kept, "shape_counts": shape_counts}
            if method == "Hough Lines":
                edges = cv2.Canny(gray, 50, 150)
                lines = cv2.HoughLinesP(edges, 1, np.pi / 180, int(params["threshold"]),
                                        minLineLength=int(params["min_length"]), maxLineGap=int(params["gap"]))
                output = image.copy()
                count = 0
                if lines is not None:
                    count = len(lines)
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        cv2.line(output, (x1, y1), (x2, y2), (40, 220, 60), 2)
                return {"image": output, "line_count": count}
            if method == "Hough Circles":
                blurred = cv2.medianBlur(gray, 5 if min(gray.shape) >= 5 else 3)
                circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2,
                                           minDist=float(params["min_distance"]), param1=100,
                                           param2=float(params["sensitivity"]), minRadius=2, maxRadius=0)
                output = image.copy()
                count = 0
                if circles is not None:
                    count = len(circles[0])
                    for x, y, radius in np.uint16(np.around(circles[0])):
                        cv2.circle(output, (int(x), int(y)), int(radius), (20, 230, 70), 2)
                        cv2.circle(output, (int(x), int(y)), 2, (255, 70, 30), 3)
                return {"image": output, "circle_count": count}
            raise ValueError("Unknown edge detection operation.")

        signature = (image_token(), method, tuple(sorted(params.items())))
        result = cached_operation({
            "Canny Edges": "Run Canny Edge Detection",
            "Sobel Gradients": "Calculate Sobel Gradients",
            "Laplacian Edges": "Run Laplacian Edge Detection",
            "Contours and Shapes": "Detect Contours and Shapes",
            "Hough Lines": "Detect Hough Lines",
            "Hough Circles": "Detect Hough Circles",
        }[method], "edges", signature, edge_operation)
        explanations = {
            "Canny Edges": "Canny smooths noise, estimates gradients, and uses two thresholds with edge tracking to create a binary edge map.",
            "Sobel Gradients": "Sobel estimates horizontal and vertical intensity derivatives. Their magnitude highlights strong changes in either direction.",
            "Laplacian Edges": "The Laplacian estimates a second derivative, highlighting rapid intensity changes in all directions.",
            "Contours and Shapes": "Contours trace connected boundaries. Polygon approximation reduces points so simple shapes can be described by their number of vertices. Labels are heuristic, not a general object recognizer.",
            "Hough Lines": "The probabilistic Hough transform groups edge pixels into line segments.",
            "Hough Circles": "The Hough circle transform searches for circular patterns and is sensitive to contrast and its sensitivity threshold.",
        }
        if result is not None:
            show_comparison(image, result["image"], "Detection Result", "edges_output")
            if "contour_count" in result:
                st.metric("Contours kept", result["contour_count"])
                st.write(result["shape_counts"])
            if "line_count" in result:
                st.metric("Lines detected", result["line_count"])
            if "circle_count" in result:
                st.metric("Circles detected", result["circle_count"])
        st.info(explanations[method])
        examples = {
            "Canny Edges": "gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)\nedges = cv2.Canny(gray, 80, 160)",
            "Sobel Gradients": "gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)\ndx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)\ndy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)\ngradient = cv2.convertScaleAbs(cv2.magnitude(dx, dy))",
            "Laplacian Edges": "gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)\nedges = cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_64F))",
            "Contours and Shapes": "contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)\nfor contour in contours:\n    polygon = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)",
            "Hough Lines": "edges = cv2.Canny(gray, 50, 150)\nlines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, minLineLength=40, maxLineGap=10)",
            "Hough Circles": "circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 25, param1=100, param2=35)",
        }
        show_code(examples[method], explanations[method])
# endregion


# region 09: HSV Color Detection and Segmentation
elif page == "Color Detection":
    section_intro(
        "Color Detection",
        "Select pixels by color and inspect the binary mask and extracted regions.",
        "RGB-to-HSV conversion, inRange masks, color presets",
    )
    if image is None:
        st.info("Upload an image using the shared uploader in the sidebar to begin.")
    else:
        preset = st.selectbox("Color preset", ["Red", "Orange", "Yellow", "Green", "Blue", "Purple", "Custom HSV range"], key="color_preset")
        presets = {
            "Red": ((0, 80, 50), (10, 255, 255)),
            "Orange": ((11, 80, 50), (24, 255, 255)),
            "Yellow": ((25, 70, 50), (35, 255, 255)),
            "Green": ((36, 50, 40), (85, 255, 255)),
            "Blue": ((90, 60, 40), (135, 255, 255)),
            "Purple": ((136, 40, 40), (160, 255, 255)),
        }
        if preset == "Custom HSV range":
            c1, c2, c3 = st.columns(3)
            h_low = c1.slider("Minimum Hue", 0, 179, 0, key="custom_h_low")
            h_high = c1.slider("Maximum Hue", 0, 179, 20, key="custom_h_high")
            s_low = c2.slider("Minimum Saturation", 0, 255, 70, key="custom_s_low")
            s_high = c2.slider("Maximum Saturation", 0, 255, 255, key="custom_s_high")
            v_low = c3.slider("Minimum Value", 0, 255, 50, key="custom_v_low")
            v_high = c3.slider("Maximum Value", 0, 255, 255, key="custom_v_high")
            lower = (h_low, s_low, v_low)
            upper = (h_high, s_high, v_high)
        else:
            lower, upper = presets[preset]
            c1, c2 = st.columns(2)
            s_adjust = c1.slider("Minimum saturation", 0, 255, lower[1], key="preset_saturation")
            v_adjust = c2.slider("Minimum brightness (value)", 0, 255, lower[2], key="preset_value")
            lower = (lower[0], s_adjust, v_adjust)

        def color_operation():
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            low_array = np.array(lower, dtype=np.uint8)
            high_array = np.array(upper, dtype=np.uint8)
            if preset == "Red":
                red_tail = cv2.inRange(hsv, np.array([170, lower[1], lower[2]], dtype=np.uint8),
                                       np.array([179, upper[1], upper[2]], dtype=np.uint8))
                mask = cv2.bitwise_or(cv2.inRange(hsv, low_array, high_array), red_tail)
            elif lower[0] <= upper[0]:
                mask = cv2.inRange(hsv, low_array, high_array)
            else:
                # A wrapped hue range, useful for custom red-like colors.
                first = cv2.inRange(hsv, low_array, np.array([179, upper[1], upper[2]], dtype=np.uint8))
                second = cv2.inRange(hsv, np.array([0, lower[1], lower[2]], dtype=np.uint8), high_array)
                mask = cv2.bitwise_or(first, second)
            return {"mask": mask, "image": cv2.bitwise_and(image, image, mask=mask)}

        signature = (image_token(), preset, tuple(lower), tuple(upper))
        result = cached_operation("Segment Selected Color", "color", signature, color_operation)
        st.caption("OpenCV uses H in 0–179 and S/V in 0–255. HSV separates hue from brightness and saturation, which often makes color selection easier than independent RGB thresholds.")
        if result is not None:
            left, right = st.columns(2)
            with left:
                st.subheader("Binary Color Mask")
                st.image(result["mask"], clamp=True, width="stretch")
            with right:
                st.subheader("Segmented Output")
                st.image(result["image"], width="stretch")
            st.download_button("Download Color Mask", image_to_png_bytes(result["mask"]), "color_mask.png", "image/png", key="download_color_mask", width="stretch")
            st.download_button("Download Segmented Image", image_to_png_bytes(result["image"]), "color_segment.png", "image/png", key="download_color_segment", width="stretch")
        show_code(
            "hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)\nmask = cv2.inRange(hsv, lower_hsv, upper_hsv)\nsegmented = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)",
            "The HSV conversion changes color representation; inRange marks pixels within the inclusive lower and upper bounds. The mask is white where selected and black elsewhere."
        )
# endregion


# region 10: Feature Detection
elif page == "Feature Detection":
    section_intro(
        "Feature Detection",
        "Find corners and local keypoints that can help describe an image or match it to another view.",
        "Harris, Shi-Tomasi, ORB, and optional SIFT",
    )
    if image is None:
        st.info("Upload an image using the shared uploader in the sidebar to begin.")
    else:
        method = st.selectbox("Feature detector", ["Harris Corners", "Shi-Tomasi Corners", "ORB Keypoints", "SIFT Keypoints"], key="feature_method")
        params = {"maximum": st.slider("Maximum features", 50, 2500, 500, step=50, key="feature_maximum")}
        if method == "Harris Corners":
            params["strength"] = st.slider("Relative response threshold", 0.001, 0.10, 0.01, step=0.001, key="harris_threshold")
        if method == "Shi-Tomasi Corners":
            params["quality"] = st.slider("Corner quality level", 0.001, 0.20, 0.01, step=0.001, key="shitomasi_quality")
            params["distance"] = st.slider("Minimum distance between corners", 1, 30, 7, key="shitomasi_distance")

        def feature_operation():
            gray = to_gray(image)
            output = image.copy()
            if method == "Harris Corners":
                response = cv2.cornerHarris(np.float32(gray), 2, 3, 0.04)
                local_max = response == cv2.dilate(response, None)
                selected = local_max & (response > float(params["strength"]) * max(0.0, float(response.max())))
                ys, xs = np.where(selected)
                if len(xs) > int(params["maximum"]):
                    order = np.argsort(response[ys, xs])[-int(params["maximum"]):]
                    xs, ys = xs[order], ys[order]
                for x, y in zip(xs, ys):
                    cv2.circle(output, (int(x), int(y)), 3, (255, 60, 50), -1)
                return {"image": output, "count": len(xs)}
            if method == "Shi-Tomasi Corners":
                corners = cv2.goodFeaturesToTrack(gray, maxCorners=int(params["maximum"]),
                                                  qualityLevel=float(params["quality"]),
                                                  minDistance=float(params["distance"]))
                if corners is None:
                    return {"image": output, "count": 0}
                for corner in corners:
                    x, y = corner.ravel().astype(int)
                    cv2.circle(output, (x, y), 4, (255, 60, 50), -1)
                return {"image": output, "count": len(corners)}
            if method == "ORB Keypoints":
                detector = cv2.ORB_create(nfeatures=int(params["maximum"]))
            else:
                if not hasattr(cv2, "SIFT_create"):
                    raise RuntimeError("SIFT is unavailable in this OpenCV build. Replace opencv-python with opencv-contrib-python, then restart the app.")
                detector = cv2.SIFT_create(nfeatures=int(params["maximum"]))
            keypoints = detector.detect(gray, None)
            output = cv2.drawKeypoints(image, keypoints, None, color=(255, 80, 40),
                                       flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            return {"image": output, "count": len(keypoints)}

        signature = (image_token(), method, tuple(sorted(params.items())))
        result = cached_operation("Detect Corners" if "Corners" in method else "Detect Keypoints", "features", signature, feature_operation)
        explanation = {
            "Harris Corners": "Harris uses local image gradients to find locations with strong intensity changes in multiple directions. Many adjacent pixels can score highly, so this view keeps local maxima.",
            "Shi-Tomasi Corners": "Shi-Tomasi selects points whose two local gradient directions are both strong. Quality and minimum-distance controls affect how many corners remain.",
            "ORB Keypoints": "ORB detects keypoints and computes binary descriptors. It is fast, free to use, and commonly paired with a Hamming-distance matcher.",
            "SIFT Keypoints": "SIFT creates scale- and rotation-aware local features. If your OpenCV build does not expose SIFT, install opencv-contrib-python in a clean environment and restart.",
        }
        if result is not None:
            show_comparison(image, result["image"], "Detected Features", "features_output")
            st.metric("Features detected", result["count"])
            if result["count"] == 0:
                st.warning("No features were detected. Try an image with more corners, texture, or contrast.")
        st.info(explanation[method])
        examples = {
            "Harris Corners": "gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)\nresponse = cv2.cornerHarris(np.float32(gray), 2, 3, 0.04)",
            "Shi-Tomasi Corners": "corners = cv2.goodFeaturesToTrack(gray, maxCorners=500, qualityLevel=0.01, minDistance=7)",
            "ORB Keypoints": "orb = cv2.ORB_create(nfeatures=500)\nkeypoints, descriptors = orb.detectAndCompute(gray, None)",
            "SIFT Keypoints": "sift = cv2.SIFT_create(nfeatures=500)\nkeypoints, descriptors = sift.detectAndCompute(gray, None)",
        }
        show_code(examples[method], explanation[method])
# endregion


# region 11: Image Matching
elif page == "Image Matching":
    section_intro(
        "Image Matching",
        "Match local features between the shared image and a second image. The global image remains unchanged.",
        "ORB descriptors, Brute-Force Matcher, Lowe's ratio test, RANSAC homography",
    )
    if image is None:
        st.info("Upload the first image using the shared uploader in the sidebar.")
    else:
        st.subheader("Second Image")
        second_upload = st.file_uploader(
            "Choose a second image for matching",
            type=["png", "jpg", "jpeg", "bmp", "webp", "tif", "tiff"],
            key="matching_second_uploader",
        )
        second_image = None
        second_bytes = None
        if second_upload is not None:
            try:
                second_bytes = second_upload.getvalue()
                second_image, _, _, _ = decode_uploaded_image(second_bytes)
            except Exception as error:
                st.error(f"Unable to decode the second image: {error}")
        if second_image is not None:
            left, right = st.columns(2)
            with left:
                st.image(image, caption="Image A — shared image", width="stretch")
            with right:
                st.image(second_image, caption="Image B — matching image", width="stretch")
            c1, c2 = st.columns(2)
            max_features = c1.slider("Maximum ORB features", 100, 3000, 1000, step=100, key="matching_max_features")
            ratio = c2.slider("Lowe ratio threshold", 0.50, 0.95, 0.75, step=0.05, key="matching_ratio")

            def matching_operation():
                gray_a, gray_b = to_gray(image), to_gray(second_image)
                orb = cv2.ORB_create(nfeatures=int(max_features))
                keypoints_a, descriptors_a = orb.detectAndCompute(gray_a, None)
                keypoints_b, descriptors_b = orb.detectAndCompute(gray_b, None)
                if descriptors_a is None or descriptors_b is None or len(keypoints_a) < 2 or len(keypoints_b) < 2:
                    return {"error": "Not enough local features were found. Try images with more texture, corners, or detail."}
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
                pairs = matcher.knnMatch(descriptors_a, descriptors_b, k=2)
                good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < float(ratio) * pair[1].distance]
                visualization = cv2.drawMatches(image, keypoints_a, second_image, keypoints_b, good[:100], None,
                                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                inliers = None
                homography_text = None
                if len(good) >= 4:
                    points_a = np.float32([keypoints_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                    points_b = np.float32([keypoints_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                    homography, mask = cv2.findHomography(points_a, points_b, cv2.RANSAC, 5.0)
                    if homography is not None and mask is not None:
                        inliers = int(mask.ravel().sum())
                        homography_text = np.array2string(homography, precision=3, suppress_small=True)
                return {"image": visualization, "keypoints_a": len(keypoints_a), "keypoints_b": len(keypoints_b),
                        "matches": len(good), "inliers": inliers, "homography": homography_text}

            signature = (image_token(), hashlib.sha1(second_bytes).hexdigest()[:16], max_features, ratio)
            result = cached_operation("Run Image Matching", "matching", signature, matching_operation)
            if result is not None:
                if "error" in result:
                    st.warning(result["error"])
                else:
                    metric_cols = st.columns(4)
                    metric_cols[0].metric("Keypoints A", result["keypoints_a"])
                    metric_cols[1].metric("Keypoints B", result["keypoints_b"])
                    metric_cols[2].metric("Good matches", result["matches"])
                    metric_cols[3].metric("RANSAC inliers", result["inliers"] if result["inliers"] is not None else "N/A")
                    st.subheader("Feature Correspondences")
                    st.image(result["image"], width="stretch")
                    st.download_button("Download Match Visualization", image_to_png_bytes(result["image"]), "image_matching.png", "image/png", key="download_matching_image", width="stretch")
                    if result["homography"]:
                        st.caption("Estimated homography matrix (Image A coordinates mapped toward Image B):")
                        st.code(result["homography"], language="text")
                    else:
                        st.info("Homography was not estimated. At least four reliable matches are usually needed.")
        else:
            st.info("Upload a second image here to enable feature matching.")
        show_code(
            "orb = cv2.ORB_create(nfeatures=1000)\nkeypoints_a, desc_a = orb.detectAndCompute(gray_a, None)\nkeypoints_b, desc_b = orb.detectAndCompute(gray_b, None)\npairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desc_a, desc_b, k=2)\ngood = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]",
            "ORB descriptors are binary, so Hamming distance is appropriate. KNN returns the two closest candidates; Lowe's ratio test keeps matches whose best candidate is clearly better than the second."
        )
# endregion


# region 12: Image Segmentation
elif page == "Image Segmentation":
    section_intro(
        "Image Segmentation",
        "Separate foreground, color regions, or dominant colors from the rest of an image.",
        "Binary thresholding, HSV masking, GrabCut, K-means",
    )
    if image is None:
        st.info("Upload an image using the shared uploader in the sidebar to begin.")
    else:
        method = st.selectbox("Segmentation method", ["Binary Threshold", "HSV Color Segmentation", "GrabCut Foreground", "K-Means Color Clustering"], key="segmentation_method")
        params = {}
        if method == "Binary Threshold":
            params["threshold"] = st.slider("Threshold value", 0, 255, 127, key="seg_threshold")
            params["inverse"] = st.checkbox("Invert selected foreground", False, key="seg_inverse")
        elif method == "HSV Color Segmentation":
            c1, c2, c3 = st.columns(3)
            params["h1"] = c1.slider("Hue minimum", 0, 179, 0, key="seg_h1")
            params["h2"] = c1.slider("Hue maximum", 0, 179, 25, key="seg_h2")
            params["s"] = c2.slider("Minimum saturation", 0, 255, 60, key="seg_s")
            params["v"] = c3.slider("Minimum value", 0, 255, 40, key="seg_v")
        elif method == "GrabCut Foreground":
            params["iterations"] = st.slider("GrabCut iterations", 1, 10, 5, key="grabcut_iterations")
            params["margin"] = st.slider("Initial rectangle margin (pixels)", 0, max(0, min(image.shape[:2]) // 4), max(0, min(10, min(image.shape[:2]) // 10)), key="grabcut_margin")
        else:
            params["clusters"] = st.slider("Number of color clusters (K)", 2, 8, 4, key="kmeans_clusters")
            params["iterations"] = st.slider("Maximum iterations", 5, 50, 20, step=5, key="kmeans_iterations")

        def segmentation_operation():
            gray = to_gray(image)
            if method == "Binary Threshold":
                threshold_type = cv2.THRESH_BINARY_INV if params["inverse"] else cv2.THRESH_BINARY
                mask = cv2.threshold(gray, int(params["threshold"]), 255, threshold_type)[1]
                return {"mask": mask, "image": cv2.bitwise_and(image, image, mask=mask)}
            if method == "HSV Color Segmentation":
                hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                lower = np.array([params["h1"], params["s"], params["v"]], dtype=np.uint8)
                upper = np.array([params["h2"], 255, 255], dtype=np.uint8)
                if params["h1"] <= params["h2"]:
                    mask = cv2.inRange(hsv, lower, upper)
                else:
                    mask = cv2.bitwise_or(
                        cv2.inRange(hsv, lower, np.array([179, 255, 255], dtype=np.uint8)),
                        cv2.inRange(hsv, np.array([0, params["s"], params["v"]], dtype=np.uint8), upper),
                    )
                return {"mask": mask, "image": cv2.bitwise_and(image, image, mask=mask)}
            if method == "GrabCut Foreground":
                h, w = gray.shape
                margin = int(params["margin"])
                if w < 3 or h < 3 or w - 2 * margin < 2 or h - 2 * margin < 2:
                    raise ValueError("The image is too small for this GrabCut rectangle. Use a larger image or reduce the margin.")
                mask = np.zeros((h, w), dtype=np.uint8)
                background_model = np.zeros((1, 65), dtype=np.float64)
                foreground_model = np.zeros((1, 65), dtype=np.float64)
                rectangle = (margin, margin, w - 2 * margin, h - 2 * margin)
                bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.grabCut(bgr, mask, rectangle, background_model, foreground_model,
                            int(params["iterations"]), cv2.GC_INIT_WITH_RECT)
                foreground = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
                return {"mask": foreground, "image": cv2.bitwise_and(image, image, mask=foreground)}
            if method == "K-Means Color Clustering":
                h, w = image.shape[:2]
                scale = min(1.0, 400.0 / max(h, w))
                small = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
                pixels = small.reshape((-1, 3)).astype(np.float32)
                k = min(int(params["clusters"]), len(pixels))
                if len(pixels) < 2:
                    raise ValueError("Image is too small for color clustering.")
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, int(params["iterations"]), 1.0)
                _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
                clustered_small = centers[labels.flatten()].reshape(small.shape).astype(np.uint8)
                clustered = cv2.resize(clustered_small, (w, h), interpolation=cv2.INTER_NEAREST)
                return {"mask": None, "image": clustered, "cluster_count": k}
            raise ValueError("Unknown segmentation method.")

        signature = (image_token(), method, tuple(sorted(params.items())))
        result = cached_operation({
            "Binary Threshold": "Run Binary Threshold Segmentation",
            "HSV Color Segmentation": "Segment Selected HSV Color",
            "GrabCut Foreground": "Extract Foreground with GrabCut",
            "K-Means Color Clustering": "Run K-Means Color Clustering",
        }[method], "segmentation", signature, segmentation_operation)
        if result is not None:
            if result["mask"] is not None:
                left, right = st.columns(2)
                with left:
                    st.subheader("Segmentation Mask")
                    st.image(result["mask"], clamp=True, width="stretch")
                    st.download_button("Download Mask", image_to_png_bytes(result["mask"]), "segmentation_mask.png", "image/png", key="download_segmentation_mask", width="stretch")
                with right:
                    st.subheader("Segmented Output")
                    st.image(result["image"], width="stretch")
            show_comparison(image, result["image"], "Segmented Output", "segmentation_output")
            if "cluster_count" in result:
                st.metric("Color clusters", result["cluster_count"])
        descriptions = {
            "Binary Threshold": "A global intensity threshold separates lighter from darker pixels. It works best when foreground and background brightness differ.",
            "HSV Color Segmentation": "HSV masking selects colors within a hue/saturation/value range. It is often easier to tune for a target color than RGB thresholds.",
            "GrabCut Foreground": "GrabCut estimates foreground and background using color models and graph cuts. The initial rectangle assumes the object is mostly inside the rectangle; complex scenes may need refinement.",
            "K-Means Color Clustering": "K-means groups pixels by color into K clusters. This produces a simplified color image, not semantic object labels.",
        }
        st.info(descriptions[method])
        examples = {
            "Binary Threshold": "gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)\nmask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]",
            "HSV Color Segmentation": "hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)\nmask = cv2.inRange(hsv, lower_hsv, upper_hsv)\nsegmented = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)",
            "GrabCut Foreground": "mask = np.zeros(image.shape[:2], np.uint8)\nbgd_model = np.zeros((1, 65), np.float64)\nfgd_model = np.zeros((1, 65), np.float64)\ncv2.grabCut(image_bgr, mask, rectangle, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)",
            "K-Means Color Clustering": "pixels = image_rgb.reshape((-1, 3)).astype(np.float32)\n_, labels, centers = cv2.kmeans(pixels, K, None, criteria, 3, cv2.KMEANS_PP_CENTERS)\nclustered = centers[labels.flatten()].reshape(image_rgb.shape).astype(np.uint8)",
        }
        show_code(examples[method], descriptions[method])
# endregion


# region 13: Optional YOLO Object Detection
elif page == "Object Detection (YOLO)":
    section_intro(
        "Object Detection (YOLO)",
        "Detect and label objects using local YOLO weights. This module never downloads model weights automatically.",
        "Ultralytics YOLO inference and confidence filtering",
    )
    if image is None:
        st.info("Upload an image using the shared uploader in the sidebar to begin.")
    else:
        weights_path = st.text_input("Local model weights path", value="models/yolo11n.pt", help="Enter the path to a model file that already exists on this computer. The app will not trigger a download.")
        confidence = st.slider("Confidence threshold", 0.05, 0.95, 0.25, step=0.05, key="yolo_confidence")
        if importlib.util.find_spec("ultralytics") is None:
            st.warning("Optional dependency not installed. Install it with: pip install ultralytics")
            st.write("After installation, place a compatible YOLO weights file on disk (for example, at models/yolo11n.pt) and enter its path above.")
        elif not Path(weights_path).expanduser().is_file():
            st.warning(f"Model weights were not found at: {weights_path}")
            st.write("Download a compatible weights file separately from the official Ultralytics model resources, save it locally, and enter the exact path here. No model is loaded or downloaded until a valid local file is available.")
        else:
            def yolo_operation():
                from ultralytics import YOLO
                weights = str(Path(weights_path).expanduser())
                if not Path(weights).is_file():
                    raise FileNotFoundError(f"Model weights do not exist: {weights}")
                model = YOLO(weights)
                source_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                predictions = model.predict(source=source_bgr, conf=float(confidence), verbose=False)
                prediction = predictions[0]
                plotted_bgr = prediction.plot()
                plotted_rgb = cv2.cvtColor(plotted_bgr, cv2.COLOR_BGR2RGB)
                number_detected = 0 if prediction.boxes is None else len(prediction.boxes)
                return {"image": plotted_rgb, "count": number_detected, "names": prediction.names}
            signature = (image_token(), str(Path(weights_path).expanduser()), confidence)
            result = cached_operation("Run YOLO Object Detection", "yolo", signature, yolo_operation)
            if result is not None:
                show_comparison(image, result["image"], "Detected Objects", "yolo_output")
                st.metric("Detections", result["count"])
                st.caption("A prediction count of zero means the model ran but did not find objects above the selected confidence threshold.")
        show_code(
            "from ultralytics import YOLO\nmodel = YOLO('models/yolo11n.pt')  # local weights file\nresults = model.predict(source=image_rgb, conf=0.25, verbose=False)\nannotated_bgr = results[0].plot()\nannotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)",
            "The code loads only a local weights file, performs inference, and converts Ultralytics' annotated BGR output back to RGB for Streamlit. A model file and the optional ultralytics package are both required."
        )
# endregion


# region 14: Video Processing
elif page == "Video Processing":
    section_intro(
        "Video Processing",
        "Upload a video, select a frame, and compare it with a grayscale or Canny-processed version.",
        "OpenCV VideoCapture, frame seeking, grayscale, Canny edge maps",
    )
    video_upload = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi", "mkv", "webm"], key="video_uploader")
    if video_upload is None:
        st.info("The video uploader appears only in this module. Upload a supported video to extract a frame.")
    else:
        video_bytes = video_upload.getvalue()
        st.session_state["video_bytes"] = video_bytes
        st.session_state["video_name"] = video_upload.name
        st.video(video_bytes)
        suffix = Path(video_upload.name).suffix or ".mp4"

        def inspect_video():
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
                handle.write(video_bytes)
                video_path = handle.name
            try:
                capture = cv2.VideoCapture(video_path)
                if not capture.isOpened():
                    raise ValueError("OpenCV could not open this video. Try converting it to MP4 (H.264).")
                frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = float(capture.get(cv2.CAP_PROP_FPS))
                width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                capture.release()
                return frame_count, fps, width, height
            finally:
                try:
                    os.remove(video_path)
                except OSError:
                    pass

        try:
            frame_count, fps, video_width, video_height = inspect_video()
            if frame_count <= 0:
                st.warning("No frames were found. The file may be unsupported or corrupted.")
            else:
                metric_cols = st.columns(4)
                metric_cols[0].metric("Frames", frame_count)
                metric_cols[1].metric("FPS", f"{fps:.2f}")
                metric_cols[2].metric("Width", f"{video_width} px")
                metric_cols[3].metric("Height", f"{video_height} px")
                c1, c2 = st.columns(2)
                frame_index = c1.slider("Frame index", 0, frame_count - 1, min(frame_count // 2, frame_count - 1), key="video_frame_index")
                mode = c2.selectbox("Frame processing", ["Original frame", "Grayscale", "Canny edges"], key="video_frame_mode")
                low = 80
                high = 160
                if mode == "Canny edges":
                    c3, c4 = st.columns(2)
                    low = c3.slider("Canny lower threshold", 0, 254, 80, key="video_canny_low")
                    high = c4.slider("Canny upper threshold", 1, 255, 160, key="video_canny_high")

                def extract_frame():
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
                        handle.write(video_bytes)
                        video_path = handle.name
                    try:
                        capture = cv2.VideoCapture(video_path)
                        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                        ok, frame_bgr = capture.read()
                        capture.release()
                        if not ok or frame_bgr is None:
                            raise ValueError("The selected frame could not be decoded.")
                        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                        if mode == "Grayscale":
                            processed = gray
                        elif mode == "Canny edges":
                            processed = cv2.Canny(gray, int(low), max(int(low) + 1, int(high)))
                        else:
                            processed = frame_rgb
                        return {"frame": frame_rgb, "processed": processed}
                    finally:
                        try:
                            os.remove(video_path)
                        except OSError:
                            pass

                signature = (hashlib.sha1(video_bytes).hexdigest()[:16], frame_index, mode, low, high)
                result = cached_operation("Extract and Process Video Frame", "video", signature, extract_frame)
                if result is not None:
                    show_comparison(result["frame"], result["processed"], mode, "video_output")
                    st.download_button("Download Extracted Frame", image_to_png_bytes(result["frame"]), "video_frame.png", "image/png", key="download_video_frame", width="stretch")
                st.info("Video frame extraction uses a temporary file because OpenCV's VideoCapture expects a path or camera stream. Large videos may take longer to inspect; Streamlit's built-in video player handles playback separately.")
        except (cv2.error, ValueError, OSError) as error:
            st.error(f"Video processing could not start: {error}")
    show_code(
        "capture = cv2.VideoCapture('input.mp4')\ncapture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)\nok, frame_bgr = capture.read()\ncapture.release()\nif ok:\n    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)\n    edges = cv2.Canny(gray, 80, 160)",
        "VideoCapture reads frames sequentially or seeks to an approximate frame index. Many compressed formats do not support perfect random access, so a selected frame can be approximate on some codecs."
    )
# endregion


# region 15: Machine Learning Demo
elif page == "Machine Learning Demo":
    section_intro(
        "Machine Learning Demo",
        "Train a simple classifier on scikit-learn's built-in handwritten digits dataset and inspect its evaluation.",
        "K-nearest neighbors, support vector machines, accuracy, classification report, confusion matrix",
    )
    if not SKLEARN_AVAILABLE:
        st.warning("The optional scikit-learn dependency is not installed. Install it with: pip install scikit-learn")
    else:
        model_name = st.selectbox("Classifier", ["K-Nearest Neighbors (KNN)", "Support Vector Machine (SVM)"], key="ml_model")
        test_size = st.slider("Test set proportion", 0.10, 0.40, 0.20, step=0.05, key="ml_test_size")
        neighbors = st.slider("K neighbors (KNN only)", 1, 15, 5, key="ml_neighbors", disabled=(model_name != "K-Nearest Neighbors (KNN)"))
        def train_digits_model():
            dataset = load_digits()
            x_train, x_test, y_train, y_test = train_test_split(
                dataset.data, dataset.target, test_size=float(test_size), random_state=42, stratify=dataset.target
            )
            model = KNeighborsClassifier(n_neighbors=int(neighbors)) if model_name == "K-Nearest Neighbors (KNN)" else SVC(kernel="rbf", gamma="scale")
            model.fit(x_train, y_train)
            predictions = model.predict(x_test)
            return {
                "accuracy": float(accuracy_score(y_test, predictions)),
                "report": classification_report(y_test, predictions, output_dict=False, zero_division=0),
                "matrix": confusion_matrix(y_test, predictions),
                "train_count": len(x_train),
                "test_count": len(x_test),
            }
        signature = (model_name, test_size, neighbors if model_name == "K-Nearest Neighbors (KNN)" else None)
        ml_result = cached_operation("Train and Evaluate Classifier", "machine_learning", signature, train_digits_model)
        if ml_result is not None:
            m1, m2, m3 = st.columns(3)
            m1.metric("Test accuracy", f"{ml_result['accuracy']:.1%}")
            m2.metric("Training samples", ml_result["train_count"])
            m3.metric("Test samples", ml_result["test_count"])
            st.subheader("Confusion Matrix")
            st.dataframe(ml_result["matrix"], width="stretch")
            st.subheader("Classification Report")
            st.code(ml_result["report"], language="text")
        st.info("Each sample in this classic dataset is an 8 × 8 grayscale image represented by 64 numeric features. This small exercise demonstrates a training/test split and evaluation; it is not a general-purpose image classifier.")
        show_code(
            "from sklearn.datasets import load_digits\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.neighbors import KNeighborsClassifier\nfrom sklearn.metrics import accuracy_score\n\ndigits = load_digits()\nX_train, X_test, y_train, y_test = train_test_split(\n    digits.data, digits.target, test_size=0.2, random_state=42, stratify=digits.target\n)\nmodel = KNeighborsClassifier(n_neighbors=5)\nmodel.fit(X_train, y_train)\npredictions = model.predict(X_test)\nprint(accuracy_score(y_test, predictions))",
            "The model learns from the training split and is evaluated on held-out samples. Stratification keeps class proportions similar in both splits."
        )
# endregion
