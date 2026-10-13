# ============================================================
# VisionLab - Computer Vision Learning Studio
# Built with Streamlit, OpenCV, NumPy, Pillow and scikit-learn
# ============================================================

# region 1: Imports and Page Configuration

from io import BytesIO
import tempfile
import os
import time

import streamlit as st
import cv2
import numpy as np

from PIL import Image
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

st.set_page_config(
    page_title="VisionLab | Computer Vision Studio",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# endregion


# region 2: Custom Styling

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    .main-title {
        font-size: 2.7rem;
        font-weight: 800;
        letter-spacing: -1px;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #8b95a5;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    div[data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.08);
        padding: 14px;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.18);
    }

    div[data-testid="stFileUploader"] {
        border-radius: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# endregion


# region 3: Reusable Helper Functions

def load_image(uploaded_file):
    """Convert an uploaded image into an RGB NumPy array."""
    if uploaded_file is None:
        return None

    try:
        return np.array(Image.open(uploaded_file).convert("RGB"))
    except Exception as error:
        st.error(f"Could not read the image: {error}")
        return None


def to_gray(image):
    """Convert an RGB image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def prepare_display_image(image):
    """Prepare an image for Streamlit display."""
    if image is None:
        return None

    if image.dtype != np.uint8:
        image = cv2.convertScaleAbs(image)

    if image.ndim == 2:
        return image

    if image.ndim == 3 and image.shape[2] == 3:
        return image

    return image


def image_to_png_bytes(image):
    """Encode an image as PNG bytes for download."""
    image = prepare_display_image(image)

    if image.ndim == 2:
        pil_image = Image.fromarray(image)
    else:
        pil_image = Image.fromarray(image)

    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    return buffer.getvalue()


def show_comparison(original, result, label="Processed Image"):
    """Display the original and processed images side by side."""
    left, right = st.columns(2)

    with left:
        st.subheader("Original Image")
        st.image(original, width="stretch")

    with right:
        st.subheader(label)
        st.image(
            prepare_display_image(result),
            clamp=True,
            width="stretch",
        )

    st.download_button(
        label="Download Result",
        data=image_to_png_bytes(result),
        file_name="visionlab_result.png",
        mime="image/png",
        key=f"download_{label.replace(' ', '_').lower()}",
    )


def upload_image_widget(label, key):
    """Create an image uploader and return its RGB array."""
    uploaded = st.file_uploader(
        label,
        type=["png", "jpg", "jpeg", "bmp", "webp"],
        key=key,
    )
    return load_image(uploaded)


def show_image_information(image):
    """Display basic image dimensions and data information."""
    height, width = image.shape[:2]
    channels = 1 if image.ndim == 2 else image.shape[2]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Width", f"{width} px")
    col2.metric("Height", f"{height} px")
    col3.metric("Channels", channels)
    col4.metric("Data Type", str(image.dtype))


def section_intro(title, description, algorithm):
    """Display a consistent introduction for each tool."""
    st.header(title)
    st.write(description)
    st.caption(f"Algorithms and methods: {algorithm}")


# endregion


# region 4: Sidebar Navigation

st.sidebar.markdown("# 🔬 VisionLab")
st.sidebar.caption("Computer Vision Learning Studio")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Image Fundamentals",
        "Image Processing",
        "Edge and Shape Detection",
        "Feature Detection",
        "Image Matching",
        "Image Segmentation",
        "Video Processing",
        "Machine Learning",
    ],
)

st.sidebar.divider()
st.sidebar.caption("Powered by OpenCV and Streamlit")

# endregion


# region 5: Home Page

if page == "Home":
    st.markdown(
        '<div class="main-title">VisionLab</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="subtitle">'
        'Explore Computer Vision algorithms interactively.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "Upload images, experiment with algorithms, compare results, "
        "and download processed images."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🖼️ Image Processing")
        st.write(
            "Reduce noise, change contrast, apply thresholding, "
            "and perform morphological operations."
        )

    with col2:
        st.subheader("🔎 Feature Analysis")
        st.write(
            "Detect corners and keypoints, then match features "
            "between two related images."
        )

    with col3:
        st.subheader("🧠 Machine Learning")
        st.write(
            "Train basic classifiers and evaluate their predictions "
            "using a sample handwritten-digit dataset."
        )

    st.divider()
    st.subheader("Available Modules")

    st.markdown(
        """
        - **Image Fundamentals:** dimensions, channels, pixels and statistics.
        - **Image Processing:** blurring, thresholding and morphology.
        - **Edge and Shape Detection:** Sobel, Laplacian, Canny and Hough.
        - **Feature Detection:** Harris, Shi-Tomasi, FAST, ORB and SIFT.
        - **Image Matching:** ORB descriptors, BFMatcher and Homography.
        - **Image Segmentation:** HSV masks, K-Means and Watershed.
        - **Video Processing:** frame extraction and background subtraction.
        - **Machine Learning:** KNN, SVM and classification metrics.
        """
    )

# endregion


# region 6: Image Fundamentals

elif page == "Image Fundamentals":
    section_intro(
        "Image Fundamentals",
        "Inspect image dimensions, color channels, pixel values and basic statistics.",
        "RGB, Grayscale, NumPy arrays and descriptive statistics",
    )

    image = upload_image_widget(
        "Upload an image",
        "fundamentals_image",
    )

    if image is not None:
        show_image_information(image)

        gray = to_gray(image)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("RGB Image")
            st.image(image, width="stretch")

        with col2:
            st.subheader("Grayscale Image")
            st.image(gray, clamp=True, width="stretch")

        st.subheader("Pixel Inspector")

        max_y, max_x = image.shape[:2]

        x = st.slider(
            "Horizontal coordinate (X)",
            0,
            max_x - 1,
            max_x // 2,
        )
        y = st.slider(
            "Vertical coordinate (Y)",
            0,
            max_y - 1,
            max_y // 2,
        )

        pixel = image[y, x]

        st.write(f"Pixel coordinate: `(x={x}, y={y})`")
        st.write(f"RGB value: `{pixel.tolist()}`")
        st.write(f"Grayscale value: `{int(gray[y, x])}`")

        st.subheader("Channel Statistics")

        stats = {
            "Red": image[:, :, 0],
            "Green": image[:, :, 1],
            "Blue": image[:, :, 2],
            "Grayscale": gray,
        }

        stat_cols = st.columns(4)

        for column, (name, channel) in zip(stat_cols, stats.items()):
            column.metric(name, f"{channel.mean():.2f}")
            column.caption(
                f"Min: {channel.min()} | Max: {channel.max()}"
            )

# endregion


# region 7: Image Processing

elif page == "Image Processing":
    section_intro(
        "Image Processing",
        "Apply common image enhancement and transformation techniques.",
        "Gaussian Blur, Median Blur, Thresholding, Histogram Equalization and Morphology",
    )

    image = upload_image_widget(
        "Upload an image",
        "processing_image",
    )

    if image is not None:
        gray = to_gray(image)

        method = st.selectbox(
            "Select a processing method",
            [
                "Grayscale",
                "Gaussian Blur",
                "Median Blur",
                "Binary Threshold",
                "Adaptive Threshold",
                "Histogram Equalization",
                "Erosion",
                "Dilation",
                "Opening",
                "Closing",
            ],
        )

        result = gray.copy()

        if method == "Gaussian Blur":
            kernel = st.slider(
                "Kernel size",
                3,
                31,
                7,
                step=2,
            )
            result = cv2.GaussianBlur(
                image,
                (kernel, kernel),
                0,
            )

        elif method == "Median Blur":
            kernel = st.slider(
                "Kernel size",
                3,
                31,
                5,
                step=2,
            )
            result = cv2.medianBlur(image, kernel)

        elif method == "Binary Threshold":
            threshold = st.slider("Threshold", 0, 255, 127)
            result = cv2.threshold(
                gray,
                threshold,
                255,
                cv2.THRESH_BINARY,
            )[1]

        elif method == "Adaptive Threshold":
            block_size = st.slider(
                "Neighborhood size",
                3,
                51,
                11,
                step=2,
            )
            constant = st.slider("Constant C", -10, 20, 2)

            result = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                block_size,
                constant,
            )

        elif method == "Histogram Equalization":
            result = cv2.equalizeHist(gray)

        elif method in ["Erosion", "Dilation", "Opening", "Closing"]:
            kernel_size = st.slider(
                "Morphology kernel size",
                3,
                21,
                5,
                step=2,
            )
            iterations = st.slider("Iterations", 1, 5, 1)

            kernel = np.ones(
                (kernel_size, kernel_size),
                dtype=np.uint8,
            )

            if method == "Erosion":
                result = cv2.erode(
                    gray,
                    kernel,
                    iterations=iterations,
                )

            elif method == "Dilation":
                result = cv2.dilate(
                    gray,
                    kernel,
                    iterations=iterations,
                )

            elif method == "Opening":
                result = cv2.morphologyEx(
                    gray,
                    cv2.MORPH_OPEN,
                    kernel,
                    iterations=iterations,
                )

            else:
                result = cv2.morphologyEx(
                    gray,
                    cv2.MORPH_CLOSE,
                    kernel,
                    iterations=iterations,
                )

        show_comparison(image, result)

        descriptions = {
            "Grayscale": "Converts RGB color values into intensity values.",
            "Gaussian Blur": "Smooths the image using a Gaussian kernel.",
            "Median Blur": "Reduces impulse noise using neighborhood medians.",
            "Binary Threshold": "Separates pixels using a fixed intensity threshold.",
            "Adaptive Threshold": "Computes local thresholds for varying illumination.",
            "Histogram Equalization": "Redistributes grayscale intensities to improve contrast.",
            "Erosion": "Shrinks bright regions in a grayscale or binary image.",
            "Dilation": "Expands bright regions in a grayscale or binary image.",
            "Opening": "Erosion followed by dilation; helps remove small bright noise.",
            "Closing": "Dilation followed by erosion; helps fill small dark gaps.",
        }

        st.info(descriptions[method])

# endregion


# region 8: Edge and Shape Detection

elif page == "Edge and Shape Detection":
    section_intro(
        "Edge and Shape Detection",
        "Find boundaries, lines and circular structures in an image.",
        "Sobel, Laplacian, Canny, Contours and Hough Transform",
    )

    image = upload_image_widget(
        "Upload an image",
        "edges_image",
    )

    if image is not None:
        gray = to_gray(image)

        method = st.selectbox(
            "Select a detection method",
            [
                "Canny",
                "Sobel",
                "Laplacian",
                "Contours",
                "Hough Lines",
                "Hough Circles",
            ],
        )

        if method == "Canny":
            col1, col2 = st.columns(2)
            low = col1.slider("Lower threshold", 0, 255, 80)
            high = col2.slider("Upper threshold", 0, 255, 160)

            result = cv2.Canny(gray, low, high)

        elif method == "Sobel":
            dx = cv2.Sobel(
                gray,
                cv2.CV_64F,
                1,
                0,
                ksize=3,
            )
            dy = cv2.Sobel(
                gray,
                cv2.CV_64F,
                0,
                1,
                ksize=3,
            )
            result = cv2.convertScaleAbs(
                cv2.magnitude(dx, dy)
            )

        elif method == "Laplacian":
            laplacian = cv2.Laplacian(
                gray,
                cv2.CV_64F,
            )
            result = cv2.convertScaleAbs(laplacian)

        elif method == "Contours":
            threshold = st.slider(
                "Binary threshold",
                0,
                255,
                120,
            )
            binary = cv2.threshold(
                gray,
                threshold,
                255,
                cv2.THRESH_BINARY,
            )[1]

            contours, _ = cv2.findContours(
                binary,
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            result = image.copy()
            cv2.drawContours(
                result,
                contours,
                -1,
                (255, 0, 0),
                2,
            )

            st.metric("Contours Found", len(contours))

        elif method == "Hough Lines":
            edges = cv2.Canny(gray, 50, 150)

            threshold = st.slider(
                "Hough threshold",
                20,
                200,
                100,
            )

            lines = cv2.HoughLinesP(
                edges,
                1,
                np.pi / 180,
                threshold,
                minLineLength=30,
                maxLineGap=10,
            )

            result = image.copy()
            line_count = 0

            if lines is not None:
                line_count = len(lines)

                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(
                        result,
                        (x1, y1),
                        (x2, y2),
                        (255, 0, 0),
                        2,
                    )

            st.metric("Lines Detected", line_count)

        else:
            blurred = cv2.medianBlur(gray, 5)

            param2 = st.slider(
                "Circle detection sensitivity",
                15,
                100,
                35,
            )

            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=25,
                param1=100,
                param2=param2,
                minRadius=5,
                maxRadius=0,
            )

            result = image.copy()
            circle_count = 0

            if circles is not None:
                circles = np.uint16(np.around(circles))
                circle_count = len(circles[0])

                for circle in circles[0]:
                    x, y, radius = map(int, circle)

                    cv2.circle(
                        result,
                        (x, y),
                        radius,
                        (0, 255, 0),
                        2,
                    )

                    cv2.circle(
                        result,
                        (x, y),
                        2,
                        (255, 0, 0),
                        3,
                    )

            st.metric("Circles Detected", circle_count)

        show_comparison(image, result)

# endregion


# region 9: Feature Detection

elif page == "Feature Detection":
    section_intro(
        "Feature Detection",
        "Detect distinctive points such as corners and local image features.",
        "Harris, Shi-Tomasi, FAST, ORB and SIFT",
    )

    image = upload_image_widget(
        "Upload an image",
        "features_image",
    )

    if image is not None:
        gray = to_gray(image)

        method = st.selectbox(
            "Select a feature detector",
            [
                "Harris Corners",
                "Shi-Tomasi Corners",
                "FAST",
                "ORB",
                "SIFT",
            ],
        )

        max_features = st.slider(
            "Maximum features",
            50,
            2000,
            500,
            step=50,
        )

        result = image.copy()
        feature_count = 0

        if method == "Harris Corners":
            response = cv2.cornerHarris(
                np.float32(gray),
                2,
                3,
                0.04,
            )

            threshold = 0.01 * response.max()
            result[response > threshold] = [255, 0, 0]
            feature_count = int(np.sum(response > threshold))

        elif method == "Shi-Tomasi Corners":
            corners = cv2.goodFeaturesToTrack(
                gray,
                maxCorners=max_features,
                qualityLevel=0.01,
                minDistance=7,
            )

            if corners is not None:
                feature_count = len(corners)

                for corner in corners:
                    x, y = corner.ravel().astype(int)
                    cv2.circle(
                        result,
                        (x, y),
                        5,
                        (255, 0, 0),
                        -1,
                    )

        elif method == "FAST":
            detector = cv2.FastFeatureDetector_create()
            keypoints = detector.detect(gray, None)
            keypoints = sorted(
                keypoints,
                key=lambda point: point.response,
                reverse=True,
            )[:max_features]

            feature_count = len(keypoints)

            result = cv2.drawKeypoints(
                image,
                keypoints,
                None,
                color=(255, 0, 0),
            )

        elif method == "ORB":
            detector = cv2.ORB_create(
                nfeatures=max_features
            )
            keypoints = detector.detect(gray, None)
            feature_count = len(keypoints)

            result = cv2.drawKeypoints(
                image,
                keypoints,
                None,
                color=(255, 0, 0),
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
            )

        else:
            if not hasattr(cv2, "SIFT_create"):
                st.error(
                    "SIFT is unavailable in this OpenCV installation."
                )
                st.stop()

            detector = cv2.SIFT_create(
                nfeatures=max_features
            )
            keypoints = detector.detect(gray, None)
            feature_count = len(keypoints)

            result = cv2.drawKeypoints(
                image,
                keypoints,
                None,
                color=(255, 0, 0),
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
            )

        st.metric("Detected Features", feature_count)
        show_comparison(image, result)

        st.caption(
            "Harris and Shi-Tomasi detect corners. FAST detects corners "
            "quickly. ORB and SIFT detect local keypoints that can also "
            "be described and matched between images."
        )

# endregion


# region 10: Image Matching

elif page == "Image Matching":
    section_intro(
        "Image Matching",
        "Compare two related images and visualize corresponding local features.",
        "ORB, BFMatcher, KNN Matching, Lowe's Ratio Test, Homography and RANSAC",
    )

    col1, col2 = st.columns(2)

    with col1:
        image_a = upload_image_widget(
            "Upload the first image",
            "matching_image_a",
        )

    with col2:
        image_b = upload_image_widget(
            "Upload the second image",
            "matching_image_b",
        )

    if image_a is not None and image_b is not None:
        st.subheader("Input Images")

        preview1, preview2 = st.columns(2)

        with preview1:
            st.image(
                image_a,
                caption="Image A",
                width="stretch",
            )

        with preview2:
            st.image(
                image_b,
                caption="Image B",
                width="stretch",
            )

        max_features = st.slider(
            "Maximum ORB features",
            100,
            3000,
            1000,
            step=100,
        )

        ratio = st.slider(
            "Lowe's ratio threshold",
            0.50,
            0.95,
            0.75,
            step=0.05,
        )

        if st.button("Run Image Matching", type="primary"):
            gray_a = to_gray(image_a)
            gray_b = to_gray(image_b)

            orb = cv2.ORB_create(nfeatures=max_features)

            keypoints_a, descriptors_a = orb.detectAndCompute(
                gray_a,
                None,
            )
            keypoints_b, descriptors_b = orb.detectAndCompute(
                gray_b,
                None,
            )

            if (
                descriptors_a is None
                or descriptors_b is None
                or len(keypoints_a) < 2
                or len(keypoints_b) < 2
            ):
                st.warning(
                    "Not enough features were found. Try more detailed "
                    "images with visible texture and corners."
                )

            else:
                matcher = cv2.BFMatcher(
                    cv2.NORM_HAMMING
                )

                pairs = matcher.knnMatch(
                    descriptors_a,
                    descriptors_b,
                    k=2,
                )

                good_matches = [
                    pair[0]
                    for pair in pairs
                    if len(pair) == 2
                    and pair[0].distance < ratio * pair[1].distance
                ]

                st.metric(
                    "Keypoints in Image A",
                    len(keypoints_a),
                )
                st.metric(
                    "Keypoints in Image B",
                    len(keypoints_b),
                )
                st.metric(
                    "Good Matches",
                    len(good_matches),
                )

                matched_image = cv2.drawMatches(
                    image_a,
                    keypoints_a,
                    image_b,
                    keypoints_b,
                    good_matches[:100],
                    None,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                )

                st.subheader("Feature Correspondences")
                st.image(
                    matched_image,
                    caption="Matching features between Image A and Image B",
                    width="stretch",
                )

                st.download_button(
                    "Download Matching Visualization",
                    data=image_to_png_bytes(matched_image),
                    file_name="image_matching.png",
                    mime="image/png",
                    key="download_matching_image",
                )

                if len(good_matches) >= 4:
                    source_points = np.float32(
                        [
                            keypoints_a[m.queryIdx].pt
                            for m in good_matches
                        ]
                    ).reshape(-1, 1, 2)

                    destination_points = np.float32(
                        [
                            keypoints_b[m.trainIdx].pt
                            for m in good_matches
                        ]
                    ).reshape(-1, 1, 2)

                    homography, mask = cv2.findHomography(
                        source_points,
                        destination_points,
                        cv2.RANSAC,
                        5.0,
                    )

                    if homography is not None and mask is not None:
                        inliers = int(mask.ravel().sum())

                        st.metric(
                            "RANSAC Inliers",
                            inliers,
                        )

                        st.write(
                            "RANSAC estimates a geometric transformation "
                            "and rejects many inconsistent feature matches."
                        )

                        st.code(
                            np.array2string(
                                homography,
                                precision=3,
                                suppress_small=True,
                            ),
                            language="text",
                        )

                        st.caption(
                            "Homography is most useful when the images "
                            "show the same mostly planar scene or a "
                            "compatible camera viewpoint."
                        )
                    else:
                        st.warning(
                            "Homography estimation failed for these images."
                        )
                else:
                    st.warning(
                        "At least four good matches are needed to attempt "
                        "Homography estimation."
                    )

    elif image_a is not None or image_b is not None:
        st.info("Upload both images to compare their visual features.")

# endregion


# region 11: Image Segmentation

elif page == "Image Segmentation":
    section_intro(
        "Image Segmentation",
        "Separate pixels or regions using color, clustering or region-based methods.",
        "HSV Color Masking, K-Means, Connected Components and Watershed",
    )

    image = upload_image_widget(
        "Upload an image",
        "segmentation_image",
    )

    if image is not None:
        method = st.selectbox(
            "Select a segmentation method",
            [
                "HSV Color Mask",
                "K-Means Clustering",
                "Connected Components",
                "Watershed",
            ],
        )

        if method == "HSV Color Mask":
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

            st.write(
                "Adjust the HSV range to select a color region. "
                "Hue uses OpenCV's range of 0 to 179."
            )

            c1, c2, c3 = st.columns(3)

            with c1:
                h_min = st.slider("Minimum Hue", 0, 179, 0)
                s_min = st.slider("Minimum Saturation", 0, 255, 70)
                v_min = st.slider("Minimum Value", 0, 255, 50)

            with c2:
                h_max = st.slider("Maximum Hue", 0, 179, 20)
                s_max = st.slider("Maximum Saturation", 0, 255, 255)
                v_max = st.slider("Maximum Value", 0, 255, 255)

            lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
            upper = np.array([h_max, s_max, v_max], dtype=np.uint8)

            mask = cv2.inRange(hsv, lower, upper)
            result = cv2.bitwise_and(image, image, mask=mask)

            left, right = st.columns(2)

            with left:
                st.subheader("Binary Color Mask")
                st.image(mask, clamp=True, width="stretch")

            with right:
                st.subheader("Selected Color Regions")
                st.image(result, width="stretch")

            st.download_button