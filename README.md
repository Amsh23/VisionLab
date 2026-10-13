# VisionLab — Computer Vision Learning Studio

VisionLab is a Streamlit application for learning classical OpenCV algorithms, feature matching, segmentation, video processing, machine learning, and electronic component detection.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Electronic Component Analyzer

Open **Electronic Component Analyzer** from the sidebar.

### Webcam or uploaded image

- Select **Take a webcam photo** to open your browser's camera capture widget, point it at components on your workbench, and take a photo.
- Or select **Use shared image** to analyze the image already loaded in the sidebar.
- The camera widget captures a still image for analysis; it does not continuously process live video.

### Detect components in the full image

The recommended detector is the hosted Roboflow model **electronics-components-sveck/7**. The project's dataset page currently lists classes such as resistor, capacitor, diode, inductor, transistor, IC, LED, cable, crystal oscillator, and potentiometer. The published dataset page lists CC BY 4.0. Model performance figures shown on third-party pages are not guarantees for images from your desk. The dataset/model may work best on images similar to its training data, so verify results on your own camera and setup.

To use the hosted model:

1. Create a Roboflow account and obtain your API key at https://app.roboflow.com/settings/api.
2. Run locally and paste the key into the password field, or add it to Streamlit Community Cloud under **App settings → Secrets**:

```toml
ROBOFLOW_API_KEY = "your_roboflow_api_key"
```

3. The selected image is sent to the hosted model for inference. Do not upload confidential images, and check the provider's current terms and quota.

No API key is committed to this repository.

### Local detector and custom training

If you prefer not to send images to a hosted service, download the public dataset and train your own local YOLO weights. The full dataset is intentionally not committed to this repository because it contains thousands of images and should be downloaded/exported from its source, with license attribution preserved.

Optional local training dependencies:

```bash
python -m pip install roboflow ultralytics
```

Set an environment variable rather than putting the API key directly in source code:

**Windows PowerShell**
```powershell
$env:ROBOFLOW_API_KEY="your_roboflow_api_key"
python scripts/download_component_dataset.py
```

**macOS / Linux**
```bash
export ROBOFLOW_API_KEY="your_roboflow_api_key"
python scripts/download_component_dataset.py
```

Then train a small model (GPU recommended):

```bash
python scripts/train_components_yolo.py --data data/component_dataset/data.yaml --epochs 40
```

After training, copy the generated best.pt file to `models/components.pt` and select **Local YOLO weights** inside the app. The detector needs a compatible model file; the application does not silently download model weights.

## Dataset contents in this repository

- `data/resistor_color_reference.csv`: resistor band digits, multipliers, and tolerances.
- `data/synthetic_resistors/images/` and `data/synthetic_resistors/labels.csv`: 12 small synthetic examples to demonstrate the label format. They are generated illustrations, not real photographs of electronic components.
- `scripts/generate_resistor_dataset.py`: generate more synthetic resistor images for pipeline demonstrations.
- `scripts/download_component_dataset.py`: download the real public component dataset from Roboflow with your account key.
- `scripts/train_components_yolo.py`: train a local object detector.
- `resistor_analyzer.py`: webcam capture, whole-scene detection, and color-band value estimation.

A detector finding a resistor does **not** automatically measure its resistance. The crop reader estimates resistance from visible 4-band or 5-band colors, and its automatic color detector is heuristic. Use a close, well-lit image. Always check important resistor values with a multimeter.

## Deploy to Streamlit Community Cloud

1. Open https://share.streamlit.io/ and choose **Create app**.
2. Select repository **Amsh23/VisionLab**, branch **main**, and main file path **app.py**.
3. Deploy the app.
4. To enable hosted full-scene detection, open app settings and add the `ROBOFLOW_API_KEY` secret shown above.

The image-processing modules and manual resistor decoder work without the Roboflow key. Hosted object detection needs the key and network access; local YOLO mode needs Ultralytics and a compatible local .pt weights file.

## Public dataset and license

- [Electronics components dataset and YOLO model (Roboflow Universe)](https://universe.roboflow.com/vishnupriyas-workspace/electronics-components-sveck)
- The dataset page lists **CC BY 4.0**. Review the current license/attribution requirements before redistributing its data or derivative exports.
- The model is a third-party hosted model. Its predictions are not guaranteed to detect every loose component or read resistor values from every photo.
