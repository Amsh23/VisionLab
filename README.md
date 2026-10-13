# VisionLab — Electronic Component Analyzer

**The main project in VisionLab is the Electronic Component Analyzer** — a practical computer-vision application for identifying electronic components in a workbench photo and reading resistor color bands.

Instead of presenting component detection as one isolated model demo, the project combines a photo/upload workflow, hosted or local object detection, resistor-region selection, adaptive color-band analysis, editable color suggestions, resistance/tolerance calculation, and short explanations of what the detected parts are commonly used for.

The rest of VisionLab's OpenCV filters, feature matching, segmentation, video, and ML examples are supporting learning tools. They are not the project's main focus.

## Main project: Electronic Component Analyzer

### What makes this project useful
- **Photo to component list:** capture a still photo with a webcam or upload an image of the workbench.
- **Find components:** use the hosted Roboflow model, or a local YOLO weights file if you have one.
- **Read resistor bands:** analyze standard 4-band and 5-band resistor color codes from a close crop.
- **Human-reviewed color reading:** inspect or correct the suggested band colors before calculating the value.
- **Explain the code:** show each band's digit, multiplier or tolerance, the resistance formula, the expected range, and common uses of the component.
- **Learn from the output:** inspect the annotated image and download the result.

**Project-specific contribution:** integrating component detection with an interactive resistor-reading workflow and understandable code explanations. The hosted component detector is a third-party Roboflow model; the project does not claim to have trained that hosted model from scratch. Automatic color reading is a classical OpenCV heuristic, so the displayed value is an estimate and should be checked with a multimeter when accuracy matters.

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

A detector finding a resistor does **not** automatically measure its resistance. The crop reader estimates resistance from visible 4-band or 5-band colors using an adaptive Lab-contrast scan and HSV color estimates. Automatic colors can be reviewed/corrected individually; the results display each band's digit, multiplier or tolerance, the resistance formula, expected range, and a short component-use explanation. Use a tight crop, even lighting, and no direct flash glare. Because similar colors can still be confused under real camera conditions, check important resistor values with a multimeter.

## Deploy to Streamlit Community Cloud

1. Open https://share.streamlit.io/ and choose **Create app**.
2. Select repository **Amsh23/VisionLab**, branch **main**, and main file path **app.py**.
3. In **Advanced settings**, select **Python 3.12**. The dependency versions are validated in the repository's Python 3.12 CI environment.
4. Deploy the app.
5. To enable hosted full-scene detection, open app settings and add the `ROBOFLOW_API_KEY` secret shown above.

If the app already exists and it was deployed with Python 3.14, Community Cloud requires deleting and redeploying the app to change its Python version. Save its URL/subdomain and secrets first, then select Python 3.12 in Advanced settings during redeployment. See [Streamlit's Python-version instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/upgrade-python).

The image-processing modules and manual resistor decoder work without the Roboflow key. Hosted object detection needs the key and network access; local YOLO mode needs Ultralytics and a compatible local .pt weights file.

## Public dataset and license

- [Electronics components dataset and YOLO model (Roboflow Universe)](https://universe.roboflow.com/vishnupriyas-workspace/electronics-components-sveck)
- The dataset page lists **CC BY 4.0**. Review the current license/attribution requirements before redistributing its data or derivative exports.
- The model is a third-party hosted model. Its predictions are not guaranteed to detect every loose component or read resistor values from every photo.
