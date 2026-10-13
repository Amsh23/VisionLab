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

**Project-specific contribution:** integrating component detection with an interactive resistor-reading workflow and understandable code explanations. The hosted component detector is a third-party Roboflow model; the project does not claim to have trained that hosted model from scratch. The default automatic color reader remains an OpenCV heuristic; an optional local YOLO segmentation model can be trained from resistor-band mask data and selected in the UI. Any decoded value should be checked with a multimeter when accuracy matters.

**Model training:** the repository includes repeatable download/prepare/train scripts for a general electronic-components dataset and a separate IoT development-board dataset. See [DATASETS.md](DATASETS.md). Large raw image archives and trained weights are kept out of ordinary Git history by default.

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

The analyzer automatically runs two hosted Roboflow models and merges overlapping same-class predictions:

- **Electronics Components, version 7** — loose parts such as resistors, capacitors, diodes, inductors, transistors, LEDs, ICs, cables, crystal oscillators, and potentiometers.
- **IoT Dataset, version 2** — a complementary set of IoT devices and learning components, including Arduino, jumper wires, resistors, transistors, capacitors, and an ultrasonic sensor.

These are separate model classes and are not a universal detector for every electronic part. Their accuracy depends on image quality and similarity to their training images. Test with independent photos from your own setup; no accuracy guarantee is implied.

The app reads the existing secret from Streamlit Community Cloud. Configure it once under **App settings → Secrets**:

```toml
ROBOFLOW_API_KEY = "your_roboflow_api_key"
```

There is no API-key input in the user interface. The selected image is sent to both hosted inference endpoints. Do not upload confidential images, and check Roboflow's current terms and quota. The key is never stored in this repository or printed into inference errors.

### Local detector and custom training

Training is optional and should be done on a PC, not in the Streamlit Cloud app environment. Use the dedicated local environment from `requirements-training.txt`; it includes Streamlit, Ultralytics, Roboflow, and the non-headless OpenCV package so the same environment can train and run local YOLO weights.

**Windows PowerShell**
```powershell
py -3.12 -m venv .venv-train
.\.venv-train\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-training.txt
```

**Train a detector for loose electronic components**
```powershell
$env:ROBOFLOW_API_KEY="your_roboflow_api_key"
python scripts/download_component_dataset.py
python scripts/train_components_yolo.py --data data/component_dataset/data.yaml --model yolo11n.pt --epochs 60 --imgsz 640 --batch 8 --weights-out models/components.pt
```

**Train a detector using the IoT_Dataset you supplied**
```powershell
$env:ROBOFLOW_API_KEY="your_roboflow_api_key"
python scripts/download_roboflow_dataset.py --dataset iot
python scripts/train_components_yolo.py --data data/iot_roboflow/data.yaml --model yolo11n.pt --epochs 100 --imgsz 640 --batch 8 --name iot_parts --weights-out models/iot_components.pt
```

**Download the resistor semantic-segmentation dataset**
```powershell
$env:ROBOFLOW_API_KEY="your_roboflow_api_key"
python scripts/download_roboflow_dataset.py --dataset resistor-segmentation
```

That export contains semantic masks, not ordinary bounding boxes. Convert the masks to YOLO segmentation polygons and train a segmentation checkpoint (not the default detection checkpoint):

```powershell
python scripts/prepare_resistor_segmentation_yolo.py --input data/resistor_value_segmentation --output data/resistor_seg_yolo --background-value 0
python scripts/train_components_yolo.py --data data/resistor_seg_yolo/data.yaml --model yolo11n-seg.pt --epochs 80 --imgsz 640 --batch 4 --name resistor_bands --weights-out models/resistor_seg.pt
```

Inspect sample labels before training and verify the mask background value. The converter approximates semantic masks as polygons and training accuracy must be evaluated on independent real photos. In the resistor reader, choose **Local YOLO segmentation model** and set `models/resistor_seg.pt` to use the trained band-color classes.

**Download the Kaggle resistor segmentation dataset**
```powershell
python scripts/download_kaggle_resistor_dataset.py
```

Configure Kaggle API access first and inspect the annotation files and license after download.

See [DATASETS.md](DATASETS.md) for all three sources, licenses, citations, download instructions, formats, and troubleshooting. Raw datasets and model weights are ignored by Git by default to avoid large repository history.

The main Electronic Component Analyzer uses the two hosted Roboflow models by default and does not expose a backend or API-key selector. The local training scripts are for experimentation on a PC; they do not automatically replace the hosted models in Streamlit Cloud. The separate **Object Detection (YOLO)** learning module can run a local weights file when Ultralytics is installed and the file is present in the app environment.

## Dataset contents in this repository

- `data/resistor_color_reference.csv`: resistor band digits, multipliers, and tolerances.
- `data/synthetic_resistors/images/` and `data/synthetic_resistors/labels.csv`: 12 small synthetic examples to demonstrate the label format. They are generated illustrations, not real photographs of electronic components.
- `scripts/generate_resistor_dataset.py`: generate more synthetic resistor images for pipeline demonstrations.
- `DATASETS.md`: public component and IoT-board datasets, licensing notes, and end-to-end training commands.
- `scripts/download_component_dataset.py`: download the larger general-components dataset from Roboflow.
- `scripts/download_roboflow_dataset.py`: download the supplied Roboflow IoT or resistor-segmentation datasets.
- `scripts/prepare_resistor_segmentation_yolo.py`: convert compatible semantic masks into YOLO segmentation polygons.
- `scripts/download_kaggle_resistor_dataset.py`: download and inventory the Kaggle resistor-segmentation archive.
- `scripts/download_iotkits_dataset.py`: download the IoTKITs embedded-board dataset from Mendeley Data.
- `scripts/prepare_iotkits_yolo.py`: convert supported IoTKITs JSON/polygon annotations to YOLO format.
- `scripts/train_components_yolo.py`: train a local YOLO detector and copy its best weights to a configurable destination.
- `resistor_analyzer.py`: webcam capture, whole-scene detection, and resistor color-band value estimation.

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
