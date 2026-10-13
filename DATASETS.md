# Datasets and Model Training

VisionLab uses two different computer-vision tasks, so train and evaluate separate models rather than mixing all labels into one model without a class mapping.

1. **Electronic component detection** finds discrete parts such as resistors, capacitors, diodes, inductors, transistors, LEDs, ICs, and potentiometers.
2. **IoT board recognition** finds development boards and educational kits such as Arduino, ESP32, and Raspberry Pi.
3. **Resistor value decoding** reads color bands after a resistor has been cropped. A generic component-detection YOLO model does not understand resistor color codes or determine resistance by itself.

## Recommended datasets

### 1. Electronics Components (general loose-part detector)

- **Source:** [Roboflow Universe — Electronics Components](https://universe.roboflow.com/vishnupriyas-workspace/electronics-components-sveck)
- **Reported size:** approximately 8,248 images.
- **Labels:** 10 component classes, including resistor, capacitor, diode, inductor, transistor, cable, crystal oscillator, IC, LED, and potentiometer.
- **License shown on the source page:** CC BY 4.0.
- **Download:** `scripts/download_component_dataset.py`; exports YOLO labels and a `data.yaml` file.

### 2. IoT_Dataset (your supplied Roboflow source)

- **Source:** [Design Project — IoT_Dataset](https://universe.roboflow.com/design-project-zeico/iot_dataset)
- **Reported size:** 209 images across seven classes: Arduino, Capacitor, IoT-Devices, Jumper Wires, Resistor, Transistor, and Ultrasonic Sensor.
- **License shown on the source page:** MIT.
- **Dataset version:** version 2. The project currently shows **Models: 0** on Roboflow Universe, so it is not a valid hosted inference model ID. Export its labels and train a local detector, or use the existing trained IoT Components Learning model in the app.
- **Download:** `scripts/download_roboflow_dataset.py --dataset iot`, format `yolov8`.

This is a useful starting point, but 209 images is small. Evaluate with independent photos from your own workbench. Do not merge this dataset blindly with the larger general-components dataset: overlapping classes such as `IoT-Devices` and specific board labels need a consistent unified label map.


### Hosted IoT model fallback (working inference endpoint)

- **Source:** [Roboflow Universe — IoT Components Learning](https://universe.roboflow.com/robertos-workspace-zpauf/iot-components-learning-ogliq)
- **Model ID:** `iot-components-learning-ogliq/1` (serverless inference API).
- **Published model-card metrics:** mAP@50 94.3%, precision 93.1%, recall 93.8% across 58 classes. These are the publisher's reported metrics, not a VisionLab independent test.
- **Usage:** the Streamlit analyzer calls this existing model through Roboflow's serverless inference SDK because the supplied `design-project-zeico/iot_dataset` has no hosted model (Model 0). Source page: https://universe.roboflow.com/robertos-workspace-zpauf/iot-components-learning-ogliq

### 3. Resistor Value Determination (your supplied Roboflow source)

- **Source:** [Resistor Color Code — Resistor Value Determination](https://universe.roboflow.com/resistor-color-code/resistor-value-determination)
- **Task:** semantic segmentation.
- **License shown on the source page:** CC BY 4.0.
- **Labels shown:** Orange, Red, Green, Yellow, Blue, Gold, Black, Brown, Violet, and several resistor-value labels.
- **Version:** the source model card references `resistor-value-determination/6` with 655 images, while the overview count displays 296 images. Verify the count in the exported version.
- **Model quality warning:** the public page reports mIoU of 5.9% for the listed model. Treat the annotations as a candidate training source; do not treat that pretrained model's predictions as ground truth.
- **Download:** `scripts/download_roboflow_dataset.py --dataset resistor-segmentation`, format `png-mask-semantic`.

This is semantic-mask data, not an ordinary bounding-box dataset. Keep every image paired with its exact mask and inspect `_classes.csv` and mask pixel IDs.

Convert the paired class masks into YOLO segmentation polygons, then fine-tune a segmentation checkpoint:

~~~powershell
python scripts/prepare_resistor_segmentation_yolo.py --input data/resistor_value_segmentation --output data/resistor_seg_yolo --background-value 0
python scripts/train_components_yolo.py --data data/resistor_seg_yolo/data.yaml --model yolo11n-seg.pt --epochs 80 --imgsz 640 --batch 4 --name resistor_bands --weights-out models/resistor_seg.pt
~~~

Review the converter's sample counts, class names, and generated labels before training. It polygonizes each connected region of each mask class, so it approximates semantic masks as instance-segmentation polygons. If the mask convention uses a different background pixel value, change `--background-value`. Semantic segmentation may identify band colors without decoding the complete resistor value; evaluate the final model separately on real photos.

### 4. Resistor Images Dataset for Segmentation (your supplied Kaggle source)

- **Source:** [Kaggle — Resistor Images Dataset for Segmentation](https://www.kaggle.com/datasets/alptugaydin/resistor-images-dataset-for-segmentation)
- **Task:** resistor-image segmentation, according to the dataset title.
- **License:** check the current Kaggle page and included metadata before redistributing images or trained derivatives. This guide does not assume a license because the page metadata could not be verified here.
- **Download:** `scripts/download_kaggle_resistor_dataset.py`. Configure Kaggle API access first. The script lists image-like files and potential annotation files after extraction.

The archive's exact annotation structure must be inspected after download. A folder of images alone is not a segmentation dataset; each image must have a matching verified mask or annotation.

If the archive contains single-channel or indexed-PNG masks that pair with images by filename stem, try the same converter after inspecting the printed file inventory:

~~~powershell
python scripts/prepare_resistor_segmentation_yolo.py --input data/resistor_kaggle_raw --output data/resistor_kaggle_yolo --background-value 0
python scripts/train_components_yolo.py --data data/resistor_kaggle_yolo/data.yaml --model yolo11n-seg.pt --epochs 80 --imgsz 640 --batch 4 --name kaggle_resistor_bands --weights-out models/kaggle_resistor_seg.pt
~~~

The converter will stop if it cannot find valid image/mask pairs. If the Kaggle dataset uses a different annotation format, do not rename arbitrary files to look like masks; inspect the actual format and write an appropriate converter first.

### 5. IoTKITs (IoT boards and educational kits)

- **Source:** [Mendeley Data — IoTKITs](https://data.mendeley.com/datasets/x5thzmkxhy/1)
- **Reported size:** 3,200 annotated images across 32 kit/board classes.
- **Examples:** Arduino Uno, Arduino Nano, Arduino Mega, ESP32, Raspberry Pi, and related educational embedded platforms.
- **Annotations:** polygon-style JSON annotations in the original dataset.
- **License shown on the source page:** CC BY 4.0.
- **Citation:** Nguyen, T.-T.; Nguyen Do, A.-T.; Vu, D.-L. (2025). *IoTKITs: A novel dataset for IoT education kit recognition*. Data in Brief, 60, 111650. https://doi.org/10.1016/j.dib.2025.111650

Download with `scripts/download_iotkits_dataset.py`. Then convert supported per-image Roboflow/LabelMe or COCO JSON annotations with `scripts/prepare_iotkits_yolo.py`.

**Important:** if the annotation converter reports an unsupported structure, do not train on unlabeled images. Import the original dataset into Roboflow, verify its labels, export in a compatible format, and use that exported configuration file.

### 6. PCB Component Detection (optional secondary source)

- **Source:** [wangshnegqi/pcb-component-detection-dataset-yolo](https://github.com/wangshnegqi/pcb-component-detection-dataset-yolo)
- **Labels:** resistor, capacitor, diode, LED, IC, connector, switch/button, header, crystal, and debug/test point.
- **License stated by the repository:** dataset images and annotations CC BY 4.0; supplied weights and repository code/docs Apache 2.0.
- **Usage:** clone or download the repository, review its license files, and try its dataset configuration with the training script.

This dataset focuses on populated PCBs, so it may not generalize to loose components on a desk or to development boards. Evaluate each task on separate real photos.

### 7. Resistor Color Band Segmentation

- **Source:** [Roboflow Universe — Resistor Color Detection](https://universe.roboflow.com/resister/resistor-color-detection)
- **Task:** instance segmentation of individual bands and edge color regions.
- **Reported classes:** Orange, Red, Green, Yellow, Blue, Black, Brown, Side-Gold, and Side-Silver.
- **License shown on the source page:** CC BY 4.0.
- **Reported version:** `resistor-color-detection/2` (the page lists 1,168 images for this export).
- **Download:** `scripts/download_roboflow_dataset.py --dataset resistor-color-bands`.

This is the real-photo dataset intended for a separate `yolo11n-seg.pt` band-color model. The class names for `Side-Gold` and `Side-Silver` are normalized in the app to Gold and Silver. Check the exported labels before training, especially whether the Roboflow export contains segmentation polygons.

### 8. Public Circuit Elements (no Roboflow key required)

- **Source:** [Hugging Face — LibreYOLO/Circuit Elements](https://huggingface.co/datasets/LibreYOLO/circuit-elements)
- **Reported size:** 772 annotated images (672 train, 64 validation, 36 test) across 45 source classes.
- **Format:** YOLO object-detection labels.
- **License shown on the dataset card:** CC BY 4.0.
- **Usage:** `.github/workflows/train-electronics-models.yml` downloads this dataset as a fallback when the Actions Roboflow secret is unavailable. `scripts/prepare_circuit_elements_yolo.py` merges spelling variants and emits a normalized YOLO dataset.

This dataset mostly shows discrete circuit parts; it is not a comprehensive IoT board dataset. Keep a separate board model when board/module recognition is the goal.

### Automated training and secret scopes

The GitHub Actions workflow can train models from publicly downloadable sources. The `ROBOFLOW_API_KEY` currently stored in Streamlit Community Cloud is **not automatically available** to GitHub Actions. To enable its Roboflow dataset downloads, add the key as a repository Actions secret named `ROBOFLOW_API_KEY` under **Settings → Secrets and variables → Actions**. Never add the actual value to Git.

The resistor-band patch classifier can be trained on synthetic augmented color patches as an offline baseline. Synthetic-only metrics are not evidence of real-camera accuracy. The Roboflow resistor band segmentation checkpoint is preferred when real annotations can be downloaded and verified.

### Dataset policy

The full raw datasets are intentionally not tracked by this repository. They can make Git history very large and require their source attribution to be retained. Downloaded assets are ignored by .gitignore; dataset metadata, training scripts, and instructions are tracked. The automated workflow publishes only the trained model checkpoints required by the app, after a training run successfully produces them.

The existing synthetic resistor pictures are for code/tests only. They are not a substitute for real resistor photographs. Collect and label real close-up images under varied light, angles, resistor body colors, and camera white balance before making accuracy claims.

## Training setup (recommended on a PC, not Streamlit Cloud)

Create a dedicated local environment for training and testing the local YOLO weights. This environment uses opencv-python rather than the headless OpenCV wheel in the Streamlit Cloud requirements, and it includes Streamlit so you can run the analyzer locally with the trained model.

### Windows PowerShell

~~~powershell
py -3.12 -m venv .venv-train
.\.venv-train\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-training.txt
~~~

### Train a component detector

1. Create a Roboflow API key at https://app.roboflow.com/settings/api. For local training, put it in the `ROBOFLOW_API_KEY` environment variable. For GitHub Actions, separately add it as an Actions repository secret.
2. Download the component dataset and train:

~~~powershell
$env:ROBOFLOW_API_KEY="YOUR_ROBOFLOW_API_KEY"
python scripts/download_component_dataset.py
python scripts/train_components_yolo.py --data data/component_dataset/data.yaml --model yolo11n.pt --epochs 60 --imgsz 640 --batch 8 --weights-out models/components.pt
~~~

### Train a detector using the supplied Roboflow IoT_Dataset

~~~powershell
$env:ROBOFLOW_API_KEY="YOUR_ROBOFLOW_API_KEY"
python scripts/download_roboflow_dataset.py --dataset iot
python scripts/train_components_yolo.py --data data/iot_roboflow/data.yaml --model yolo11n.pt --epochs 100 --imgsz 640 --batch 8 --name iot_parts --weights-out models/iot_components.pt
~~~

This source has only 209 images, so 100 epochs is a starting experiment, not a promise of good accuracy. Review validation metrics and test on photos that were not in the dataset.

### Download the supplied resistor semantic-segmentation dataset

~~~powershell
$env:ROBOFLOW_API_KEY="YOUR_ROBOFLOW_API_KEY"
python scripts/download_roboflow_dataset.py --dataset resistor-segmentation
~~~

This export contains semantic masks. Inspect the folder layout, class mapping, and pixel IDs first. Do not train it as ordinary object detection. Use a segmentation architecture and loader compatible with these masks.

### Download the supplied Kaggle resistor dataset

~~~powershell
python scripts/download_kaggle_resistor_dataset.py
~~~

Configure Kaggle API access as described below, then inspect the printed annotation file list and check the dataset's license. The download script does not guess which files are masks.

### Configure Kaggle downloads

1. Sign in to Kaggle and open your account's **Settings → API** section.
2. Create/download your Kaggle API token using Kaggle's current instructions.
3. Configure the token using Kaggle's documented method for your operating system. Never commit `kaggle.json` or an API token to GitHub.
4. Run `python scripts/download_kaggle_resistor_dataset.py`.

If you use a Kaggle credentials file, store it in Kaggle's expected user configuration directory and restrict access to it.

If training memory is limited, reduce `--batch` to `4` or `2`. A GPU is recommended; CPU training is slower.

## Use your trained model

Run VisionLab from a machine where the weights file exists. The main Electronic Component Analyzer calls `electronics-components-sveck/7` and `iot_dataset/2` automatically through hosted inference. There is no model-backend selector in that view.

These separate weights remain useful for local experiments:

- **models/components.pt** for loose electronic components.
- **models/iot_components.pt** for the supplied Roboflow IoT_Dataset labels.
- **models/iot_boards.pt** for the separate IoTKITs board/kit labels.
- **models/resistor_seg.pt** for a YOLO segmentation model trained on converted resistor masks; select it in the resistor reader's `Local YOLO segmentation model` method.

Raw datasets and generated model weights are not committed by default because they can make the Git history large and may have redistribution requirements. The download scripts keep datasets local and preserve source attribution. For an offline deployed detector, train a model separately and deliberately arrange model hosting/versioning; hosted inference does not train a model on the downloaded data.

## Resistor color recognition is a different task

The component dataset is useful for finding a resistor but typically does not label every color band or label the full resistance code. To train a dedicated color-band model, you need images annotated at the band level or individual band crops with verified color labels. Use real photos and split train/validation/test by physical resistor or photo session to avoid almost-identical images leaking across splits. Until such a dedicated model is trained and validated, automatic band colors are estimates and must be reviewable in the UI.
