# Datasets and Model Training

VisionLab uses two different computer-vision tasks, so train and evaluate separate models rather than mixing all labels into one model without a class mapping.

1. **Electronic component detection** finds discrete parts such as resistors, capacitors, diodes, inductors, transistors, LEDs, ICs, and potentiometers.
2. **IoT board recognition** finds development boards and educational kits such as Arduino, ESP32, and Raspberry Pi.
3. **Resistor value decoding** reads color bands after a resistor has been cropped. A generic component-detection YOLO model does not understand resistor color codes or determine resistance by itself.

## Recommended datasets

### 1. Electronics Components (primary component detector)

- **Source:** [Roboflow Universe — Electronics Components](https://universe.roboflow.com/vishnupriyas-workspace/electronics-components-sveck)
- **Reported size:** approximately 8,248 images.
- **Labels:** 10 component classes, including resistor, capacitor, diode, inductor, transistor, connection cable, crystal oscillator, IC, LED, and potentiometer.
- **License shown on the source page:** CC BY 4.0.
- **Download:** requires a Roboflow account/API key, then exports YOLOv8 labels and a data.yaml file.

The current downloader is **scripts/download_component_dataset.py**. Keep your API key in an environment variable; never commit it to GitHub.

### 2. IoTKITs (IoT boards and educational kits)

- **Source:** [Mendeley Data — IoTKITs](https://data.mendeley.com/datasets/x5thzmkxhy/1)
- **Reported size:** 3,200 annotated images across 32 kit/board classes.
- **Examples:** Arduino Uno, Arduino Nano, Arduino Mega, ESP32, Raspberry Pi, and related educational embedded platforms.
- **Annotations:** polygon-style JSON annotations in the original dataset.
- **License shown on the source page:** CC BY 4.0.
- **Citation:** Nguyen, T.-T.; Nguyen Do, A.-T.; Vu, D.-L. (2025). *IoTKITs: A novel dataset for IoT education kit recognition*. Data in Brief, 60, 111650. https://doi.org/10.1016/j.dib.2025.111650

Download with **scripts/download_iotkits_dataset.py**. Then convert the supported per-image Roboflow/LabelMe or COCO JSON annotations with **scripts/prepare_iotkits_yolo.py**. The converter writes a YOLO-format directory and a data.yaml file.

**Important:** if the converter reports that the annotation structure is unsupported, do not train on unlabeled images. Import the original dataset into Roboflow, verify its labels, export as YOLOv8, and point the training script at the exported data.yaml.

### 3. PCB Component Detection (optional secondary source)

- **Source:** [wangshnegqi/pcb-component-detection-dataset-yolo](https://github.com/wangshnegqi/pcb-component-detection-dataset-yolo)
- **Labels:** resistor, capacitor, diode, LED, IC, connector, switch/button, header, crystal, and debug/test point.
- **License stated by the repository:** dataset images and annotations CC BY 4.0; supplied weights and repository code/docs Apache 2.0.
- **Usage:** clone or download the repository, review its license files, and try its dataset/data.yaml with the training script.

This dataset focuses on populated PCBs, so it may not generalize to loose components on a desk or to development boards. Evaluate each task on separate real photos.

### Dataset policy

The full raw datasets are intentionally not tracked by this repository by default. They contain thousands of images, can make Git history very large, and need their source attribution kept with them. Downloaded assets are ignored by .gitignore; DATASETS.md, download/prepare scripts, and training instructions are tracked in GitHub instead.

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

1. Create a free Roboflow API key at https://app.roboflow.com/settings/api.
2. Download the component dataset and train:

~~~powershell
$env:ROBOFLOW_API_KEY="YOUR_ROBOFLOW_API_KEY"
python scripts/download_component_dataset.py
python scripts/train_components_yolo.py --data data/component_dataset/data.yaml --model yolo11n.pt --epochs 60 --imgsz 640 --batch 8 --weights-out models/components.pt
~~~

### Train a separate IoT-board detector

~~~powershell
python scripts/download_iotkits_dataset.py
python scripts/prepare_iotkits_yolo.py
python scripts/train_components_yolo.py --data data/iotkits_yolo/data.yaml --model yolo11n.pt --epochs 60 --imgsz 640 --batch 8 --name iot_boards --weights-out models/iot_boards.pt
~~~

If training memory is limited, reduce --batch to 4 or 2. A GPU is recommended; CPU training is slower. Training will download the initial YOLO checkpoint if it is not cached.

## Use your trained model

Run VisionLab from a machine where the weights file exists. In **Electronic Component Analyzer → Detection backend → Local YOLO weights**, set one of these paths:

- **models/components.pt** for loose electronic components.
- **models/iot_boards.pt** for IoT development boards and kits.

These are separate models and will return the class names they were trained with. Use a model with the task and image domain that match your input.

Model weights are excluded from Git by default. For Streamlit Cloud deployment, the weights must be provided separately or intentionally versioned. Do not commit a large raw dataset or a large .pt file without considering GitHub's file-size limits; use Git LFS or model storage for large assets. The app's hosted Roboflow mode can remain available without deploying a local model.

## Resistor color recognition is a different task

The component dataset is useful for finding a resistor but typically does not label every color band or label the full resistance code. To train a dedicated color-band model, you need images annotated at the band level or individual band crops with verified color labels. Use real photos and split train/validation/test by physical resistor or photo session to avoid almost-identical images leaking across splits. Until such a dedicated model is trained and validated, automatic band colors are estimates and must be reviewable in the UI.
