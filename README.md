# VisionLab — Computer Vision Learning Studio

VisionLab is a Streamlit application for experimenting with image processing, edge detection, feature matching, segmentation, video frames, and a small machine-learning demo.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Electronic Component Analyzer

The **Electronic Component Analyzer** page estimates a resistor's value from its color bands.

- Upload a photo using the shared uploader in the sidebar.
- Crop tightly around one through-hole resistor, showing the complete body and all bands.
- Rotate the crop so the resistor is horizontal.
- Try automatic OpenCV band detection, or select the band colors manually.
- Read the estimated resistance and tolerance range, and download the annotated crop.

The automatic detector is an educational heuristic, not a trained object detector. It works best with one resistor, neutral background, diffuse lighting, and the whole body in the crop. Manual color selection is provided as a fallback. A photo-derived value should be checked with a multimeter before using it in a circuit.

## Dataset

- **data/resistor_color_reference.csv**: resistor digit colors, multiplier values, tolerance values, and reference RGB swatches.
- **scripts/generate_resistor_dataset.py**: reproducible generator for synthetic resistor images plus labels.csv.

Generate 300 synthetic examples:

```bash
python scripts/generate_resistor_dataset.py --count 300 --output data/generated_resistor_dataset
```

This creates an **images/** folder and labels with the band colors, resistance in ohms, tolerance, and train/validation/test split. The images are rendered programmatically, not collected from physical components; do not use synthetic-only scores as evidence of real-world accuracy.

For a real component-localization dataset, see the [Electronic Components 8 Classes dataset on Roboflow Universe](https://universe.roboflow.com/electronic-components-dataset-for-yolo/electronic-components-8-classes). The dataset page lists a CC BY 4.0 license. Download/export the dataset yourself, review its current attribution instructions, and keep any license-required attribution if you redistribute it. A detector that locates a resistor and a color-band reader solve different problems.

## Deploy to Streamlit Community Cloud

1. Push the repository to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/) and select **Create app**.
3. Choose repository **Amsh23/VisionLab**, branch **main**, and entrypoint **app.py**.
4. Click **Deploy**.

Community Cloud installs the dependencies from the root requirements.txt. The app does not require a secret key for the built-in image-processing or resistor color-code pages. See the [official deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) for the current workflow.

## Optional YOLO module

The existing YOLO module needs the optional ultralytics package and a compatible local weights file. Install it with:

```bash
python -m pip install ultralytics
```

The app does not download model weights automatically. Large model files should be managed separately and included only after checking their size and license.
