# Datasets for VisionLab

## Resistor color-code reference

- `resistor_color_reference.csv` maps standard band colors to digit, multiplier, common tolerance, and reference RGB values.
- `synthetic_resistors/images/` contains 12 small generated resistor illustrations. Their expected band colors and resistance values are stored in `synthetic_resistors/labels.csv`.
- Run `python scripts/generate_resistor_dataset.py --count 300 --output data/generated_resistor_dataset` to generate a larger synthetic training/demo set.

The synthetic data is useful for testing parsing and data pipelines but is not a substitute for real camera photographs.

## Full electronic component detection dataset

Recommended source: [Electronics components dataset and YOLO model on Roboflow Universe](https://universe.roboflow.com/vishnupriyas-workspace/electronics-components-sveck).

The dataset page lists approximately 8,248 images and ten component categories, including resistor, capacitor, diode, inductor, transistor, cable, crystal oscillator, IC, LED, and potentiometer. The project page lists the dataset license as CC BY 4.0; check the source page for any updates before use or redistribution.

The complete dataset is not vendored into this GitHub repository because thousands of images can make the app repo unnecessarily large, and users should retrieve the chosen version from the provider so its attribution/version information remains clear.

### Download and train

1. Create a Roboflow account and create an API key at https://app.roboflow.com/settings/api.
2. Install optional development dependencies with `python -m pip install roboflow ultralytics`.
3. Set the `ROBOFLOW_API_KEY` environment variable.
4. Run `python scripts/download_component_dataset.py`.
5. Train with `python scripts/train_components_yolo.py --data data/component_dataset/data.yaml --epochs 40`.

For Streamlit deployment, the hosted detection option uses the model endpoint electronics-components-sveck/7 and reads the key from Streamlit secrets. A detector labels the component class; reading resistor bands is a separate image-analysis task.
