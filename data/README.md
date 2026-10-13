# Resistor datasets

This folder contains a small, standards-based resistor color-code reference table used by the VisionLab Electronic Component Analyzer.

## Reference dataset

- `resistor_color_reference.csv` lists the ten digit colors, their multipliers, common tolerance values, and reference color swatches.
- These are reference values, not photographs and not a trained machine-learning model.
- `synthetic_resistors/images/` contains 12 small synthetic PNG samples. Their labels and expected resistor values are in `synthetic_resistors/labels.csv`. These are illustrative generated images, not photographs of physical resistors.

## Generate an image dataset

Install the project's core dependencies, then run this from the repository root:

```bash
python scripts/generate_resistor_dataset.py --count 300 --output data/generated_resistor_dataset
```

The script creates `images/` with synthetic 4-band and 5-band resistor images plus `labels.csv` with exact band labels, resistance values, tolerance values, and train/validation/test splits.

**Important limitation:** these images are procedurally drawn and do not represent real photographs. They are useful for demos, testing label parsing, and explaining data pipelines, but they should not be used as the only training or evaluation set for a real-world vision model. Collect real photos with varied cameras, backgrounds, orientations, and lighting before claiming practical accuracy.

## Real component-detection dataset

For training a model to locate resistors among other electronic parts, see the public [Electronic Components 8 Classes dataset on Roboflow Universe](https://universe.roboflow.com/electronic-components-dataset-for-yolo/electronic-components-8-classes). Its page lists a CC BY 4.0 license and classes for resistor, capacitor, diode, and several component/value categories. Review the current dataset version and attribution instructions before redistributing an export.

A detector that finds a resistor is not the same as a model that reads its color bands. For a general-purpose application, train and evaluate these two stages separately.
