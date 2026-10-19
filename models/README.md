# Model checkpoints

## Live webcam checkpoint priority

The Electronic Component Analyzer checks local checkpoints in this order:

1. `models/best.pt` — optional combined Electronics + IoT checkpoint exported by the Google Colab notebook.
2. `models/components.pt` — existing general-electronics checkpoint.
3. `models/iot_components.pt` and `models/iot_boards.pt` — optional task-specific checkpoints.

The webcam loads the first available checkpoint when Ultralytics is installed. If no local checkpoint can be loaded, the live view can use the configured hosted Roboflow detector.

## Add the Colab-trained checkpoint

Download `best.pt` from [the Colab notebook](https://colab.research.google.com/github/Amsh23/VisionLab/blob/main/notebooks/train_electronics_iot_colab.ipynb), then place it at this directory path as `models/best.pt`. The checkpoint is optional and must be validated on independent real photos before use. The notebook's quick-training preset is intended as a baseline, not an accuracy guarantee.

Do not commit raw datasets, API keys, Colab runtime files, or training logs.
