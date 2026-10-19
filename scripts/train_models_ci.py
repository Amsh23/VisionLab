"""Download public datasets and train the lightweight models used by VisionLab CI.

The script keeps raw dataset files under data/ and publishes only trained model
checkpoints under models/. Roboflow datasets are optional and require a
ROBOFLOW_API_KEY environment variable; public fallback sources are used when
the key is not available.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
PYTHON = sys.executable
TRAIN_BUDGET_MIN = max(1, int(os.environ.get("VISIONLAB_TRAIN_BUDGET_MIN", "150")))
TRAIN_STARTED = time.monotonic()


def remaining_budget_seconds() -> float:
    return TRAIN_BUDGET_MIN * 60 - (time.monotonic() - TRAIN_STARTED)


def run_command(arguments: list[str]) -> None:
    """Run a subprocess without copying credentials into command-line arguments."""
    printable = " ".join(str(value) for value in arguments)
    print(f"\n[VisionLab training] $ {printable}", flush=True)
    remaining = remaining_budget_seconds()
    if remaining <= 0:
        raise RuntimeError(
            f"Training time budget ({TRAIN_BUDGET_MIN} minutes) has been exhausted."
        )
    try:
        completed = subprocess.run(
            arguments,
            cwd=ROOT,
            env=os.environ.copy(),
            check=False,
            timeout=remaining,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"Training command exceeded the remaining time budget "
            f"({TRAIN_BUDGET_MIN} minutes)."
        ) from error
    if completed.returncode:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {arguments[1]}")


def dataset_yaml(directory: Path) -> Path | None:
    candidates = sorted(directory.rglob("data.yaml")) + sorted(directory.rglob("data.yml"))
    return candidates[0] if candidates else None


def download_circuit_elements() -> Path:
    """Download a public YOLO electronics dataset and normalize spelling variants."""
    from huggingface_hub import snapshot_download

    raw = DATA / "circuit-elements"
    raw.mkdir(parents=True, exist_ok=True)
    errors = []
    for repo_id in ("LibreYOLO/circuit-elements", "Libre-YOLO/circuit-elements"):
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=str(raw),
                max_workers=8,
            )
            print(f"Downloaded public dataset: {repo_id}", flush=True)
            break
        except Exception as error:
            errors.append(f"{repo_id}: {type(error).__name__}")
            if raw.exists():
                shutil.rmtree(raw, ignore_errors=True)
    else:
        raise RuntimeError("Circuit Elements download failed: " + "; ".join(errors))

    output = DATA / "circuit-elements-normalized"
    run_command([
        PYTHON, "scripts/prepare_circuit_elements_yolo.py",
        "--input", str(raw), "--output", str(output),
    ])
    result = output / "data.yaml"
    if not result.is_file():
        raise RuntimeError("Normalized Circuit Elements data.yaml was not created.")
    return result


def train_yolo(data_file: Path, output: Path, name: str, epochs: int, segmentation: bool = False) -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    run_command([
        PYTHON, "scripts/train_components_yolo.py",
        "--data", str(data_file),
        "--model", "yolo11s-seg.pt" if segmentation else "yolo11s.pt",
        "--epochs", str(epochs),
        "--imgsz", "640",
        "--batch", "4" if segmentation else "8",
        "--device", "cpu",
        "--project", f"runs/{name}",
        "--name", "github_actions",
        "--weights-out", str(output),
    ])
    if not output.is_file():
        raise RuntimeError(f"Training did not create the expected checkpoint: {output.name}")


def train_electronics(report: dict) -> None:
    key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    data_file: Path | None = None
    cached_candidate = DATA / "component_dataset" / "data.yaml"
    if cached_candidate.is_file():
        data_file = cached_candidate
        print(f"Reusing cached electronics dataset: {data_file}", flush=True)
    elif key:
        try:
            run_command([PYTHON, "scripts/download_component_dataset.py", "--output", "data/component_dataset"])
            candidate = DATA / "component_dataset" / "data.yaml"
            if candidate.is_file():
                data_file = candidate
        except Exception as error:
            report["warnings"].append(f"Roboflow electronics download: {type(error).__name__}")

    if data_file is None:
        data_file = download_circuit_elements()

    train_yolo(
        data_file,
        MODELS / "components.pt",
        "electronics",
        epochs=60,
        segmentation=False,
    )
    report["models"].append({
        "file": "models/components.pt",
        "dataset": "Roboflow Electronics Components" if "component_dataset" in str(data_file) else "LibreYOLO Circuit Elements",
        "task": "electronics component detection",
    })


def train_iot(report: dict) -> None:
    key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    data_file: Path | None = None
    model_path: Path | None = None
    dataset_name = ""

    cached_iot = DATA / "iot_roboflow" / "data.yaml"
    if cached_iot.is_file():
        data_file = cached_iot
        model_path = MODELS / "iot_components.pt"
        dataset_name = "Roboflow IoT_Dataset"
        print(f"Reusing cached IoT dataset: {data_file}", flush=True)
    elif key:
        try:
            run_command([PYTHON, "scripts/download_roboflow_dataset.py", "--dataset", "iot"])
            candidate = DATA / "iot_roboflow" / "data.yaml"
            if candidate.is_file():
                data_file = candidate
                model_path = MODELS / "iot_components.pt"
                dataset_name = "Roboflow IoT_Dataset"
        except Exception as error:
            report["warnings"].append(f"Roboflow IoT download: {type(error).__name__}")

    # If the private-key Roboflow export is unavailable, still try the public
    # IoTKITs dataset and its annotations instead of skipping local IoT training.
    if data_file is None:
        try:
            run_command([
                PYTHON, "scripts/download_iotkits_dataset.py",
                "--output", "data/iotkits_raw",
            ])
            run_command([
                PYTHON, "scripts/prepare_iotkits_yolo.py",
                "--input", "data/iotkits_raw",
                "--output", "data/iotkits_yolo",
            ])
            candidate = DATA / "iotkits_yolo" / "data.yaml"
            if candidate.is_file():
                data_file = candidate
                model_path = MODELS / "iot_boards.pt"
                dataset_name = "IoTKITs Mendeley Data"
        except Exception as error:
            report["warnings"].append(f"IoTKITs download or annotation conversion: {type(error).__name__}")
            print(
                "WARNING: IoTKITs could not be prepared. The existing hosted IoT detector remains available "
                "when ROBOFLOW_API_KEY is configured in Streamlit.",
                flush=True,
            )
            return

    if data_file is None or model_path is None:
        report["warnings"].append("No usable annotated IoT dataset was found.")
        return

    train_yolo(
        data_file,
        model_path,
        "iot",
        epochs=60 if dataset_name == "Roboflow IoT_Dataset" else 40,
        segmentation=False,
    )
    report["models"].append({
        "file": str(model_path.relative_to(ROOT)),
        "dataset": dataset_name,
        "task": "IoT boards or modules",
    })


def has_segmentation_polygons(directory: Path) -> bool:
    """Require at least one polygon annotation before training a segmentation model."""
    for label_file in directory.rglob("*.txt"):
        if "labels" not in {part.casefold() for part in label_file.parts}:
            continue
        for line in label_file.read_text(encoding="utf-8", errors="replace").splitlines():
            if len(line.split()) >= 7:
                return True
    return False


def train_resistor_colors(report: dict) -> None:
    run_command([
        PYTHON, "scripts/train_resistor_band_classifier.py",
        "--samples-per-class", "1000",
        "--output", str(MODELS / "resistor_band_classifier.joblib"),
    ])
    report["models"].append({
        "file": "models/resistor_band_classifier.joblib",
        "dataset": "Augmented synthetic color-band patches",
        "task": "12-class resistor-band patch color classification",
        "warning": "Synthetic-only metrics are not estimates of accuracy on real resistor photographs.",
    })

    key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not key:
        report["warnings"].append(
            "Real-photo resistor-band models were skipped because the GitHub Actions "
            "ROBOFLOW_API_KEY secret is not configured."
        )
        return

    # Model A: instance segmentation of individual resistor color bands.
    try:
        raw = DATA / "resistor_color_bands"
        data_file = dataset_yaml(raw)
        if data_file is None:
            run_command([
                PYTHON, "scripts/download_roboflow_dataset.py",
                "--dataset", "resistor-color-bands",
            ])
            data_file = dataset_yaml(raw)
        if data_file is None or not has_segmentation_polygons(raw):
            raise RuntimeError("No valid YOLO segmentation polygon annotations were found.")

        train_yolo(
            data_file,
            MODELS / "resistor_seg.pt",
            "resistor-color-bands",
            epochs=30,
            segmentation=True,
        )
        report["models"].append({
            "file": "models/resistor_seg.pt",
            "dataset": "Roboflow Resistor Color Detection",
            "task": "real-photo resistor-band instance segmentation",
        })
    except Exception as error:
        report["warnings"].append(
            f"Resistor color-band instance segmentation was not produced: {type(error).__name__}: {error}"
        )

    # Model B: preserve the supplied semantic-mask dataset as a separate checkpoint.
    # It has a different label schema from individual band-color polygons, so it
    # must not overwrite models/resistor_seg.pt.
    try:
        raw = DATA / "resistor_value_segmentation"
        converted = DATA / "resistor_value_seg_yolo"
        if dataset_yaml(raw) is None:
            run_command([
                PYTHON, "scripts/download_roboflow_dataset.py",
                "--dataset", "resistor-segmentation",
            ])
        run_command([
            PYTHON, "scripts/prepare_resistor_segmentation_yolo.py",
            "--input", str(raw), "--output", str(converted), "--background-value", "0",
        ])
        data_file = dataset_yaml(converted)
        if data_file is None or not has_segmentation_polygons(converted):
            raise RuntimeError("The semantic-mask export did not produce valid YOLO polygons.")
        train_yolo(
            data_file,
            MODELS / "resistor_value_seg.pt",
            "resistor-value-segmentation",
            epochs=12,
            segmentation=True,
        )
        report["models"].append({
            "file": "models/resistor_value_seg.pt",
            "dataset": "Roboflow Resistor Value Determination",
            "task": "experimental resistor semantic-mask segmentation converted to polygons",
            "warning": "Converted semantic masks are approximate; validate class mapping on real photos.",
        })
    except Exception as error:
        report["warnings"].append(
            f"Resistor value semantic-segmentation model was not produced: {type(error).__name__}: {error}. "
            "Inspect mask background values and class IDs before relying on this model."
        )


def main() -> None:
    task = sys.argv[1] if len(sys.argv) > 1 else "all"
    if task not in {"all", "electronics", "iot", "resistor-colors"}:
        raise SystemExit("Usage: python scripts/train_models_ci.py [all|electronics|iot|resistor-colors]")

    DATA.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "models": [],
        "warnings": [],
    }
    tasks = {
        "electronics": train_electronics,
        "iot": train_iot,
        "resistor-colors": train_resistor_colors,
    }
    selected = list(tasks) if task == "all" else [task]
    for task_name in selected:
        try:
            remaining = remaining_budget_seconds()
            if remaining <= 0:
                report["warnings"].append(
                    f"Skipped {task_name}: the {TRAIN_BUDGET_MIN}-minute training budget was exhausted."
                )
                break
            print(
                f"\n========== Training task: {task_name} "
                f"(about {remaining / 60:.1f} minutes budget remaining) ==========",
                flush=True,
            )
            tasks[task_name](report)
        except Exception as error:
            report["warnings"].append(f"{task_name}: {type(error).__name__}: {error}")
            print(f"WARNING: {task_name} failed: {type(error).__name__}: {error}", flush=True)

    report_path = MODELS / "training-report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nTraining summary:")
    print(json.dumps(report, indent=2))
    if not report["models"]:
        raise SystemExit("No model checkpoint was produced; see the warnings above.")


if __name__ == "__main__":
    main()
