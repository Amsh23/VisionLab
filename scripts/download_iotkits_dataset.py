"""Download the public IoTKITs embedded-board dataset from Mendeley Data.

The dataset is distributed under CC BY 4.0. This script downloads the official
archive, verifies that it is a ZIP file, safely extracts it, and writes an
attribution note. The raw archive and extracted images are intentionally kept
outside Git by default because they are large dataset assets.
"""
from __future__ import annotations

import argparse
import json
import time
import zipfile
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import urlparse

import requests

DATASET_ID = "x5thzmkxhy"
DATASET_VERSION = 1
API_ROOT = "https://api.data.mendeley.com"


def get_download_url(session: requests.Session, dataset_id: str, version: int) -> str:
    """Get the official signed download URL from Mendeley Data."""
    metadata_url = f"{API_ROOT}/datasets/{dataset_id}/zip"
    metadata: dict = {}

    for attempt in range(1, 13):
        response = session.get(metadata_url, params={"version": version}, timeout=30)
        response.raise_for_status()
        try:
            metadata = response.json()
        except ValueError:
            metadata = {}

        download_url = metadata.get("url") if isinstance(metadata, dict) else None
        status = str(metadata.get("status", "")).upper() if isinstance(metadata, dict) else ""
        if download_url:
            return str(download_url)
        if status and status not in {"IN_PROGRESS", "PENDING", "PROCESSING"}:
            break
        if attempt < 12:
            time.sleep(5)

    # Public dataset archives may be served via a redirect endpoint when the
    # metadata endpoint does not expose a direct signed URL.
    redirect_url = f"{API_ROOT}/datasets/{dataset_id}/zip/file_downloaded"
    response = session.get(
        redirect_url,
        params={"version": version},
        allow_redirects=False,
        timeout=30,
    )
    if response.is_redirect and response.headers.get("Location"):
        return response.headers["Location"]

    detail = json.dumps(metadata, ensure_ascii=False)[:800]
    raise RuntimeError(
        "Mendeley Data did not return a ZIP download URL. "
        f"Dataset metadata: {detail or response.text[:300]}"
    )


def safe_extract(archive_path: Path, destination: Path) -> None:
    """Extract a ZIP archive while preventing path traversal entries."""
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if target != destination_resolved and destination_resolved not in target.parents:
                raise RuntimeError(f"Unsafe ZIP path detected: {member.filename}")
        archive.extractall(destination)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download IoTKITs, an annotated dataset of embedded and IoT development boards."
    )
    parser.add_argument("--dataset-id", default=DATASET_ID, help="Mendeley Data dataset ID.")
    parser.add_argument("--version", type=int, default=DATASET_VERSION)
    parser.add_argument("--output", type=Path, default=Path("data/iotkits_raw"))
    parser.add_argument("--keep-archive", action="store_true", help="Keep the downloaded ZIP in the output directory.")
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    archive_path = output / "iotkits_download.zip"

    with requests.Session() as session:
        session.headers.update({"User-Agent": "VisionLab dataset helper/1.0"})
        download_url = get_download_url(session, args.dataset_id, args.version)
        parsed_url = urlparse(download_url)
        if parsed_url.scheme not in {"https", "http"}:
            raise SystemExit(f"Unexpected dataset download URL scheme: {parsed_url.scheme!r}")

        print(f"Downloading IoTKITs {args.dataset_id}, version {args.version}...")
        with session.get(download_url, stream=True, timeout=(30, 180)) as response:
            response.raise_for_status()
            with archive_path.open("wb") as archive_file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        archive_file.write(chunk)

    if not zipfile.is_zipfile(archive_path):
        preview = archive_path.read_bytes()[:300].decode("utf-8", errors="replace")
        archive_path.unlink(missing_ok=True)
        raise SystemExit(
            "The server response was not a ZIP archive. Mendeley may have changed its download flow. "
            f"Response preview: {preview}"
        )

    print(f"Extracting archive to: {output}")
    safe_extract(archive_path, output)

    attribution = (
        "Dataset: IoTKITs\n"
        "Dataset DOI: 10.17632/x5thzmkxhy.1\n"
        "Source: https://data.mendeley.com/datasets/x5thzmkxhy/1\n"
        "License: CC BY 4.0\n"
        "Citation: Nguyen, Thanh-Thien; Nguyen Do, Anh-Tuan; Vu, Duc-Lung (2025). "
        "IoTKITs: A novel dataset for IoT education kit recognition. Data in Brief, 60, 111650. "
        "https://doi.org/10.1016/j.dib.2025.111650\n"
        "Downloaded by scripts/download_iotkits_dataset.py. Review source metadata before redistribution.\n"
    )
    (output / "DATASET_SOURCE.txt").write_text(attribution, encoding="utf-8")

    if not args.keep_archive:
        archive_path.unlink(missing_ok=True)

    images = [
        path for path in output.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ]
    json_files = list(output.rglob("*.json"))
    print(f"Download complete. Found {len(images)} image files and {len(json_files)} JSON annotation/metadata files.")
    print(f"Dataset folder: {output}")
    print("Next: run scripts/prepare_iotkits_yolo.py to convert supported annotations into YOLO format.")


if __name__ == "__main__":
    main()
