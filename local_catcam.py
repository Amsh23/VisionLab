"""Open a local CatCam/webcam preview with OpenCV.

Run on the Windows PC that has CatCam installed:
    python local_catcam.py --camera 0

Press Q or Esc to quit. Press S to save the current frame.
Use --list to probe camera indexes 0 through 9.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import time

import cv2


def open_camera(index: int) -> cv2.VideoCapture:
    """Open a camera using the Windows DirectShow backend first."""
    if hasattr(cv2, "CAP_DSHOW"):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture(index)


def list_cameras(max_index: int = 10) -> None:
    print("Probing camera indexes (close other camera apps if a device is busy)...")
    found = []
    for index in range(max_index):
        cap = open_camera(index)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                height, width = frame.shape[:2]
                print(f"Camera index {index}: available ({width}x{height})")
                found.append(index)
        cap.release()
    if not found:
        print("No working camera found. Check CatCam, Windows camera permissions, and USB/Wi-Fi connection.")
    else:
        print("Use one of the indexes above, for example: python local_catcam.py --camera 1")


def main() -> None:
    parser = argparse.ArgumentParser(description="Show a live webcam feed using OpenCV.")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0).")
    parser.add_argument("--list", action="store_true", help="Probe camera indexes 0-9 and exit.")
    args = parser.parse_args()

    if args.list:
        list_cameras()
        return

    cap = open_camera(args.camera)
    if not cap.isOpened():
        cap.release()
        raise SystemExit(
            f"Could not open camera index {args.camera}. Run 'python local_catcam.py --list' "
            "to find CatCam's index, and make sure no other app is holding the camera."
        )

    print(f"Live camera started at index {args.camera}. Press Q/Esc to quit; S saves a frame.")
    captures_dir = Path("captures")
    previous_time = time.perf_counter()
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Camera stopped returning frames.")
                break

            now = time.perf_counter()
            elapsed = now - previous_time
            previous_time = now
            if elapsed > 0:
                instant_fps = 1.0 / elapsed
                fps = instant_fps if fps == 0 else (0.85 * fps + 0.15 * instant_fps)

            preview = frame.copy()
            cv2.putText(
                preview,
                f"CatCam / OpenCV | {fps:.1f} FPS | Q: quit | S: save",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (40, 230, 80),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("VisionLab - Live CatCam", preview)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                captures_dir.mkdir(parents=True, exist_ok=True)
                filename = captures_dir / f"catcam_{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"
                if cv2.imwrite(str(filename), frame):
                    print(f"Saved original camera frame: {filename}")
                else:
                    print("Could not save the frame.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
