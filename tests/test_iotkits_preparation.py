import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.prepare_iotkits_yolo import (
    build_image_index,
    parse_box,
    parse_coco_json,
    parse_per_image_json,
    parse_points,
    write_yolo_dataset,
)


class IoTKITsPreparationTests(unittest.TestCase):
    def test_parses_coco_flat_polygon_points(self):
        points = parse_points([[10, 20, 50, 20, 50, 60, 10, 60]])
        self.assertEqual(points, [(10.0, 20.0), (50.0, 20.0), (50.0, 60.0), (10.0, 60.0)])

    def test_parses_polygon_into_pixel_box(self):
        box = parse_box(
            {"points": [{"x": 10, "y": 20}, {"x": 50, "y": 60}]},
            width=100,
            height=80,
        )
        self.assertEqual(box, (10.0, 20.0, 50.0, 60.0))

    def test_parses_labelme_sidecar(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            Image.new("RGB", (100, 80), "white").save(root / "board.jpg")
            (root / "board.json").write_text(
                json.dumps({
                    "imagePath": "board.jpg",
                    "imageWidth": 100,
                    "imageHeight": 80,
                    "shapes": [{
                        "label": "ESP32",
                        "points": [[10, 20], [60, 20], [60, 50], [10, 50]],
                    }],
                }),
                encoding="utf-8",
            )
            by_name, by_stem = build_image_index(root)
            record = parse_per_image_json(root / "board.json", root, by_name, by_stem)

            self.assertIsNotNone(record)
            self.assertEqual(record["objects"][0][0], "ESP32")
            self.assertEqual(record["objects"][0][1:], (10.0, 20.0, 60.0, 50.0))

    def test_converts_coco_boxes_to_yolo_dataset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            Image.new("RGB", (100, 80), "white").save(root / "board.jpg")
            coco_path = root / "annotations.json"
            coco_path.write_text(
                json.dumps({
                    "images": [{"id": 1, "file_name": "board.jpg", "width": 100, "height": 80}],
                    "categories": [{"id": 3, "name": "Arduino Uno"}],
                    "annotations": [{"id": 1, "image_id": 1, "category_id": 3, "bbox": [10, 20, 50, 30]}],
                }),
                encoding="utf-8",
            )
            by_name, by_stem = build_image_index(root)
            records = parse_coco_json(coco_path, root, by_name, by_stem)
            output = root / "prepared"
            write_yolo_dataset(records, output, val_ratio=0.1, test_ratio=0.1, seed=42)

            label_files = list((output / "labels" / "train").glob("*.txt"))
            self.assertEqual(len(label_files), 1)
            label = label_files[0].read_text(encoding="utf-8").strip()
            self.assertEqual(label, "0 0.350000 0.437500 0.500000 0.375000")
            self.assertIn('0: "Arduino Uno"', (output / "data.yaml").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
