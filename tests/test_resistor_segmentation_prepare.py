import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from scripts.prepare_resistor_segmentation_yolo import (
    contours_for_class,
    list_images,
    list_masks,
    pair_image_mask,
    read_mask,
)


class ResistorSemanticMaskPreparationTests(unittest.TestCase):
    def test_pairs_image_with_mask_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "train" / "images" / "resistor_01.jpg"
            mask_path = root / "train" / "masks" / "resistor_01.png"
            image_path.parent.mkdir(parents=True)
            mask_path.parent.mkdir(parents=True)
            Image.new("RGB", (64, 32), "white").save(image_path)
            mask = np.zeros((32, 64), dtype=np.uint8)
            mask[10:22, 12:20] = 1
            Image.fromarray(mask).save(mask_path)

            self.assertIn(image_path, list_images(root))
            mask_files = list_masks(root)
            self.assertIn(mask_path, mask_files)
            self.assertEqual(pair_image_mask(image_path, root, mask_files), mask_path)

    def test_reads_mask_class_ids_without_scaling(self):
        with tempfile.TemporaryDirectory() as temporary:
            mask_path = Path(temporary) / "sample.png"
            mask = np.array([[0, 1, 7], [255, 2, 3]], dtype=np.uint8)
            Image.fromarray(mask).save(mask_path)
            actual = read_mask(mask_path)
            np.testing.assert_array_equal(actual, mask)

    def test_contour_extraction_returns_a_polygon(self):
        mask = np.zeros((40, 60), dtype=np.uint8)
        cv2.rectangle(mask, (10, 8), (30, 28), 4, thickness=-1)
        polygons = contours_for_class(mask, class_value=4, min_area=6)
        self.assertEqual(len(polygons), 1)
        self.assertGreaterEqual(len(polygons[0]), 3)


if __name__ == "__main__":
    unittest.main()
