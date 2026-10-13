import csv
import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_ROOT = ROOT / "data" / "synthetic_resistors"


class SyntheticResistorDatasetTests(unittest.TestCase):
    def test_sample_labels_point_to_valid_images(self):
        labels_file = SAMPLE_ROOT / "labels.csv"
        self.assertTrue(labels_file.is_file())

        with labels_file.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 12)
        for row in rows:
            image_path = SAMPLE_ROOT / row["image"]
            self.assertTrue(image_path.is_file(), f"Missing image: {image_path}")

            with Image.open(image_path) as image:
                self.assertEqual(image.format, "PNG")
                self.assertEqual(image.size, (160, 48))
                image.verify()

            bands = [row[f"band_{index}"] for index in range(1, int(row["band_count"]) + 1)]
            expected_digit_count = int(row["band_count"]) - 2
            digits = "".join(str(["Black", "Brown", "Red", "Orange", "Yellow", "Green", "Blue", "Violet", "Gray", "White"].index(color))
                             for color in bands[:expected_digit_count])
            multipliers = {
                "Silver": 0.01, "Gold": 0.1, "Black": 1, "Brown": 10, "Red": 100,
                "Orange": 1000, "Yellow": 10000, "Green": 100000, "Blue": 1000000,
                "Violet": 10000000,
            }
            expected_ohms = int(digits) * multipliers[bands[-2]]
            self.assertAlmostEqual(float(row["resistance_ohms"]), expected_ohms)


if __name__ == "__main__":
    unittest.main()
