import random
import unittest

import numpy as np

from resistor_analyzer import detect_band_runs
from scripts.generate_resistor_dataset import render_synthetic_resistor


class ResistorBandDetectionTests(unittest.TestCase):
    def test_detects_four_band_synthetic_resistor(self):
        expected = ["Yellow", "Violet", "Red", "Gold"]
        image = render_synthetic_resistor(
            expected,
            random.Random(42),
            np.random.default_rng(42),
        )

        runs, candidates = detect_band_runs(image, expected_count=4)

        self.assertGreaterEqual(len(candidates), 4)
        self.assertEqual([run["color"] for run in runs], expected)

    def test_detects_five_band_synthetic_resistor(self):
        expected = ["Yellow", "Violet", "Black", "Brown", "Brown"]
        image = render_synthetic_resistor(
            expected,
            random.Random(43),
            np.random.default_rng(43),
        )

        runs, candidates = detect_band_runs(image, expected_count=5)

        self.assertGreaterEqual(len(candidates), 5)
        self.assertEqual([run["color"] for run in runs], expected)


if __name__ == "__main__":
    unittest.main()
