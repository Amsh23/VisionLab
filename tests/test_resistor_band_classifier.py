"""Tests for the trained resistor band-color feature pipeline."""

import unittest

import numpy as np

from resistor_band_classifier import (
    BAND_COLOR_NAMES,
    BAND_COLOR_RGB,
    FEATURE_VERSION,
    extract_color_patch_features,
)


class ResistorBandClassifierFeatureTests(unittest.TestCase):
    def test_all_standard_colors_have_finite_features(self):
        for name in BAND_COLOR_NAMES:
            with self.subTest(color=name):
                rgb = np.array(BAND_COLOR_RGB[name], dtype=np.uint8)
                features = extract_color_patch_features(rgb)
                self.assertEqual(features.shape, (18,))
                self.assertTrue(np.isfinite(features).all())

    def test_image_patch_feature_extraction_is_deterministic(self):
        patch = np.full((32, 12, 3), BAND_COLOR_RGB["Gold"], dtype=np.uint8)
        first = extract_color_patch_features(patch)
        second = extract_color_patch_features(patch)
        np.testing.assert_allclose(first, second)
        self.assertEqual(FEATURE_VERSION, 1)

    def test_invalid_patch_is_rejected(self):
        with self.assertRaises(ValueError):
            extract_color_patch_features(np.zeros((10, 10), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
