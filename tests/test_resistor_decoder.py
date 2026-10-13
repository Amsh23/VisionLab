import unittest

from resistor_analyzer import decode_band_sequence, format_resistance


class ResistorDecoderTests(unittest.TestCase):
    def test_four_band_4k7_resistor(self):
        decoded = decode_band_sequence(["Yellow", "Violet", "Red", "Gold"], 4)
        self.assertEqual(decoded["ohms"], 4700)
        self.assertEqual(decoded["tolerance_percent"], 5)
        self.assertEqual(decoded["minimum_ohms"], 4465)
        self.assertEqual(decoded["maximum_ohms"], 4935)

    def test_five_band_4k7_resistor(self):
        decoded = decode_band_sequence(["Yellow", "Violet", "Black", "Brown", "Brown"], 5)
        self.assertEqual(decoded["ohms"], 4700)
        self.assertEqual(decoded["tolerance_percent"], 1)

    def test_gold_multiplier(self):
        decoded = decode_band_sequence(["Brown", "Black", "Gold", "Silver"], 4)
        self.assertAlmostEqual(decoded["ohms"], 1.0)
        self.assertEqual(decoded["tolerance_percent"], 10)

    def test_invalid_significant_band_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "cannot be Gold or Silver"):
            decode_band_sequence(["Gold", "Violet", "Red", "Gold"], 4)

    def test_formatter(self):
        self.assertEqual(format_resistance(4700), "4.7 kΩ")
        self.assertEqual(format_resistance(2_200_000), "2.2 MΩ")


if __name__ == "__main__":
    unittest.main()
