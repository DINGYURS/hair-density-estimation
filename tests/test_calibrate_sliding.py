from pathlib import Path
import tempfile
import unittest

from calibrate_sliding import (
    SlidingCalibration,
    apply_calibration,
    fit_linear_calibration,
    load_calibration,
    save_calibration,
)


class CalibrateSlidingTest(unittest.TestCase):
    def test_fit_linear_calibration_maps_pred_to_gt(self) -> None:
        calibration = fit_linear_calibration(
            pred_counts=[10.0, 20.0, 30.0],
            gt_counts=[9.0, 17.0, 25.0],
            source="val.csv",
        )

        self.assertAlmostEqual(calibration.slope, 0.8)
        self.assertAlmostEqual(calibration.intercept, 1.0)
        self.assertEqual(apply_calibration(20.0, calibration), 17.0)

    def test_save_and_load_calibration_json(self) -> None:
        calibration = SlidingCalibration(
            slope=0.748972,
            intercept=1.647411,
            source="val_sliding_predictions.csv",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sliding_calibration.json"
            save_calibration(calibration, path)

            loaded = load_calibration(path)

        self.assertEqual(loaded, calibration)


if __name__ == "__main__":
    unittest.main()
