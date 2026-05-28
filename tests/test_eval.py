from pathlib import Path
import tempfile
import unittest

from eval import PredictionRecord, compute_summary, write_predictions_csv


class EvalTest(unittest.TestCase):
    def test_compute_summary_uses_prediction_records(self) -> None:
        records = [
            PredictionRecord("a", 10.0, 8.0),
            PredictionRecord("b", 12.0, 15.0),
        ]

        summary = compute_summary(records)

        self.assertEqual(summary["samples"], 2)
        self.assertEqual(summary["mae"], 2.5)
        self.assertEqual(summary["rmse"], (13.0 / 2.0) ** 0.5)

    def test_write_predictions_csv_has_stage_six_columns(self) -> None:
        records = [PredictionRecord("img_001", 5.0, 6.5)]

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "predictions.csv"
            write_predictions_csv(records, csv_path)

            self.assertEqual(
                csv_path.read_text(encoding="utf-8").splitlines(),
                [
                    "image_id,gt_count,pred_count,abs_error,squared_error",
                    "img_001,5.000000,6.500000,1.500000,2.250000",
                ],
            )


if __name__ == "__main__":
    unittest.main()
