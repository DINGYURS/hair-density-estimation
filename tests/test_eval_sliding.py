from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import torch

from eval_sliding import (
    evaluate_sliding_image,
    list_all_image_ids,
    read_split_image_ids,
    resolve_eval_image_ids,
    resolve_sliding_output_path,
)
from eval_sliding import write_sliding_predictions_csv
from eval import PredictionRecord
from calibrate_sliding import SlidingCalibration


class ZeroDensityModel(torch.nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch_size, _channels, height, width = images.shape
        return torch.zeros((batch_size, 1, height // 8, width // 8), dtype=torch.float32)


class EvalSlidingTest(unittest.TestCase):
    def test_read_split_image_ids_skips_blank_lines(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            split_path = Path(temp_dir) / "test.txt"
            split_path.write_text("a\n\n b \n", encoding="utf-8")

            self.assertEqual(read_split_image_ids(split_path), ["a", "b"])

    def test_list_all_image_ids_reads_sorted_image_stems(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_dir = Path(temp_dir) / "Images"
            image_dir.mkdir()
            (image_dir / "b.jpg").write_bytes(b"")
            (image_dir / "a.jpg").write_bytes(b"")
            (image_dir / "ignore.txt").write_text("x", encoding="utf-8")

            self.assertEqual(list_all_image_ids(Path(temp_dir)), ["a", "b"])

    def test_resolve_eval_image_ids_can_use_all_images(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir)
            image_dir = data_root / "Images"
            image_dir.mkdir()
            (image_dir / "img_002.jpg").write_bytes(b"")
            (image_dir / "img_001.jpg").write_bytes(b"")

            image_ids = resolve_eval_image_ids(
                data_cfg={"root": str(data_root)},
                split="all",
            )

            self.assertEqual(image_ids, ["img_001", "img_002"])

    def test_resolve_sliding_output_path_uses_split_name(self) -> None:
        output_path = resolve_sliding_output_path(Path("outputs/predictions"), split="test")

        self.assertEqual(output_path, Path("outputs/predictions/test_sliding_predictions.csv"))

    def test_evaluate_sliding_image_uses_full_image_gt_with_class_filter(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir) / "FDU"
            image_dir = data_root / "Images"
            annotation_dir = data_root / "Annotations"
            image_dir.mkdir(parents=True)
            annotation_dir.mkdir(parents=True)

            image_path = image_dir / "sample.jpg"
            image = np.full((512, 512, 3), 128, dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(image_path), image))

            xml_path = annotation_dir / "sample.xml"
            xml_path.write_text(
                """<annotation>
  <filename>sample.jpg</filename>
  <size><width>512</width><height>512</height></size>
  <object><name>premium</name><bndbox><xmin>10</xmin><ymin>10</ymin><xmax>20</xmax><ymax>20</ymax></bndbox></object>
  <object><name>abnormal</name><bndbox><xmin>70</xmin><ymin>50</ymin><xmax>80</xmax><ymax>60</ymax></bndbox></object>
</annotation>
""",
                encoding="utf-8",
            )

            record = evaluate_sliding_image(
                model=ZeroDensityModel(),
                image_id="sample",
                data_root=data_root,
                device=torch.device("cpu"),
                input_size=512,
                stride=256,
                include_classes=None,
                exclude_classes=["abnormal"],
            )

            self.assertEqual(record.image_id, "sample")
            self.assertEqual(record.gt_count, 1.0)
            self.assertEqual(record.pred_count, 0.0)

    def test_write_sliding_predictions_csv_can_include_calibrated_counts(self) -> None:
        records = [PredictionRecord("img_001", gt_count=10.0, pred_count=12.0)]
        calibration = SlidingCalibration(slope=0.5, intercept=1.0, source="val.csv")

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "predictions.csv"
            write_sliding_predictions_csv(records, path, calibration=calibration)

            self.assertEqual(
                path.read_text(encoding="utf-8").splitlines(),
                [
                    "image_id,gt_count,pred_count,corrected_pred_count,abs_error,squared_error,corrected_abs_error,corrected_squared_error",
                    "img_001,10.000000,12.000000,7.000000,2.000000,4.000000,3.000000,9.000000",
                ],
            )


if __name__ == "__main__":
    unittest.main()
