from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import torch

from infer import (
    InferenceResult,
    format_inference_label_lines,
    load_crop_gt_count,
    preprocess_image,
    save_inference_visualizations,
)


class InferTest(unittest.TestCase):
    def test_preprocess_image_center_crops_and_normalizes_for_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "sample.jpg"
            image = np.full((80, 120, 3), 128, dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(image_path), image))

            tensor, cropped_bgr = preprocess_image(image_path, input_size=64)

            self.assertEqual(tuple(tensor.shape), (1, 3, 64, 64))
            self.assertEqual(cropped_bgr.shape, (64, 64, 3))
            self.assertEqual(tensor.dtype, torch.float32)

    def test_save_inference_visualizations_writes_demo_images(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            image = np.full((64, 64, 3), 80, dtype=np.uint8)
            density = np.zeros((8, 8), dtype=np.float32)
            density[3, 4] = 2.5

            result = save_inference_visualizations(
                image_bgr=image,
                pred_density=density,
                output_dir=output_dir,
                image_id="sample",
                pred_count=2.5,
                gt_count=3.0,
                alpha=0.4,
            )

            self.assertIsInstance(result, InferenceResult)
            self.assertEqual(result.image_id, "sample")
            self.assertEqual(result.gt_count, 3.0)
            self.assertEqual(result.pred_count, 2.5)
            self.assertEqual(result.abs_error, 0.5)
            self.assertTrue(result.heatmap_path.exists())
            self.assertTrue(result.overlay_path.exists())

    def test_load_crop_gt_count_applies_abnormal_exclusion_for_stage8_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / "FDU"
            image_dir = data_root / "Images"
            annotation_dir = data_root / "Annotations"
            image_dir.mkdir(parents=True)
            annotation_dir.mkdir(parents=True)

            image_path = image_dir / "sample.jpg"
            image = np.full((80, 120, 3), 128, dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(image_path), image))

            xml_path = annotation_dir / "sample.xml"
            xml_path.write_text(
                """<annotation>
  <filename>sample.jpg</filename>
  <size><width>120</width><height>80</height></size>
  <object><name>premium</name><bndbox><xmin>30</xmin><ymin>30</ymin><xmax>40</xmax><ymax>40</ymax></bndbox></object>
  <object><name>abnormal</name><bndbox><xmin>45</xmin><ymin>30</ymin><xmax>55</xmax><ymax>40</ymax></bndbox></object>
</annotation>
""",
                encoding="utf-8",
            )

            gt_count = load_crop_gt_count(
                image_path=image_path,
                data_root=data_root,
                input_size=64,
                exclude_classes=["abnormal"],
            )

            self.assertEqual(gt_count, 1.0)

    def test_format_inference_label_lines_splits_long_text(self) -> None:
        lines = format_inference_label_lines(
            image_id="230219_A177_4",
            pred_count=2.47,
            gt_count=2.0,
        )

        self.assertEqual(lines, ["230219_A177_4", "gt=2.00  pred=2.47  err=0.47"])


if __name__ == "__main__":
    unittest.main()
