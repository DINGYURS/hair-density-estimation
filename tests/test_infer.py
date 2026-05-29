from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import torch

from infer import (
    InferenceResult,
    format_inference_label_lines,
    generate_sliding_windows,
    load_crop_gt_count,
    load_full_gt_count,
    preprocess_image,
    save_inference_visualizations,
    stitch_window_densities,
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

    def test_generate_sliding_windows_covers_full_fdu_image(self) -> None:
        windows = generate_sliding_windows(height=1024, width=1280, patch_size=512, stride=512)

        self.assertEqual(windows[0], (0, 0, 512, 512))
        self.assertIn((768, 512, 1280, 1024), windows)
        self.assertTrue(all(right - left == 512 for left, _top, right, _bottom in windows))
        self.assertTrue(all(bottom - top == 512 for _left, top, _right, bottom in windows))
        self.assertEqual(max(right for _left, _top, right, _bottom in windows), 1280)
        self.assertEqual(max(bottom for _left, _top, _right, bottom in windows), 1024)

    def test_stitch_window_densities_returns_full_size_count_preserved_map(self) -> None:
        windows = [
            (0, 0, 512, 512),
            (512, 0, 1024, 512),
        ]
        densities = [
            np.full((64, 64), 1.0 / (64 * 64), dtype=np.float32),
            np.full((64, 64), 2.0 / (64 * 64), dtype=np.float32),
        ]

        full_density = stitch_window_densities(
            windows=windows,
            densities=densities,
            output_size=(512, 1024),
        )

        self.assertEqual(full_density.shape, (512, 1024))
        self.assertAlmostEqual(float(full_density.sum()), 3.0, places=4)

    def test_load_full_gt_count_applies_class_filter_without_cropping(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / "FDU"
            annotation_dir = data_root / "Annotations"
            annotation_dir.mkdir(parents=True)

            xml_path = annotation_dir / "sample.xml"
            xml_path.write_text(
                """<annotation>
  <filename>sample.jpg</filename>
  <size><width>120</width><height>80</height></size>
  <object><name>premium</name><bndbox><xmin>10</xmin><ymin>10</ymin><xmax>20</xmax><ymax>20</ymax></bndbox></object>
  <object><name>abnormal</name><bndbox><xmin>70</xmin><ymin>50</ymin><xmax>80</xmax><ymax>60</ymax></bndbox></object>
</annotation>
""",
                encoding="utf-8",
            )

            gt_count = load_full_gt_count(
                image_path=data_root / "Images" / "sample.jpg",
                data_root=data_root,
                exclude_classes=["abnormal"],
            )

            self.assertEqual(gt_count, 1.0)


if __name__ == "__main__":
    unittest.main()
