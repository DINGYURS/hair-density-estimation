from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import torch

from infer import (
    InferenceResult,
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


if __name__ == "__main__":
    unittest.main()
