import math
import unittest

import numpy as np
import torch

from src.utils.metrics import count_from_density, mae, rmse


class MetricsTest(unittest.TestCase):
    def test_count_from_density_accepts_numpy_and_torch(self) -> None:
        density_np = np.array([[0.25, 0.75], [1.0, 2.0]], dtype=np.float32)
        density_torch = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])

        self.assertEqual(count_from_density(density_np), 4.0)
        self.assertEqual(count_from_density(density_torch), 10.0)

    def test_mae_and_rmse_use_per_image_count_errors(self) -> None:
        pred_counts = [10.0, 14.0, 20.0]
        gt_counts = [12.0, 13.0, 16.0]

        self.assertEqual(mae(pred_counts, gt_counts), 7.0 / 3.0)
        self.assertEqual(rmse(pred_counts, gt_counts), math.sqrt(21.0 / 3.0))

    def test_metrics_reject_empty_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty"):
            mae([], [])


if __name__ == "__main__":
    unittest.main()
