from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from src.datasets.fdu_dataset import filter_objects_by_class, resize_image_and_points
from src.utils.voc_parser import VocObject
from tools.ablation_matrix import ExperimentSpec, build_eval_command, build_train_command


class Stage8AblationTest(unittest.TestCase):
    def test_filter_objects_by_class_can_exclude_abnormal(self) -> None:
        objects = [
            VocObject("premium", (0, 0, 10, 10), (0, 0, 10, 10)),
            VocObject("abnormal", (10, 10, 20, 20), (10, 10, 20, 20)),
            VocObject("single", (20, 20, 30, 30), (20, 20, 30, 30)),
        ]

        filtered = filter_objects_by_class(objects, exclude_classes={"abnormal"})

        self.assertEqual([obj.class_name for obj in filtered], ["premium", "single"])

    def test_resize_image_and_points_scales_full_image_without_losing_points(self) -> None:
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        points = [(50.0, 25.0), (150.0, 75.0)]

        resized, scaled_points = resize_image_and_points(image, points, output_size=50)

        self.assertEqual(resized.shape, (50, 50, 3))
        self.assertEqual(scaled_points, [(12.5, 12.5), (37.5, 37.5)])

    def test_ablation_commands_are_scoped_to_experiment_output_dir(self) -> None:
        spec = ExperimentSpec(
            experiment_id="e2_mse_only",
            description="MSE without count loss",
            config=Path("configs/ablations/e2_mse_only.yaml"),
            output_dir=Path("outputs/ablations/e2_mse_only"),
        )

        train_command = build_train_command(spec, device="cuda")
        eval_command = build_eval_command(spec, split="test", device="cuda")

        self.assertIn("--config configs/ablations/e2_mse_only.yaml", train_command)
        self.assertIn("--output-dir outputs/ablations/e2_mse_only", train_command)
        self.assertIn("--checkpoint outputs/ablations/e2_mse_only/checkpoints/csrnet_best.pth", eval_command)
        self.assertIn("--output-dir outputs/ablations/e2_mse_only/predictions", eval_command)


if __name__ == "__main__":
    unittest.main()
