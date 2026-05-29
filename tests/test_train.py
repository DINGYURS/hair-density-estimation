from __future__ import annotations

import unittest

from src.models import CSRNet
from train import apply_freeze_frontend, trainable_parameters


class TrainConfigurationTest(unittest.TestCase):
    def test_apply_freeze_frontend_keeps_backend_trainable(self) -> None:
        model = CSRNet(pretrained=False)

        apply_freeze_frontend(model, freeze_frontend=True)

        self.assertTrue(all(not parameter.requires_grad for parameter in model.frontend.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.backend.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.output_layer.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in trainable_parameters(model)))


if __name__ == "__main__":
    unittest.main()
