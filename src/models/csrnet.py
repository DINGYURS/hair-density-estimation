from __future__ import annotations

import logging
from collections.abc import Sequence

import torch
from torch import nn
import torch.nn.functional as F


LOGGER = logging.getLogger(__name__)


VGG_FRONTEND_CHANNELS = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512]
BACKEND_CHANNELS = [512, 512, 512, 256, 128, 64]


def _make_layers(
    cfg: Sequence[int | str],
    in_channels: int = 3,
    dilation: int = 1,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_channels = in_channels
    for item in cfg:
        if item == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            continue
        if not isinstance(item, int):
            raise ValueError(f"Unsupported layer config item: {item!r}")
        layers.extend(
            [
                nn.Conv2d(
                    current_channels,
                    item,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation,
                ),
                nn.ReLU(inplace=True),
            ]
        )
        current_channels = item
    return nn.Sequential(*layers)


class CSRNet(nn.Module):
    """CSRNet density estimator with a VGG-16 frontend and dilated backend."""

    def __init__(self, pretrained: bool = False, non_negative: bool = True) -> None:
        super().__init__()
        self.pretrained = bool(pretrained)
        self.non_negative = bool(non_negative)
        self.frontend = _make_layers(VGG_FRONTEND_CHANNELS)
        self.backend = _make_layers(BACKEND_CHANNELS, in_channels=512, dilation=2)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        self._initialize_weights()
        if self.pretrained:
            self._load_vgg16_frontend_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        if self.non_negative:
            x = F.relu(x)
        return x

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

    def _load_vgg16_frontend_weights(self) -> None:
        try:
            from torchvision.models import VGG16_Weights, vgg16
        except ImportError:
            LOGGER.warning("torchvision is not installed; CSRNet frontend uses random weights.")
            self.pretrained = False
            return

        try:
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        except Exception as exc:
            LOGGER.warning("Failed to load torchvision VGG16 weights: %s", exc)
            self.pretrained = False
            return

        frontend_state = self.frontend.state_dict()
        vgg_state = vgg.features[:23].state_dict()
        matched = {
            key: value
            for key, value in vgg_state.items()
            if key in frontend_state and frontend_state[key].shape == value.shape
        }
        frontend_state.update(matched)
        self.frontend.load_state_dict(frontend_state)
        LOGGER.info("Loaded ImageNet VGG16 weights for %d frontend tensors.", len(matched))


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable
