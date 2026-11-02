"""Small multi-class U-Net."""

import numpy as np
import torch
from torch import nn

from .simulate import NUM_CLASSES


def _block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(True),
        nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(True),
    )


class UNet(nn.Module):
    def __init__(self, base: int = 16, classes: int = NUM_CLASSES):
        super().__init__()
        self.e1, self.e2 = _block(1, base), _block(base, base * 2)
        self.mid = _block(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.d2 = _block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.d1 = _block(base * 2, base)
        self.head = nn.Conv2d(base, classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        mid = self.mid(self.pool(e2))
        d2 = self.d2(torch.cat([self.up2(mid), e2], 1))
        return self.head(self.d1(torch.cat([self.up1(d2), e1], 1)))


@torch.no_grad()
def predict(model: UNet, image: np.ndarray) -> np.ndarray:
    model.eval()
    return model(torch.from_numpy(image)[None, None])[0].argmax(0).numpy()
