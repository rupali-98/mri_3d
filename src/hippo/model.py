"""3D U-Net built with MONAI."""
from __future__ import annotations

from monai.networks.nets import UNet

from . import NUM_CLASSES


def build_model(cfg: dict) -> UNet:
    m = cfg["model"]
    return UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=NUM_CLASSES,
        channels=tuple(m["channels"]),
        strides=tuple(m["strides"]),
        num_res_units=m["num_res_units"],
        dropout=m["dropout"],
    )
