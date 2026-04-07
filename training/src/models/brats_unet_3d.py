from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any  # noqa: F401
import torch.nn as nn

from dynamic_network_architectures.architectures.unet import PlainConvUNet

BRATS_INPUT_CHANNELS = 4
BRATS_OUTPUT_CHANNELS = 3
BRATS_3D_PATCH_SIZE = (128, 128, 128)


@dataclass(frozen=True)
class BratsUNet3DConfig:
    input_channels: int = BRATS_INPUT_CHANNELS
    output_channels: int = BRATS_OUTPUT_CHANNELS
    patch_size: tuple[int, int, int] = BRATS_3D_PATCH_SIZE
    deep_supervision: bool = True
    n_stages: int = 6
    features_per_stage: tuple[int, ...] = (32, 64, 128, 256, 320, 320)
    kernel_sizes: tuple[tuple[int, int, int], ...] = (
        (3, 3, 3),
        (3, 3, 3),
        (3, 3, 3),
        (3, 3, 3),
        (3, 3, 3),
        (3, 3, 3),
    )
    strides: tuple[tuple[int, int, int], ...] = (
        (1, 1, 1),
        (2, 2, 2),
        (2, 2, 2),
        (2, 2, 2),
        (2, 2, 2),
        (2, 2, 2),
    )
    n_conv_per_stage: tuple[int, ...] = (2, 2, 2, 2, 2, 2)
    n_conv_per_stage_decoder: tuple[int, ...] = (2, 2, 2, 2, 2)
    conv_bias: bool = True
    norm_op: type[nn.Module] = nn.InstanceNorm3d
    norm_op_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"eps": 1e-5, "affine": True}
    )
    dropout_op: type[nn.Module] | None = None
    dropout_op_kwargs: dict[str, Any] | None = None
    nonlin: type[nn.Module] = nn.LeakyReLU
    nonlin_kwargs: dict[str, Any] = field(default_factory=lambda: {"inplace": True})


class BratsUNet3D(PlainConvUNet):
    """Fixed 3D segmentation network for the BraTS_final project."""

    def __init__(self, config: BratsUNet3DConfig | None = None) -> None:
        cfg = config or BratsUNet3DConfig()
        super().__init__(
            input_channels=cfg.input_channels,
            num_classes=cfg.output_channels,
            n_stages=cfg.n_stages,
            features_per_stage=list(cfg.features_per_stage),
            conv_op=nn.Conv3d,
            kernel_sizes=[list(v) for v in cfg.kernel_sizes],
            strides=[list(v) for v in cfg.strides],
            n_conv_per_stage=list(cfg.n_conv_per_stage),
            n_conv_per_stage_decoder=list(cfg.n_conv_per_stage_decoder),
            conv_bias=cfg.conv_bias,
            norm_op=cfg.norm_op,
            norm_op_kwargs=cfg.norm_op_kwargs,
            dropout_op=cfg.dropout_op,
            dropout_op_kwargs=cfg.dropout_op_kwargs,
            nonlin=cfg.nonlin,
            nonlin_kwargs=cfg.nonlin_kwargs,
            deep_supervision=cfg.deep_supervision,
        )
        self.config = cfg


def build_brats_training_model(
    input_channels: int = BRATS_INPUT_CHANNELS,
    output_channels: int = BRATS_OUTPUT_CHANNELS,
) -> BratsUNet3D:
    return BratsUNet3D(
        BratsUNet3DConfig(
            input_channels=input_channels,
            output_channels=output_channels,
            deep_supervision=True,
        )
    )


def build_brats_inference_model(
    input_channels: int = BRATS_INPUT_CHANNELS,
    output_channels: int = BRATS_OUTPUT_CHANNELS,
) -> BratsUNet3D:
    return BratsUNet3D(
        BratsUNet3DConfig(
            input_channels=input_channels,
            output_channels=output_channels,
            deep_supervision=False,
        )
    )
