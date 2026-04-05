from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class BratsTrainingConfig:
    epochs: int = 150
    initial_lr: float = 1e-2
    weight_decay: float = 3e-5
    momentum: float = 0.99
    nesterov: bool = True
    lr_poly_exponent: float = 0.9
    batch_size: int = 2
    oversample_foreground_percent: float = 0.33
    probabilistic_oversampling: bool = False
    num_iterations_per_epoch: int = 250
    num_val_iterations_per_epoch: int = 50
    use_deep_supervision: bool = True
    use_compile: bool = True
    inference_mirroring_axes: tuple[int, int, int] = (0, 1, 2)


@dataclass(frozen=True)
class BratsAugmentationConfig:
    rotation_degrees: float = 30.0
    scale_range: tuple[float, float] = (0.7, 1.4)
    gaussian_noise_probability: float = 0.1
    gaussian_blur_probability: float = 0.2
    brightness_probability: float = 0.15
    contrast_probability: float = 0.15
    low_resolution_probability: float = 0.25
    gamma_invert_probability: float = 0.1
    gamma_probability: float = 0.3
    mirror_axes: tuple[int, int, int] = (0, 1, 2)


def get_default_training_config() -> BratsTrainingConfig:
    cfg = BratsTrainingConfig()
    overrides = {
        "epochs": ("BRATS_FINAL_EPOCHS", int),
        "batch_size": ("BRATS_FINAL_BATCH_SIZE", int),
        "num_iterations_per_epoch": ("BRATS_FINAL_TRAIN_ITERS", int),
        "num_val_iterations_per_epoch": ("BRATS_FINAL_VAL_ITERS", int),
        "use_compile": ("BRATS_FINAL_USE_COMPILE", lambda v: v.lower() in ("1", "true", "yes")),
    }
    payload = {}
    for field_name, (env_name, caster) in overrides.items():
        value = os.environ.get(env_name)
        if value is not None:
            payload[field_name] = caster(value)
    return BratsTrainingConfig(**payload)


def get_default_augmentation_config() -> BratsAugmentationConfig:
    return BratsAugmentationConfig()
