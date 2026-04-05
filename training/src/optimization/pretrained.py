from __future__ import annotations

import torch
from torch import nn


def _unwrap_module(model: nn.Module) -> nn.Module:
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def load_pretrained_weights(model: nn.Module, checkpoint_path: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("network_weights", checkpoint)
    model = _unwrap_module(model)
    current = model.state_dict()

    skip_markers = (".seg_layers.",)
    transferred = {}
    for key, value in state_dict.items():
        if key in current and current[key].shape == value.shape and not any(
            marker in key for marker in skip_markers
        ):
            transferred[key] = value

    if not transferred:
        raise RuntimeError(
            f"No compatible pretrained weights found in {checkpoint_path} for the fixed BraTS model."
        )

    current.update(transferred)
    model.load_state_dict(current)
