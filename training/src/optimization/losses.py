from __future__ import annotations

from typing import Callable, Optional, Sequence

import torch
from torch import Tensor, nn


def softmax_helper_dim1(x: Tensor) -> Tensor:
    return torch.softmax(x, dim=1)


class MemoryEfficientSoftDiceLoss(nn.Module):
    def __init__(
        self,
        apply_nonlin: Optional[Callable[[Tensor], Tensor]] = None,
        do_bg: bool = True,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.apply_nonlin = apply_nonlin
        self.do_bg = do_bg
        self.smooth = smooth

    def forward(self, x: Tensor, y: Tensor, loss_mask: Tensor | None = None) -> Tensor:
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        axes = tuple(range(2, x.ndim))

        with torch.no_grad():
            if x.ndim != y.ndim:
                y = y.view((y.shape[0], 1, *y.shape[1:]))

            if x.shape == y.shape:
                y_onehot = y.to(torch.float32)
            else:
                y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.float32)
                y_onehot.scatter_(1, y.long(), 1)

            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]

            sum_gt = (
                y_onehot.sum(axes, dtype=torch.float32)
                if loss_mask is None
                else (y_onehot * loss_mask).sum(axes, dtype=torch.float32)
            )

        if not self.do_bg:
            x = x[:, 1:]

        if loss_mask is None:
            intersect = (x * y_onehot).sum(axes, dtype=torch.float32)
            sum_pred = x.sum(axes, dtype=torch.float32)
        else:
            intersect = (x * y_onehot * loss_mask).sum(axes, dtype=torch.float32)
            sum_pred = (x * loss_mask).sum(axes, dtype=torch.float32)

        dc = (2 * intersect + self.smooth) / (
            sum_gt + sum_pred + float(self.smooth)
        ).clamp_min(1e-8)
        return -dc.mean()


class RegionDiceBCELoss(nn.Module):
    """Dice + BCE loss for region-based BraTS supervision."""

    def __init__(self, smooth: float = 1e-5, use_ignore_label: bool = False) -> None:
        super().__init__()
        self.use_ignore_label = use_ignore_label
        self.bce = nn.BCEWithLogitsLoss(reduction="none" if use_ignore_label else "mean")
        self.dice = MemoryEfficientSoftDiceLoss(
            apply_nonlin=torch.sigmoid,
            do_bg=True,
            smooth=smooth,
        )

    def forward(self, net_output: Tensor, target: Tensor) -> Tensor:
        if self.use_ignore_label:
            if target.dtype == torch.bool:
                mask = ~target[:, -1:]
            else:
                mask = (1 - target[:, -1:]).bool()
            target_regions = target[:, :-1]
        else:
            mask = None
            target_regions = target

        dice_loss = self.dice(net_output, target_regions, loss_mask=mask)
        target_regions = target_regions.float()

        if mask is not None:
            bce_loss = (self.bce(net_output, target_regions) * mask).sum() / torch.clamp(
                mask.sum(), min=1e-8
            )
        else:
            bce_loss = self.bce(net_output, target_regions)

        return bce_loss + dice_loss


class DeepSupervisionWrapper(nn.Module):
    def __init__(self, loss: nn.Module, weight_factors: Sequence[float]) -> None:
        super().__init__()
        if not any(weight != 0 for weight in weight_factors):
            raise ValueError("At least one deep supervision weight must be non-zero.")
        self.loss = loss
        self.weight_factors = tuple(weight_factors)

    def forward(self, *args):
        if not all(isinstance(arg, (tuple, list)) for arg in args):
            raise TypeError("All deep supervision inputs must be tuples or lists.")

        return sum(
            self.weight_factors[index] * self.loss(*inputs)
            for index, inputs in enumerate(zip(*args))
            if self.weight_factors[index] != 0.0
        )


def get_deep_supervision_weights(num_outputs: int) -> tuple[float, ...]:
    if num_outputs < 1:
        raise ValueError("num_outputs must be at least 1")
    weights = [1 / (2**i) for i in range(num_outputs)]
    weights[-1] = 0.0
    weight_sum = sum(weights)
    return tuple(weight / weight_sum for weight in weights)


def build_region_loss(
    use_deep_supervision: bool = True,
    num_deep_supervision_outputs: int = 5,
) -> nn.Module:
    base_loss = RegionDiceBCELoss(smooth=1e-5, use_ignore_label=False)
    if not use_deep_supervision:
        return base_loss
    return DeepSupervisionWrapper(
        base_loss, get_deep_supervision_weights(num_deep_supervision_outputs)
    )
