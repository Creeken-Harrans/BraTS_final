from __future__ import annotations

from torch import nn

from ..core.config import BratsTrainingConfig
from .scheduler import PolyLRScheduler


def build_optimizer_and_scheduler(
    model: nn.Module,
    config: BratsTrainingConfig | None = None,
):
    cfg = config or BratsTrainingConfig()
    optimizer = __import__("torch").optim.SGD(
        model.parameters(),
        lr=cfg.initial_lr,
        weight_decay=cfg.weight_decay,
        momentum=cfg.momentum,
        nesterov=cfg.nesterov,
    )
    scheduler = PolyLRScheduler(
        optimizer=optimizer,
        initial_lr=cfg.initial_lr,
        max_steps=cfg.epochs,
        exponent=cfg.lr_poly_exponent,
    )
    return optimizer, scheduler
