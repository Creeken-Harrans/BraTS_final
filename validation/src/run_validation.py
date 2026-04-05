from __future__ import annotations

from typing import Any

from training.src.core.runtime import build_device, resolve_validation_checkpoint
from training.src.core.trainer import BratsTrainer


def run_validation_for_fold(
    fold: int,
    *,
    use_best: bool = False,
    export_probabilities: bool = False,
) -> dict[str, Any]:
    trainer = BratsTrainer(
        fold=fold,
        device=build_device(),
        export_validation_probabilities=export_probabilities,
        validation_only=True,
        val_with_best=use_best,
    )
    trainer.initialize()
    checkpoint = resolve_validation_checkpoint(fold, use_best=use_best)
    if checkpoint is None:
        raise RuntimeError(
            f"No checkpoint available for fold {fold}. Run training first or provide a completed fold."
        )
    trainer.load_checkpoint(checkpoint)
    return trainer.perform_actual_validation()


__all__ = ["run_validation_for_fold"]
