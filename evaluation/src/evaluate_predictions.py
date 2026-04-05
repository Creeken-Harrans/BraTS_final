from __future__ import annotations

from pathlib import Path
from typing import Optional

from .metrics import (
    compute_metrics_on_folder,
    evaluate_prediction_folder,
    load_summary_json,
    save_summary_json,
)


def evaluate_folder(
    folder_ref: str | Path,
    folder_pred: str | Path,
    *,
    output_file: str | Path | None = None,
    file_ending: str = ".nii.gz",
    labels_or_regions: list[int | tuple[int, ...]] | None = None,
    ignore_label: Optional[int] = None,
    num_processes: int = 1,
    chill: bool = True,
) -> dict:
    if labels_or_regions is None:
        raise ValueError("labels_or_regions is required when using evaluate_folder directly.")
    return compute_metrics_on_folder(
        folder_ref,
        folder_pred,
        output_file,
        labels_or_regions,
        ignore_label=ignore_label,
        num_processes=num_processes,
        chill=chill,
        file_ending=file_ending,
    )


__all__ = [
    "compute_metrics_on_folder",
    "evaluate_folder",
    "evaluate_prediction_folder",
    "load_summary_json",
    "save_summary_json",
]
