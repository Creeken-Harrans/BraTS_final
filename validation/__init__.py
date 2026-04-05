from __future__ import annotations

from importlib import import_module

__all__ = [
    "DEFAULT_METRIC",
    "compute_case_metrics",
    "find_best_config",
    "generate_overlay",
    "predict_training_cases",
    "predict_sliding_window_logits",
    "render_progress_plot",
    "restore_prediction_to_original_space",
    "summarize_validation_metrics",
    "write_segmentation_nifti",
    "write_validation_overlay",
]

_MODULE_BY_NAME = {
    "DEFAULT_METRIC": "validation.src.find_best_config",
    "compute_case_metrics": "validation.src.inference",
    "find_best_config": "validation.src.find_best_config",
    "generate_overlay": "validation.src.visualization",
    "predict_training_cases": "validation.src.predict",
    "predict_sliding_window_logits": "validation.src.inference",
    "render_progress_plot": "validation.src.visualization",
    "restore_prediction_to_original_space": "validation.src.inference",
    "summarize_validation_metrics": "validation.src.inference",
    "write_segmentation_nifti": "validation.src.inference",
    "write_validation_overlay": "validation.src.visualization",
}


def __getattr__(name: str):
    if name not in _MODULE_BY_NAME:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_MODULE_BY_NAME[name])
    return getattr(module, name)
