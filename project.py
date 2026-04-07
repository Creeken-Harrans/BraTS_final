from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


@lru_cache(maxsize=1)
def get_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "project_config.json").is_file():
            return parent
    raise RuntimeError("Unable to locate BraTS_final project root")


@lru_cache(maxsize=1)
def load_project_config() -> dict[str, Any]:
    return json.loads((get_project_root() / "project_config.json").read_text())


def resolve_project_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (get_project_root() / candidate).resolve()


def get_dataset_name() -> str:
    return load_project_config()["dataset"]["name"]


def get_default_folds() -> list[int]:
    return list(load_project_config()["dataset"]["default_folds"])


def get_workspace_root() -> Path:
    return get_project_root().parent


def get_raw_dataset_dir() -> Path:
    cfg = load_project_config()
    return resolve_project_path(cfg["paths"]["raw_root"]) / cfg["dataset"]["name"]


def get_preprocessed_dataset_dir() -> Path:
    cfg = load_project_config()
    return resolve_project_path(cfg["paths"]["preprocessed_root"]) / cfg["dataset"]["name"]


def get_external_dataset_root() -> Path:
    return (get_workspace_root() / "BraTS-Dataset").resolve()


def get_external_preprocessed_dataset_dir() -> Path:
    return get_external_dataset_root() / "nnUNet_preprocessed" / get_dataset_name()


def get_external_raw_dataset_dir() -> Path:
    return get_external_dataset_root() / "nnUNet_raw" / get_dataset_name()


def get_primary_raw_dataset_dir() -> Path:
    raw_dir = get_raw_dataset_dir()
    images_dir = raw_dir / "imagesTr"
    labels_dir = raw_dir / "labelsTr"
    if images_dir.is_dir() and labels_dir.is_dir():
        return raw_dir
    return get_external_raw_dataset_dir()


def get_primary_preprocessed_dataset_dir() -> Path:
    dataset_dir = get_preprocessed_dataset_dir()
    if (dataset_dir / "dataset.json").is_file() and (dataset_dir / "splits_final.json").is_file():
        return dataset_dir
    return get_external_preprocessed_dataset_dir()


def get_training_cases_dir() -> Path:
    internal_dir = get_preprocessed_dataset_dir() / "ProjectPlans_3d_fullres"
    if internal_dir.is_dir() and any(internal_dir.glob("*.pkl")):
        return internal_dir
    return get_external_preprocessed_dataset_dir() / "ProjectPlans_3d_fullres"


def get_gt_segmentations_dir() -> Path:
    internal_dir = get_preprocessed_dataset_dir() / "gt_segmentations"
    if internal_dir.is_dir():
        return internal_dir
    return get_external_preprocessed_dataset_dir() / "gt_segmentations"


def get_results_root() -> Path:
    cfg = load_project_config()
    return resolve_project_path(cfg["paths"]["results_root"])


def get_fold_root(fold: int) -> Path:
    return get_results_root() / f"fold{int(fold)}"


def get_legacy_fold_artifacts_dir(fold: int) -> Path:
    return get_fold_root(fold) / "legacy_braTS_artifacts"


def resolve_fold_artifacts_dir(fold: int) -> Path:
    fold_root = get_fold_root(fold)
    legacy_dir = get_legacy_fold_artifacts_dir(fold)
    if any((fold_root / name).exists() for name in ("training_state.json", "checkpoint_final.pth", "validation")):
        return fold_root
    if legacy_dir.is_dir():
        return legacy_dir
    return fold_root


def resolve_fold_validation_dir(fold: int) -> Path:
    active_validation = get_fold_root(fold) / "validation"
    if active_validation.is_dir() and any(active_validation.glob("*.nii.gz")):
        return active_validation
    legacy_validation = get_legacy_fold_artifacts_dir(fold) / "validation"
    if legacy_validation.is_dir():
        return legacy_validation
    return active_validation


def get_model_results_root() -> Path:
    return get_results_root()


def get_evaluation_root() -> Path:
    return get_project_root() / "evaluation" / "results"


def get_evaluation_report_root() -> Path:
    return get_project_root() / "evaluation" / "report"


def get_model_name() -> str:
    return load_project_config()["training"]["model_name"]


def get_configuration_name() -> str:
    return load_project_config()["training"]["configuration_name"]


def get_checkpoint_priority() -> list[str]:
    return list(load_project_config()["training"]["checkpoint_priority"])
