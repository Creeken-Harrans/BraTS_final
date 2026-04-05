from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from project import (
    get_checkpoint_priority,
    get_configuration_name,
    get_dataset_name,
    get_default_folds,
    get_fold_root,
    get_model_name,
    get_gt_segmentations_dir,
    get_preprocessed_dataset_dir,
    get_primary_preprocessed_dataset_dir,
    get_project_root,
    get_legacy_fold_artifacts_dir,
    get_results_root,
    get_training_cases_dir,
    resolve_fold_artifacts_dir,
)
from .config import BratsTrainingConfig, get_default_training_config


def build_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this project, but no CUDA device is available."
        )
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    return torch.device("cuda")


def get_training_output_dir(fold: int) -> Path:
    return get_fold_root(fold)


def get_results_snapshot_dir(fold: int) -> Path:
    return get_training_output_dir(fold) / "snapshot"


def get_training_logs_dir(fold: int) -> Path:
    return get_training_output_dir(fold) / "logs"


def resolve_resume_checkpoint(fold: int) -> Path | None:
    for output_dir in (get_training_output_dir(fold), get_legacy_fold_artifacts_dir(fold)):
        for checkpoint_name in get_checkpoint_priority():
            candidate = output_dir / checkpoint_name
            if candidate.is_file():
                return candidate
    return None


def resolve_training_resume_checkpoint(fold: int) -> Path | None:
    for output_dir in (get_training_output_dir(fold), get_legacy_fold_artifacts_dir(fold)):
        for checkpoint_name in ("checkpoint_latest.pth", "checkpoint_best.pth", "checkpoint_final.pth"):
            candidate = output_dir / checkpoint_name
            if candidate.is_file():
                return candidate
    return None


def resolve_validation_checkpoint(fold: int, *, use_best: bool = False) -> Path | None:
    preferred = (
        ("checkpoint_best.pth", "checkpoint_final.pth", "checkpoint_latest.pth")
        if use_best
        else ("checkpoint_final.pth", "checkpoint_best.pth", "checkpoint_latest.pth")
    )
    for output_dir in (get_training_output_dir(fold), get_legacy_fold_artifacts_dir(fold)):
        for checkpoint_name in preferred:
            candidate = output_dir / checkpoint_name
            if candidate.is_file():
                return candidate
    return None


def read_training_state(fold: int) -> dict[str, Any] | None:
    for training_state_file in (
        get_training_output_dir(fold) / "training_state.json",
        get_legacy_fold_artifacts_dir(fold) / "training_state.json",
    ):
        if not training_state_file.is_file():
            continue
        try:
            return json.loads(training_state_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
    return None


def resolve_target_num_epochs() -> int:
    return get_default_training_config().epochs


def training_fold_is_complete(fold: int) -> bool:
    state = read_training_state(fold)
    if state is None:
        return False

    try:
        next_epoch = int(state.get("next_epoch", 0))
    except (TypeError, ValueError):
        next_epoch = 0

    if state.get("status") == "completed" and next_epoch >= resolve_target_num_epochs():
        return True

    try:
        remaining_epochs = int(state.get("remaining_epochs", 1))
    except (TypeError, ValueError):
        return False
    return remaining_epochs <= 0 or next_epoch >= resolve_target_num_epochs()


def describe_training_action(
    fold: int,
    *,
    restart_training: bool,
    validation_only: bool,
    pretrained_weights: str | None,
) -> tuple[str, str]:
    if validation_only:
        return "validation-only", "explicit --validation-only"
    if restart_training:
        return "restart", "explicit --restart-training"
    if pretrained_weights is not None:
        return "pretrained", "explicit --pretrained-weights"

    checkpoint = resolve_resume_checkpoint(fold)
    if checkpoint is None:
        return "start-from-scratch", "no checkpoint found"
    if training_fold_is_complete(fold):
        return "skip-completed", f"completed fold ({checkpoint.name})"
    return "resume", f"checkpoint available ({checkpoint.name})"


def ensure_preprocessed_training_inputs() -> tuple[Path, Path]:
    preprocessed_dir = get_primary_preprocessed_dataset_dir()
    dataset_json = preprocessed_dir / "dataset.json"
    splits_json = preprocessed_dir / "splits_final.json"
    if not dataset_json.is_file():
        raise RuntimeError(f"Missing required dataset metadata: {dataset_json}")
    if not splits_json.is_file():
        raise RuntimeError(f"Missing required split file: {splits_json}")
    training_cases_dir = get_training_cases_dir()
    if not training_cases_dir.is_dir():
        raise RuntimeError(f"Missing required preprocessed training cases directory: {training_cases_dir}")
    return dataset_json, splits_json


def resolve_training_data_layout() -> dict[str, str]:
    dataset_json, splits_json = ensure_preprocessed_training_inputs()
    return {
        "metadata_dir": str(get_primary_preprocessed_dataset_dir()),
        "training_cases_dir": str(get_training_cases_dir()),
        "gt_segmentations_dir": str(get_gt_segmentations_dir()),
        "dataset_json": str(dataset_json),
        "splits_json": str(splits_json),
    }


def build_training_metadata(
    fold: int,
    device: torch.device,
    config: BratsTrainingConfig | None = None,
) -> dict[str, Any]:
    cfg = config or get_default_training_config()
    output_dir = get_training_output_dir(fold)
    checkpoint = resolve_resume_checkpoint(fold)
    return {
        "status": "planned",
        "dataset_name": get_dataset_name(),
        "model_name": get_model_name(),
        "configuration_name": get_configuration_name(),
        "fold": fold,
        "device": str(device),
        "output_folder": str(output_dir),
        "resume_checkpoint_path": str(checkpoint) if checkpoint is not None else None,
        "data_layout": resolve_training_data_layout(),
        "training_hyperparameters": asdict(cfg),
    }


def write_training_state(fold: int, state: dict[str, Any]) -> Path:
    output_dir = get_training_output_dir(fold)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "training_state.json"
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def sync_training_snapshot(fold: int) -> None:
    source = get_training_output_dir(fold)
    target = get_results_snapshot_dir(fold)
    if not source.is_dir():
        return
    target.mkdir(parents=True, exist_ok=True)
    logs_source = get_training_logs_dir(fold)
    logs_target = target / "logs"
    if logs_source.is_dir():
        logs_target.mkdir(parents=True, exist_ok=True)
        for pattern in ("training_log_*.txt",):
            for path in logs_source.glob(pattern):
                logs_target.joinpath(path.name).write_bytes(path.read_bytes())
    for pattern in ("debug.json", "progress.png", "summary.json"):
        for path in source.glob(pattern):
            target.joinpath(path.name).write_bytes(path.read_bytes())


def get_train_all_plan(
    restart_training: bool,
    validation_only: bool,
    pretrained_weights: str | None,
) -> list[str]:
    lines: list[str] = []
    for fold in get_default_folds():
        action, reason = describe_training_action(
            fold,
            restart_training=restart_training,
            validation_only=validation_only,
            pretrained_weights=pretrained_weights,
        )
        lines.append(f"fold {fold}: {action} ({reason})")
    return lines
