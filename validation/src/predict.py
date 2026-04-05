from __future__ import annotations

import json
import pickle
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch

from evaluation.src.metrics import evaluate_prediction_folder
from models import BRATS_3D_PATCH_SIZE, build_brats_inference_model
from project import get_dataset_name, get_default_folds, get_evaluation_root, get_gt_segmentations_dir, get_primary_preprocessed_dataset_dir, get_results_root, get_training_cases_dir, resolve_fold_validation_dir
from training.src.data.dataset import infer_preprocessed_dataset_class
from training.src.data.labels import load_brats_label_manager
from training.src.core.runtime import build_device, resolve_validation_checkpoint
from .inference import predict_sliding_window_logits, restore_prediction_to_original_space, write_segmentation_nifti


def _resolve_output_dir(
    *,
    output_dir: str | Path | None,
    fold: int,
    sample_training_cases: int,
    sample_seed: int,
) -> Path:
    if output_dir is not None:
        return Path(output_dir).resolve()
    base = (
        get_evaluation_root()
        / f"predict_training_sample_n{sample_training_cases}_seed{sample_seed}_fold{fold}"
    ).resolve()
    if not base.exists() or not any(base.iterdir()):
        return base
    suffix = 1
    while True:
        candidate = base.with_name(f"{base.name}_repeat{suffix}")
        if not candidate.exists() or not any(candidate.iterdir()):
            return candidate
        suffix += 1


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_fold_score(fold: int) -> float | None:
    summary_path = resolve_fold_validation_dir(fold) / "summary.json"
    if not summary_path.is_file():
        return None
    try:
        summary = _load_json(summary_path)
    except json.JSONDecodeError:
        return None
    value = summary.get("foreground_mean", {}).get("Dice")
    if not isinstance(value, (int, float)):
        return None
    return float(value)


def _load_recommended_folds() -> tuple[int, ...] | None:
    inference_information = get_results_root() / get_dataset_name() / "inference_information.json"
    if not inference_information.is_file():
        return None
    try:
        payload = _load_json(inference_information)
    except json.JSONDecodeError:
        return None
    folds = payload.get("folds")
    if not isinstance(folds, list) or not all(isinstance(fold, int) for fold in folds):
        return None
    return tuple(folds)


def _resolve_prediction_fold(use_best_checkpoint: bool) -> tuple[int, Path, str]:
    scored_candidates: list[tuple[float, int, Path]] = []
    fallback_candidates: list[tuple[int, Path]] = []
    candidate_folds = _load_recommended_folds() or tuple(get_default_folds())
    for fold in candidate_folds:
        checkpoint = resolve_validation_checkpoint(fold, use_best=use_best_checkpoint)
        if checkpoint is None:
            continue
        fallback_candidates.append((fold, checkpoint))
        score = _load_fold_score(fold)
        if score is not None:
            scored_candidates.append((score, fold, checkpoint))

    if scored_candidates:
        scored_candidates.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        _, fold, checkpoint = scored_candidates[0]
        return fold, checkpoint, "best available validation Dice from find-best-config folds"

    if fallback_candidates:
        fallback_candidates.sort(key=lambda item: item[0])
        fold, checkpoint = fallback_candidates[0]
        return fold, checkpoint, "first fold with an available checkpoint from find-best-config folds"

    raise RuntimeError("No validation checkpoint is available in any default fold.")


def _sample_training_case_ids(sample_training_cases: int, sample_seed: int) -> list[str]:
    training_cases_dir = get_training_cases_dir()
    dataset_class = infer_preprocessed_dataset_class(str(training_cases_dir))
    all_case_ids = dataset_class.get_identifiers(str(training_cases_dir))
    all_case_ids = sorted(all_case_ids)
    if sample_training_cases > len(all_case_ids):
        raise RuntimeError(
            f"Requested {sample_training_cases} training cases, but only {len(all_case_ids)} are available."
        )
    rng = random.Random(sample_seed)
    return rng.sample(all_case_ids, sample_training_cases)


def predict_training_cases(
    *,
    sample_training_cases: int,
    sample_seed: int,
    use_best_checkpoint: bool = False,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
    export_probabilities: bool = False,
) -> dict[str, Any]:
    if sample_training_cases <= 0:
        raise RuntimeError("--sample-training-cases must be a positive integer.")

    resolved_fold, checkpoint_path, fold_reason = _resolve_prediction_fold(use_best_checkpoint)
    fold_was_auto_selected = True

    destination = _resolve_output_dir(
        output_dir=output_dir,
        fold=resolved_fold,
        sample_training_cases=sample_training_cases,
        sample_seed=sample_seed,
    )
    if destination.exists() and not destination.is_dir():
        raise RuntimeError(f"Output path exists and is not a directory: {destination}")
    if destination.is_dir() and any(destination.iterdir()) and not overwrite:
        raise RuntimeError(
            f"Output directory already exists and is not empty: {destination}. Use --overwrite or choose --output-dir."
        )
    if destination.is_dir() and overwrite:
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    model = build_brats_inference_model()
    torch_device = build_device()
    checkpoint = torch.load(checkpoint_path, map_location=torch_device, weights_only=False)
    model.load_state_dict(checkpoint["network_weights"])
    model = model.to(torch_device)
    model.eval()

    checkpoint_config = checkpoint.get("config") or {}
    mirror_axes = tuple(checkpoint_config.get("inference_mirroring_axes", (0, 1, 2)))

    metadata_dir = get_primary_preprocessed_dataset_dir()
    dataset_json = _load_json(metadata_dir / "dataset.json")
    label_manager = load_brats_label_manager(dataset_json)
    training_cases_dir = get_training_cases_dir()
    dataset_class = infer_preprocessed_dataset_class(str(training_cases_dir))
    dataset = dataset_class(str(training_cases_dir))
    selected_case_ids = _sample_training_case_ids(sample_training_cases, sample_seed)

    for case_id in selected_case_ids:
        data, _, _, properties = dataset.load_case(case_id)
        image = torch.from_numpy(np.asarray(data)).float()
        logits = predict_sliding_window_logits(
            model,
            image=image,
            patch_size=BRATS_3D_PATCH_SIZE,
            device=torch_device,
            mirror_axes=mirror_axes,
        )
        probabilities = label_manager.apply_inference_nonlin(logits).cpu().numpy()
        restored_probabilities, segmentation = restore_prediction_to_original_space(
            probabilities,
            properties,
            label_manager,
        )
        output_stem = destination / case_id
        if export_probabilities:
            np.savez_compressed(str(output_stem) + ".npz", probabilities=restored_probabilities)
            with Path(str(output_stem) + ".pkl").open("wb") as handle:
                pickle.dump(properties, handle)
        write_segmentation_nifti(segmentation, properties, Path(str(output_stem) + ".nii.gz"))

    sample_selection = {
        "sample_training_cases": int(sample_training_cases),
        "sample_seed": int(sample_seed),
        "selected_case_ids": selected_case_ids,
        "fold": resolved_fold,
        "fold_was_auto_selected": fold_was_auto_selected,
        "fold_selection_reason": fold_reason,
        "checkpoint_path": str(checkpoint_path),
        "device": str(torch_device),
        "export_probabilities": bool(export_probabilities),
        "output_dir": str(destination),
    }
    sample_selection_path = destination / "sample_selection.json"
    sample_selection_path.write_text(
        json.dumps(sample_selection, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    summary = evaluate_prediction_folder(
        destination,
        gt_dir=get_gt_segmentations_dir(),
        output_file=destination / "summary.json",
        num_processes=1,
        chill=True,
    )
    return {
        "summary": summary,
        "output_dir": str(destination),
        "summary_path": str(destination / "summary.json"),
        "sample_selection_path": str(sample_selection_path),
        "selected_case_ids": selected_case_ids,
        "fold": resolved_fold,
        "fold_was_auto_selected": fold_was_auto_selected,
        "fold_selection_reason": fold_reason,
        "checkpoint_path": str(checkpoint_path),
    }


__all__ = ["predict_training_cases"]
