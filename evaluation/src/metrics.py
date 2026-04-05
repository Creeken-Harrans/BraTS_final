from __future__ import annotations

import json
import multiprocessing
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Optional

import numpy as np
import SimpleITK as sitk

from project import (
    get_default_folds,
    get_gt_segmentations_dir,
    get_model_results_root,
    get_primary_preprocessed_dataset_dir,
    resolve_fold_validation_dir,
)
from training.src.data.labels import load_brats_label_manager
from training.src.utils import make_json_safe


def label_or_region_to_key(label_or_region: int | tuple[int, ...]) -> str:
    return str(label_or_region)


def key_to_label_or_region(key: str) -> int | tuple[int, ...]:
    try:
        return int(key)
    except ValueError:
        key = key.replace("(", "").replace(")", "")
        return tuple(int(part) for part in key.split(",") if part)


def save_summary_json(results: dict, output_file: str | Path) -> None:
    converted = deepcopy(results)
    converted["mean"] = {
        label_or_region_to_key(key): value for key, value in results["mean"].items()
    }
    for index in range(len(converted["metric_per_case"])):
        converted["metric_per_case"][index]["metrics"] = {
            label_or_region_to_key(key): value
            for key, value in results["metric_per_case"][index]["metrics"].items()
        }
    Path(output_file).write_text(
        json.dumps(make_json_safe(converted), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_summary_json(filename: str | Path) -> dict:
    results = json.loads(Path(filename).read_text(encoding="utf-8"))
    results["mean"] = {
        key_to_label_or_region(key): value for key, value in results["mean"].items()
    }
    for index in range(len(results["metric_per_case"])):
        results["metric_per_case"][index]["metrics"] = {
            key_to_label_or_region(key): value
            for key, value in results["metric_per_case"][index]["metrics"].items()
        }
    return results


def region_or_label_to_mask(
    segmentation: np.ndarray, region_or_label: int | tuple[int, ...]
) -> np.ndarray:
    if np.isscalar(region_or_label):
        return segmentation == region_or_label
    return np.isin(segmentation, list(region_or_label))


def compute_tp_fp_fn_tn(
    mask_ref: np.ndarray,
    mask_pred: np.ndarray,
    ignore_mask: np.ndarray | None = None,
) -> tuple[int, int, int, int]:
    use_mask = np.ones_like(mask_ref, dtype=bool) if ignore_mask is None else ~ignore_mask
    tp = int(np.sum((mask_ref & mask_pred) & use_mask))
    fp = int(np.sum(((~mask_ref) & mask_pred) & use_mask))
    fn = int(np.sum((mask_ref & (~mask_pred)) & use_mask))
    tn = int(np.sum(((~mask_ref) & (~mask_pred)) & use_mask))
    return tp, fp, fn, tn


def compute_metrics(
    reference_file: str,
    prediction_file: str,
    labels_or_regions: list[int | tuple[int, ...]],
    ignore_label: Optional[int] = None,
) -> dict:
    seg_ref = sitk.GetArrayFromImage(sitk.ReadImage(reference_file))
    seg_pred = sitk.GetArrayFromImage(sitk.ReadImage(prediction_file))
    ignore_mask = seg_ref == ignore_label if ignore_label is not None else None

    results = {
        "reference_file": reference_file,
        "prediction_file": prediction_file,
        "metrics": {},
    }
    for label_or_region in labels_or_regions:
        mask_ref = region_or_label_to_mask(seg_ref, label_or_region)
        mask_pred = region_or_label_to_mask(seg_pred, label_or_region)
        tp, fp, fn, tn = compute_tp_fp_fn_tn(mask_ref, mask_pred, ignore_mask)
        if tp + fp + fn == 0:
            dice = float("nan")
            iou = float("nan")
        else:
            dice = 2 * tp / (2 * tp + fp + fn)
            iou = tp / (tp + fp + fn)
        results["metrics"][label_or_region] = {
            "Dice": dice,
            "IoU": iou,
            "FP": fp,
            "TP": tp,
            "FN": fn,
            "TN": tn,
            "n_pred": fp + tp,
            "n_ref": fn + tp,
        }
    return results


def compute_metrics_on_folder(
    folder_ref: str | Path,
    folder_pred: str | Path,
    output_file: str | Path | None,
    labels_or_regions: list[int | tuple[int, ...]],
    *,
    ignore_label: Optional[int] = None,
    num_processes: int = 1,
    chill: bool = True,
    file_ending: str = ".nii.gz",
) -> dict:
    folder_ref = Path(folder_ref)
    folder_pred = Path(folder_pred)
    files_pred = sorted(path.name for path in folder_pred.glob(f"*{file_ending}"))
    files_ref = sorted(path.name for path in folder_ref.glob(f"*{file_ending}"))
    if len(files_pred) == 0:
        raise RuntimeError(
            f"No prediction files with suffix {file_ending} were found in {folder_pred}."
        )
    missing_refs = [name for name in files_pred if not (folder_ref / name).is_file()]
    if missing_refs:
        raise FileNotFoundError(
            f"Predictions without matching ground-truth files were found in {folder_pred}: {missing_refs}"
        )
    if not chill:
        missing_predictions = [name for name in files_ref if not (folder_pred / name).is_file()]
        if missing_predictions:
            raise RuntimeError(
                f"Not all files in {folder_ref} exist in {folder_pred}: {missing_predictions[:5]}"
            )

    work_items = [
        (str(folder_ref / name), str(folder_pred / name), labels_or_regions, ignore_label)
        for name in files_pred
    ]
    if num_processes > 1:
        with multiprocessing.get_context("spawn").Pool(num_processes) as pool:
            results = pool.starmap(compute_metrics, work_items)
    else:
        results = [compute_metrics(*item) for item in work_items]

    metric_names = list(results[0]["metrics"][labels_or_regions[0]].keys())
    means = {}
    for label_or_region in labels_or_regions:
        means[label_or_region] = {
            metric_name: float(
                np.nanmean([item["metrics"][label_or_region][metric_name] for item in results])
            )
            for metric_name in metric_names
        }
    foreground_mean = {
        metric_name: float(np.nanmean([means[key][metric_name] for key in means]))
        for metric_name in metric_names
    }
    summary = {
        "metric_per_case": results,
        "mean": means,
        "foreground_mean": foreground_mean,
    }
    if output_file is not None:
        save_summary_json(summary, output_file)
    return summary


def evaluate_prediction_folder(
    pred_dir: str | Path,
    *,
    gt_dir: str | Path | None = None,
    output_file: str | Path | None = None,
    num_processes: int = 1,
    chill: bool = False,
) -> dict:
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir) if gt_dir is not None else get_gt_segmentations_dir()
    training_state_candidates = (
        pred_dir.parent / "training_state.json",
        pred_dir.parent.parent / "training_state.json",
    )
    dataset_json = None
    for candidate in training_state_candidates:
        if candidate.is_file():
            dataset_json = json.loads(candidate.read_text(encoding="utf-8"))
            break
    if dataset_json is None:
        dataset_json = json.loads((get_primary_preprocessed_dataset_dir() / "dataset.json").read_text(encoding="utf-8"))
    else:
        data_layout = dataset_json.get("data_layout")
        if data_layout and data_layout.get("dataset_json"):
            dataset_json = json.loads(
                Path(data_layout["dataset_json"]).read_text(encoding="utf-8")
            )
        else:
            dataset_json = json.loads((get_primary_preprocessed_dataset_dir() / "dataset.json").read_text(encoding="utf-8"))
    label_manager = load_brats_label_manager(dataset_json)
    return compute_metrics_on_folder(
        gt_dir,
        pred_dir,
        output_file,
        label_manager.foreground_regions if label_manager.has_regions else label_manager.foreground_labels,
        ignore_label=label_manager.ignore_label,
        num_processes=num_processes,
        chill=chill,
        file_ending=dataset_json["file_ending"],
    )


def accumulate_cv_results(
    merged_output_folder: str | Path,
    folds: list[int] | tuple[int, ...] | None = None,
    *,
    num_processes: int = 1,
    overwrite: bool = True,
) -> dict:
    model_results_root = get_model_results_root()
    merged_output_folder = Path(merged_output_folder)
    folds = tuple(get_default_folds() if folds is None else folds)

    if overwrite and merged_output_folder.is_dir():
        shutil.rmtree(merged_output_folder)
    merged_output_folder.mkdir(parents=True, exist_ok=True)

    copied_any = False
    for fold in folds:
        validation_dir = resolve_fold_validation_dir(fold)
        if not validation_dir.is_dir():
            raise RuntimeError(f"fold {fold} is missing validation output: {validation_dir}")
        for prediction_file in sorted(validation_dir.glob("*.nii.gz")):
            target = merged_output_folder / prediction_file.name
            if target.exists():
                raise RuntimeError(f"More than one fold has a prediction for {prediction_file.name}")
            shutil.copy2(prediction_file, target)
            copied_any = True

    if not copied_any:
        raise RuntimeError("No validation predictions were copied while accumulating CV results.")

    summary = evaluate_prediction_folder(
        merged_output_folder,
        output_file=merged_output_folder / "summary.json",
        num_processes=num_processes,
        chill=True,
    )
    return summary
