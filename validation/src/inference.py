from __future__ import annotations

from pathlib import Path
import itertools
from contextlib import nullcontext
from typing import Iterable

import numpy as np
import SimpleITK as sitk
import torch

from training.src.data.labels import RegionLabelManager


def _compute_steps(size: int, patch: int, step_fraction: float = 0.5) -> list[int]:
    if size <= patch:
        return [0]
    target_step = max(int(round(patch * step_fraction)), 1)
    num_steps = int(np.ceil((size - patch) / target_step)) + 1
    actual_step = (size - patch) / max(num_steps - 1, 1)
    return [int(round(actual_step * index)) for index in range(num_steps)]


def _gaussian_importance_map(patch_size: tuple[int, int, int], sigma_scale: float = 0.125) -> torch.Tensor:
    center = [(size - 1) / 2 for size in patch_size]
    sigmas = [max(size * sigma_scale, 1e-6) for size in patch_size]
    coords = np.meshgrid(*[np.arange(size, dtype=np.float32) for size in patch_size], indexing="ij")
    gaussian = np.ones(patch_size, dtype=np.float32)
    for axis, coord in enumerate(coords):
        gaussian *= np.exp(-((coord - center[axis]) ** 2) / (2 * sigmas[axis] ** 2))
    gaussian = gaussian / np.maximum(gaussian.max(), 1e-8)
    gaussian = np.clip(gaussian, 1e-4, None)
    return torch.from_numpy(gaussian)


def _predict_with_mirroring(
    model: torch.nn.Module,
    patch: torch.Tensor,
    mirror_axes: tuple[int, ...] | None,
) -> torch.Tensor:
    prediction = model(patch)
    if isinstance(prediction, (tuple, list)):
        prediction = prediction[0]

    if mirror_axes:
        axes = [axis + 2 for axis in mirror_axes]
        for combo_len in range(1, len(axes) + 1):
            for combo in itertools.combinations(axes, combo_len):
                mirrored = model(torch.flip(patch, combo))
                if isinstance(mirrored, (tuple, list)):
                    mirrored = mirrored[0]
                prediction += torch.flip(mirrored, combo)
        prediction /= 2 ** len(axes)
    return prediction


@torch.inference_mode()
def predict_sliding_window_logits(
    model: torch.nn.Module,
    image: torch.Tensor,
    patch_size: tuple[int, int, int],
    device: torch.device,
    mirror_axes: tuple[int, ...] | None = None,
) -> torch.Tensor:
    spatial_shape = tuple(int(v) for v in image.shape[1:])
    padded_shape = tuple(max(size, patch) for size, patch in zip(spatial_shape, patch_size))
    padded = torch.zeros((image.shape[0], *padded_shape), dtype=image.dtype)
    slices = tuple(slice(0, size) for size in spatial_shape)
    padded[(slice(None), *slices)] = image

    gaussian = _gaussian_importance_map(patch_size).to(device=device, dtype=torch.float32)
    output_channels = None
    aggregated = None
    normalization = torch.zeros(padded_shape, device=device, dtype=torch.float32)

    xs = _compute_steps(padded_shape[0], patch_size[0])
    ys = _compute_steps(padded_shape[1], patch_size[1])
    zs = _compute_steps(padded_shape[2], patch_size[2])

    for x in xs:
        for y in ys:
            for z in zs:
                patch = padded[
                    :,
                    x : x + patch_size[0],
                    y : y + patch_size[1],
                    z : z + patch_size[2],
                ].unsqueeze(0).to(device=device, dtype=torch.float32)
                with (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if device.type == "cuda"
                    else nullcontext()
                ):
                    prediction = _predict_with_mirroring(model, patch, mirror_axes)
                prediction = prediction[0]

                if aggregated is None:
                    output_channels = int(prediction.shape[0])
                    aggregated = torch.zeros(
                        (output_channels, *padded_shape), device=device, dtype=torch.float32
                    )

                aggregated[
                    :,
                    x : x + patch_size[0],
                    y : y + patch_size[1],
                    z : z + patch_size[2],
                ] += prediction * gaussian
                normalization[
                    x : x + patch_size[0],
                    y : y + patch_size[1],
                    z : z + patch_size[2],
                ] += gaussian

    if aggregated is None:
        raise RuntimeError("Sliding-window validation produced no predictions.")

    aggregated /= normalization.unsqueeze(0)
    return aggregated[(slice(None), *slices)].cpu()


def restore_prediction_to_original_space(
    probabilities: np.ndarray,
    properties: dict,
    label_manager: RegionLabelManager,
) -> tuple[np.ndarray, np.ndarray]:
    restored_probabilities = label_manager.revert_cropping_on_probabilities(
        probabilities,
        properties["bbox_used_for_cropping"],
        properties["shape_before_cropping"],
    )
    segmentation = label_manager.convert_probabilities_to_segmentation(restored_probabilities)
    if isinstance(segmentation, torch.Tensor):
        segmentation = segmentation.cpu().numpy()
    return restored_probabilities, segmentation.astype(np.uint16, copy=False)


def write_segmentation_nifti(segmentation: np.ndarray, properties: dict, output_path: Path) -> None:
    image = sitk.GetImageFromArray(segmentation.astype(np.uint16, copy=False))
    sitk_stuff = properties["sitk_stuff"]
    image.SetSpacing(tuple(float(v) for v in sitk_stuff["spacing"]))
    image.SetOrigin(tuple(float(v) for v in sitk_stuff["origin"]))
    image.SetDirection(tuple(float(v) for v in sitk_stuff["direction"]))
    sitk.WriteImage(image, str(output_path), useCompression=True)


def compute_case_metrics(
    reference_segmentation: np.ndarray,
    predicted_segmentation: np.ndarray,
    regions: Iterable[int | tuple[int, ...]],
) -> dict:
    metrics: dict[str, dict[str, float]] = {}
    for region in regions:
        if np.isscalar(region):
            ref_mask = reference_segmentation == region
            pred_mask = predicted_segmentation == region
            region_key = str(int(region))
        else:
            ref_mask = np.isin(reference_segmentation, list(region))
            pred_mask = np.isin(predicted_segmentation, list(region))
            region_key = str(tuple(int(v) for v in region))

        tp = float(np.logical_and(ref_mask, pred_mask).sum())
        fp = float(np.logical_and(~ref_mask, pred_mask).sum())
        fn = float(np.logical_and(ref_mask, ~pred_mask).sum())
        if tp + fp + fn == 0:
            dice = float("nan")
            iou = float("nan")
        else:
            dice = 2 * tp / (2 * tp + fp + fn)
            iou = tp / (tp + fp + fn)

        metrics[region_key] = {
            "Dice": dice,
            "IoU": iou,
            "TP": tp,
            "FP": fp,
            "FN": fn,
        }
    return metrics


def summarize_validation_metrics(case_metrics: list[dict]) -> dict:
    if not case_metrics:
        return {"metric_per_case": [], "mean": {}, "foreground_mean": {}}

    labels = list(case_metrics[0]["metrics"].keys())
    mean_metrics: dict[str, dict[str, float]] = {}
    for label in labels:
        metric_names = case_metrics[0]["metrics"][label].keys()
        mean_metrics[label] = {
            metric_name: float(
                np.nanmean([case["metrics"][label][metric_name] for case in case_metrics])
            )
            for metric_name in metric_names
        }

    foreground_mean = {}
    for metric_name in case_metrics[0]["metrics"][labels[0]].keys():
        foreground_mean[metric_name] = float(
            np.nanmean([mean_metrics[label][metric_name] for label in labels])
        )

    return {
        "metric_per_case": case_metrics,
        "mean": mean_metrics,
        "foreground_mean": foreground_mean,
    }
