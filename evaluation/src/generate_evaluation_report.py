#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from project import get_project_root
from training.src.utils import make_json_safe

FIG_DPI = 220
MODALITY_ORDER = ("t1", "t1ce", "t2", "flair")
PLANE_ORDER = ("axial", "coronal", "sagittal")
REGION_DISPLAY_NAMES = {
    "(1, 2, 3)": "Whole Tumor (WT)",
    "(2, 3)": "Tumor Core (TC)",
    "(3,)": "Enhancing Tumor (ET)",
}
REGION_COLORS = {
    "(1, 2, 3)": "#355070",
    "(2, 3)": "#b56576",
    "(3,)": "#e56b6f",
}
REGION_TO_LABELS = {
    "(1, 2, 3)": (1, 2, 3),
    "(2, 3)": (2, 3),
    "(3,)": (3,),
}
SEG_COLORS = {
    0: "#000000",
    1: "#e63946",
    2: "#ffb703",
    3: "#8d99ae",
    4: "#00b4d8",
}
ERROR_COLORS = {
    "tp": "#06d6a0",
    "fp": "#ef476f",
    "fn": "#ffd166",
}


WORKSPACE_ROOT = get_project_root()


def resolve_workspace_path(relative_or_absolute: str | Path) -> Path:
    candidate = Path(relative_or_absolute)
    if candidate.is_absolute():
        return candidate.resolve()
    return (WORKSPACE_ROOT / candidate).resolve()


def to_workspace_relative_string(path: Path) -> str:
    absolute = path.resolve()
    return os.path.relpath(absolute, WORKSPACE_ROOT)


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compute_display_range(volume: np.ndarray) -> tuple[float, float]:
    nonzero = volume[volume != 0]
    values = nonzero if nonzero.size else volume.reshape(-1)
    if values.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(values, [1.0, 99.0])
    if (
        not np.isfinite(vmin)
        or not np.isfinite(vmax)
        or math.isclose(float(vmin), float(vmax))
    ):
        vmin = float(np.min(values))
        vmax = float(np.max(values))
        if math.isclose(vmin, vmax):
            vmax = vmin + 1.0
    return float(vmin), float(vmax)


def get_display_slice(volume: np.ndarray, plane: str, index: int) -> np.ndarray:
    if plane == "axial":
        slice_2d = volume[:, :, index]
    elif plane == "coronal":
        slice_2d = volume[:, index, :]
    elif plane == "sagittal":
        slice_2d = volume[index, :, :]
    else:
        raise ValueError(f"Unsupported plane: {plane}")
    return np.asarray(slice_2d).T


def count_nonzero_per_slice(mask: np.ndarray, plane: str) -> np.ndarray:
    if plane == "axial":
        return np.count_nonzero(mask, axis=(0, 1))
    if plane == "coronal":
        return np.count_nonzero(mask, axis=(0, 2))
    if plane == "sagittal":
        return np.count_nonzero(mask, axis=(1, 2))
    raise ValueError(f"Unsupported plane: {plane}")


def select_best_slice(mask: np.ndarray, plane: str) -> int:
    counts = count_nonzero_per_slice(mask, plane)
    if counts.size == 0:
        return 0
    return int(np.argmax(counts))


def build_seg_colormap() -> tuple[Any, Any]:
    colors = [SEG_COLORS.get(index, "#cccccc") for index in range(max(SEG_COLORS) + 1)]
    cmap = ListedColormap(colors)
    bounds = np.arange(-0.5, len(colors) + 0.5, 1.0)
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm


def build_error_handles() -> list[Any]:
    return [
        Patch(facecolor=ERROR_COLORS["tp"], edgecolor="black", label="TP"),
        Patch(facecolor=ERROR_COLORS["fp"], edgecolor="black", label="FP"),
        Patch(facecolor=ERROR_COLORS["fn"], edgecolor="black", label="FN"),
    ]


def save_figure(fig: Any, output_path: Path) -> None:
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return math.nan
    return float(numerator / denominator)


def safe_nanmean(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return math.nan
    return float(np.mean(finite))


def metric_sort_value(value: float) -> float:
    return float(value) if np.isfinite(value) else float("-inf")


def metric_display(value: float) -> str:
    return f"{value:.3f}" if np.isfinite(value) else "n/a"


def metric_plot_value(value: float) -> float:
    return float(value) if np.isfinite(value) else 0.0


def parse_case_metrics(summary: dict[str, Any]) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    for item in summary["metric_per_case"]:
        prediction_file = Path(item["prediction_file"]).resolve()
        reference_file = Path(item["reference_file"]).resolve()
        case_id = prediction_file.name.replace(".nii.gz", "")
        wt = float(item["metrics"]["(1, 2, 3)"]["Dice"])
        tc = float(item["metrics"]["(2, 3)"]["Dice"])
        et = float(item["metrics"]["(3,)"]["Dice"])
        parsed.append(
            {
                "case_id": case_id,
                "prediction_file": prediction_file,
                "reference_file": reference_file,
                "metrics": item["metrics"],
                "dice_wt": wt,
                "dice_tc": tc,
                "dice_et": et,
                "dice_mean": float(np.mean([wt, tc, et])),
                "tp_wt": float(item["metrics"]["(1, 2, 3)"]["TP"]),
                "fp_wt": float(item["metrics"]["(1, 2, 3)"]["FP"]),
                "fn_wt": float(item["metrics"]["(1, 2, 3)"]["FN"]),
                "tp_tc": float(item["metrics"]["(2, 3)"]["TP"]),
                "fp_tc": float(item["metrics"]["(2, 3)"]["FP"]),
                "fn_tc": float(item["metrics"]["(2, 3)"]["FN"]),
                "tp_et": float(item["metrics"]["(3,)"]["TP"]),
                "fp_et": float(item["metrics"]["(3,)"]["FP"]),
                "fn_et": float(item["metrics"]["(3,)"]["FN"]),
                "precision_wt": safe_ratio(
                    item["metrics"]["(1, 2, 3)"]["TP"],
                    item["metrics"]["(1, 2, 3)"]["TP"]
                    + item["metrics"]["(1, 2, 3)"]["FP"],
                ),
                "precision_tc": safe_ratio(
                    item["metrics"]["(2, 3)"]["TP"],
                    item["metrics"]["(2, 3)"]["TP"] + item["metrics"]["(2, 3)"]["FP"],
                ),
                "precision_et": safe_ratio(
                    item["metrics"]["(3,)"]["TP"],
                    item["metrics"]["(3,)"]["TP"] + item["metrics"]["(3,)"]["FP"],
                ),
                "recall_wt": safe_ratio(
                    item["metrics"]["(1, 2, 3)"]["TP"],
                    item["metrics"]["(1, 2, 3)"]["TP"]
                    + item["metrics"]["(1, 2, 3)"]["FN"],
                ),
                "recall_tc": safe_ratio(
                    item["metrics"]["(2, 3)"]["TP"],
                    item["metrics"]["(2, 3)"]["TP"] + item["metrics"]["(2, 3)"]["FN"],
                ),
                "recall_et": safe_ratio(
                    item["metrics"]["(3,)"]["TP"],
                    item["metrics"]["(3,)"]["TP"] + item["metrics"]["(3,)"]["FN"],
                ),
            }
        )
        parsed[-1]["dice_mean"] = safe_nanmean([wt, tc, et])
    parsed.sort(key=lambda item: metric_sort_value(item["dice_mean"]), reverse=True)
    return parsed


def find_case_modalities(case_id: str, raw_dataset_dir: Path) -> dict[str, Path]:
    images_tr = raw_dataset_dir / "imagesTr"
    modality_files = {
        role: images_tr / f"{case_id}_{index:04d}.nii.gz"
        for index, role in enumerate(MODALITY_ORDER)
    }
    missing = [str(path) for path in modality_files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing raw image modalities for {case_id}: {missing}"
        )
    return modality_files


def load_nifti_array(path: Path, *, as_int: bool = False) -> np.ndarray:
    image = nib.load(str(path))
    if as_int:
        return np.rint(np.asanyarray(image.dataobj)).astype(np.int16)
    return np.asarray(image.get_fdata(dtype=np.float32), dtype=np.float32)


def plot_region_mean_dice(summary: dict[str, Any], output_path: Path) -> None:
    region_keys = ["(1, 2, 3)", "(2, 3)", "(3,)"]
    values = [float(summary["mean"][key]["Dice"]) for key in region_keys]
    plot_values = [metric_plot_value(value) for value in values]
    labels = [REGION_DISPLAY_NAMES[key] for key in region_keys]
    colors = [REGION_COLORS[key] for key in region_keys]

    fig, ax = plt.subplots(figsize=(9, 6), dpi=FIG_DPI)
    bars = ax.bar(labels, plot_values, color=colors)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Dice")
    ax.set_title("Mean Dice by BraTS region")
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.02,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    save_figure(fig, output_path)


def plot_case_ranking(case_metrics: list[dict[str, Any]], output_path: Path) -> None:
    labels = [item["case_id"] for item in case_metrics]
    values = [item["dice_mean"] for item in case_metrics]
    plot_values = [metric_plot_value(value) for value in values]
    colors = [
        "#2a9d8f" if idx < 3 else "#e76f51" if idx >= len(values) - 3 else "#577590"
        for idx in range(len(values))
    ]

    fig, ax = plt.subplots(figsize=(12, 6), dpi=FIG_DPI)
    bars = ax.bar(range(len(labels)), plot_values, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Mean Dice across WT / TC / ET")
    ax.set_title("Per-case Dice ranking on the evaluated subset")
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            metric_plot_value(value) + 0.01,
            metric_display(value),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    save_figure(fig, output_path)


def plot_case_region_heatmap(
    case_metrics: list[dict[str, Any]], output_path: Path
) -> None:
    region_keys = ["(1, 2, 3)", "(2, 3)", "(3,)"]
    data = np.asarray(
        [
            [float(item["metrics"][key]["Dice"]) for key in region_keys]
            for item in case_metrics
        ],
        dtype=float,
    )
    labels = [item["case_id"] for item in case_metrics]
    region_labels = [REGION_DISPLAY_NAMES[key] for key in region_keys]

    fig, ax = plt.subplots(figsize=(8, max(4, len(labels) * 0.7)), dpi=FIG_DPI)
    image = ax.imshow(data, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(region_labels)))
    ax.set_xticklabels(region_labels)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Per-case region Dice heatmap")
    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            ax.text(
                col,
                row,
                metric_display(float(data[row, col])),
                ha="center",
                va="center",
                color="white",
                fontsize=8,
            )
    fig.colorbar(image, ax=ax, label="Dice")
    fig.tight_layout()
    save_figure(fig, output_path)


def plot_region_error_breakdown(
    case_metrics: list[dict[str, Any]], output_path: Path
) -> None:
    region_specs = [
        ("WT", "fp_wt", "fn_wt", "#355070"),
        ("TC", "fp_tc", "fn_tc", "#b56576"),
        ("ET", "fp_et", "fn_et", "#e56b6f"),
    ]
    labels = [spec[0] for spec in region_specs]
    fp_values = [
        float(np.mean([item[spec[1]] for item in case_metrics]))
        for spec in region_specs
    ]
    fn_values = [
        float(np.mean([item[spec[2]] for item in case_metrics]))
        for spec in region_specs
    ]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 6), dpi=FIG_DPI)
    bars_fp = ax.bar(x - width / 2, fp_values, width, label="FP", color="#ef476f")
    bars_fn = ax.bar(x + width / 2, fn_values, width, label="FN", color="#ffd166")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean voxel count per evaluated case")
    ax.set_title("Average false-positive and false-negative burden by region")
    ax.legend()
    for bars in (bars_fp, bars_fn):
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                value,
                f"{value:.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.tight_layout()
    save_figure(fig, output_path)


def plot_region_precision_recall(
    case_metrics: list[dict[str, Any]], output_path: Path
) -> None:
    region_specs = [
        ("WT", "precision_wt", "recall_wt", "#355070"),
        ("TC", "precision_tc", "recall_tc", "#b56576"),
        ("ET", "precision_et", "recall_et", "#e56b6f"),
    ]
    labels = [spec[0] for spec in region_specs]
    precision_values = [
        safe_nanmean([item[spec[1]] for item in case_metrics]) for spec in region_specs
    ]
    recall_values = [
        safe_nanmean([item[spec[2]] for item in case_metrics]) for spec in region_specs
    ]
    plot_precision_values = [metric_plot_value(value) for value in precision_values]
    plot_recall_values = [metric_plot_value(value) for value in recall_values]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 6), dpi=FIG_DPI)
    bars_precision = ax.bar(
        x - width / 2, plot_precision_values, width, label="Precision", color="#2a9d8f"
    )
    bars_recall = ax.bar(
        x + width / 2, plot_recall_values, width, label="Recall", color="#e9c46a"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Average precision and recall by region")
    ax.legend()
    for bar, value in zip(bars_precision, precision_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            metric_display(value),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    for bar, value in zip(bars_recall, recall_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            metric_display(value),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    save_figure(fig, output_path)


def plot_dice_vs_reference_volume(
    case_metrics: list[dict[str, Any]], output_path: Path
) -> None:
    x = [item["metrics"]["(1, 2, 3)"]["n_ref"] for item in case_metrics]
    y = [item["dice_wt"] for item in case_metrics]
    labels = [item["case_id"] for item in case_metrics]

    fig, ax = plt.subplots(figsize=(9, 6), dpi=FIG_DPI)
    ax.scatter(x, y, c="#355070", s=70, alpha=0.85)
    ax.set_xlabel("WT reference voxels")
    ax.set_ylabel("WT Dice")
    ax.set_title("WT Dice vs reference tumor burden")
    for case_id, xv, yv in zip(labels, x, y):
        ax.text(xv, yv, case_id.split("_")[-1], fontsize=8, ha="left", va="bottom")
    fig.tight_layout()
    save_figure(fig, output_path)


def plot_region_volume_bias(
    case_metrics: list[dict[str, Any]], output_path: Path
) -> None:
    region_specs = [
        ("WT", "(1, 2, 3)", "#355070"),
        ("TC", "(2, 3)", "#b56576"),
        ("ET", "(3,)", "#e56b6f"),
    ]
    labels = [spec[0] for spec in region_specs]
    pred_minus_ref = [
        float(
            np.mean(
                [
                    item["metrics"][spec[1]]["n_pred"]
                    - item["metrics"][spec[1]]["n_ref"]
                    for item in case_metrics
                ]
            )
        )
        for spec in region_specs
    ]

    fig, ax = plt.subplots(figsize=(9, 6), dpi=FIG_DPI)
    bars = ax.bar(labels, pred_minus_ref, color=[spec[2] for spec in region_specs])
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_ylabel("Mean predicted voxels - reference voxels")
    ax.set_title("Average volume bias by region")
    for bar, value in zip(bars, pred_minus_ref):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value,
            f"{value:.0f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=8,
        )
    fig.tight_layout()
    save_figure(fig, output_path)


def plot_region_case_ranking(
    case_metrics: list[dict[str, Any]], region_key: str, output_path: Path
) -> None:
    region_to_attr = {
        "(1, 2, 3)": "dice_wt",
        "(2, 3)": "dice_tc",
        "(3,)": "dice_et",
    }
    region_name = REGION_DISPLAY_NAMES[region_key]
    attr = region_to_attr[region_key]
    ordered = sorted(
        case_metrics, key=lambda item: metric_sort_value(item[attr]), reverse=True
    )
    labels = [item["case_id"] for item in ordered]
    values = [item[attr] for item in ordered]
    plot_values = [metric_plot_value(value) for value in values]

    fig, ax = plt.subplots(figsize=(12, 6), dpi=FIG_DPI)
    bars = ax.bar(range(len(labels)), plot_values, color=REGION_COLORS[region_key])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Dice")
    ax.set_title(f"{region_name} case ranking")
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            metric_plot_value(value) + 0.01,
            metric_display(value),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    save_figure(fig, output_path)


def compute_case_error_analysis(
    case_metrics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    analysis: list[dict[str, Any]] = []
    for item in case_metrics:
        dominant_issue = max(
            [
                ("WT_FN", item["fn_wt"]),
                ("WT_FP", item["fp_wt"]),
                ("TC_FN", item["fn_tc"]),
                ("TC_FP", item["fp_tc"]),
                ("ET_FN", item["fn_et"]),
                ("ET_FP", item["fp_et"]),
            ],
            key=lambda pair: pair[1],
        )[0]
        finite_regions = [
            (name, value)
            for name, value in [
                ("WT", item["dice_wt"]),
                ("TC", item["dice_tc"]),
                ("ET", item["dice_et"]),
            ]
            if np.isfinite(value)
        ]
        strongest_region = (
            min(finite_regions, key=lambda pair: pair[1])[0]
            if finite_regions
            else "n/a"
        )
        analysis.append(
            {
                "case_id": item["case_id"],
                "dice_mean": item["dice_mean"],
                "dice_wt": item["dice_wt"],
                "dice_tc": item["dice_tc"],
                "dice_et": item["dice_et"],
                "wt_ref_voxels": item["metrics"]["(1, 2, 3)"]["n_ref"],
                "wt_pred_voxels": item["metrics"]["(1, 2, 3)"]["n_pred"],
                "dominant_error_mode": dominant_issue,
                "weakest_region": strongest_region,
                "fp_fn_balance_wt": item["fp_wt"] - item["fn_wt"],
                "fp_fn_balance_tc": item["fp_tc"] - item["fn_tc"],
                "fp_fn_balance_et": item["fp_et"] - item["fn_et"],
                "precision_wt": item["precision_wt"],
                "precision_tc": item["precision_tc"],
                "precision_et": item["precision_et"],
                "recall_wt": item["recall_wt"],
                "recall_tc": item["recall_tc"],
                "recall_et": item["recall_et"],
            }
        )
    return analysis


def write_case_metrics_csv(
    case_metrics: list[dict[str, Any]], output_path: Path
) -> None:
    fieldnames = [
        "case_id",
        "dice_mean",
        "dice_wt",
        "dice_tc",
        "dice_et",
        "tp_wt",
        "fp_wt",
        "fn_wt",
        "tp_tc",
        "fp_tc",
        "fn_tc",
        "tp_et",
        "fp_et",
        "fn_et",
        "precision_wt",
        "precision_tc",
        "precision_et",
        "recall_wt",
        "recall_tc",
        "recall_et",
        "prediction_file",
        "reference_file",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in case_metrics:
            writer.writerow({key: item[key] for key in fieldnames})


def write_case_analysis_csv(
    case_analysis: list[dict[str, Any]], output_path: Path
) -> None:
    fieldnames = list(case_analysis[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in case_analysis:
            writer.writerow(item)


def build_dataset_analysis(
    case_metrics: list[dict[str, Any]], summary: dict[str, Any]
) -> dict[str, Any]:
    sorted_by_mean = sorted(
        case_metrics,
        key=lambda item: metric_sort_value(item["dice_mean"]),
        reverse=True,
    )
    weakest_cases = [item["case_id"] for item in sorted_by_mean[-3:]]
    strongest_cases = [item["case_id"] for item in sorted_by_mean[:3]]
    return {
        "evaluated_case_count": len(case_metrics),
        "foreground_mean_dice": float(summary["foreground_mean"]["Dice"]),
        "mean_region_dice": {
            REGION_DISPLAY_NAMES[key]: float(summary["mean"][key]["Dice"])
            for key in ["(1, 2, 3)", "(2, 3)", "(3,)"]
        },
        "best_case": strongest_cases[0],
        "worst_case": weakest_cases[-1],
        "top3_cases": strongest_cases,
        "bottom3_cases": weakest_cases,
        "mean_dice_std": (
            float(
                np.std(
                    [
                        item["dice_mean"]
                        for item in case_metrics
                        if np.isfinite(item["dice_mean"])
                    ]
                )
            )
            if any(np.isfinite(item["dice_mean"]) for item in case_metrics)
            else math.nan
        ),
        "cases_with_et_zero_dice": [
            item["case_id"]
            for item in case_metrics
            if np.isfinite(item["dice_et"]) and item["dice_et"] <= 1e-8
        ],
    }


def write_analysis_json(analysis: dict[str, Any], output_path: Path) -> None:
    output_path.write_text(
        json.dumps(make_json_safe(analysis), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def render_case_analysis_markdown(
    case_analysis: list[dict[str, Any]], output_path: Path
) -> None:
    lines = [
        "# Case Analysis",
        "",
        "| Case | Mean Dice | Weakest Region | Dominant Error | WT FP-FN | TC FP-FN | ET FP-FN | WT P/R | TC P/R | ET P/R |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for item in case_analysis:
        lines.append(
            f"| `{item['case_id']}` | {metric_display(item['dice_mean'])} | {item['weakest_region']} | "
            f"{item['dominant_error_mode']} | {item['fp_fn_balance_wt']:.0f} | "
            f"{item['fp_fn_balance_tc']:.0f} | {item['fp_fn_balance_et']:.0f} | "
            f"{metric_display(item['precision_wt'])}/{metric_display(item['recall_wt'])} | "
            f"{metric_display(item['precision_tc'])}/{metric_display(item['recall_tc'])} | "
            f"{metric_display(item['precision_et'])}/{metric_display(item['recall_et'])} |"
        )
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _region_mask(seg: np.ndarray, region_key: str) -> np.ndarray:
    labels = REGION_TO_LABELS[region_key]
    mask = np.zeros_like(seg, dtype=bool)
    for label in labels:
        mask |= seg == label
    return mask


def plot_case_region_error_map(
    case_metrics: dict[str, Any],
    raw_dataset_dir: Path,
    region_key: str,
    output_path: Path,
) -> None:
    case_id = case_metrics["case_id"]
    modality_files = find_case_modalities(case_id, raw_dataset_dir)
    base_volume = load_nifti_array(modality_files["flair"])
    seg_gt = load_nifti_array(case_metrics["reference_file"], as_int=True)
    seg_pred = load_nifti_array(case_metrics["prediction_file"], as_int=True)
    gt_mask = _region_mask(seg_gt, region_key)
    pred_mask = _region_mask(seg_pred, region_key)
    focus_mask = gt_mask | pred_mask
    vmin, vmax = compute_display_range(base_volume)

    fig, axes = plt.subplots(2, len(PLANE_ORDER), figsize=(18, 8), dpi=FIG_DPI)
    handles = build_error_handles()
    region_name = REGION_DISPLAY_NAMES[region_key]

    for col, plane in enumerate(PLANE_ORDER):
        index = select_best_slice(focus_mask, plane)
        base_slice = get_display_slice(base_volume, plane, index)
        gt_slice = get_display_slice(gt_mask.astype(np.uint8), plane, index).astype(
            bool
        )
        pred_slice = get_display_slice(pred_mask.astype(np.uint8), plane, index).astype(
            bool
        )

        tp = gt_slice & pred_slice
        fp = (~gt_slice) & pred_slice
        fn = gt_slice & (~pred_slice)

        ax_mask = axes[0, col]
        ax_err = axes[1, col]
        ax_mask.imshow(base_slice, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        overlay = np.zeros((*gt_slice.shape, 4), dtype=float)
        overlay[gt_slice] = [0.0, 0.7, 1.0, 0.30]
        overlay[pred_slice] = [1.0, 0.0, 0.6, 0.30]
        ax_mask.imshow(overlay, origin="lower")
        ax_mask.set_title(
            f"{region_name} | {plane} | GT cyan / Pred magenta", fontsize=11
        )
        ax_mask.axis("off")

        ax_err.imshow(base_slice, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        error_rgb = np.zeros((*tp.shape, 4), dtype=float)
        error_rgb[tp] = [0.023, 0.839, 0.627, 0.55]
        error_rgb[fp] = [0.937, 0.278, 0.435, 0.60]
        error_rgb[fn] = [1.0, 0.820, 0.400, 0.60]
        ax_err.imshow(error_rgb, origin="lower")
        ax_err.set_title(f"{region_name} error | {plane}", fontsize=11)
        ax_err.axis("off")

    fig.suptitle(
        f"{case_id} | {region_name} region-specific error analysis", fontsize=16
    )
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=3)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    save_figure(fig, output_path)


def build_conclusion_lines(
    case_metrics: list[dict[str, Any]], dataset_analysis: dict[str, Any]
) -> list[str]:
    mean_wt = safe_nanmean([item["dice_wt"] for item in case_metrics])
    mean_tc = safe_nanmean([item["dice_tc"] for item in case_metrics])
    mean_et = safe_nanmean([item["dice_et"] for item in case_metrics])
    finite_regions = [
        (name, value)
        for name, value in [("WT", mean_wt), ("TC", mean_tc), ("ET", mean_et)]
        if np.isfinite(value)
    ]
    weakest_region = (
        min(finite_regions, key=lambda pair: pair[1])[0] if finite_regions else "n/a"
    )
    dominant_error_mode = max(
        [
            (
                "WT false negatives",
                float(np.mean([item["fn_wt"] for item in case_metrics])),
            ),
            (
                "WT false positives",
                float(np.mean([item["fp_wt"] for item in case_metrics])),
            ),
            (
                "TC false negatives",
                float(np.mean([item["fn_tc"] for item in case_metrics])),
            ),
            (
                "TC false positives",
                float(np.mean([item["fp_tc"] for item in case_metrics])),
            ),
            (
                "ET false negatives",
                float(np.mean([item["fn_et"] for item in case_metrics])),
            ),
            (
                "ET false positives",
                float(np.mean([item["fp_et"] for item in case_metrics])),
            ),
        ],
        key=lambda pair: pair[1],
    )[0]
    lines = [
        f"- 当前评估子集共有 `{len(case_metrics)}` 个病例，foreground mean Dice 为 `{dataset_analysis['foreground_mean_dice']:.3f}`。",
        f"- 三个 BraTS region 里当前最弱的是 `{weakest_region}`，说明这一块最值得优先排查。",
        f"- 平均来看，当前最主要的失败模式是 `{dominant_error_mode}`。",
        f"- 最好病例是 `{dataset_analysis['best_case']}`，最差病例是 `{dataset_analysis['worst_case']}`，两者之间的表现差异很大。",
    ]
    et_zero = dataset_analysis.get("cases_with_et_zero_dice", [])
    if et_zero:
        lines.append(
            f"- `ET Dice = 0` 的病例有：`{', '.join(et_zero)}`，这通常意味着增强肿瘤区域几乎没有被正确抓到。"
        )
    return lines


def render_per_case_markdown(
    case_metrics: dict[str, Any], output_path: Path, overlay_filename: str
) -> None:
    lines = [
        f"# {case_metrics['case_id']}",
        "",
        f"- mean Dice: `{metric_display(case_metrics['dice_mean'])}`",
        f"- WT Dice: `{metric_display(case_metrics['dice_wt'])}`",
        f"- TC Dice: `{metric_display(case_metrics['dice_tc'])}`",
        f"- ET Dice: `{metric_display(case_metrics['dice_et'])}`",
        f"- WT Precision / Recall: `{metric_display(case_metrics['precision_wt'])} / {metric_display(case_metrics['recall_wt'])}`",
        f"- TC Precision / Recall: `{metric_display(case_metrics['precision_tc'])} / {metric_display(case_metrics['recall_tc'])}`",
        f"- ET Precision / Recall: `{metric_display(case_metrics['precision_et'])} / {metric_display(case_metrics['recall_et'])}`",
        f"- prediction: `{to_workspace_relative_string(case_metrics['prediction_file'])}`",
        f"- reference: `{to_workspace_relative_string(case_metrics['reference_file'])}`",
        "",
        "## Metric Formulas",
        "",
        r"$$\mathrm{Dice} = \frac{2TP}{2TP + FP + FN}$$",
        "",
        r"$$\mathrm{IoU} = \frac{TP}{TP + FP + FN}$$",
        "",
        r"$$\mathrm{Precision} = \frac{TP}{TP + FP}$$",
        "",
        r"$$\mathrm{Recall} = \frac{TP}{TP + FN}$$",
        "",
        "## Formula Substitution",
        "",
        rf"WT Dice: $$\frac{{2 \times {case_metrics['tp_wt']:.0f}}}{{2 \times {case_metrics['tp_wt']:.0f} + {case_metrics['fp_wt']:.0f} + {case_metrics['fn_wt']:.0f}}} = {metric_display(case_metrics['dice_wt'])}$$",
        "",
        rf"TC Dice: $$\frac{{2 \times {case_metrics['tp_tc']:.0f}}}{{2 \times {case_metrics['tp_tc']:.0f} + {case_metrics['fp_tc']:.0f} + {case_metrics['fn_tc']:.0f}}} = {metric_display(case_metrics['dice_tc'])}$$",
        "",
        rf"ET Dice: $$\frac{{2 \times {case_metrics['tp_et']:.0f}}}{{2 \times {case_metrics['tp_et']:.0f} + {case_metrics['fp_et']:.0f} + {case_metrics['fn_et']:.0f}}} = {metric_display(case_metrics['dice_et'])}$$",
        "",
        rf"WT Precision: $$\frac{{{case_metrics['tp_wt']:.0f}}}{{{case_metrics['tp_wt']:.0f} + {case_metrics['fp_wt']:.0f}}} = {metric_display(case_metrics['precision_wt'])}$$",
        "",
        rf"WT Recall: $$\frac{{{case_metrics['tp_wt']:.0f}}}{{{case_metrics['tp_wt']:.0f} + {case_metrics['fn_wt']:.0f}}} = {metric_display(case_metrics['recall_wt'])}$$",
        "",
        "## Overlay",
        "",
        f"![{case_metrics['case_id']} Overlay]({overlay_filename})",
        "",
        "## Region-Specific Error Maps",
        "",
        f"![WT Error]({case_metrics['case_id']}_wt_error.png)",
        "",
        f"![TC Error]({case_metrics['case_id']}_tc_error.png)",
        "",
        f"![ET Error]({case_metrics['case_id']}_et_error.png)",
        "",
        "## Error Summary",
        "",
        f"- dominant error mode: `{max([('WT_FN', case_metrics['fn_wt']), ('WT_FP', case_metrics['fp_wt']), ('TC_FN', case_metrics['fn_tc']), ('TC_FP', case_metrics['fp_tc']), ('ET_FN', case_metrics['fn_et']), ('ET_FP', case_metrics['fp_et'])], key=lambda pair: pair[1])[0]}`",
        f"- WT FP/FN: `{case_metrics['fp_wt']:.0f} / {case_metrics['fn_wt']:.0f}`",
        f"- TC FP/FN: `{case_metrics['fp_tc']:.0f} / {case_metrics['fn_tc']:.0f}`",
        f"- ET FP/FN: `{case_metrics['fp_et']:.0f} / {case_metrics['fn_et']:.0f}`",
    ]
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def plot_case_overlay_panel(
    case_metrics: dict[str, Any],
    raw_dataset_dir: Path,
    output_path: Path,
) -> None:
    case_id = case_metrics["case_id"]
    modality_files = find_case_modalities(case_id, raw_dataset_dir)
    base_volume = load_nifti_array(modality_files["flair"])
    seg_gt = load_nifti_array(case_metrics["reference_file"], as_int=True)
    seg_pred = load_nifti_array(case_metrics["prediction_file"], as_int=True)
    vmin, vmax = compute_display_range(base_volume)
    seg_cmap, seg_norm = build_seg_colormap()

    fig, axes = plt.subplots(3, len(PLANE_ORDER), figsize=(18, 12), dpi=FIG_DPI)
    handles = build_error_handles()
    title_lines = [
        f"{case_id}",
        f"Mean Dice={metric_display(case_metrics['dice_mean'])}",
        f"WT={metric_display(case_metrics['dice_wt'])} | TC={metric_display(case_metrics['dice_tc'])} | ET={metric_display(case_metrics['dice_et'])}",
    ]

    for col, plane in enumerate(PLANE_ORDER):
        focus_mask = (seg_gt > 0) | (seg_pred > 0)
        index = select_best_slice(focus_mask, plane)
        base_slice = get_display_slice(base_volume, plane, index)
        gt_slice = get_display_slice(seg_gt, plane, index)
        pred_slice = get_display_slice(seg_pred, plane, index)

        tp = (gt_slice > 0) & (pred_slice > 0)
        fp = (gt_slice == 0) & (pred_slice > 0)
        fn = (gt_slice > 0) & (pred_slice == 0)

        ax_gt = axes[0, col]
        ax_pred = axes[1, col]
        ax_err = axes[2, col]

        for ax in (ax_gt, ax_pred, ax_err):
            ax.imshow(base_slice, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
            ax.axis("off")

        ax_gt.imshow(
            np.ma.masked_where(gt_slice == 0, gt_slice),
            cmap=seg_cmap,
            norm=seg_norm,
            origin="lower",
            alpha=0.45,
        )
        ax_gt.set_title(f"GT | {plane} | slice {index}", fontsize=11)

        ax_pred.imshow(
            np.ma.masked_where(pred_slice == 0, pred_slice),
            cmap=seg_cmap,
            norm=seg_norm,
            origin="lower",
            alpha=0.45,
        )
        ax_pred.set_title(f"Prediction | {plane} | slice {index}", fontsize=11)

        error_rgb = np.zeros((*tp.shape, 4), dtype=float)
        error_rgb[tp] = [0.023, 0.839, 0.627, 0.55]
        error_rgb[fp] = [0.937, 0.278, 0.435, 0.60]
        error_rgb[fn] = [1.0, 0.820, 0.400, 0.60]
        ax_err.imshow(error_rgb, origin="lower")
        ax_err.set_title(f"Error map | TP green / FP red / FN yellow", fontsize=11)

    fig.suptitle("\n".join(title_lines), fontsize=16)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=3)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    save_figure(fig, output_path)


def render_report_markdown(
    output_path: Path,
    summary_path: Path,
    case_metrics: list[dict[str, Any]],
    sample_selection: dict[str, Any] | None,
    dataset_analysis: dict[str, Any],
) -> None:
    best_case = case_metrics[0]
    worst_case = case_metrics[-1]
    median_case = case_metrics[len(case_metrics) // 2]
    lines: list[str] = []
    lines.append("# Evaluation Report")
    lines.append("")
    lines.append(f"- summary source: `{to_workspace_relative_string(summary_path)}`")
    lines.append(f"- evaluated cases: `{len(case_metrics)}`")
    if sample_selection is not None:
        lines.append(
            f"- sampled from training set: `{sample_selection.get('sample_count')}` cases"
        )
        lines.append(f"- sample seed: `{sample_selection.get('sample_seed')}`")
    lines.append("")
    lines.append("## Conclusions")
    lines.append("")
    lines.extend(build_conclusion_lines(case_metrics, dataset_analysis))
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append("![Mean Dice by Region](mean_dice_by_region.png)")
    lines.append("")
    lines.append("![Per-case Ranking](case_ranking.png)")
    lines.append("")
    lines.append("![Per-case Heatmap](case_region_heatmap.png)")
    lines.append("")
    lines.append("![FP/FN Burden by Region](region_error_breakdown.png)")
    lines.append("")
    lines.append("![Precision / Recall by Region](region_precision_recall.png)")
    lines.append("")
    lines.append("![Region Volume Bias](region_volume_bias.png)")
    lines.append("")
    lines.append("![WT Dice vs Reference Volume](dice_vs_reference_volume.png)")
    lines.append("")
    lines.append("![WT Case Ranking](wt_case_ranking.png)")
    lines.append("")
    lines.append("![TC Case Ranking](tc_case_ranking.png)")
    lines.append("")
    lines.append("![ET Case Ranking](et_case_ranking.png)")
    lines.append("")
    lines.append("## Case Highlights")
    lines.append("")
    lines.append(
        f"- best case: `{best_case['case_id']}` with mean Dice `{metric_display(best_case['dice_mean'])}`"
    )
    lines.append(
        f"- median case: `{median_case['case_id']}` with mean Dice `{metric_display(median_case['dice_mean'])}`"
    )
    lines.append(
        f"- worst case: `{worst_case['case_id']}` with mean Dice `{metric_display(worst_case['dice_mean'])}`"
    )
    lines.append("")
    lines.append(f"### Best Case: `{best_case['case_id']}`")
    lines.append("")
    lines.append("![Best Case Overlay](best_case_overlay.png)")
    lines.append("")
    lines.append(f"### Worst Case: `{worst_case['case_id']}`")
    lines.append("")
    lines.append("![Worst Case Overlay](worst_case_overlay.png)")
    lines.append("")
    lines.append("## Detailed Analysis Outputs")
    lines.append("")
    lines.append("## Metric Formulas")
    lines.append("")
    lines.append(
        r"$$TP = |Y_{pred}=1 \land Y_{ref}=1|, \quad FP = |Y_{pred}=1 \land Y_{ref}=0|, \quad FN = |Y_{pred}=0 \land Y_{ref}=1|$$"
    )
    lines.append("")
    lines.append(r"$$\mathrm{Dice} = \frac{2TP}{2TP + FP + FN}$$")
    lines.append("")
    lines.append(r"$$\mathrm{IoU} = \frac{TP}{TP + FP + FN}$$")
    lines.append("")
    lines.append(r"$$\mathrm{Precision} = \frac{TP}{TP + FP}$$")
    lines.append("")
    lines.append(r"$$\mathrm{Recall} = \frac{TP}{TP + FN}$$")
    lines.append("")
    lines.append("## Detailed Analysis Outputs")
    lines.append("")
    lines.append("- `analysis.json`: dataset-level summary and failure-pattern digest")
    lines.append("- `case_metrics.csv`: per-case Dice and TP/FP/FN statistics")
    lines.append(
        "- `case_analysis.csv`: per-case dominant failure mode and FP/FN balance"
    )
    lines.append("- `case_analysis.md`: markdown table version of the case analysis")
    lines.append("- `cases/`: per-case markdown pages and overlay figures")
    lines.append("")
    lines.append("## Per-case Table")
    lines.append("")
    lines.append("| Case | Mean Dice | WT | TC | ET |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for item in case_metrics:
        lines.append(
            f"| `{item['case_id']}` | {metric_display(item['dice_mean'])} | {metric_display(item['dice_wt'])} | {metric_display(item['dice_tc'])} | {metric_display(item['dice_et'])} |"
        )
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def generate_evaluation_report(
    summary_file: Path,
    predictions_dir: Path,
    raw_dataset_dir: Path,
    output_dir: Path,
    sample_selection_file: Path | None = None,
) -> None:
    ensure_output_dir(output_dir)
    cases_output_dir = output_dir / "cases"
    ensure_output_dir(cases_output_dir)
    summary = load_json(summary_file)
    case_metrics = parse_case_metrics(summary)
    if len(case_metrics) == 0:
        raise RuntimeError(f"No cases found in summary file: {summary_file}")

    summary_copy = output_dir / "summary.copy.json"
    summary_copy.write_text(
        json.dumps(make_json_safe(summary), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    case_analysis = compute_case_error_analysis(case_metrics)
    dataset_analysis = build_dataset_analysis(case_metrics, summary)
    write_analysis_json(dataset_analysis, output_dir / "analysis.json")
    write_case_metrics_csv(case_metrics, output_dir / "case_metrics.csv")
    write_case_analysis_csv(case_analysis, output_dir / "case_analysis.csv")
    render_case_analysis_markdown(case_analysis, output_dir / "case_analysis.md")

    plot_region_mean_dice(summary, output_dir / "mean_dice_by_region.png")
    plot_case_ranking(case_metrics, output_dir / "case_ranking.png")
    plot_case_region_heatmap(case_metrics, output_dir / "case_region_heatmap.png")
    plot_region_error_breakdown(case_metrics, output_dir / "region_error_breakdown.png")
    plot_region_precision_recall(
        case_metrics, output_dir / "region_precision_recall.png"
    )
    plot_region_volume_bias(case_metrics, output_dir / "region_volume_bias.png")
    plot_dice_vs_reference_volume(
        case_metrics, output_dir / "dice_vs_reference_volume.png"
    )
    plot_region_case_ranking(
        case_metrics, "(1, 2, 3)", output_dir / "wt_case_ranking.png"
    )
    plot_region_case_ranking(case_metrics, "(2, 3)", output_dir / "tc_case_ranking.png")
    plot_region_case_ranking(case_metrics, "(3,)", output_dir / "et_case_ranking.png")
    plot_case_overlay_panel(
        case_metrics[0], raw_dataset_dir, output_dir / "best_case_overlay.png"
    )
    plot_case_overlay_panel(
        case_metrics[-1], raw_dataset_dir, output_dir / "worst_case_overlay.png"
    )

    for item in case_metrics:
        case_overlay = cases_output_dir / f"{item['case_id']}_overlay.png"
        case_markdown = cases_output_dir / f"{item['case_id']}.md"
        plot_case_overlay_panel(item, raw_dataset_dir, case_overlay)
        plot_case_region_error_map(
            item,
            raw_dataset_dir,
            "(1, 2, 3)",
            cases_output_dir / f"{item['case_id']}_wt_error.png",
        )
        plot_case_region_error_map(
            item,
            raw_dataset_dir,
            "(2, 3)",
            cases_output_dir / f"{item['case_id']}_tc_error.png",
        )
        plot_case_region_error_map(
            item,
            raw_dataset_dir,
            "(3,)",
            cases_output_dir / f"{item['case_id']}_et_error.png",
        )
        render_per_case_markdown(item, case_markdown, case_overlay.name)

    sample_selection = None
    if sample_selection_file is not None and sample_selection_file.is_file():
        sample_selection = load_json(sample_selection_file)
        (output_dir / "sample_selection.copy.json").write_text(
            json.dumps(make_json_safe(sample_selection), indent=2, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )

    render_report_markdown(
        output_dir / "report.md",
        summary_file,
        case_metrics,
        sample_selection,
        dataset_analysis,
    )


def main(
    summary_file: Path,
    predictions_dir: Path,
    raw_dataset_dir: Path,
    output_dir: Path,
    sample_selection_file: Path | None = None,
) -> None:
    generate_evaluation_report(
        summary_file=summary_file,
        predictions_dir=predictions_dir,
        raw_dataset_dir=raw_dataset_dir,
        output_dir=output_dir,
        sample_selection_file=sample_selection_file,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a markdown evaluation report with visualizations."
    )
    parser.add_argument("--summary-file", required=True)
    parser.add_argument("--predictions-dir", required=True)
    parser.add_argument("--raw-dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-selection-file", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        summary_file=resolve_workspace_path(args.summary_file),
        predictions_dir=resolve_workspace_path(args.predictions_dir),
        raw_dataset_dir=resolve_workspace_path(args.raw_dataset_dir),
        output_dir=resolve_workspace_path(args.output_dir),
        sample_selection_file=(
            resolve_workspace_path(args.sample_selection_file)
            if args.sample_selection_file is not None
            else None
        ),
    )


__all__ = ["generate_evaluation_report", "main", "parse_args"]
