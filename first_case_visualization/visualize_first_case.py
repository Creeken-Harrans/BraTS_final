#!/usr/bin/env python3
"""
Dependencies:
    nibabel
    numpy
    matplotlib
    pandas

This script scans the BraTS archive recursively, finds the first BraTS-style
patient directory in sorted-path order, loads the 3D NIfTI volumes, and writes
text/JSON/CSV summaries plus multiple high-resolution visualizations.

BraTS data is a 3D medical image volume instead of a 2D JPG/PNG:
    - volume shape describes the size of the 3D array
    - axial/coronal/sagittal refer to three orthogonal slice directions
    - voxel spacing comes from the NIfTI header
    - seg stores voxel-wise tumor labels in the same 3D space
    - bounding box and overlay are defined in 3D and then projected to slices
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Protocol, cast

import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch, Rectangle

matplotlib.use("Agg")


DEFAULT_DATA_ROOT_REL = Path(
    "BraTS_final_Dataset/archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
)
DEFAULT_OUTPUT_DIR_REL = Path("first_case_visualization/output")

MODALITY_ORDER = ("t1", "t1ce", "t2", "flair")
PLANE_ORDER = ("axial", "coronal", "sagittal")
FIG_DPI = 220
MONTAGE_SLICES = 16
SEG_MONTAGE_SLICES = 12
DISPLAY_PERCENTILES = (1.0, 99.0)

KNOWN_LABEL_MEANINGS = {
    0: "Background",
    1: "Necrotic / non-enhancing tumor core (NCR/NET)",
    2: "Peritumoral edema (ED)",
    4: "Enhancing tumor (ET)",
}

SEG_COLORS = {
    0: "#000000",
    1: "#e63946",
    2: "#ffb703",
    3: "#8d99ae",
    4: "#00b4d8",
}

MODALITY_COLORS = {
    "t1": "#355070",
    "t1ce": "#6d597a",
    "t2": "#b56576",
    "flair": "#e56b6f",
}


class NiftiImageLike(Protocol):
    shape: tuple[int, ...]
    dataobj: Any
    affine: Any
    header: Any

    def get_fdata(self, dtype: Any = np.float32) -> np.ndarray: ...


def log(message: str) -> None:
    print(f"[BraTS visualize] {message}")


def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "first_case_visualization").is_dir():
            return parent
    raise RuntimeError(
        "Unable to locate BraTS_final project root from visualize_first_case.py"
    )


PROJECT_ROOT = find_project_root()


def resolve_workspace_path(relative_or_absolute: Path | str) -> Path:
    candidate = Path(relative_or_absolute)
    if candidate.is_absolute():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()


def to_workspace_relative_string(path: Path) -> str:
    absolute = path.resolve()
    return os.path.relpath(absolute, PROJECT_ROOT)


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def pretty_modality_name(role: str) -> str:
    return {
        "t1": "T1",
        "t1ce": "T1CE",
        "t2": "T2",
        "flair": "FLAIR",
        "seg": "SEG",
    }.get(role, role.upper())


def is_nifti_file(path: Path) -> bool:
    lowered = path.name.lower()
    return path.is_file() and (lowered.endswith(".nii") or lowered.endswith(".nii.gz"))


def strip_nifti_suffix(path: Path) -> str:
    lowered = path.name.lower()
    if lowered.endswith(".nii.gz"):
        return path.name[:-7]
    if lowered.endswith(".nii"):
        return path.name[:-4]
    return path.stem


def detect_series_role(path: Path) -> str | None:
    stem = strip_nifti_suffix(path).lower()
    tokens = [token for token in re.split(r"[^a-z0-9]+", stem) if token]
    compact = "".join(tokens)
    token_set = set(tokens)

    if "seg" in token_set or "segmentation" in token_set or compact.endswith("seg"):
        return "seg"
    if "flair" in token_set or compact.endswith("flair"):
        return "flair"
    if (
        "t1ce" in token_set
        or "t1gd" in token_set
        or compact.endswith("t1ce")
        or compact.endswith("t1gd")
    ):
        return "t1ce"
    if "t2" in token_set or compact.endswith("t2"):
        return "t2"
    if ("t1" in token_set or compact.endswith("t1")) and not (
        compact.endswith("t1ce") or compact.endswith("t1gd")
    ):
        return "t1"
    return None


def find_case_candidate_dirs(data_root: Path) -> list[Path]:
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")
    if not data_root.is_dir():
        raise NotADirectoryError(f"Data root is not a directory: {data_root}")

    grouped: dict[Path, list[tuple[str, Path]]] = {}
    for path in data_root.rglob("*"):
        if not is_nifti_file(path):
            continue
        role = detect_series_role(path)
        if role is None:
            continue
        grouped.setdefault(path.parent, []).append((role, path))

    candidates: list[Path] = []
    for directory, role_files in grouped.items():
        distinct_roles = {role for role, _ in role_files}
        if len(role_files) >= 4 and len(distinct_roles) >= 3:
            candidates.append(directory)

    return sorted(candidates)


def select_first_case_dir(data_root: Path) -> Path:
    candidates = find_case_candidate_dirs(data_root)
    if not candidates:
        raise FileNotFoundError(
            f"No BraTS-style patient directory was found under {data_root}. "
            "Expected a directory containing multiple recognized NIfTI files."
        )
    return candidates[0]


def discover_case_files(case_dir: Path) -> dict[str, Path]:
    nifti_files = sorted(path for path in case_dir.iterdir() if is_nifti_file(path))
    if not nifti_files:
        raise FileNotFoundError(
            f"Selected case directory contains no NIfTI files: {case_dir}"
        )

    discovered: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}
    unrecognized: list[Path] = []

    for path in nifti_files:
        role = detect_series_role(path)
        if role is None:
            unrecognized.append(path)
            continue
        if role in discovered:
            duplicates.setdefault(role, [discovered[role]]).append(path)
        else:
            discovered[role] = path

    if duplicates:
        duplicate_lines = []
        for role, files in sorted(duplicates.items()):
            duplicate_lines.append(
                f"{role}: {', '.join(str(path) for path in sorted(files))}"
            )
        raise ValueError(
            "Duplicate files matched the same role in the selected case directory:\n"
            + "\n".join(duplicate_lines)
        )

    missing = [role for role in (*MODALITY_ORDER, "seg") if role not in discovered]
    if missing:
        found_roles = ", ".join(sorted(discovered))
        found_files = "\n".join(f"  - {path.name}" for path in nifti_files)
        extra_text = ""
        if unrecognized:
            extra_text = "\nUnrecognized NIfTI files:\n" + "\n".join(
                f"  - {path.name}" for path in unrecognized
            )
        raise FileNotFoundError(
            f"The first patient directory in sorted order is incomplete: {case_dir}\n"
            f"Missing required roles: {', '.join(missing)}\n"
            f"Recognized roles: {found_roles or 'none'}\n"
            f"Files in directory:\n{found_files}{extra_text}"
        )

    return discovered


def load_volume(path: Path, role: str) -> dict[str, Any]:
    image = cast(NiftiImageLike, cast(object, nib.load(str(path))))

    if len(image.shape) != 3:
        raise ValueError(
            f"{role} file must be a 3D volume, but got shape {image.shape} from {path}"
        )

    if role == "seg":
        raw_data = np.asanyarray(image.dataobj)
        if not np.allclose(raw_data, np.rint(raw_data)):
            raise ValueError(
                f"Segmentation labels must be integer-like, but found non-integer values in {path}"
            )
        data = np.rint(raw_data).astype(np.int16)
    else:
        data = image.get_fdata(dtype=np.float32)
        data = np.asarray(data, dtype=np.float32)

    return {
        "role": role,
        "path": path,
        "image": image,
        "data": data,
        "shape": tuple(int(v) for v in image.shape),
        "dtype": str(image.header.get_data_dtype()),
        "spacing": tuple(float(v) for v in image.header.get_zooms()[:3]),
        "affine": np.asarray(image.affine, dtype=float),
    }


def validate_volume_alignment(records: dict[str, dict[str, Any]]) -> None:
    names = list(records)
    reference = records[names[0]]
    ref_shape = reference["shape"]
    ref_spacing = np.asarray(reference["spacing"], dtype=float)
    ref_affine = np.asarray(reference["affine"], dtype=float)

    mismatches: list[str] = []
    for role, record in records.items():
        if record["shape"] != ref_shape:
            mismatches.append(
                f"{role} shape {record['shape']} does not match reference shape {ref_shape}"
            )
        if not np.allclose(record["spacing"], ref_spacing, atol=1e-5):
            mismatches.append(
                f"{role} voxel spacing {record['spacing']} does not match reference spacing {tuple(ref_spacing)}"
            )
        if not np.allclose(record["affine"], ref_affine, atol=1e-4):
            mismatches.append(f"{role} affine does not match the reference affine")

    if mismatches:
        raise ValueError(
            "Volumes are not aligned in the same 3D space, so overlays would be unreliable:\n"
            + "\n".join(mismatches)
        )


def compute_seg_label_stats(seg_data: np.ndarray) -> list[dict[str, Any]]:
    labels, counts = np.unique(seg_data, return_counts=True)
    total_voxels = int(seg_data.size)
    stats: list[dict[str, Any]] = []
    for label, count in zip(labels, counts):
        label_int = int(label)
        voxel_count = int(count)
        stats.append(
            {
                "label": label_int,
                "name": KNOWN_LABEL_MEANINGS.get(
                    label_int, f"Unknown label {label_int}"
                ),
                "voxel_count": voxel_count,
                "fraction_of_volume": voxel_count / total_voxels,
            }
        )
    return stats


def serializable_affine(affine: np.ndarray) -> list[list[float]]:
    return [[round(float(value), 6) for value in row] for row in affine.tolist()]


def build_case_summary(
    case_dir: Path,
    case_files: dict[str, Path],
    records: dict[str, dict[str, Any]],
    seg_label_stats: list[dict[str, Any]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "case_dir": to_workspace_relative_string(case_dir),
        "files": {
            role: to_workspace_relative_string(case_files[role])
            for role in (*MODALITY_ORDER, "seg")
        },
        "volumes": {},
        "seg_labels": seg_label_stats,
    }

    for role in (*MODALITY_ORDER, "seg"):
        record = records[role]
        summary["volumes"][role] = {
            "path": to_workspace_relative_string(record["path"]),
            "shape": [int(v) for v in record["shape"]],
            "dtype": record["dtype"],
            "voxel_spacing": [float(v) for v in record["spacing"]],
            "affine": serializable_affine(record["affine"]),
        }

    return summary


def write_case_summary_text(summary: dict[str, Any], output_path: Path) -> None:
    lines: list[str] = []
    lines.append("BraTS First Case Summary")
    lines.append("=" * 80)
    lines.append(f"Case directory: {summary['case_dir']}")
    lines.append("")
    lines.append("Files")
    lines.append("-" * 80)

    for role in (*MODALITY_ORDER, "seg"):
        lines.append(f"{role}: {summary['files'][role]}")

    lines.append("")
    lines.append("Volume Metadata")
    lines.append("-" * 80)

    for role in (*MODALITY_ORDER, "seg"):
        info = summary["volumes"][role]
        lines.append(f"[{role}]")
        lines.append(f"  path: {info['path']}")
        lines.append(f"  shape: {tuple(info['shape'])}")
        lines.append(f"  dtype: {info['dtype']}")
        lines.append(f"  voxel spacing: {tuple(info['voxel_spacing'])}")
        lines.append("  affine:")
        for row in info["affine"]:
            lines.append("    " + " ".join(f"{value:10.6f}" for value in row))
        lines.append("")

    lines.append("Segmentation Labels")
    lines.append("-" * 80)
    for item in summary["seg_labels"]:
        lines.append(
            f"label {item['label']}: {item['name']} | "
            f"voxel_count={item['voxel_count']} | "
            f"fraction_of_volume={item['fraction_of_volume']:.6f}"
        )

    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def compute_basic_stats(values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0:
        return {"min": None, "max": None, "mean": None, "std": None}
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def compute_intensity_stats(records: dict[str, dict[str, Any]]) -> Any:
    rows: list[dict[str, Any]] = []
    for role in MODALITY_ORDER:
        volume = np.asarray(records[role]["data"], dtype=np.float32)
        all_values = volume.reshape(-1)
        nonzero_values = all_values[all_values != 0]

        all_stats = compute_basic_stats(all_values)
        nonzero_stats = compute_basic_stats(nonzero_values)

        rows.append(
            {
                "modality": role,
                "path": to_workspace_relative_string(records[role]["path"]),
                "shape": "x".join(str(v) for v in records[role]["shape"]),
                "dtype": records[role]["dtype"],
                "nonzero_voxel_ratio": float(np.count_nonzero(volume) / volume.size),
                "all_min": all_stats["min"],
                "all_max": all_stats["max"],
                "all_mean": all_stats["mean"],
                "all_std": all_stats["std"],
                "nonzero_min": nonzero_stats["min"],
                "nonzero_max": nonzero_stats["max"],
                "nonzero_mean": nonzero_stats["mean"],
                "nonzero_std": nonzero_stats["std"],
            }
        )
    return pd.DataFrame(rows)


def write_seg_labels_summary(
    seg_label_stats: list[dict[str, Any]], output_path: Path
) -> None:
    total_voxels = sum(item["voxel_count"] for item in seg_label_stats)
    tumor_voxels = sum(
        item["voxel_count"] for item in seg_label_stats if item["label"] != 0
    )

    lines = [
        "BraTS Segmentation Labels Summary",
        "=" * 80,
        "BraTS segmentation is a 3D label volume aligned with the MRI volumes.",
        f"Total voxels in the volume: {total_voxels}",
        f"Total non-background tumor voxels: {tumor_voxels}",
        "",
        "Labels",
        "-" * 80,
    ]

    for item in seg_label_stats:
        lines.append(
            f"label {item['label']}: {item['name']}\n"
            f"  voxel count: {item['voxel_count']}\n"
            f"  fraction of full volume: {item['fraction_of_volume']:.6f}"
        )

    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def compute_display_range(volume: np.ndarray) -> tuple[float, float]:
    nonzero = volume[volume != 0]
    values = nonzero if nonzero.size else volume.reshape(-1)

    if values.size == 0:
        return 0.0, 1.0

    vmin, vmax = np.percentile(values, DISPLAY_PERCENTILES)
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin = float(np.min(values))
        vmax = float(np.max(values))
    if math.isclose(float(vmin), float(vmax)):
        vmax = float(vmin) + 1.0
    return float(vmin), float(vmax)


def get_plane_length(volume: np.ndarray, plane: str) -> int:
    if plane == "axial":
        return int(volume.shape[2])
    if plane == "coronal":
        return int(volume.shape[1])
    if plane == "sagittal":
        return int(volume.shape[0])
    raise ValueError(f"Unsupported plane: {plane}")


def get_display_slice(volume: np.ndarray, plane: str, index: int) -> np.ndarray:
    if plane == "axial":
        slice_2d = volume[:, :, index]
    elif plane == "coronal":
        slice_2d = volume[:, index, :]
    elif plane == "sagittal":
        slice_2d = volume[index, :, :]
    else:
        raise ValueError(f"Unsupported plane: {plane}")

    # Transpose so that x/y are displayed in a consistent medical-image view.
    return np.asarray(slice_2d).T


def count_nonzero_per_slice(mask: np.ndarray, plane: str) -> np.ndarray:
    if plane == "axial":
        return np.count_nonzero(mask, axis=(0, 1))
    if plane == "coronal":
        return np.count_nonzero(mask, axis=(0, 2))
    if plane == "sagittal":
        return np.count_nonzero(mask, axis=(1, 2))
    raise ValueError(f"Unsupported plane: {plane}")


def finalize_indices(
    preferred: np.ndarray, total_length: int, target_count: int
) -> list[int]:
    if total_length <= 0:
        return []

    seen: set[int] = set()
    result: list[int] = []

    for raw_index in np.round(preferred).astype(int):
        index = int(max(0, min(total_length - 1, raw_index)))
        if index not in seen:
            seen.add(index)
            result.append(index)

    if len(result) < min(target_count, total_length):
        fallback = np.round(
            np.linspace(0, total_length - 1, max(target_count * 4, total_length))
        ).astype(int)
        for index in fallback:
            int_index = int(index)
            if int_index not in seen:
                seen.add(int_index)
                result.append(int_index)
            if len(result) >= min(target_count, total_length):
                break

    return result[: min(target_count, total_length)]


def select_montage_indices(
    volume: np.ndarray, plane: str, num_slices: int
) -> list[int]:
    total_length = get_plane_length(volume, plane)
    counts = count_nonzero_per_slice(volume != 0, plane)
    informative = np.where(counts > 0)[0]

    if informative.size >= num_slices:
        preferred = informative[
            np.round(np.linspace(0, informative.size - 1, num_slices)).astype(int)
        ]
    elif informative.size > 0:
        preferred = np.linspace(int(informative[0]), int(informative[-1]), num_slices)
    else:
        preferred = np.linspace(0, total_length - 1, num_slices)

    return finalize_indices(preferred, total_length, num_slices)


def select_best_seg_slice(seg_data: np.ndarray, plane: str) -> int:
    counts = count_nonzero_per_slice(seg_data > 0, plane)
    max_count = int(np.max(counts)) if counts.size else 0
    if max_count <= 0:
        raise ValueError(
            "Segmentation contains only background label 0, so tumor-focused visualizations cannot be created."
        )
    return int(np.argmax(counts))


def build_seg_colormap(seg_label_stats: list[dict[str, Any]]) -> tuple[Any, Any]:
    labels = [item["label"] for item in seg_label_stats]
    max_label = max(max(labels), max(SEG_COLORS))
    colors = [SEG_COLORS.get(index, "#cccccc") for index in range(max_label + 1)]
    cmap = ListedColormap(colors)
    bounds = np.arange(-0.5, max_label + 1.5, 1.0)
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm


def build_label_handles(
    seg_label_stats: list[dict[str, Any]], include_background: bool = False
) -> list[Any]:
    handles: list[Any] = []
    for item in seg_label_stats:
        label = item["label"]
        if label == 0 and not include_background:
            continue
        handles.append(
            Patch(
                facecolor=SEG_COLORS.get(label, "#cccccc"),
                edgecolor="black",
                label=f"{label}: {item['name']}",
            )
        )
    return handles


def save_figure(fig: Any, output_path: Path) -> None:
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_modalities_mid_slices(
    records: dict[str, dict[str, Any]], output_path: Path
) -> None:
    fig, axes = plt.subplots(
        nrows=len(MODALITY_ORDER),
        ncols=len(PLANE_ORDER),
        figsize=(14, 18),
        dpi=FIG_DPI,
    )

    for row, role in enumerate(MODALITY_ORDER):
        volume = records[role]["data"]
        vmin, vmax = compute_display_range(volume)
        for col, plane in enumerate(PLANE_ORDER):
            index = get_plane_length(volume, plane) // 2
            ax = axes[row, col]
            ax.imshow(
                get_display_slice(volume, plane, index),
                cmap="gray",
                origin="lower",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_title(
                f"{pretty_modality_name(role)} | {plane} | slice {index}",
                fontsize=10,
            )
            ax.axis("off")

    fig.suptitle(
        "BraTS 3D volume middle slices across axial / coronal / sagittal planes",
        fontsize=16,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_figure(fig, output_path)


def plot_modality_montage(role: str, record: dict[str, Any], output_path: Path) -> None:
    volume = record["data"]
    indices = select_montage_indices(volume, plane="axial", num_slices=MONTAGE_SLICES)
    columns = 4
    rows = math.ceil(len(indices) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(16, rows * 4), dpi=FIG_DPI)
    axes = np.atleast_1d(axes).ravel()
    vmin, vmax = compute_display_range(volume)

    for ax, index in zip(axes, indices):
        ax.imshow(
            get_display_slice(volume, "axial", index),
            cmap="gray",
            origin="lower",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"axial slice {index}", fontsize=10)
        ax.axis("off")

    for ax in axes[len(indices) :]:
        ax.axis("off")

    fig.suptitle(
        f"{pretty_modality_name(role)} axial montage "
        "(3D volume sampled across multiple slices, avoiding all-black regions)",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, output_path)


def plot_segmentation_montage(
    seg_record: dict[str, Any],
    seg_label_stats: list[dict[str, Any]],
    output_path: Path,
) -> None:
    seg_data = seg_record["data"]
    indices = select_montage_indices(
        seg_data, plane="axial", num_slices=SEG_MONTAGE_SLICES
    )
    columns = 4
    rows = math.ceil(len(indices) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(16, rows * 4), dpi=FIG_DPI)
    axes = np.atleast_1d(axes).ravel()
    cmap, norm = build_seg_colormap(seg_label_stats)

    for ax, index in zip(axes, indices):
        display_slice = get_display_slice(seg_data, "axial", index)
        tumor_voxels = int(np.count_nonzero(display_slice))
        ax.imshow(
            display_slice,
            cmap=cmap,
            norm=norm,
            origin="lower",
            interpolation="nearest",
        )
        ax.set_facecolor("black")
        ax.set_title(
            f"axial slice {index} | labeled voxels {tumor_voxels}", fontsize=10
        )
        ax.axis("off")

    for ax in axes[len(indices) :]:
        ax.axis("off")

    handles = build_label_handles(seg_label_stats, include_background=False)
    fig.suptitle("Segmentation axial montage with discrete tumor labels", fontsize=15)
    if handles:
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=max(1, len(handles)),
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    else:
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, output_path)


def plot_overlay_best_slices(
    base_record: dict[str, Any],
    seg_record: dict[str, Any],
    seg_label_stats: list[dict[str, Any]],
    output_path: Path,
) -> None:
    base_volume = base_record["data"]
    seg_data = seg_record["data"]
    vmin, vmax = compute_display_range(base_volume)
    cmap, norm = build_seg_colormap(seg_label_stats)
    fig, axes = plt.subplots(1, len(PLANE_ORDER), figsize=(18, 6), dpi=FIG_DPI)

    for ax, plane in zip(axes, PLANE_ORDER):
        index = select_best_seg_slice(seg_data, plane)
        base_slice = get_display_slice(base_volume, plane, index)
        seg_slice = get_display_slice(seg_data, plane, index)
        tumor_voxels = int(np.count_nonzero(seg_slice))

        ax.imshow(base_slice, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        ax.imshow(
            np.ma.masked_where(seg_slice == 0, seg_slice),
            cmap=cmap,
            norm=norm,
            origin="lower",
            alpha=0.45,
            interpolation="nearest",
        )
        ax.set_title(
            f"{plane} | slice {index} | tumor voxels {tumor_voxels}", fontsize=11
        )
        ax.axis("off")

    handles = build_label_handles(seg_label_stats, include_background=False)
    fig.suptitle(
        f"{pretty_modality_name(base_record['role'])} + segmentation overlay on tumor-dominant slices",
        fontsize=16,
    )
    if handles:
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=max(1, len(handles)),
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.08, 1, 0.92))
    else:
        fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_figure(fig, output_path)


def compute_3d_bounding_box(
    mask: np.ndarray, margin: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.argwhere(mask)
    if coordinates.size == 0:
        raise ValueError(
            "Cannot compute a tumor bounding box because segmentation has no non-zero labels."
        )

    mins = coordinates.min(axis=0)
    maxs = coordinates.max(axis=0) + 1
    mins = np.maximum(mins - margin, 0)
    maxs = np.minimum(maxs + margin, np.asarray(mask.shape))
    return mins.astype(int), maxs.astype(int)


def project_bbox_to_plane(
    mins: np.ndarray, maxs: np.ndarray, plane: str
) -> tuple[int, int, int, int]:
    if plane == "axial":
        return (
            int(mins[0]),
            int(mins[1]),
            int(maxs[0] - mins[0]),
            int(maxs[1] - mins[1]),
        )
    if plane == "coronal":
        return (
            int(mins[0]),
            int(mins[2]),
            int(maxs[0] - mins[0]),
            int(maxs[2] - mins[2]),
        )
    if plane == "sagittal":
        return (
            int(mins[1]),
            int(mins[2]),
            int(maxs[1] - mins[1]),
            int(maxs[2] - mins[2]),
        )
    raise ValueError(f"Unsupported plane: {plane}")


def crop_display_slice(
    display_slice: np.ndarray,
    mins: np.ndarray,
    maxs: np.ndarray,
    plane: str,
) -> np.ndarray:
    x0, y0, width, height = project_bbox_to_plane(mins, maxs, plane)
    return display_slice[y0 : y0 + height, x0 : x0 + width]


def plot_tumor_bbox_views(
    base_record: dict[str, Any],
    seg_record: dict[str, Any],
    seg_label_stats: list[dict[str, Any]],
    output_path: Path,
) -> None:
    base_volume = base_record["data"]
    seg_data = seg_record["data"]
    mins, maxs = compute_3d_bounding_box(seg_data > 0, margin=5)
    vmin, vmax = compute_display_range(base_volume)
    cmap, norm = build_seg_colormap(seg_label_stats)

    fig, axes = plt.subplots(2, len(PLANE_ORDER), figsize=(18, 10), dpi=FIG_DPI)

    for col, plane in enumerate(PLANE_ORDER):
        index = select_best_seg_slice(seg_data, plane)
        base_slice = get_display_slice(base_volume, plane, index)
        seg_slice = get_display_slice(seg_data, plane, index)
        masked_seg = np.ma.masked_where(seg_slice == 0, seg_slice)

        ax_full = axes[0, col]
        ax_crop = axes[1, col]

        ax_full.imshow(base_slice, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        ax_full.imshow(
            masked_seg,
            cmap=cmap,
            norm=norm,
            origin="lower",
            alpha=0.35,
            interpolation="nearest",
        )
        x0, y0, width, height = project_bbox_to_plane(mins, maxs, plane)
        ax_full.add_patch(
            Rectangle(
                (x0, y0),
                width,
                height,
                fill=False,
                edgecolor="white",
                linewidth=2.2,
                linestyle="--",
            )
        )
        ax_full.set_title(f"{plane} full slice {index} with tumor bbox", fontsize=11)
        ax_full.axis("off")

        crop_base = crop_display_slice(base_slice, mins, maxs, plane)
        crop_seg = crop_display_slice(seg_slice, mins, maxs, plane)
        ax_crop.imshow(crop_base, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        ax_crop.imshow(
            np.ma.masked_where(crop_seg == 0, crop_seg),
            cmap=cmap,
            norm=norm,
            origin="lower",
            alpha=0.45,
            interpolation="nearest",
        )
        ax_crop.set_title(f"{plane} bbox crop", fontsize=11)
        ax_crop.axis("off")

    handles = build_label_handles(seg_label_stats, include_background=False)
    fig.suptitle(
        f"Tumor 3D bounding-box views on {pretty_modality_name(base_record['role'])}",
        fontsize=16,
    )
    if handles:
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=max(1, len(handles)),
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    else:
        fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, output_path)


def plot_intensity_histograms(
    records: dict[str, dict[str, Any]], output_path: Path
) -> None:
    fig, axes = plt.subplots(
        len(MODALITY_ORDER),
        2,
        figsize=(16, 18),
        dpi=FIG_DPI,
    )

    for row, role in enumerate(MODALITY_ORDER):
        volume = np.asarray(records[role]["data"], dtype=np.float32).reshape(-1)
        nonzero = volume[volume != 0]
        color = MODALITY_COLORS[role]

        ax_all = axes[row, 0]
        ax_nonzero = axes[row, 1]

        ax_all.hist(volume, bins=120, color=color, alpha=0.85)
        ax_all.set_title(f"{pretty_modality_name(role)} | all voxels", fontsize=11)
        ax_all.set_xlabel("Intensity")
        ax_all.set_ylabel("Voxel count")

        if nonzero.size:
            ax_nonzero.hist(nonzero, bins=120, color=color, alpha=0.85)
        else:
            ax_nonzero.text(
                0.5,
                0.5,
                "No non-zero voxels",
                ha="center",
                va="center",
                transform=ax_nonzero.transAxes,
            )
        ax_nonzero.set_title(
            f"{pretty_modality_name(role)} | non-zero voxels only", fontsize=11
        )
        ax_nonzero.set_xlabel("Intensity")
        ax_nonzero.set_ylabel("Voxel count")

    fig.suptitle("Intensity distributions for BraTS modalities", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_figure(fig, output_path)


def add_bar_labels(ax: Any, bars: Any, percentages: list[float]) -> None:
    for bar, percentage in zip(bars, percentages):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{percentage:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_seg_label_distribution(
    seg_label_stats: list[dict[str, Any]], output_path: Path
) -> None:
    labels = [item["label"] for item in seg_label_stats]
    counts = [item["voxel_count"] for item in seg_label_stats]
    total = float(sum(counts))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=FIG_DPI)

    full_percentages = [count / total * 100.0 for count in counts]
    full_colors = [SEG_COLORS.get(label, "#cccccc") for label in labels]
    bars_all = axes[0].bar(
        [str(label) for label in labels], full_percentages, color=full_colors
    )
    add_bar_labels(axes[0], bars_all, full_percentages)
    axes[0].set_title("Label proportion in the full 3D volume", fontsize=12)
    axes[0].set_xlabel("Label value")
    axes[0].set_ylabel("Percentage of all voxels")

    tumor_items = [
        item
        for item in seg_label_stats
        if item["label"] != 0 and item["voxel_count"] > 0
    ]
    if tumor_items:
        tumor_total = float(sum(item["voxel_count"] for item in tumor_items))
        tumor_labels = [item["label"] for item in tumor_items]
        tumor_percentages = [
            item["voxel_count"] / tumor_total * 100.0 for item in tumor_items
        ]
        tumor_colors = [SEG_COLORS.get(label, "#cccccc") for label in tumor_labels]
        bars_tumor = axes[1].bar(
            [str(label) for label in tumor_labels],
            tumor_percentages,
            color=tumor_colors,
        )
        add_bar_labels(axes[1], bars_tumor, tumor_percentages)
        axes[1].set_title(
            "Non-background label proportion inside tumor voxels", fontsize=12
        )
        axes[1].set_xlabel("Label value")
        axes[1].set_ylabel("Percentage of tumor voxels")
    else:
        axes[1].text(
            0.5,
            0.5,
            "No non-background labels found",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )
        axes[1].set_title(
            "Non-background label proportion inside tumor voxels", fontsize=12
        )
        axes[1].set_axis_off()

    fig.suptitle("BraTS segmentation label distribution", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, output_path)


def write_readme(output_dir: Path, case_dir: Path, data_root: Path) -> None:
    content = f"""# BraTS 第一个病例可视化结果

本目录由 `visualize_first_case.py` 自动生成。

- 数据根目录：`{to_workspace_relative_string(data_root)}`
- 当前自动选中的病例目录：`{to_workspace_relative_string(case_dir)}`
- 输出目录：`{to_workspace_relative_string(output_dir)}`

## 为什么 BraTS 数据不是 jpg/png，而是 3D NIfTI

BraTS 使用的是医学影像常见的 **NIfTI** 格式，因为它保存的是完整的三维 MRI 体数据，而不是单张二维图片。

- 一份 NIfTI 文件通常对应一个 3D volume，例如 `(240, 240, 155)` 这样的体素网格。
- 每个体素不仅有强度值，还带有空间信息，例如 **voxel spacing** 和 **affine**。
- 这样我们才能在 **axial / coronal / sagittal** 三个方向上切片观察，而不是只看一张静态图。
- 分割标签 `seg` 也是一个与 MRI 完全对齐的 3D 体数据，所以可以做逐体素 overlay、3D bounding box 和标签统计。

## 模态说明

- `t1`：T1 加权 MRI，适合看基础解剖结构。
- `t1ce`：T1 对比增强 MRI，也常写作 `t1gd`，增强区通常更明显。
- `t2`：T2 加权 MRI，液体相关信号更亮。
- `flair`：抑制脑脊液后的 T2 风格图像，常更利于观察水肿区域。
- `seg`：分割标签体数据，不是普通图片，而是每个体素一个类别值。

## 文件名和术语速查

为了帮助你读懂后面的图片文件名，这里先把最常见的英文词解释一下。

- **modality / modalities**：模态。意思是“同一个病例的不同成像方式”。BraTS 里常见的是 `t1`、`t1ce`、`t2`、`flair` 四个模态。
- **slice / slices**：切片。三维 MRI volume 可以沿不同方向切成很多二维图，每一张就是一个 slice。
- **mid slices**：中间切片。通常指某个方向上索引居中的切片，用来快速看整体结构。
- **montage**：拼图、拼接图。把多张切片按网格拼在一张大图里，方便连续观察。
- **overlay**：叠加图。把分割标签半透明盖到原始 MRI 上，让你同时看到影像和标注。
- **best slices**：最有代表性的切片。这里不是随便选中间层，而是根据 seg 自动挑出肿瘤最明显的层面。
- **tumor**：肿瘤。
- **bbox / bounding box**：边界框、包围框。用一个长方体把肿瘤所在空间范围框起来。
- **views**：视图。这里通常指不同方向或不同观察尺度下的图。
- **intensity**：强度。MRI 里的亮和暗，本质上是信号强弱。
- **histogram**：直方图。用来统计某些数值出现得多不多，比如强度分布。
- **distribution**：分布。表示各个数值、各个标签或各个类别各占多少。
- **seg**：`segmentation` 的缩写，意思是分割标签。
- **label**：标签值。比如 `0/1/2/4`，每个值对应一种语义。
- **axial**：轴位、横断面。可以想成“从头顶往脚底方向一层层切”。
- **coronal**：冠状位。可以想成“从脸往后脑勺方向一层层切”。
- **sagittal**：矢状位。可以想成“从左耳往右耳方向一层层切”。
- **voxel**：体素。三维版像素，是 MRI 体数据里的最小小方块。
- **spacing**：间距。相邻 voxel 在真实世界中的距离，单位通常是毫米 mm。
- **affine**：仿射矩阵。它负责把“数组坐标”映射到“真实空间坐标”。

## 看图前先建立 6 个概念

- **volume shape**：例如 `(240, 240, 155)`，表示这不是一张图片，而是一个三维体数据。你可以把它理解成 155 张连续切片叠起来形成的一个脑部“立体盒子”。
- **voxel**：voxel 可以理解成三维版的像素。普通图片里是 pixel，MRI 体数据里是 voxel。每个 voxel 都有一个强度值。
- **voxel spacing**：表示相邻 voxel 在真实空间中的距离，例如 `(1.0, 1.0, 1.0)` mm。它告诉你这个三维体数据在现实世界里有多“密”。
- **axial / coronal / sagittal**：它们是三种看三维数据的切片方式。axial 像从上往下切，coronal 像从前往后切，sagittal 像从左往右切。
- **intensity**：MRI 里的亮和暗不是颜色，而是不同组织在这个模态下的信号强弱。不同模态里，同一块组织可能看起来完全不一样。
- **seg / overlay / bounding box**：`seg` 是三维标签体数据；overlay 是把标签叠到原始 MRI 上；bounding box 是把肿瘤所在的空间范围框出来，便于看局部。

## 建议阅读顺序

1. 先看 `case_summary.txt` 和 `seg_labels_summary.txt`
2. 再看 `modalities_mid_slices.png`
3. 再看四个模态的 montage：`t1_montage.png`、`t1ce_montage.png`、`t2_montage.png`、`flair_montage.png`
4. 再看 `seg_montage.png`
5. 再看 `flair_overlay_best_slices.png` 和 `t1ce_overlay_best_slices.png`
6. 再看 `tumor_bbox_views.png`
7. 最后看 `intensity_histograms.png` 和 `seg_label_distribution.png`

这样看最容易从“先认识数据长什么样”，逐步过渡到“理解肿瘤标签和影像之间的关系”。

## 输出文件详细说明

### `case_summary.txt`

这是最适合新手先读的文件。

- 它告诉你当前分析的是哪个病例、每个模态文件在哪里、seg 文件在哪里。
- 它列出了每个文件的 `shape`、`dtype`、`voxel spacing` 和 `affine`。
- 你可以把它当成这份 BraTS 病例的“身份证”和“体检单”。

你从这里最应该学到的是：

- BraTS 不是普通图片集，而是一组彼此对齐的 3D 体数据。
- 四个模态和一个 seg 在同一个空间里，因此后面才能做 overlay。
- seg 里的 unique labels 和体素数，决定了这个病例肿瘤各成分大概有多少。

### `case_summary.json`

这个文件和 `case_summary.txt` 内容类似，但更适合程序继续读取。

- 如果你以后想自己写代码继续分析，这个 JSON 会比纯文本更方便。
- 你可以把它理解成“给程序看的摘要”，而 `case_summary.txt` 更像“给人看的摘要”。

### `intensity_stats.csv`

这个文件回答的问题是：“每个模态整体上亮不亮？背景多不多？非零区域的信号分布大概怎样？”

- `all_*` 系列统计的是整个 3D volume，包括大量背景 0。
- `nonzero_*` 系列统计的是非零区域，更接近真实脑组织范围。
- `nonzero_voxel_ratio` 告诉你整个 volume 里真正有信号的体素占比。

新手最容易忽略的一点是：

- 医学图像里背景 0 往往特别多，所以只看全体素统计很容易被“背景”带偏。
- 因此同时看全体素统计和非零体素统计，会更容易理解为什么训练模型前常做 brain region normalization。

### `seg_labels_summary.txt`

这个文件专门解释分割标签。

- 它会告诉你每个标签值代表什么，以及各自有多少个体素。
- 如果你不清楚 BraTS 里 `1 / 2 / 4` 是什么，这个文件就是最快的参考。

看这个文件时，可以重点记住：

- `0` 通常是背景。
- `1` 通常表示坏死/非增强肿瘤核心。
- `2` 通常表示水肿。
- `4` 通常表示增强肿瘤。

### `modalities_mid_slices.png`（中文翻译：四个模态的中间切片三视图）

这个文件名可以拆开理解：

- `modalities` = 多个模态
- `mid` = 中间的
- `slices` = 切片

所以它的直译就是：“多个模态的中间切片图”。

这张图是“全局入门图”，非常重要。

- 四行分别是 `T1 / T1CE / T2 / FLAIR`。
- 三列分别是 `axial / coronal / sagittal`。
- 每个小图都是对应方向上的中间切片。

这里面最重要的专有名词是：

- **modality（模态）**：同一个病人的不同成像方式。
- **mid slice（中间切片）**：不是最有病灶代表性的切片，只是几何位置居中的切片。
- **three-view（三视图）**：就是 axial、coronal、sagittal 三个方向一起看。

你应该怎么读这张图：

- 先横向看一整行：固定一个模态，比较它在三个方向上的样子。
- 再纵向看一整列：固定一个方向，比较四个模态在同一空间位置上的差异。
- 最后留意肿瘤在三个方向里的外形有没有变化，因为同一个 3D 结构从不同方向切开后，形状经常不同。

新手常见误区是：

- 误以为中间切片就是“最重要的切片”。其实不一定，中间层可能刚好没切到肿瘤最明显的位置。
- 误以为四个模态只是“颜色不同”。其实不是，它们反映的是不同成像机制。

### `t1_montage.png`（中文翻译：T1 多切片拼接图）

这个文件名可以拆开理解：

- `t1` = T1 模态
- `montage` = 拼接图、拼图

所以它的意思是：“把很多张 T1 切片拼到一张大图里”。

这张图是 T1 模态在多个 axial 切片上的连续观察图。

- 你可以把它理解成“从头顶一路往下翻很多层切片”。
- 因为是均匀采样，所以能看到脑部结构如何随层面变化。

专有名词解释：

- **T1-weighted MRI**：T1 加权 MRI，通常更适合看基础解剖轮廓。
- **montage**：把多张切片按网格排布到同一张图中，便于一次性比较。

重点看什么：

- 正常脑结构的大体轮廓是不是清楚。
- 肿瘤有没有让某些结构受到挤压、偏移或变形。
- 和后面的 T1CE / FLAIR 对照时，T1 可以提供比较好的解剖参照。

如果你是新手，可以这样理解它的价值：

- 它不一定是最能看出肿瘤的模态。
- 但它非常适合拿来当“底图”，帮助你判断脑结构在哪里、病灶相对正常组织的位置在哪里。

### `t1ce_montage.png`（中文翻译：T1 对比增强多切片拼接图）

这个文件名可以拆开理解：

- `t1ce` = T1 contrast-enhanced，T1 对比增强
- `montage` = 多切片拼接图

有些数据集也会把 `t1ce` 写成 `t1gd`。

- `ce` 是 `contrast-enhanced` 的缩写，意思是“对比增强后”
- `gd` 常指 `gadolinium`，也就是钆对比剂

这是新手最值得重点看的图之一。

- T1CE 是加入对比增强后的 T1。
- 很多时候增强肿瘤在这里更明显，因此它常用来观察肿瘤核心附近的强化区域。

你可以重点留意：

- 哪些层面开始出现明显异常。
- 异常区域是否是局部亮起来，而不是整脑一起亮。
- 后面再对照 `t1ce_overlay_best_slices.png`，会更容易理解增强区和标签是怎么对应的。

专有名词解释：

- **contrast enhancement（对比增强）**：注射对比剂后，某些区域在图像上会更亮，常提示血脑屏障破坏或活跃病变。
- **enhancing tumor（增强肿瘤）**：BraTS 中常对应标签 `4`，往往是临床上很关注的一个区域。

### `t2_montage.png`（中文翻译：T2 多切片拼接图）

这个文件名很直观：

- `t2` = T2 模态
- `montage` = 多切片拼接图

T2 更适合看高水含量区域。

- 在这个模态里，液体相关信号通常更亮。
- 因此肿瘤周围的异常区域在 T2 里往往比较显眼。

你从这张图里可以学到：

- 病灶不仅仅是“一个小核心”，周围往往还有更大范围的异常组织环境。
- 和 T1 相比，T2 常常更强调异常范围，而不是正常解剖细节。

专有名词解释：

- **high water content（高水含量）**：比如水肿、液体增多的组织，在 T2 里往往更亮。
- **abnormal hyperintensity（异常高信号）**：就是某一块区域比周围明显更亮，常常提示异常。

### `flair_montage.png`（中文翻译：FLAIR 多切片拼接图）

这个文件名可以拆开理解：

- `flair` = FLAIR 模态
- `montage` = 多切片拼接图

FLAIR 是一个很重要的专有名词，它的全称是：

- **Fluid-Attenuated Inversion Recovery**

你可以把它粗略理解成：

- “一种把脑脊液亮度压下去的 T2 风格 MRI”

这通常是 BraTS 新手最容易看出“问题在哪里”的图。

- FLAIR 可以抑制脑脊液信号，让水肿和异常高信号更容易被看到。
- 因此它常被用来观察 edema 和病灶总体扩展范围。

你可以把它当成：

- “病灶总体地图”
- 尤其适合和 `seg_montage.png`、`flair_overlay_best_slices.png` 联合着看

专有名词解释：

- **edema（水肿）**：组织内液体增多，BraTS 中常和较大范围的异常高信号相关。
- **CSF（cerebrospinal fluid，脑脊液）**：在 FLAIR 里会被抑制，从而让病灶更突出。

### `seg_montage.png`（中文翻译：分割标签多切片拼接图）

这个文件名可以拆开理解：

- `seg` = segmentation，分割标签
- `montage` = 多切片拼接图

所以它不是 MRI 原图，而是“标签本身的拼接展示图”。

这张图不是原始 MRI，而是分割标签本身。

- 不同颜色代表不同 label。
- 这些彩色区域不是画上去的装饰，而是每个 voxel 的类别。

你应该怎么理解它：

- 如果某种颜色在连续很多层都出现，说明这个标签对应的结构在 3D 空间中是连续存在的。
- 如果某个标签只在少数几层出现，说明它在体积上比较小，或者更局限。
- 这张图能帮助你从“2D mask”思维切换到“3D label volume”思维。

专有名词解释：

- **segmentation（分割）**：给每个 voxel 指定类别。
- **label（标签）**：类别编号，比如 `0/1/2/4`。
- **label volume（标签体）**：不是一张 mask，而是完整的三维标签数据。

### `flair_overlay_best_slices.png`（中文翻译：FLAIR 上叠加分割标签的代表性切片图）

这个文件名可以拆开理解：

- `flair` = FLAIR 模态
- `overlay` = 叠加
- `best_slices` = 最有代表性的切片

直白一点说，就是：

- “在 FLAIR 上叠加分割结果，并选出最能体现肿瘤的切片”

这张图把 `seg` 叠加到了 FLAIR 上，而且不是机械地取中间层，而是自动选了更有代表性的肿瘤切片。

它为什么重要：

- FLAIR 本来就适合看病灶总体异常范围。
- 再叠加 seg 之后，你就能直接看出“标注认为的病灶”是否和“肉眼看到的异常高信号”对得上。

新手应该重点看：

- 标签是否落在异常区域上，而不是跑到正常脑组织里。
- 三个方向上，病灶边界是否都合理。
- 某些区域在 FLAIR 上看起来很大，但标签内部可能还分成不同子区域。

专有名词解释：

- **overlay（叠加）**：把彩色标签半透明覆盖到灰度 MRI 上。
- **best slice（代表性切片）**：根据 seg 非零区域自动挑选，不是随便取的。
- **legend（图例）**：图中颜色和标签名称的对应表。

### `t1ce_overlay_best_slices.png`（中文翻译：T1 对比增强图上叠加分割标签的代表性切片图）

这个文件名和上一张非常像，只是底图从 FLAIR 换成了 T1CE。

- `t1ce` = T1 对比增强
- `overlay` = 叠加
- `best_slices` = 最有代表性的切片

这张图和上一张类似，但底图换成了 T1CE。

它特别适合回答：

- 增强肿瘤到底出现在什么位置？
- 增强区域和肿瘤核心、水肿之间是什么关系？

如果你把这张图和 FLAIR overlay 对着看，会很容易理解：

- FLAIR 常更强调总体异常范围。
- T1CE 常更强调增强活跃区域。
- 同一个肿瘤，在不同模态下突出显示的成分并不相同。

专有名词解释：

- **active enhancing region（活跃增强区）**：在 T1CE 上更容易显现的区域。
- **core（核心）**：肿瘤内部较中心的部分，不一定和周围水肿一样大。

### `tumor_bbox_views.png`（中文翻译：肿瘤边界框全局与局部视图）

这个文件名可以拆开理解：

- `tumor` = 肿瘤
- `bbox` = bounding box，边界框、包围框
- `views` = 视图

所以它的意思是：

- “展示肿瘤包围框的不同观察视图”

这张图非常适合帮助新手建立“全局 + 局部”两种观察尺度。

- 上排是整幅切片，你能看到整个脑和肿瘤相对位置。
- 下排是根据 seg 算出来的 bounding box 局部放大图，你能更清楚看肿瘤附近的细节。

这张图想教你的核心概念是：

- 肿瘤既要放在全脑背景里看，也要在局部放大后看。
- 只看全图容易忽略局部结构。
- 只看局部又容易失去它在整个脑内的位置感。

专有名词解释：

- **bounding box（边界框）**：包住目标区域的最小长方体范围。
- **crop（裁剪）**：把全图里肿瘤附近那一块单独截出来放大看。
- **context（上下文）**：肿瘤周围以及它在整个脑中的位置关系。

### `intensity_histograms.png`（中文翻译：四个模态的强度直方图）

这个文件名可以拆开理解：

- `intensity` = 强度
- `histograms` = 直方图，复数形式表示这里有多张直方图

这张图是在看“信号分布”，不是在看空间位置。

- 每一行一个模态。
- 左图统计所有体素，所以背景 0 会很多。
- 右图只统计非零体素，更接近真正脑组织的信号分布。

你从这张图里最应该学到的是：

- MRI 的“亮和暗”是有分布的，不是拍脑袋决定的。
- 不同模态的分布差别很大，所以不同模态不能简单地当成普通 RGB 通道。
- 背景过多会严重影响整体统计，因此很多预处理会强调非零区域。

专有名词解释：

- **histogram（直方图）**：统计某个数值范围里有多少数据。
- **nonzero voxels（非零体素）**：不等于 0 的体素，通常更接近真正有成像信息的区域。
- **normalization（归一化）**：为了让不同病例、不同模态的数值更好比较，常会做尺度调整。

### `seg_label_distribution.png`（中文翻译：分割标签占比分布图）

这个文件名可以拆开理解：

- `seg` = segmentation，分割标签
- `label` = 标签值
- `distribution` = 分布

所以它的意思是：

- “分割标签各类别所占比例的图”

这张图是在看“标签比例”。

- 左图把背景也算进去，所以你会看到背景占绝大多数。
- 右图只看肿瘤内部的非背景标签，更适合比较 tumor 内部各成分比例。

它最适合帮助新手理解：

- 在医学分割里，类别往往极不平衡，背景远大于病灶。
- 即使只看病灶内部，不同标签之间的比例也可能差很多。
- 这会直接影响后续做分割模型时的损失函数、采样策略和评价指标。

专有名词解释：

- **class imbalance（类别不平衡）**：某些类别很多，某些类别很少。
- **ratio / proportion（占比）**：某个标签体素数占总数的比例。
- **evaluation metric（评价指标）**：比如 Dice，用来衡量分割结果好不好。

## 学习建议

- 第一步：先读 `case_summary.txt` 和 `seg_labels_summary.txt`，把数据结构和标签含义弄清楚。
- 第二步：看 `modalities_mid_slices.png`，建立模态差异和三视图差异的第一印象。
- 第三步：看四个 montage，感受 3D volume 随切片变化的过程。
- 第四步：看 `seg_montage.png`，理解标签是怎样在 3D 空间里分布的。
- 第五步：看两张 overlay，把“原始影像外观”和“分割标签位置”联系起来。
- 第六步：看 `tumor_bbox_views.png`，理解局部放大和全脑上下文之间的关系。
- 第七步：最后看 `intensity_histograms.png` 和 `seg_label_distribution.png`，建立对强度分布和类别分布的直觉。

如果你是第一次接触 BraTS，最重要的一件事不是立刻记住所有术语，而是先真正理解：

- 这是一份 **3D 体数据**
- 有 **多个模态**
- 有一个与之对齐的 **3D seg 标签**
- 不同图是在从 **不同角度** 帮你理解同一个病例
"""
    (output_dir / "README.md").write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate visualization assets for the first BraTS case in the dataset"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT_REL,
        help="BraTS archive root. Relative paths are resolved from the BraTS_final project root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR_REL,
        help="Output directory. Relative paths are resolved from the BraTS_final project root.",
    )
    return parser.parse_args()


def main(data_root: Path | None = None, output_dir: Path | None = None) -> None:
    if data_root is None or output_dir is None:
        args = parse_args()
        if data_root is None:
            data_root = resolve_workspace_path(args.data_root)
        if output_dir is None:
            output_dir = resolve_workspace_path(args.output_dir)
    else:
        data_root = Path(data_root).resolve()
        output_dir = Path(output_dir).resolve()

    ensure_output_dir(output_dir)

    log(f"Scanning data root: {data_root}")
    case_dir = select_first_case_dir(data_root)
    log(f"Selected first case directory: {case_dir}")

    case_files = discover_case_files(case_dir)
    log("Loading 3D NIfTI volumes and headers")

    records: dict[str, dict[str, Any]] = {}
    for role in (*MODALITY_ORDER, "seg"):
        records[role] = load_volume(case_files[role], role)

    validate_volume_alignment(records)

    seg_label_stats = compute_seg_label_stats(records["seg"]["data"])

    summary = build_case_summary(case_dir, case_files, records, seg_label_stats)
    log("Writing summary text, JSON, and CSV files")
    write_case_summary_text(summary, output_dir / "case_summary.txt")
    (output_dir / "case_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    compute_intensity_stats(records).to_csv(
        output_dir / "intensity_stats.csv",
        index=False,
    )
    write_seg_labels_summary(seg_label_stats, output_dir / "seg_labels_summary.txt")

    log("Creating multi-view and tumor-focused figures")
    plot_modalities_mid_slices(records, output_dir / "modalities_mid_slices.png")
    for role in MODALITY_ORDER:
        plot_modality_montage(role, records[role], output_dir / f"{role}_montage.png")
    plot_segmentation_montage(
        records["seg"], seg_label_stats, output_dir / "seg_montage.png"
    )
    plot_overlay_best_slices(
        records["flair"],
        records["seg"],
        seg_label_stats,
        output_dir / "flair_overlay_best_slices.png",
    )
    plot_overlay_best_slices(
        records["t1ce"],
        records["seg"],
        seg_label_stats,
        output_dir / "t1ce_overlay_best_slices.png",
    )
    plot_tumor_bbox_views(
        records["flair"],
        records["seg"],
        seg_label_stats,
        output_dir / "tumor_bbox_views.png",
    )
    plot_intensity_histograms(records, output_dir / "intensity_histograms.png")
    plot_seg_label_distribution(
        seg_label_stats, output_dir / "seg_label_distribution.png"
    )

    log("Writing README")
    write_readme(output_dir, case_dir, data_root)
    log(f"All results were written to: {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"Error: {exc}") from exc
