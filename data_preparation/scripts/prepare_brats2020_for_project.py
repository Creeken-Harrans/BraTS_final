from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Iterable

# Source of truth for dataset conversion in the standalone BraTS project.
# This script is maintained from the migrated nnUNet_test converter, with
# project-local path handling and safety checks kept here as the active entrypoint.

DATASET_ID = 220
DATASET_NAME = "BraTS2020"
DEFAULT_SRC_ROOT_REL = Path(
    "BraTS_final_Dataset/archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
)
DEFAULT_PROJECT_RAW_REL = Path("BraTS_final_Dataset/nnUNet_raw")


def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "data_preparation").is_dir():
            return parent
    raise RuntimeError(
        "Unable to locate BraTS_final project root from prepare_brats2020_for_project.py"
    )


def resolve_workspace_path(relative_or_absolute: Path | str) -> Path:
    candidate = Path(relative_or_absolute)
    if candidate.is_absolute():
        return candidate.resolve()
    return (find_project_root() / candidate).resolve()


def find_case_file(case_dir: Path, suffix: str) -> Path:
    """
    支持 .nii.gz 和 .nii 两种输入。
    优先找 .nii.gz；找不到再找 .nii。
    """
    stem = case_dir.name
    candidates = [
        case_dir / f"{stem}_{suffix}.nii.gz",
        case_dir / f"{stem}_{suffix}.nii",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Missing file for case={stem}, suffix={suffix}")


def write_image_as_niigz(src: Path, dst: Path) -> None:
    import SimpleITK as sitk

    img = sitk.ReadImage(str(src))
    sitk.WriteImage(img, str(dst))


def convert_brats_seg_to_project(src: Path, dst: Path) -> None:
    import numpy as np
    import SimpleITK as sitk

    """
    BraTS 原标签:
        0 = background
        1 = NCR/NET
        2 = ED
        4 = ET

    按官方 BraTS nnUNet 脚本的思路，转换成连续整数:
        0 -> 0
        2 -> 1
        1 -> 2
        4 -> 3
    """
    img = sitk.ReadImage(str(src))
    arr = sitk.GetArrayFromImage(img)

    uniques = set(np.unique(arr).tolist())
    allowed = {0, 1, 2, 4}
    if not uniques.issubset(allowed):
        raise RuntimeError(
            f"Unexpected labels in {src}: got {sorted(uniques)}, allowed {sorted(allowed)}"
        )

    new_arr = np.zeros_like(arr, dtype=np.uint8)
    new_arr[arr == 2] = 1
    new_arr[arr == 1] = 2
    new_arr[arr == 4] = 3

    out = sitk.GetImageFromArray(new_arr)
    out.CopyInformation(img)
    sitk.WriteImage(out, str(dst))


def validate_case_geometry(case_id: str, files: dict[str, Path]) -> None:
    import SimpleITK as sitk

    metadata: dict[
        str,
        tuple[tuple[int, ...], tuple[float, ...], tuple[float, ...], tuple[float, ...]],
    ] = {}
    for role, path in files.items():
        image = sitk.ReadImage(str(path))
        metadata[role] = (
            tuple(int(v) for v in image.GetSize()),
            tuple(float(v) for v in image.GetSpacing()),
            tuple(float(v) for v in image.GetOrigin()),
            tuple(float(v) for v in image.GetDirection()),
        )

    reference_role = next(iter(metadata))
    reference = metadata[reference_role]
    mismatches: list[str] = []
    for role, info in metadata.items():
        if info != reference:
            mismatches.append(
                f"{case_id} {role} geometry mismatch against {reference_role}: {info} != {reference}"
            )

    if mismatches:
        raise RuntimeError("\n".join(mismatches))


def collect_cases(src_root: Path) -> list[tuple[str, dict[str, Path]]]:
    required = ["t1", "t1ce", "t2", "flair", "seg"]
    valid_cases: list[tuple[str, dict[str, Path]]] = []

    for d in sorted(src_root.iterdir()):
        if not d.is_dir():
            continue
        try:
            files = {k: find_case_file(d, k) for k in required}
            validate_case_geometry(d.name, files)
            valid_cases.append((d.name, files))
        except FileNotFoundError:
            # 跳过不符合 BraTS case 结构的目录
            continue

    if not valid_cases:
        raise RuntimeError(f"No valid BraTS case folders found under: {src_root}")

    return valid_cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert BraTS2020 raw data into Dataset220_BraTS2020 for this project"
    )
    parser.add_argument(
        "--src-root",
        type=Path,
        default=DEFAULT_SRC_ROOT_REL,
        help="BraTS2020 raw source root. Relative paths are resolved from the BraTS_final project root.",
    )
    parser.add_argument(
        "--project-raw",
        type=Path,
        default=None,
        help="Target raw dataset root. Relative paths are resolved from the BraTS_final project root. "
        "Defaults to PROJECT_RAW or BraTS_final_Dataset/nnUNet_raw.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete the existing Dataset220_BraTS2020 directory and rebuild it from scratch.",
    )
    return parser.parse_args()


def _existing_dataset_looks_complete(
    out_base: Path,
    expected_modalities: int,
) -> bool:
    dataset_json_path = out_base / "dataset.json"
    images_tr = out_base / "imagesTr"
    labels_tr = out_base / "labelsTr"
    if (
        not dataset_json_path.is_file()
        or not images_tr.is_dir()
        or not labels_tr.is_dir()
    ):
        return False

    image_files = list(images_tr.glob("*.nii.gz"))
    label_files = list(labels_tr.glob("*.nii.gz"))
    try:
        dataset_json = json.loads(dataset_json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    expected_case_count = int(dataset_json.get("numTraining", -1))
    expected_image_files = expected_case_count * expected_modalities
    if expected_case_count <= 0:
        return False
    return (
        len(image_files) == expected_image_files
        and len(label_files) == expected_case_count
    )


def main(
    src_root: Path | None = None, project_raw: Path | None = None, force: bool = False
) -> None:
    if src_root is None or project_raw is None:
        args = parse_args()
        if src_root is None:
            src_root = resolve_workspace_path(args.src_root)
        if project_raw is None:
            env_project_raw = os.environ.get("PROJECT_RAW", "").strip()
            if args.project_raw is not None:
                project_raw = resolve_workspace_path(args.project_raw)
            elif env_project_raw:
                project_raw = Path(env_project_raw).resolve()
            else:
                project_raw = resolve_workspace_path(DEFAULT_PROJECT_RAW_REL)
        force = force or args.force
    else:
        src_root = Path(src_root).resolve()
        project_raw = Path(project_raw).resolve()

    out_base = project_raw / f"Dataset{DATASET_ID:03d}_{DATASET_NAME}"
    imagesTr = out_base / "imagesTr"
    labelsTr = out_base / "labelsTr"
    expected_modalities = 4

    if out_base.exists() and any(out_base.iterdir()) and not force:
        if _existing_dataset_looks_complete(out_base, expected_modalities):
            print(f"[INFO] Reusing existing dataset directory: {out_base}")
            print(f"[OK] dataset.json: {out_base / 'dataset.json'}")
            return

    cases = collect_cases(src_root)
    print(f"[INFO] Found {len(cases)} valid BraTS cases")

    # 按官方 BraTS v2 转换脚本的通道顺序:
    # 0000=T1, 0001=T1ce, 0002=T2, 0003=Flair
    channel_map = [
        ("t1", "0000"),
        ("t1ce", "0001"),
        ("t2", "0002"),
        ("flair", "0003"),
    ]
    expected_modalities = len(channel_map)

    if out_base.exists() and any(out_base.iterdir()):
        if force:
            print(
                f"[INFO] Removing existing dataset directory before rebuild: {out_base}"
            )
            shutil.rmtree(out_base)
        else:
            raise RuntimeError(
                f"Target dataset folder already exists but is incomplete or inconsistent: {out_base}\n"
                "Use --force to rebuild it from scratch."
            )

    imagesTr.mkdir(parents=True, exist_ok=True)
    labelsTr.mkdir(parents=True, exist_ok=True)

    for case_id, files in cases:
        for modal_name, channel_id in channel_map:
            dst = imagesTr / f"{case_id}_{channel_id}.nii.gz"
            write_image_as_niigz(files[modal_name], dst)

        seg_dst = labelsTr / f"{case_id}.nii.gz"
        convert_brats_seg_to_project(files["seg"], seg_dst)

    # 注意：region-based training 中 labels 的顺序不能乱。
    # json.dump 不要 sort_keys=True。
    dataset_json = {
        "channel_names": {"0": "T1", "1": "T1ce", "2": "T2", "3": "Flair"},
        "labels": {
            "background": 0,
            "whole_tumor": [1, 2, 3],
            "tumor_core": [2, 3],
            "enhancing_tumor": [3],
        },
        "numTraining": len(cases),
        "file_ending": ".nii.gz",
        "regions_class_order": [1, 2, 3],
        "description": "BraTS2020 converted for the standalone BraTS project region-based training",
    }

    with open(out_base / "dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset_json, f, indent=2, ensure_ascii=False)

    print(f"[OK] Converted dataset written to: {out_base}")
    print(f"[OK] imagesTr: {imagesTr}")
    print(f"[OK] labelsTr: {labelsTr}")
    print(f"[OK] dataset.json: {out_base / 'dataset.json'}")


if __name__ == "__main__":
    main()
