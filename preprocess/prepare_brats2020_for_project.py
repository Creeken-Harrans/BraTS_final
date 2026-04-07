from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

DATASET_ID = 220
DATASET_NAME = "BraTS2020"
DEFAULT_SRC_ROOT_REL = Path(
    "BraTS_final_Dataset/archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
)
DEFAULT_PROJECT_RAW_REL = Path("BraTS_final_Dataset/nnUNet_raw")
DEFAULT_PROJECT_PREPROCESSED_REL = Path("BraTS_final_Dataset/nnUNet_preprocessed")
DEFAULT_PREPROCESSED_IDENTIFIER = "ProjectPlans_3d_fullres"
DEFAULT_NUM_SPLITS = 5
DEFAULT_SPLIT_SEED = 12345
MAX_CLASS_LOCATIONS_PER_REGION = 10000
PREPROCESSED_METADATA_FILES = ("dataset.json", "splits_final.json")
PREPROCESSED_REQUIRED_DIRS = (DEFAULT_PREPROCESSED_IDENTIFIER, "gt_segmentations")
PREPROCESSED_LEGACY_FILES = (
    "ProjectPlans.json",
    "reference_ProjectPlans.json",
    "nnUNetPlans.json",
    "dataset_fingerprint.json",
    "ProjectPlans_2d",
    "nnUNetPlans_2d",
    "nnUNetPlans_3d_fullres",
)
REGION_DEFINITIONS: dict[str, tuple[int, ...]] = {
    "whole_tumor": (1, 2, 3),
    "tumor_core": (2, 3),
    "enhancing_tumor": (3,),
}


def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "preprocess").is_dir():
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
    stem = case_dir.name
    candidates = [
        case_dir / f"{stem}_{suffix}.nii.gz",
        case_dir / f"{stem}_{suffix}.nii",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing file for case={stem}, suffix={suffix}")


def strip_nii_gz_suffix(path: Path) -> str:
    if path.name.endswith(".nii.gz"):
        return path.name[:-7]
    return path.stem


def write_image_as_niigz(src: Path, dst: Path) -> None:
    import SimpleITK as sitk

    image = sitk.ReadImage(str(src))
    sitk.WriteImage(image, str(dst))


def convert_brats_seg_to_project(src: Path, dst: Path) -> None:
    import numpy as np
    import SimpleITK as sitk

    image = sitk.ReadImage(str(src))
    array = sitk.GetArrayFromImage(image)

    uniques = set(np.unique(array).tolist())
    allowed = {0, 1, 2, 4}
    if not uniques.issubset(allowed):
        raise RuntimeError(
            f"Unexpected labels in {src}: got {sorted(uniques)}, allowed {sorted(allowed)}"
        )

    converted = np.zeros_like(array, dtype=np.uint8)
    converted[array == 2] = 1
    converted[array == 1] = 2
    converted[array == 4] = 3

    output = sitk.GetImageFromArray(converted)
    output.CopyInformation(image)
    sitk.WriteImage(output, str(dst))


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
    for case_dir in sorted(src_root.iterdir()):
        if not case_dir.is_dir():
            continue
        try:
            files = {key: find_case_file(case_dir, key) for key in required}
            validate_case_geometry(case_dir.name, files)
            valid_cases.append((case_dir.name, files))
        except FileNotFoundError:
            continue
    if not valid_cases:
        raise RuntimeError(f"No valid BraTS case folders found under: {src_root}")
    return valid_cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert BraTS2020 archive data into the project raw dataset and "
            "training-ready preprocessed dataset."
        )
    )
    parser.add_argument("--src-root", type=Path, default=DEFAULT_SRC_ROOT_REL)
    parser.add_argument("--project-raw", type=Path, default=None)
    parser.add_argument("--project-preprocessed", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-preprocessing", action="store_true")
    parser.add_argument("--force-preprocessing", action="store_true")
    parser.add_argument("--num-splits", type=int, default=DEFAULT_NUM_SPLITS)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    return parser.parse_args()


def _existing_dataset_looks_complete(out_base: Path, expected_modalities: int) -> bool:
    dataset_json_path = out_base / "dataset.json"
    images_tr = out_base / "imagesTr"
    labels_tr = out_base / "labelsTr"
    if not dataset_json_path.is_file() or not images_tr.is_dir() or not labels_tr.is_dir():
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
    return len(image_files) == expected_image_files and len(label_files) == expected_case_count


def _managed_preprocessed_paths(preprocessed_root: Path) -> tuple[Path, ...]:
    return tuple(
        preprocessed_root / name
        for name in (
            *PREPROCESSED_METADATA_FILES,
            *PREPROCESSED_REQUIRED_DIRS,
            *PREPROCESSED_LEGACY_FILES,
        )
    )


def _existing_preprocessed_looks_complete(
    preprocessed_root: Path,
    *,
    expected_case_ids: list[str],
) -> bool:
    dataset_json_path = preprocessed_root / "dataset.json"
    splits_json_path = preprocessed_root / "splits_final.json"
    training_dir = preprocessed_root / DEFAULT_PREPROCESSED_IDENTIFIER
    gt_dir = preprocessed_root / "gt_segmentations"
    if (
        not dataset_json_path.is_file()
        or not splits_json_path.is_file()
        or not training_dir.is_dir()
        or not gt_dir.is_dir()
    ):
        return False
    try:
        dataset_json = json.loads(dataset_json_path.read_text(encoding="utf-8"))
        splits = json.loads(splits_json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if int(dataset_json.get("numTraining", -1)) != len(expected_case_ids):
        return False
    if not isinstance(splits, list) or len(splits) == 0:
        return False
    expected = set(expected_case_ids)
    npz_files = {path.stem for path in training_dir.glob("*.npz")}
    pkl_files = {path.stem for path in training_dir.glob("*.pkl")}
    gt_files = {strip_nii_gz_suffix(path) for path in gt_dir.glob("*.nii.gz")}
    return npz_files == expected and pkl_files == expected and gt_files == expected


def _stable_seed_from_case_id(case_id: str, base_seed: int) -> int:
    digest = hashlib.sha256(f"{base_seed}:{case_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**32)


def _compute_bbox_from_mask(mask: "np.ndarray") -> list[list[int]]:
    import numpy as np

    if not np.any(mask):
        return [[0, int(size)] for size in mask.shape]
    bbox: list[list[int]] = []
    for axis_indices in np.where(mask):
        bbox.append([int(axis_indices.min()), int(axis_indices.max()) + 1])
    return bbox


def _crop_spatial(array: "np.ndarray", bbox: list[list[int]]) -> "np.ndarray":
    slices = tuple(slice(int(start), int(end)) for start, end in bbox)
    if array.ndim == 4:
        return array[(slice(None), *slices)]
    return array[slices]


def _normalize_cropped_modalities(data: "np.ndarray") -> "np.ndarray":
    import numpy as np

    normalized = np.zeros_like(data, dtype=np.float32)
    for channel in range(data.shape[0]):
        channel_data = data[channel].astype(np.float32, copy=False)
        foreground = channel_data != 0
        if not np.any(foreground):
            continue
        values = channel_data[foreground]
        mean = float(values.mean())
        std = float(values.std())
        if std < 1e-8:
            normalized[channel][foreground] = channel_data[foreground] - mean
        else:
            normalized[channel][foreground] = (channel_data[foreground] - mean) / std
    return normalized


def _build_class_locations(
    segmentation: "np.ndarray",
    *,
    case_id: str,
    max_locations_per_region: int,
    base_seed: int,
) -> dict[tuple[int, ...], "np.ndarray"]:
    import numpy as np

    rng = np.random.default_rng(_stable_seed_from_case_id(case_id, base_seed))
    result: dict[tuple[int, ...], np.ndarray] = {}
    for region in REGION_DEFINITIONS.values():
        coords = np.argwhere(np.isin(segmentation, np.asarray(region)))
        if coords.size == 0:
            result[region] = np.empty((0, segmentation.ndim + 1), dtype=np.int16)
            continue
        if coords.shape[0] > max_locations_per_region:
            indices = rng.choice(coords.shape[0], size=max_locations_per_region, replace=False)
            coords = coords[indices]
        prefix = np.zeros((coords.shape[0], 1), dtype=np.int16)
        result[region] = np.concatenate((prefix, coords.astype(np.int16, copy=False)), axis=1)
    return result


def _write_preprocessed_metadata(
    preprocessed_root: Path,
    *,
    dataset_json: dict,
    splits: list[dict[str, list[str]]],
) -> None:
    preprocessed_root.mkdir(parents=True, exist_ok=True)
    (preprocessed_root / "dataset.json").write_text(
        json.dumps(dataset_json, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (preprocessed_root / "splits_final.json").write_text(
        json.dumps(splits, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _generate_splits(case_ids: list[str], num_splits: int, split_seed: int) -> list[dict[str, list[str]]]:
    import numpy as np

    if num_splits < 2:
        raise RuntimeError("--num-splits must be at least 2.")
    if num_splits > len(case_ids):
        raise RuntimeError(
            f"--num-splits={num_splits} is larger than the number of cases ({len(case_ids)})."
        )
    ordered_case_ids = sorted(case_ids)
    rng = np.random.default_rng(split_seed)
    shuffled_indices = np.arange(len(ordered_case_ids))
    rng.shuffle(shuffled_indices)
    fold_indices = np.array_split(shuffled_indices, num_splits)
    splits: list[dict[str, list[str]]] = []
    for val_indices in fold_indices:
        val_set = {ordered_case_ids[int(index)] for index in val_indices.tolist()}
        splits.append(
            {
                "train": [case_id for case_id in ordered_case_ids if case_id not in val_set],
                "val": [case_id for case_id in ordered_case_ids if case_id in val_set],
            }
        )
    return splits


def build_training_ready_preprocessed_dataset(
    *,
    raw_dataset_dir: Path,
    preprocessed_root: Path,
    force: bool,
    num_splits: int,
    split_seed: int,
) -> None:
    import numpy as np
    import SimpleITK as sitk

    from training.src.data.dataset import NpzCaseDataset

    images_tr = raw_dataset_dir / "imagesTr"
    labels_tr = raw_dataset_dir / "labelsTr"
    dataset_json = json.loads((raw_dataset_dir / "dataset.json").read_text(encoding="utf-8"))
    case_ids = sorted(strip_nii_gz_suffix(path) for path in labels_tr.glob("*.nii.gz"))

    if _existing_preprocessed_looks_complete(
        preprocessed_root, expected_case_ids=case_ids
    ) and not force:
        print(f"[INFO] Reusing existing preprocessed dataset: {preprocessed_root}")
        print(f"[OK] training_cases: {preprocessed_root / DEFAULT_PREPROCESSED_IDENTIFIER}")
        print(f"[OK] gt_segmentations: {preprocessed_root / 'gt_segmentations'}")
        return

    if force:
        for path in _managed_preprocessed_paths(preprocessed_root):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            elif path.exists():
                path.unlink()

    training_dir = preprocessed_root / DEFAULT_PREPROCESSED_IDENTIFIER
    gt_dir = preprocessed_root / "gt_segmentations"
    training_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    for case_id in case_ids:
        modality_arrays: list[np.ndarray] = []
        first_image = None
        for channel in range(4):
            image = sitk.ReadImage(str(images_tr / f"{case_id}_{channel:04d}.nii.gz"))
            if first_image is None:
                first_image = image
            modality_arrays.append(sitk.GetArrayFromImage(image).astype(np.float32, copy=False))
        if first_image is None:
            raise RuntimeError(f"Unable to read modalities for case: {case_id}")

        label_path = labels_tr / f"{case_id}.nii.gz"
        label_image = sitk.ReadImage(str(label_path))
        segmentation = sitk.GetArrayFromImage(label_image).astype(np.int16, copy=False)

        data = np.stack(modality_arrays, axis=0)
        foreground_mask = np.any(data != 0, axis=0)
        bbox = _compute_bbox_from_mask(foreground_mask)
        cropped_mask = _crop_spatial(foreground_mask.astype(np.uint8, copy=False), bbox).astype(bool, copy=False)
        cropped_data = _crop_spatial(data, bbox).astype(np.float32, copy=False)
        cropped_seg = _crop_spatial(segmentation, bbox).astype(np.int16, copy=False)

        stored_seg = cropped_seg.copy()
        stored_seg[~cropped_mask] = -1
        properties = {
            "bbox_used_for_cropping": bbox,
            "shape_before_cropping": tuple(int(v) for v in data.shape[1:]),
            "shape_after_cropping_and_before_resampling": tuple(int(v) for v in cropped_data.shape[1:]),
            "spacing": [float(v) for v in first_image.GetSpacing()[::-1]],
            "sitk_stuff": {
                "spacing": [float(v) for v in label_image.GetSpacing()],
                "origin": [float(v) for v in label_image.GetOrigin()],
                "direction": [float(v) for v in label_image.GetDirection()],
            },
            "class_locations": _build_class_locations(
                cropped_seg,
                case_id=case_id,
                max_locations_per_region=MAX_CLASS_LOCATIONS_PER_REGION,
                base_seed=split_seed,
            ),
        }
        NpzCaseDataset.save_case(
            _normalize_cropped_modalities(cropped_data),
            stored_seg[None].astype(np.int16, copy=False),
            properties,
            str(training_dir / case_id),
        )
        shutil.copy2(label_path, gt_dir / label_path.name)

    dataset_json["description"] = (
        "BraTS2020 converted and cropped for the standalone BraTS project training pipeline"
    )
    _write_preprocessed_metadata(
        preprocessed_root,
        dataset_json=dataset_json,
        splits=_generate_splits(case_ids, num_splits=num_splits, split_seed=split_seed),
    )
    print(f"[OK] Preprocessed dataset root: {preprocessed_root}")
    print(f"[OK] training_cases: {training_dir}")
    print(f"[OK] gt_segmentations: {gt_dir}")
    print(f"[OK] splits_final.json: {preprocessed_root / 'splits_final.json'}")


def main() -> None:
    args = parse_args()
    src_root = resolve_workspace_path(args.src_root)

    env_project_raw = os.environ.get("PROJECT_RAW", "").strip()
    if args.project_raw is not None:
        project_raw = resolve_workspace_path(args.project_raw)
    elif env_project_raw:
        project_raw = Path(env_project_raw).resolve()
    else:
        project_raw = resolve_workspace_path(DEFAULT_PROJECT_RAW_REL)

    env_project_preprocessed = os.environ.get("PROJECT_PREPROCESSED", "").strip()
    if args.project_preprocessed is not None:
        project_preprocessed = resolve_workspace_path(args.project_preprocessed)
    elif env_project_preprocessed:
        project_preprocessed = Path(env_project_preprocessed).resolve()
    else:
        project_preprocessed = resolve_workspace_path(DEFAULT_PROJECT_PREPROCESSED_REL)

    out_base = project_raw / f"Dataset{DATASET_ID:03d}_{DATASET_NAME}"
    images_tr = out_base / "imagesTr"
    labels_tr = out_base / "labelsTr"
    expected_modalities = 4

    if out_base.exists() and any(out_base.iterdir()) and not args.force:
        if _existing_dataset_looks_complete(out_base, expected_modalities):
            print(f"[INFO] Reusing existing dataset directory: {out_base}")
            print(f"[OK] dataset.json: {out_base / 'dataset.json'}")
            if args.skip_preprocessing:
                return
            build_training_ready_preprocessed_dataset(
                raw_dataset_dir=out_base,
                preprocessed_root=project_preprocessed,
                force=args.force or args.force_preprocessing,
                num_splits=int(args.num_splits),
                split_seed=int(args.split_seed),
            )
            return
        print(f"[WARN] Incomplete raw dataset detected, rebuilding from scratch: {out_base}")
        shutil.rmtree(out_base)

    cases = collect_cases(src_root)
    print(f"[INFO] Found {len(cases)} valid BraTS cases")
    channel_map = [("t1", "0000"), ("t1ce", "0001"), ("t2", "0002"), ("flair", "0003")]

    if out_base.exists() and any(out_base.iterdir()):
        shutil.rmtree(out_base)

    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    for case_id, files in cases:
        for modal_name, channel_id in channel_map:
            write_image_as_niigz(files[modal_name], images_tr / f"{case_id}_{channel_id}.nii.gz")
        convert_brats_seg_to_project(files["seg"], labels_tr / f"{case_id}.nii.gz")

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
    (out_base / "dataset.json").write_text(
        json.dumps(dataset_json, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"[OK] Converted dataset written to: {out_base}")
    print(f"[OK] imagesTr: {images_tr}")
    print(f"[OK] labelsTr: {labels_tr}")
    print(f"[OK] dataset.json: {out_base / 'dataset.json'}")

    if args.skip_preprocessing:
        return

    build_training_ready_preprocessed_dataset(
        raw_dataset_dir=out_base,
        preprocessed_root=project_preprocessed,
        force=args.force or args.force_preprocessing,
        num_splits=int(args.num_splits),
        split_seed=int(args.split_seed),
    )


if __name__ == "__main__":
    main()
