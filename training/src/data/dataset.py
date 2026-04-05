from __future__ import annotations

import math
import multiprocessing
import os
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Optional

import blosc2
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import (
    isfile,
    load_pickle,
    subfiles,
    write_pickle,
)


def _convert_npz_case_to_npy(args: tuple[str, bool, bool, bool]) -> None:
    npz_file, unpack_segmentation, overwrite_existing, verify = args
    data_npy = npz_file[:-3] + "npy"
    seg_npy = npz_file[:-4] + "_seg.npy"
    npz_content = None
    if overwrite_existing or not isfile(data_npy):
        npz_content = np.load(npz_file) if npz_content is None else npz_content
        np.save(data_npy, npz_content["data"])
    if unpack_segmentation and (overwrite_existing or not isfile(seg_npy)):
        npz_content = np.load(npz_file) if npz_content is None else npz_content
        np.save(seg_npy, npz_content["seg"])
    if verify:
        np.load(data_npy, mmap_mode="r")
        if isfile(seg_npy):
            np.load(seg_npy, mmap_mode="r")


def unpack_preprocessed_cases(
    folder: str,
    unpack_segmentation: bool = True,
    overwrite_existing: bool = False,
    num_processes: int = 8,
    verify: bool = False,
) -> None:
    npz_files = subfiles(folder, True, None, ".npz", True)
    work_items = [
        (npz_file, unpack_segmentation, overwrite_existing, verify)
        for npz_file in npz_files
    ]
    with multiprocessing.get_context("spawn").Pool(num_processes) as pool:
        pool.map(_convert_npz_case_to_npy, work_items)


class PreprocessedCaseDataset(ABC):
    def __init__(
        self,
        folder: str,
        identifiers: Optional[list[str]] = None,
        folder_with_segs_from_previous_stage: Optional[str] = None,
    ) -> None:
        self.source_folder = folder
        self.folder_with_segs_from_previous_stage = folder_with_segs_from_previous_stage
        self.identifiers = identifiers or self.get_identifiers(folder)
        self.identifiers.sort()

    def __getitem__(self, identifier: str):
        return self.load_case(identifier)

    @abstractmethod
    def load_case(self, identifier: str):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def save_case(data: np.ndarray, seg: np.ndarray, properties: dict, output_filename_truncated: str):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_identifiers(folder: str) -> list[str]:
        raise NotImplementedError


class NpzCaseDataset(PreprocessedCaseDataset):
    def load_case(self, identifier: str):
        data_npy_file = os.path.join(self.source_folder, identifier + ".npy")
        if not isfile(data_npy_file):
            data = np.load(os.path.join(self.source_folder, identifier + ".npz"))["data"].copy()
        else:
            data = np.load(data_npy_file)

        seg_npy_file = os.path.join(self.source_folder, identifier + "_seg.npy")
        if not isfile(seg_npy_file):
            seg = np.load(os.path.join(self.source_folder, identifier + ".npz"))["seg"].copy()
        else:
            seg = np.load(seg_npy_file)

        if self.folder_with_segs_from_previous_stage is not None:
            prev_seg_npy_file = os.path.join(self.folder_with_segs_from_previous_stage, identifier + ".npy")
            if isfile(prev_seg_npy_file):
                seg_prev = np.load(prev_seg_npy_file)
            else:
                seg_prev = np.load(
                    os.path.join(self.folder_with_segs_from_previous_stage, identifier + ".npz")
                )["seg"].copy()
        else:
            seg_prev = None

        properties = load_pickle(os.path.join(self.source_folder, identifier + ".pkl"))
        return data, seg, seg_prev, properties

    @staticmethod
    def save_case(data: np.ndarray, seg: np.ndarray, properties: dict, output_filename_truncated: str):
        np.savez_compressed(output_filename_truncated + ".npz", data=data, seg=seg)
        write_pickle(properties, output_filename_truncated + ".pkl")

    @staticmethod
    def get_identifiers(folder: str) -> list[str]:
        return [name[:-4] for name in os.listdir(folder) if name.endswith("npz")]


class BloscCaseDataset(PreprocessedCaseDataset):
    def __init__(self, folder: str, identifiers: Optional[list[str]] = None, folder_with_segs_from_previous_stage: Optional[str] = None) -> None:
        super().__init__(folder, identifiers, folder_with_segs_from_previous_stage)
        blosc2.set_nthreads(1)

    def load_case(self, identifier: str):
        dparams = {"nthreads": 1}
        mmap_kwargs = {} if os.name == "nt" else {"mmap_mode": "r"}
        data = blosc2.open(urlpath=os.path.join(self.source_folder, identifier + ".b2nd"), mode="r", dparams=dparams, **mmap_kwargs)
        seg = blosc2.open(urlpath=os.path.join(self.source_folder, identifier + "_seg.b2nd"), mode="r", dparams=dparams, **mmap_kwargs)
        if self.folder_with_segs_from_previous_stage is not None:
            seg_prev = blosc2.open(
                urlpath=os.path.join(self.folder_with_segs_from_previous_stage, identifier + ".b2nd"),
                mode="r",
                dparams=dparams,
                **mmap_kwargs,
            )
        else:
            seg_prev = None
        properties = load_pickle(os.path.join(self.source_folder, identifier + ".pkl"))
        return data, seg, seg_prev, properties

    @staticmethod
    def save_case(
        data: np.ndarray,
        seg: np.ndarray,
        properties: dict,
        output_filename_truncated: str,
        chunks=None,
        blocks=None,
        chunks_seg=None,
        blocks_seg=None,
        clevel: int = 8,
        codec=blosc2.Codec.ZSTD,
    ):
        blosc2.set_nthreads(1)
        chunks_seg = chunks if chunks_seg is None else chunks_seg
        blocks_seg = blocks if blocks_seg is None else blocks_seg
        cparams = {"codec": codec, "clevel": clevel}
        for path in (
            output_filename_truncated + ".b2nd",
            output_filename_truncated + "_seg.b2nd",
            output_filename_truncated + ".pkl",
        ):
            if isfile(path):
                os.remove(path)
        blosc2.asarray(np.ascontiguousarray(data), urlpath=output_filename_truncated + ".b2nd", chunks=chunks, blocks=blocks, cparams=cparams)
        blosc2.asarray(np.ascontiguousarray(seg), urlpath=output_filename_truncated + "_seg.b2nd", chunks=chunks_seg, blocks=blocks_seg, cparams=cparams)
        write_pickle(properties, output_filename_truncated + ".pkl")

    @staticmethod
    def get_identifiers(folder: str) -> list[str]:
        return [
            name[:-5]
            for name in os.listdir(folder)
            if name.endswith(".b2nd") and not name.endswith("_seg.b2nd")
        ]

    @staticmethod
    def comp_blosc2_params(
        image_size: tuple[int, int, int, int],
        patch_size: tuple[int, int] | tuple[int, int, int],
        bytes_per_pixel: int = 4,
        l1_cache_size_per_core_in_bytes: int = 32768,
        l3_cache_size_per_core_in_bytes: int = 1441792,
        safety_factor: float = 0.8,
    ):
        num_channels = image_size[0]
        if len(patch_size) == 2:
            patch_size = (1, *patch_size)
        patch_size = np.array(patch_size)
        block_size = np.array((num_channels, *[2 ** (max(0, math.ceil(math.log2(v)))) for v in patch_size]))

        estimated_nbytes_block = np.prod(block_size) * bytes_per_pixel
        while estimated_nbytes_block > (l1_cache_size_per_core_in_bytes * safety_factor):
            axis_order = np.argsort(block_size[1:] / patch_size)[::-1]
            idx = 0
            picked_axis = axis_order[idx]
            while block_size[picked_axis + 1] == 1:
                idx += 1
                picked_axis = axis_order[idx]
            block_size[picked_axis + 1] = 2 ** max(0, math.floor(math.log2(block_size[picked_axis + 1] - 1)))
            block_size[picked_axis + 1] = min(block_size[picked_axis + 1], image_size[picked_axis + 1])
            estimated_nbytes_block = np.prod(block_size) * bytes_per_pixel
        block_size = np.array([min(i, j) for i, j in zip(image_size, block_size)])

        chunk_size = deepcopy(block_size)
        estimated_nbytes_chunk = np.prod(chunk_size) * bytes_per_pixel
        while estimated_nbytes_chunk < (l3_cache_size_per_core_in_bytes * safety_factor):
            if patch_size[0] == 1 and all(i == j for i, j in zip(chunk_size[2:], image_size[2:])):
                break
            if all(i == j for i, j in zip(chunk_size, image_size)):
                break
            axis_order = np.argsort(chunk_size[1:] / block_size[1:])
            idx = 0
            picked_axis = axis_order[idx]
            while chunk_size[picked_axis + 1] == image_size[picked_axis + 1] or patch_size[picked_axis] == 1:
                idx += 1
                picked_axis = axis_order[idx]
            chunk_size[picked_axis + 1] += block_size[picked_axis + 1]
            chunk_size[picked_axis + 1] = min(chunk_size[picked_axis + 1], image_size[picked_axis + 1])
            estimated_nbytes_chunk = np.prod(chunk_size) * bytes_per_pixel
            if np.mean([i / j for i, j in zip(chunk_size[1:], patch_size)]) > 1.5:
                chunk_size[picked_axis + 1] -= block_size[picked_axis + 1]
                break
        return tuple(min(i, j) for i, j in zip(image_size, chunk_size)), tuple(block_size)


def infer_preprocessed_dataset_class(folder: str):
    file_endings = {Path(path).suffix.lstrip(".") for path in subfiles(folder, join=False)}
    file_endings -= {"pkl", "npy"}
    if len(file_endings) != 1:
        raise RuntimeError(f"Unable to infer preprocessed dataset variant from {folder}")
    ending = next(iter(file_endings))
    if ending == "npz":
        return NpzCaseDataset
    if ending == "b2nd":
        return BloscCaseDataset
    raise RuntimeError(f"Unsupported preprocessed dataset ending: {ending}")
