from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd
from batchgenerators.dataloading.data_loader import DataLoader
from threadpoolctl import threadpool_limits

from .dataset import PreprocessedCaseDataset
from .labels import RegionLabelManager


class PreprocessedBatchLoader(DataLoader):
    def __init__(
        self,
        data: PreprocessedCaseDataset,
        batch_size: int,
        patch_size,
        final_patch_size,
        label_manager: RegionLabelManager,
        oversample_foreground_percent: float = 0.0,
        sampling_probabilities=None,
        pad_sides=None,
        probabilistic_oversampling: bool = False,
        transforms=None,
    ) -> None:
        super().__init__(data, batch_size, 1, None, True, False, True, sampling_probabilities)
        if len(patch_size) == 2:
            final_patch_size = (1, *final_patch_size)
            patch_size = (1, *patch_size)
            self.patch_size_was_2d = True
        else:
            self.patch_size_was_2d = False

        self.indices = data.identifiers
        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
        self.need_to_pad = (np.array(patch_size) - np.array(final_patch_size)).astype(int)
        if pad_sides is not None:
            if self.patch_size_was_2d:
                pad_sides = (0, *pad_sides)
            for index in range(len(self.need_to_pad)):
                self.need_to_pad[index] += pad_sides[index]
        self.transforms = transforms
        self.annotated_classes_key = tuple([-1] + label_manager.all_labels)
        self.has_ignore = label_manager.has_ignore_label
        self.get_do_oversample = (
            self._oversample_last_xx_percent
            if not probabilistic_oversampling
            else self._probabilistic_oversampling
        )
        self.data_shape, self.seg_shape = self.determine_shapes()

    def _oversample_last_xx_percent(self, sample_idx: int) -> bool:
        return not sample_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))

    def _probabilistic_oversampling(self, _: int) -> bool:
        return np.random.uniform() < self.oversample_foreground_percent

    def determine_shapes(self):
        data, seg, seg_prev, _ = self._data.load_case(self._data.identifiers[0])
        num_channels = data.shape[0]
        spatial_shape = self.final_patch_size if self.transforms is not None else self.patch_size
        if self.patch_size_was_2d:
            spatial_shape = spatial_shape[1:]
        data_shape = (self.batch_size, num_channels, *spatial_shape)
        channels_seg = seg.shape[0] + (1 if seg_prev is not None else 0)
        seg_shape = (self.batch_size, channels_seg, *spatial_shape)
        return data_shape, seg_shape

    def get_bbox(self, data_shape, force_fg: bool, class_locations: Optional[dict], overwrite_class=None):
        need_to_pad = self.need_to_pad.copy()
        dim = len(data_shape)
        for index in range(dim):
            if need_to_pad[index] + data_shape[index] < self.patch_size[index]:
                need_to_pad[index] = self.patch_size[index] - data_shape[index]
        lbs = [-need_to_pad[i] // 2 for i in range(dim)]
        ubs = [
            data_shape[i] + need_to_pad[i] // 2 + need_to_pad[i] % 2 - self.patch_size[i]
            for i in range(dim)
        ]

        if not force_fg and not self.has_ignore:
            bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]
        else:
            if not force_fg and self.has_ignore:
                selected_class = self.annotated_classes_key
                if len(class_locations[selected_class]) == 0:
                    warnings.warn("Warning: no annotated pixels in image.")
                    selected_class = None
            elif force_fg:
                if class_locations is None:
                    raise RuntimeError("class_locations is required for foreground oversampling")
                eligible_classes_or_regions = [key for key in class_locations if len(class_locations[key]) > 0]
                tmp = [key == self.annotated_classes_key if isinstance(key, tuple) else False for key in eligible_classes_or_regions]
                if any(tmp) and len(eligible_classes_or_regions) > 1:
                    eligible_classes_or_regions.pop(np.where(tmp)[0][0])
                if len(eligible_classes_or_regions) == 0:
                    selected_class = None
                else:
                    selected_class = (
                        eligible_classes_or_regions[np.random.choice(len(eligible_classes_or_regions))]
                        if overwrite_class is None or overwrite_class not in eligible_classes_or_regions
                        else overwrite_class
                    )
            else:
                raise RuntimeError("Invalid oversampling state")

            if selected_class is not None:
                voxels = class_locations[selected_class]
                selected_voxel = voxels[np.random.choice(len(voxels))]
                bbox_lbs = [
                    max(lbs[i], selected_voxel[i + 1] - self.patch_size[i] // 2)
                    for i in range(dim)
                ]
            else:
                bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]

        bbox_ubs = [bbox_lbs[i] + self.patch_size[i] for i in range(dim)]
        return bbox_lbs, bbox_ubs

    def generate_train_batch(self):
        selected_keys = self.get_indices()
        data_all = torch.empty(self.data_shape, dtype=torch.float32)
        seg_all = None

        with torch.no_grad():
            with threadpool_limits(limits=1, user_api=None):
                for batch_index, case_id in enumerate(selected_keys):
                    force_fg = self.get_do_oversample(batch_index)
                    data, seg, seg_prev, properties = self._data.load_case(case_id)
                    bbox_lbs, bbox_ubs = self.get_bbox(data.shape[1:], force_fg, properties["class_locations"])
                    bbox = [[low, high] for low, high in zip(bbox_lbs, bbox_ubs)]

                    data_cropped = torch.from_numpy(crop_and_pad_nd(data, bbox, 0)).float()
                    seg_cropped = torch.from_numpy(crop_and_pad_nd(seg, bbox, -1)).to(torch.int16)
                    if seg_prev is not None:
                        seg_prev_cropped = torch.from_numpy(crop_and_pad_nd(seg_prev, bbox, -1)).to(torch.int16)
                        seg_cropped = torch.cat((seg_cropped, seg_prev_cropped[None]), dim=0)

                    if self.patch_size_was_2d:
                        data_cropped = data_cropped[:, 0]
                        seg_cropped = seg_cropped[:, 0]

                    if self.transforms is not None:
                        transformed = self.transforms(image=data_cropped, segmentation=seg_cropped)
                        data_sample = transformed["image"]
                        seg_sample = transformed["segmentation"]
                    else:
                        data_sample = data_cropped
                        seg_sample = seg_cropped

                    data_all[batch_index] = data_sample
                    if isinstance(seg_sample, list):
                        if seg_all is None:
                            seg_all = [torch.empty((self.batch_size, *item.shape), dtype=item.dtype) for item in seg_sample]
                        for seg_index, item in enumerate(seg_sample):
                            seg_all[seg_index][batch_index] = item
                    else:
                        if seg_all is None:
                            seg_all = torch.empty((self.batch_size, *seg_sample.shape), dtype=seg_sample.dtype)
                        seg_all[batch_index] = seg_sample

        return {"data": data_all, "target": seg_all, "keys": selected_keys}
