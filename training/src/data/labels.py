from __future__ import annotations

from typing import Callable, Optional, Union

import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image

# from torch import Tensor


def softmax_helper_dim0(x: Tensor) -> Tensor:
    return torch.softmax(x, dim=0)


class RegionLabelManager:
    def __init__(
        self,
        label_dict: dict,
        regions_class_order: list[int] | None,
        force_use_labels: bool = False,
        inference_nonlin: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        self._sanity_check(label_dict)
        self.label_dict = label_dict
        self.regions_class_order = regions_class_order
        self._force_use_labels = force_use_labels
        self._has_regions = (
            False
            if force_use_labels
            else any(
                isinstance(value, (tuple, list)) and len(value) > 1
                for value in self.label_dict.values()
            )
        )
        self._ignore_label = self._determine_ignore_label()
        self._all_labels = self._get_all_labels()
        self._regions = self._get_regions()
        self.inference_nonlin = (
            inference_nonlin
            if inference_nonlin is not None
            else (torch.sigmoid if self.has_regions else softmax_helper_dim0)
        )

    def _sanity_check(self, label_dict: dict) -> None:
        if "background" not in label_dict:
            raise RuntimeError("Background label must be declared and must be 0.")
        if int(label_dict["background"]) != 0:
            raise RuntimeError("Background label must be 0.")

    def _determine_ignore_label(self) -> Optional[int]:
        ignore_label = self.label_dict.get("ignore")
        if ignore_label is not None and not isinstance(ignore_label, int):
            raise TypeError("Ignore label must be an integer.")
        return ignore_label

    def _get_all_labels(self) -> list[int]:
        labels: list[int] = []
        for name, value in self.label_dict.items():
            if name == "ignore":
                continue
            if isinstance(value, (tuple, list)):
                labels.extend(int(v) for v in value)
            else:
                labels.append(int(value))
        result = sorted(np.unique(labels).tolist())
        return [int(v) for v in result]

    def _get_regions(self) -> list[Union[int, tuple[int, ...]]] | None:
        if not self._has_regions or self._force_use_labels:
            return None
        if self.regions_class_order is None:
            raise RuntimeError(
                "regions_class_order is required for region-based training."
            )
        regions: list[Union[int, tuple[int, ...]]] = []
        for name, value in self.label_dict.items():
            if name == "ignore":
                continue
            if (np.isscalar(value) and value == 0) or (
                isinstance(value, (tuple, list))
                and len(np.unique(value)) == 1
                and int(np.unique(value)[0]) == 0
            ):
                continue
            if isinstance(value, list):
                value = tuple(value)
            regions.append(value)
        if len(regions) != len(self.regions_class_order):
            raise RuntimeError(
                "regions_class_order length must match number of regions."
            )
        return regions

    @staticmethod
    def filter_background(
        classes_or_regions: list[Union[int, tuple[int, ...]]],
    ) -> list[Union[int, tuple[int, ...]]]:
        return [
            item
            for item in classes_or_regions
            if ((not isinstance(item, (tuple, list))) and item != 0)
            or (
                isinstance(item, (tuple, list))
                and not (len(np.unique(item)) == 1 and int(np.unique(item)[0]) == 0)
            )
        ]

    @property
    def has_regions(self) -> bool:
        return self._has_regions

    @property
    def has_ignore_label(self) -> bool:
        return self._ignore_label is not None

    @property
    def all_regions(self) -> list[Union[int, tuple[int, ...]]] | None:
        return self._regions

    @property
    def all_labels(self) -> list[int]:
        return self._all_labels

    @property
    def ignore_label(self) -> Optional[int]:
        return self._ignore_label

    @property
    def foreground_regions(self):
        return self.filter_background(self.all_regions or [])

    @property
    def foreground_labels(self):
        return self.filter_background(self.all_labels)

    @property
    def num_segmentation_heads(self) -> int:
        return (
            len(self.foreground_regions) if self.has_regions else len(self.all_labels)
        )

    def apply_inference_nonlin(
        self, logits: Union[np.ndarray, Tensor]
    ) -> Union[np.ndarray, Tensor]:
        input_is_numpy = isinstance(logits, np.ndarray)
        tensor = torch.from_numpy(logits) if input_is_numpy else logits
        with torch.no_grad():
            probabilities = self.inference_nonlin(tensor.float())
        return probabilities.cpu().numpy() if input_is_numpy else probabilities

    def convert_probabilities_to_segmentation(
        self, predicted_probabilities: Union[np.ndarray, Tensor]
    ) -> Union[np.ndarray, Tensor]:
        if self.has_regions:
            if isinstance(predicted_probabilities, np.ndarray):
                segmentation = np.zeros(
                    predicted_probabilities.shape[1:], dtype=np.uint16
                )
            else:
                segmentation = torch.zeros(
                    predicted_probabilities.shape[1:],
                    dtype=torch.int16,
                    device=predicted_probabilities.device,
                )
            for index, cls in enumerate(self.regions_class_order or []):
                segmentation[predicted_probabilities[index] > 0.5] = cls
            return segmentation

        input_is_numpy = isinstance(predicted_probabilities, np.ndarray)
        if not input_is_numpy:
            predicted_probabilities = predicted_probabilities.cpu().numpy()
        segmentation = predicted_probabilities.argmax(0)
        return segmentation if input_is_numpy else torch.from_numpy(segmentation)

    def convert_logits_to_segmentation(
        self, predicted_logits: Union[np.ndarray, Tensor]
    ) -> Union[np.ndarray, Tensor]:
        probabilities = (
            self.apply_inference_nonlin(predicted_logits)
            if self.has_regions
            else predicted_logits
        )
        return self.convert_probabilities_to_segmentation(probabilities)

    def revert_cropping_on_probabilities(
        self,
        predicted_probabilities: Union[Tensor, np.ndarray],
        bbox: list[list[int]],
        original_shape: list[int] | tuple[int, ...],
    ):
        output = (
            np.zeros(
                (predicted_probabilities.shape[0], *original_shape),
                dtype=predicted_probabilities.dtype,
            )
            if isinstance(predicted_probabilities, np.ndarray)
            else torch.zeros(
                (predicted_probabilities.shape[0], *original_shape),
                dtype=predicted_probabilities.dtype,
                device=predicted_probabilities.device,
            )
        )
        if not self.has_regions:
            output[0] = 1
        return insert_crop_into_image(output, predicted_probabilities, bbox)


def load_brats_label_manager(dataset_json: dict) -> RegionLabelManager:
    return RegionLabelManager(
        label_dict=dataset_json["labels"],
        regions_class_order=dataset_json.get("regions_class_order"),
    )
