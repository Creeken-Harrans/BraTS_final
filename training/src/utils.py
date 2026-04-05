from __future__ import annotations

from typing import Any

import numpy as np


def make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return make_json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def collate_outputs(outputs: list[dict[str, Any]]) -> dict[str, Any]:
    collated: dict[str, Any] = {}
    for key in outputs[0]:
        first = outputs[0][key]
        if np.isscalar(first):
            collated[key] = [item[key] for item in outputs]
        elif isinstance(first, np.ndarray):
            collated[key] = np.vstack([item[key][None] for item in outputs])
        elif isinstance(first, list):
            collated[key] = [value for item in outputs for value in item[key]]
        else:
            raise TypeError(f"Unsupported collate type for {key}: {type(first)!r}")
    return collated
