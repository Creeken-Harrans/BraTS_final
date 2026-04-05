from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from project import get_evaluation_root, get_results_root


DEFAULT_METRIC = "foreground_mean.Dice"


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def _default_search_roots() -> list[Path]:
    return _dedupe_paths([get_results_root(), get_evaluation_root()])


def _discover_summary_files(search_roots: list[str | Path] | None = None) -> list[Path]:
    roots = _default_search_roots() if not search_roots else [Path(root).resolve() for root in search_roots]
    summary_files: list[Path] = []
    for root in _dedupe_paths(roots):
        if not root.exists():
            continue
        if root.is_file():
            if root.name == "summary.json":
                summary_files.append(root)
            continue
        summary_files.extend(sorted(path.resolve() for path in root.rglob("summary.json")))
    return _dedupe_paths(summary_files)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_metric(summary: dict[str, Any], metric_path: str) -> float:
    value: Any = summary
    for key in metric_path.split("."):
        if not isinstance(value, dict) or key not in value:
            raise KeyError(metric_path)
        value = value[key]
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"Metric {metric_path} is not a finite number")
    return float(value)


def _find_training_state(summary_path: Path) -> Path | None:
    for candidate in (
        summary_path.parent / "training_state.json",
        summary_path.parent.parent / "training_state.json",
    ):
        if candidate.is_file():
            return candidate.resolve()
    return None


def _infer_candidate_kind(summary_path: Path) -> str:
    if summary_path.parent.name == "validation":
        return "fold_validation"
    if "crossval" in summary_path.parent.name:
        return "cross_validation"
    return "summary"


def _build_candidate(summary_path: Path, metric_path: str) -> dict[str, Any]:
    summary = _load_json(summary_path)
    metric_value = _resolve_metric(summary, metric_path)
    training_state_path = _find_training_state(summary_path)
    training_state = _load_json(training_state_path) if training_state_path else {}
    hyperparameters = training_state.get("training_hyperparameters")

    return {
        "summary_path": str(summary_path),
        "metric_path": metric_path,
        "metric_value": metric_value,
        "kind": _infer_candidate_kind(summary_path),
        "fold": training_state.get("fold"),
        "configuration_name": training_state.get("configuration_name") or training_state.get("configuration"),
        "model_name": training_state.get("model_name") or training_state.get("trainer_name"),
        "status": training_state.get("status"),
        "validation_only": training_state.get("validation_only"),
        "val_with_best": training_state.get("val_with_best"),
        "training_state_path": str(training_state_path) if training_state_path else None,
        "hyperparameters": hyperparameters,
    }


def _metric_name(metric_path: str) -> str:
    return metric_path.rsplit(".", maxsplit=1)[-1]


def _is_lower_better(metric_path: str) -> bool:
    return _metric_name(metric_path) in {"FP", "FN"}


def find_best_config(
    *,
    search_roots: list[str | Path] | None = None,
    metric_path: str = DEFAULT_METRIC,
    lower_is_better: bool | None = None,
) -> dict[str, Any]:
    summary_files = _discover_summary_files(search_roots)
    if not summary_files:
        raise RuntimeError("No summary.json files were found in the requested search roots.")

    candidates: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for summary_path in summary_files:
        try:
            candidates.append(_build_candidate(summary_path, metric_path))
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            skipped.append({"summary_path": str(summary_path), "reason": str(exc)})

    if not candidates:
        raise RuntimeError(
            f"Found {len(summary_files)} summary.json files, but none contained a usable {metric_path} metric."
        )

    use_lower_is_better = _is_lower_better(metric_path) if lower_is_better is None else lower_is_better
    ranked = sorted(
        candidates,
        key=lambda item: (item["metric_value"], item["summary_path"]),
        reverse=not use_lower_is_better,
    )
    return {
        "metric_path": metric_path,
        "lower_is_better": use_lower_is_better,
        "searched_roots": [str(path) for path in (_default_search_roots() if not search_roots else _dedupe_paths([Path(root) for root in search_roots]))],
        "num_candidates": len(ranked),
        "best": ranked[0],
        "ranking": ranked,
        "skipped": skipped,
    }


__all__ = ["DEFAULT_METRIC", "find_best_config"]
