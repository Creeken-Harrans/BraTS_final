from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from project import (
    get_configuration_name,
    get_dataset_name,
    get_default_folds,
    get_model_name,
    get_project_root,
    get_reference_plans_file,
    get_results_root,
    resolve_fold_validation_dir,
)
from evaluation.src.metrics import accumulate_cv_results


DEFAULT_METRIC = "foreground_mean.Dice"
PLANS_IDENTIFIER = "ProjectPlans"


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
    return _dedupe_paths([get_results_root()])


def _legacy_search_root_candidates() -> list[Path]:
    workspace_root = get_project_root().parent
    return _dedupe_paths(
        [
            workspace_root
            / "BraTS"
            / "03_training_and_results"
            / "artifacts"
            / "nnUNet_results",
            workspace_root / "BraTS" / "04_inference_and_evaluation" / "evaluation",
        ]
    )


def _format_folds_label(folds: tuple[int, ...]) -> str:
    return "_".join(str(fold) for fold in folds)


def _model_identifier() -> str:
    return f"{get_model_name()}__{PLANS_IDENTIFIER}__{get_configuration_name()}"


def _crossval_output_dir(folds: tuple[int, ...]) -> Path:
    return get_results_root() / _model_identifier() / f"crossval_results_folds_{_format_folds_label(folds)}"


def _inference_output_dir() -> Path:
    return get_results_root() / get_dataset_name()


def _discover_available_validation_folds(
    requested_folds: tuple[int, ...] | None = None,
) -> tuple[int, ...]:
    candidates = requested_folds if requested_folds is not None else tuple(get_default_folds())
    available: list[int] = []
    for fold in candidates:
        validation_dir = resolve_fold_validation_dir(fold)
        if validation_dir.is_dir() and any(validation_dir.glob("*.nii.gz")):
            available.append(int(fold))
    return tuple(available)


def _prepare_default_crossval_results() -> tuple[Path, tuple[int, ...]] | None:
    available_folds = _discover_available_validation_folds()
    if not available_folds:
        return None

    output_dir = _crossval_output_dir(available_folds)
    accumulate_cv_results(
        output_dir,
        folds=available_folds,
        num_processes=1,
        overwrite=True,
    )
    return output_dir, available_folds


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


def _build_no_summary_error(search_roots: list[str | Path] | None) -> RuntimeError:
    searched_roots = _default_search_roots() if not search_roots else _dedupe_paths([Path(root) for root in search_roots])
    message_lines = [
        "No summary.json files were found in the requested search roots.",
        "Searched roots:",
        *[f"  - {path}" for path in searched_roots],
        "Generate results first with `train`, `validate`, `predict`, or `accumulate-cv`.",
    ]

    legacy_roots = [path for path in _legacy_search_root_candidates() if path.exists()]
    legacy_summaries = _discover_summary_files(legacy_roots) if legacy_roots else []
    if legacy_summaries:
        message_lines.extend(
            [
                "Found summary.json files in the neighboring legacy BraTS project.",
                "You can scan them with:",
                "  python run.py find-best-config --search-root "
                + " ".join(str(path) for path in legacy_roots),
            ]
        )

    return RuntimeError("\n".join(message_lines))


def _shared_inference_dir() -> Path:
    return _inference_output_dir()


def _write_inference_artifacts(best: dict[str, Any]) -> tuple[Path, Path]:
    output_dir = _shared_inference_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    folds = tuple(int(fold) for fold in get_default_folds())
    if best.get("shared_folds"):
        folds = tuple(int(fold) for fold in best["shared_folds"])
    elif best.get("fold") is not None:
        folds = (int(best["fold"]),)

    postprocessing_file = output_dir / "postprocessing.json"
    postprocessing_file.write_text(
        json.dumps(
            {
                "enabled": False,
                "reason": "BraTS_final does not implement automatic postprocessing selection.",
                "summary_path": best["summary_path"],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    payload = {
        "dataset_name_or_id": get_dataset_name(),
        "folds": folds,
        "considered_models": [
            {
                "trainer": get_model_name(),
                "plans": PLANS_IDENTIFIER,
                "configuration": get_configuration_name(),
            }
        ],
        "ensembling_allowed": False,
        "all_results": {best["summary_path"]: best["metric_value"]},
        "best_model_or_ensemble": {
            "result_on_crossval_pre_pp": best["metric_value"],
            "result_on_crossval_post_pp": best["metric_value"],
            "summary_path": best["summary_path"],
            "postprocessing_file": str(postprocessing_file),
            "some_plans_file": str(get_reference_plans_file()),
            "selected_model_or_models": [
                {
                    "trainer": get_model_name(),
                    "configuration": get_configuration_name(),
                    "plans_identifier": PLANS_IDENTIFIER,
                }
            ],
        },
    }
    inference_information = output_dir / "inference_information.json"
    inference_information.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    instructions = [
        "***Run inference like this:***",
        "",
        "python run.py predict --sample-training-cases 12 --sample-seed 123",
        "",
        "***Best configuration summary***",
        "",
        f"dataset: {get_dataset_name()}",
        f"model: {get_model_name()}",
        f"configuration: {get_configuration_name()}",
        f"plans: {PLANS_IDENTIFIER}",
        f"folds: {' '.join(str(fold) for fold in folds)}",
        f"metric: {best['metric_path']}",
        f"value: {best['metric_value']:.6f}",
        f"summary: {best['summary_path']}",
        "",
        "***Postprocessing***",
        "",
        f"metadata: {postprocessing_file}",
    ]
    inference_instructions = output_dir / "inference_instructions.txt"
    inference_instructions.write_text("\n".join(instructions) + "\n", encoding="utf-8")
    return inference_information, inference_instructions


def find_best_config(
    *,
    search_roots: list[str | Path] | None = None,
    metric_path: str = DEFAULT_METRIC,
    lower_is_better: bool | None = None,
) -> dict[str, Any]:
    summary_files = _discover_summary_files(search_roots)
    auto_prepared_cv_dir: Path | None = None
    auto_prepared_folds: tuple[int, ...] | None = None
    if not summary_files:
        if search_roots is None:
            prepared = _prepare_default_crossval_results()
            if prepared is not None:
                auto_prepared_cv_dir, auto_prepared_folds = prepared
                summary_files = _discover_summary_files([auto_prepared_cv_dir])
        if not summary_files:
            raise _build_no_summary_error(search_roots)

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
    best = dict(ranked[0])
    if auto_prepared_cv_dir is not None:
        best["auto_prepared_cv_dir"] = str(auto_prepared_cv_dir)
    if auto_prepared_folds is not None:
        best["shared_folds"] = list(auto_prepared_folds)
    inference_information_path, inference_instructions_path = _write_inference_artifacts(best)
    all_results = {_model_identifier(): best["metric_value"]}
    return {
        "metric_path": metric_path,
        "lower_is_better": use_lower_is_better,
        "searched_roots": [str(path) for path in (_default_search_roots() if not search_roots else _dedupe_paths([Path(root) for root in search_roots]))],
        "num_candidates": len(ranked),
        "best": best,
        "ranking": ranked,
        "skipped": skipped,
        "all_results": all_results,
        "considered_models": [
            {
                "trainer": get_model_name(),
                "plans": PLANS_IDENTIFIER,
                "configuration": get_configuration_name(),
            }
        ],
        "ensembling_allowed": False,
        "folds": best.get("shared_folds", list(get_default_folds())),
        "inference_information_path": str(inference_information_path),
        "inference_instructions_path": str(inference_instructions_path),
    }


__all__ = ["DEFAULT_METRIC", "find_best_config"]
