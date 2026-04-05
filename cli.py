from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from project import (
    get_evaluation_root,
    get_evaluation_report_root,
    get_configuration_name,
    get_dataset_name,
    get_default_folds,
    get_fold_root,
    get_model_name,
    get_gt_segmentations_dir,
    get_preprocessed_dataset_dir,
    get_primary_preprocessed_dataset_dir,
    get_primary_raw_dataset_dir,
    get_project_root,
    resolve_fold_validation_dir,
)
def cmd_doctor(_: argparse.Namespace) -> int:
    from training.src.core.runtime import resolve_training_data_layout

    data_layout = resolve_training_data_layout()
    print("BraTS_final doctor")
    print(f"project_root: {get_project_root()}")
    print(f"dataset: {get_dataset_name()}")
    print(f"model: {get_model_name()}")
    print(f"configuration: {get_configuration_name()}")
    print(f"preprocessed_dir: {get_preprocessed_dataset_dir()}")
    print(f"active_metadata_dir: {get_primary_preprocessed_dataset_dir()}")
    print(f"training_cases_dir: {data_layout['training_cases_dir']}")
    print(f"gt_segmentations_dir: {data_layout['gt_segmentations_dir']}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from training.src.core.runtime import (
        build_device,
        ensure_preprocessed_training_inputs,
        resolve_training_resume_checkpoint,
        resolve_validation_checkpoint,
        sync_training_snapshot,
        training_fold_is_complete,
    )
    from training.src.core.trainer import BratsTrainer

    ensure_preprocessed_training_inputs()
    device = build_device()
    training_checkpoint = resolve_training_resume_checkpoint(args.fold)
    validation_checkpoint = resolve_validation_checkpoint(args.fold, use_best=args.val_best)

    if (
        not args.restart_training
        and not args.validation_only
        and args.pretrained_weights is None
        and training_checkpoint is not None
        and training_fold_is_complete(args.fold)
        ):
        print(
            "[INFO] Fold is already marked completed. "
            "Skipping trainer startup.\n"
            "Use --validation-only to rerun validation, or --restart-training to force a new run."
        )
        return 0

    trainer = BratsTrainer(
        fold=args.fold,
        device=device,
        export_validation_probabilities=args.npz,
        validation_only=args.validation_only,
        restart_training=args.restart_training,
        pretrained_weights=args.pretrained_weights,
        disable_checkpointing=args.disable_checkpointing,
        val_with_best=args.val_best,
    )
    trainer.initialize()
    trainer.maybe_resume(training_checkpoint)
    if args.validation_only:
        if validation_checkpoint is None:
            raise RuntimeError(
                f"No checkpoint available for fold {args.fold}. "
                "Run training first or provide a completed fold."
            )
        trainer.load_checkpoint(validation_checkpoint)
    else:
        trainer.run_training()
    trainer.perform_actual_validation()
    sync_training_snapshot(args.fold)
    return 0


def cmd_train_all(args: argparse.Namespace) -> int:
    from training.src.core.runtime import get_train_all_plan

    print("[INFO] train-all plan:")
    for line in get_train_all_plan(
        restart_training=args.restart_training,
        validation_only=args.validation_only,
        pretrained_weights=args.pretrained_weights,
    ):
        print(f"  - {line}")

    for fold in get_default_folds():
        fold_args = argparse.Namespace(**vars(args))
        fold_args.fold = fold
        cmd_train(fold_args)
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    from validation.src.run_validation import run_validation_for_fold

    summary = run_validation_for_fold(
        args.fold,
        use_best=args.val_best,
        export_probabilities=args.npz,
    )
    print(
        f"[OK] Validation summary written to: "
        f"{resolve_fold_validation_dir(args.fold).parent / 'summary.json'}"
    )
    print(
        f"[OK] Foreground mean Dice: "
        f"{summary['foreground_mean'].get('Dice')}"
    )
    return 0


def cmd_find_best_config(args: argparse.Namespace) -> int:
    from validation.src.find_best_config import find_best_config

    result = find_best_config(
        search_roots=args.search_root,
    )
    best = result["best"]
    print()
    print("***All results:***")
    for key, value in result["all_results"].items():
        print(f"{key}: {value}")
    best_key = next(iter(result["all_results"]))
    print(f"\n*Best*: {best_key}: {best['metric_value']}")
    print()
    print("***Determining postprocessing for best model/ensemble***")
    print(f"[OK] Cross-validation summary: {best['summary_path']}")
    if best.get("auto_prepared_cv_dir"):
        print(f"[OK] Cross-validation results prepared in: {best['auto_prepared_cv_dir']}")
    print(f"[OK] Inference info: {result['inference_information_path']}")
    print(f"[OK] Inference instructions: {result['inference_instructions_path']}")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    from validation.src.predict import predict_training_cases

    result = predict_training_cases(
        sample_training_cases=args.sample_training_cases,
        sample_seed=args.sample_seed,
        use_best_checkpoint=args.val_best,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        export_probabilities=args.npz,
    )
    summary = result["summary"]
    print(f"[OK] Prediction output written to: {result['output_dir']}")
    print(f"[OK] Summary written to: {result['summary_path']}")
    print(f"[OK] Sample selection written to: {result['sample_selection_path']}")
    print(f"[OK] Foreground mean Dice: {summary['foreground_mean'].get('Dice')}")
    print(f"[OK] Fold used: {result['fold']} ({result['fold_selection_reason']})")
    print(f"[OK] Checkpoint: {result['checkpoint_path']}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from evaluation.src.evaluate_predictions import evaluate_prediction_folder

    gt_dir = Path(args.gt_dir).resolve() if args.gt_dir else get_gt_segmentations_dir()
    if args.pred_dir is not None:
        pred_dir = Path(args.pred_dir).resolve()
    elif args.fold is not None:
        pred_dir = resolve_fold_validation_dir(args.fold)
    else:
        raise RuntimeError("evaluate requires either --pred-dir or --fold")

    output_file = (
        Path(args.output_file).resolve()
        if args.output_file
        else (get_evaluation_root() / f"{pred_dir.name}_summary.json")
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    evaluate_prediction_folder(
        pred_dir,
        gt_dir=gt_dir,
        output_file=output_file,
        num_processes=args.num_processes,
        chill=True if args.fold is not None and args.pred_dir is None else args.chill,
    )
    print(f"[OK] Evaluation summary written to: {output_file}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from evaluation.src.evaluate_predictions import evaluate_prediction_folder
    from evaluation.src.generate_evaluation_report import generate_evaluation_report

    raw_dataset_dir = (
        Path(args.raw_dataset_dir).resolve()
        if args.raw_dataset_dir
        else get_primary_raw_dataset_dir()
    )

    if args.pred_dir is not None:
        pred_dir = Path(args.pred_dir).resolve()
    elif args.fold is not None:
        pred_dir = resolve_fold_validation_dir(args.fold)
    else:
        raise RuntimeError("report requires either --pred-dir or --fold")

    if args.summary_file is not None:
        summary_file = Path(args.summary_file).resolve()
    else:
        summary_file = pred_dir / "summary.json"

    if args.output_dir is not None:
        output_dir = Path(args.output_dir).resolve()
    elif args.fold is not None:
        output_dir = get_fold_root(args.fold) / "report"
    elif pred_dir.parent == get_project_root():
        output_dir = get_evaluation_report_root() / pred_dir.name
    else:
        output_dir = pred_dir.parent / "report"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not summary_file.is_file():
        evaluate_prediction_folder(
            pred_dir,
            gt_dir=Path(args.gt_dir).resolve() if args.gt_dir else get_gt_segmentations_dir(),
            output_file=summary_file,
            num_processes=args.num_processes,
            chill=args.chill or args.fold is not None,
        )

    sample_selection_file = (
        Path(args.sample_selection_file).resolve()
        if args.sample_selection_file is not None
        else None
    )
    generate_evaluation_report(
        summary_file=summary_file,
        predictions_dir=pred_dir,
        raw_dataset_dir=raw_dataset_dir,
        output_dir=output_dir,
        sample_selection_file=sample_selection_file,
    )
    print(f"[OK] Evaluation report written to: {output_dir}")
    return 0


def cmd_accumulate_cv(args: argparse.Namespace) -> int:
    from evaluation.src.accumulate_cv_results import accumulate_cv_results

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (get_evaluation_root() / f"crossval_folds_{'_'.join(str(f) for f in args.folds)}")
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    accumulate_cv_results(
        output_dir,
        folds=args.folds,
        num_processes=args.num_processes,
        overwrite=not args.no_overwrite,
    )
    print(f"[OK] Cross-validation summary written to: {output_dir / 'summary.json'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BraTS_final project entrypoint")
    subparsers = parser.add_subparsers(dest="command")

    doctor = subparsers.add_parser("doctor", help="Show project configuration")
    doctor.set_defaults(func=cmd_doctor)

    train = subparsers.add_parser("train", help="Prepare one training fold")
    train.add_argument("--fold", type=int, required=True, help="Fold index")
    train.add_argument("--npz", action="store_true", help="Keep validation probabilities")
    train.add_argument("--validation-only", action="store_true")
    train.add_argument("--restart-training", action="store_true")
    train.add_argument("--pretrained-weights", type=str, default=None)
    train.add_argument("--disable-checkpointing", action="store_true")
    train.add_argument("--val-best", action="store_true")
    train.set_defaults(func=cmd_train)

    train_all = subparsers.add_parser("train-all", help="Prepare all default folds")
    train_all.add_argument("--npz", action="store_true", help="Keep validation probabilities")
    train_all.add_argument("--validation-only", action="store_true")
    train_all.add_argument("--restart-training", action="store_true")
    train_all.add_argument("--pretrained-weights", type=str, default=None)
    train_all.add_argument("--disable-checkpointing", action="store_true")
    train_all.add_argument("--val-best", action="store_true")
    train_all.set_defaults(func=cmd_train_all)

    validate = subparsers.add_parser("validate", help="Run validation for one fold from an existing checkpoint")
    validate.add_argument("--fold", type=int, required=True, help="Fold index")
    validate.add_argument("--npz", action="store_true", help="Keep validation probabilities")
    validate.add_argument("--val-best", action="store_true")
    validate.set_defaults(func=cmd_validate)

    find_best = subparsers.add_parser(
        "find-best-config",
        help="Aggregate available validation folds and determine the best configuration",
    )
    find_best.add_argument(
        "--search-root",
        nargs="+",
        default=None,
        help="Optional directories or summary.json files to scan instead of the default project workflow.",
    )
    find_best.set_defaults(func=cmd_find_best_config)

    predict = subparsers.add_parser(
        "predict",
        help="Run inference on a sampled subset of training cases",
    )
    predict.add_argument("--sample-training-cases", type=int, required=True, help="Number of training cases to sample")
    predict.add_argument("--sample-seed", type=int, required=True, help="Random seed used for sampling cases")
    predict.add_argument("--npz", action="store_true", help="Keep restored probability maps")
    predict.add_argument("--val-best", action="store_true", help="Prefer checkpoint_best.pth over checkpoint_final.pth")
    predict.add_argument("--overwrite", action="store_true", help="Allow writing into an existing non-empty output directory")
    predict.add_argument("--output-dir", type=str, default=None, help="Prediction output directory")
    predict.set_defaults(func=cmd_predict)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate a validation/prediction folder")
    evaluate.add_argument("--fold", type=int, default=None, help="Use fold<n>/validation as prediction folder")
    evaluate.add_argument("--pred-dir", type=str, default=None, help="Prediction directory to evaluate")
    evaluate.add_argument("--gt-dir", type=str, default=None, help="Ground-truth directory")
    evaluate.add_argument("--output-file", type=str, default=None)
    evaluate.add_argument("--num-processes", type=int, default=1)
    evaluate.add_argument("--chill", action="store_true")
    evaluate.set_defaults(func=cmd_evaluate)

    report = subparsers.add_parser("report", help="Generate a full evaluation report directory")
    report.add_argument("--fold", type=int, default=None, help="Use fold<n>/validation as prediction folder")
    report.add_argument("--pred-dir", type=str, default=None, help="Prediction directory to report on")
    report.add_argument("--summary-file", type=str, default=None, help="Existing summary.json to reuse")
    report.add_argument("--gt-dir", type=str, default=None, help="Ground-truth directory for auto-evaluate fallback")
    report.add_argument("--raw-dataset-dir", type=str, default=None, help="Raw dataset directory with imagesTr")
    report.add_argument("--output-dir", type=str, default=None, help="Report output directory")
    report.add_argument("--sample-selection-file", type=str, default=None)
    report.add_argument("--num-processes", type=int, default=1)
    report.add_argument("--chill", action="store_true")
    report.set_defaults(func=cmd_report)

    accumulate = subparsers.add_parser(
        "accumulate-cv",
        help="Merge validation outputs from multiple folds and re-evaluate them",
    )
    accumulate.add_argument("--folds", nargs="+", type=int, default=get_default_folds())
    accumulate.add_argument("--output-dir", type=str, default=None)
    accumulate.add_argument("--num-processes", type=int, default=1)
    accumulate.add_argument("--no-overwrite", action="store_true")
    accumulate.set_defaults(func=cmd_accumulate_cv)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    return int(args.func(args))
