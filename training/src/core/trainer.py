from __future__ import annotations

import json
import os
import pickle
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from time import time
from typing import Any

import numpy as np
import SimpleITK as sitk
import torch
from batchgenerators.dataloading.single_threaded_augmenter import (
    SingleThreadedAugmenter,
)

from project import (
    get_gt_segmentations_dir,
    get_primary_preprocessed_dataset_dir,
    get_primary_raw_dataset_dir,
    get_training_cases_dir,
)
from evaluation.src.metrics import evaluate_prediction_folder, save_summary_json
from validation.src.inference import (
    compute_case_metrics,
    predict_sliding_window_logits,
    restore_prediction_to_original_space,
    summarize_validation_metrics,
    write_segmentation_nifti,
)
from validation.src.visualization import write_validation_overlay

from ..data.data_loader import PreprocessedBatchLoader
from ..data.dataset import infer_preprocessed_dataset_class
from ..data.labels import load_brats_label_manager
from ..models import BRATS_3D_PATCH_SIZE, build_brats_training_model
from ..data.transforms import (
    build_training_transforms,
    build_validation_transforms,
    get_deep_supervision_scales,
)
from ..monitoring.async_plotter import AsyncPlotWorker
from ..monitoring.logger import TrainingLogger
from ..optimization.losses import build_region_loss
from ..optimization.optim import build_optimizer_and_scheduler
from ..optimization.pretrained import load_pretrained_weights
from ..utils import collate_outputs, make_json_safe
from .config import BratsTrainingConfig, get_default_training_config
from .runtime import (
    build_training_metadata,
    get_training_logs_dir,
    get_training_output_dir,
)


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model._orig_mod if hasattr(model, "_orig_mod") else model


class BratsTrainer:
    def __init__(
        self,
        fold: int,
        device: torch.device,
        *,
        export_validation_probabilities: bool = False,
        validation_only: bool = False,
        restart_training: bool = False,
        pretrained_weights: str | None = None,
        disable_checkpointing: bool = False,
        val_with_best: bool = False,
        config: BratsTrainingConfig | None = None,
    ) -> None:
        self.fold = int(fold)
        self.device = device
        self.config = config or get_default_training_config()
        self.export_validation_probabilities = bool(export_validation_probabilities)
        self.validation_only = bool(validation_only)
        self.restart_training = bool(restart_training)
        self.pretrained_weights = pretrained_weights
        self.disable_checkpointing = bool(disable_checkpointing)
        self.val_with_best = bool(val_with_best)

        metadata_dir = get_primary_preprocessed_dataset_dir()
        self.training_cases_dir = get_training_cases_dir()
        self.gt_segmentations_dir = get_gt_segmentations_dir()
        self.raw_dataset_dir = get_primary_raw_dataset_dir()
        self.dataset_json = json.loads(
            (metadata_dir / "dataset.json").read_text(encoding="utf-8")
        )
        self.splits = json.loads(
            (metadata_dir / "splits_final.json").read_text(encoding="utf-8")
        )
        self.dataset_class = infer_preprocessed_dataset_class(
            str(self.training_cases_dir)
        )
        self.label_manager = load_brats_label_manager(self.dataset_json)
        self.output_dir = get_training_output_dir(self.fold)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir = get_training_logs_dir(self.fold)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now()
        self.log_file = self.logs_dir / (
            f"training_log_{timestamp.year}_{timestamp.month}_{timestamp.day}_"
            f"{timestamp.hour:02d}_{timestamp.minute:02d}_{timestamp.second:02d}.txt"
        )
        self.training_state_file = self.output_dir / "training_state.json"
        self.debug_file = self.output_dir / "debug.json"
        self.logger = TrainingLogger()

        self.model = build_brats_training_model()
        self.loss = build_region_loss(
            use_deep_supervision=self.config.use_deep_supervision,
            num_deep_supervision_outputs=5,
        )
        self.optimizer, self.lr_scheduler = build_optimizer_and_scheduler(
            self.model, self.config
        )
        self.grad_scaler = (
            torch.amp.GradScaler("cuda") if self.device.type == "cuda" else None
        )

        self.current_epoch = 0
        self.best_ema: float | None = None
        self.resume_checkpoint_path: str | None = None
        self.latest_checkpoint_path: str | None = None
        self.best_checkpoint_path: str | None = None
        self.final_checkpoint_path: str | None = None

        self.train_loader: SingleThreadedAugmenter | None = None
        self.val_loader: SingleThreadedAugmenter | None = None
        self.plot_worker: AsyncPlotWorker | None = None

        if self.disable_checkpointing and self.val_with_best:
            raise RuntimeError(
                "--val-best is not compatible with --disable-checkpointing"
            )

    @property
    def epochs(self) -> int:
        return int(self.config.epochs)

    def print_to_log_file(
        self, *parts: Any, also_print_to_console: bool = True
    ) -> None:
        line = " ".join(str(part) for part in parts)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        with self.log_file.open("a", encoding="utf-8") as handle:
            handle.write(f"{datetime.now().isoformat()} {line}\n")
        if also_print_to_console:
            print(line)

    def _write_training_state(self, status: str) -> None:
        payload = build_training_metadata(self.fold, self.device, self.config)
        payload.update(
            {
                "status": status,
                "timestamp": datetime.now().isoformat(),
                "validation_only": self.validation_only,
                "export_validation_probabilities": self.export_validation_probabilities,
                "pretrained_weights": self.pretrained_weights,
                "restart_training": self.restart_training,
                "disable_checkpointing": self.disable_checkpointing,
                "val_with_best": self.val_with_best,
                "log_file": str(self.log_file),
                "resume_checkpoint_path": self.resume_checkpoint_path,
                "latest_checkpoint_path": self.latest_checkpoint_path,
                "best_checkpoint_path": self.best_checkpoint_path,
                "final_checkpoint_path": self.final_checkpoint_path,
                "logged_epochs": self.logger.get_num_logged_epochs(),
                "next_epoch": self.current_epoch,
                "remaining_epochs": max(self.epochs - self.current_epoch, 0),
            }
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.training_state_file.write_text(
            json.dumps(make_json_safe(payload), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        self._write_debug_state(status)

    def _write_debug_state(self, status: str) -> None:
        payload = {
            "fold": self.fold,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "device": str(self.device),
            "output_dir": str(self.output_dir),
            "logs_dir": str(self.logs_dir),
            "log_file": str(self.log_file),
            "training_state_file": str(self.training_state_file),
            "history": self.logger.get_history_snapshot(),
            "logged_epochs": self.logger.get_num_logged_epochs(),
            "current_epoch": self.current_epoch,
            "total_epochs": self.epochs,
            "resume_checkpoint_path": self.resume_checkpoint_path,
            "latest_checkpoint_path": self.latest_checkpoint_path,
            "best_checkpoint_path": self.best_checkpoint_path,
            "final_checkpoint_path": self.final_checkpoint_path,
            "best_ema": self.best_ema,
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.debug_file.write_text(
            json.dumps(make_json_safe(payload), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def _clear_restart_artifacts(self) -> None:
        for path in (
            self.output_dir / "checkpoint_latest.pth",
            self.output_dir / "checkpoint_best.pth",
            self.output_dir / "checkpoint_final.pth",
            self.output_dir / "training_state.json",
            self.output_dir / "debug.json",
            self.output_dir / "summary.json",
        ):
            if path.exists():
                path.unlink()
        validation_dir = self.output_dir / "validation"
        if validation_dir.is_dir():
            for child in validation_dir.iterdir():
                if child.is_file():
                    child.unlink()
        if self.logs_dir.is_dir():
            for child in self.logs_dir.iterdir():
                if child.is_file():
                    child.unlink()

    def initialize(self) -> None:
        if self.restart_training:
            self._clear_restart_artifacts()

        self.model = self.model.to(self.device)
        self.loss = self.loss.to(self.device)

        use_compile = (
            self.config.use_compile
            and hasattr(torch, "compile")
            and self.device.type == "cuda"
            and os.name != "nt"
        )
        if use_compile:
            self.model = torch.compile(self.model)

        if self.pretrained_weights and not self.validation_only:
            load_pretrained_weights(self.model, self.pretrained_weights)
            self.print_to_log_file(
                f"Loaded pretrained weights: {self.pretrained_weights}"
            )

        self.plot_worker = AsyncPlotWorker()
        self.train_loader, self.val_loader = self._build_dataloaders()
        if not self.validation_only:
            self._write_training_state("running")

    def close(self) -> None:
        if self.plot_worker is not None:
            self.plot_worker.close()
            self.plot_worker = None

    def _fold_split(self) -> tuple[list[str], list[str]]:
        if self.fold >= len(self.splits):
            raise RuntimeError(
                f"Fold {self.fold} is outside available splits ({len(self.splits)})."
            )
        return list(self.splits[self.fold]["train"]), list(
            self.splits[self.fold]["val"]
        )

    def _build_dataloaders(
        self,
    ) -> tuple[SingleThreadedAugmenter, SingleThreadedAugmenter]:
        tr_keys, val_keys = self._fold_split()
        dataset_train = self.dataset_class(str(self.training_cases_dir), tr_keys)
        dataset_val = self.dataset_class(str(self.training_cases_dir), val_keys)

        model = _unwrap_model(self.model)
        strides = getattr(model, "strides", None) or model.config.strides
        deep_supervision_scales = (
            get_deep_supervision_scales(strides)
            if self.config.use_deep_supervision
            else None
        )
        regions = (
            self.label_manager.foreground_regions
            if self.label_manager.has_regions
            else None
        )

        train_transforms = build_training_transforms(
            patch_size=BRATS_3D_PATCH_SIZE,
            rotation_for_da=(-30.0 / 360 * 2.0 * np.pi, 30.0 / 360 * 2.0 * np.pi),
            deep_supervision_scales=deep_supervision_scales,
            mirror_axes=self.config.inference_mirroring_axes,
            do_dummy_2d_data_aug=False,
            use_mask_for_norm=[True, True, True, True],
            is_cascaded=False,
            foreground_labels=self.label_manager.foreground_labels,
            regions=regions,
            ignore_label=self.label_manager.ignore_label,
        )
        val_transforms = build_validation_transforms(
            deep_supervision_scales=deep_supervision_scales,
            is_cascaded=False,
            foreground_labels=self.label_manager.foreground_labels,
            regions=regions,
            ignore_label=self.label_manager.ignore_label,
        )

        train_loader = PreprocessedBatchLoader(
            dataset_train,
            batch_size=self.config.batch_size,
            patch_size=BRATS_3D_PATCH_SIZE,
            final_patch_size=BRATS_3D_PATCH_SIZE,
            label_manager=self.label_manager,
            oversample_foreground_percent=self.config.oversample_foreground_percent,
            probabilistic_oversampling=self.config.probabilistic_oversampling,
            transforms=train_transforms,
        )
        val_loader = PreprocessedBatchLoader(
            dataset_val,
            batch_size=self.config.batch_size,
            patch_size=BRATS_3D_PATCH_SIZE,
            final_patch_size=BRATS_3D_PATCH_SIZE,
            label_manager=self.label_manager,
            oversample_foreground_percent=self.config.oversample_foreground_percent,
            probabilistic_oversampling=self.config.probabilistic_oversampling,
            transforms=val_transforms,
        )
        return SingleThreadedAugmenter(train_loader, None), SingleThreadedAugmenter(
            val_loader, None
        )

    def save_checkpoint(self, filename: str) -> None:
        if self.disable_checkpointing:
            self.print_to_log_file("No checkpoint written, checkpointing is disabled")
            self._write_training_state("running")
            return

        checkpoint = {
            "network_weights": _unwrap_model(self.model).state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "grad_scaler_state": (
                self.grad_scaler.state_dict() if self.grad_scaler is not None else None
            ),
            "logging": self.logger.get_checkpoint(),
            "best_ema": self.best_ema,
            "current_epoch": self.current_epoch,
            "config": self.config.__dict__,
        }
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, filename)
        basename = Path(filename).name
        if basename == "checkpoint_latest.pth":
            self.latest_checkpoint_path = filename
        elif basename == "checkpoint_best.pth":
            self.best_checkpoint_path = filename
        elif basename == "checkpoint_final.pth":
            self.final_checkpoint_path = filename
        self._write_training_state("running")

    def load_checkpoint(self, checkpoint_path: str | os.PathLike[str]) -> None:
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        _unwrap_model(self.model).load_state_dict(checkpoint["network_weights"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if self.grad_scaler is not None and checkpoint["grad_scaler_state"] is not None:
            self.grad_scaler.load_state_dict(checkpoint["grad_scaler_state"])
        self.logger.load_checkpoint(checkpoint["logging"])
        self.best_ema = checkpoint.get("best_ema")
        self.current_epoch = int(checkpoint.get("current_epoch", 0))
        checkpoint_path = str(checkpoint_path)
        self.resume_checkpoint_path = checkpoint_path
        self.latest_checkpoint_path = None
        self.best_checkpoint_path = None
        self.final_checkpoint_path = None
        basename = Path(checkpoint_path).name
        if basename == "checkpoint_latest.pth":
            self.latest_checkpoint_path = checkpoint_path
        elif basename == "checkpoint_best.pth":
            self.best_checkpoint_path = checkpoint_path
        elif basename == "checkpoint_final.pth":
            self.final_checkpoint_path = checkpoint_path
        self.print_to_log_file(
            "Loaded checkpoint state",
            {
                "checkpoint_path": checkpoint_path,
                "resuming_at_epoch": self.current_epoch,
                "logged_epochs": self.logger.get_num_logged_epochs(),
            },
        )
        if not self.validation_only:
            self._write_training_state("running")

    def maybe_resume(self, checkpoint_path: str | os.PathLike[str] | None) -> None:
        if (
            checkpoint_path is not None
            and not self.validation_only
            and not self.restart_training
            and self.pretrained_weights is None
        ):
            self.load_checkpoint(checkpoint_path)

    def _autocast_context(self):
        if self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def train_step(self, batch: dict[str, Any]) -> dict[str, np.ndarray]:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            target = [
                item.to(self.device, non_blocking=True).float() for item in target
            ]
        else:
            target = target.to(self.device, non_blocking=True).float()

        self.optimizer.zero_grad(set_to_none=True)
        with self._autocast_context():
            output = self.model(data)
            loss = self.loss(output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 12.0)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 12.0)
            self.optimizer.step()

        return {"loss": np.asarray(loss.detach().cpu().item(), dtype=np.float32)}

    def validation_step(self, batch: dict[str, Any]) -> dict[str, np.ndarray]:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            target = [
                item.to(self.device, non_blocking=True).float() for item in target
            ]
        else:
            target = target.to(self.device, non_blocking=True).float()

        with self._autocast_context():
            output = self.model(data)
            loss = self.loss(output, target)

        if self.config.use_deep_supervision:
            output = output[0]
            target = target[0]

        predicted = (torch.sigmoid(output) > 0.5).long()
        tp = (predicted * target).sum(dim=(0, 2, 3, 4)).detach().cpu().numpy()
        fp = (predicted * (1 - target)).sum(dim=(0, 2, 3, 4)).detach().cpu().numpy()
        fn = ((1 - predicted) * target).sum(dim=(0, 2, 3, 4)).detach().cpu().numpy()
        return {
            "loss": np.asarray(loss.detach().cpu().item(), dtype=np.float32),
            "tp_hard": tp.astype(np.float32, copy=False),
            "fp_hard": fp.astype(np.float32, copy=False),
            "fn_hard": fn.astype(np.float32, copy=False),
        }

    def _set_deep_supervision_enabled(self, enabled: bool) -> None:
        model = _unwrap_model(self.model)
        model.decoder.deep_supervision = enabled

    def on_epoch_end(
        self,
        train_outputs: list[dict[str, np.ndarray]],
        val_outputs: list[dict[str, np.ndarray]],
    ) -> None:
        train_collated = collate_outputs(train_outputs)
        val_collated = collate_outputs(val_outputs)

        train_loss = float(np.mean(train_collated["loss"]))
        val_loss = float(np.mean(val_collated["loss"]))
        tp = np.sum(val_collated["tp_hard"], axis=0)
        fp = np.sum(val_collated["fp_hard"], axis=0)
        fn = np.sum(val_collated["fn_hard"], axis=0)
        dices = [
            float(2 * i / (2 * i + j + k)) if (2 * i + j + k) > 0 else float("nan")
            for i, j, k in zip(tp, fp, fn)
        ]
        mean_fg_dice = float(np.nanmean(dices))

        self.logger.log("train_losses", train_loss, self.current_epoch)
        self.logger.log("val_losses", val_loss, self.current_epoch)
        self.logger.log("dice_per_class_or_region", dices, self.current_epoch)
        self.logger.log("mean_fg_dice", mean_fg_dice, self.current_epoch)
        self.logger.log("epoch_end_timestamps", time(), self.current_epoch)

        self.print_to_log_file(f"Epoch {self.current_epoch} summary")
        self.print_to_log_file("train_loss", round(train_loss, 4))
        self.print_to_log_file("val_loss", round(val_loss, 4))
        self.print_to_log_file("Pseudo dice", [round(value, 4) for value in dices])

        if self.current_epoch != self.epochs - 1:
            self.save_checkpoint(str(self.output_dir / "checkpoint_latest.pth"))
        current_ema = float(self.logger.get_value("ema_fg_dice", -1))
        if self.best_ema is None or current_ema > self.best_ema:
            self.best_ema = current_ema
            self.save_checkpoint(str(self.output_dir / "checkpoint_best.pth"))

        if self.plot_worker is not None:
            self.plot_worker.submit_progress(
                self.logger.get_history_snapshot(),
                str(self.output_dir / "progress.png"),
                total_epochs=self.epochs,
                current_epoch=self.current_epoch,
            )
            self.plot_worker.check_errors()
        else:
            self.logger.plot_progress_png(
                self.output_dir,
                total_epochs=self.epochs,
                current_epoch=self.current_epoch,
            )
        self.current_epoch += 1
        self._write_training_state("running")

    def run_training(self) -> None:
        if self.train_loader is None or self.val_loader is None:
            raise RuntimeError("Trainer is not initialized.")

        for epoch in range(self.current_epoch, self.epochs):
            self.logger.log("epoch_start_timestamps", time(), self.current_epoch)
            self.lr_scheduler.step(self.current_epoch)
            self.logger.log(
                "lrs", float(self.optimizer.param_groups[0]["lr"]), self.current_epoch
            )
            self.print_to_log_file(
                f"Epoch {self.current_epoch} (run {self.current_epoch + 1}/{self.epochs})"
            )
            self.print_to_log_file(
                f"Current learning rate: {self.optimizer.param_groups[0]['lr']:.6f}"
            )

            self.model.train()
            train_outputs = [
                self.train_step(next(self.train_loader))
                for _ in range(int(self.config.num_iterations_per_epoch))
            ]

            self.model.eval()
            with torch.no_grad():
                val_outputs = [
                    self.validation_step(next(self.val_loader))
                    for _ in range(int(self.config.num_val_iterations_per_epoch))
                ]

            self.on_epoch_end(train_outputs, val_outputs)

        self.current_epoch = self.epochs
        self.save_checkpoint(str(self.output_dir / "checkpoint_final.pth"))
        latest = self.output_dir / "checkpoint_latest.pth"
        if latest.exists():
            latest.unlink()
            self.latest_checkpoint_path = None
        self._write_training_state("completed")
        self.print_to_log_file("Training done.")

    def perform_actual_validation(self) -> dict[str, Any]:
        self._set_deep_supervision_enabled(False)
        self.model.eval()
        validation_dir = self.output_dir / "validation"
        validation_dir.mkdir(parents=True, exist_ok=True)

        _, val_keys = self._fold_split()
        case_limit = os.environ.get("BRATS_FINAL_VAL_CASE_LIMIT")
        if case_limit is not None:
            val_keys = val_keys[: int(case_limit)]

        metrics_per_case = []
        regions = self.label_manager.foreground_regions

        for case_id in val_keys:
            data, _, _, properties = self.dataset_class(
                str(self.training_cases_dir), [case_id]
            ).load_case(case_id)
            image = torch.from_numpy(np.asarray(data)).float()
            logits = predict_sliding_window_logits(
                self.model,
                image=image,
                patch_size=BRATS_3D_PATCH_SIZE,
                device=self.device,
                mirror_axes=self.config.inference_mirroring_axes,
            )
            probabilities = (
                self.label_manager.apply_inference_nonlin(logits).cpu().numpy()
            )
            restored_probabilities, segmentation = restore_prediction_to_original_space(
                probabilities,
                properties,
                self.label_manager,
            )

            output_stem = validation_dir / case_id
            if self.export_validation_probabilities:
                np.savez_compressed(
                    str(output_stem) + ".npz", probabilities=restored_probabilities
                )
                with Path(str(output_stem) + ".pkl").open("wb") as handle:
                    pickle.dump(properties, handle)

            write_segmentation_nifti(
                segmentation, properties, Path(str(output_stem) + ".nii.gz")
            )
            flair_image = self.raw_dataset_dir / "imagesTr" / f"{case_id}_0003.nii.gz"
            if flair_image.is_file():
                if self.plot_worker is not None:
                    self.plot_worker.submit_overlay(
                        str(flair_image),
                        segmentation,
                        str(Path(str(output_stem) + "_overlay.png")),
                    )
                else:
                    write_validation_overlay(
                        flair_image,
                        segmentation,
                        Path(str(output_stem) + "_overlay.png"),
                    )

            reference_file = self.gt_segmentations_dir / f"{case_id}.nii.gz"
            reference_segmentation = sitk.GetArrayFromImage(
                sitk.ReadImage(str(reference_file))
            ).astype(np.uint16)
            if flair_image.is_file():
                if self.plot_worker is not None:
                    self.plot_worker.submit_overlay(
                        str(flair_image),
                        reference_segmentation,
                        str(Path(str(output_stem) + "_reference_overlay.png")),
                    )
                else:
                    write_validation_overlay(
                        flair_image,
                        reference_segmentation,
                        Path(str(output_stem) + "_reference_overlay.png"),
                    )
            metrics_per_case.append(
                {
                    "reference_file": str(reference_file),
                    "prediction_file": str(Path(str(output_stem) + ".nii.gz")),
                    "metrics": compute_case_metrics(
                        reference_segmentation, segmentation, regions
                    ),
                }
            )
            self.print_to_log_file(f"Validated {case_id}")

        summary = evaluate_prediction_folder(
            validation_dir,
            gt_dir=self.gt_segmentations_dir,
            output_file=validation_dir / "summary.json",
            num_processes=1,
            chill=True,
        )
        save_summary_json(summary, self.output_dir / "summary.json")
        if self.plot_worker is not None:
            self.close()
        self._set_deep_supervision_enabled(True)
        self.print_to_log_file(
            "Validation complete", summary["foreground_mean"].get("Dice")
        )
        return summary
