from __future__ import annotations

from pathlib import Path
from copy import deepcopy
from typing import Any

from validation.src.visualization import render_progress_plot


class TrainingLogger:
    def __init__(self) -> None:
        self.history: dict[str, list[Any]] = {
            "mean_fg_dice": [],
            "ema_fg_dice": [],
            "dice_per_class_or_region": [],
            "train_losses": [],
            "val_losses": [],
            "lrs": [],
            "epoch_start_timestamps": [],
            "epoch_end_timestamps": [],
        }

    def log(self, key: str, value: Any, epoch: int) -> None:
        series = self.history[key]
        if len(series) < epoch + 1:
            series.append(value)
        else:
            series[epoch] = value

        if key == "mean_fg_dice":
            if len(self.history["ema_fg_dice"]) == 0:
                ema = float(value)
            else:
                previous = float(self.history["ema_fg_dice"][epoch - 1])
                ema = previous * 0.9 + 0.1 * float(value)
            self.log("ema_fg_dice", ema, epoch)

    def get_value(self, key: str, step: int | None) -> Any:
        values = self.history[key]
        return values if step is None else values[step]

    def get_num_logged_epochs(self) -> int:
        return min(len(values) for values in self.history.values())

    def get_checkpoint(self) -> dict[str, list[Any]]:
        return self.history

    def load_checkpoint(self, checkpoint: dict[str, list[Any]]) -> None:
        self.history = checkpoint

    def get_history_snapshot(self) -> dict[str, list[Any]]:
        return deepcopy(self.history)

    def plot_progress_png(
        self,
        output_folder: str | Path,
        total_epochs: int | None = None,
        current_epoch: int | None = None,
    ) -> None:
        render_progress_plot(
            self.history,
            Path(output_folder) / "progress.png",
            total_epochs=total_epochs,
            current_epoch=current_epoch,
        )
