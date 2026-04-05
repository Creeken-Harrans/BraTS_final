from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
import seaborn as sns


COLOR_CYCLE = (
    "000000",
    "4363d8",
    "f58231",
    "3cb44b",
    "e6194B",
    "911eb4",
    "ffe119",
    "bfef45",
    "42d4f4",
    "f032e6",
)


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    return tuple(int(hex_color[index : index + 2], 16) for index in (0, 2, 4))


def generate_overlay(
    input_image: np.ndarray,
    segmentation: np.ndarray,
    overlay_intensity: float = 0.6,
) -> np.ndarray:
    image = np.copy(input_image)
    if image.ndim == 2:
        image = np.tile(image[:, :, None], (1, 1, 3))
    elif image.ndim == 3 and image.shape[2] == 1:
        image = np.tile(image, (1, 1, 3))
    elif image.ndim != 3:
        raise RuntimeError(f"Unsupported image shape for overlay: {image.shape}")

    image = image.astype(np.float32)
    image -= image.min()
    image /= max(float(image.max()), 1e-8)
    image *= 255.0

    labels = [label for label in np.sort(pd.unique(segmentation.ravel())) if int(label) > 0]
    mapping = {int(label): index + 1 for index, label in enumerate(labels)}
    for label, color_index in mapping.items():
        image[segmentation == label] += overlay_intensity * np.asarray(
            _hex_to_rgb(COLOR_CYCLE[color_index % len(COLOR_CYCLE)]),
            dtype=np.float32,
        )

    image /= max(float(image.max()), 1e-8)
    image *= 255.0
    return image.astype(np.uint8)


def select_slice_to_plot(segmentation: np.ndarray) -> int:
    foreground_per_slice = (segmentation > 0).sum(axis=(1, 2))
    return int(np.argmax(foreground_per_slice))


def write_validation_overlay(
    image_file: str | Path,
    segmentation: np.ndarray,
    output_file: str | Path,
) -> None:
    image = sitk.GetArrayFromImage(sitk.ReadImage(str(image_file))).astype(np.float32)
    selected_slice = select_slice_to_plot(segmentation)
    overlay = generate_overlay(image[selected_slice], segmentation[selected_slice])
    plt.imsave(str(output_file), overlay)


def render_progress_plot(
    history: dict[str, list],
    output_file: str | Path,
    *,
    total_epochs: int | None = None,
    current_epoch: int | None = None,
) -> None:
    epoch = min(len(values) for values in history.values()) - 1
    if epoch < 0:
        return

    sns.set(font_scale=2.5)
    fig, ax = plt.subplots(1, 1, figsize=(30, 18))
    ax2 = ax.twinx()
    x_values = list(range(epoch + 1))

    ax.plot(
        x_values,
        history["train_losses"][: epoch + 1],
        color="b",
        ls="-",
        label="loss_tr",
        linewidth=4,
    )
    ax.plot(
        x_values,
        history["val_losses"][: epoch + 1],
        color="r",
        ls="-",
        label="loss_val",
        linewidth=4,
    )
    ax2.plot(
        x_values,
        history["mean_fg_dice"][: epoch + 1],
        color="g",
        ls="dotted",
        label="pseudo dice",
        linewidth=3,
    )
    ax2.plot(
        x_values,
        history["ema_fg_dice"][: epoch + 1],
        color="g",
        ls="-",
        label="pseudo dice (mov. avg.)",
        linewidth=4,
    )

    if total_epochs is not None and total_epochs > 0:
        limit = max(total_epochs - 1, 0)
        ax.set_xlim(0, limit)
        ax2.set_xlim(0, limit)
        tick_count = min(total_epochs, 6)
        tick_positions = np.linspace(0, limit, num=tick_count, dtype=int)
        tick_positions = np.unique(tick_positions)
        ax.set_xticks(tick_positions)
    else:
        ax.set_xlim(0, max(epoch, 0))
    if total_epochs is not None and current_epoch is not None:
        ax.set_title(
            f"Training Progress ({current_epoch + 1}/{total_epochs} epochs)",
            pad=30,
            fontsize=42,
        )

    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax2.set_ylabel("pseudo dice")
    ax.margins(x=0)
    ax2.margins(x=0)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.01, 0.995),
        frameon=True,
        borderaxespad=0.6,
    )
    ax2.legend(
        loc="lower right",
        bbox_to_anchor=(0.99, 0.995),
        frameon=True,
        borderaxespad=0.6,
    )
    plt.tight_layout(rect=(0, 0, 1, 1))
    fig.savefig(str(output_file))
    plt.close(fig)
