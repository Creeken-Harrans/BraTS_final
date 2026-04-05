from __future__ import annotations

import multiprocessing
from queue import Empty
from typing import Any

from validation.src.visualization import (
    render_progress_plot,
    write_validation_overlay,
)


def _plot_worker_main(task_queue, error_queue) -> None:
    while True:
        try:
            task = task_queue.get(timeout=0.2)
        except Empty:
            continue
        if task is None:
            break

        try:
            task_type = task["type"]
            if task_type == "progress":
                render_progress_plot(
                    task["history"],
                    task["output_file"],
                    total_epochs=task.get("total_epochs"),
                    current_epoch=task.get("current_epoch"),
                )
            elif task_type == "overlay":
                write_validation_overlay(
                    task["image_file"],
                    task["segmentation"],
                    task["output_file"],
                )
            else:
                raise RuntimeError(f"Unsupported plot task: {task_type}")
        except Exception as exc:  # pragma: no cover
            error_queue.put({"task": task, "error": repr(exc)})


class AsyncPlotWorker:
    def __init__(self) -> None:
        ctx = multiprocessing.get_context("spawn")
        self.task_queue = ctx.Queue()
        self.error_queue = ctx.Queue()
        self.process = ctx.Process(
            target=_plot_worker_main,
            args=(self.task_queue, self.error_queue),
            daemon=True,
        )
        self.process.start()

    def submit_progress(
        self,
        history: dict[str, list[Any]],
        output_file: str,
        *,
        total_epochs: int | None,
        current_epoch: int | None,
    ) -> None:
        self.task_queue.put(
            {
                "type": "progress",
                "history": history,
                "output_file": output_file,
                "total_epochs": total_epochs,
                "current_epoch": current_epoch,
            }
        )

    def submit_overlay(self, image_file: str, segmentation, output_file: str) -> None:
        self.task_queue.put(
            {
                "type": "overlay",
                "image_file": image_file,
                "segmentation": segmentation,
                "output_file": output_file,
            }
        )

    def check_errors(self) -> None:
        if self.error_queue.empty():
            return
        payload = self.error_queue.get()
        raise RuntimeError(f"Async plot worker failed: {payload['error']}")

    def close(self) -> None:
        if self.process.is_alive():
            self.task_queue.put(None)
            self.process.join(timeout=30)
        self.check_errors()
