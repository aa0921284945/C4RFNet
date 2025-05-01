# -*- coding: utf-8 -*-
"""
LossLogger 2.0
♻️  Cleaner API, English comments, optional smoothing & CSV export.
"""

from __future__ import annotations
import csv
import datetime as dt
from pathlib import Path
from typing import Iterable, Optional
import matplotlib.pyplot as plt


class LossLogger:
    """Lightweight loss tracker + matplotlib exporter."""

    def __init__(self, log_dir: str | Path = "logs") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Histories
        self.total: list[float] = []
        self.global_: list[float] = []
        self.local_: list[float] = []
        self.val: list[float] = []      # may be shorter than others

    # ------------------------------------------------------------------ #
    #                              LOGGING                               #
    # ------------------------------------------------------------------ #
    def log(self,
            total: float,
            global_loss: float,
            local_loss: float,
            val_loss: Optional[float] = None) -> None:
        """Append one epoch of losses."""
        self.total.append(total)
        self.global_.append(global_loss)
        self.local_.append(local_loss)
        if val_loss is not None:
            self.val.append(val_loss)

    # ------------------------------------------------------------------ #
    #                         SAVING & PLOTTING                          #
    # ------------------------------------------------------------------ #
    def _smooth(self, series: Iterable[float], k: int) -> list[float]:
        """Simple moving average with window `k`. `k<=1` returns raw data."""
        if k <= 1:
            return list(series)
        cumsum = [0.0]
        for x in series:
            cumsum.append(cumsum[-1] + x)
        return [
            (cumsum[i + k] - cumsum[i]) / k
            for i in range(len(series) - k + 1)
        ]

    def plot(
        self,
        filename: str | None = None,
        show: bool = False,
        smooth_k: int = 1,
    ) -> Path:
        """Render and save the loss curves. Returns the saved path."""
        if filename is None:
            ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"loss_curve_{ts}.png"
        save_path = self.log_dir / filename

        epochs = range(1, len(self.total) + 1)
        plt.figure(figsize=(10, 6))

        def _plot(y, label):
            if smooth_k > 1:
                y = self._smooth(y, smooth_k)
                xs = range(smooth_k, len(y) + smooth_k)
            else:
                xs = epochs
            plt.plot(xs, y, label=label)

        _plot(self.total,  "Total")
        _plot(self.global_, "Global")
        _plot(self.local_, "Local")

        if len(self.val) == len(self.total):
            _plot(self.val, "Validation")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training / Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        if show:
            plt.show()
        plt.close()
        print(f"[📈] Loss curve saved: {save_path}")
        return save_path

    def to_csv(self,
               filename: str | None = None,
               include_val: bool = True) -> Path:
        """Dump all logged values to a CSV file."""
        if filename is None:
            ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"loss_log_{ts}.csv"
        csv_path = self.log_dir / filename

        headers = ["epoch", "total", "global", "local"]
        rows = zip(
            range(1, len(self.total) + 1),
            self.total, self.global_, self.local_
        )

        if include_val:
            headers.append("val")
            rows = [
                (*row, self.val[i] if i < len(self.val) else float("nan"))
                for i, row in enumerate(rows)
            ]

        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)

        print(f"[💾] Loss log exported: {csv_path}")
        return csv_path
