#!/usr/bin/env python3
"""Plot geometric vs KLT GT confusion matrix (G+C, -18 s offset)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot KLT geometric confusion matrix PNG")
    p.add_argument(
        "--out-png",
        type=Path,
        default=Path("experiments/results/klt_geom_confusion_matrix.png"),
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    # Verified aggregate (G+C, offset -18 s, 8 KLT sequences with GT)
    # Rows: ground truth, Columns: geometric prediction
    cm = np.array(
        [
            [29550, 227],   # GT LOS
            [10017, 4326],  # GT NLOS
        ],
        dtype=np.int64,
    )

    fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=150)
    im = ax.imshow(cm, cmap="Blues")

    labels_x = ["Predicted LOS", "Predicted NLOS"]
    labels_y = ["GT LOS", "GT NLOS"]
    ax.set_xticks([0, 1], labels=labels_x, fontsize=11)
    ax.set_yticks([0, 1], labels=labels_y, fontsize=11)
    ax.set_xlabel("Geometric model", fontsize=12, labelpad=8)
    ax.set_ylabel("Ground truth", fontsize=12, labelpad=8)

    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            val = int(cm[i, j])
            pct = 100.0 * val / cm.sum()
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(
                j,
                i,
                f"{val:,}\n({pct:.1f}%)",
                ha="center",
                va="center",
                fontsize=11,
                color=color,
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Count", fontsize=10)

    fig.tight_layout()
    out = args.out_png.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
