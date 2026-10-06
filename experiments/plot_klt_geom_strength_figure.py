#!/usr/bin/env python3
"""Paper figure highlighting geometric model strengths on KLT (G+C, -18 s)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot KLT geometric strength figure")
    p.add_argument(
        "--out-png",
        type=Path,
        default=Path("experiments/results/klt_geom_nlos_reliability.png"),
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    # Verified aggregate (G+C, offset -18 s)
    tp, tn, fp, fn = 29550, 4326, 10017, 227
    nlos_prec = tn / (tn + fn)
    los_rec = tp / (tp + fn)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=150)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.22, top=0.88, wspace=0.32)

    # Panel A: when geometric predicts NLOS, how often is GT NLOS?
    ax = axes[0]
    correct = tn
    wrong = fn
    total = correct + wrong
    ax.barh(
        [0],
        [correct],
        left=[0],
        height=0.5,
        color="#2166ac",
        edgecolor="white",
        linewidth=1.2,
        label=f"True NLOS: {correct:,} ({100 * correct / total:.1f}%)",
    )
    ax.barh(
        [0],
        [wrong],
        left=[correct],
        height=0.5,
        color="#d6604d",
        edgecolor="white",
        linewidth=1.2,
        label=f"False NLOS: {wrong:,} ({100 * wrong / total:.1f}%)",
    )
    ax.set_xlim(0, total * 1.02)
    ax.set_ylim(-0.65, 0.65)
    ax.set_yticks([])
    ax.set_xlabel("Count", fontsize=11)
    ax.set_title("When geometric model predicts NLOS", fontsize=11, fontweight="bold", pad=10)
    ax.text(
        correct / 2,
        0,
        f"True NLOS\n{correct:,}",
        ha="center",
        va="center",
        fontsize=10,
        color="white",
        fontweight="bold",
    )
    # Small slice: label outside the bar so it stays readable.
    ax.annotate(
        f"False NLOS\n{wrong:,} ({100 * wrong / total:.1f}%)",
        xy=(correct + wrong, 0),
        xytext=(total * 0.72, 0.42),
        fontsize=9,
        fontweight="bold",
        color="#b2182b",
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="-|>", color="#b2182b", lw=1.0, shrinkA=2, shrinkB=2),
    )
    ax.text(0.0, -0.52, f"n = {total:,} predictions", transform=ax.transAxes, fontsize=9, ha="left", color="#444444")
    legend_handles = [
        mpatches.Patch(color="#2166ac", label=f"True NLOS ({100 * correct / total:.1f}%)"),
        mpatches.Patch(color="#d6604d", label=f"False NLOS ({100 * wrong / total:.1f}%)"),
    ]
    ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=9, frameon=False)

    # Panel B: two headline metrics where the model is strong
    ax = axes[1]
    metrics = ["NLOS\nprecision", "LOS\nrecall"]
    values = [nlos_prec, los_rec]
    colors = ["#2166ac", "#4393c3"]
    x = np.arange(len(metrics))
    bars = ax.bar(x, values, color=colors, width=0.52, edgecolor="white", linewidth=1.2)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_xticks(x, metrics, fontsize=10)
    ax.axhline(0.9, color="#888888", linestyle="--", linewidth=0.8, alpha=0.7)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.03,
            f"{100 * val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    out = args.out_png.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="white", pad_inches=0.15)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
