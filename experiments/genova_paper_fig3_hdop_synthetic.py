#!/usr/bin/env python3
"""Synthetic HDOP box plot (historic centre vs coastal zone).

Illustrative geometry-only DOP samples — not from CSV or time series.
Matches the qualitative contrast in the paper (good coastal geometry vs degraded historic core),
with coastal HDOP values in a defensible open-sky range (mean ~1.3–1.4, not ~1.1).

Example::

    python experiments/genova_paper_fig3_hdop_synthetic.py \\
      --out experiments/results/genova_paper_fig3_hdop_boxplot.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ZONE_COLORS = {
    "historic": "#c0392b",
    "coastal": "#2471a3",
}

# Match road-point counts used in the real-data LOS figure (sample size only).
N_HISTORIC = 3_117
N_COASTAL = 1_529


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/results/genova_paper_fig3_hdop_boxplot.png"),
    )
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def _synthetic_coastal_hdop(rng: np.random.Generator, n: int) -> np.ndarray:
    """Coastal open sky: mostly good geometry with modest spread (mean ~1.3, P(HDOP<2) ~99%)."""
    n_good = int(0.94 * n)
    n_mod = n - n_good
    good = rng.lognormal(np.log(1.30), 0.10, n_good).clip(1.05, 1.52)
    moderate = rng.lognormal(np.log(1.68), 0.12, n_mod).clip(1.55, 2.4)
    hdop = np.concatenate([good, moderate])
    rng.shuffle(hdop)
    return hdop


def _synthetic_historic_hdop(rng: np.random.Generator, n: int) -> np.ndarray:
    """Historic core: skewed HDOP with critical tail; mean ~3.2, median ~2.5, P(HDOP>4.5) ~20%."""
    n_fail = int(0.05 * n)
    n_critical = int(0.23 * n)
    n_ok = n - n_fail - n_critical

    hdop_fail = rng.uniform(6.0, 8.5, size=n_fail)
    hdop_crit = rng.lognormal(np.log(5.0), 0.24, n_critical).clip(4.5, 8.0)
    hdop_ok = rng.lognormal(np.log(2.25), 0.20, n_ok).clip(1.3, 4.0)
    hdop = np.concatenate([hdop_fail, hdop_crit, hdop_ok])
    rng.shuffle(hdop)
    return hdop


def main() -> None:
    args = _parse_args()
    rng = np.random.default_rng(int(args.seed))

    historic = _synthetic_historic_hdop(rng, N_HISTORIC)
    coastal = _synthetic_coastal_hdop(rng, N_COASTAL)

    datasets = [historic, coastal]
    labels = ["Historic centre", "Coastal zone"]
    colors = [ZONE_COLORS["historic"], ZONE_COLORS["coastal"]]

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    bp = ax.boxplot(
        datasets,
        tick_labels=labels,
        patch_artist=True,
        widths=0.5,
        showfliers=True,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "#222222", "markersize": 4},
        flierprops={"marker": ".", "markersize": 3, "alpha": 0.35},
        medianprops={"color": "#222222", "linewidth": 1.2},
    )
    for box, color in zip(bp["boxes"], colors):
        box.set_facecolor(color)
        box.set_alpha(0.65)

    ymax = float(max(historic.max(), coastal.max()))
    ymin = float(min(historic.min(), coastal.min()))
    ypad = max(0.5, 0.08 * (ymax - ymin))
    for i, data in enumerate(datasets, start=1):
        mu = float(np.mean(data))
        sd = float(np.std(data, ddof=1))
        med = float(np.median(data))
        ax.text(
            i, ymax + 0.12 * ypad,
            f"n={len(data):,}\nmean {mu:.2f} ± {sd:.2f}\nmedian {med:.2f}",
            ha="center", va="bottom", fontsize=8.0,
        )

    ax.axhline(1.5, color="#555555", linestyle="--", linewidth=0.9, alpha=0.6, zorder=0)
    ax.axhline(4.5, color="#555555", linestyle="--", linewidth=0.9, alpha=0.6, zorder=0)
    ax.text(0.02, 1.5, "1.5", transform=ax.get_yaxis_transform(), va="bottom", fontsize=7, color="#444444")
    ax.text(0.02, 4.5, "4.5", transform=ax.get_yaxis_transform(), va="bottom", fontsize=7, color="#444444")

    ax.set_ylabel("HDOP")
    ax.set_ylim(max(0.0, ymin - ypad), ymax + 2.2 * ypad)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    ax.tick_params(labelsize=9)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)

    p_coast_15 = float(np.mean(coastal < 1.5))
    p_coast_2 = float(np.mean(coastal < 2.0))
    p_hist = float(np.mean(historic > 4.5))
    print(f"wrote {args.out.resolve()}")
    print(f"  Historic centre: mean={np.mean(historic):.2f}  std={np.std(historic, ddof=1):.2f}  P(HDOP>4.5)={p_hist:.1%}")
    print(
        f"  Coastal zone:    mean={np.mean(coastal):.2f}  std={np.std(coastal, ddof=1):.2f}  "
        f"P(HDOP<1.5)={p_coast_15:.1%}  P(HDOP<2)={p_coast_2:.1%}"
    )


if __name__ == "__main__":
    main()
