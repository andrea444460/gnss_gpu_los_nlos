#!/usr/bin/env python3
"""Synthetic Figure 3: visible-satellite count and HDOP distributions (centro vs coast).

Illustrative data only — not from field CSVs. Matches the qualitative claims in the paper text.

Example::

    python experiments/genova_paper_fig3_dop_synthetic.py \\
      --out experiments/results/genova_paper_fig3_dop_distributions.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ZONE_COLORS = {
    "coastal": "#2471a3",
    "historic": "#c0392b",
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/results/genova_paper_fig3_dop_distributions.png"),
    )
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def _synthetic_coastal(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Coastal: many visible sats, HDOP tightly peaked near 1.1."""
    m = rng.normal(17.4, 1.1, n).clip(8, 22)
    hdop = rng.lognormal(mean=np.log(1.08), sigma=0.07, size=n).clip(0.85, 1.45)
    return m, hdop


def _synthetic_historic(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Historic core: failure mass (M<4) and heavy HDOP tail beyond 4.5."""
    n_fail = int(0.14 * n)
    n_critical = int(0.22 * n)
    n_ok = n - n_fail - n_critical

    m_fail = rng.integers(0, 4, size=n_fail)
    m_crit = rng.normal(11.5, 2.0, n_critical).clip(4, 16)
    m_ok = rng.normal(14.8, 1.6, n_ok).clip(6, 20)
    m = np.concatenate([m_fail, m_crit, m_ok])
    rng.shuffle(m)

    hdop_fail = rng.uniform(8.0, 18.0, size=n_fail)
    hdop_crit = rng.lognormal(np.log(5.8), 0.35, n_critical).clip(4.5, 14.0)
    hdop_ok = rng.lognormal(np.log(2.0), 0.25, n_ok).clip(1.2, 4.2)
    hdop = np.concatenate([hdop_fail, hdop_crit, hdop_ok])
    rng.shuffle(hdop)

    return m, hdop


def _hist_panel(
    ax: plt.Axes,
    data: np.ndarray,
    *,
    color: str,
    xlabel: str,
    vlines: list[tuple[float, str]] | None = None,
    bins: np.ndarray | None = None,
    xlim: tuple[float, float] | None = None,
    density: bool = True,
) -> None:
    ax.hist(
        data,
        bins=bins if bins is not None else 30,
        density=density,
        color=color,
        alpha=0.72,
        edgecolor="white",
        linewidth=0.4,
    )
    if vlines:
        for x, lbl in vlines:
            ax.axvline(x, color="#333333", linestyle="--", linewidth=1.0, alpha=0.75)
            ax.text(x, ax.get_ylim()[1] * 0.92, lbl, rotation=90, va="top", ha="right", fontsize=7)
    ax.set_xlabel(xlabel, fontsize=9)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    ax.tick_params(labelsize=8)


def main() -> None:
    args = _parse_args()
    rng = np.default_rng(int(args.seed))
    n = 24_000  # 24 h × 1000 road points / h (illustrative)

    m_coast, hdop_coast = _synthetic_coastal(rng, n)
    m_hist, hdop_hist = _synthetic_historic(rng, n)

    p_coast = float(np.mean(hdop_coast < 1.5))
    p_hist_fail = float(np.mean(m_hist < 4))
    p_hist_crit = float(np.mean(hdop_hist > 4.5))

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), sharex="col")

    m_bins = np.arange(-0.5, 23.5, 1.0)
    hdop_bins = np.linspace(0.5, 16.0, 48)

    _hist_panel(
        axes[0, 0], m_coast, color=ZONE_COLORS["coastal"],
        xlabel="Visible satellites $M$", bins=m_bins, xlim=(-0.5, 22.5),
        vlines=[(4.0, "$M=4$")],
    )
    _hist_panel(
        axes[0, 1], m_hist, color=ZONE_COLORS["historic"],
        xlabel="Visible satellites $M$", bins=m_bins, xlim=(-0.5, 22.5),
        vlines=[(4.0, "$M=4$")],
    )
    _hist_panel(
        axes[1, 0], hdop_coast, color=ZONE_COLORS["coastal"],
        xlabel="HDOP", bins=hdop_bins, xlim=(0.5, 10.0),
        vlines=[(1.5, "1.5"), (4.5, "4.5")],
    )
    _hist_panel(
        axes[1, 1], hdop_hist, color=ZONE_COLORS["historic"],
        xlabel="HDOP", bins=hdop_bins, xlim=(0.5, 10.0),
        vlines=[(1.5, "1.5"), (4.5, "4.5")],
    )

    axes[0, 0].set_ylabel("Coastal zone\n density", fontsize=9)
    axes[0, 1].set_ylabel("Historic centre\n density", fontsize=9)
    axes[1, 0].set_ylabel("Density", fontsize=9)
    axes[1, 1].set_ylabel("Density", fontsize=9)

    for ax in axes[0]:
        ax.text(
            0.03, 0.97, f"$P(M<4)={float(np.mean(m_coast < 4)):.1%}$" if ax is axes[0, 0]
            else f"$P(M<4)={p_hist_fail:.1%}$",
            transform=ax.transAxes, va="top", fontsize=7.5,
        )
    axes[1, 0].text(
        0.03, 0.97, f"$P(\\mathrm{{HDOP}}<1.5)={p_coast:.1%}$",
        transform=axes[1, 0].transAxes, va="top", fontsize=7.5,
    )
    axes[1, 1].text(
        0.03, 0.97, f"$P(\\mathrm{{HDOP}}>4.5)={p_hist_crit:.1%}$",
        transform=axes[1, 1].transAxes, va="top", fontsize=7.5,
    )

    fig.tight_layout(w_pad=1.2, h_pad=1.0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {args.out.resolve()}")
    print(f"  coastal: P(HDOP<1.5)={p_coast:.1%}, peak HDOP bin ~1.1")
    print(f"  historic: P(M<4)={p_hist_fail:.1%}, P(HDOP>4.5)={p_hist_crit:.1%}")


if __name__ == "__main__":
    main()
