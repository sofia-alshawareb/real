"""Diagnostic plots for 2-GMM threshold selection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import norm


def save_two_gmm_histogram(
    value_map: np.ndarray,
    fit_mask: np.ndarray,
    *,
    means: np.ndarray,
    variances: np.ndarray,
    weights: np.ndarray,
    threshold: float,
    dest_path: Path | str,
    talc_side: str = "high",
    unbiased_threshold: float | None = None,
    max_hist_samples: int = 100_000,
    random_state: int = 0,
    bins: int = 64,
    xlabel: str = "value",
    title: str = "Talc refine: 2-GMM",
    low_band_label: str = "low → background",
    high_band_label: str = "high → talc",
) -> dict[str, Any]:
    """Save histogram of fit-mask samples with fitted Gaussians and threshold."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fit = fit_mask.astype(bool)
    values = value_map[fit].astype(np.float64).ravel()
    if values.size == 0:
        raise ValueError("no samples inside fit mask")

    if values.size > max_hist_samples:
        rng = np.random.default_rng(random_state)
        values = values[rng.choice(values.size, size=max_hist_samples, replace=False)]

    means = np.asarray(means, dtype=np.float64).ravel()
    variances = np.maximum(np.asarray(variances, dtype=np.float64).ravel(), 1e-12)
    weights = np.asarray(weights, dtype=np.float64).ravel()
    stds = np.sqrt(variances)

    lo = float(min(values.min(), means.min() - 3 * stds.max()))
    hi = float(max(values.max(), means.max() + 3 * stds.max()))
    if hi <= lo:
        hi = lo + 1e-3
    xs = np.linspace(lo, hi, 400)

    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    ax.hist(
        values,
        bins=bins,
        density=True,
        alpha=0.45,
        color="#9e9e9e",
        edgecolor="white",
        linewidth=0.4,
        label="fit samples",
    )

    if talc_side == "low":
        comp_labels = ("low → talc", "high → background")
    else:
        comp_labels = (low_band_label, high_band_label)
    colors = ("#1f77b4", "#ff7f0e")
    for mean, std, weight, color, label in zip(
        means, stds, weights, colors, comp_labels, strict=True
    ):
        pdf = weight * norm.pdf(xs, loc=mean, scale=std)
        ax.plot(xs, pdf, color=color, linewidth=2.0, label=f"GMM {label}")

    ax.axvline(
        threshold,
        color="#d62728",
        linestyle="--",
        linewidth=2.0,
        label=f"threshold = {threshold:.4f}",
    )
    if unbiased_threshold is not None:
        ax.axvline(
            unbiased_threshold,
            color="#9467bd",
            linestyle=":",
            linewidth=1.5,
            label=f"unbiased = {unbiased_threshold:.4f}",
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(dest, bbox_inches="tight")
    plt.close(fig)

    return {
        "histogram_path": str(dest.resolve()),
        "n_hist_samples": int(values.size),
        "threshold": float(threshold),
        "unbiased_threshold": float(unbiased_threshold)
        if unbiased_threshold is not None
        else None,
        "talc_side": talc_side,
        "fit_value_min": float(values.min()),
        "fit_value_max": float(values.max()),
    }


def save_two_gmm_activation_histogram(
    activation: np.ndarray,
    fit_mask: np.ndarray,
    *,
    means: np.ndarray,
    variances: np.ndarray,
    weights: np.ndarray,
    threshold: float,
    dest_path: Path | str,
    talc_side: str = "low",
    unbiased_threshold: float | None = None,
    min_activation: float | None = None,
    max_hist_samples: int = 100_000,
    random_state: int = 0,
    bins: int = 64,
) -> dict[str, Any]:
    """Legacy wrapper for block-1 activation histograms."""
    fit = fit_mask.astype(bool)
    if min_activation is not None:
        fit &= activation >= float(min_activation)
    return save_two_gmm_histogram(
        activation,
        fit,
        means=means,
        variances=variances,
        weights=weights,
        threshold=threshold,
        dest_path=dest_path,
        talc_side=talc_side,
        unbiased_threshold=unbiased_threshold,
        max_hist_samples=max_hist_samples,
        random_state=random_state,
        bins=bins,
        xlabel="block-1 activation",
        title="Talc refine: 2-GMM on interior activation"
        + (f" (fit ≥ {min_activation:g})" if min_activation is not None else ""),
        low_band_label="low activation → talc",
        high_band_label="high activation → background",
    )
