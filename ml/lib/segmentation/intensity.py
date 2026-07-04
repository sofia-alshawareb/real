"""Intensity construction and GMM helpers."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import brentq
from scipy.stats import norm
from skimage.filters import sobel
from skimage.restoration import denoise_bilateral
from sklearn.mixture import GaussianMixture

from ml.lib.constants import N_GAUSSIANS_BINARY, N_GAUSSIANS_REGIONS


def build_intensity(
    gray: np.ndarray,
    *,
    preprocess: bool,
    illum_sigma: float,
    denoise: bool,
) -> np.ndarray:
    if not preprocess:
        return gray.astype(np.float32)
    illumination = ndi.gaussian_filter(gray, sigma=illum_sigma)
    illumination = np.maximum(illumination, 1e-6)
    corrected = gray / illumination
    corrected = corrected / max(float(np.percentile(corrected, 99.5)), 1e-6)
    corrected = np.clip(corrected, 0.0, 1.0).astype(np.float32)
    if not denoise:
        return corrected
    return denoise_bilateral(
        corrected,
        sigma_color=0.05,
        sigma_spatial=2,
        channel_axis=None,
    ).astype(np.float32)


def normalize01(arr: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(arr)), float(np.max(arr))
    if hi > lo:
        return ((arr - lo) / (hi - lo)).astype(np.float32)
    return np.zeros_like(arr, dtype=np.float32)


def intensity_gradient_map(intensity: np.ndarray) -> np.ndarray:
    return normalize01(sobel(intensity))


def fit_gmm(
    value_map: np.ndarray,
    n_components: int,
    max_samples: int,
    random_state: int,
    pixel_mask: np.ndarray | None = None,
) -> GaussianMixture:
    if pixel_mask is not None:
        values = value_map[pixel_mask.astype(bool)].astype(np.float64).ravel()
    else:
        values = value_map.ravel().astype(np.float64)
    if values.size < n_components:
        raise ValueError(f"Need ≥{n_components} pixels for GMM, got {values.size}.")
    if values.size > max_samples:
        rng = np.random.default_rng(random_state)
        values = values[rng.choice(values.size, size=max_samples, replace=False)]
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        random_state=random_state,
        n_init=5,
        max_iter=300,
    )
    gmm.fit(values.reshape(-1, 1))
    return gmm


def ordered_gmm_params(
    gmm: GaussianMixture,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = gmm.means_.ravel()
    order = np.argsort(means)
    means = means[order]
    weights = gmm.weights_[order]
    variances = np.array([float(gmm.covariances_[i].ravel()[0]) for i in order])
    variances = np.maximum(variances, 1e-12)
    return means, variances, weights


def _component_pdf(x: np.ndarray, weight: float, mean: float, std: float) -> np.ndarray:
    std = max(float(std), 1e-6)
    return weight * norm.pdf(x, loc=mean, scale=std)


def adjacent_intersection(
    mean_lo: float,
    var_lo: float,
    weight_lo: float,
    mean_hi: float,
    var_hi: float,
    weight_hi: float,
) -> float:
    lo, hi = float(mean_lo), float(mean_hi)
    if hi - lo < 1e-6:
        return lo
    std_lo = float(np.sqrt(max(var_lo, 1e-12)))
    std_hi = float(np.sqrt(max(var_hi, 1e-12)))

    def diff(x: float) -> float:
        xv = np.atleast_1d(x)
        return float(
            _component_pdf(xv, weight_lo, mean_lo, std_lo)[0]
            - _component_pdf(xv, weight_hi, mean_hi, std_hi)[0]
        )

    xs = np.linspace(lo, hi, 400)
    vals = np.array([diff(float(x)) for x in xs])
    sign_changes = np.where(np.sign(vals[:-1]) * np.sign(vals[1:]) < 0)[0]
    if len(sign_changes) > 0:
        i = int(sign_changes[0])
        return float(brentq(diff, xs[i], xs[i + 1]))
    return float(xs[int(np.argmin(np.abs(vals)))])


def thresholds_from_adjacent_intersections(
    means: np.ndarray,
    variances: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    thresholds = [
        adjacent_intersection(
            means[i],
            variances[i],
            weights[i],
            means[i + 1],
            variances[i + 1],
            weights[i + 1],
        )
        for i in range(len(means) - 1)
    ]
    thresholds = np.asarray(thresholds, dtype=np.float64)
    for i in range(1, len(thresholds)):
        if thresholds[i] <= thresholds[i - 1]:
            thresholds[i] = min(
                0.999999,
                thresholds[i - 1] + max(1e-4, 0.25 * (means[i + 1] - means[i])),
            )
    return thresholds


def two_gmm_threshold(
    intensity: np.ndarray,
    *,
    max_samples: int,
    random_state: int,
    pixel_mask: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    gmm = fit_gmm(
        intensity,
        N_GAUSSIANS_BINARY,
        max_samples,
        random_state,
        pixel_mask=pixel_mask,
    )
    means, variances, weights = ordered_gmm_params(gmm)
    threshold = adjacent_intersection(
        means[0],
        variances[0],
        weights[0],
        means[1],
        variances[1],
        weights[1],
    )
    return threshold, means, variances, weights


def fit_region_gmm(
    activation: np.ndarray,
    max_samples: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gmm = fit_gmm(activation, N_GAUSSIANS_REGIONS, max_samples, random_state)
    return ordered_gmm_params(gmm)
