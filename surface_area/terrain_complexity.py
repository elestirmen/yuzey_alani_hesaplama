"""Local terrain complexity descriptors for analytic benchmark rasters."""

from __future__ import annotations

import math

import numpy as np


def compute_complexity_descriptors(
    z: np.ndarray,
    *,
    dx: float,
    dy: float,
    window_size: int = 5,
) -> dict[str, np.ndarray]:
    """Compute a compact set of local complexity descriptors."""
    z_arr = np.asarray(z, dtype=np.float64)
    if z_arr.ndim != 2:
        raise ValueError("z must be a 2D array")
    if z_arr.shape[0] < 3 or z_arr.shape[1] < 3:
        raise ValueError("z must be at least 3x3")
    if float(dx) <= 0 or float(dy) <= 0:
        raise ValueError("dx and dy must be > 0")

    size = int(window_size)
    if size < 3:
        raise ValueError("window_size must be >= 3")
    if size % 2 == 0:
        size += 1

    try:
        from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter, uniform_filter
    except Exception as e:
        raise RuntimeError("scipy is required for terrain complexity descriptors") from e

    grad_y, grad_x = np.gradient(z_arr, float(dy), float(dx), edge_order=2)
    slope_magnitude = np.hypot(grad_x, grad_y)

    grad_x_y, grad_x_x = np.gradient(grad_x, float(dy), float(dx), edge_order=2)
    grad_y_y, _grad_y_x = np.gradient(grad_y, float(dy), float(dx), edge_order=2)
    curvature_magnitude = np.sqrt(grad_x_x * grad_x_x + 2.0 * grad_x_y * grad_x_y + grad_y_y * grad_y_y)

    local_relief_range = maximum_filter(z_arr, size=size, mode="nearest") - minimum_filter(z_arr, size=size, mode="nearest")

    slope_mean = uniform_filter(slope_magnitude, size=size, mode="nearest")
    slope_mean_sq = uniform_filter(slope_magnitude * slope_magnitude, size=size, mode="nearest")
    slope_variance = np.maximum(slope_mean_sq - slope_mean * slope_mean, 0.0)

    sigma = max(0.85, size / 3.0)
    trend = gaussian_filter(z_arr, sigma=sigma, mode="nearest")
    roughness_residual = np.abs(z_arr - trend)

    return {
        "slope_magnitude": slope_magnitude.astype(np.float64, copy=False),
        "curvature_magnitude": curvature_magnitude.astype(np.float64, copy=False),
        "local_relief_range": local_relief_range.astype(np.float64, copy=False),
        "slope_variance": slope_variance.astype(np.float64, copy=False),
        "roughness_residual": roughness_residual.astype(np.float64, copy=False),
    }


def summarize_complexity_descriptors(
    descriptors: dict[str, np.ndarray],
    *,
    mask: np.ndarray | None = None,
) -> dict[str, dict[str, float | int]]:
    """Summarize descriptor rasters with robust scalar statistics."""
    if mask is not None:
        mask_arr = np.asarray(mask, dtype=bool)
    else:
        mask_arr = None

    summary: dict[str, dict[str, float | int]] = {}
    for name, values in descriptors.items():
        arr = np.asarray(values, dtype=np.float64)
        valid = np.isfinite(arr)
        if mask_arr is not None:
            if mask_arr.shape != arr.shape:
                raise ValueError(f"Mask shape mismatch for descriptor {name!r}")
            valid &= ~mask_arr

        if not np.any(valid):
            summary[name] = {
                "count": 0,
                "min": math.nan,
                "max": math.nan,
                "mean": math.nan,
                "std": math.nan,
                "p90": math.nan,
                "p95": math.nan,
            }
            continue

        vv = arr[valid]
        summary[name] = {
            "count": int(vv.size),
            "min": float(vv.min()),
            "max": float(vv.max()),
            "mean": float(vv.mean(dtype=np.float64)),
            "std": float(vv.std(dtype=np.float64)),
            "p90": float(np.percentile(vv, 90.0)),
            "p95": float(np.percentile(vv, 95.0)),
        }
    return summary
