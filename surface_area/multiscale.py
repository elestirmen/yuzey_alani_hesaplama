"""Multiscale (low/high frequency) surface area decomposition."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
from scipy.ndimage import gaussian_filter

from surface_area.io import block_window_count, gaussian_overlap_pixels, iter_block_windows, read_window_float32
from surface_area.methods import AreaResult, SlopeMethod, gradient_multiplier_cell_areas

ProgressFn = Callable[[str, int, int], None]


@dataclass(frozen=True, slots=True)
class MultiscaleAreaResult:
    sigma_m: float
    a_topo: float
    a_micro: float
    a_total: float
    micro_ratio: float
    valid_cells: int


def _lowpass_gaussian_normalized(
    z: np.ndarray,
    valid: np.ndarray,
    *,
    sigma_px_row: float,
    sigma_px_col: float,
    truncate: float = 4.0,
) -> np.ndarray:
    """Low-pass filter with nodata-aware normalized convolution."""
    if sigma_px_row <= 0 or sigma_px_col <= 0:
        return z.astype(np.float64, copy=False)

    z0 = np.where(valid, z.astype(np.float64, copy=False), 0.0)
    w0 = valid.astype(np.float64, copy=False)

    # Note: gaussian_filter sigma order is (rows, cols)
    zf = gaussian_filter(z0, sigma=(sigma_px_row, sigma_px_col), mode="nearest", truncate=truncate)
    wf = gaussian_filter(w0, sigma=(sigma_px_row, sigma_px_col), mode="nearest", truncate=truncate)

    with np.errstate(divide="ignore", invalid="ignore"):
        zL = np.where(wf > 0.0, zf / wf, np.nan)
    return zL


def compute_area_multiscale(
    z: np.ndarray,
    dx: float,
    dy: float,
    valid: np.ndarray,
    *,
    base_method: SlopeMethod = "horn",
    sigma_m_list: Iterable[float],
    truncate: float = 4.0,
) -> list[MultiscaleAreaResult]:
    """Compute multiscale decomposition on an in-memory array."""
    a_total_res = AreaResult(*compute_total_gradient(z, dx, dy, valid, method=base_method))
    a_total = float(a_total_res.a3d)
    n_total = int(a_total_res.valid_cells)

    out: list[MultiscaleAreaResult] = []
    for sigma_m in sigma_m_list:
        sigma_m_f = float(sigma_m)
        if sigma_m_f <= 0:
            raise ValueError(f"sigma_m must be > 0, got {sigma_m}")

        zL = _lowpass_gaussian_normalized(
            z,
            valid,
            sigma_px_row=sigma_m_f / float(dy),
            sigma_px_col=sigma_m_f / float(dx),
            truncate=truncate,
        )
        a_topo_res = AreaResult(*compute_total_gradient(zL, dx, dy, valid, method=base_method))
        a_topo = float(a_topo_res.a3d)
        a_micro = a_total - a_topo
        micro_ratio = float(a_micro / a_total) if a_total > 0 else float("nan")
        out.append(
            MultiscaleAreaResult(
                sigma_m=sigma_m_f,
                a_topo=a_topo,
                a_micro=a_micro,
                a_total=a_total,
                micro_ratio=micro_ratio,
                valid_cells=n_total,
            )
        )
    return out


def compute_total_gradient(
    z: np.ndarray, dx: float, dy: float, valid: np.ndarray, *, method: SlopeMethod
) -> tuple[float, int]:
    areas, cell_valid = gradient_multiplier_cell_areas(z, dx, dy, valid, method=method)
    return float(areas[cell_valid].sum(dtype=np.float64)), int(cell_valid.sum())


def compute_multiscale_on_raster(
    raster_path: str,
    *,
    nodata: float | None,
    base_method: SlopeMethod,
    sigma_m_list: list[float],
    truncate: float = 4.0,
    a_total: AreaResult | None = None,
    progress: ProgressFn | None = None,
) -> list[MultiscaleAreaResult]:
    """Blockwise multiscale decomposition for large rasters.

    - Low-pass is computed with nodata-aware normalized convolution.
    - A_topo and A_total are computed with gradient-multiplier (Horn or ZT).
    - Validity for A_topo uses the *original* valid-mask stencil to avoid counting
      nodata-influenced areas.
    """
    import rasterio  # local import

    if not sigma_m_list:
        raise ValueError("sigma_m_list is empty")

    sigma_m_list_f = [float(s) for s in sigma_m_list]
    if any(s <= 0 for s in sigma_m_list_f):
        raise ValueError(f"All sigma_m must be > 0, got: {sigma_m_list}")

    # Accumulators per sigma.
    acc_topo: dict[float, float] = {s: 0.0 for s in sigma_m_list_f}
    acc_n: dict[float, int] = {s: 0 for s in sigma_m_list_f}

    with rasterio.open(raster_path) as ds:
        dx = float(abs(ds.transform.a))
        dy = float(abs(ds.transform.e))
        if dx <= 0 or dy <= 0:
            raise ValueError(f"Invalid pixel sizes from transform: dx={dx}, dy={dy}")

        total_blocks = block_window_count(ds)

        # Compute a_total if not provided (single-pass).
        if a_total is None:
            total_a3d = 0.0
            total_n = 0
            block_i = 0
            for w in iter_block_windows(ds):
                block_i += 1
                z, valid, inner = read_window_float32(ds, w, nodata=nodata, overlap=1)
                areas, cell_valid = gradient_multiplier_cell_areas(z, dx, dy, valid, method=base_method)
                areas_in = areas[inner]
                valid_in = cell_valid[inner]
                total_a3d += float(areas_in[valid_in].sum(dtype=np.float64))
                total_n += int(valid_in.sum())
                if progress is not None:
                    progress("multiscale_total", block_i, total_blocks)
            a_total = AreaResult(a3d=total_a3d, valid_cells=total_n)

        # Choose an overlap large enough for the largest sigma and the gradient stencil.
        max_sigma_px_row = max(s / dy for s in sigma_m_list_f)
        max_sigma_px_col = max(s / dx for s in sigma_m_list_f)
        pad = max(
            gaussian_overlap_pixels(max_sigma_px_row, truncate=truncate),
            gaussian_overlap_pixels(max_sigma_px_col, truncate=truncate),
        )
        overlap = pad + 1  # +1 for gradient stencil

        total_units = int(total_blocks) * int(len(sigma_m_list_f))
        unit_i = 0

        for w in iter_block_windows(ds):
            z, valid, inner = read_window_float32(ds, w, nodata=nodata, overlap=overlap)

            # Precompute arrays for normalized convolution once per block.
            z0 = np.where(valid, z.astype(np.float64, copy=False), 0.0)
            w0 = valid.astype(np.float64, copy=False)

            for sigma_m in sigma_m_list_f:
                sigma_px_row = sigma_m / dy
                sigma_px_col = sigma_m / dx

                zf = gaussian_filter(
                    z0, sigma=(sigma_px_row, sigma_px_col), mode="nearest", truncate=truncate
                )
                wf = gaussian_filter(
                    w0, sigma=(sigma_px_row, sigma_px_col), mode="nearest", truncate=truncate
                )
                with np.errstate(divide="ignore", invalid="ignore"):
                    zL = np.where(wf > 0.0, zf / wf, np.nan)

                areas, cell_valid = gradient_multiplier_cell_areas(zL, dx, dy, valid, method=base_method)
                areas_in = areas[inner]
                valid_in = cell_valid[inner]
                acc_topo[sigma_m] += float(areas_in[valid_in].sum(dtype=np.float64))
                acc_n[sigma_m] += int(valid_in.sum())
                unit_i += 1
                if progress is not None:
                    progress("multiscale", unit_i, total_units)

    assert a_total is not None
    out: list[MultiscaleAreaResult] = []
    for sigma_m in sigma_m_list_f:
        a_topo = float(acc_topo[sigma_m])
        a_total_val = float(a_total.a3d)
        a_micro = a_total_val - a_topo
        micro_ratio = float(a_micro / a_total_val) if a_total_val > 0 else float("nan")
        out.append(
            MultiscaleAreaResult(
                sigma_m=sigma_m,
                a_topo=a_topo,
                a_micro=a_micro,
                a_total=a_total_val,
                micro_ratio=micro_ratio,
                valid_cells=int(acc_n[sigma_m]),
            )
        )
    return out
