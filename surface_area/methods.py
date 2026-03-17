"""Surface area estimators.

All computations assume DEM values represent elevations at cell centers.
Cell sizes are given by (dx, dy) in meters (or dataset linear units).

Validity masking:
- `valid` is a boolean mask where True indicates a valid DEM sample.
- For stencil-based methods (3x3), a cell is counted only if the full stencil is valid.
- For corner-based methods (TIN / bilinear patch), corners are derived only when all
  contributing center cells are valid (count==4). This excludes a 1-cell border and
  nodata-adjacent cells.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Literal

import numpy as np
try:
    import numba
    from numba import njit
except Exception:  # pragma: no cover - optional dependency
    numba = None
    njit = None

from surface_area.io import iter_block_windows, read_window_float32

ProgressFn = Callable[[str, int, int], None]

NUMBA_AVAILABLE = numba is not None


def _optional_njit(*args, **kwargs):
    if njit is None:
        def decorator(fn):
            return fn

        return decorator
    return njit(*args, **kwargs)


SlopeMethod = Literal["horn", "zt"]


@dataclass(frozen=True, slots=True)
class AreaResult:
    a3d: float
    valid_cells: int
    # Optional diagnostics for adaptive_bilinear_patch_integral.
    adaptive_avg_level: float | None = None
    adaptive_max_level_used: int | None = None
    adaptive_refined_cell_fraction: float | None = None
    adaptive_total_subcells_evaluated: int | None = None
    # Optional diagnostics for sector_adaptive_jenness_integral.
    sector_jenness_avg_level: float | None = None
    sector_jenness_max_level_used: int | None = None
    sector_jenness_refined_fraction: float | None = None


_SUPPORTED_METHODS = {
    "jenness_window_8tri",
    "sector_adaptive_jenness_integral",
    "tin_2tri_cell",
    "gradient_multiplier",
    "bilinear_patch_integral",
    "adaptive_bilinear_patch_integral",
}


@dataclass(frozen=True, slots=True)
class _MethodComputeJob:
    raster_path: str
    nodata: float | None
    windows: tuple[tuple[int, int, int, int], ...]
    dx: float
    dy: float
    methods: tuple[str, ...]
    jenness_weight: float
    slope_method: SlopeMethod
    integral_N: int
    adaptive_rel_tol: float
    adaptive_abs_tol: float
    adaptive_max_level: int
    adaptive_min_N: int
    adaptive_roughness_fastpath: bool
    adaptive_roughness_threshold: float | None
    sector_jenness_rel_tol: float
    sector_jenness_abs_tol: float
    sector_jenness_max_level: int
    sector_jenness_min_samples: int
    include_timings: bool


@dataclass(frozen=True, slots=True)
class _MethodComputeChunkResult:
    acc_a3d: dict[str, float]
    acc_n: dict[str, int]
    acc_t: dict[str, float]
    ad_level_sum: int
    ad_refined: int
    ad_max_level: int
    ad_subcells: int
    sj_level_sum: int
    sj_refined: int
    sj_max_level: int
    blocks_done: int


def _triangle_area_heron(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    s = (a + b + c) * 0.5
    # Heron is numerically fragile; clamp to 0 for tiny negatives.
    v = s * (s - a) * (s - b) * (s - c)
    return np.sqrt(np.maximum(v, 0.0))


def _stencil_all9(valid: np.ndarray) -> np.ndarray:
    """Return mask for center cells where full 3x3 neighborhood is valid."""
    c = valid[1:-1, 1:-1]
    return (
        c
        & valid[:-2, 1:-1]
        & valid[:-2, 2:]
        & valid[1:-1, 2:]
        & valid[2:, 2:]
        & valid[2:, 1:-1]
        & valid[2:, :-2]
        & valid[1:-1, :-2]
        & valid[:-2, :-2]
    )


def _stencil_cross(valid: np.ndarray) -> np.ndarray:
    """Return mask for center cells where N,S,E,W and center are valid."""
    c = valid[1:-1, 1:-1]
    return c & valid[:-2, 1:-1] & valid[2:, 1:-1] & valid[1:-1, :-2] & valid[1:-1, 2:]


_QUADRATIC_FIT_MAX_CONDITION = 1e10


def _adaptive_bilinear_area_result(
    *,
    a3d: float,
    valid_cells: int,
    level_sum: float,
    max_level_used: int,
    refined_cells: int,
    total_subcells: int,
) -> AreaResult:
    if valid_cells <= 0:
        return AreaResult(
            a3d=a3d,
            valid_cells=0,
            adaptive_avg_level=float("nan"),
            adaptive_max_level_used=0,
            adaptive_refined_cell_fraction=float("nan"),
            adaptive_total_subcells_evaluated=0,
        )
    return AreaResult(
        a3d=a3d,
        valid_cells=valid_cells,
        adaptive_avg_level=float(level_sum / float(valid_cells)),
        adaptive_max_level_used=int(max_level_used),
        adaptive_refined_cell_fraction=float(refined_cells / float(valid_cells)),
        adaptive_total_subcells_evaluated=int(total_subcells),
    )


def _sector_jenness_area_result(
    *,
    a3d: float,
    valid_cells: int,
    level_sum: float,
    max_level_used: int,
    refined_cells: int,
) -> AreaResult:
    if valid_cells <= 0:
        return AreaResult(
            a3d=a3d,
            valid_cells=0,
            sector_jenness_avg_level=float("nan"),
            sector_jenness_max_level_used=0,
            sector_jenness_refined_fraction=float("nan"),
        )
    return AreaResult(
        a3d=a3d,
        valid_cells=valid_cells,
        sector_jenness_avg_level=float(level_sum / float(valid_cells)),
        sector_jenness_max_level_used=int(max_level_used),
        sector_jenness_refined_fraction=float(refined_cells / float(valid_cells)),
    )


@lru_cache(maxsize=16)
def _quadratic_fit_operator(dx: float, dy: float) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return a numerically scaled pseudoinverse for the 3x3 quadratic fit."""
    coords = np.array(
        [
            (-float(dx), -float(dy)),
            (0.0, -float(dy)),
            (float(dx), -float(dy)),
            (-float(dx), 0.0),
            (0.0, 0.0),
            (float(dx), 0.0),
            (-float(dx), float(dy)),
            (0.0, float(dy)),
            (float(dx), float(dy)),
        ],
        dtype=np.float64,
    )
    x = coords[:, 0]
    y = coords[:, 1]
    design = np.column_stack((x * x, y * y, x * y, x, y, np.ones_like(x)))

    # Scale columns before forming the pseudoinverse so the fit remains stable
    # across anisotropic cell sizes and small/large grid units.
    col_scale = np.maximum(np.linalg.norm(design, axis=0), 1.0)
    design_scaled = design / col_scale[None, :]

    svals = np.linalg.svd(design_scaled, compute_uv=False)
    if svals.size < design.shape[1] or not np.all(np.isfinite(svals)):
        return None, None
    if svals[-1] <= np.finfo(np.float64).eps * svals[0]:
        return None, None

    cond = float(svals[0] / svals[-1])
    if not math.isfinite(cond) or cond > _QUADRATIC_FIT_MAX_CONDITION:
        return None, None

    pinv_scaled = np.linalg.pinv(design_scaled, rcond=1.0 / _QUADRATIC_FIT_MAX_CONDITION)
    return pinv_scaled.astype(np.float64, copy=False), col_scale.astype(np.float64, copy=False)


def _quadratic_coefficients_from_stencil(
    z: np.ndarray,
    dx: float,
    dy: float,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit z(x, y)=ax^2+by^2+cxy+dx+ey+f on each valid 3x3 neighborhood."""
    center_rows = max(0, z.shape[0] - 2)
    center_cols = max(0, z.shape[1] - 2)
    coeffs = np.full((center_rows, center_cols, 6), np.nan, dtype=np.float64)
    cell_valid = np.zeros((center_rows, center_cols), dtype=bool)
    if z.shape[0] < 3 or z.shape[1] < 3:
        return coeffs, cell_valid

    pinv_scaled, col_scale = _quadratic_fit_operator(float(dx), float(dy))
    if pinv_scaled is None or col_scale is None:
        return coeffs, cell_valid

    v = _stencil_all9(valid)
    if not np.any(v):
        return coeffs, cell_valid

    z64 = z.astype(np.float64, copy=False)
    samples = np.stack(
        (
            z64[:-2, :-2],
            z64[:-2, 1:-1],
            z64[:-2, 2:],
            z64[1:-1, :-2],
            z64[1:-1, 1:-1],
            z64[1:-1, 2:],
            z64[2:, :-2],
            z64[2:, 1:-1],
            z64[2:, 2:],
        ),
        axis=-1,
    )

    flat_valid = np.flatnonzero(v.reshape(-1))
    if flat_valid.size == 0:
        return coeffs, cell_valid

    sample_v = samples.reshape(-1, 9)[flat_valid]
    coeff_v_scaled = sample_v @ pinv_scaled.T
    coeff_v = coeff_v_scaled / col_scale[None, :]
    finite = np.all(np.isfinite(coeff_v), axis=1)
    if not np.any(finite):
        return coeffs, cell_valid

    coeffs.reshape(-1, 6)[flat_valid[finite]] = coeff_v[finite]
    cell_valid.reshape(-1)[flat_valid[finite]] = True
    return coeffs, cell_valid


def _triangle_area_2d(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    v1 = p1 - p0
    v2 = p2 - p0
    return 0.5 * abs(float(v1[0] * v2[1] - v1[1] * v2[0]))


def _subdivide_triangle(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...]:
    m01 = 0.5 * (p0 + p1)
    m12 = 0.5 * (p1 + p2)
    m20 = 0.5 * (p2 + p0)
    return (
        (p0, m01, m20),
        (m01, p1, m12),
        (m20, m12, p2),
        (m01, m12, m20),
    )


@lru_cache(maxsize=16)
def _sector_jenness_triangles(dx: float, dy: float) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...]:
    """Eight triangular sectors partitioning the current cell footprint exactly."""
    hx = 0.5 * float(dx)
    hy = 0.5 * float(dy)
    center = np.array((0.0, 0.0), dtype=np.float64)
    ring = (
        np.array((0.0, -hy), dtype=np.float64),
        np.array((hx, -hy), dtype=np.float64),
        np.array((hx, 0.0), dtype=np.float64),
        np.array((hx, hy), dtype=np.float64),
        np.array((0.0, hy), dtype=np.float64),
        np.array((-hx, hy), dtype=np.float64),
        np.array((-hx, 0.0), dtype=np.float64),
        np.array((-hx, -hy), dtype=np.float64),
    )
    sectors = tuple((center, ring[i], ring[(i + 1) % len(ring)]) for i in range(len(ring)))
    total_area = sum(_triangle_area_2d(p0, p1, p2) for p0, p1, p2 in sectors)
    if not math.isclose(total_area, float(dx) * float(dy), rel_tol=1e-12, abs_tol=1e-12):
        raise RuntimeError("Sector Jenness footprint partition does not match dx*dy")
    return sectors


@lru_cache(maxsize=16)
def _sector_jenness_geometry_arrays(dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    """Return cached array geometry for the sector footprint and its first subdivision."""
    sectors = _sector_jenness_triangles(float(dx), float(dy))
    sector_points = np.empty((len(sectors), 3, 2), dtype=np.float64)
    sector_children = np.empty((len(sectors), 4, 3, 2), dtype=np.float64)
    for sector_i, (p0, p1, p2) in enumerate(sectors):
        sector_points[sector_i, 0] = p0
        sector_points[sector_i, 1] = p1
        sector_points[sector_i, 2] = p2
        children = _subdivide_triangle(p0, p1, p2)
        for child_i, (c0, c1, c2) in enumerate(children):
            sector_children[sector_i, child_i, 0] = c0
            sector_children[sector_i, child_i, 1] = c1
            sector_children[sector_i, child_i, 2] = c2
    return sector_points, sector_children


@lru_cache(maxsize=8)
def _sector_jenness_triangle_rule(min_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic triangular quadrature rule with at least min_samples points."""
    if min_samples <= 1:
        bary = np.array(((1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),), dtype=np.float64)
        weights = np.array((1.0,), dtype=np.float64)
        return bary, weights
    if min_samples <= 3:
        bary = np.array(
            (
                (1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0),
                (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0),
                (2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0),
            ),
            dtype=np.float64,
        )
        weights = np.full((3,), 1.0 / 3.0, dtype=np.float64)
        return bary, weights

    bary = np.array(
        (
            (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
            (0.059715871789770, 0.470142064105115, 0.470142064105115),
            (0.470142064105115, 0.059715871789770, 0.470142064105115),
            (0.470142064105115, 0.470142064105115, 0.059715871789770),
            (0.101286507323456, 0.101286507323456, 0.797426985353087),
            (0.101286507323456, 0.797426985353087, 0.101286507323456),
            (0.797426985353087, 0.101286507323456, 0.101286507323456),
        ),
        dtype=np.float64,
    )
    weights = np.array(
        (
            0.225000000000000,
            0.132394152788506,
            0.132394152788506,
            0.132394152788506,
            0.125939180544827,
            0.125939180544827,
            0.125939180544827,
        ),
        dtype=np.float64,
    )
    return bary, weights


def _quadratic_area_integrand(coeff: np.ndarray, x: float, y: float) -> float:
    a, b, c, d, e, _ = (float(v) for v in coeff)
    dzdx = (2.0 * a * x) + (c * y) + d
    dzdy = (2.0 * b * y) + (c * x) + e
    return math.hypot(1.0, dzdx, dzdy)


def _triangle_quadrature_integral(
    coeff: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    bary: np.ndarray,
    weights: np.ndarray,
) -> float:
    x0 = float(p0[0])
    y0 = float(p0[1])
    x1 = float(p1[0])
    y1 = float(p1[1])
    x2 = float(p2[0])
    y2 = float(p2[1])

    area = 0.5 * abs((x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0))
    if area <= 0.0:
        return 0.0

    # Evaluate all quadrature points at once to avoid per-point Python overhead.
    x = bary[:, 0] * x0 + bary[:, 1] * x1 + bary[:, 2] * x2
    y = bary[:, 0] * y0 + bary[:, 1] * y1 + bary[:, 2] * y2

    a = float(coeff[0])
    b = float(coeff[1])
    c = float(coeff[2])
    d = float(coeff[3])
    e = float(coeff[4])
    if weights.size <= 3:
        total = 0.0
        for i in range(weights.size):
            bx = float(bary[i, 0])
            by = float(bary[i, 1])
            bz = float(bary[i, 2])
            xi = bx * x0 + by * x1 + bz * x2
            yi = bx * y0 + by * y1 + bz * y2
            dzdx_i = (2.0 * a * xi) + (c * yi) + d
            dzdy_i = (2.0 * b * yi) + (c * xi) + e
            total += float(weights[i]) * math.sqrt(1.0 + dzdx_i * dzdx_i + dzdy_i * dzdy_i)
        return area * total

    dzdx = (2.0 * a * x) + (c * y) + d
    dzdy = (2.0 * b * y) + (c * x) + e
    vals = np.sqrt(1.0 + dzdx * dzdx + dzdy * dzdy)
    return area * float(np.dot(weights, vals))


def _triangle_quadrature_integral_batch(
    coeffs: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    bary: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Evaluate one triangle quadrature rule for many quadratic surfaces at once."""
    if coeffs.ndim != 2 or coeffs.shape[1] < 5:
        raise ValueError("coeffs must have shape (n, >=5)")
    if coeffs.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)

    x0 = float(p0[0])
    y0 = float(p0[1])
    x1 = float(p1[0])
    y1 = float(p1[1])
    x2 = float(p2[0])
    y2 = float(p2[1])

    area = 0.5 * abs((x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0))
    if area <= 0.0:
        return np.zeros((coeffs.shape[0],), dtype=np.float64)

    x = bary[:, 0] * x0 + bary[:, 1] * x1 + bary[:, 2] * x2
    y = bary[:, 0] * y0 + bary[:, 1] * y1 + bary[:, 2] * y2

    a = coeffs[:, 0:1]
    b = coeffs[:, 1:2]
    c = coeffs[:, 2:3]
    d = coeffs[:, 3:4]
    e = coeffs[:, 4:5]

    dzdx = (2.0 * a * x[None, :]) + (c * y[None, :]) + d
    dzdy = (2.0 * b * y[None, :]) + (c * x[None, :]) + e
    vals = np.sqrt(1.0 + dzdx * dzdx + dzdy * dzdy)
    return area * (vals @ weights)


@_optional_njit(cache=True)
def _triangle_quadrature_integral_numba(
    coeff: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    bary: np.ndarray,
    weights: np.ndarray,
) -> float:
    area = 0.5 * abs((x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0))
    if area <= 0.0:
        return 0.0

    a = float(coeff[0])
    b = float(coeff[1])
    c = float(coeff[2])
    d = float(coeff[3])
    e = float(coeff[4])

    total = 0.0
    for i in range(weights.shape[0]):
        bx = float(bary[i, 0])
        by = float(bary[i, 1])
        bz = float(bary[i, 2])
        xi = bx * x0 + by * x1 + bz * x2
        yi = bx * y0 + by * y1 + bz * y2
        dzdx_i = (2.0 * a * xi) + (c * yi) + d
        dzdy_i = (2.0 * b * yi) + (c * xi) + e
        total += float(weights[i]) * math.sqrt(1.0 + dzdx_i * dzdx_i + dzdy_i * dzdy_i)
    return area * total


@_optional_njit(cache=True)
def _adaptive_triangle_integral_numba(
    coeff: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    bary: np.ndarray,
    weights: np.ndarray,
    rel_tol: float,
    abs_tol: float,
    max_level: int,
    level: int,
    coarse: float,
) -> tuple[float, int, bool]:
    coarse_here = float(coarse)
    if not math.isfinite(coarse_here):
        return math.nan, level, False
    if level >= max_level:
        return coarse_here, level, True

    remaining_levels = max_level - level
    if remaining_levels < 0:
        remaining_levels = 0
    stack_cap = 1 + (3 * remaining_levels)
    if stack_cap < 1:
        stack_cap = 1

    stack_x0 = np.empty((stack_cap,), dtype=np.float64)
    stack_y0 = np.empty((stack_cap,), dtype=np.float64)
    stack_x1 = np.empty((stack_cap,), dtype=np.float64)
    stack_y1 = np.empty((stack_cap,), dtype=np.float64)
    stack_x2 = np.empty((stack_cap,), dtype=np.float64)
    stack_y2 = np.empty((stack_cap,), dtype=np.float64)
    stack_abs_tol = np.empty((stack_cap,), dtype=np.float64)
    stack_coarse = np.empty((stack_cap,), dtype=np.float64)
    stack_level = np.empty((stack_cap,), dtype=np.int64)

    stack_size = 1
    stack_x0[0] = x0
    stack_y0[0] = y0
    stack_x1[0] = x1
    stack_y1[0] = y1
    stack_x2[0] = x2
    stack_y2[0] = y2
    stack_abs_tol[0] = abs_tol
    stack_coarse[0] = coarse_here
    stack_level[0] = level

    total = 0.0
    level_used = level

    while stack_size > 0:
        stack_size -= 1
        cur_x0 = stack_x0[stack_size]
        cur_y0 = stack_y0[stack_size]
        cur_x1 = stack_x1[stack_size]
        cur_y1 = stack_y1[stack_size]
        cur_x2 = stack_x2[stack_size]
        cur_y2 = stack_y2[stack_size]
        cur_abs_tol = stack_abs_tol[stack_size]
        cur_coarse = stack_coarse[stack_size]
        cur_level = int(stack_level[stack_size])

        if not math.isfinite(cur_coarse):
            return math.nan, level_used, False
        if cur_level >= max_level:
            total += cur_coarse
            if cur_level > level_used:
                level_used = cur_level
            continue

        m01x = 0.5 * (cur_x0 + cur_x1)
        m01y = 0.5 * (cur_y0 + cur_y1)
        m12x = 0.5 * (cur_x1 + cur_x2)
        m12y = 0.5 * (cur_y1 + cur_y2)
        m20x = 0.5 * (cur_x2 + cur_x0)
        m20y = 0.5 * (cur_y2 + cur_y0)

        child0 = _triangle_quadrature_integral_numba(coeff, cur_x0, cur_y0, m01x, m01y, m20x, m20y, bary, weights)
        child1 = _triangle_quadrature_integral_numba(coeff, m01x, m01y, cur_x1, cur_y1, m12x, m12y, bary, weights)
        child2 = _triangle_quadrature_integral_numba(coeff, m20x, m20y, m12x, m12y, cur_x2, cur_y2, bary, weights)
        child3 = _triangle_quadrature_integral_numba(coeff, m01x, m01y, m12x, m12y, m20x, m20y, bary, weights)
        if (
            (not math.isfinite(child0))
            or (not math.isfinite(child1))
            or (not math.isfinite(child2))
            or (not math.isfinite(child3))
        ):
            return math.nan, level_used, False

        fine = child0 + child1 + child2 + child3
        tol = max(float(cur_abs_tol), float(rel_tol) * abs(fine))
        next_level = cur_level + 1
        if abs(fine - cur_coarse) <= tol or next_level >= max_level:
            total += fine
            if next_level > level_used:
                level_used = next_level
            continue

        next_abs_tol = float(cur_abs_tol) * 0.25

        stack_x0[stack_size] = m01x
        stack_y0[stack_size] = m01y
        stack_x1[stack_size] = m12x
        stack_y1[stack_size] = m12y
        stack_x2[stack_size] = m20x
        stack_y2[stack_size] = m20y
        stack_abs_tol[stack_size] = next_abs_tol
        stack_coarse[stack_size] = child3
        stack_level[stack_size] = next_level
        stack_size += 1

        stack_x0[stack_size] = m20x
        stack_y0[stack_size] = m20y
        stack_x1[stack_size] = m12x
        stack_y1[stack_size] = m12y
        stack_x2[stack_size] = cur_x2
        stack_y2[stack_size] = cur_y2
        stack_abs_tol[stack_size] = next_abs_tol
        stack_coarse[stack_size] = child2
        stack_level[stack_size] = next_level
        stack_size += 1

        stack_x0[stack_size] = m01x
        stack_y0[stack_size] = m01y
        stack_x1[stack_size] = cur_x1
        stack_y1[stack_size] = cur_y1
        stack_x2[stack_size] = m12x
        stack_y2[stack_size] = m12y
        stack_abs_tol[stack_size] = next_abs_tol
        stack_coarse[stack_size] = child1
        stack_level[stack_size] = next_level
        stack_size += 1

        stack_x0[stack_size] = cur_x0
        stack_y0[stack_size] = cur_y0
        stack_x1[stack_size] = m01x
        stack_y1[stack_size] = m01y
        stack_x2[stack_size] = m20x
        stack_y2[stack_size] = m20y
        stack_abs_tol[stack_size] = next_abs_tol
        stack_coarse[stack_size] = child0
        stack_level[stack_size] = next_level
        stack_size += 1

    return total, level_used, True


@_optional_njit(cache=True)
def _integrate_sector_jenness_cell_from_level1_numba(
    coeff: np.ndarray,
    bary: np.ndarray,
    weights: np.ndarray,
    rel_tol: float,
    abs_tol: float,
    max_level: int,
    sector_accepted: np.ndarray,
    sector_fine: np.ndarray,
    sector_child_coarse: np.ndarray,
    sector_children: np.ndarray,
) -> tuple[float, int, bool]:
    total = 0.0
    level_used = 0
    sector_count = int(sector_children.shape[0])
    sector_abs_tol = float(abs_tol) / float(sector_count) if abs_tol > 0 else 0.0
    child_abs_tol = sector_abs_tol * 0.25

    for sector_i in range(sector_count):
        if bool(sector_accepted[sector_i]):
            area = float(sector_fine[sector_i])
            sector_level = 1
        else:
            area = 0.0
            sector_level = 1
            for child_i in range(sector_children.shape[1]):
                child = sector_children[sector_i, child_i]
                child_area, child_level, ok = _adaptive_triangle_integral_numba(
                    coeff,
                    float(child[0, 0]),
                    float(child[0, 1]),
                    float(child[1, 0]),
                    float(child[1, 1]),
                    float(child[2, 0]),
                    float(child[2, 1]),
                    bary,
                    weights,
                    rel_tol,
                    child_abs_tol,
                    max_level,
                    1,
                    float(sector_child_coarse[sector_i, child_i]),
                )
                if not ok:
                    return math.nan, level_used, False
                area += child_area
                if child_level > sector_level:
                    sector_level = child_level

        if not math.isfinite(area):
            return math.nan, level_used, False
        total += area
        if sector_level > level_used:
            level_used = sector_level
    return total, level_used, True


@_optional_njit(cache=True)
def _integrate_sector_jenness_cells_from_level1_numba(
    coeffs: np.ndarray,
    bary: np.ndarray,
    weights: np.ndarray,
    rel_tol: float,
    abs_tol: float,
    max_level: int,
    sector_accepted: np.ndarray,
    sector_fine: np.ndarray,
    sector_child_coarse: np.ndarray,
    sector_children: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(coeffs.shape[0])
    areas = np.empty((n,), dtype=np.float64)
    levels = np.zeros((n,), dtype=np.uint8)
    finite = np.ones((n,), dtype=np.bool_)
    for cell_i in range(n):
        area, level_used, ok = _integrate_sector_jenness_cell_from_level1_numba(
            coeffs[cell_i],
            bary,
            weights,
            rel_tol,
            abs_tol,
            max_level,
            sector_accepted[cell_i],
            sector_fine[cell_i],
            sector_child_coarse[cell_i],
            sector_children,
        )
        if ok:
            areas[cell_i] = area
            levels[cell_i] = np.uint8(level_used)
        else:
            areas[cell_i] = math.nan
            levels[cell_i] = np.uint8(0)
            finite[cell_i] = False
    return areas, levels, finite


def _integrate_sector_jenness_fallback_cells_from_level1(
    coeffs: np.ndarray,
    dx: float,
    dy: float,
    bary: np.ndarray,
    weights: np.ndarray,
    *,
    rel_tol: float,
    abs_tol: float,
    max_level: int,
    sector_accepted: np.ndarray,
    sector_fine: np.ndarray,
    sector_child_coarse: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(coeffs.shape[0])
    if n == 0:
        return (
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=np.uint8),
            np.zeros((0,), dtype=bool),
        )

    if NUMBA_AVAILABLE and max_level > 0 and n >= 8:
        _, sector_children = _sector_jenness_geometry_arrays(float(dx), float(dy))
        coeffs_c = np.ascontiguousarray(coeffs, dtype=np.float64)
        accepted_c = np.ascontiguousarray(sector_accepted.T, dtype=np.bool_)
        fine_c = np.ascontiguousarray(sector_fine.T, dtype=np.float64)
        child_c = np.ascontiguousarray(np.moveaxis(sector_child_coarse, -1, 0), dtype=np.float64)
        bary_c = np.ascontiguousarray(bary, dtype=np.float64)
        weights_c = np.ascontiguousarray(weights, dtype=np.float64)
        areas, levels, finite = _integrate_sector_jenness_cells_from_level1_numba(
            coeffs_c,
            bary_c,
            weights_c,
            float(rel_tol),
            float(abs_tol),
            int(max_level),
            accepted_c,
            fine_c,
            child_c,
            np.ascontiguousarray(sector_children, dtype=np.float64),
        )
        return areas, levels, finite

    areas = np.empty((n,), dtype=np.float64)
    levels = np.zeros((n,), dtype=np.uint8)
    finite = np.ones((n,), dtype=bool)
    for cell_i, coeff in enumerate(coeffs):
        if max_level <= 0:
            area, level_used = _integrate_sector_jenness_cell(
                coeff,
                dx,
                dy,
                bary,
                weights,
                rel_tol=rel_tol,
                abs_tol=abs_tol,
                max_level=max_level,
            )
        else:
            area, level_used = _integrate_sector_jenness_cell_from_level1(
                coeff,
                dx,
                dy,
                bary,
                weights,
                rel_tol=rel_tol,
                abs_tol=abs_tol,
                max_level=max_level,
                sector_accepted=sector_accepted[:, cell_i],
                sector_fine=sector_fine[:, cell_i],
                sector_child_coarse=sector_child_coarse[:, :, cell_i],
            )
        if math.isfinite(area):
            areas[cell_i] = area
            levels[cell_i] = np.uint8(level_used)
        else:
            areas[cell_i] = float("nan")
            finite[cell_i] = False
    return areas, levels, finite


def _sector_jenness_level1_presolve(
    coeffs: np.ndarray,
    dx: float,
    dy: float,
    bary: np.ndarray,
    weights: np.ndarray,
    *,
    rel_tol: float,
    abs_tol: float,
    max_level: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Precompute level-1 sector refinement results for many cells.

    Returns:
      cell_accepted: (n,) bool, True where every sector converged at level 1
      cell_area: (n,) float64, sum of level-1 sector areas for accepted cells
      sector_accepted: (8, n) bool, per-sector level-1 convergence flags
      sector_fine: (8, n) float64, per-sector level-1 fine areas
      sector_child_coarse: (8, 4, n) float64, child triangle coarse values
    """
    n = int(coeffs.shape[0])
    if n <= 0 or max_level <= 0:
        return (
            np.zeros((n,), dtype=bool),
            np.zeros((n,), dtype=np.float64),
            np.zeros((0, n), dtype=bool),
            np.zeros((0, n), dtype=np.float64),
            np.zeros((0, 4, n), dtype=np.float64),
        )

    sectors = _sector_jenness_triangles(float(dx), float(dy))
    sector_count = len(sectors)
    sector_accepted = np.ones((sector_count, n), dtype=bool)
    sector_fine = np.zeros((sector_count, n), dtype=np.float64)
    sector_child_coarse = np.zeros((sector_count, 4, n), dtype=np.float64)
    sector_abs_tol = float(abs_tol) / float(sector_count) if abs_tol > 0 else 0.0

    for sector_i, (p0, p1, p2) in enumerate(sectors):
        coarse = _triangle_quadrature_integral_batch(coeffs, p0, p1, p2, bary, weights)
        child_sum = np.zeros((n,), dtype=np.float64)
        child_finite = np.ones((n,), dtype=bool)
        for child_i, (c0, c1, c2) in enumerate(_subdivide_triangle(p0, p1, p2)):
            child_area = _triangle_quadrature_integral_batch(coeffs, c0, c1, c2, bary, weights)
            sector_child_coarse[sector_i, child_i] = child_area
            child_sum += child_area
            child_finite &= np.isfinite(child_area)

        finite = np.isfinite(coarse) & child_finite & np.isfinite(child_sum)
        if max_level > 1:
            tol = np.maximum(sector_abs_tol, float(rel_tol) * np.abs(child_sum))
            sector_accepted[sector_i] = finite & (np.abs(child_sum - coarse) <= tol)
        else:
            sector_accepted[sector_i] = finite
        sector_fine[sector_i] = child_sum

    cell_accepted = sector_accepted.all(axis=0)
    cell_area = sector_fine.sum(axis=0, dtype=np.float64)
    cell_area[~cell_accepted] = 0.0
    return cell_accepted, cell_area, sector_accepted, sector_fine, sector_child_coarse


def _adaptive_triangle_integral(
    coeff: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    bary: np.ndarray,
    weights: np.ndarray,
    *,
    rel_tol: float,
    abs_tol: float,
    max_level: int,
    level: int,
    coarse: float | None = None,
) -> tuple[float, int]:
    """Adaptive triangle integration by recursive 4-way subdivision."""
    coarse_here = (
        _triangle_quadrature_integral(coeff, p0, p1, p2, bary, weights)
        if coarse is None
        else float(coarse)
    )
    if not math.isfinite(coarse_here):
        return float("nan"), level
    if level >= max_level:
        return coarse_here, level

    children = _subdivide_triangle(p0, p1, p2)
    child_coarse = [
        _triangle_quadrature_integral(coeff, c0, c1, c2, bary, weights)
        for c0, c1, c2 in children
    ]
    if not all(math.isfinite(v) for v in child_coarse):
        return float("nan"), level

    fine = float(sum(child_coarse))
    tol = max(float(abs_tol), float(rel_tol) * abs(fine))
    if abs(fine - coarse_here) <= tol or (level + 1) >= max_level:
        return fine, level + 1

    child_abs_tol = float(abs_tol) * 0.25
    total = 0.0
    level_used = level + 1
    for (c0, c1, c2), child_val in zip(children, child_coarse, strict=False):
        child_area, child_level = _adaptive_triangle_integral(
            coeff,
            c0,
            c1,
            c2,
            bary,
            weights,
            rel_tol=rel_tol,
            abs_tol=child_abs_tol,
            max_level=max_level,
            level=level + 1,
            coarse=child_val,
        )
        if not math.isfinite(child_area):
            return float("nan"), level_used
        total += child_area
        level_used = max(level_used, child_level)
    return total, level_used


def _sector_jenness_planar_fastpath(
    coeff_v: np.ndarray,
    dx: float,
    dy: float,
    *,
    rel_tol: float,
    abs_tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return cells whose fitted quadratic is close enough to planar over the cell."""
    if coeff_v.size == 0:
        return np.zeros((0,), dtype=bool), np.zeros((0,), dtype=np.float64)

    cell_area_2d = float(dx) * float(dy)
    p0 = coeff_v[:, 3]
    q0 = coeff_v[:, 4]
    plane_area = cell_area_2d * np.sqrt(1.0 + p0 * p0 + q0 * q0)

    hx = 0.5 * float(dx)
    hy = 0.5 * float(dy)
    cx = np.array((-hx, hx, hx, -hx), dtype=np.float64)
    cy = np.array((-hy, -hy, hy, hy), dtype=np.float64)

    a = coeff_v[:, 0:1]
    b = coeff_v[:, 1:2]
    c = coeff_v[:, 2:3]
    delta_dx = (2.0 * a * cx[None, :]) + (c * cy[None, :])
    delta_dy = (c * cx[None, :]) + (2.0 * b * cy[None, :])
    max_delta = np.sqrt(delta_dx * delta_dx + delta_dy * delta_dy).max(axis=1)

    tol = np.maximum(float(abs_tol), float(rel_tol) * plane_area)
    use_plane = (cell_area_2d * max_delta) <= tol
    return use_plane, plane_area


def _sector_jenness_level1_fastpath(
    coeffs: np.ndarray,
    dx: float,
    dy: float,
    bary: np.ndarray,
    weights: np.ndarray,
    *,
    rel_tol: float,
    abs_tol: float,
    max_level: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve cells that converge after the first adaptive refinement step."""
    accepted, area, _, _, _ = _sector_jenness_level1_presolve(
        coeffs,
        dx,
        dy,
        bary,
        weights,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
        max_level=max_level,
    )
    return accepted, area


def _integrate_sector_jenness_cell(
    coeff: np.ndarray,
    dx: float,
    dy: float,
    bary: np.ndarray,
    weights: np.ndarray,
    *,
    rel_tol: float,
    abs_tol: float,
    max_level: int,
) -> tuple[float, int]:
    total = 0.0
    level_used = 0
    sector_abs_tol = float(abs_tol) / 8.0 if abs_tol > 0 else 0.0

    for p0, p1, p2 in _sector_jenness_triangles(float(dx), float(dy)):
        if max_level <= 0:
            area = _triangle_quadrature_integral(coeff, p0, p1, p2, bary, weights)
            sector_level = 0
        else:
            area, sector_level = _adaptive_triangle_integral(
                coeff,
                p0,
                p1,
                p2,
                bary,
                weights,
                rel_tol=rel_tol,
                abs_tol=sector_abs_tol,
                max_level=max_level,
                level=0,
            )
        if not math.isfinite(area):
            return float("nan"), level_used
        total += area
        level_used = max(level_used, sector_level)
    return total, level_used


def _integrate_sector_jenness_cell_from_level1(
    coeff: np.ndarray,
    dx: float,
    dy: float,
    bary: np.ndarray,
    weights: np.ndarray,
    *,
    rel_tol: float,
    abs_tol: float,
    max_level: int,
    sector_accepted: np.ndarray,
    sector_fine: np.ndarray,
    sector_child_coarse: np.ndarray,
) -> tuple[float, int]:
    """Continue adaptive sector integration from precomputed level-1 results."""
    total = 0.0
    level_used = 0
    sector_abs_tol = float(abs_tol) / 8.0 if abs_tol > 0 else 0.0
    child_abs_tol = sector_abs_tol * 0.25

    for sector_i, (p0, p1, p2) in enumerate(_sector_jenness_triangles(float(dx), float(dy))):
        if bool(sector_accepted[sector_i]):
            area = float(sector_fine[sector_i])
            sector_level = 1
        else:
            area = 0.0
            sector_level = 1
            for child_i, (c0, c1, c2) in enumerate(_subdivide_triangle(p0, p1, p2)):
                child_area, child_level = _adaptive_triangle_integral(
                    coeff,
                    c0,
                    c1,
                    c2,
                    bary,
                    weights,
                    rel_tol=rel_tol,
                    abs_tol=child_abs_tol,
                    max_level=max_level,
                    level=1,
                    coarse=float(sector_child_coarse[sector_i, child_i]),
                )
                if not math.isfinite(child_area):
                    return float("nan"), level_used
                area += child_area
                sector_level = max(sector_level, child_level)

        if not math.isfinite(area):
            return float("nan"), level_used
        total += area
        level_used = max(level_used, sector_level)
    return total, level_used


def compute_area_jenness(
    z: np.ndarray,
    dx: float,
    dy: float,
    valid: np.ndarray,
    *,
    weight: float = 0.25,
) -> AreaResult:
    """Jenness-style 3x3 window around each center cell using 8 triangles."""
    areas, cell_valid = jenness_window_8tri_cell_areas(z, dx, dy, valid, weight=weight)
    return AreaResult(a3d=float(areas[cell_valid].sum(dtype=np.float64)), valid_cells=int(cell_valid.sum()))


def compute_area_sector_adaptive_jenness_integral(
    z: np.ndarray,
    dx: float,
    dy: float,
    valid: np.ndarray,
    *,
    rel_tol: float = 1e-4,
    abs_tol: float = 0.0,
    max_level: int = 5,
    min_samples: int = 3,
) -> AreaResult:
    areas, cell_valid, levels = sector_adaptive_jenness_integral_cell_areas(
        z,
        dx,
        dy,
        valid,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
        max_level=max_level,
        min_samples=min_samples,
    )
    v = cell_valid
    n = int(v.sum())
    a3d = float(areas[v].sum(dtype=np.float64))
    if n <= 0:
        return _sector_jenness_area_result(
            a3d=a3d,
            valid_cells=0,
            level_sum=0.0,
            max_level_used=0,
            refined_cells=0,
        )

    levels_v = levels[v]
    return _sector_jenness_area_result(
        a3d=a3d,
        valid_cells=n,
        level_sum=float(levels_v.sum(dtype=np.int64)),
        max_level_used=int(levels_v.max(initial=0)),
        refined_cells=int((levels_v > 0).sum()),
    )


def jenness_window_8tri_cell_areas(
    z: np.ndarray,
    dx: float,
    dy: float,
    valid: np.ndarray,
    *,
    weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell areas for the Jenness 8-triangle method.

    Returns:
      areas: float64 array (rows, cols), 0 where invalid/uncomputed
      cell_valid: bool array (rows, cols), True where a cell contributed
    """
    if z.shape != valid.shape:
        raise ValueError("z and valid must have the same shape")
    if z.ndim != 2:
        raise ValueError("z must be 2D")
    if dx <= 0 or dy <= 0:
        raise ValueError("dx and dy must be > 0")
    if weight <= 0:
        raise ValueError("weight must be > 0")

    rows, cols = z.shape
    out = np.zeros((rows, cols), dtype=np.float64)
    cell_valid = np.zeros((rows, cols), dtype=bool)
    if rows < 3 or cols < 3:
        return out, cell_valid

    diag = math.hypot(dx, dy)

    C = z[1:-1, 1:-1].astype(np.float64, copy=False)
    N = z[:-2, 1:-1].astype(np.float64, copy=False)
    NE = z[:-2, 2:].astype(np.float64, copy=False)
    E = z[1:-1, 2:].astype(np.float64, copy=False)
    SE = z[2:, 2:].astype(np.float64, copy=False)
    S = z[2:, 1:-1].astype(np.float64, copy=False)
    SW = z[2:, :-2].astype(np.float64, copy=False)
    W = z[1:-1, :-2].astype(np.float64, copy=False)
    NW = z[:-2, :-2].astype(np.float64, copy=False)

    v = _stencil_all9(valid)
    if not np.any(v):
        return out, cell_valid

    # Distances from center.
    dCN = np.sqrt(dy * dy + (C - N) ** 2)
    dCNE = np.sqrt(diag * diag + (C - NE) ** 2)
    dCE = np.sqrt(dx * dx + (C - E) ** 2)
    dCSE = np.sqrt(diag * diag + (C - SE) ** 2)
    dCS = np.sqrt(dy * dy + (C - S) ** 2)
    dCSW = np.sqrt(diag * diag + (C - SW) ** 2)
    dCW = np.sqrt(dx * dx + (C - W) ** 2)
    dCNW = np.sqrt(diag * diag + (C - NW) ** 2)

    # Neighbor-neighbor distances around the ring.
    dN_NE = np.sqrt(dx * dx + (N - NE) ** 2)
    dNE_E = np.sqrt(dy * dy + (NE - E) ** 2)
    dE_SE = np.sqrt(dy * dy + (E - SE) ** 2)
    dSE_S = np.sqrt(dx * dx + (SE - S) ** 2)
    dS_SW = np.sqrt(dx * dx + (S - SW) ** 2)
    dSW_W = np.sqrt(dy * dy + (SW - W) ** 2)
    dW_NW = np.sqrt(dy * dy + (W - NW) ** 2)
    dNW_N = np.sqrt(dx * dx + (NW - N) ** 2)

    a1 = _triangle_area_heron(dCN, dCNE, dN_NE)
    a2 = _triangle_area_heron(dCNE, dCE, dNE_E)
    a3 = _triangle_area_heron(dCE, dCSE, dE_SE)
    a4 = _triangle_area_heron(dCSE, dCS, dSE_S)
    a5 = _triangle_area_heron(dCS, dCSW, dS_SW)
    a6 = _triangle_area_heron(dCSW, dCW, dSW_W)
    a7 = _triangle_area_heron(dCW, dCNW, dW_NW)
    a8 = _triangle_area_heron(dCNW, dCN, dNW_N)

    areas_center = (a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8) * float(weight)

    out_center = np.where(v, areas_center, 0.0)
    out[1:-1, 1:-1] = out_center
    cell_valid[1:-1, 1:-1] = v
    return out, cell_valid


def sector_adaptive_jenness_integral_cell_areas(
    z: np.ndarray,
    dx: float,
    dy: float,
    valid: np.ndarray,
    *,
    rel_tol: float,
    abs_tol: float,
    max_level: int,
    min_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-cell quadratic sector integration over the current cell footprint.

    The local surface is fit on a full 3x3 neighborhood and then integrated over
    8 triangular sectors inside the current cell footprint itself. The sectors are
    defined by the cell center and the alternating edge-midpoint/corner ring.
    """
    if z.shape != valid.shape:
        raise ValueError("z and valid must have the same shape")
    if z.ndim != 2:
        raise ValueError("z must be 2D")
    if dx <= 0 or dy <= 0:
        raise ValueError("dx and dy must be > 0")
    if rel_tol < 0 or abs_tol < 0:
        raise ValueError("rel_tol and abs_tol must be >= 0")
    if max_level < 0:
        raise ValueError("max_level must be >= 0")
    if min_samples <= 0:
        raise ValueError("min_samples must be >= 1")

    rows, cols = z.shape
    out = np.zeros((rows, cols), dtype=np.float64)
    cell_valid = np.zeros((rows, cols), dtype=bool)
    levels = np.zeros((rows, cols), dtype=np.uint8)
    if rows < 3 or cols < 3:
        return out, cell_valid, levels

    coeffs_center, coeff_valid = _quadratic_coefficients_from_stencil(z, dx, dy, valid)
    if not np.any(coeff_valid):
        return out, cell_valid, levels

    bary, weights = _sector_jenness_triangle_rule(int(min_samples))

    coeff_flat = coeffs_center.reshape(-1, 6)
    valid_flat_idx = np.flatnonzero(coeff_valid.reshape(-1))
    coeff_v = coeff_flat[valid_flat_idx]

    center_area = np.zeros((rows - 2, cols - 2), dtype=np.float64)
    center_levels = np.zeros((rows - 2, cols - 2), dtype=np.uint8)
    center_valid = coeff_valid.copy()

    use_plane, plane_area = _sector_jenness_planar_fastpath(
        coeff_v,
        dx,
        dy,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
    )
    center_area.reshape(-1)[valid_flat_idx[use_plane]] = plane_area[use_plane]

    active_idx = valid_flat_idx[~use_plane]
    active_coeffs = coeff_v[~use_plane]
    fast_mask, fast_area, sector_fast, sector_fine, sector_child_coarse = _sector_jenness_level1_presolve(
        active_coeffs,
        dx,
        dy,
        bary,
        weights,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
        max_level=max_level,
    )
    if np.any(fast_mask):
        center_area.reshape(-1)[active_idx[fast_mask]] = fast_area[fast_mask]
        center_levels.reshape(-1)[active_idx[fast_mask]] = np.uint8(1)

    fallback_idx = active_idx[~fast_mask]
    fallback_coeffs = active_coeffs[~fast_mask]
    fallback_sector_fast = sector_fast[:, ~fast_mask]
    fallback_sector_fine = sector_fine[:, ~fast_mask]
    fallback_child_coarse = sector_child_coarse[:, :, ~fast_mask]
    fallback_areas, fallback_levels, fallback_finite = _integrate_sector_jenness_fallback_cells_from_level1(
        fallback_coeffs,
        dx,
        dy,
        bary,
        weights,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
        max_level=max_level,
        sector_accepted=fallback_sector_fast,
        sector_fine=fallback_sector_fine,
        sector_child_coarse=fallback_child_coarse,
    )
    if fallback_idx.size > 0:
        center_area.reshape(-1)[fallback_idx[fallback_finite]] = fallback_areas[fallback_finite]
        center_levels.reshape(-1)[fallback_idx[fallback_finite]] = fallback_levels[fallback_finite]
        center_valid.reshape(-1)[fallback_idx[~fallback_finite]] = False

    out[1:-1, 1:-1] = np.where(center_valid, center_area, 0.0)
    cell_valid[1:-1, 1:-1] = center_valid
    levels[1:-1, 1:-1] = np.where(center_valid, center_levels, 0).astype(np.uint8, copy=False)
    return out, cell_valid, levels


def compute_area_gradient(
    z: np.ndarray,
    dx: float,
    dy: float,
    valid: np.ndarray,
    *,
    method: SlopeMethod = "horn",
) -> AreaResult:
    areas, cell_valid = gradient_multiplier_cell_areas(z, dx, dy, valid, method=method)
    return AreaResult(a3d=float(areas[cell_valid].sum(dtype=np.float64)), valid_cells=int(cell_valid.sum()))


def gradient_multiplier_cell_areas(
    z: np.ndarray,
    dx: float,
    dy: float,
    valid: np.ndarray,
    *,
    method: SlopeMethod,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell areas using local area factor sqrt(1+p^2+q^2)."""
    if z.shape != valid.shape:
        raise ValueError("z and valid must have the same shape")
    if z.ndim != 2:
        raise ValueError("z must be 2D")
    if dx <= 0 or dy <= 0:
        raise ValueError("dx and dy must be > 0")

    rows, cols = z.shape
    out = np.zeros((rows, cols), dtype=np.float64)
    cell_valid = np.zeros((rows, cols), dtype=bool)
    if rows < 3 or cols < 3:
        return out, cell_valid

    method_n = method.strip().lower()
    if method_n not in {"horn", "zt"}:
        raise ValueError(f"Unknown slope method: {method!r} (use horn|zt)")

    z64 = z.astype(np.float64, copy=False)
    C = z64[1:-1, 1:-1]
    N = z64[:-2, 1:-1]
    S = z64[2:, 1:-1]
    E = z64[1:-1, 2:]
    W = z64[1:-1, :-2]

    if method_n == "horn":
        NW = z64[:-2, :-2]
        NE = z64[:-2, 2:]
        SW = z64[2:, :-2]
        SE = z64[2:, 2:]

        dzdx = ((NE + 2.0 * E + SE) - (NW + 2.0 * W + SW)) / (8.0 * dx)
        dzdy = ((SW + 2.0 * S + SE) - (NW + 2.0 * N + NE)) / (8.0 * dy)
        v = _stencil_all9(valid)
    else:
        dzdx = (E - W) / (2.0 * dx)
        dzdy = (S - N) / (2.0 * dy)
        v = _stencil_cross(valid)

    # local_factor = sqrt(1 + p^2 + q^2)
    local = np.sqrt(1.0 + dzdx * dzdx + dzdy * dzdy)
    areas_center = (dx * dy) * local

    out_center = np.where(v, areas_center, 0.0)
    out[1:-1, 1:-1] = out_center
    cell_valid[1:-1, 1:-1] = v
    return out, cell_valid


def _corners_from_centers(
    z: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute corner elevations from center cells (count==4 only).

    Returns p00,p10,p01,p11 (each rows x cols) and cell_valid mask.
    Cells are valid when:
    - center is valid
    - all 4 corners are defined from 4 valid centers (count==4 for each corner)
    """
    rows, cols = z.shape

    z_nan = np.where(valid, z.astype(np.float64, copy=False), np.nan)
    pad = np.pad(z_nan, ((1, 1), (1, 1)), mode="constant", constant_values=np.nan)

    a = pad[0 : rows + 1, 0 : cols + 1]
    b = pad[0 : rows + 1, 1 : cols + 2]
    c = pad[1 : rows + 2, 0 : cols + 1]
    d = pad[1 : rows + 2, 1 : cols + 2]

    fa = np.isfinite(a)
    fb = np.isfinite(b)
    fc = np.isfinite(c)
    fd = np.isfinite(d)
    count = fa.astype(np.uint8) + fb.astype(np.uint8) + fc.astype(np.uint8) + fd.astype(np.uint8)

    corner_sum = (
        np.where(fa, a, 0.0)
        + np.where(fb, b, 0.0)
        + np.where(fc, c, 0.0)
        + np.where(fd, d, 0.0)
    )
    # Only accept fully-supported corners to avoid nodata bleed and edge artifacts.
    corner = np.where(count == 4, corner_sum * 0.25, np.nan)

    p00 = corner[0:rows, 0:cols]
    p10 = corner[0:rows, 1 : cols + 1]
    p01 = corner[1 : rows + 1, 0:cols]
    p11 = corner[1 : rows + 1, 1 : cols + 1]

    cell_valid = (
        valid
        & np.isfinite(p00)
        & np.isfinite(p10)
        & np.isfinite(p01)
        & np.isfinite(p11)
    )
    return p00, p10, p01, p11, cell_valid


def compute_area_tin_2tri(
    z: np.ndarray,
    dx: float,
    dy: float,
    valid: np.ndarray,
) -> AreaResult:
    areas, cell_valid = tin_2tri_cell_areas(z, dx, dy, valid)
    return AreaResult(a3d=float(areas[cell_valid].sum(dtype=np.float64)), valid_cells=int(cell_valid.sum()))


def tin_2tri_cell_areas(
    z: np.ndarray, dx: float, dy: float, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell two-triangle (TIN) areas using corner elevations estimated from centers."""
    if z.shape != valid.shape:
        raise ValueError("z and valid must have the same shape")
    if z.ndim != 2:
        raise ValueError("z must be 2D")
    if dx <= 0 or dy <= 0:
        raise ValueError("dx and dy must be > 0")

    rows, cols = z.shape
    out = np.zeros((rows, cols), dtype=np.float64)
    cell_valid = np.zeros((rows, cols), dtype=bool)
    if rows < 3 or cols < 3:
        return out, cell_valid

    p00, p10, p01, p11, v = _corners_from_centers(z, valid)
    if not np.any(v):
        return out, cell_valid

    dz_b = p10 - p00
    dz_c = p11 - p00
    mag1 = np.sqrt((dz_b * dy) ** 2 + (dx * (dz_b - dz_c)) ** 2 + (dx * dy) ** 2)

    dz_b2 = p11 - p00
    dz_c2 = p01 - p00
    mag2 = np.sqrt((dy * (dz_c2 - dz_b2)) ** 2 + (dx * dz_c2) ** 2 + (dx * dy) ** 2)

    areas = 0.5 * (mag1 + mag2)
    out = np.where(v, areas, 0.0)
    cell_valid = v
    return out, cell_valid


def compute_area_bilinear_integral(
    z: np.ndarray,
    dx: float,
    dy: float,
    valid: np.ndarray,
    *,
    N: int = 5,
) -> AreaResult:
    areas, cell_valid = bilinear_patch_integral_cell_areas(z, dx, dy, valid, N=N)
    return AreaResult(a3d=float(areas[cell_valid].sum(dtype=np.float64)), valid_cells=int(cell_valid.sum()))


def compute_area_adaptive_bilinear_integral(
    z: np.ndarray,
    dx: float,
    dy: float,
    valid: np.ndarray,
    *,
    rel_tol: float = 1e-4,
    abs_tol: float = 0.0,
    max_level: int = 5,
    min_N: int = 2,
    roughness_fastpath: bool = True,
    roughness_threshold: float | None = None,
) -> AreaResult:
    areas, cell_valid, levels, subcells = adaptive_bilinear_patch_integral_cell_areas(
        z,
        dx,
        dy,
        valid,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
        max_level=max_level,
        min_N=min_N,
        roughness_fastpath=roughness_fastpath,
        roughness_threshold=roughness_threshold,
    )
    v = cell_valid
    n = int(v.sum())
    a3d = float(areas[v].sum(dtype=np.float64))
    levels_v = levels[v]
    refined = int((levels_v > 1).sum())
    return _adaptive_bilinear_area_result(
        a3d=a3d,
        valid_cells=n,
        level_sum=float(levels_v.sum(dtype=np.int64)),
        max_level_used=int(levels_v.max(initial=0)) if n > 0 else 0,
        refined_cells=refined,
        total_subcells=int(subcells[v].sum(dtype=np.int64)),
    )


def bilinear_patch_integral_cell_areas(
    z: np.ndarray, dx: float, dy: float, valid: np.ndarray, *, N: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell numeric surface-area integration over a bilinear patch.

    Each cell is treated as a bilinear surface defined by 4 corner z-values, and
    integrated by subdividing the cell into NxN subcells, each split into 2 triangles.
    """
    if z.shape != valid.shape:
        raise ValueError("z and valid must have the same shape")
    if z.ndim != 2:
        raise ValueError("z must be 2D")
    if dx <= 0 or dy <= 0:
        raise ValueError("dx and dy must be > 0")
    if N <= 0:
        raise ValueError("N must be >= 1")

    rows, cols = z.shape
    out = np.zeros((rows, cols), dtype=np.float64)
    cell_valid = np.zeros((rows, cols), dtype=bool)
    if rows < 3 or cols < 3:
        return out, cell_valid

    p00, p10, p01, p11, v = _corners_from_centers(z, valid)
    if not np.any(v):
        return out, cell_valid

    out = _bilinear_patch_integral_from_corners(p00, p10, p01, p11, v, dx, dy, N=N)
    cell_valid = v
    return out, cell_valid


def adaptive_bilinear_patch_integral_cell_areas(
    z: np.ndarray,
    dx: float,
    dy: float,
    valid: np.ndarray,
    *,
    rel_tol: float,
    abs_tol: float,
    max_level: int,
    min_N: int,
    roughness_fastpath: bool,
    roughness_threshold: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-cell adaptive bilinear patch integration.

    Returns:
      areas: float64 array (rows, cols), 0 where invalid/uncomputed
      cell_valid: bool array (rows, cols), True where a cell contributed
      final_level: uint8 array (rows, cols), refinement level used per cell
      subcells_total: int32 array (rows, cols), total subcells evaluated per cell

    Notes:
    - Corners are derived from centers using the same `_corners_from_centers` logic as the
      fixed-N `bilinear_patch_integral` method.
    - Refinement levels correspond to N = min_N * 2**level. When max_level>=1, the
      algorithm compares consecutive levels (coarse L vs fine L+1) and returns the fine.
    """
    if z.shape != valid.shape:
        raise ValueError("z and valid must have the same shape")
    if z.ndim != 2:
        raise ValueError("z must be 2D")
    if dx <= 0 or dy <= 0:
        raise ValueError("dx and dy must be > 0")
    if min_N <= 0:
        raise ValueError("min_N must be >= 1")
    if max_level < 0:
        raise ValueError("max_level must be >= 0")
    if rel_tol < 0 or abs_tol < 0:
        raise ValueError("rel_tol and abs_tol must be >= 0")

    rows, cols = z.shape
    areas = np.zeros((rows, cols), dtype=np.float64)
    cell_valid = np.zeros((rows, cols), dtype=bool)
    levels = np.zeros((rows, cols), dtype=np.uint8)
    subcells_total = np.zeros((rows, cols), dtype=np.int32)
    if rows < 3 or cols < 3:
        return areas, cell_valid, levels, subcells_total

    p00, p10, p01, p11, v = _corners_from_centers(z, valid)
    if not np.any(v):
        return areas, cell_valid, levels, subcells_total

    a, lvl, sub = _adaptive_bilinear_patch_integral_from_corners(
        p00,
        p10,
        p01,
        p11,
        v,
        dx,
        dy,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
        max_level=max_level,
        min_N=min_N,
        roughness_fastpath=roughness_fastpath,
        roughness_threshold=roughness_threshold,
    )
    areas = a
    cell_valid = v
    levels = lvl
    subcells_total = sub
    return areas, cell_valid, levels, subcells_total


def _bilinear_patch_integral_from_corners(
    p00: np.ndarray,
    p10: np.ndarray,
    p01: np.ndarray,
    p11: np.ndarray,
    v: np.ndarray,
    dx: float,
    dy: float,
    *,
    N: int,
) -> np.ndarray:
    """Compute per-cell bilinear patch integral areas from corner arrays."""
    rows, cols = p00.shape
    du = float(dx) / float(N)
    dv = float(dy) / float(N)

    # Avoid NaN propagation: use 0 corners for invalid cells; mask them out at the end.
    p00m = np.where(v, p00, 0.0)
    p10m = np.where(v, p10, 0.0)
    p01m = np.where(v, p01, 0.0)
    p11m = np.where(v, p11, 0.0)

    u = np.linspace(0.0, 1.0, N + 1, dtype=np.float64)
    w = (1.0 - u).astype(np.float64, copy=False)

    # Precompute bilinear node z on the (N+1)x(N+1) grid for all cells in the block.
    # Shape: (N+1, N+1, rows, cols)
    U = u[:, None, None, None]
    W = w[:, None, None, None]
    V = u[None, :, None, None]
    T = w[None, :, None, None]

    z_nodes = (W * T) * p00m + (U * T) * p10m + (W * V) * p01m + (U * V) * p11m

    area = np.zeros((rows, cols), dtype=np.float64)
    base = (du * dv) ** 2

    for i in range(N):
        for j in range(N):
            za = z_nodes[i, j]
            zb = z_nodes[i + 1, j]
            zc = z_nodes[i + 1, j + 1]
            zd = z_nodes[i, j + 1]

            dz_ab = zb - za
            dz_ac = zc - za
            mag1 = np.sqrt((dz_ab * dv) ** 2 + (du * (dz_ab - dz_ac)) ** 2 + base)

            dz_b2 = zc - za
            dz_c2 = zd - za
            mag2 = np.sqrt((dv * (dz_c2 - dz_b2)) ** 2 + (du * dz_c2) ** 2 + base)

            area += 0.5 * (mag1 + mag2)

    return np.where(v, area, 0.0)


def _tin_2tri_area_from_corners(
    p00: np.ndarray, p10: np.ndarray, p01: np.ndarray, p11: np.ndarray, dx: float, dy: float
) -> np.ndarray:
    dz_b = p10 - p00
    dz_c = p11 - p00
    mag1 = np.sqrt((dz_b * dy) ** 2 + (dx * (dz_b - dz_c)) ** 2 + (dx * dy) ** 2)

    dz_b2 = p11 - p00
    dz_c2 = p01 - p00
    mag2 = np.sqrt((dy * (dz_c2 - dz_b2)) ** 2 + (dx * dz_c2) ** 2 + (dx * dy) ** 2)

    return 0.5 * (mag1 + mag2)


def _bilinear_patch_integral_1d(
    p00: np.ndarray,
    p10: np.ndarray,
    p01: np.ndarray,
    p11: np.ndarray,
    dx: float,
    dy: float,
    *,
    N: int,
) -> np.ndarray:
    """Compute bilinear patch areas for a 1D list of valid cells, chunked for memory safety."""
    n = int(p00.size)
    if n == 0:
        return np.zeros((0,), dtype=np.float64)

    # Memory budget for z_nodes (~8 bytes per float). Keep comfortably below a few hundred MB.
    # z_nodes shape is (N+1, N+1, chunk, 1) => (N+1)^2 * chunk floats.
    nodes_per_cell = int((int(N) + 1) * (int(N) + 1))
    max_nodes = 8_000_000  # ~64 MB of float64
    chunk = max(1, int(max_nodes // max(nodes_per_cell, 1)))

    out = np.empty((n,), dtype=np.float64)
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        c00 = p00[start:end].reshape(-1, 1)
        c10 = p10[start:end].reshape(-1, 1)
        c01 = p01[start:end].reshape(-1, 1)
        c11 = p11[start:end].reshape(-1, 1)
        v = np.ones_like(c00, dtype=bool)
        a2 = _bilinear_patch_integral_from_corners(c00, c10, c01, c11, v, dx, dy, N=N)
        out[start:end] = a2.reshape(-1)
    return out


def _adaptive_bilinear_patch_integral_from_corners(
    p00: np.ndarray,
    p10: np.ndarray,
    p01: np.ndarray,
    p11: np.ndarray,
    v: np.ndarray,
    dx: float,
    dy: float,
    *,
    rel_tol: float,
    abs_tol: float,
    max_level: int,
    min_N: int,
    roughness_fastpath: bool,
    roughness_threshold: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Adaptive refinement over bilinear patch triangulation, per cell.

    Returns:
      area: float64 array (rows, cols)
      final_level: uint8 array (rows, cols)
      subcells_total: int32 array (rows, cols)
    """
    rows, cols = p00.shape
    out_area = np.zeros((rows, cols), dtype=np.float64)
    out_level = np.zeros((rows, cols), dtype=np.uint8)
    out_subcells = np.zeros((rows, cols), dtype=np.int32)
    if not np.any(v):
        return out_area, out_level, out_subcells

    if max_level == 0 and min_N == 1:
        # Exact match with N=1 bilinear integral (TIN 2-triangle) and avoids extra work.
        out_area = np.where(v, _tin_2tri_area_from_corners(p00, p10, p01, p11, dx, dy), 0.0)
        out_level = np.where(v, 0, 0).astype(np.uint8, copy=False)
        out_subcells = np.where(v, 0, 0).astype(np.int32, copy=False)
        return out_area, out_level, out_subcells

    flat_idx = np.flatnonzero(v.reshape(-1))
    n = int(flat_idx.size)
    if n == 0:
        return out_area, out_level, out_subcells

    p00v = p00.reshape(-1)[flat_idx].astype(np.float64, copy=False)
    p10v = p10.reshape(-1)[flat_idx].astype(np.float64, copy=False)
    p01v = p01.reshape(-1)[flat_idx].astype(np.float64, copy=False)
    p11v = p11.reshape(-1)[flat_idx].astype(np.float64, copy=False)

    final_area = np.empty((n,), dtype=np.float64)
    final_level = np.zeros((n,), dtype=np.uint8)
    subcells = np.zeros((n,), dtype=np.int64)

    planar = np.zeros((n,), dtype=bool)
    if roughness_fastpath:
        # Bilinear cross-term d=u*v coefficient; zero means the patch is planar.
        d = p00v - p10v - p01v + p11v
        sx = np.maximum(np.abs(p10v - p00v), np.abs(p11v - p01v))
        sy = np.maximum(np.abs(p01v - p00v), np.abs(p11v - p10v))
        denom = sx + sy + 1e-12
        metric = np.abs(d) / denom
        thr = 0.0 if roughness_threshold is None else float(roughness_threshold)
        planar = metric <= thr
        if np.any(planar):
            final_area[planar] = _tin_2tri_area_from_corners(
                p00v[planar], p10v[planar], p01v[planar], p11v[planar], dx, dy
            )
            final_level[planar] = 0
            subcells[planar] = 0

    active = ~planar
    if not np.any(active):
        out_area.reshape(-1)[flat_idx] = final_area
        out_level.reshape(-1)[flat_idx] = final_level
        out_subcells.reshape(-1)[flat_idx] = np.minimum(subcells, np.iinfo(np.int32).max).astype(np.int32)
        return out_area, out_level, out_subcells

    # Coarse evaluation at N=min_N.
    idx_active = np.flatnonzero(active)
    area_prev = np.zeros((n,), dtype=np.float64)
    if min_N == 1:
        area_prev[idx_active] = _tin_2tri_area_from_corners(
            p00v[idx_active], p10v[idx_active], p01v[idx_active], p11v[idx_active], dx, dy
        )
    else:
        area_prev[idx_active] = _bilinear_patch_integral_1d(
            p00v[idx_active],
            p10v[idx_active],
            p01v[idx_active],
            p11v[idx_active],
            dx,
            dy,
            N=int(min_N),
        )
    subcells[idx_active] += int(min_N) * int(min_N)

    if max_level == 0:
        final_area[idx_active] = area_prev[idx_active]
        final_level[idx_active] = 0
        out_area.reshape(-1)[flat_idx] = final_area
        out_level.reshape(-1)[flat_idx] = final_level
        out_subcells.reshape(-1)[flat_idx] = np.minimum(subcells, np.iinfo(np.int32).max).astype(np.int32)
        return out_area, out_level, out_subcells

    # Refine by doubling N each level; compare consecutive levels (L vs L+1).
    active2 = idx_active
    for fine_level in range(1, int(max_level) + 1):
        if active2.size == 0:
            break
        N = int(min_N) * (2**int(fine_level))
        if N <= 1:
            area_fine = _tin_2tri_area_from_corners(
                p00v[active2], p10v[active2], p01v[active2], p11v[active2], dx, dy
            )
        else:
            area_fine = _bilinear_patch_integral_1d(
                p00v[active2], p10v[active2], p01v[active2], p11v[active2], dx, dy, N=N
            )
        subcells[active2] += int(N) * int(N)

        err = np.abs(area_fine - area_prev[active2])
        tol = np.maximum(float(abs_tol), float(rel_tol) * area_fine)

        if fine_level == int(max_level):
            converged = np.ones_like(err, dtype=bool)
        else:
            converged = err <= tol

        if np.any(converged):
            done_idx = active2[converged]
            final_area[done_idx] = area_fine[converged]
            final_level[done_idx] = np.uint8(fine_level)

        not_done = ~converged
        if not np.any(not_done):
            active2 = np.zeros((0,), dtype=np.int64)
            break

        # Keep refining the remaining cells.
        keep_idx = active2[not_done]
        area_prev[keep_idx] = area_fine[not_done]
        active2 = keep_idx

    out_area.reshape(-1)[flat_idx] = final_area
    out_level.reshape(-1)[flat_idx] = final_level
    out_subcells.reshape(-1)[flat_idx] = np.minimum(subcells, np.iinfo(np.int32).max).astype(np.int32)
    return out_area, out_level, out_subcells


def _normalize_requested_methods(methods: list[str]) -> set[str]:
    wanted = {m.strip().lower() for m in methods}
    unknown = sorted(wanted - _SUPPORTED_METHODS)
    if unknown:
        raise ValueError(f"Unknown method(s): {unknown}. Supported: {sorted(_SUPPORTED_METHODS)}")
    return wanted


def _window_to_tuple(window: object) -> tuple[int, int, int, int]:
    return (
        int(getattr(window, "col_off")),
        int(getattr(window, "row_off")),
        int(getattr(window, "width")),
        int(getattr(window, "height")),
    )


def _chunk_window_tuples(
    windows: list[tuple[int, int, int, int]],
    *,
    workers: int,
) -> list[tuple[tuple[int, int, int, int], ...]]:
    if not windows:
        return []
    # Use smaller work batches so parallel progress updates remain responsive.
    target_tasks = max(1, min(len(windows), int(workers) * 16))
    chunk_size = max(1, int(math.ceil(len(windows) / float(target_tasks))))
    return [tuple(windows[i : i + chunk_size]) for i in range(0, len(windows), chunk_size)]


def _build_method_results(
    acc_a3d: dict[str, float],
    acc_n: dict[str, int],
    *,
    ad_level_sum: int,
    ad_refined: int,
    ad_max_level: int,
    ad_subcells: int,
    sj_level_sum: int,
    sj_refined: int,
    sj_max_level: int,
) -> dict[str, AreaResult]:
    results: dict[str, AreaResult] = {}
    for m in acc_a3d:
        if m == "adaptive_bilinear_patch_integral":
            n = int(acc_n[m])
            results[m] = _adaptive_bilinear_area_result(
                a3d=acc_a3d[m],
                valid_cells=n,
                level_sum=float(ad_level_sum),
                max_level_used=int(ad_max_level),
                refined_cells=int(ad_refined),
                total_subcells=int(ad_subcells),
            )
            continue
        if m == "sector_adaptive_jenness_integral":
            n = int(acc_n[m])
            results[m] = _sector_jenness_area_result(
                a3d=acc_a3d[m],
                valid_cells=n,
                level_sum=float(sj_level_sum),
                max_level_used=int(sj_max_level),
                refined_cells=int(sj_refined),
            )
            continue
        results[m] = AreaResult(a3d=acc_a3d[m], valid_cells=acc_n[m])
    return results


def _accumulate_window_metrics(
    z: np.ndarray,
    valid: np.ndarray,
    inner: tuple[slice, slice],
    *,
    dx: float,
    dy: float,
    wanted: set[str],
    jenness_weight: float,
    slope_method: SlopeMethod,
    integral_N: int,
    adaptive_rel_tol: float,
    adaptive_abs_tol: float,
    adaptive_max_level: int,
    adaptive_min_N: int,
    adaptive_roughness_fastpath: bool,
    adaptive_roughness_threshold: float | None,
    sector_jenness_rel_tol: float,
    sector_jenness_abs_tol: float,
    sector_jenness_max_level: int,
    sector_jenness_min_samples: int,
    include_timings: bool,
    acc_a3d: dict[str, float],
    acc_n: dict[str, int],
    acc_t: dict[str, float],
    diag: dict[str, int],
) -> None:
    from time import perf_counter

    need_corners = bool({"tin_2tri_cell", "bilinear_patch_integral", "adaptive_bilinear_patch_integral"} & wanted)

    p00 = p10 = p01 = p11 = None
    corners_valid = None
    if need_corners:
        if include_timings:
            t0 = perf_counter()
        p00, p10, p01, p11, corners_valid = _corners_from_centers(z, valid)
        if include_timings:
            t_corner = perf_counter() - t0
            if "tin_2tri_cell" in wanted:
                acc_t["tin_2tri_cell"] += t_corner
            if "bilinear_patch_integral" in wanted:
                acc_t["bilinear_patch_integral"] += t_corner
            if "adaptive_bilinear_patch_integral" in wanted:
                acc_t["adaptive_bilinear_patch_integral"] += t_corner

    if "jenness_window_8tri" in wanted:
        if include_timings:
            t0 = perf_counter()
        a, v = jenness_window_8tri_cell_areas(z, dx, dy, valid, weight=jenness_weight)
        if include_timings:
            acc_t["jenness_window_8tri"] += perf_counter() - t0
        a_in = a[inner]
        v_in = v[inner]
        acc_a3d["jenness_window_8tri"] += float(a_in[v_in].sum(dtype=np.float64))
        acc_n["jenness_window_8tri"] += int(v_in.sum())

    if "sector_adaptive_jenness_integral" in wanted:
        if include_timings:
            t0 = perf_counter()
        a, v, levels = sector_adaptive_jenness_integral_cell_areas(
            z,
            dx,
            dy,
            valid,
            rel_tol=float(sector_jenness_rel_tol),
            abs_tol=float(sector_jenness_abs_tol),
            max_level=int(sector_jenness_max_level),
            min_samples=int(sector_jenness_min_samples),
        )
        if include_timings:
            acc_t["sector_adaptive_jenness_integral"] += perf_counter() - t0
        a_in = a[inner]
        v_in = v[inner]
        acc_a3d["sector_adaptive_jenness_integral"] += float(a_in[v_in].sum(dtype=np.float64))
        acc_n["sector_adaptive_jenness_integral"] += int(v_in.sum())

        lvl_in = levels[inner][v_in]
        if lvl_in.size:
            diag["sj_level_sum"] += int(lvl_in.sum(dtype=np.int64))
            diag["sj_max_level"] = max(diag["sj_max_level"], int(lvl_in.max(initial=0)))
            diag["sj_refined"] += int((lvl_in > 0).sum())

    if "gradient_multiplier" in wanted:
        if include_timings:
            t0 = perf_counter()
        a, v = gradient_multiplier_cell_areas(z, dx, dy, valid, method=slope_method)
        if include_timings:
            acc_t["gradient_multiplier"] += perf_counter() - t0
        a_in = a[inner]
        v_in = v[inner]
        acc_a3d["gradient_multiplier"] += float(a_in[v_in].sum(dtype=np.float64))
        acc_n["gradient_multiplier"] += int(v_in.sum())

    if "tin_2tri_cell" in wanted:
        assert p00 is not None and corners_valid is not None
        if include_timings:
            t0 = perf_counter()
        dz_b = p10 - p00
        dz_c = p11 - p00
        mag1 = np.sqrt((dz_b * dy) ** 2 + (dx * (dz_b - dz_c)) ** 2 + (dx * dy) ** 2)
        dz_b2 = p11 - p00
        dz_c2 = p01 - p00
        mag2 = np.sqrt((dy * (dz_c2 - dz_b2)) ** 2 + (dx * dz_c2) ** 2 + (dx * dy) ** 2)
        areas = 0.5 * (mag1 + mag2)
        areas = np.where(corners_valid, areas, 0.0)
        if include_timings:
            acc_t["tin_2tri_cell"] += perf_counter() - t0
        v = corners_valid
        a_in = areas[inner]
        v_in = v[inner]
        acc_a3d["tin_2tri_cell"] += float(a_in[v_in].sum(dtype=np.float64))
        acc_n["tin_2tri_cell"] += int(v_in.sum())

    if "bilinear_patch_integral" in wanted:
        assert p00 is not None and corners_valid is not None
        if include_timings:
            t0 = perf_counter()
        areas = _bilinear_patch_integral_from_corners(p00, p10, p01, p11, corners_valid, dx, dy, N=integral_N)
        if include_timings:
            acc_t["bilinear_patch_integral"] += perf_counter() - t0
        v = corners_valid
        a_in = areas[inner]
        v_in = v[inner]
        acc_a3d["bilinear_patch_integral"] += float(a_in[v_in].sum(dtype=np.float64))
        acc_n["bilinear_patch_integral"] += int(v_in.sum())

    if "adaptive_bilinear_patch_integral" in wanted:
        assert p00 is not None and corners_valid is not None
        if include_timings:
            t0 = perf_counter()
        areas, levels, subcells = _adaptive_bilinear_patch_integral_from_corners(
            p00,
            p10,
            p01,
            p11,
            corners_valid,
            dx,
            dy,
            rel_tol=float(adaptive_rel_tol),
            abs_tol=float(adaptive_abs_tol),
            max_level=int(adaptive_max_level),
            min_N=int(adaptive_min_N),
            roughness_fastpath=bool(adaptive_roughness_fastpath),
            roughness_threshold=adaptive_roughness_threshold,
        )
        if include_timings:
            acc_t["adaptive_bilinear_patch_integral"] += perf_counter() - t0
        v = corners_valid
        a_in = areas[inner]
        v_in = v[inner]
        acc_a3d["adaptive_bilinear_patch_integral"] += float(a_in[v_in].sum(dtype=np.float64))
        acc_n["adaptive_bilinear_patch_integral"] += int(v_in.sum())

        lvl_in = levels[inner][v_in]
        if lvl_in.size:
            diag["ad_level_sum"] += int(lvl_in.sum(dtype=np.int64))
            diag["ad_max_level"] = max(diag["ad_max_level"], int(lvl_in.max(initial=0)))
            diag["ad_refined"] += int((lvl_in > 1).sum())
            diag["ad_subcells"] += int(subcells[inner][v_in].sum(dtype=np.int64))


def _compute_method_chunk(job: _MethodComputeJob) -> _MethodComputeChunkResult:
    import rasterio
    from rasterio.windows import Window

    wanted = set(job.methods)
    acc_a3d: dict[str, float] = {m: 0.0 for m in job.methods}
    acc_n: dict[str, int] = {m: 0 for m in job.methods}
    acc_t: dict[str, float] = {m: 0.0 for m in job.methods}
    diag = {
        "ad_level_sum": 0,
        "ad_refined": 0,
        "ad_max_level": 0,
        "ad_subcells": 0,
        "sj_level_sum": 0,
        "sj_refined": 0,
        "sj_max_level": 0,
    }

    with rasterio.open(job.raster_path) as ds:
        for w_tuple in job.windows:
            window = Window(*w_tuple)
            z, valid, inner = read_window_float32(ds, window, nodata=job.nodata, overlap=1)
            _accumulate_window_metrics(
                z,
                valid,
                inner,
                dx=job.dx,
                dy=job.dy,
                wanted=wanted,
                jenness_weight=job.jenness_weight,
                slope_method=job.slope_method,
                integral_N=job.integral_N,
                adaptive_rel_tol=job.adaptive_rel_tol,
                adaptive_abs_tol=job.adaptive_abs_tol,
                adaptive_max_level=job.adaptive_max_level,
                adaptive_min_N=job.adaptive_min_N,
                adaptive_roughness_fastpath=job.adaptive_roughness_fastpath,
                adaptive_roughness_threshold=job.adaptive_roughness_threshold,
                sector_jenness_rel_tol=job.sector_jenness_rel_tol,
                sector_jenness_abs_tol=job.sector_jenness_abs_tol,
                sector_jenness_max_level=job.sector_jenness_max_level,
                sector_jenness_min_samples=job.sector_jenness_min_samples,
                include_timings=job.include_timings,
                acc_a3d=acc_a3d,
                acc_n=acc_n,
                acc_t=acc_t,
                diag=diag,
            )

    return _MethodComputeChunkResult(
        acc_a3d=acc_a3d,
        acc_n=acc_n,
        acc_t=acc_t,
        ad_level_sum=diag["ad_level_sum"],
        ad_refined=diag["ad_refined"],
        ad_max_level=diag["ad_max_level"],
        ad_subcells=diag["ad_subcells"],
        sj_level_sum=diag["sj_level_sum"],
        sj_refined=diag["sj_refined"],
        sj_max_level=diag["sj_max_level"],
        blocks_done=len(job.windows),
    )


def _compute_methods_on_raster_impl(
    raster_path: str,
    *,
    nodata: float | None,
    methods: list[str],
    jenness_weight: float,
    slope_method: SlopeMethod,
    integral_N: int,
    adaptive_rel_tol: float,
    adaptive_abs_tol: float,
    adaptive_max_level: int,
    adaptive_min_N: int,
    adaptive_roughness_fastpath: bool,
    adaptive_roughness_threshold: float | None,
    sector_jenness_rel_tol: float,
    sector_jenness_abs_tol: float,
    sector_jenness_max_level: int,
    sector_jenness_min_samples: int,
    include_timings: bool,
    progress: ProgressFn | None,
    workers: int,
) -> tuple[dict[str, AreaResult], dict[str, float]]:
    import rasterio
    from rasterio.windows import Window

    wanted = _normalize_requested_methods(methods)
    worker_count = int(workers)
    if worker_count <= 0:
        raise ValueError(f"workers must be >= 1, got: {worker_count}")

    method_names = tuple(sorted(wanted))
    acc_a3d: dict[str, float] = {m: 0.0 for m in method_names}
    acc_n: dict[str, int] = {m: 0 for m in method_names}
    acc_t: dict[str, float] = {m: 0.0 for m in method_names}
    diag = {
        "ad_level_sum": 0,
        "ad_refined": 0,
        "ad_max_level": 0,
        "ad_subcells": 0,
        "sj_level_sum": 0,
        "sj_refined": 0,
        "sj_max_level": 0,
    }

    with rasterio.open(raster_path) as ds:
        dx = float(abs(ds.transform.a))
        dy = float(abs(ds.transform.e))
        if dx <= 0 or dy <= 0:
            raise ValueError(f"Invalid pixel sizes from transform: dx={dx}, dy={dy}")

        windows = [_window_to_tuple(w) for w in iter_block_windows(ds)]
        total_blocks = len(windows)
        if progress is not None:
            progress("compute", 0, total_blocks)

        if worker_count == 1 or total_blocks <= 1:
            block_i = 0
            for w_tuple in windows:
                block_i += 1
                window = Window(*w_tuple)
                z, valid, inner = read_window_float32(ds, window, nodata=nodata, overlap=1)
                _accumulate_window_metrics(
                    z,
                    valid,
                    inner,
                    dx=dx,
                    dy=dy,
                    wanted=wanted,
                    jenness_weight=jenness_weight,
                    slope_method=slope_method,
                    integral_N=integral_N,
                    adaptive_rel_tol=adaptive_rel_tol,
                    adaptive_abs_tol=adaptive_abs_tol,
                    adaptive_max_level=adaptive_max_level,
                    adaptive_min_N=adaptive_min_N,
                    adaptive_roughness_fastpath=adaptive_roughness_fastpath,
                    adaptive_roughness_threshold=adaptive_roughness_threshold,
                    sector_jenness_rel_tol=sector_jenness_rel_tol,
                    sector_jenness_abs_tol=sector_jenness_abs_tol,
                    sector_jenness_max_level=sector_jenness_max_level,
                    sector_jenness_min_samples=sector_jenness_min_samples,
                    include_timings=include_timings,
                    acc_a3d=acc_a3d,
                    acc_n=acc_n,
                    acc_t=acc_t,
                    diag=diag,
                )
                if progress is not None:
                    progress("compute", block_i, total_blocks)

            return (
                _build_method_results(
                    acc_a3d,
                    acc_n,
                    ad_level_sum=diag["ad_level_sum"],
                    ad_refined=diag["ad_refined"],
                    ad_max_level=diag["ad_max_level"],
                    ad_subcells=diag["ad_subcells"],
                    sj_level_sum=diag["sj_level_sum"],
                    sj_refined=diag["sj_refined"],
                    sj_max_level=diag["sj_max_level"],
                ),
                acc_t,
            )

    chunks = _chunk_window_tuples(windows, workers=worker_count)
    jobs = [
        _MethodComputeJob(
            raster_path=raster_path,
            nodata=nodata,
            windows=chunk,
            dx=dx,
            dy=dy,
            methods=method_names,
            jenness_weight=jenness_weight,
            slope_method=slope_method,
            integral_N=integral_N,
            adaptive_rel_tol=adaptive_rel_tol,
            adaptive_abs_tol=adaptive_abs_tol,
            adaptive_max_level=adaptive_max_level,
            adaptive_min_N=adaptive_min_N,
            adaptive_roughness_fastpath=adaptive_roughness_fastpath,
            adaptive_roughness_threshold=adaptive_roughness_threshold,
            sector_jenness_rel_tol=sector_jenness_rel_tol,
            sector_jenness_abs_tol=sector_jenness_abs_tol,
            sector_jenness_max_level=sector_jenness_max_level,
            sector_jenness_min_samples=sector_jenness_min_samples,
            include_timings=include_timings,
        )
        for chunk in chunks
    ]

    completed_blocks = 0
    with ProcessPoolExecutor(max_workers=min(worker_count, len(jobs))) as executor:
        future_map = {executor.submit(_compute_method_chunk, job): len(job.windows) for job in jobs}
        for future in as_completed(future_map):
            chunk_res = future.result()
            for m in method_names:
                acc_a3d[m] += float(chunk_res.acc_a3d.get(m, 0.0))
                acc_n[m] += int(chunk_res.acc_n.get(m, 0))
                acc_t[m] += float(chunk_res.acc_t.get(m, 0.0))
            diag["ad_level_sum"] += int(chunk_res.ad_level_sum)
            diag["ad_refined"] += int(chunk_res.ad_refined)
            diag["ad_max_level"] = max(diag["ad_max_level"], int(chunk_res.ad_max_level))
            diag["ad_subcells"] += int(chunk_res.ad_subcells)
            diag["sj_level_sum"] += int(chunk_res.sj_level_sum)
            diag["sj_refined"] += int(chunk_res.sj_refined)
            diag["sj_max_level"] = max(diag["sj_max_level"], int(chunk_res.sj_max_level))
            completed_blocks += int(chunk_res.blocks_done)
            if progress is not None:
                progress("compute", completed_blocks, total_blocks)

    return (
        _build_method_results(
            acc_a3d,
            acc_n,
            ad_level_sum=diag["ad_level_sum"],
            ad_refined=diag["ad_refined"],
            ad_max_level=diag["ad_max_level"],
            ad_subcells=diag["ad_subcells"],
            sj_level_sum=diag["sj_level_sum"],
            sj_refined=diag["sj_refined"],
            sj_max_level=diag["sj_max_level"],
        ),
        acc_t,
    )


def compute_methods_on_raster(
    raster_path: str,
    *,
    nodata: float | None,
    methods: list[str],
    jenness_weight: float,
    slope_method: SlopeMethod,
    integral_N: int,
    adaptive_rel_tol: float = 1e-4,
    adaptive_abs_tol: float = 0.0,
    adaptive_max_level: int = 5,
    adaptive_min_N: int = 2,
    adaptive_roughness_fastpath: bool = True,
    adaptive_roughness_threshold: float | None = None,
    sector_jenness_rel_tol: float = 1e-4,
    sector_jenness_abs_tol: float = 0.0,
    sector_jenness_max_level: int = 5,
    sector_jenness_min_samples: int = 3,
    workers: int = 1,
) -> dict[str, AreaResult]:
    """Compute multiple methods in a single blockwise pass over a raster."""
    results, _ = _compute_methods_on_raster_impl(
        raster_path,
        nodata=nodata,
        methods=methods,
        jenness_weight=jenness_weight,
        slope_method=slope_method,
        integral_N=integral_N,
        adaptive_rel_tol=adaptive_rel_tol,
        adaptive_abs_tol=adaptive_abs_tol,
        adaptive_max_level=adaptive_max_level,
        adaptive_min_N=adaptive_min_N,
        adaptive_roughness_fastpath=adaptive_roughness_fastpath,
        adaptive_roughness_threshold=adaptive_roughness_threshold,
        sector_jenness_rel_tol=sector_jenness_rel_tol,
        sector_jenness_abs_tol=sector_jenness_abs_tol,
        sector_jenness_max_level=sector_jenness_max_level,
        sector_jenness_min_samples=sector_jenness_min_samples,
        include_timings=False,
        progress=None,
        workers=workers,
    )
    return results


def compute_methods_on_raster_with_timings(
    raster_path: str,
    *,
    nodata: float | None,
    methods: list[str],
    jenness_weight: float,
    slope_method: SlopeMethod,
    integral_N: int,
    adaptive_rel_tol: float = 1e-4,
    adaptive_abs_tol: float = 0.0,
    adaptive_max_level: int = 5,
    adaptive_min_N: int = 2,
    adaptive_roughness_fastpath: bool = True,
    adaptive_roughness_threshold: float | None = None,
    sector_jenness_rel_tol: float = 1e-4,
    sector_jenness_abs_tol: float = 0.0,
    sector_jenness_max_level: int = 5,
    sector_jenness_min_samples: int = 3,
    progress: ProgressFn | None = None,
    workers: int = 1,
) -> tuple[dict[str, AreaResult], dict[str, float]]:
    """Like compute_methods_on_raster, but also returns per-method compute time (seconds).

    Timing is *compute-only* (excludes raster IO), accumulated across blocks.
    """
    return _compute_methods_on_raster_impl(
        raster_path,
        nodata=nodata,
        methods=methods,
        jenness_weight=jenness_weight,
        slope_method=slope_method,
        integral_N=integral_N,
        adaptive_rel_tol=adaptive_rel_tol,
        adaptive_abs_tol=adaptive_abs_tol,
        adaptive_max_level=adaptive_max_level,
        adaptive_min_N=adaptive_min_N,
        adaptive_roughness_fastpath=adaptive_roughness_fastpath,
        adaptive_roughness_threshold=adaptive_roughness_threshold,
        sector_jenness_rel_tol=sector_jenness_rel_tol,
        sector_jenness_abs_tol=sector_jenness_abs_tol,
        sector_jenness_max_level=sector_jenness_max_level,
        sector_jenness_min_samples=sector_jenness_min_samples,
        include_timings=True,
        progress=progress,
        workers=workers,
    )
