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

import math
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

from surface_area.io import block_window_count, iter_block_windows, read_window_float32

ProgressFn = Callable[[str, int, int], None]


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
    if n <= 0:
        return AreaResult(
            a3d=a3d,
            valid_cells=0,
            adaptive_avg_level=float("nan"),
            adaptive_max_level_used=0,
            adaptive_refined_cell_fraction=float("nan"),
            adaptive_total_subcells_evaluated=0,
        )

    levels_v = levels[v]
    refined = levels_v > 1
    return AreaResult(
        a3d=a3d,
        valid_cells=n,
        adaptive_avg_level=float(levels_v.mean(dtype=np.float64)),
        adaptive_max_level_used=int(levels_v.max(initial=0)),
        adaptive_refined_cell_fraction=float(refined.mean(dtype=np.float64)),
        adaptive_total_subcells_evaluated=int(subcells[v].sum(dtype=np.int64)),
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
) -> dict[str, AreaResult]:
    """Compute multiple methods in a single blockwise pass over a raster."""
    import rasterio  # local import for faster module import in tests

    wanted = {m.strip().lower() for m in methods}
    supported = {
        "jenness_window_8tri",
        "tin_2tri_cell",
        "gradient_multiplier",
        "bilinear_patch_integral",
        "adaptive_bilinear_patch_integral",
    }
    unknown = sorted(wanted - supported)
    if unknown:
        raise ValueError(f"Unknown method(s): {unknown}. Supported: {sorted(supported)}")

    acc_a3d: dict[str, float] = {m: 0.0 for m in supported if m in wanted}
    acc_n: dict[str, int] = {m: 0 for m in supported if m in wanted}

    # Adaptive diagnostics accumulators (only used when requested).
    ad_level_sum = 0
    ad_refined = 0
    ad_max_level = 0
    ad_subcells = 0

    with rasterio.open(raster_path) as ds:
        dx = float(abs(ds.transform.a))
        dy = float(abs(ds.transform.e))
        if dx <= 0 or dy <= 0:
            raise ValueError(f"Invalid pixel sizes from transform: dx={dx}, dy={dy}")

        overlap = 1
        need_corners = bool({"tin_2tri_cell", "bilinear_patch_integral", "adaptive_bilinear_patch_integral"} & wanted)

        for w in iter_block_windows(ds):
            z, valid, inner = read_window_float32(ds, w, nodata=nodata, overlap=overlap)

            # Common corner-derived values for TIN / bilinear.
            p00 = p10 = p01 = p11 = None
            corners_valid = None
            if need_corners:
                p00, p10, p01, p11, corners_valid = _corners_from_centers(z, valid)

            if "jenness_window_8tri" in wanted:
                a, v = jenness_window_8tri_cell_areas(z, dx, dy, valid, weight=jenness_weight)
                a_in = a[inner]
                v_in = v[inner]
                acc_a3d["jenness_window_8tri"] += float(a_in[v_in].sum(dtype=np.float64))
                acc_n["jenness_window_8tri"] += int(v_in.sum())

            if "gradient_multiplier" in wanted:
                a, v = gradient_multiplier_cell_areas(z, dx, dy, valid, method=slope_method)
                a_in = a[inner]
                v_in = v[inner]
                acc_a3d["gradient_multiplier"] += float(a_in[v_in].sum(dtype=np.float64))
                acc_n["gradient_multiplier"] += int(v_in.sum())

            if "tin_2tri_cell" in wanted:
                assert p00 is not None and corners_valid is not None
                dz_b = p10 - p00
                dz_c = p11 - p00
                mag1 = np.sqrt((dz_b * dy) ** 2 + (dx * (dz_b - dz_c)) ** 2 + (dx * dy) ** 2)
                dz_b2 = p11 - p00
                dz_c2 = p01 - p00
                mag2 = np.sqrt((dy * (dz_c2 - dz_b2)) ** 2 + (dx * dz_c2) ** 2 + (dx * dy) ** 2)
                areas = 0.5 * (mag1 + mag2)
                areas = np.where(corners_valid, areas, 0.0)
                v = corners_valid
                a_in = areas[inner]
                v_in = v[inner]
                acc_a3d["tin_2tri_cell"] += float(a_in[v_in].sum(dtype=np.float64))
                acc_n["tin_2tri_cell"] += int(v_in.sum())

            if "bilinear_patch_integral" in wanted:
                assert p00 is not None and corners_valid is not None
                areas = _bilinear_patch_integral_from_corners(
                    p00, p10, p01, p11, corners_valid, dx, dy, N=integral_N
                )
                v = corners_valid
                a_in = areas[inner]
                v_in = v[inner]
                acc_a3d["bilinear_patch_integral"] += float(a_in[v_in].sum(dtype=np.float64))
                acc_n["bilinear_patch_integral"] += int(v_in.sum())

            if "adaptive_bilinear_patch_integral" in wanted:
                assert p00 is not None and corners_valid is not None
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
                v = corners_valid
                a_in = areas[inner]
                v_in = v[inner]
                acc_a3d["adaptive_bilinear_patch_integral"] += float(a_in[v_in].sum(dtype=np.float64))
                acc_n["adaptive_bilinear_patch_integral"] += int(v_in.sum())

                lvl_in = levels[inner][v_in]
                if lvl_in.size:
                    ad_level_sum += int(lvl_in.sum(dtype=np.int64))
                    ad_max_level = max(ad_max_level, int(lvl_in.max(initial=0)))
                    ad_refined += int((lvl_in > 1).sum())
                    ad_subcells += int(subcells[inner][v_in].sum(dtype=np.int64))

    results: dict[str, AreaResult] = {}
    for m in acc_a3d:
        if m != "adaptive_bilinear_patch_integral":
            results[m] = AreaResult(a3d=acc_a3d[m], valid_cells=acc_n[m])
            continue

        n = int(acc_n[m])
        if n <= 0:
            results[m] = AreaResult(
                a3d=acc_a3d[m],
                valid_cells=0,
                adaptive_avg_level=float("nan"),
                adaptive_max_level_used=0,
                adaptive_refined_cell_fraction=float("nan"),
                adaptive_total_subcells_evaluated=0,
            )
        else:
            results[m] = AreaResult(
                a3d=acc_a3d[m],
                valid_cells=n,
                adaptive_avg_level=float(ad_level_sum / float(n)),
                adaptive_max_level_used=int(ad_max_level),
                adaptive_refined_cell_fraction=float(ad_refined / float(n)),
                adaptive_total_subcells_evaluated=int(ad_subcells),
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
    progress: ProgressFn | None = None,
) -> tuple[dict[str, AreaResult], dict[str, float]]:
    """Like compute_methods_on_raster, but also returns per-method compute time (seconds).

    Timing is *compute-only* (excludes raster IO), accumulated across blocks.
    """
    from time import perf_counter

    wanted = {m.strip().lower() for m in methods}
    supported = {
        "jenness_window_8tri",
        "tin_2tri_cell",
        "gradient_multiplier",
        "bilinear_patch_integral",
        "adaptive_bilinear_patch_integral",
    }
    unknown = sorted(wanted - supported)
    if unknown:
        raise ValueError(f"Unknown method(s): {unknown}. Supported: {sorted(supported)}")

    acc_a3d: dict[str, float] = {m: 0.0 for m in supported if m in wanted}
    acc_n: dict[str, int] = {m: 0 for m in supported if m in wanted}
    acc_t: dict[str, float] = {m: 0.0 for m in supported if m in wanted}

    # Adaptive diagnostics accumulators (only used when requested).
    ad_level_sum = 0
    ad_refined = 0
    ad_max_level = 0
    ad_subcells = 0

    import rasterio  # local import

    with rasterio.open(raster_path) as ds:
        dx = float(abs(ds.transform.a))
        dy = float(abs(ds.transform.e))
        if dx <= 0 or dy <= 0:
            raise ValueError(f"Invalid pixel sizes from transform: dx={dx}, dy={dy}")

        total_blocks = block_window_count(ds)
        block_i = 0

        overlap = 1
        need_corners = bool({"tin_2tri_cell", "bilinear_patch_integral", "adaptive_bilinear_patch_integral"} & wanted)

        for w in iter_block_windows(ds):
            block_i += 1
            z, valid, inner = read_window_float32(ds, w, nodata=nodata, overlap=overlap)

            # Common corner-derived values for TIN / bilinear.
            p00 = p10 = p01 = p11 = None
            corners_valid = None
            if need_corners:
                t0 = perf_counter()
                p00, p10, p01, p11, corners_valid = _corners_from_centers(z, valid)
                t_corner = perf_counter() - t0
                # Count corner derivation time towards both corner-based methods if requested.
                if "tin_2tri_cell" in wanted:
                    acc_t["tin_2tri_cell"] += t_corner
                if "bilinear_patch_integral" in wanted:
                    acc_t["bilinear_patch_integral"] += t_corner
                if "adaptive_bilinear_patch_integral" in wanted:
                    acc_t["adaptive_bilinear_patch_integral"] += t_corner

            if "jenness_window_8tri" in wanted:
                t0 = perf_counter()
                a, v = jenness_window_8tri_cell_areas(z, dx, dy, valid, weight=jenness_weight)
                acc_t["jenness_window_8tri"] += perf_counter() - t0
                a_in = a[inner]
                v_in = v[inner]
                acc_a3d["jenness_window_8tri"] += float(a_in[v_in].sum(dtype=np.float64))
                acc_n["jenness_window_8tri"] += int(v_in.sum())

            if "gradient_multiplier" in wanted:
                t0 = perf_counter()
                a, v = gradient_multiplier_cell_areas(z, dx, dy, valid, method=slope_method)
                acc_t["gradient_multiplier"] += perf_counter() - t0
                a_in = a[inner]
                v_in = v[inner]
                acc_a3d["gradient_multiplier"] += float(a_in[v_in].sum(dtype=np.float64))
                acc_n["gradient_multiplier"] += int(v_in.sum())

            if "tin_2tri_cell" in wanted:
                assert p00 is not None and corners_valid is not None
                t0 = perf_counter()
                dz_b = p10 - p00
                dz_c = p11 - p00
                mag1 = np.sqrt((dz_b * dy) ** 2 + (dx * (dz_b - dz_c)) ** 2 + (dx * dy) ** 2)
                dz_b2 = p11 - p00
                dz_c2 = p01 - p00
                mag2 = np.sqrt((dy * (dz_c2 - dz_b2)) ** 2 + (dx * dz_c2) ** 2 + (dx * dy) ** 2)
                areas = 0.5 * (mag1 + mag2)
                areas = np.where(corners_valid, areas, 0.0)
                v = corners_valid
                acc_t["tin_2tri_cell"] += perf_counter() - t0
                a_in = areas[inner]
                v_in = v[inner]
                acc_a3d["tin_2tri_cell"] += float(a_in[v_in].sum(dtype=np.float64))
                acc_n["tin_2tri_cell"] += int(v_in.sum())

            if "bilinear_patch_integral" in wanted:
                t0 = perf_counter()
                assert p00 is not None and corners_valid is not None
                areas = _bilinear_patch_integral_from_corners(
                    p00, p10, p01, p11, corners_valid, dx, dy, N=integral_N
                )
                v = corners_valid
                acc_t["bilinear_patch_integral"] += perf_counter() - t0
                a_in = areas[inner]
                v_in = v[inner]
                acc_a3d["bilinear_patch_integral"] += float(a_in[v_in].sum(dtype=np.float64))
                acc_n["bilinear_patch_integral"] += int(v_in.sum())

            if "adaptive_bilinear_patch_integral" in wanted:
                t0 = perf_counter()
                assert p00 is not None and corners_valid is not None
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
                acc_t["adaptive_bilinear_patch_integral"] += perf_counter() - t0
                v = corners_valid
                a_in = areas[inner]
                v_in = v[inner]
                acc_a3d["adaptive_bilinear_patch_integral"] += float(a_in[v_in].sum(dtype=np.float64))
                acc_n["adaptive_bilinear_patch_integral"] += int(v_in.sum())

                lvl_in = levels[inner][v_in]
                if lvl_in.size:
                    ad_level_sum += int(lvl_in.sum(dtype=np.int64))
                    ad_max_level = max(ad_max_level, int(lvl_in.max(initial=0)))
                    ad_refined += int((lvl_in > 1).sum())
                    ad_subcells += int(subcells[inner][v_in].sum(dtype=np.int64))

            if progress is not None:
                progress("compute", block_i, total_blocks)

    results: dict[str, AreaResult] = {}
    for m in acc_a3d:
        if m != "adaptive_bilinear_patch_integral":
            results[m] = AreaResult(a3d=acc_a3d[m], valid_cells=acc_n[m])
            continue

        n = int(acc_n[m])
        if n <= 0:
            results[m] = AreaResult(
                a3d=acc_a3d[m],
                valid_cells=0,
                adaptive_avg_level=float("nan"),
                adaptive_max_level_used=0,
                adaptive_refined_cell_fraction=float("nan"),
                adaptive_total_subcells_evaluated=0,
            )
        else:
            results[m] = AreaResult(
                a3d=acc_a3d[m],
                valid_cells=n,
                adaptive_avg_level=float(ad_level_sum / float(n)),
                adaptive_max_level_used=int(ad_max_level),
                adaptive_refined_cell_fraction=float(ad_refined / float(n)),
                adaptive_total_subcells_evaluated=int(ad_subcells),
            )
    return results, acc_t
