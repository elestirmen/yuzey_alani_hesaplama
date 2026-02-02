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


def compute_methods_on_raster(
    raster_path: str,
    *,
    nodata: float | None,
    methods: list[str],
    jenness_weight: float,
    slope_method: SlopeMethod,
    integral_N: int,
) -> dict[str, AreaResult]:
    """Compute multiple methods in a single blockwise pass over a raster."""
    import rasterio  # local import for faster module import in tests

    wanted = {m.strip().lower() for m in methods}
    supported = {
        "jenness_window_8tri",
        "tin_2tri_cell",
        "gradient_multiplier",
        "bilinear_patch_integral",
    }
    unknown = sorted(wanted - supported)
    if unknown:
        raise ValueError(f"Unknown method(s): {unknown}. Supported: {sorted(supported)}")

    acc_a3d: dict[str, float] = {m: 0.0 for m in supported if m in wanted}
    acc_n: dict[str, int] = {m: 0 for m in supported if m in wanted}

    with rasterio.open(raster_path) as ds:
        dx = float(abs(ds.transform.a))
        dy = float(abs(ds.transform.e))
        if dx <= 0 or dy <= 0:
            raise ValueError(f"Invalid pixel sizes from transform: dx={dx}, dy={dy}")

        overlap = 1
        need_corners = bool({"tin_2tri_cell", "bilinear_patch_integral"} & wanted)

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

    return {m: AreaResult(a3d=acc_a3d[m], valid_cells=acc_n[m]) for m in acc_a3d}


def compute_methods_on_raster_with_timings(
    raster_path: str,
    *,
    nodata: float | None,
    methods: list[str],
    jenness_weight: float,
    slope_method: SlopeMethod,
    integral_N: int,
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
    }
    unknown = sorted(wanted - supported)
    if unknown:
        raise ValueError(f"Unknown method(s): {unknown}. Supported: {sorted(supported)}")

    acc_a3d: dict[str, float] = {m: 0.0 for m in supported if m in wanted}
    acc_n: dict[str, int] = {m: 0 for m in supported if m in wanted}
    acc_t: dict[str, float] = {m: 0.0 for m in supported if m in wanted}

    import rasterio  # local import

    with rasterio.open(raster_path) as ds:
        dx = float(abs(ds.transform.a))
        dy = float(abs(ds.transform.e))
        if dx <= 0 or dy <= 0:
            raise ValueError(f"Invalid pixel sizes from transform: dx={dx}, dy={dy}")

        total_blocks = block_window_count(ds)
        block_i = 0

        overlap = 1
        need_corners = bool({"tin_2tri_cell", "bilinear_patch_integral"} & wanted)

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

            if progress is not None:
                progress("compute", block_i, total_blocks)

    results = {m: AreaResult(a3d=acc_a3d[m], valid_cells=acc_n[m]) for m in acc_a3d}
    return results, acc_t
