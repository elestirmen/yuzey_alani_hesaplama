from __future__ import annotations

import math

import numpy as np

from surface_area.methods import (
    compute_area_bilinear_integral,
    compute_area_gradient,
    compute_area_jenness,
    compute_area_tin_2tri,
)
from surface_area.synthetic import (
    SyntheticGrid,
    compute_reference_surface_area,
    paraboloid,
    plane,
    reference_area_two_triangles,
    sample_surface_centers,
    sinusoid,
)


def _relative_error(est: float, ref: float) -> float:
    if ref == 0:
        return abs(est - ref)
    return abs(est - ref) / abs(ref)


def test_plane_all_methods_high_accuracy() -> None:
    grid = SyntheticGrid(rows=60, cols=60, dx=1.0, dy=1.0)
    f = plane(a=0.12, b=-0.07, c=10.0)
    z = sample_surface_centers(grid, f)
    valid = np.isfinite(z)

    x0, x1 = grid.dx, (grid.cols - 1) * grid.dx
    y0, y1 = grid.dy, (grid.rows - 1) * grid.dy
    ref = reference_area_two_triangles(
        func=f, x0=x0, x1=x1, y0=y0, y1=y1, fine_dx=grid.dx / 10.0, fine_dy=grid.dy / 10.0
    )

    expected_cells = (grid.rows - 2) * (grid.cols - 2)

    r_j = compute_area_jenness(z, grid.dx, grid.dy, valid, weight=0.25)
    assert r_j.valid_cells == expected_cells
    assert _relative_error(r_j.a3d, ref) < 1e-3

    r_t = compute_area_tin_2tri(z, grid.dx, grid.dy, valid)
    assert r_t.valid_cells == expected_cells
    assert _relative_error(r_t.a3d, ref) < 1e-3

    r_g_h = compute_area_gradient(z, grid.dx, grid.dy, valid, method="horn")
    assert r_g_h.valid_cells == expected_cells
    assert _relative_error(r_g_h.a3d, ref) < 1e-3

    r_g_zt = compute_area_gradient(z, grid.dx, grid.dy, valid, method="zt")
    assert r_g_zt.valid_cells == expected_cells
    assert _relative_error(r_g_zt.a3d, ref) < 1e-3

    r_b = compute_area_bilinear_integral(z, grid.dx, grid.dy, valid, N=5)
    assert r_b.valid_cells == expected_cells
    assert _relative_error(r_b.a3d, ref) < 1e-3


def test_sinusoid_reasonable_accuracy() -> None:
    grid = SyntheticGrid(rows=60, cols=60, dx=1.0, dy=1.0)
    kx = 2.0 * math.pi * 2.0 / grid.width
    ky = 2.0 * math.pi * 3.0 / grid.height
    f = sinusoid(amplitude=2.0, kx=kx, ky=ky)
    z = sample_surface_centers(grid, f)
    valid = np.isfinite(z)

    x0, x1 = grid.dx, (grid.cols - 1) * grid.dx
    y0, y1 = grid.dy, (grid.rows - 1) * grid.dy
    ref = reference_area_two_triangles(
        func=f, x0=x0, x1=x1, y0=y0, y1=y1, fine_dx=grid.dx / 10.0, fine_dy=grid.dy / 10.0
    )

    # Thresholds are intentionally loose; DEM discretization methods differ.
    tol = 0.05

    assert _relative_error(compute_area_jenness(z, grid.dx, grid.dy, valid, weight=0.25).a3d, ref) < tol
    assert _relative_error(compute_area_tin_2tri(z, grid.dx, grid.dy, valid).a3d, ref) < tol
    assert _relative_error(compute_area_gradient(z, grid.dx, grid.dy, valid, method="horn").a3d, ref) < tol
    assert _relative_error(compute_area_bilinear_integral(z, grid.dx, grid.dy, valid, N=5).a3d, ref) < tol


def test_paraboloid_reasonable_accuracy() -> None:
    grid = SyntheticGrid(rows=60, cols=60, dx=1.0, dy=1.0)
    f = paraboloid(scale=2000.0, x0=grid.width * 0.5, y0=grid.height * 0.5)
    z = sample_surface_centers(grid, f)
    valid = np.isfinite(z)

    x0, x1 = grid.dx, (grid.cols - 1) * grid.dx
    y0, y1 = grid.dy, (grid.rows - 1) * grid.dy
    ref = reference_area_two_triangles(
        func=f, x0=x0, x1=x1, y0=y0, y1=y1, fine_dx=grid.dx / 10.0, fine_dy=grid.dy / 10.0
    )

    tol = 0.05

    assert _relative_error(compute_area_jenness(z, grid.dx, grid.dy, valid, weight=0.25).a3d, ref) < tol
    assert _relative_error(compute_area_tin_2tri(z, grid.dx, grid.dy, valid).a3d, ref) < tol
    assert _relative_error(compute_area_gradient(z, grid.dx, grid.dy, valid, method="horn").a3d, ref) < tol
    assert _relative_error(compute_area_bilinear_integral(z, grid.dx, grid.dy, valid, N=5).a3d, ref) < tol


def test_bilinear_n1_matches_tin() -> None:
    grid = SyntheticGrid(rows=40, cols=40, dx=2.0, dy=3.0)
    f = plane(a=0.03, b=0.02, c=0.0)
    z = sample_surface_centers(grid, f)
    valid = np.isfinite(z)

    tin = compute_area_tin_2tri(z, grid.dx, grid.dy, valid).a3d
    bil = compute_area_bilinear_integral(z, grid.dx, grid.dy, valid, N=1).a3d
    assert abs(tin - bil) / tin < 1e-12


def test_reference_surface_area_reports_cell_counts_consistently() -> None:
    z = np.arange(36, dtype=np.float64).reshape(6, 6)
    z[2, 3] = np.nan

    res = compute_reference_surface_area(z, dx=1.0, dy=2.0, nodata_value=None)

    valid_samples = np.isfinite(z)
    valid_cells = (
        valid_samples[:-1, :-1]
        & valid_samples[:-1, 1:]
        & valid_samples[1:, :-1]
        & valid_samples[1:, 1:]
    )

    assert res.valid_cells == int(valid_cells.sum())
    assert res.nodata_cells == int(valid_cells.size - valid_cells.sum())
    assert res.valid_samples == int(valid_samples.sum())
    assert res.nodata_samples == int((~valid_samples).sum())
    assert abs(res.planar_area_m2 - (res.valid_cells * 2.0)) < 1e-12
