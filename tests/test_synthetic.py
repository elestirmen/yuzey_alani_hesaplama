from __future__ import annotations

import math

import numpy as np

import surface_area.synthetic as synthetic_mod
from surface_area.analytic_surfaces import build_analytic_surface, compute_continuous_surface_reference
from surface_area.methods import (
    compute_area_bilinear_integral,
    compute_area_gradient,
    compute_area_jenness,
    compute_area_tin_2tri,
)
from surface_area.synthetic import (
    SyntheticGrid,
    compute_reference_surface_area,
    generate_synthetic_dsm,
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


def test_analytic_tilted_plane_continuous_reference_matches_closed_form() -> None:
    surface = build_analytic_surface(
        "analytic_tilted_plane",
        extent_width_m=40.0,
        extent_height_m=24.0,
        relief=1.25,
        roughness_m=0.0,
        seed=7,
    )

    ref = compute_continuous_surface_reference(
        surface,
        extent_width_m=40.0,
        extent_height_m=24.0,
    )

    slope_x = float(surface.parameters["slope_x"])
    slope_y = float(surface.parameters["slope_y"])
    expected = 40.0 * 24.0 * math.sqrt(1.0 + slope_x * slope_x + slope_y * slope_y)

    assert ref.integration_method == "exact_closed_form"
    assert abs(ref.surface_area_m2 - expected) / expected < 1e-12


def test_generate_synthetic_dsm_is_deterministic_per_preset_and_seed() -> None:
    kwargs = dict(
        rows=48,
        cols=48,
        dx=1.0,
        dy=1.0,
        preset="waves",
        seed=17,
        relief=0.0,
        roughness_m=1.0,
    )

    z1 = generate_synthetic_dsm(**kwargs)
    z2 = generate_synthetic_dsm(**kwargs)

    assert np.array_equal(z1, z2)


def test_generate_synthetic_dsm_uses_preset_specific_roughness_patterns() -> None:
    common = dict(
        rows=48,
        cols=48,
        dx=1.0,
        dy=1.0,
        seed=11,
        relief=0.0,
    )

    plane_base = generate_synthetic_dsm(preset="plane", roughness_m=0.0, **common)
    waves_base = generate_synthetic_dsm(preset="waves", roughness_m=0.0, **common)
    plane_rough = generate_synthetic_dsm(preset="plane", roughness_m=1.0, **common)
    waves_rough = generate_synthetic_dsm(preset="waves", roughness_m=1.0, **common)

    assert np.allclose(plane_base, waves_base)
    assert not np.allclose(plane_rough, waves_rough)
    assert not np.allclose(plane_rough - plane_base, waves_rough - waves_base)


def test_generate_synthetic_dsm_parallel_roughness_is_deterministic(monkeypatch) -> None:
    monkeypatch.setattr(synthetic_mod, "_ROUGHNESS_PARALLEL_MIN_CELLS", 1)

    kwargs = dict(
        rows=48,
        cols=48,
        dx=1.0,
        dy=1.0,
        preset="mountain",
        seed=23,
        relief=0.0,
        roughness_m=1.0,
    )

    z1 = generate_synthetic_dsm(**kwargs)
    z2 = generate_synthetic_dsm(**kwargs)

    assert np.array_equal(z1, z2)
