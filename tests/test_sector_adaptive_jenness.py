from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

from main import RunConfig
from surface_area.cli import DEFAULT_METHODS, METHOD_CHOICES, build_parser
from surface_area.methods import (
    _integrate_sector_jenness_cell,
    _integrate_sector_jenness_cell_from_level1,
    _sector_jenness_level1_fastpath,
    _sector_jenness_level1_presolve,
    _sector_jenness_triangle_rule,
    compute_area_sector_adaptive_jenness_integral,
    compute_methods_on_raster,
)


def _centered_xy(rows: int, cols: int, *, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    xs = (np.arange(cols, dtype=np.float64) - 0.5 * float(cols - 1)) * float(dx)
    ys = (np.arange(rows, dtype=np.float64) - 0.5 * float(rows - 1)) * float(dy)
    return np.meshgrid(xs, ys)


def _write_dem_geotiff(
    path: Path,
    z: np.ndarray,
    *,
    dx: float,
    dy: float,
    crs: CRS,
) -> None:
    rows, cols = z.shape
    transform = from_origin(0.0, float(rows) * float(dy), float(dx), float(dy))
    profile = {
        "driver": "GTiff",
        "height": rows,
        "width": cols,
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(z.astype(np.float32, copy=False), 1)


def _relative_error(est: float, ref: float) -> float:
    if ref == 0.0:
        return abs(est - ref)
    return abs(est - ref) / abs(ref)


def test_sector_adaptive_jenness_flat_plane_matches_a2d() -> None:
    rows, cols = 20, 18
    dx, dy = 2.0, 1.5
    z = np.full((rows, cols), 12.5, dtype=np.float64)
    valid = np.isfinite(z)

    res = compute_area_sector_adaptive_jenness_integral(z, dx, dy, valid)
    expected_cells = (rows - 2) * (cols - 2)
    expected_a2d = float(expected_cells) * dx * dy

    assert res.valid_cells == expected_cells
    assert abs(res.a3d - expected_a2d) < 1e-12
    assert (res.sector_jenness_max_level_used or 0) == 0
    assert (res.sector_jenness_refined_fraction or 0.0) == 0.0


def test_sector_adaptive_jenness_tilted_plane_matches_exact_area() -> None:
    rows, cols = 17, 19
    dx, dy = 1.25, 0.8
    px, py = 0.18, -0.11
    X, Y = _centered_xy(rows, cols, dx=dx, dy=dy)
    z = (px * X + py * Y + 3.0).astype(np.float64, copy=False)
    valid = np.isfinite(z)

    res = compute_area_sector_adaptive_jenness_integral(z, dx, dy, valid)
    expected_cells = (rows - 2) * (cols - 2)
    expected = float(expected_cells) * dx * dy * math.sqrt(1.0 + px * px + py * py)

    assert res.valid_cells == expected_cells
    assert _relative_error(res.a3d, expected) < 1e-12
    assert (res.sector_jenness_max_level_used or 0) == 0


def test_sector_adaptive_jenness_mild_quadratic_is_finite_stable_and_above_a2d() -> None:
    rows = cols = 25
    dx = dy = 1.0
    X, Y = _centered_xy(rows, cols, dx=dx, dy=dy)
    z = (0.004 * X * X + 0.006 * Y * Y + 0.002 * X * Y + 5.0).astype(np.float64, copy=False)
    valid = np.isfinite(z)

    coarse = compute_area_sector_adaptive_jenness_integral(
        z,
        dx,
        dy,
        valid,
        rel_tol=1e-5,
        abs_tol=0.0,
        max_level=4,
        min_samples=3,
    )
    fine = compute_area_sector_adaptive_jenness_integral(
        z,
        dx,
        dy,
        valid,
        rel_tol=1e-6,
        abs_tol=0.0,
        max_level=5,
        min_samples=7,
    )

    expected_cells = (rows - 2) * (cols - 2)
    expected_a2d = float(expected_cells) * dx * dy

    assert coarse.valid_cells == expected_cells
    assert fine.valid_cells == expected_cells
    assert math.isfinite(coarse.a3d)
    assert math.isfinite(fine.a3d)
    assert fine.a3d > expected_a2d
    assert _relative_error(coarse.a3d, fine.a3d) < 1e-3


def test_sector_adaptive_jenness_skips_incomplete_3x3_stencils() -> None:
    rows = cols = 9
    dx = dy = 1.0
    px, py = 0.07, 0.03
    X, Y = _centered_xy(rows, cols, dx=dx, dy=dy)
    z = (px * X + py * Y).astype(np.float64, copy=False)
    z[4, 4] = np.nan
    valid = np.isfinite(z)

    res = compute_area_sector_adaptive_jenness_integral(z, dx, dy, valid)
    expected_valid_cells = (rows - 2) * (cols - 2) - 9
    expected_ratio = math.sqrt(1.0 + px * px + py * py)
    a2d = float(res.valid_cells) * dx * dy

    assert res.valid_cells == expected_valid_cells
    assert math.isfinite(res.a3d)
    assert _relative_error(res.a3d / a2d, expected_ratio) < 1e-12


def test_sector_adaptive_jenness_level1_fastpath_matches_recursive() -> None:
    coeffs = np.array(
        [
            [0.004, 0.006, 0.002, 0.08, -0.03, 5.0],
            [-0.003, 0.002, -0.001, 0.04, 0.02, 1.5],
            [0.001, -0.002, 0.0005, -0.01, 0.05, -2.0],
        ],
        dtype=np.float64,
    )
    dx = dy = 1.0
    bary, weights = _sector_jenness_triangle_rule(3)

    accepted, areas = _sector_jenness_level1_fastpath(
        coeffs,
        dx,
        dy,
        bary,
        weights,
        rel_tol=1e-4,
        abs_tol=0.0,
        max_level=1,
    )

    assert accepted.all()
    for i, coeff in enumerate(coeffs):
        area, level = _integrate_sector_jenness_cell(
            coeff,
            dx,
            dy,
            bary,
            weights,
            rel_tol=1e-4,
            abs_tol=0.0,
            max_level=1,
        )
        assert level == 1
        assert abs(areas[i] - area) < 1e-12


def test_sector_adaptive_jenness_level1_resume_matches_recursive() -> None:
    coeffs = np.array(
        [
            [0.8, 0.6, 0.4, 0.08, -0.03, 5.0],
            [-0.7, 0.5, -0.35, 0.04, 0.02, 1.5],
            [0.65, -0.45, 0.3, -0.01, 0.05, -2.0],
        ],
        dtype=np.float64,
    )
    dx = dy = 1.0
    bary, weights = _sector_jenness_triangle_rule(3)

    accepted, _, sector_accepted, sector_fine, sector_child_coarse = _sector_jenness_level1_presolve(
        coeffs,
        dx,
        dy,
        bary,
        weights,
        rel_tol=1e-4,
        abs_tol=0.0,
        max_level=5,
    )

    assert (~accepted).any()
    for i, coeff in enumerate(coeffs):
        expected_area, expected_level = _integrate_sector_jenness_cell(
            coeff,
            dx,
            dy,
            bary,
            weights,
            rel_tol=1e-4,
            abs_tol=0.0,
            max_level=5,
        )
        resumed_area, resumed_level = _integrate_sector_jenness_cell_from_level1(
            coeff,
            dx,
            dy,
            bary,
            weights,
            rel_tol=1e-4,
            abs_tol=0.0,
            max_level=5,
            sector_accepted=sector_accepted[:, i],
            sector_fine=sector_fine[:, i],
            sector_child_coarse=sector_child_coarse[:, :, i],
        )
        assert resumed_level == expected_level
        assert abs(resumed_area - expected_area) < 1e-12


def test_sector_adaptive_jenness_cli_smoke_writes_results_and_metadata(tmp_path: Path) -> None:
    from surface_area.cli import main as cli_main

    rows = cols = 12
    dx = dy = 1.0
    X, Y = _centered_xy(rows, cols, dx=dx, dy=dy)
    z = (0.08 * X - 0.04 * Y + 2.0).astype(np.float64, copy=False)
    dem_path = tmp_path / "dem.tif"
    outdir = tmp_path / "out"
    _write_dem_geotiff(dem_path, z, dx=dx, dy=dy, crs=CRS.from_epsg(3857))

    rc = cli_main(
        [
            "run",
            "--dem",
            str(dem_path),
            "--outdir",
            str(outdir),
            "--gsd",
            "1",
            "--methods",
            "sector_adaptive_jenness_integral",
            "--sector_jenness_rel_tol",
            "1e-5",
            "--sector_jenness_abs_tol",
            "0",
            "--sector_jenness_max_level",
            "4",
            "--sector_jenness_min_samples",
            "7",
        ]
    )
    assert rc == 0

    results_path = outdir / "results_long.csv"
    info_path = outdir / "run_info.json"
    assert results_path.exists()
    assert info_path.exists()

    df = pd.read_csv(results_path)
    assert set(df["method"]) == {"sector_adaptive_jenness_integral"}
    assert {"sector_jenness_avg_level", "sector_jenness_max_level_used", "sector_jenness_refined_fraction"} <= set(
        df.columns
    )
    row = df.iloc[0]
    assert row["method"] == "sector_adaptive_jenness_integral"
    assert math.isfinite(float(row["A3D"]))
    assert "min_samples=7" in str(row["note"])
    assert "max_level=4" in str(row["note"])

    payload = json.loads(info_path.read_text(encoding="utf-8"))
    params = payload["params"]
    assert params["methods"] == ["sector_adaptive_jenness_integral"]
    assert params["sector_jenness_rel_tol"] == 1e-5
    assert params["sector_jenness_abs_tol"] == 0.0
    assert params["sector_jenness_max_level"] == 4
    assert params["sector_jenness_min_samples"] == 7


def test_sector_adaptive_jenness_runconfig_to_argv_includes_new_flags(tmp_path: Path) -> None:
    dem_path = tmp_path / "dem.tif"
    _write_dem_geotiff(
        dem_path,
        np.zeros((5, 5), dtype=np.float64),
        dx=1.0,
        dy=1.0,
        crs=CRS.from_epsg(3857),
    )

    cfg = RunConfig(
        dem=str(dem_path),
        outdir=str(tmp_path / "out"),
        gsd=[1.0],
        methods=["sector_adaptive_jenness_integral"],
        plots=False,
        sector_jenness_rel_tol=1e-5,
        sector_jenness_abs_tol=1e-6,
        sector_jenness_max_level=4,
        sector_jenness_min_samples=7,
    )
    argv = cfg.to_argv()

    assert "--sector_jenness_rel_tol" in argv
    assert "--sector_jenness_abs_tol" in argv
    assert "--sector_jenness_max_level" in argv
    assert "--sector_jenness_min_samples" in argv
    assert "sector_adaptive_jenness_integral" in argv
    assert argv[argv.index("--sector_jenness_rel_tol") + 1] == "1e-05"
    assert argv[argv.index("--sector_jenness_abs_tol") + 1] == "1e-06"
    assert argv[argv.index("--sector_jenness_max_level") + 1] == "4"
    assert argv[argv.index("--sector_jenness_min_samples") + 1] == "7"


def test_sector_adaptive_jenness_keeps_existing_registry_and_methods_compatible(tmp_path: Path) -> None:
    legacy_methods = [
        "jenness_window_8tri",
        "tin_2tri_cell",
        "gradient_multiplier",
        "bilinear_patch_integral",
        "adaptive_bilinear_patch_integral",
    ]
    for method in legacy_methods:
        assert method in METHOD_CHOICES
    assert "sector_adaptive_jenness_integral" in METHOD_CHOICES

    parser = build_parser()
    args = parser.parse_args(
        [
            "run",
            "--dem",
            "dem.tif",
            "--outdir",
            "out",
            "--methods",
            *legacy_methods,
            "sector_adaptive_jenness_integral",
            "--sector_jenness_rel_tol",
            "1e-5",
            "--sector_jenness_max_level",
            "4",
            "--sector_jenness_min_samples",
            "7",
        ]
    )
    assert args.methods[-1] == "sector_adaptive_jenness_integral"

    rows = cols = 12
    dx = dy = 1.0
    X, Y = _centered_xy(rows, cols, dx=dx, dy=dy)
    z = (0.08 * X - 0.04 * Y + 2.0).astype(np.float64, copy=False)
    dem_path = tmp_path / "dem.tif"
    _write_dem_geotiff(dem_path, z, dx=dx, dy=dy, crs=CRS.from_epsg(3857))

    methods = legacy_methods + ["sector_adaptive_jenness_integral"]
    results = compute_methods_on_raster(
        str(dem_path),
        nodata=None,
        methods=methods,
        jenness_weight=0.25,
        slope_method="horn",
        integral_N=5,
        sector_jenness_rel_tol=1e-5,
        sector_jenness_abs_tol=0.0,
        sector_jenness_max_level=4,
        sector_jenness_min_samples=7,
    )

    assert set(results) == set(methods)
    assert all(results[m].valid_cells > 0 for m in methods)


def test_default_methods_enable_all_six_base_methods() -> None:
    assert DEFAULT_METHODS == [
        "jenness_window_8tri",
        "sector_adaptive_jenness_integral",
        "tin_2tri_cell",
        "gradient_multiplier",
        "bilinear_patch_integral",
        "adaptive_bilinear_patch_integral",
    ]
    assert "multiscale_decomposed_area" not in DEFAULT_METHODS
