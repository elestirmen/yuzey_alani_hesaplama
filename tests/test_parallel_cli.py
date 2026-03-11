from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

from main import RunConfig


def _write_dem_geotiff(path: Path, z: np.ndarray, *, dx: float, dy: float, crs: CRS) -> None:
    rows, cols = z.shape
    profile = {
        "driver": "GTiff",
        "height": rows,
        "width": cols,
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": from_origin(0.0, float(rows) * float(dy), float(dx), float(dy)),
        "nodata": None,
        "compress": "deflate",
        "predictor": 2,
        "tiled": False,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(z.astype(np.float32, copy=False), 1)


def _demo_dem(rows: int, cols: int, *, dx: float, dy: float) -> np.ndarray:
    xs = (np.arange(cols, dtype=np.float64) + 0.5) * float(dx)
    ys = (np.arange(rows, dtype=np.float64) + 0.5) * float(dy)
    x_grid, y_grid = np.meshgrid(xs, ys)
    plane = 0.08 * x_grid - 0.03 * y_grid + 10.0
    waves = 0.6 * np.sin(x_grid / 3.0) * np.cos(y_grid / 4.0)
    return (plane + waves).astype(np.float64, copy=False)


def test_runconfig_to_argv_includes_workers(tmp_path: Path) -> None:
    dem_path = tmp_path / "dem.tif"
    _write_dem_geotiff(
        dem_path,
        np.zeros((8, 8), dtype=np.float64),
        dx=1.0,
        dy=1.0,
        crs=CRS.from_epsg(3857),
    )

    cfg = RunConfig(
        dem=str(dem_path),
        outdir=str(tmp_path / "out"),
        gsd=[1.0, 2.0],
        methods=["gradient_multiplier"],
        plots=False,
        workers=3,
    )
    argv = cfg.to_argv()

    assert "--workers" in argv
    assert argv[argv.index("--workers") + 1] == "3"


def test_cli_parallel_workers_matches_serial(tmp_path: Path) -> None:
    from surface_area.cli import main as cli_main

    dem_path = tmp_path / "dem.tif"
    out_serial = tmp_path / "out_serial"
    out_parallel = tmp_path / "out_parallel"
    z = _demo_dem(40, 40, dx=1.0, dy=1.0)
    _write_dem_geotiff(dem_path, z, dx=1.0, dy=1.0, crs=CRS.from_epsg(3857))

    base_args = [
        "run",
        "--dem",
        str(dem_path),
        "--gsd",
        "1",
        "2",
        "--methods",
        "gradient_multiplier",
        "tin_2tri_cell",
    ]

    rc_serial = cli_main([*base_args, "--outdir", str(out_serial), "--workers", "1"])
    rc_parallel = cli_main([*base_args, "--outdir", str(out_parallel), "--workers", "2"])

    assert rc_serial == 0
    assert rc_parallel == 0

    cols = ["gsd_m", "method", "dx", "dy", "A2D", "A3D", "ratio", "valid_cells", "note"]
    df_serial = pd.read_csv(out_serial / "results_long.csv")[cols].sort_values(["gsd_m", "method"]).reset_index(
        drop=True
    )
    df_parallel = pd.read_csv(out_parallel / "results_long.csv")[cols].sort_values(["gsd_m", "method"]).reset_index(
        drop=True
    )

    assert list(df_serial["method"]) == list(df_parallel["method"])
    assert list(df_serial["valid_cells"]) == list(df_parallel["valid_cells"])
    assert list(df_serial["note"]) == list(df_parallel["note"])
    for col in ["gsd_m", "dx", "dy", "A2D", "A3D", "ratio"]:
        assert np.allclose(df_serial[col].to_numpy(), df_parallel[col].to_numpy(), rtol=0.0, atol=1e-9)

    info_parallel = json.loads((out_parallel / "run_info.json").read_text(encoding="utf-8"))
    assert info_parallel["params"]["workers"] == 2


def test_compute_methods_on_raster_workers_matches_serial(tmp_path: Path) -> None:
    from surface_area.methods import compute_methods_on_raster

    dem_path = tmp_path / "dem_methods.tif"
    z = _demo_dem(36, 36, dx=1.0, dy=1.0)
    _write_dem_geotiff(dem_path, z, dx=1.0, dy=1.0, crs=CRS.from_epsg(3857))

    serial = compute_methods_on_raster(
        str(dem_path),
        nodata=None,
        methods=["gradient_multiplier", "tin_2tri_cell", "adaptive_bilinear_patch_integral"],
        jenness_weight=0.25,
        slope_method="horn",
        integral_N=5,
        workers=1,
    )
    parallel = compute_methods_on_raster(
        str(dem_path),
        nodata=None,
        methods=["gradient_multiplier", "tin_2tri_cell", "adaptive_bilinear_patch_integral"],
        jenness_weight=0.25,
        slope_method="horn",
        integral_N=5,
        workers=2,
    )

    assert serial.keys() == parallel.keys()
    for key in serial:
        s = serial[key]
        p = parallel[key]
        assert s.valid_cells == p.valid_cells
        assert np.isclose(s.a3d, p.a3d, rtol=0.0, atol=1e-9)
        assert s.adaptive_max_level_used == p.adaptive_max_level_used
        assert s.adaptive_total_subcells_evaluated == p.adaptive_total_subcells_evaluated
