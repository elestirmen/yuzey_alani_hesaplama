from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

import main


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


def _argv_value(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def test_runconfig_resolved_dem_paths_reads_directory_tifs_only(tmp_path: Path) -> None:
    dem_dir = tmp_path / "dems"
    dem_dir.mkdir()
    z = np.zeros((4, 4), dtype=np.float64)
    _write_dem_geotiff(dem_dir / "b.TIF", z, dx=1.0, dy=1.0, crs=CRS.from_epsg(3857))
    _write_dem_geotiff(dem_dir / "a.tiff", z, dx=1.0, dy=1.0, crs=CRS.from_epsg(3857))
    (dem_dir / "ignore.txt").write_text("not a raster", encoding="utf-8")

    cfg = main.RunConfig(
        dem=str(dem_dir),
        outdir=str(tmp_path / "out"),
        gsd=["native"],
        methods=["gradient_multiplier"],
        plots=False,
    )

    assert [path.name for path in cfg.resolved_dem_paths()] == ["a.tiff", "b.TIF"]
    cfg.validate()


def test_runconfig_resolved_dem_paths_reads_demlist_entries(tmp_path: Path) -> None:
    dem_dir = tmp_path / "dems"
    dem_dir.mkdir()
    z = np.zeros((4, 4), dtype=np.float64)
    alpha = dem_dir / "alpha.tif"
    beta = dem_dir / "beta.tiff"
    _write_dem_geotiff(alpha, z, dx=1.0, dy=1.0, crs=CRS.from_epsg(3857))
    _write_dem_geotiff(beta, z, dx=1.0, dy=1.0, crs=CRS.from_epsg(3857))
    dem_list = tmp_path / "latest_generated.demlist"
    dem_list.write_text(f"# latest batch\ndems/{alpha.name}\n{beta.resolve()}\ndems/{alpha.name}\n", encoding="utf-8")

    cfg = main.RunConfig(
        dem=str(dem_list),
        outdir=str(tmp_path / "out"),
        gsd=["native"],
        methods=["gradient_multiplier"],
        plots=False,
    )

    assert [path.name for path in cfg.resolved_dem_paths()] == ["alpha.tif", "beta.tiff"]
    cfg.validate()


def test_runconfig_to_argv_rejects_directory_input(tmp_path: Path) -> None:
    dem_dir = tmp_path / "dems"
    dem_dir.mkdir()
    _write_dem_geotiff(dem_dir / "only.tif", np.zeros((4, 4), dtype=np.float64), dx=1.0, dy=1.0, crs=CRS.from_epsg(3857))

    cfg = main.RunConfig(
        dem=str(dem_dir),
        outdir=str(tmp_path / "out"),
        gsd=["native"],
        methods=["gradient_multiplier"],
        plots=False,
    )

    with pytest.raises(ValueError, match="single DEM GeoTIFF file"):
        cfg.to_argv()


def test_runconfig_to_argv_rejects_demlist_input(tmp_path: Path) -> None:
    dem_path = tmp_path / "only.tif"
    _write_dem_geotiff(dem_path, np.zeros((4, 4), dtype=np.float64), dx=1.0, dy=1.0, crs=CRS.from_epsg(3857))
    dem_list = tmp_path / "latest_generated.demlist"
    dem_list.write_text(f"{dem_path.resolve()}\n", encoding="utf-8")

    cfg = main.RunConfig(
        dem=str(dem_list),
        outdir=str(tmp_path / "out"),
        gsd=["native"],
        methods=["gradient_multiplier"],
        plots=False,
    )

    with pytest.raises(ValueError, match="single DEM GeoTIFF file"):
        cfg.to_argv()


def test_main_cli_directory_dispatches_each_tif_to_subdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dem_dir = tmp_path / "dems"
    dem_dir.mkdir()
    z = np.zeros((4, 4), dtype=np.float64)
    _write_dem_geotiff(dem_dir / "alpha.tif", z, dx=1.0, dy=1.0, crs=CRS.from_epsg(3857))
    _write_dem_geotiff(dem_dir / "beta.tiff", z, dx=1.0, dy=1.0, crs=CRS.from_epsg(3857))

    calls: list[tuple[Path, Path, list[str] | None, list[str] | None]] = []

    def fake_cmd_run(args) -> int:
        calls.append((Path(args.dem), Path(args.outdir), args.gsd, args.methods))
        return 0

    monkeypatch.setattr(main.surface_area_cli, "cmd_run", fake_cmd_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "run",
            "--dem",
            str(dem_dir),
            "--outdir",
            str(tmp_path / "out"),
            "--gsd",
            "native",
            "2",
            "--methods",
            "gradient_multiplier",
        ],
    )

    rc = main.main()

    assert rc == 0
    assert [dem.name for dem, _, _, _ in calls] == ["alpha.tif", "beta.tiff"]
    assert [outdir for _, outdir, _, _ in calls] == [tmp_path / "out" / "alpha", tmp_path / "out" / "beta"]
    assert calls[0][2] == ["native", "2"]
    assert calls[0][3] == ["gradient_multiplier"]


def test_main_cli_demlist_dispatches_each_tif_to_subdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dem_dir = tmp_path / "dems"
    dem_dir.mkdir()
    z = np.zeros((4, 4), dtype=np.float64)
    alpha = dem_dir / "alpha.tif"
    beta = dem_dir / "beta.tiff"
    _write_dem_geotiff(alpha, z, dx=1.0, dy=1.0, crs=CRS.from_epsg(3857))
    _write_dem_geotiff(beta, z, dx=1.0, dy=1.0, crs=CRS.from_epsg(3857))
    dem_list = tmp_path / "latest_generated.demlist"
    dem_list.write_text(f"{alpha.resolve()}\n{beta.resolve()}\n", encoding="utf-8")

    calls: list[tuple[Path, Path, list[str] | None, list[str] | None]] = []

    def fake_cmd_run(args) -> int:
        calls.append((Path(args.dem), Path(args.outdir), args.gsd, args.methods))
        return 0

    monkeypatch.setattr(main.surface_area_cli, "cmd_run", fake_cmd_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "run",
            "--dem",
            str(dem_list),
            "--outdir",
            str(tmp_path / "out"),
            "--gsd",
            "native",
            "--methods",
            "gradient_multiplier",
        ],
    )

    rc = main.main()

    assert rc == 0
    assert [dem.name for dem, _, _, _ in calls] == ["alpha.tif", "beta.tiff"]
    assert [outdir for _, outdir, _, _ in calls] == [tmp_path / "out" / "alpha", tmp_path / "out" / "beta"]


def test_main_default_config_directory_dispatches_each_tif_to_subdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dem_dir = tmp_path / "dems"
    dem_dir.mkdir()
    z = np.zeros((4, 4), dtype=np.float64)
    _write_dem_geotiff(dem_dir / "alpha.tif", z, dx=1.0, dy=1.0, crs=CRS.from_epsg(3857))
    _write_dem_geotiff(dem_dir / "beta.tif", z, dx=1.0, dy=1.0, crs=CRS.from_epsg(3857))

    calls: list[list[str]] = []

    def fake_cli_main(argv: list[str]) -> int:
        calls.append(list(argv))
        return 0

    monkeypatch.setattr(main.surface_area_cli, "main", fake_cli_main)
    monkeypatch.setitem(main.config, "dem", str(dem_dir))
    monkeypatch.setitem(main.config, "outdir", str(tmp_path / "batch_out"))
    monkeypatch.setitem(main.config, "gsd", ["native"])
    monkeypatch.setitem(main.config, "methods", ["gradient_multiplier"])
    monkeypatch.setitem(main.config, "plots", False)
    monkeypatch.setattr(sys, "argv", ["main.py"])

    rc = main.main()

    assert rc == 0
    assert len(calls) == 2
    assert [_argv_value(argv, "--dem") for argv in calls] == [str(dem_dir / "alpha.tif"), str(dem_dir / "beta.tif")]
    assert [_argv_value(argv, "--outdir") for argv in calls] == [
        str(tmp_path / "batch_out" / "alpha"),
        str(tmp_path / "batch_out" / "beta"),
    ]


def test_main_default_config_demlist_dispatches_each_tif_to_subdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dem_dir = tmp_path / "dems"
    dem_dir.mkdir()
    z = np.zeros((4, 4), dtype=np.float64)
    alpha = dem_dir / "alpha.tif"
    beta = dem_dir / "beta.tif"
    _write_dem_geotiff(alpha, z, dx=1.0, dy=1.0, crs=CRS.from_epsg(3857))
    _write_dem_geotiff(beta, z, dx=1.0, dy=1.0, crs=CRS.from_epsg(3857))
    dem_list = tmp_path / "latest_generated.demlist"
    dem_list.write_text(f"{alpha.resolve()}\n{beta.resolve()}\n", encoding="utf-8")

    calls: list[list[str]] = []

    def fake_cli_main(argv: list[str]) -> int:
        calls.append(list(argv))
        return 0

    monkeypatch.setattr(main.surface_area_cli, "main", fake_cli_main)
    monkeypatch.setitem(main.config, "dem", str(dem_list))
    monkeypatch.setitem(main.config, "outdir", str(tmp_path / "batch_out"))
    monkeypatch.setitem(main.config, "gsd", ["native"])
    monkeypatch.setitem(main.config, "methods", ["gradient_multiplier"])
    monkeypatch.setitem(main.config, "plots", False)
    monkeypatch.setattr(sys, "argv", ["main.py"])

    rc = main.main()

    assert rc == 0
    assert len(calls) == 2
    assert [_argv_value(argv, "--dem") for argv in calls] == [str(alpha.resolve()), str(beta.resolve())]
    assert [_argv_value(argv, "--outdir") for argv in calls] == [
        str(tmp_path / "batch_out" / "alpha"),
        str(tmp_path / "batch_out" / "beta"),
    ]


def test_main_batch_writes_batch_summary_workbook(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dem_dir = tmp_path / "dems"
    dem_dir.mkdir()
    z_alpha = np.linspace(0.0, 1.0, 64, dtype=np.float64).reshape(8, 8)
    z_beta = np.linspace(1.0, 2.0, 64, dtype=np.float64).reshape(8, 8)
    _write_dem_geotiff(dem_dir / "alpha.tif", z_alpha, dx=1.0, dy=1.0, crs=CRS.from_epsg(3857))
    _write_dem_geotiff(dem_dir / "beta.tif", z_beta, dx=1.0, dy=1.0, crs=CRS.from_epsg(3857))

    out_root = tmp_path / "batch_out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "run",
            "--dem",
            str(dem_dir),
            "--outdir",
            str(out_root),
            "--gsd",
            "1",
            "--methods",
            "gradient_multiplier",
        ],
    )

    rc = main.main()

    assert rc == 0
    workbook_path = out_root / "batch_summary.xlsx"
    assert workbook_path.exists()

    xls = pd.ExcelFile(workbook_path)
    assert {"batch_results_long", "batch_summary", "batch_runs", "pivot_runtime"} <= set(xls.sheet_names)

    df_long = pd.read_excel(workbook_path, sheet_name="batch_results_long")
    assert set(df_long["area_id"]) == {"alpha", "beta"}
    assert set(df_long["method"]) == {"gradient_multiplier"}
    assert set(df_long["comparison_reference_kind"]) == {"none"}

    df_summary = pd.read_excel(workbook_path, sheet_name="batch_summary")
    assert len(df_summary) == 1
    assert df_summary.iloc[0]["method"] == "gradient_multiplier"
    assert int(df_summary.iloc[0]["area_count"]) == 2
