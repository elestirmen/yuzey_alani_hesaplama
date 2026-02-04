from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

from surface_area.io import get_raster_info
from surface_area.methods import (
    compute_area_adaptive_bilinear_integral,
    compute_area_bilinear_integral,
    compute_methods_on_raster,
)


def _relative_error(est: float, ref: float) -> float:
    if ref == 0:
        return abs(est - ref)
    return abs(est - ref) / abs(ref)


def _write_dem_geotiff(
    path: Path,
    z: np.ndarray,
    *,
    dx: float,
    dy: float,
    crs: CRS,
    nodata: float | None = None,
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
        "nodata": nodata,
        "compress": "deflate",
        "predictor": 2,
        "tiled": False,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(z.astype(np.float32, copy=False), 1)


def _plane_dem(rows: int, cols: int, *, dx: float, dy: float, a: float, b: float, c: float) -> np.ndarray:
    xs = (np.arange(cols, dtype=np.float64) + 0.5) * float(dx)
    ys = float(rows) * float(dy) - (np.arange(rows, dtype=np.float64) + 0.5) * float(dy)
    X, Y = np.meshgrid(xs, ys)
    return (a * X + b * Y + c).astype(np.float64, copy=False)


def _write_geojson_polygon(path: Path, *, roi_id: str, polygon_coords: list[list[tuple[float, float]]], crs: str) -> None:
    payload = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": crs}},
        "features": [
            {
                "type": "Feature",
                "properties": {"id": roi_id},
                "geometry": {"type": "Polygon", "coordinates": polygon_coords},
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_adaptive_bilinear_plane_ratio_and_low_refinement() -> None:
    rows, cols = 20, 20
    dx = dy = 1.0
    a, b, c = 0.12, -0.07, 10.0
    z = _plane_dem(rows, cols, dx=dx, dy=dy, a=a, b=b, c=c)
    valid = np.isfinite(z)

    r = compute_area_adaptive_bilinear_integral(z, dx, dy, valid)
    A2D = float(r.valid_cells) * dx * dy
    ratio = float(r.a3d / A2D)
    expected = math.sqrt(1.0 + a * a + b * b)

    assert _relative_error(ratio, expected) < 1e-3
    assert (r.adaptive_max_level_used or 0) <= 1
    assert (r.adaptive_refined_cell_fraction or 0.0) == 0.0


def test_adaptive_bilinear_sinusoid_converges_and_saves_work() -> None:
    rows, cols = 22, 22
    dx = dy = 1.0
    xs = (np.arange(cols, dtype=np.float64) + 0.5) * dx
    ys = (np.arange(rows, dtype=np.float64) + 0.5) * dy
    X, Y = np.meshgrid(xs, ys)
    kx = 2.0 * math.pi * 2.0 / float(cols * dx)
    ky = 2.0 * math.pi * 3.0 / float(rows * dy)
    z = (2.0 * np.sin(kx * X) * np.sin(ky * Y)).astype(np.float64, copy=False)
    valid = np.isfinite(z)

    # High-N fixed subdivision reference (still using the same bilinear-from-corners model).
    ref_N = 80
    ref = compute_area_bilinear_integral(z, dx, dy, valid, N=ref_N).a3d

    r = compute_area_adaptive_bilinear_integral(z, dx, dy, valid)
    assert _relative_error(r.a3d, ref) < 5e-3

    fixed_subcells = int(r.valid_cells) * int(ref_N) * int(ref_N)
    assert (r.adaptive_total_subcells_evaluated or 0) < fixed_subcells


def test_roi_fraction_partial_pixel_rectangle_matches_expected_fraction(tmp_path: Path) -> None:
    # Small planar DEM written to GeoTIFF.
    rows, cols = 10, 10
    dx = dy = 1.0
    a, b, c = 0.1, 0.05, 0.0
    z = _plane_dem(rows, cols, dx=dx, dy=dy, a=a, b=b, c=c)
    crs = CRS.from_epsg(3857)
    dem_path = tmp_path / "dem.tif"
    _write_dem_geotiff(dem_path, z, dx=dx, dy=dy, crs=crs, nodata=None)

    # ROI cuts through pixel column 5 at x=5.5 -> half coverage for that column.
    roi_path = tmp_path / "roi.geojson"
    x0, x1 = 0.0, 5.5
    y0, y1 = 0.0, float(rows) * float(dy)
    poly = [[(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]]
    _write_geojson_polygon(roi_path, roi_id="p1", polygon_coords=poly, crs="EPSG:3857")

    from surface_area.roi import compute_roi_areas_on_raster, load_rois

    info = get_raster_info(dem_path)
    rois = load_rois(roi_path, raster_crs=info.crs, roi_id_field=None)
    rows_out, _ = compute_roi_areas_on_raster(
        dem_path,
        nodata=None,
        rois=rois,
        roi_mode="fraction",
        roi_all_touched=False,
        methods=["adaptive_bilinear_patch_integral"],
        jenness_weight=0.25,
        slope_method="horn",
        integral_N=5,
        adaptive_rel_tol=1e-4,
        adaptive_abs_tol=0.0,
        adaptive_max_level=5,
        adaptive_min_N=2,
        adaptive_roughness_fastpath=True,
        adaptive_roughness_threshold=None,
        progress=None,
    )

    assert len(rows_out) == 1
    roi_row = rows_out[0]

    # Global result on the same raster/method for comparison.
    global_res = compute_methods_on_raster(
        str(dem_path),
        nodata=None,
        methods=["adaptive_bilinear_patch_integral"],
        jenness_weight=0.25,
        slope_method="horn",
        integral_N=5,
    )["adaptive_bilinear_patch_integral"]
    A2D_global = float(global_res.valid_cells) * dx * dy

    # Corners-based methods exclude a 1-cell border: valid cols/rows are 1..8.
    # Within those, ROI covers cols 1..4 fully and col 5 half => 4.5/8 coverage.
    expected_A2D = 8.0 * 4.5 * dx * dy
    assert abs(float(roi_row["A2D"]) - expected_A2D) < 1e-9
    assert _relative_error(float(roi_row["A2D"]) / A2D_global, 4.5 / 8.0) < 1e-12

    # For a plane, both A2D and A3D scale linearly with the coverage fraction.
    expected_ratio = math.sqrt(1.0 + a * a + b * b)
    assert _relative_error(float(roi_row["ratio"]), expected_ratio) < 1e-3
    assert _relative_error(float(roi_row["A3D"]) / float(global_res.a3d), 4.5 / 8.0) < 1e-3


def test_adaptive_bilinear_nodata_stripe_is_skipped() -> None:
    rows, cols = 30, 30
    dx = dy = 1.0
    a, b, c = 0.05, -0.02, 3.0
    z = _plane_dem(rows, cols, dx=dx, dy=dy, a=a, b=b, c=c)
    z[:, 14] = np.nan
    valid = np.isfinite(z)

    r = compute_area_adaptive_bilinear_integral(z, dx, dy, valid)
    assert 0 < r.valid_cells < (rows - 2) * (cols - 2)
    assert math.isfinite(r.a3d)

    A2D = float(r.valid_cells) * dx * dy
    ratio = float(r.a3d / A2D)
    expected = math.sqrt(1.0 + a * a + b * b)
    assert _relative_error(ratio, expected) < 1e-3

