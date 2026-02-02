"""Raster IO helpers (read, resample, and block-window access)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.windows import Window
from rasterio.warp import calculate_default_transform, reproject


@dataclass(frozen=True, slots=True)
class RasterInfo:
    path: Path
    crs: CRS | None
    transform: Affine
    width: int
    height: int
    nodata: float | None
    dtype: str

    @property
    def dx(self) -> float:
        return float(abs(self.transform.a))

    @property
    def dy(self) -> float:
        return float(abs(self.transform.e))


def get_raster_info(path: str | Path) -> RasterInfo:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Raster not found: {p}")
    with rasterio.open(p) as ds:
        return RasterInfo(
            path=p,
            crs=ds.crs,
            transform=ds.transform,
            width=ds.width,
            height=ds.height,
            nodata=ds.nodata,
            dtype=str(ds.dtypes[0]),
        )


def crs_linear_unit_name(crs: CRS | None) -> str | None:
    if crs is None:
        return None
    try:
        return crs.linear_units
    except Exception:
        return None


def crs_is_meter(crs: CRS | None) -> bool | None:
    """Return True/False if CRS unit is clearly meter, else None (unknown)."""
    if crs is None:
        return None
    if crs.is_geographic:
        return False
    unit = crs_linear_unit_name(crs)
    if unit is None:
        return None
    unit_norm = unit.strip().lower()
    if unit_norm in {"metre", "meter", "metres", "meters", "m"}:
        return True
    return False


def parse_resampling(name: str) -> Resampling:
    n = name.strip().lower()
    if n in {"bilinear", "linear"}:
        return Resampling.bilinear
    if n in {"nearest", "near"}:
        return Resampling.nearest
    if n in {"cubic"}:
        return Resampling.cubic
    raise ValueError(f"Unsupported resampling: {name!r} (use bilinear|nearest|cubic)")


def safe_gsd_tag(gsd_m: float) -> str:
    s = f"{gsd_m:g}"
    return s.replace(".", "p").replace("-", "m")


def resample_dem(
    *,
    src_path: str | Path,
    dst_path: str | Path,
    target_gsd_m: float,
    resampling: Resampling,
    nodata: float | None,
) -> RasterInfo:
    """Resample a DEM to a target GSD (in dataset CRS units).

    Notes:
    - Uses rasterio.warp.reproject.
    - Writes a tiled, compressed GeoTIFF to dst_path.
    - Output dtype is float32.
    """
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    if target_gsd_m <= 0:
        raise ValueError(f"target_gsd_m must be > 0, got {target_gsd_m}")

    with rasterio.open(src_path) as src:
        src_crs = src.crs
        if src_crs is None:
            raise ValueError("Input DEM has no CRS; GSD units are ambiguous.")

        src_nodata = nodata if nodata is not None else src.nodata

        dst_transform, dst_width, dst_height = calculate_default_transform(
            src_crs,
            src_crs,
            src.width,
            src.height,
            *src.bounds,
            resolution=(target_gsd_m, target_gsd_m),
        )

        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            count=1,
            dtype="float32",
            nodata=src_nodata,
            width=dst_width,
            height=dst_height,
            transform=dst_transform,
            compress="deflate",
            predictor=2,
            tiled=True,
        )

        # Choose a reasonable block size (fallback if src has none).
        block_x = int(profile.get("blockxsize") or 512)
        block_y = int(profile.get("blockysize") or 512)
        block_x = max(128, min(1024, block_x))
        block_y = max(128, min(1024, block_y))
        profile.update(blockxsize=block_x, blockysize=block_y)

        with rasterio.open(dst_path, "w", **profile) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=src_crs,
                resampling=resampling,
                src_nodata=src_nodata,
                dst_nodata=src_nodata,
                num_threads=2,
            )

    return get_raster_info(dst_path)


def _nodata_fill_value(ds: rasterio.DatasetReader, nodata: float | None) -> float:
    if nodata is not None:
        return float(nodata)
    if ds.nodata is not None:
        return float(ds.nodata)
    # We read as float32, so NaN is a safe boundless fill.
    return float("nan")


def window_with_overlap(window: Window, overlap: int) -> Window:
    if overlap <= 0:
        return window
    return Window(
        col_off=int(window.col_off) - overlap,
        row_off=int(window.row_off) - overlap,
        width=int(window.width) + 2 * overlap,
        height=int(window.height) + 2 * overlap,
    )


def read_window_float32(
    ds: rasterio.DatasetReader,
    window: Window,
    *,
    nodata: float | None,
    overlap: int = 0,
) -> tuple[np.ndarray, np.ndarray, tuple[slice, slice]]:
    """Read a raster window (with optional overlap) as float32 plus a validity mask.

    Returns:
      z: float32 array (rows, cols)
      valid: bool array, True for valid cells
      inner_slices: slices selecting the original (non-overlapped) window
    """
    ow = window_with_overlap(window, overlap)
    fill = _nodata_fill_value(ds, nodata)

    z = ds.read(
        1,
        window=ow,
        boundless=True,
        fill_value=fill,
        out_dtype="float32",
    )

    valid = np.isfinite(z)
    if nodata is not None and math.isfinite(float(nodata)):
        valid &= z != float(nodata)
    elif ds.nodata is not None and math.isfinite(float(ds.nodata)):
        valid &= z != float(ds.nodata)

    inner = (
        slice(overlap, overlap + int(window.height)),
        slice(overlap, overlap + int(window.width)),
    )
    return z, valid, inner


def iter_block_windows(ds: rasterio.DatasetReader) -> Iterator[Window]:
    """Iterate dataset windows in block order for band 1."""
    for _, w in ds.block_windows(1):
        yield w


def block_window_count(ds: rasterio.DatasetReader) -> int:
    """Return total number of block windows for band 1.

    Uses dataset block shape when available; falls back to iterating block_windows.
    """
    try:
        block_y, block_x = ds.block_shapes[0]
        if int(block_x) > 0 and int(block_y) > 0:
            return int(math.ceil(ds.width / float(block_x)) * math.ceil(ds.height / float(block_y)))
    except Exception:
        pass
    return sum(1 for _ in ds.block_windows(1))


def gaussian_overlap_pixels(sigma_px: float, *, truncate: float = 4.0) -> int:
    if sigma_px <= 0:
        return 0
    return int(math.ceil(truncate * float(sigma_px)))
