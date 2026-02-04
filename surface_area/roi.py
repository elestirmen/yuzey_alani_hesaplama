"""ROI (polygon) support for per-parcel surface area aggregation.

This module is intentionally imported only when ROI functionality is requested,
so optional dependencies (shapely / geopandas) do not affect the base CLI.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Literal

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.features import rasterize
from rasterio.windows import bounds as window_bounds
from rasterio.windows import transform as window_transform
from rasterio.warp import transform_geom

from surface_area.io import block_window_count, iter_block_windows, read_window_float32
from surface_area.methods import (
    ProgressFn,
    SlopeMethod,
    _adaptive_bilinear_patch_integral_from_corners,
    _bilinear_patch_integral_from_corners,
    _corners_from_centers,
    gradient_multiplier_cell_areas,
    jenness_window_8tri_cell_areas,
)

RoiMode = Literal["mask", "fraction"]


class RoiError(RuntimeError):
    pass


def _require_shapely() -> Any:
    try:
        import shapely  # type: ignore[import-not-found]

        return shapely
    except Exception as e:  # pragma: no cover - depends on environment
        raise RoiError(
            "ROI functionality requires 'shapely'. Install it (e.g., `pip install shapely`) "
            "or run without --roi."
        ) from e


@dataclass(frozen=True, slots=True)
class RoiGeometry:
    roi_id: str
    geometry: Any  # shapely geometry
    bounds: tuple[float, float, float, float]


def _crs_from_geojson(payload: dict[str, Any]) -> CRS | None:
    # RFC 7946: GeoJSON coordinates are WGS84 lon/lat by default (EPSG:4326),
    # and the `crs` member is deprecated. If `crs` is present, honor it.
    crs_obj = payload.get("crs")
    if isinstance(crs_obj, dict):
        if crs_obj.get("type") == "name" and isinstance(crs_obj.get("properties"), dict):
            name = crs_obj["properties"].get("name")
            if isinstance(name, str) and name.strip():
                try:
                    return CRS.from_string(name.strip())
                except Exception:
                    return None
    return CRS.from_epsg(4326)


def _load_geojson(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise RoiError(f"Failed to read GeoJSON: {path}: {e}") from e


def _iter_geojson_features(payload: dict[str, Any]) -> Iterable[dict[str, Any]]:
    t = payload.get("type")
    if t == "FeatureCollection":
        feats = payload.get("features")
        if not isinstance(feats, list):
            raise RoiError("GeoJSON FeatureCollection missing 'features' list")
        for f in feats:
            if isinstance(f, dict):
                yield f
        return
    if t == "Feature":
        yield payload
        return
    # Allow geometry-only GeoJSON.
    if isinstance(t, str) and t in {"Polygon", "MultiPolygon"}:
        yield {"type": "Feature", "geometry": payload, "properties": {}}
        return
    raise RoiError(f"Unsupported GeoJSON type: {t!r} (expected FeatureCollection/Feature/Polygon/MultiPolygon)")


def _pick_id_field(props: dict[str, Any], *, requested: str | None) -> str | None:
    if requested is not None and requested.strip():
        return requested.strip()
    if "id" in props:
        return "id"
    # Pick first property key as a fallback.
    for k in props.keys():
        return str(k)
    return None


def load_rois(
    path: str | Path,
    *,
    raster_crs: CRS | None,
    roi_id_field: str | None,
) -> list[RoiGeometry]:
    """Load ROI polygons from GeoJSON or a vector dataset (Shapefile, etc.).

    - If a GeoJSON has no explicit CRS, it is assumed to be EPSG:4326 (RFC 7946).
    - If the ROI CRS differs from the raster CRS, geometries are reprojected.
    - Multiple features with the same ROI id are unioned.
    """
    shapely = _require_shapely()
    from shapely.geometry import shape as shp_shape  # type: ignore[import-not-found]
    from shapely.ops import unary_union  # type: ignore[import-not-found]

    p = Path(path)
    if not p.exists():
        raise RoiError(f"ROI file not found: {p}")

    suffix = p.suffix.lower()
    if suffix in {".json", ".geojson"}:
        payload = _load_geojson(p)
        src_crs = _crs_from_geojson(payload)
        feats = list(_iter_geojson_features(payload))
        if not feats:
            raise RoiError(f"No features found in ROI GeoJSON: {p}")

        # Determine id field from first feature properties.
        first_props = feats[0].get("properties") if isinstance(feats[0].get("properties"), dict) else {}
        id_field = _pick_id_field(first_props, requested=roi_id_field)

        items: list[tuple[str, Any]] = []
        for i, f in enumerate(feats):
            geom = f.get("geometry")
            if not isinstance(geom, dict):
                continue
            props = f.get("properties") if isinstance(f.get("properties"), dict) else {}
            if id_field is None:
                roi_id = str(f.get("id") or i)
            else:
                roi_id = str(props.get(id_field) if id_field in props else f.get("id") or i)
            items.append((roi_id, geom))
    else:
        # Prefer geopandas (handles many formats), fall back to fiona.
        try:
            import geopandas as gpd  # type: ignore[import-not-found]

            gdf = gpd.read_file(p)
            if gdf.empty:
                raise RoiError(f"ROI file contains no features: {p}")
            src_crs = CRS.from_user_input(gdf.crs) if gdf.crs is not None else None
            # Pick id field.
            cols = [c for c in list(gdf.columns) if c != "geometry"]
            props0 = {c: True for c in cols}
            id_field = _pick_id_field(props0, requested=roi_id_field)
            if id_field is None:
                ids = [str(i) for i in range(len(gdf))]
            else:
                ids = [str(v) for v in gdf[id_field].tolist()]
            items = list(zip(ids, [geom.__geo_interface__ for geom in gdf.geometry.tolist()]))
        except Exception:
            try:
                import fiona  # type: ignore[import-not-found]
            except Exception as e:  # pragma: no cover - depends on environment
                raise RoiError(
                    "Reading Shapefile/OGR ROIs requires 'geopandas' or 'fiona'. "
                    "Install one of them, or provide a GeoJSON ROI."
                ) from e

            with fiona.open(p) as src:
                if len(src) == 0:
                    raise RoiError(f"ROI file contains no features: {p}")
                src_crs = None
                try:
                    if src.crs_wkt:
                        src_crs = CRS.from_wkt(src.crs_wkt)
                    elif src.crs:
                        src_crs = CRS.from_user_input(src.crs)
                except Exception:
                    src_crs = None

                first = next(iter(src))
                props0 = first.get("properties") if isinstance(first.get("properties"), dict) else {}
                id_field = _pick_id_field(props0, requested=roi_id_field)
                items = []
                # Include the first feature.
                feats = [first]
                feats.extend(list(src))
                for i, f in enumerate(feats):
                    geom = f.get("geometry")
                    if not isinstance(geom, dict):
                        continue
                    props = f.get("properties") if isinstance(f.get("properties"), dict) else {}
                    if id_field is None:
                        roi_id = str(f.get("id") or i)
                    else:
                        roi_id = str(props.get(id_field) if id_field in props else f.get("id") or i)
                    items.append((roi_id, geom))

    if raster_crs is None:
        raise RoiError("Raster has no CRS; cannot align ROI geometries.")

    if src_crs is None:
        raise RoiError(
            "ROI CRS is missing/unknown. Provide a GeoJSON (assumed EPSG:4326 unless 'crs' is present), "
            "or a vector dataset with a valid CRS."
        )

    # Reproject geometry mappings as needed before converting to shapely.
    if src_crs != raster_crs:
        items = [(roi_id, transform_geom(src_crs, raster_crs, geom, precision=9)) for roi_id, geom in items]

    # Group by roi_id and union multi-part ROIs.
    by_id: dict[str, list[Any]] = {}
    for roi_id, geom_mapping in items:
        try:
            g = shp_shape(geom_mapping)
        except Exception:
            continue
        if g.is_empty:
            continue
        if g.geom_type not in {"Polygon", "MultiPolygon"}:
            continue
        by_id.setdefault(str(roi_id), []).append(g)

    out: list[RoiGeometry] = []
    for roi_id, geoms in by_id.items():
        g = unary_union(geoms) if len(geoms) > 1 else geoms[0]
        if g.is_empty:
            continue
        # Fix simple invalidities.
        try:
            if not g.is_valid:  # type: ignore[attr-defined]
                g = g.buffer(0)
        except Exception:
            pass
        if g.is_empty:
            continue
        if g.geom_type not in {"Polygon", "MultiPolygon"}:
            continue
        b = tuple(float(x) for x in g.bounds)
        out.append(RoiGeometry(roi_id=str(roi_id), geometry=g, bounds=b))  # type: ignore[arg-type]

    if not out:
        raise RoiError(f"No polygon geometries found in ROI file: {p}")

    return sorted(out, key=lambda r: r.roi_id)


def _roi_indices_overlapping_window(
    rois: list[RoiGeometry], bounds_arr: np.ndarray, win_bounds: tuple[float, float, float, float]
) -> np.ndarray:
    left, bottom, right, top = (float(x) for x in win_bounds)
    minx = bounds_arr[:, 0]
    miny = bounds_arr[:, 1]
    maxx = bounds_arr[:, 2]
    maxy = bounds_arr[:, 3]
    hit = (maxx > left) & (minx < right) & (maxy > bottom) & (miny < top)
    return np.flatnonzero(hit)


def compute_roi_areas_on_raster(
    raster_path: str | Path,
    *,
    nodata: float | None,
    rois: list[RoiGeometry],
    roi_mode: RoiMode,
    roi_all_touched: bool,
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
    progress: ProgressFn | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Compute per-ROI areas for selected methods in a blockwise pass.

    Returns:
      rows: list of dicts (one per ROI per method)
      timings: compute-only per-method seconds (excludes raster IO)
    """
    mode = roi_mode.strip().lower()
    if mode not in {"mask", "fraction"}:
        raise ValueError(f"roi_mode must be mask|fraction, got {roi_mode!r}")

    wanted = [m.strip().lower() for m in methods]
    supported = {
        "jenness_window_8tri",
        "tin_2tri_cell",
        "gradient_multiplier",
        "bilinear_patch_integral",
        "adaptive_bilinear_patch_integral",
    }
    unknown = sorted(set(wanted) - supported)
    if unknown:
        raise ValueError(f"ROI computation does not support method(s): {unknown}. Supported: {sorted(supported)}")

    n_rois = int(len(rois))
    if n_rois <= 0:
        return [], {}

    roi_ids = [r.roi_id for r in rois]
    bounds_arr = np.array([r.bounds for r in rois], dtype=np.float64)

    acc_a2d: dict[str, np.ndarray] = {m: np.zeros((n_rois,), dtype=np.float64) for m in wanted}
    acc_a3d: dict[str, np.ndarray] = {m: np.zeros((n_rois,), dtype=np.float64) for m in wanted}
    acc_n: dict[str, np.ndarray] = {m: np.zeros((n_rois,), dtype=np.int64) for m in wanted}
    acc_t: dict[str, float] = {m: 0.0 for m in wanted}

    # Adaptive diagnostics accumulators (per ROI).
    ad_level_sum = np.zeros((n_rois,), dtype=np.float64)
    ad_refined = np.zeros((n_rois,), dtype=np.int64)
    ad_max_level = np.zeros((n_rois,), dtype=np.int64)
    ad_subcells = np.zeros((n_rois,), dtype=np.int64)

    roi_time = np.zeros((n_rois,), dtype=np.float64)

    need_corners = bool({"tin_2tri_cell", "bilinear_patch_integral", "adaptive_bilinear_patch_integral"} & set(wanted))

    shapely = None
    if mode == "fraction":
        shapely = _require_shapely()
        from shapely.geometry import Polygon, box  # type: ignore[import-not-found]

        pixel_poly_box = box
        PixelPolygon = Polygon

    with rasterio.open(raster_path) as ds:
        dx = float(abs(ds.transform.a))
        dy = float(abs(ds.transform.e))
        if dx <= 0 or dy <= 0:
            raise ValueError(f"Invalid pixel sizes from transform: dx={dx}, dy={dy}")

        pixel_area = float(abs(ds.transform.a * ds.transform.e - ds.transform.b * ds.transform.d))
        if pixel_area <= 0:
            pixel_area = float(dx) * float(dy)

        # For stencil methods, overlap=1 is sufficient. (Multiscale uses larger overlap, but is not ROI-supported here.)
        overlap = 1
        total_blocks = block_window_count(ds)
        block_i = 0

        # Precompute inner (eroded) geometries for fraction mode based on half-pixel diagonal.
        inner_geoms: list[Any] | None = None
        if mode == "fraction":
            assert shapely is not None
            r = 0.5 * math.hypot(dx, dy)
            inner_geoms = []
            for roi in rois:
                g = roi.geometry
                try:
                    gi = g.buffer(-r) if r > 0 else g
                except Exception:
                    gi = None
                if gi is None or getattr(gi, "is_empty", True):
                    inner_geoms.append(None)
                else:
                    inner_geoms.append(gi)

        for w in iter_block_windows(ds):
            block_i += 1
            z, valid, inner = read_window_float32(ds, w, nodata=nodata, overlap=overlap)
            h = int(w.height)
            ww = int(w.width)
            wt = window_transform(w, ds.transform)
            win_b = window_bounds(w, ds.transform)

            roi_idx = _roi_indices_overlapping_window(rois, bounds_arr, win_b)
            if roi_idx.size == 0:
                if progress is not None:
                    progress("roi", block_i, total_blocks)
                continue

            # Method areas for this block.
            a_j = v_j = None
            a_g = v_g = None
            a_t = v_t = None
            a_b = v_b = None
            a_a = v_a = None
            a_levels = a_subcells = None

            # Common corner-derived values for TIN / bilinear / adaptive.
            p00 = p10 = p01 = p11 = None
            corners_valid = None
            if need_corners:
                p00, p10, p01, p11, corners_valid = _corners_from_centers(z, valid)

            if "jenness_window_8tri" in wanted:
                t0 = perf_counter()
                a, v = jenness_window_8tri_cell_areas(z, dx, dy, valid, weight=jenness_weight)
                acc_t["jenness_window_8tri"] += perf_counter() - t0
                a_j = a[inner]
                v_j = v[inner]

            if "gradient_multiplier" in wanted:
                t0 = perf_counter()
                a, v = gradient_multiplier_cell_areas(z, dx, dy, valid, method=slope_method)
                acc_t["gradient_multiplier"] += perf_counter() - t0
                a_g = a[inner]
                v_g = v[inner]

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
                acc_t["tin_2tri_cell"] += perf_counter() - t0
                a_t = areas[inner]
                v_t = corners_valid[inner]

            if "bilinear_patch_integral" in wanted:
                assert p00 is not None and corners_valid is not None
                t0 = perf_counter()
                areas = _bilinear_patch_integral_from_corners(p00, p10, p01, p11, corners_valid, dx, dy, N=integral_N)
                acc_t["bilinear_patch_integral"] += perf_counter() - t0
                a_b = areas[inner]
                v_b = corners_valid[inner]

            if "adaptive_bilinear_patch_integral" in wanted:
                assert p00 is not None and corners_valid is not None
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
                acc_t["adaptive_bilinear_patch_integral"] += perf_counter() - t0
                a_a = areas[inner]
                v_a = corners_valid[inner]
                a_levels = levels[inner]
                a_subcells = subcells[inner]

            # ROI weighting and accumulation.
            if mode == "mask":
                t0 = perf_counter()
                shapes = [(rois[int(i)].geometry, int(i)) for i in roi_idx]
                labels = rasterize(
                    shapes=shapes,
                    out_shape=(h, ww),
                    transform=wt,
                    fill=-1,
                    dtype="int32",
                    all_touched=bool(roi_all_touched),
                )
                dt = perf_counter() - t0
                # Distribute shared rasterization time across ROIs touched by this window.
                roi_time[roi_idx] += float(dt) / float(max(1, int(roi_idx.size)))

                def _accumulate(method: str, areas_in: np.ndarray, valid_in: np.ndarray) -> None:
                    m = (labels >= 0) & valid_in
                    if not np.any(m):
                        return
                    idx2 = labels[m].astype(np.int64, copy=False)
                    acc_a3d[method] += np.bincount(idx2, weights=areas_in[m], minlength=n_rois)
                    cnt = np.bincount(idx2, minlength=n_rois).astype(np.int64, copy=False)
                    acc_a2d[method] += cnt.astype(np.float64) * pixel_area
                    acc_n[method] += cnt

                if a_j is not None and v_j is not None:
                    _accumulate("jenness_window_8tri", a_j, v_j)
                if a_g is not None and v_g is not None:
                    _accumulate("gradient_multiplier", a_g, v_g)
                if a_t is not None and v_t is not None:
                    _accumulate("tin_2tri_cell", a_t, v_t)
                if a_b is not None and v_b is not None:
                    _accumulate("bilinear_patch_integral", a_b, v_b)
                if a_a is not None and v_a is not None:
                    _accumulate("adaptive_bilinear_patch_integral", a_a, v_a)

                # Adaptive diagnostics (mask mode) keyed only on which cells contribute (labels>=0 and v_a).
                if a_a is not None and v_a is not None and a_levels is not None and a_subcells is not None:
                    m = (labels >= 0) & v_a
                    if np.any(m):
                        idx2 = labels[m].astype(np.int64, copy=False)
                        lvl = a_levels[m].astype(np.float64, copy=False)
                        ad_level_sum += np.bincount(idx2, weights=lvl, minlength=n_rois)
                        ad_refined += np.bincount(idx2, weights=(a_levels[m] > 1).astype(np.int64), minlength=n_rois)
                        ad_subcells += np.bincount(
                            idx2, weights=a_subcells[m].astype(np.float64, copy=False), minlength=n_rois
                        ).astype(np.int64)
                        np.maximum.at(ad_max_level, idx2, a_levels[m].astype(np.int64, copy=False))

            else:
                assert inner_geoms is not None
                assert shapely is not None

                # Pixel polygon factory optimized for the common axis-aligned case.
                axis_aligned = float(wt.b) == 0.0 and float(wt.d) == 0.0

                def _pixel_polygon(col: int, row: int) -> Any:
                    if axis_aligned:
                        x0, y0 = wt * (col, row)
                        x1, y1 = wt * (col + 1, row + 1)
                        return pixel_poly_box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
                    x00, y00 = wt * (col, row)
                    x10, y10 = wt * (col + 1, row)
                    x11, y11 = wt * (col + 1, row + 1)
                    x01, y01 = wt * (col, row + 1)
                    return PixelPolygon([(x00, y00), (x10, y10), (x11, y11), (x01, y01)])

                for i in roi_idx.tolist():
                    roi_i = int(i)
                    geom = rois[roi_i].geometry
                    inner_geom = inner_geoms[roi_i]

                    t0 = perf_counter()
                    touched = rasterize(
                        shapes=[geom],
                        out_shape=(h, ww),
                        transform=wt,
                        fill=0,
                        default_value=1,
                        dtype="uint8",
                        all_touched=True,
                    ).astype(bool, copy=False)
                    if inner_geom is None:
                        inside = np.zeros((h, ww), dtype=bool)
                    else:
                        inside = rasterize(
                            shapes=[inner_geom],
                            out_shape=(h, ww),
                            transform=wt,
                            fill=0,
                            default_value=1,
                            dtype="uint8",
                            all_touched=False,
                        ).astype(bool, copy=False)
                    roi_time[roi_i] += perf_counter() - t0

                    boundary = touched & ~inside

                    br, bc = np.nonzero(boundary)
                    fracs = None
                    if br.size:
                        fr = np.zeros((br.size,), dtype=np.float64)
                        for k in range(br.size):
                            poly = _pixel_polygon(int(bc[k]), int(br[k]))
                            try:
                                a = float(geom.intersection(poly).area)
                            except Exception:
                                a = 0.0
                            fr[k] = max(0.0, min(1.0, a / pixel_area)) if pixel_area > 0 else 0.0
                        fracs = fr

                    def _accum_fraction(method: str, areas_in: np.ndarray, valid_in: np.ndarray) -> None:
                        # Full inside pixels: fraction=1.
                        inside_valid = inside & valid_in
                        if np.any(inside_valid):
                            acc_a3d[method][roi_i] += float(areas_in[inside_valid].sum(dtype=np.float64))
                            n_inside = int(inside_valid.sum())
                            acc_a2d[method][roi_i] += float(n_inside) * pixel_area
                            acc_n[method][roi_i] += n_inside

                        # Boundary pixels: fraction in (0,1].
                        if fracs is None:
                            return
                        vb = valid_in[br, bc]
                        if not np.any(vb):
                            return
                        fb = fracs[vb]
                        if not np.any(fb > 0):
                            return
                        ab = areas_in[br[vb], bc[vb]]
                        acc_a3d[method][roi_i] += float((ab * fb).sum(dtype=np.float64))
                        acc_a2d[method][roi_i] += float(fb.sum(dtype=np.float64)) * pixel_area
                        acc_n[method][roi_i] += int((fb > 0).sum())

                    if a_j is not None and v_j is not None:
                        _accum_fraction("jenness_window_8tri", a_j, v_j)
                    if a_g is not None and v_g is not None:
                        _accum_fraction("gradient_multiplier", a_g, v_g)
                    if a_t is not None and v_t is not None:
                        _accum_fraction("tin_2tri_cell", a_t, v_t)
                    if a_b is not None and v_b is not None:
                        _accum_fraction("bilinear_patch_integral", a_b, v_b)
                    if a_a is not None and v_a is not None:
                        _accum_fraction("adaptive_bilinear_patch_integral", a_a, v_a)

                    # Adaptive diagnostics (fraction mode): count cells with any coverage (inside or boundary frac>0).
                    if a_levels is not None and a_subcells is not None and v_a is not None:
                        inside_valid = inside & v_a
                        if np.any(inside_valid):
                            lvl = a_levels[inside_valid].astype(np.int64, copy=False)
                            ad_level_sum[roi_i] += float(lvl.sum(dtype=np.int64))
                            ad_refined[roi_i] += int((lvl > 1).sum())
                            ad_max_level[roi_i] = max(ad_max_level[roi_i], int(lvl.max(initial=0)))
                            ad_subcells[roi_i] += int(a_subcells[inside_valid].sum(dtype=np.int64))

                        if fracs is not None and br.size:
                            vb = v_a[br, bc]
                            if np.any(vb):
                                fb = fracs[vb]
                                use = fb > 0
                                if np.any(use):
                                    lvlb = a_levels[br[vb][use], bc[vb][use]].astype(np.int64, copy=False)
                                    ad_level_sum[roi_i] += float(lvlb.sum(dtype=np.int64))
                                    ad_refined[roi_i] += int((lvlb > 1).sum())
                                    ad_max_level[roi_i] = max(ad_max_level[roi_i], int(lvlb.max(initial=0)))
                                    ad_subcells[roi_i] += int(
                                        a_subcells[br[vb][use], bc[vb][use]].sum(dtype=np.int64)
                                    )

            if progress is not None:
                progress("roi", block_i, total_blocks)

    rows: list[dict[str, Any]] = []
    for roi_i, roi_id in enumerate(roi_ids):
        for m in wanted:
            a2d = float(acc_a2d[m][roi_i])
            a3d = float(acc_a3d[m][roi_i])
            ratio = float(a3d / a2d) if a2d > 0 else float("nan")
            n = int(acc_n[m][roi_i])
            note = ";".join([f"roi_mode={mode}", f"all_touched={bool(roi_all_touched)}"])
            row: dict[str, Any] = {
                "roi_id": roi_id,
                "method": m,
                "A2D": a2d,
                "A3D": a3d,
                "ratio": ratio,
                "valid_cells": n,
                "runtime_sec": float(acc_t[m] + roi_time[roi_i]),
                "note": note,
            }
            if m == "adaptive_bilinear_patch_integral":
                if n > 0:
                    row.update(
                        {
                            "adaptive_avg_level": float(ad_level_sum[roi_i] / float(n)),
                            "adaptive_max_level_used": int(ad_max_level[roi_i]),
                            "adaptive_refined_cell_fraction": float(ad_refined[roi_i] / float(n)),
                            "adaptive_total_subcells_evaluated": int(ad_subcells[roi_i]),
                        }
                    )
                else:
                    row.update(
                        {
                            "adaptive_avg_level": float("nan"),
                            "adaptive_max_level_used": 0,
                            "adaptive_refined_cell_fraction": float("nan"),
                            "adaptive_total_subcells_evaluated": 0,
                        }
                    )
            rows.append(row)

    return rows, acc_t
