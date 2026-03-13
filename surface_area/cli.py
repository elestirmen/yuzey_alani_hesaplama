from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

from surface_area import __version__
from surface_area.io import (
    RasterInfo,
    crs_is_meter,
    crs_linear_unit_name,
    get_raster_info,
    parse_resampling,
    resample_dem,
    safe_gsd_tag,
    write_dem_float32_geotiff,
)
from surface_area.methods import AreaResult, SlopeMethod, compute_methods_on_raster_with_timings
from surface_area.multiscale import compute_multiscale_on_raster
from surface_area.plotting import plot_a3d_vs_gsd, plot_micro_ratio_vs_gsd, plot_ratio_vs_gsd
from surface_area.progress import ProgressPrinter
from surface_area.synthetic import SYNTHETIC_PRESETS, generate_synthetic_dsm


GSD_NATIVE_TOKEN = "native"
DEFAULT_GSD_LIST = [GSD_NATIVE_TOKEN, 0.1, 0.5, 1, 2, 5, 10, 20, 50]

METHOD_CHOICES = [
    "jenness_window_8tri",
    "sector_adaptive_jenness_integral",
    "tin_2tri_cell",
    "gradient_multiplier",
    "bilinear_patch_integral",
    "adaptive_bilinear_patch_integral",
    "multiscale_decomposed_area",
]

DEFAULT_METHODS = [
    "jenness_window_8tri",
    "sector_adaptive_jenness_integral",
    "tin_2tri_cell",
    "gradient_multiplier",
]

_RESULTS_LONG_BASE_COLUMNS = [
    "gsd_m",
    "dx",
    "dy",
    "method",
    "A2D",
    "A3D",
    "ratio",
    "valid_cells",
    "runtime_sec",
    "resample_runtime_sec",
    "note",
]
_RESULTS_ROI_BASE_COLUMNS = [
    "gsd_m",
    "dx",
    "dy",
    "roi_id",
    "method",
    "A2D",
    "A3D",
    "ratio",
    "valid_cells",
    "runtime_sec",
    "resample_runtime_sec",
    "note",
]
_RESULTS_REFERENCE_COLUMNS = ["A3D_ref", "A3D_diff", "A3D_rel_err"]
_RESULTS_SYNTHETIC_REFERENCE_COLUMNS = [
    "synthetic_native_ref_A2D",
    "synthetic_native_ref_A3D",
    "synthetic_native_ref_ratio",
    "A3D_synthetic_native_ref_diff",
    "A3D_synthetic_native_ref_rel_err",
    "continuous_gt_A2D",
    "continuous_gt_A3D",
    "continuous_gt_ratio",
    "A3D_continuous_gt_diff",
    "A3D_continuous_gt_rel_err",
]
_RESULTS_MULTISCALE_COLUMNS = ["a_topo", "a_micro", "a_total", "micro_ratio", "sigma_m"]
_RESULTS_ADAPTIVE_COLUMNS = [
    "adaptive_avg_level",
    "adaptive_max_level_used",
    "adaptive_refined_cell_fraction",
    "adaptive_total_subcells_evaluated",
]
_RESULTS_SECTOR_COLUMNS = [
    "sector_jenness_avg_level",
    "sector_jenness_max_level_used",
    "sector_jenness_refined_fraction",
]


@dataclass(frozen=True, slots=True)
class _ResolvedGsdTarget:
    gsd_m: float
    use_native_grid: bool
    label: str


def _format_gsd_value(value: str | float) -> str:
    if isinstance(value, str):
        return value
    return f"{float(value):g}"


def _grid_resolution_key(dx: float, dy: float) -> tuple[str, str]:
    return (f"{float(dx):.12g}", f"{float(dy):.12g}")


def _load_synthetic_reference_payload(dem_path: Path) -> dict[str, Any] | None:
    sidecar_path = dem_path.with_suffix(".reference.json")
    if not sidecar_path.exists():
        return None
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    payload["_sidecar_path"] = str(sidecar_path)
    return payload


def _synthetic_reference_summary(payload: dict[str, Any]) -> dict[str, Any]:
    continuous = payload.get("continuous_ground_truth")
    native = payload.get("native_grid_reference")
    multi_resolution = payload.get("multi_resolution")
    return {
        "sidecar_path": payload.get("_sidecar_path"),
        "terrain_family": payload.get("terrain_family"),
        "has_continuous_ground_truth": isinstance(continuous, dict),
        "continuous_gt_surface_area_m2": None if not isinstance(continuous, dict) else continuous.get("surface_area_m2"),
        "continuous_gt_planar_area_m2": None if not isinstance(continuous, dict) else continuous.get("planar_area_m2"),
        "native_reference_surface_area_m2": None if not isinstance(native, dict) else native.get("surface_area_m2"),
        "resolution_count": len(multi_resolution) if isinstance(multi_resolution, list) else 0,
    }


def _append_synthetic_reference_columns(df_long: pd.DataFrame, payload: dict[str, Any] | None) -> pd.DataFrame:
    if payload is None or df_long.empty:
        return df_long

    continuous = payload.get("continuous_ground_truth")
    if isinstance(continuous, dict):
        gt_a2d = continuous.get("planar_area_m2")
        gt_a3d = continuous.get("surface_area_m2")
        gt_ratio = continuous.get("surface_ratio")
        if gt_a2d is not None:
            df_long["continuous_gt_A2D"] = float(gt_a2d)
        if gt_a3d is not None:
            gt_a3d_float = float(gt_a3d)
            df_long["continuous_gt_A3D"] = gt_a3d_float
            df_long["A3D_continuous_gt_diff"] = df_long["A3D"] - gt_a3d_float
            if gt_a3d_float != 0:
                df_long["A3D_continuous_gt_rel_err"] = df_long["A3D_continuous_gt_diff"] / gt_a3d_float
        if gt_ratio is not None:
            df_long["continuous_gt_ratio"] = float(gt_ratio)

    native_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    multi_resolution = payload.get("multi_resolution")
    if isinstance(multi_resolution, list):
        for entry in multi_resolution:
            if not isinstance(entry, dict):
                continue
            grid_info = entry.get("grid_info")
            native_ref = entry.get("native_grid_reference")
            if not isinstance(grid_info, dict) or not isinstance(native_ref, dict):
                continue
            dx = grid_info.get("dx")
            dy = grid_info.get("dy")
            if dx is None or dy is None:
                continue
            native_lookup[_grid_resolution_key(float(dx), float(dy))] = native_ref

    if not native_lookup:
        grid_info = payload.get("grid_info")
        native_ref = payload.get("native_grid_reference")
        if isinstance(grid_info, dict) and isinstance(native_ref, dict):
            dx = grid_info.get("dx")
            dy = grid_info.get("dy")
            if dx is not None and dy is not None:
                native_lookup[_grid_resolution_key(float(dx), float(dy))] = native_ref

    if native_lookup:
        synthetic_native_a2d: list[float] = []
        synthetic_native_a3d: list[float] = []
        synthetic_native_ratio: list[float] = []
        for row in df_long.itertuples(index=False):
            native_ref = native_lookup.get(_grid_resolution_key(float(row.dx), float(row.dy)))
            if native_ref is None:
                synthetic_native_a2d.append(float("nan"))
                synthetic_native_a3d.append(float("nan"))
                synthetic_native_ratio.append(float("nan"))
                continue
            synthetic_native_a2d.append(float(native_ref.get("planar_area_m2", float("nan"))))
            synthetic_native_a3d.append(float(native_ref.get("surface_area_m2", float("nan"))))
            synthetic_native_ratio.append(float(native_ref.get("surface_ratio", float("nan"))))

        df_long["synthetic_native_ref_A2D"] = synthetic_native_a2d
        df_long["synthetic_native_ref_A3D"] = synthetic_native_a3d
        df_long["synthetic_native_ref_ratio"] = synthetic_native_ratio
        df_long["A3D_synthetic_native_ref_diff"] = df_long["A3D"] - df_long["synthetic_native_ref_A3D"]
        valid_den = df_long["synthetic_native_ref_A3D"].replace({0.0: float("nan")})
        df_long["A3D_synthetic_native_ref_rel_err"] = df_long["A3D_synthetic_native_ref_diff"] / valid_den

    return df_long


def _native_gsd_scalar(info: RasterInfo) -> float:
    dx = float(info.dx)
    dy = float(info.dy)
    if dx <= 0 or dy <= 0:
        raise ValueError(f"Invalid pixel sizes from raster: dx={dx}, dy={dy}")
    if math.isclose(dx, dy, rel_tol=1e-9, abs_tol=1e-12):
        return dx
    return 0.5 * (dx + dy)


def _resolve_gsd_targets(
    values: list[str] | list[str | float] | None,
    *,
    raster_info: RasterInfo,
) -> list[_ResolvedGsdTarget]:
    raw_values = list(values) if values is not None else list(DEFAULT_GSD_LIST)
    if not raw_values:
        raise ValueError("--gsd must contain at least one value")

    resolved: list[_ResolvedGsdTarget] = []
    seen: set[tuple[str, str]] = set()

    for value in raw_values:
        if isinstance(value, str):
            token = value.strip().lower()
            if not token:
                raise ValueError("--gsd must not contain empty strings")
            if token == GSD_NATIVE_TOKEN:
                key = _grid_resolution_key(raster_info.dx, raster_info.dy)
                target = _ResolvedGsdTarget(
                    gsd_m=_native_gsd_scalar(raster_info),
                    use_native_grid=True,
                    label=GSD_NATIVE_TOKEN,
                )
            else:
                try:
                    gsd_m = float(token)
                except ValueError as exc:
                    raise ValueError(
                        f"--gsd values must be positive numbers or '{GSD_NATIVE_TOKEN}', got: {value!r}"
                    ) from exc
                if gsd_m <= 0:
                    raise ValueError(f"--gsd must contain positive values, got: {raw_values}")
                key = _grid_resolution_key(gsd_m, gsd_m)
                target = _ResolvedGsdTarget(gsd_m=gsd_m, use_native_grid=False, label=f"{gsd_m:g}")
        else:
            gsd_m = float(value)
            if gsd_m <= 0:
                raise ValueError(f"--gsd must contain positive values, got: {raw_values}")
            key = _grid_resolution_key(gsd_m, gsd_m)
            target = _ResolvedGsdTarget(gsd_m=gsd_m, use_native_grid=False, label=f"{gsd_m:g}")

        if key in seen:
            continue
        seen.add(key)
        resolved.append(target)

    if not resolved:
        raise ValueError("--gsd must contain at least one unique target resolution")
    return resolved


@dataclass(frozen=True, slots=True)
class _RunJob:
    dem: str
    tmp_dir: str
    resampled_dir: str
    keep_resampled: bool
    gsd_m: float
    gsd_label: str
    use_native_grid: bool
    gsd_idx: int
    total_gsd: int
    resampling: str
    nodata: float | None
    base_methods: tuple[str, ...]
    compute_methods: tuple[str, ...]
    needs_multiscale: bool
    slope_method: SlopeMethod
    jenness_weight: float
    integral_N: int
    adaptive_rel_tol: float
    adaptive_abs_tol: float
    adaptive_max_level: int
    adaptive_min_N: int
    adaptive_roughness_fastpath: bool
    adaptive_roughness_threshold: float | None
    sector_jenness_rel_tol: float
    sector_jenness_abs_tol: float
    sector_jenness_max_level: int
    sector_jenness_min_samples: int
    sigma_mode: str
    sigma_m: tuple[float, ...]
    roi_path: str | None
    roi_id_field: str | None
    roi_mode: str
    roi_all_touched: bool
    roi_only: bool
    raster_crs: str | None
    workers: int


@dataclass(frozen=True, slots=True)
class _RunJobResult:
    gsd_idx: int
    total_gsd: int
    gsd_m: float
    rows: list[dict]
    roi_rows: list[dict]


def _run_single_gsd(job: _RunJob, *, show_progress: bool = False, loaded_rois: Any | None = None) -> _RunJobResult:
    progress = ProgressPrinter() if show_progress else None

    gsd_display = job.gsd_label
    if job.use_native_grid:
        dst_path = Path(job.dem)
        res_info = get_raster_info(job.dem)
        t_resample = 0.0
        resampling_note = "native"
        if progress is not None:
            progress.log(
                f"[{job.gsd_idx}/{job.total_gsd}] Using native DEM grid "
                f"(dx={res_info.dx:g}, dy={res_info.dy:g}) ..."
            )
    else:
        tag = safe_gsd_tag(job.gsd_m)
        dst_dir = Path(job.resampled_dir if job.keep_resampled else job.tmp_dir)
        dst_path = dst_dir / f"dem_gsd_{tag}m.tif"
        rs = parse_resampling(job.resampling)

        if progress is not None:
            progress.log(f"[{job.gsd_idx}/{job.total_gsd}] Resampling DEM at gsd={job.gsd_m:g} ...")
        t0 = perf_counter()
        res_info = resample_dem(
            src_path=job.dem,
            dst_path=dst_path,
            target_gsd_m=job.gsd_m,
            resampling=rs,
            nodata=job.nodata,
        )
        t_resample = perf_counter() - t0
        resampling_note = rs.name

    dx = res_info.dx
    dy = res_info.dy
    rows: list[dict] = []
    roi_rows: list[dict] = []
    results: dict[str, AreaResult] = {}

    if not job.roi_only:
        method_summary = ", ".join(job.compute_methods)
        if progress is not None:
            progress.log(f"[{job.gsd_idx}/{job.total_gsd}] Computing methods: {method_summary}")

        def _methods_progress(_: str, current: int, total: int) -> None:
            if progress is not None:
                progress.update(
                    label=f"[{job.gsd_idx}/{job.total_gsd}] compute (gsd={gsd_display})",
                    current=current,
                    total=total,
                )

        results, timings = compute_methods_on_raster_with_timings(
            str(dst_path),
            nodata=job.nodata,
            methods=list(job.compute_methods),
            jenness_weight=job.jenness_weight,
            slope_method=job.slope_method,
            integral_N=job.integral_N,
            adaptive_rel_tol=job.adaptive_rel_tol,
            adaptive_abs_tol=job.adaptive_abs_tol,
            adaptive_max_level=job.adaptive_max_level,
            adaptive_min_N=job.adaptive_min_N,
            adaptive_roughness_fastpath=job.adaptive_roughness_fastpath,
            adaptive_roughness_threshold=job.adaptive_roughness_threshold,
            sector_jenness_rel_tol=job.sector_jenness_rel_tol,
            sector_jenness_abs_tol=job.sector_jenness_abs_tol,
            sector_jenness_max_level=job.sector_jenness_max_level,
            sector_jenness_min_samples=job.sector_jenness_min_samples,
            progress=_methods_progress if progress is not None else None,
            workers=job.workers,
        )
        if progress is not None:
            progress.finish()

        for method in job.base_methods:
            r = results[method]
            a2d = float(r.valid_cells) * dx * dy
            a3d = float(r.a3d)
            ratio = float(a3d / a2d) if a2d > 0 else float("nan")
            note_parts = [f"resampling={resampling_note}", f"dx={dx:g}", f"dy={dy:g}"]
            if method == "jenness_window_8tri":
                note_parts.append(f"weight={job.jenness_weight:g}")
                note_parts.append("triangle=heron")
            elif method == "sector_adaptive_jenness_integral":
                note_parts.append("surface=quadratic_ls_3x3")
                note_parts.append("partition=8_sector_cell")
                note_parts.append(f"min_samples={job.sector_jenness_min_samples}")
                note_parts.append(f"rel_tol={job.sector_jenness_rel_tol:g}")
                note_parts.append(f"abs_tol={job.sector_jenness_abs_tol:g}")
                note_parts.append(f"max_level={job.sector_jenness_max_level}")
            elif method == "gradient_multiplier":
                note_parts.append(f"slope_method={job.slope_method}")
            elif method == "bilinear_patch_integral":
                note_parts.append(f"N={job.integral_N}")
            elif method == "adaptive_bilinear_patch_integral":
                note_parts.append(f"min_N={job.adaptive_min_N}")
                note_parts.append(f"rel_tol={job.adaptive_rel_tol:g}")
                note_parts.append(f"abs_tol={job.adaptive_abs_tol:g}")
                note_parts.append(f"max_level={job.adaptive_max_level}")
            elif method == "tin_2tri_cell":
                note_parts.append("triangles=2")
            note_parts.append("runtime=compute_only")

            row = {
                "gsd_m": job.gsd_m,
                "dx": dx,
                "dy": dy,
                "method": method,
                "A2D": a2d,
                "A3D": a3d,
                "ratio": ratio,
                "valid_cells": int(r.valid_cells),
                "runtime_sec": float(timings.get(method, float("nan"))),
                "resample_runtime_sec": float(t_resample),
                "note": ";".join(note_parts),
            }
            if method == "adaptive_bilinear_patch_integral":
                row.update(
                    {
                        "adaptive_avg_level": r.adaptive_avg_level,
                        "adaptive_max_level_used": r.adaptive_max_level_used,
                        "adaptive_refined_cell_fraction": r.adaptive_refined_cell_fraction,
                        "adaptive_total_subcells_evaluated": r.adaptive_total_subcells_evaluated,
                    }
                )
            if method == "sector_adaptive_jenness_integral":
                row.update(
                    {
                        "sector_jenness_avg_level": r.sector_jenness_avg_level,
                        "sector_jenness_max_level_used": r.sector_jenness_max_level_used,
                        "sector_jenness_refined_fraction": r.sector_jenness_refined_fraction,
                    }
                )
            rows.append(row)

    if job.roi_path is not None and job.base_methods:
        if progress is not None:
            progress.log(f"[{job.gsd_idx}/{job.total_gsd}] ROI aggregation (mode={job.roi_mode})")

        from surface_area.roi import compute_roi_areas_on_raster, load_rois

        rois = loaded_rois
        if rois is None:
            from rasterio.crs import CRS

            rois = load_rois(
                job.roi_path,
                raster_crs=None if job.raster_crs is None else CRS.from_string(job.raster_crs),
                roi_id_field=job.roi_id_field,
            )

        def _roi_progress(_: str, current: int, total: int) -> None:
            if progress is not None:
                progress.update(
                    label=f"[{job.gsd_idx}/{job.total_gsd}] roi (gsd={gsd_display})",
                    current=current,
                    total=total,
                )

        t0 = perf_counter()
        r_rows, _roi_timings = compute_roi_areas_on_raster(
            str(dst_path),
            nodata=job.nodata,
            rois=rois,
            roi_mode=job.roi_mode,
            roi_all_touched=job.roi_all_touched,
            methods=list(job.base_methods),
            jenness_weight=job.jenness_weight,
            slope_method=job.slope_method,
            integral_N=job.integral_N,
            adaptive_rel_tol=job.adaptive_rel_tol,
            adaptive_abs_tol=job.adaptive_abs_tol,
            adaptive_max_level=job.adaptive_max_level,
            adaptive_min_N=job.adaptive_min_N,
            adaptive_roughness_fastpath=job.adaptive_roughness_fastpath,
            adaptive_roughness_threshold=job.adaptive_roughness_threshold,
            sector_jenness_rel_tol=job.sector_jenness_rel_tol,
            sector_jenness_abs_tol=job.sector_jenness_abs_tol,
            sector_jenness_max_level=job.sector_jenness_max_level,
            sector_jenness_min_samples=job.sector_jenness_min_samples,
            progress=_roi_progress if progress is not None else None,
        )
        if progress is not None:
            progress.finish()
        t_roi = perf_counter() - t0
        for rr in r_rows:
            rr.update({"gsd_m": job.gsd_m, "dx": dx, "dy": dy, "resample_runtime_sec": float(t_resample)})
            rr["note"] = f"{rr.get('note', '')};roi_wall_sec={t_roi:g}".lstrip(";")
            roi_rows.append(rr)

    if job.needs_multiscale and not job.roi_only:
        sigma_list = _sigma_list_for_gsd(job.gsd_m, sigma_values=list(job.sigma_m), sigma_mode=job.sigma_mode)
        if progress is not None:
            progress.log(f"[{job.gsd_idx}/{job.total_gsd}] Multiscale decomposition (sigma_m={sigma_list})")

        def _ms_progress(stage: str, current: int, total: int) -> None:
            if progress is not None:
                progress.update(
                    label=f"[{job.gsd_idx}/{job.total_gsd}] {stage} (gsd={gsd_display})",
                    current=current,
                    total=total,
                )

        t0 = perf_counter()
        ms = compute_multiscale_on_raster(
            str(dst_path),
            nodata=job.nodata,
            base_method=job.slope_method,
            sigma_m_list=sigma_list,
            a_total=results.get("gradient_multiplier"),
            progress=_ms_progress if progress is not None else None,
        )
        if progress is not None:
            progress.finish()
        t_ms = perf_counter() - t0

        runtime_each = float(t_ms) / float(len(ms)) if ms else float("nan")
        for ms_res in ms:
            method_name = f"multiscale_decomposed_area_sigma{ms_res.sigma_m:g}m"
            a2d = float(ms_res.valid_cells) * dx * dy
            a_total = float(ms_res.a_total)
            ratio = float(a_total / a2d) if a2d > 0 else float("nan")
            rows.append(
                {
                    "gsd_m": job.gsd_m,
                    "dx": dx,
                    "dy": dy,
                    "method": method_name,
                    "A2D": a2d,
                    "A3D": a_total,
                    "ratio": ratio,
                    "valid_cells": int(ms_res.valid_cells),
                    "runtime_sec": runtime_each,
                    "resample_runtime_sec": float(t_resample),
                    "a_topo": float(ms_res.a_topo),
                    "a_micro": float(ms_res.a_micro),
                    "a_total": a_total,
                    "micro_ratio": float(ms_res.micro_ratio),
                    "sigma_m": float(ms_res.sigma_m),
                    "note": ";".join(
                        [
                            f"base_method={job.slope_method}",
                            f"sigma_m={ms_res.sigma_m:g}",
                            f"sigma_mode={job.sigma_mode}",
                            "lowpass=gaussian_normalized",
                        ]
                    ),
                }
            )

    if not job.keep_resampled and not job.use_native_grid:
        try:
            dst_path.unlink(missing_ok=True)
        except Exception:
            pass

    return _RunJobResult(
        gsd_idx=job.gsd_idx,
        total_gsd=job.total_gsd,
        gsd_m=job.gsd_m,
        rows=rows,
        roi_rows=roi_rows,
    )


def _env_versions() -> dict[str, str]:
    import matplotlib
    import numpy
    import pandas
    import rasterio
    import scipy

    return {
        "python": sys.version.replace("\n", " "),
        "surface_area": __version__,
        "numpy": numpy.__version__,
        "rasterio": rasterio.__version__,
        "scipy": scipy.__version__,
        "pandas": pandas.__version__,
        "matplotlib": matplotlib.__version__,
    }


def _raster_info_json(info: RasterInfo) -> dict:
    return {
        "path": str(info.path),
        "crs": info.crs.to_string() if info.crs is not None else None,
        "transform": list(info.transform) if info.transform is not None else None,
        "width": int(info.width),
        "height": int(info.height),
        "nodata": None if info.nodata is None else float(info.nodata),
        "dtype": str(info.dtype),
        "dx": float(info.dx),
        "dy": float(info.dy),
    }


def _write_run_info(outdir: Path, payload: dict) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "run_info.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _run_info_sheet(payload: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def _visit(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                child_prefix = f"{prefix}.{key}" if prefix else str(key)
                _visit(child_prefix, child)
            return
        if isinstance(value, list):
            if any(isinstance(item, (dict, list)) for item in value):
                for index, child in enumerate(value):
                    child_prefix = f"{prefix}[{index}]"
                    _visit(child_prefix, child)
            else:
                rows.append({"key": prefix, "value": json.dumps(value, ensure_ascii=False)})
            return
        rows.append({"key": prefix, "value": value})

    _visit("", payload)
    return pd.DataFrame.from_records(rows, columns=["key", "value"])


def _write_results_workbook(
    outdir: Path,
    *,
    run_info: dict[str, Any],
    df_long: pd.DataFrame | None = None,
    df_roi: pd.DataFrame | None = None,
) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    workbook_path = outdir / "results.xlsx"

    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        if df_long is not None:
            df_long.to_excel(writer, sheet_name="results_long", index=False)
            _results_wide(df_long).to_excel(writer, sheet_name="results_wide", index=False)
        if df_roi is not None and not df_roi.empty:
            df_roi.to_excel(writer, sheet_name="results_roi_long", index=False)
        _run_info_sheet(run_info).to_excel(writer, sheet_name="run_info", index=False)

    return workbook_path


def _order_result_columns(df: pd.DataFrame, *, base_columns: list[str]) -> pd.DataFrame:
    ordered: list[str] = []
    for column in (
        base_columns
        + _RESULTS_REFERENCE_COLUMNS
        + _RESULTS_SYNTHETIC_REFERENCE_COLUMNS
        + _RESULTS_MULTISCALE_COLUMNS
        + _RESULTS_ADAPTIVE_COLUMNS
        + _RESULTS_SECTOR_COLUMNS
    ):
        if column in df.columns and column not in ordered:
            ordered.append(column)
    remaining = [column for column in df.columns if column not in ordered]
    return df[ordered + remaining]


def _sigma_list_for_gsd(gsd_m: float, *, sigma_values: list[float], sigma_mode: str) -> list[float]:
    mode = sigma_mode.strip().lower()
    if mode not in {"mult", "m"}:
        raise ValueError(f"sigma_mode must be 'mult' or 'm', got {sigma_mode!r}")
    if mode == "mult":
        return [float(v) * float(gsd_m) for v in sigma_values]
    return [float(v) for v in sigma_values]


def _results_wide(df_long: pd.DataFrame) -> pd.DataFrame:
    # Excel-friendly wide output: one column per computed method, only the final area value.
    wide = df_long.set_index(["gsd_m", "method"])["A3D"].unstack("method")
    wide.columns = [f"{method}_A3D" for method in wide.columns.to_list()]
    return wide.reset_index().sort_values("gsd_m")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m surface_area", description="DEM surface area estimator")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run resampling + surface area computations")
    run.add_argument("--dem", required=True, type=Path, help="Input DEM GeoTIFF path")
    run.add_argument("--outdir", required=True, type=Path, help="Output directory")
    run.add_argument(
        "--gsd",
        type=str,
        nargs="+",
        default=None,
        help=(
            f"Target GSD list in meters or '{GSD_NATIVE_TOKEN}' for the source grid "
            f"(default: {' '.join(_format_gsd_value(v) for v in DEFAULT_GSD_LIST)})"
        ),
    )
    run.add_argument(
        "--methods",
        type=str,
        nargs="+",
        choices=METHOD_CHOICES,
        default=None,
        help=f"Methods to run (default: {DEFAULT_METHODS})",
    )
    run.add_argument("--resampling", choices=["bilinear", "nearest", "cubic"], default="bilinear")
    run.add_argument("--nodata", type=float, default=None, help="Override DEM nodata value")
    run.add_argument("--slope_method", choices=["horn", "zt"], default="horn")
    run.add_argument("--jenness_weight", type=float, default=0.25)
    run.add_argument("--integral_N", type=int, default=5)
    run.add_argument("--adaptive_rel_tol", type=float, default=1e-4)
    run.add_argument("--adaptive_abs_tol", type=float, default=0.0)
    run.add_argument("--adaptive_max_level", type=int, default=5)
    run.add_argument("--adaptive_min_N", type=int, default=2)
    run.add_argument("--adaptive_roughness_fastpath", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument("--adaptive_roughness_threshold", type=float, default=None)
    run.add_argument("--sector_jenness_rel_tol", type=float, default=1e-4)
    run.add_argument("--sector_jenness_abs_tol", type=float, default=0.0)
    run.add_argument("--sector_jenness_max_level", type=int, default=5)
    run.add_argument("--sector_jenness_min_samples", type=int, default=3)
    run.add_argument(
        "--sigma_mode",
        choices=["mult", "m"],
        default="mult",
        help="Interpret --sigma_m values as multiples of GSD (mult) or absolute meters (m).",
    )
    run.add_argument("--sigma_m", type=float, nargs="+", default=[2.0, 5.0], help="Sigma list for multiscale")
    run.add_argument("--roi", type=Path, default=None, help="Optional ROI polygons (GeoJSON or Shapefile)")
    run.add_argument("--roi_id_field", type=str, default=None, help="ROI id field (default: id if present)")
    run.add_argument("--roi_mode", choices=["mask", "fraction"], default="mask")
    run.add_argument("--roi_all_touched", action=argparse.BooleanOptionalAction, default=False)
    run.add_argument("--roi_only", action=argparse.BooleanOptionalAction, default=False)
    run.add_argument("--reference_csv", type=Path, default=None, help="Optional reference CSV to compare")
    run.add_argument("--plots", action="store_true", help="Generate PNG plots")
    run.add_argument("--keep_resampled", action="store_true", help="Keep resampled GeoTIFFs on disk")
    run.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes for blockwise raster compute (default: 1).",
    )

    synth = sub.add_parser("synth", help="Generate a synthetic DSM/DEM GeoTIFF for method comparisons")
    synth.add_argument("--out", required=True, type=Path, help="Output GeoTIFF path")
    synth.add_argument("--preset", choices=SYNTHETIC_PRESETS, default="patchwork")
    synth.add_argument("--rows", type=int, default=512)
    synth.add_argument("--cols", type=int, default=512)
    synth.add_argument("--dx", type=float, default=1.0, help="Pixel size in meters")
    synth.add_argument("--dy", type=float, default=None, help="Pixel size in meters (defaults to --dx)")
    synth.add_argument("--seed", type=int, default=0)
    synth.add_argument("--relief", type=float, default=1.0, help="Macro relief multiplier")
    synth.add_argument("--roughness_m", type=float, default=0.75, help="Micro roughness amplitude (meters)")
    synth.add_argument("--crs", type=str, default="EPSG:32636")
    synth.add_argument("--origin_x", type=float, default=500_000.0)
    synth.add_argument("--origin_y", type=float, default=4_500_000.0)
    synth.add_argument("--nodata", type=float, default=-9999.0)
    synth.add_argument("--nodata_holes", type=int, default=0, help="Number of circular nodata holes to punch in")
    synth.add_argument("--nodata_radius_m", type=float, default=12.0, help="Base radius for nodata holes (meters)")

    return p


def cmd_run(args: argparse.Namespace) -> int:
    dem: Path = args.dem
    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    progress = ProgressPrinter()

    if not dem.exists():
        print(f"ERROR: DEM not found: {dem}", file=sys.stderr)
        return 2

    info = get_raster_info(dem)
    unit_name = crs_linear_unit_name(info.crs)
    unit_is_meter = crs_is_meter(info.crs)
    synthetic_reference_payload = _load_synthetic_reference_payload(dem)
    if unit_is_meter is False:
        print(
            f"WARNING: DEM CRS unit does not look like meters (unit={unit_name!r}). "
            "GSD values and areas will be in CRS linear units.",
            file=sys.stderr,
        )
    elif unit_is_meter is None:
        print(
            f"WARNING: Could not confirm DEM CRS linear unit (unit={unit_name!r}). "
            "Ensure GSD values are in the DEM CRS linear unit.",
            file=sys.stderr,
        )

    try:
        gsd_targets = _resolve_gsd_targets(args.gsd, raster_info=info)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    gsd_list = [float(target.gsd_m) for target in gsd_targets]
    worker_count = int(args.workers)
    if worker_count <= 0:
        print(f"ERROR: --workers must be >= 1, got: {worker_count}", file=sys.stderr)
        return 2

    method_list = [m.strip().lower() for m in (args.methods if args.methods is not None else DEFAULT_METHODS)]

    rs = parse_resampling(args.resampling)
    slope_method_n: SlopeMethod = "horn" if args.slope_method.strip().lower() == "horn" else "zt"

    versions = _env_versions()
    run_dt = datetime.now(timezone.utc)
    run_ts = run_dt.isoformat()
    run_tag = run_dt.strftime("%Y%m%dT%H%M%SZ")

    run_info = {
        "timestamp_utc": run_ts,
        "dem": str(dem),
        "dem_info": _raster_info_json(info),
        "synthetic_reference": None
        if synthetic_reference_payload is None
        else _synthetic_reference_summary(synthetic_reference_payload),
        "versions": versions,
        "params": {
            "gsd_specs": [target.label for target in gsd_targets],
            "gsd_list": gsd_list,
            "methods": method_list,
            "resampling": rs.name,
            "nodata_override": args.nodata,
            "slope_method": slope_method_n,
            "jenness_weight": float(args.jenness_weight),
            "integral_N": int(args.integral_N),
            "adaptive_rel_tol": float(args.adaptive_rel_tol),
            "adaptive_abs_tol": float(args.adaptive_abs_tol),
            "adaptive_max_level": int(args.adaptive_max_level),
            "adaptive_min_N": int(args.adaptive_min_N),
            "adaptive_roughness_fastpath": bool(args.adaptive_roughness_fastpath),
            "adaptive_roughness_threshold": None
            if args.adaptive_roughness_threshold is None
            else float(args.adaptive_roughness_threshold),
            "sector_jenness_rel_tol": float(args.sector_jenness_rel_tol),
            "sector_jenness_abs_tol": float(args.sector_jenness_abs_tol),
            "sector_jenness_max_level": int(args.sector_jenness_max_level),
            "sector_jenness_min_samples": int(args.sector_jenness_min_samples),
            "sigma_mode": args.sigma_mode,
            "sigma_m_values": list(args.sigma_m),
            "roi": None if args.roi is None else str(args.roi),
            "roi_id_field": args.roi_id_field,
            "roi_mode": args.roi_mode,
            "roi_all_touched": bool(args.roi_all_touched),
            "roi_only": bool(args.roi_only),
            "workers": worker_count,
        },
    }
    _write_run_info(outdir, run_info)

    rows: list[dict] = []
    roi_rows: list[dict] = []
    df_long: pd.DataFrame | None = None
    df_roi: pd.DataFrame | None = None
    resampled_dir = outdir / "resampled"
    tmp_dir = outdir / "_tmp" / f"run_{run_tag}"
    if args.keep_resampled:
        resampled_dir.mkdir(parents=True, exist_ok=True)
    else:
        tmp_dir.mkdir(parents=True, exist_ok=True)

    base_methods = [m for m in method_list if m != "multiscale_decomposed_area"]
    needs_multiscale = "multiscale_decomposed_area" in method_list
    compute_set = set(base_methods)
    if needs_multiscale:
        compute_set.add("gradient_multiplier")  # base method for multiscale

    loaded_rois = None
    if args.roi is not None:
        try:
            from surface_area.roi import load_rois

            loaded_rois = load_rois(args.roi, raster_crs=info.crs, roi_id_field=args.roi_id_field)
            progress.log(f"Loaded {len(loaded_rois)} ROI polygon(s) from: {args.roi}")
        except Exception as e:
            progress.log(f"ERROR: failed to load ROI: {e}")
            return 2

    total_gsd = len(gsd_list)
    jobs = [
        _RunJob(
            dem=str(dem),
            tmp_dir=str(tmp_dir),
            resampled_dir=str(resampled_dir),
            keep_resampled=bool(args.keep_resampled),
            gsd_m=target.gsd_m,
            gsd_label=target.label,
            use_native_grid=target.use_native_grid,
            gsd_idx=gsd_idx,
            total_gsd=total_gsd,
            resampling=rs.name,
            nodata=args.nodata,
            base_methods=tuple(base_methods),
            compute_methods=tuple(sorted(compute_set)),
            needs_multiscale=bool(needs_multiscale),
            slope_method=slope_method_n,
            jenness_weight=float(args.jenness_weight),
            integral_N=int(args.integral_N),
            adaptive_rel_tol=float(args.adaptive_rel_tol),
            adaptive_abs_tol=float(args.adaptive_abs_tol),
            adaptive_max_level=int(args.adaptive_max_level),
            adaptive_min_N=int(args.adaptive_min_N),
            adaptive_roughness_fastpath=bool(args.adaptive_roughness_fastpath),
            adaptive_roughness_threshold=args.adaptive_roughness_threshold,
            sector_jenness_rel_tol=float(args.sector_jenness_rel_tol),
            sector_jenness_abs_tol=float(args.sector_jenness_abs_tol),
            sector_jenness_max_level=int(args.sector_jenness_max_level),
            sector_jenness_min_samples=int(args.sector_jenness_min_samples),
            sigma_mode=args.sigma_mode,
            sigma_m=tuple(float(x) for x in args.sigma_m),
            roi_path=None if args.roi is None else str(args.roi),
            roi_id_field=args.roi_id_field,
            roi_mode=args.roi_mode,
            roi_all_touched=bool(args.roi_all_touched),
            roi_only=bool(args.roi_only),
            raster_crs=None if info.crs is None else info.crs.to_string(),
            workers=worker_count,
        )
        for gsd_idx, target in enumerate(gsd_targets, start=1)
    ]

    if worker_count > 1:
        progress.log(f"Using {worker_count} worker process(es) for blockwise raster compute.")
    for job in jobs:
        result = _run_single_gsd(job, show_progress=True, loaded_rois=loaded_rois)
        rows.extend(result.rows)
        roi_rows.extend(result.roi_rows)

    if not args.roi_only:
        df_long = pd.DataFrame.from_records(rows).sort_values(["gsd_m", "method"]).reset_index(drop=True)

        if args.reference_csv is not None:
            try:
                ref = pd.read_csv(args.reference_csv)
                ref_cols = {c.lower(): c for c in ref.columns}
                gsd_col = ref_cols.get("gsd_m") or ref_cols.get("gsd") or ref_cols.get("resolution")
                method_col = ref_cols.get("method") or ref_cols.get("tool") or ref_cols.get("name")
                a3d_col = ref_cols.get("a3d") or ref_cols.get("area_3d") or ref_cols.get("surface_area")
                if gsd_col and method_col and a3d_col:
                    ref2 = ref[[gsd_col, method_col, a3d_col]].copy()
                    ref2.columns = ["gsd_m", "method", "A3D_ref"]
                    df_long = df_long.merge(ref2, on=["gsd_m", "method"], how="left")
                    df_long["A3D_diff"] = df_long["A3D"] - df_long["A3D_ref"]
                    df_long["A3D_rel_err"] = df_long["A3D_diff"] / df_long["A3D_ref"]
                else:
                    print("WARNING: reference CSV did not match expected columns; skipping merge.", file=sys.stderr)
            except Exception as e:
                print(f"WARNING: failed to read/merge reference CSV: {e}", file=sys.stderr)

        df_long = _append_synthetic_reference_columns(df_long, synthetic_reference_payload)

        df_long = _order_result_columns(df_long, base_columns=_RESULTS_LONG_BASE_COLUMNS)

        if args.plots:
            progress.log("Plotting...")
            plot_a3d_vs_gsd(df_long, outdir)
            plot_ratio_vs_gsd(df_long, outdir)
            plot_micro_ratio_vs_gsd(df_long, outdir)

        if args.plots:
            print(f"Wrote plots to: {outdir}")

    if roi_rows:
        df_roi = pd.DataFrame.from_records(roi_rows).sort_values(["gsd_m", "roi_id", "method"]).reset_index(
            drop=True
        )
        df_roi = _order_result_columns(df_roi, base_columns=_RESULTS_ROI_BASE_COLUMNS)

    workbook_path = _write_results_workbook(outdir, run_info=run_info, df_long=df_long, df_roi=df_roi)
    print(f"Wrote: {workbook_path}")

    return 0


def cmd_synth(args: argparse.Namespace) -> int:
    out: Path = args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    progress = ProgressPrinter()
    z = generate_synthetic_dsm(
        rows=int(args.rows),
        cols=int(args.cols),
        dx=float(args.dx),
        dy=None if args.dy is None else float(args.dy),
        preset=str(args.preset),
        seed=int(args.seed),
        relief=float(args.relief),
        roughness_m=float(args.roughness_m),
        nodata_value=float(args.nodata) if args.nodata is not None else None,
        nodata_holes=int(args.nodata_holes),
        nodata_radius_m=float(args.nodata_radius_m),
        progress=progress,
    )
    progress.finish()

    info = write_dem_float32_geotiff(
        path=out,
        z=z,
        dx=float(args.dx),
        dy=float(args.dx if args.dy is None else args.dy),
        crs=str(args.crs),
        nodata=float(args.nodata) if args.nodata is not None else None,
        origin_x=float(args.origin_x),
        origin_y=float(args.origin_y),
    )

    print(f"Wrote: {out}")
    print(f"  size: {info.width}x{info.height}  dx={info.dx:g}  dy={info.dy:g}  nodata={info.nodata!r}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        return cmd_run(args)
    if args.command == "synth":
        return cmd_synth(args)

    parser.print_help()
    return 2
