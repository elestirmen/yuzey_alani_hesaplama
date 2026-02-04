from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

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
)
from surface_area.methods import AreaResult, SlopeMethod, compute_methods_on_raster_with_timings
from surface_area.multiscale import compute_multiscale_on_raster
from surface_area.plotting import plot_a3d_vs_gsd, plot_micro_ratio_vs_gsd, plot_ratio_vs_gsd
from surface_area.progress import ProgressPrinter


DEFAULT_GSD_LIST = [0.1, 0.5, 1, 2, 5, 10, 20, 50]

METHOD_CHOICES = [
    "jenness_window_8tri",
    "tin_2tri_cell",
    "gradient_multiplier",
    "bilinear_patch_integral",
    "adaptive_bilinear_patch_integral",
    "multiscale_decomposed_area",
]

DEFAULT_METHODS = [
    "jenness_window_8tri",
    "tin_2tri_cell",
    "gradient_multiplier",
    "bilinear_patch_integral",
    "multiscale_decomposed_area",
]


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


def _sigma_list_for_gsd(gsd_m: float, *, sigma_values: list[float], sigma_mode: str) -> list[float]:
    mode = sigma_mode.strip().lower()
    if mode not in {"mult", "m"}:
        raise ValueError(f"sigma_mode must be 'mult' or 'm', got {sigma_mode!r}")
    if mode == "mult":
        return [float(v) * float(gsd_m) for v in sigma_values]
    return [float(v) for v in sigma_values]


def _results_wide(df_long: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "A2D",
        "A3D",
        "ratio",
        "valid_cells",
        "runtime_sec",
        "a_topo",
        "a_micro",
        "a_total",
        "micro_ratio",
    ]
    keep_metrics = [m for m in metrics if m in df_long.columns]

    wide = df_long.set_index(["gsd_m", "method"])[keep_metrics].unstack("method")
    # wide columns are MultiIndex (metric, method)
    wide.columns = [f"{method}_{metric}" for metric, method in wide.columns.to_list()]
    return wide.reset_index().sort_values("gsd_m")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m surface_area", description="DEM surface area estimator")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run resampling + surface area computations")
    run.add_argument("--dem", required=True, type=Path, help="Input DEM GeoTIFF path")
    run.add_argument("--outdir", required=True, type=Path, help="Output directory")
    run.add_argument(
        "--gsd",
        type=float,
        nargs="+",
        default=None,
        help=f"Target GSD list in meters (default: {DEFAULT_GSD_LIST})",
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

    gsd_list = [float(x) for x in (args.gsd if args.gsd is not None else DEFAULT_GSD_LIST)]
    if not gsd_list or any(x <= 0 for x in gsd_list):
        print(f"ERROR: --gsd must contain positive values, got: {gsd_list}", file=sys.stderr)
        return 2
    gsd_list = sorted(set(gsd_list))

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
        "versions": versions,
        "params": {
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
            "sigma_mode": args.sigma_mode,
            "sigma_m_values": list(args.sigma_m),
            "roi": None if args.roi is None else str(args.roi),
            "roi_id_field": args.roi_id_field,
            "roi_mode": args.roi_mode,
            "roi_all_touched": bool(args.roi_all_touched),
            "roi_only": bool(args.roi_only),
        },
    }
    _write_run_info(outdir, run_info)

    rows: list[dict] = []
    roi_rows: list[dict] = []
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

    rois = None
    if args.roi is not None:
        try:
            from surface_area.roi import RoiError, compute_roi_areas_on_raster, load_rois

            rois = load_rois(args.roi, raster_crs=info.crs, roi_id_field=args.roi_id_field)
            progress.log(f"Loaded {len(rois)} ROI polygon(s) from: {args.roi}")
        except Exception as e:
            progress.log(f"ERROR: failed to load ROI: {e}")
            return 2

    total_gsd = len(gsd_list)
    for gsd_idx, gsd_m in enumerate(gsd_list, start=1):
        tag = safe_gsd_tag(gsd_m)
        dst_path = (resampled_dir if args.keep_resampled else tmp_dir) / f"dem_gsd_{tag}m.tif"

        progress.log(f"[{gsd_idx}/{total_gsd}] Resampling DEM at gsd={gsd_m:g} ...")
        t0 = perf_counter()
        res_info = resample_dem(
            src_path=dem,
            dst_path=dst_path,
            target_gsd_m=gsd_m,
            resampling=rs,
            nodata=args.nodata,
        )
        t_resample = perf_counter() - t0

        dx = res_info.dx
        dy = res_info.dy

        method_summary = ", ".join(sorted(compute_set))
        if not args.roi_only:
            progress.log(f"[{gsd_idx}/{total_gsd}] Computing methods: {method_summary}")

        def _methods_progress(_: str, current: int, total: int) -> None:
            progress.update(label=f"[{gsd_idx}/{total_gsd}] compute (gsd={gsd_m:g})", current=current, total=total)

        results = timings = None
        if not args.roi_only:
            results, timings = compute_methods_on_raster_with_timings(
                str(dst_path),
                nodata=args.nodata,
                methods=sorted(compute_set),
                jenness_weight=float(args.jenness_weight),
                slope_method=slope_method_n,
                integral_N=int(args.integral_N),
                adaptive_rel_tol=float(args.adaptive_rel_tol),
                adaptive_abs_tol=float(args.adaptive_abs_tol),
                adaptive_max_level=int(args.adaptive_max_level),
                adaptive_min_N=int(args.adaptive_min_N),
                adaptive_roughness_fastpath=bool(args.adaptive_roughness_fastpath),
                adaptive_roughness_threshold=args.adaptive_roughness_threshold,
                progress=_methods_progress,
            )
            progress.finish()

            for method in base_methods:
                r = results[method]
                A2D = float(r.valid_cells) * dx * dy
                A3D = float(r.a3d)
                ratio = float(A3D / A2D) if A2D > 0 else float("nan")
                note_parts = [f"resampling={rs.name}", f"dx={dx:g}", f"dy={dy:g}"]
                if method == "jenness_window_8tri":
                    note_parts.append(f"weight={float(args.jenness_weight):g}")
                    note_parts.append("triangle=heron")
                elif method == "gradient_multiplier":
                    note_parts.append(f"slope_method={slope_method_n}")
                elif method == "bilinear_patch_integral":
                    note_parts.append(f"N={int(args.integral_N)}")
                elif method == "adaptive_bilinear_patch_integral":
                    note_parts.append(f"min_N={int(args.adaptive_min_N)}")
                    note_parts.append(f"rel_tol={float(args.adaptive_rel_tol):g}")
                    note_parts.append(f"abs_tol={float(args.adaptive_abs_tol):g}")
                    note_parts.append(f"max_level={int(args.adaptive_max_level)}")
                elif method == "tin_2tri_cell":
                    note_parts.append("triangles=2")
                note_parts.append("runtime=compute_only")

                row = {
                    "gsd_m": gsd_m,
                    "dx": dx,
                    "dy": dy,
                    "method": method,
                    "A2D": A2D,
                    "A3D": A3D,
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
                rows.append(row)

        if rois is not None and base_methods:
            progress.log(f"[{gsd_idx}/{total_gsd}] ROI aggregation (mode={args.roi_mode})")

            def _roi_progress(_: str, current: int, total: int) -> None:
                progress.update(label=f"[{gsd_idx}/{total_gsd}] roi (gsd={gsd_m:g})", current=current, total=total)

            t0 = perf_counter()
            r_rows, _roi_timings = compute_roi_areas_on_raster(
                str(dst_path),
                nodata=args.nodata,
                rois=rois,
                roi_mode=args.roi_mode,
                roi_all_touched=bool(args.roi_all_touched),
                methods=base_methods,
                jenness_weight=float(args.jenness_weight),
                slope_method=slope_method_n,
                integral_N=int(args.integral_N),
                adaptive_rel_tol=float(args.adaptive_rel_tol),
                adaptive_abs_tol=float(args.adaptive_abs_tol),
                adaptive_max_level=int(args.adaptive_max_level),
                adaptive_min_N=int(args.adaptive_min_N),
                adaptive_roughness_fastpath=bool(args.adaptive_roughness_fastpath),
                adaptive_roughness_threshold=args.adaptive_roughness_threshold,
                progress=_roi_progress,
            )
            progress.finish()
            t_roi = perf_counter() - t0
            for rr in r_rows:
                rr.update({"gsd_m": gsd_m, "dx": dx, "dy": dy, "resample_runtime_sec": float(t_resample)})
                # Preserve compute_roi_areas_on_raster runtime; include a top-level wall clock in note for traceability.
                rr["note"] = f"{rr.get('note', '')};roi_wall_sec={t_roi:g}".lstrip(";")
                roi_rows.append(rr)

        if needs_multiscale and not args.roi_only:
            sigma_list = _sigma_list_for_gsd(
                gsd_m, sigma_values=[float(x) for x in args.sigma_m], sigma_mode=args.sigma_mode
            )
            progress.log(f"[{gsd_idx}/{total_gsd}] Multiscale decomposition (sigma_m={sigma_list})")

            def _ms_progress(stage: str, current: int, total: int) -> None:
                label = f"[{gsd_idx}/{total_gsd}] {stage} (gsd={gsd_m:g})"
                progress.update(label=label, current=current, total=total)

            t0 = perf_counter()
            ms = compute_multiscale_on_raster(
                str(dst_path),
                nodata=args.nodata,
                base_method=slope_method_n,
                sigma_m_list=sigma_list,
                a_total=results.get("gradient_multiplier"),
                progress=_ms_progress,
            )
            t_ms = perf_counter() - t0
            progress.finish()

            runtime_each = float(t_ms) / float(len(ms)) if ms else float("nan")
            for ms_res in ms:
                method_name = f"multiscale_decomposed_area_sigma{ms_res.sigma_m:g}m"
                A2D = float(ms_res.valid_cells) * dx * dy
                A_total = float(ms_res.a_total)
                ratio = float(A_total / A2D) if A2D > 0 else float("nan")
                rows.append(
                    {
                        "gsd_m": gsd_m,
                        "dx": dx,
                        "dy": dy,
                        "method": method_name,
                        "A2D": A2D,
                        "A3D": A_total,
                        "ratio": ratio,
                        "valid_cells": int(ms_res.valid_cells),
                        "runtime_sec": runtime_each,
                        "resample_runtime_sec": float(t_resample),
                        "a_topo": float(ms_res.a_topo),
                        "a_micro": float(ms_res.a_micro),
                        "a_total": A_total,
                        "micro_ratio": float(ms_res.micro_ratio),
                        "sigma_m": float(ms_res.sigma_m),
                        "note": ";".join(
                            [
                                f"base_method={slope_method_n}",
                                f"sigma_m={ms_res.sigma_m:g}",
                                f"sigma_mode={args.sigma_mode}",
                                "lowpass=gaussian_normalized",
                            ]
                        ),
                    }
                )

        if not args.keep_resampled:
            try:
                dst_path.unlink(missing_ok=True)
            except Exception:
                pass

    if not args.roi_only:
        df_long = pd.DataFrame.from_records(rows).sort_values(["gsd_m", "method"]).reset_index(drop=True)

        adaptive_cols = [
            "adaptive_avg_level",
            "adaptive_max_level_used",
            "adaptive_refined_cell_fraction",
            "adaptive_total_subcells_evaluated",
        ]

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

        # Ensure any newly-added adaptive diagnostics are appended at the end of the CSV.
        keep = [c for c in df_long.columns if c not in adaptive_cols]
        tail = [c for c in adaptive_cols if c in df_long.columns]
        df_long = df_long[keep + tail]

        long_path = outdir / "results_long.csv"
        df_long.to_csv(long_path, index=False)

        wide_path = outdir / "results_wide.csv"
        _results_wide(df_long).to_csv(wide_path, index=False)

        if args.plots:
            progress.log("Plotting...")
            plot_a3d_vs_gsd(df_long, outdir)
            plot_ratio_vs_gsd(df_long, outdir)
            plot_micro_ratio_vs_gsd(df_long, outdir)

        print(f"Wrote: {long_path}")
        print(f"Wrote: {wide_path}")
        if args.plots:
            print(f"Wrote plots to: {outdir}")

    if roi_rows:
        df_roi = pd.DataFrame.from_records(roi_rows).sort_values(["gsd_m", "roi_id", "method"]).reset_index(
            drop=True
        )
        adaptive_cols = [
            "adaptive_avg_level",
            "adaptive_max_level_used",
            "adaptive_refined_cell_fraction",
            "adaptive_total_subcells_evaluated",
        ]
        keep = [c for c in df_roi.columns if c not in adaptive_cols]
        tail = [c for c in adaptive_cols if c in df_roi.columns]
        df_roi = df_roi[keep + tail]
        roi_path = outdir / "results_roi_long.csv"
        df_roi.to_csv(roi_path, index=False)
        print(f"Wrote: {roi_path}")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        return cmd_run(args)

    parser.print_help()
    return 2
