"""Plotting utilities for results."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import pandas as pd


def _metric_frame(df_long: pd.DataFrame, value_column: str) -> pd.DataFrame:
    if value_column not in df_long.columns:
        return pd.DataFrame()
    return df_long.dropna(subset=["gsd_m", value_column]).copy()


def _reference_curve(df_long: pd.DataFrame, value_column: str) -> pd.DataFrame:
    if value_column not in df_long.columns:
        return pd.DataFrame(columns=["gsd_m", value_column])

    reference = (
        df_long.dropna(subset=["gsd_m", value_column])[["gsd_m", value_column]]
        .drop_duplicates(subset=["gsd_m", value_column])
        .sort_values("gsd_m")
        .reset_index(drop=True)
    )
    if reference.empty:
        return reference

    # Synthetic references repeat once per method, so we keep one value per GSD.
    return reference.groupby("gsd_m", as_index=False, sort=True).first()


def _ratio_to_excess_percent(values: pd.Series) -> pd.Series:
    return (pd.to_numeric(values, errors="coerce") - 1.0) * 100.0


def _transformed_column_frame(
    df_long: pd.DataFrame,
    *,
    value_column: str,
    transform: Callable[[pd.Series], pd.Series] | None = None,
) -> pd.DataFrame:
    if value_column not in df_long.columns:
        return pd.DataFrame(columns=df_long.columns)

    frame = df_long.copy()
    if transform is not None:
        frame[value_column] = transform(frame[value_column])
    return frame


def _plot_reference_series(
    ax: plt.Axes,
    df_long: pd.DataFrame,
    *,
    value_column: str,
    label: str,
    color: str,
    linestyle: str,
) -> None:
    reference = _reference_curve(df_long, value_column)
    if reference.empty:
        return

    if reference[value_column].nunique(dropna=True) == 1:
        ax.axhline(
            float(reference[value_column].iloc[0]),
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
            label=label,
            zorder=3,
        )
        return

    ax.plot(
        reference["gsd_m"],
        reference[value_column],
        color=color,
        linestyle=linestyle,
        linewidth=2.0,
        marker="s",
        markersize=4.5,
        label=label,
        zorder=3,
    )


def _prep_axes(title: str, xlabel: str, ylabel: str) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    return fig, ax


def _plot_metric_vs_gsd(
    df_long: pd.DataFrame,
    outdir: str | Path,
    *,
    value_column: str,
    filename: str,
    title: str,
    ylabel: str,
    zero_baseline: bool = False,
) -> Path | None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    metric = _metric_frame(df_long, value_column)
    if metric.empty:
        return None

    fig, ax = _prep_axes(title, "GSD (m)", ylabel)
    ax.set_xscale("log")
    if zero_baseline:
        ax.axhline(0.0, color="0.4", linestyle=":", linewidth=1.0, zorder=1)

    for method, g in metric.groupby("method", sort=True):
        g = g.sort_values("gsd_m")
        ax.plot(g["gsd_m"], g[value_column], marker="o", linewidth=1.5, label=str(method))

    ax.legend(loc="best", fontsize=9)
    path = outdir / filename
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_a3d_vs_gsd(df_long: pd.DataFrame, outdir: str | Path) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = _prep_axes("A3D vs GSD", "GSD (m)", "A3D (m^2)")
    ax.set_xscale("log")

    for method, g in df_long.dropna(subset=["A3D"]).groupby("method", sort=True):
        g = g.sort_values("gsd_m")
        ax.plot(g["gsd_m"], g["A3D"], marker="o", linewidth=1.5, label=str(method))

    _plot_reference_series(
        ax,
        df_long,
        value_column="synthetic_native_ref_A3D",
        label="Native-grid reference (per GSD)",
        color="black",
        linestyle="--",
    )
    _plot_reference_series(
        ax,
        df_long,
        value_column="continuous_gt_A3D",
        label="Continuous ground truth (GSD-independent)",
        color="tab:red",
        linestyle=":",
    )

    ax.legend(loc="best", fontsize=9)
    path = outdir / "A3D_vs_GSD.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_ratio_family_vs_gsd(
    df_long: pd.DataFrame,
    outdir: str | Path,
    *,
    title: str,
    ylabel: str,
    filename: str,
    transform: Callable[[pd.Series], pd.Series] | None = None,
    zero_baseline: bool = False,
) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = _prep_axes(title, "GSD (m)", ylabel)
    ax.set_xscale("log")
    if zero_baseline:
        ax.axhline(0.0, color="0.4", linestyle=":", linewidth=1.0, zorder=1)

    ratio = _transformed_column_frame(df_long, value_column="ratio", transform=transform)
    for method, g in ratio.dropna(subset=["ratio"]).groupby("method", sort=True):
        g = g.sort_values("gsd_m")
        ax.plot(g["gsd_m"], g["ratio"], marker="o", linewidth=1.5, label=str(method))

    _plot_reference_series(
        ax,
        _transformed_column_frame(df_long, value_column="synthetic_native_ref_ratio", transform=transform),
        value_column="synthetic_native_ref_ratio",
        label="Native-grid reference (per GSD)",
        color="black",
        linestyle="--",
    )
    _plot_reference_series(
        ax,
        _transformed_column_frame(df_long, value_column="continuous_gt_ratio", transform=transform),
        value_column="continuous_gt_ratio",
        label="Continuous ground truth (GSD-independent)",
        color="tab:red",
        linestyle=":",
    )

    ax.legend(loc="best", fontsize=9)
    path = outdir / filename
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_ratio_vs_gsd(df_long: pd.DataFrame, outdir: str | Path) -> Path:
    return _plot_ratio_family_vs_gsd(
        df_long,
        outdir,
        title="A3D/A2D Ratio vs GSD",
        ylabel="A3D / A2D (-)",
        filename="ratio_vs_GSD.png",
    )


def plot_surface_excess_vs_gsd(df_long: pd.DataFrame, outdir: str | Path) -> Path:
    return _plot_ratio_family_vs_gsd(
        df_long,
        outdir,
        title="Surface Excess vs GSD",
        ylabel="(A3D / A2D - 1) (%)",
        filename="surface_excess_vs_GSD.png",
        transform=_ratio_to_excess_percent,
        zero_baseline=True,
    )


def plot_continuous_gt_rel_err_vs_gsd(df_long: pd.DataFrame, outdir: str | Path) -> Path | None:
    return _plot_metric_vs_gsd(
        df_long,
        outdir,
        value_column="A3D_continuous_gt_rel_err",
        filename="continuous_gt_rel_err_vs_GSD.png",
        title="Relative Error vs GSD (Continuous Ground Truth)",
        ylabel="Relative error (-)",
        zero_baseline=True,
    )


def plot_native_grid_ref_rel_err_vs_gsd(df_long: pd.DataFrame, outdir: str | Path) -> Path | None:
    return _plot_metric_vs_gsd(
        df_long,
        outdir,
        value_column="A3D_synthetic_native_ref_rel_err",
        filename="native_grid_ref_rel_err_vs_GSD.png",
        title="Relative Error vs GSD (Native-grid Reference)",
        ylabel="Relative error (-)",
        zero_baseline=True,
    )


def plot_runtime_vs_gsd(df_long: pd.DataFrame, outdir: str | Path) -> Path | None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    runtime = _metric_frame(df_long, "runtime_sec")
    if runtime.empty:
        return None

    runtime["runtime_total_sec"] = (
        runtime["runtime_sec"].astype(float) + runtime["resample_runtime_sec"].fillna(0.0).astype(float)
    )

    fig, ax = _prep_axes("Runtime vs GSD", "GSD (m)", "Runtime (s)")
    ax.set_xscale("log")

    for method, g in runtime.groupby("method", sort=True):
        g = g.sort_values("gsd_m")
        ax.plot(g["gsd_m"], g["runtime_total_sec"], marker="o", linewidth=1.5, label=str(method))

    ax.legend(loc="best", fontsize=9)
    path = outdir / "runtime_vs_GSD.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_error_vs_runtime(df_long: pd.DataFrame, outdir: str | Path) -> Path | None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    error_column = ""
    ylabel = ""
    if "A3D_continuous_gt_rel_err" in df_long.columns and df_long["A3D_continuous_gt_rel_err"].notna().any():
        error_column = "A3D_continuous_gt_rel_err"
        ylabel = "|Relative error vs continuous GT| (-)"
    elif "A3D_synthetic_native_ref_rel_err" in df_long.columns and df_long["A3D_synthetic_native_ref_rel_err"].notna().any():
        error_column = "A3D_synthetic_native_ref_rel_err"
        ylabel = "|Relative error vs native-grid ref| (-)"
    else:
        return None

    scatter = df_long.dropna(subset=["runtime_sec", error_column]).copy()
    if scatter.empty:
        return None

    scatter["runtime_total_sec"] = (
        scatter["runtime_sec"].astype(float) + scatter["resample_runtime_sec"].fillna(0.0).astype(float)
    )
    scatter["abs_rel_err"] = scatter[error_column].abs().astype(float)

    fig, ax = _prep_axes("Error vs Runtime", "Runtime (s)", ylabel)

    for method, g in scatter.groupby("method", sort=True):
        g = g.sort_values("runtime_total_sec")
        ax.plot(g["runtime_total_sec"], g["abs_rel_err"], linewidth=0.9, alpha=0.5)
        ax.scatter(g["runtime_total_sec"], g["abs_rel_err"], s=36, alpha=0.85, label=str(method))

    ax.legend(loc="best", fontsize=9)
    path = outdir / "error_vs_runtime.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_micro_ratio_vs_gsd(df_long: pd.DataFrame, outdir: str | Path) -> Path | None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if "micro_ratio" not in df_long.columns:
        return None

    ms = df_long.dropna(subset=["micro_ratio"])
    if ms.empty:
        return None

    fig, ax = _prep_axes("A_micro / A_total vs GSD (multiscale)", "GSD (m)", "A_micro / A_total (-)")
    ax.set_xscale("log")

    if "sigma_m" in ms.columns and ms["sigma_m"].notna().any():
        for sigma_m, g in ms.groupby("sigma_m", sort=True):
            g = g.sort_values("gsd_m")
            ax.plot(
                g["gsd_m"],
                g["micro_ratio"],
                marker="o",
                linewidth=1.5,
                label=f"sigma={float(sigma_m):g} m",
            )
    else:
        for method, g in ms.groupby("method", sort=True):
            g = g.sort_values("gsd_m")
            ax.plot(g["gsd_m"], g["micro_ratio"], marker="o", linewidth=1.5, label=str(method))

    ax.legend(loc="best", fontsize=9)
    path = outdir / "micro_ratio_vs_GSD.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
