"""Plotting utilities for results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _prep_axes(title: str, xlabel: str, ylabel: str) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    return fig, ax


def plot_a3d_vs_gsd(df_long: pd.DataFrame, outdir: str | Path) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = _prep_axes("A3D vs GSD", "GSD (m)", "A3D (m²)")
    ax.set_xscale("log")

    for method, g in df_long.dropna(subset=["A3D"]).groupby("method", sort=True):
        g = g.sort_values("gsd_m")
        ax.plot(g["gsd_m"], g["A3D"], marker="o", linewidth=1.5, label=str(method))

    ax.legend(loc="best", fontsize=9)
    path = outdir / "A3D_vs_GSD.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_ratio_vs_gsd(df_long: pd.DataFrame, outdir: str | Path) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = _prep_axes("A3D/A2D Ratio vs GSD", "GSD (m)", "A3D / A2D (-)")
    ax.set_xscale("log")

    for method, g in df_long.dropna(subset=["ratio"]).groupby("method", sort=True):
        g = g.sort_values("gsd_m")
        ax.plot(g["gsd_m"], g["ratio"], marker="o", linewidth=1.5, label=str(method))

    ax.legend(loc="best", fontsize=9)
    path = outdir / "ratio_vs_GSD.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_micro_ratio_vs_gsd(df_long: pd.DataFrame, outdir: str | Path) -> Path | None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ms = df_long.dropna(subset=["micro_ratio"])
    if ms.empty:
        return None

    fig, ax = _prep_axes("A_micro / A_total vs GSD (multiscale)", "GSD (m)", "A_micro / A_total (-)")
    ax.set_xscale("log")

    # Prefer sigma_m labels if available.
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

