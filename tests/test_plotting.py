from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from surface_area.plotting import (
    _reference_curve,
    plot_a3d_vs_gsd,
    plot_continuous_gt_rel_err_vs_gsd,
    plot_error_vs_runtime,
    plot_native_grid_ref_rel_err_vs_gsd,
    plot_ratio_vs_gsd,
    plot_surface_excess_vs_gsd,
    plot_runtime_vs_gsd,
)


def _demo_results_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gsd_m": [1.0, 1.0, 2.0, 2.0],
            "method": [
                "gradient_multiplier",
                "tin_2tri_cell",
                "gradient_multiplier",
                "tin_2tri_cell",
            ],
            "A3D": [101.0, 99.5, 95.0, 94.5],
            "ratio": [1.010, 0.995, 0.950, 0.945],
            "synthetic_native_ref_A3D": [100.0, 100.0, 96.0, 96.0],
            "synthetic_native_ref_ratio": [1.000, 1.000, 0.960, 0.960],
            "continuous_gt_A3D": [110.0, 110.0, 110.0, 110.0],
            "continuous_gt_ratio": [1.100, 1.100, 1.100, 1.100],
            "A3D_continuous_gt_rel_err": [
                (101.0 - 110.0) / 110.0,
                (99.5 - 110.0) / 110.0,
                (95.0 - 110.0) / 110.0,
                (94.5 - 110.0) / 110.0,
            ],
            "A3D_synthetic_native_ref_rel_err": [
                (101.0 - 100.0) / 100.0,
                (99.5 - 100.0) / 100.0,
                (95.0 - 96.0) / 96.0,
                (94.5 - 96.0) / 96.0,
            ],
            "runtime_sec": [1.0, 1.5, 2.0, 2.5],
            "resample_runtime_sec": [0.2, 0.2, 0.4, 0.4],
        }
    )


def _capture_lines(
    monkeypatch,
    plot_fn,
    df_long: pd.DataFrame,
    outdir: Path,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    captured: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def _spy_savefig(self: Figure, *args, **kwargs) -> None:
        ax = self.axes[0]
        for line in ax.get_lines():
            captured[line.get_label()] = (
                np.asarray(line.get_xdata(), dtype=float),
                np.asarray(line.get_ydata(), dtype=float),
            )

    monkeypatch.setattr(Figure, "savefig", _spy_savefig)
    plot_fn(df_long, outdir)
    return captured


def _capture_scatter_offsets(
    monkeypatch,
    plot_fn,
    df_long: pd.DataFrame,
    outdir: Path,
) -> list[np.ndarray]:
    captured: list[np.ndarray] = []

    def _spy_savefig(self: Figure, *args, **kwargs) -> None:
        ax = self.axes[0]
        for collection in ax.collections:
            offsets = np.asarray(collection.get_offsets(), dtype=float)
            if offsets.size:
                captured.append(offsets)

    monkeypatch.setattr(Figure, "savefig", _spy_savefig)
    plot_fn(df_long, outdir)
    return captured


def test_reference_curve_collapses_duplicate_methods() -> None:
    df = _demo_results_frame()

    curve = _reference_curve(df, "synthetic_native_ref_A3D")

    assert curve["gsd_m"].tolist() == [1.0, 2.0]
    assert curve["synthetic_native_ref_A3D"].tolist() == [100.0, 96.0]


def test_plot_a3d_vs_gsd_adds_reference_lines(monkeypatch, tmp_path: Path) -> None:
    captured = _capture_lines(monkeypatch, plot_a3d_vs_gsd, _demo_results_frame(), tmp_path)

    assert "Continuous ground truth (GSD-independent)" in captured
    assert "Native-grid reference (per GSD)" in captured
    assert np.allclose(captured["Continuous ground truth (GSD-independent)"][1], [110.0, 110.0])
    assert np.allclose(captured["Native-grid reference (per GSD)"][1], [100.0, 96.0])


def test_plot_ratio_vs_gsd_adds_reference_lines(monkeypatch, tmp_path: Path) -> None:
    captured = _capture_lines(monkeypatch, plot_ratio_vs_gsd, _demo_results_frame(), tmp_path)

    assert "Continuous ground truth (GSD-independent)" in captured
    assert "Native-grid reference (per GSD)" in captured
    assert np.allclose(captured["Continuous ground truth (GSD-independent)"][1], [1.1, 1.1])
    assert np.allclose(captured["Native-grid reference (per GSD)"][1], [1.0, 0.96])


def test_plot_surface_excess_vs_gsd_adds_reference_lines(monkeypatch, tmp_path: Path) -> None:
    captured = _capture_lines(monkeypatch, plot_surface_excess_vs_gsd, _demo_results_frame(), tmp_path)

    assert "Continuous ground truth (GSD-independent)" in captured
    assert "Native-grid reference (per GSD)" in captured
    assert np.allclose(captured["Continuous ground truth (GSD-independent)"][1], [10.0, 10.0])
    assert np.allclose(captured["Native-grid reference (per GSD)"][1], [0.0, -4.0])


def test_plot_surface_excess_vs_gsd_handles_missing_reference_columns(monkeypatch, tmp_path: Path) -> None:
    df = _demo_results_frame()[["gsd_m", "method", "ratio"]].copy()

    captured = _capture_lines(monkeypatch, plot_surface_excess_vs_gsd, df, tmp_path)

    assert "gradient_multiplier" in captured
    assert "tin_2tri_cell" in captured
    assert "Native-grid reference (per GSD)" not in captured
    assert "Continuous ground truth (GSD-independent)" not in captured
    assert np.allclose(captured["gradient_multiplier"][1], [1.0, -5.0])


def test_plot_continuous_gt_rel_err_vs_gsd_uses_continuous_error_column(monkeypatch, tmp_path: Path) -> None:
    captured = _capture_lines(monkeypatch, plot_continuous_gt_rel_err_vs_gsd, _demo_results_frame(), tmp_path)

    assert np.allclose(captured["gradient_multiplier"][1], [(101.0 - 110.0) / 110.0, (95.0 - 110.0) / 110.0])
    assert np.allclose(captured["tin_2tri_cell"][1], [(99.5 - 110.0) / 110.0, (94.5 - 110.0) / 110.0])


def test_plot_native_grid_ref_rel_err_vs_gsd_uses_native_reference_error_column(monkeypatch, tmp_path: Path) -> None:
    captured = _capture_lines(monkeypatch, plot_native_grid_ref_rel_err_vs_gsd, _demo_results_frame(), tmp_path)

    assert np.allclose(captured["gradient_multiplier"][1], [(101.0 - 100.0) / 100.0, (95.0 - 96.0) / 96.0])
    assert np.allclose(captured["tin_2tri_cell"][1], [(99.5 - 100.0) / 100.0, (94.5 - 96.0) / 96.0])


def test_plot_runtime_vs_gsd_uses_total_runtime(monkeypatch, tmp_path: Path) -> None:
    captured = _capture_lines(monkeypatch, plot_runtime_vs_gsd, _demo_results_frame(), tmp_path)

    assert np.allclose(captured["gradient_multiplier"][1], [1.2, 2.4])
    assert np.allclose(captured["tin_2tri_cell"][1], [1.7, 2.9])


def test_plot_error_vs_runtime_uses_absolute_continuous_relative_error(monkeypatch, tmp_path: Path) -> None:
    collections = _capture_scatter_offsets(monkeypatch, plot_error_vs_runtime, _demo_results_frame(), tmp_path)

    assert len(collections) == 2
    flattened = np.vstack(collections)
    assert flattened.shape == (4, 2)
    expected = np.array(
        [
            [1.2, abs((101.0 - 110.0) / 110.0)],
            [2.4, abs((95.0 - 110.0) / 110.0)],
            [1.7, abs((99.5 - 110.0) / 110.0)],
            [2.9, abs((94.5 - 110.0) / 110.0)],
        ]
    )
    assert np.allclose(flattened[np.argsort(flattened[:, 0])], expected[np.argsort(expected[:, 0])])
