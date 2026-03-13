"""Analytic benchmark surfaces and continuous reference integration helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np


SurfaceEvaluator = Callable[[np.ndarray, np.ndarray], np.ndarray]


ANALYTIC_PRESETS = [
    "analytic_plane",
    "analytic_tilted_plane",
    "analytic_sinusoidal",
    "analytic_gaussian_hill",
    "analytic_multi_gaussian",
    "analytic_saddle",
    "analytic_dome",
    "analytic_hybrid_multiscale",
]
_ANALYTIC_PRESET_SET = frozenset(ANALYTIC_PRESETS)


@dataclass(frozen=True, slots=True)
class CircularHoleSpec:
    center_x_m: float
    center_y_m: float
    radius_m: float


@dataclass(frozen=True, slots=True)
class AnalyticSurface:
    preset: str
    description: str
    parameters: dict[str, object]
    evaluate: SurfaceEvaluator
    gradient_x: SurfaceEvaluator
    gradient_y: SurfaceEvaluator


@dataclass(frozen=True, slots=True)
class ContinuousSurfaceReference:
    planar_area_m2: float
    surface_area_m2: float
    surface_ratio: float
    integration_method: str
    samples_x: int
    samples_y: int
    levels: int
    rel_tol: float
    abs_tol: float
    masked_fraction: float

    @property
    def surface_area_km2(self) -> float:
        return self.surface_area_m2 / 1e6

    @property
    def planar_area_km2(self) -> float:
        return self.planar_area_m2 / 1e6

    @property
    def surface_area_ha(self) -> float:
        return self.surface_area_m2 / 1e4

    @property
    def planar_area_ha(self) -> float:
        return self.planar_area_m2 / 1e4


def is_analytic_preset(preset: str) -> bool:
    return preset.strip().lower() in _ANALYTIC_PRESET_SET


def grid_centers(rows: int, cols: int, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    xs = (np.arange(int(cols), dtype=np.float64) + 0.5) * float(dx)
    ys = (np.arange(int(rows), dtype=np.float64) + 0.5) * float(dy)
    return np.meshgrid(xs, ys)


def sample_analytic_surface(
    surface: AnalyticSurface,
    *,
    rows: int,
    cols: int,
    dx: float,
    dy: float,
) -> np.ndarray:
    x, y = grid_centers(rows, cols, dx, dy)
    return surface.evaluate(x, y).astype(np.float64, copy=False)


def generate_circular_holes(
    *,
    rng: np.random.Generator,
    count: int,
    base_radius_m: float,
    width_m: float,
    height_m: float,
    progress: ProgressPrinter | None = None,
    progress_label: str = "nodata",
) -> list[CircularHoleSpec]:
    if int(count) <= 0:
        return []
    if float(base_radius_m) <= 0:
        raise ValueError("base_radius_m must be > 0")
    if float(width_m) <= 0 or float(height_m) <= 0:
        raise ValueError("width_m and height_m must be > 0")

    holes_i = int(count)
    holes: list[CircularHoleSpec] = []
    if progress is not None:
        progress.update(label=progress_label, current=0, total=holes_i)

    for i in range(holes_i):
        holes.append(
            CircularHoleSpec(
                center_x_m=float(rng.uniform(0.0, float(width_m))),
                center_y_m=float(rng.uniform(0.0, float(height_m))),
                radius_m=float(rng.uniform(0.7 * float(base_radius_m), 1.4 * float(base_radius_m))),
            )
        )
        if progress is not None:
            progress.update(label=progress_label, current=i + 1, total=holes_i)
    return holes


def circular_hole_mask(
    x: np.ndarray,
    y: np.ndarray,
    holes: list[CircularHoleSpec] | tuple[CircularHoleSpec, ...],
) -> np.ndarray:
    mask = np.zeros(np.broadcast_shapes(x.shape, y.shape), dtype=bool)
    if not holes:
        return mask
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    mask = np.zeros(xx.shape, dtype=bool)
    for hole in holes:
        mask |= (xx - float(hole.center_x_m)) ** 2 + (yy - float(hole.center_y_m)) ** 2 <= float(hole.radius_m) ** 2
    return mask


def circular_hole_mask_for_grid(
    *,
    rows: int,
    cols: int,
    dx: float,
    dy: float,
    holes: list[CircularHoleSpec] | tuple[CircularHoleSpec, ...],
) -> np.ndarray:
    x, y = grid_centers(rows, cols, dx, dy)
    return circular_hole_mask(x, y, holes)


def build_analytic_surface(
    preset: str,
    *,
    extent_width_m: float,
    extent_height_m: float,
    relief: float = 1.0,
    roughness_m: float = 0.75,
    seed: int = 0,
) -> AnalyticSurface:
    preset_n = preset.strip().lower()
    if preset_n not in _ANALYTIC_PRESET_SET:
        raise ValueError(f"Unknown analytic preset: {preset!r}. Choices: {ANALYTIC_PRESETS}")

    width = float(extent_width_m)
    height = float(extent_height_m)
    if width <= 0 or height <= 0:
        raise ValueError("extent_width_m and extent_height_m must be > 0")

    relief_f = float(relief)
    roughness_f = float(roughness_m)
    if relief_f < 0:
        raise ValueError("relief must be >= 0")
    if roughness_f < 0:
        raise ValueError("roughness_m must be >= 0")

    rng = np.random.default_rng(int(seed))
    cx = 0.5 * width
    cy = 0.5 * height

    if preset_n == "analytic_plane":
        z0 = 100.0

        def f(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            return np.full_like(np.asarray(x, dtype=np.float64), z0, dtype=np.float64)

        def gx(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            return np.zeros_like(np.asarray(x, dtype=np.float64), dtype=np.float64)

        def gy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            return np.zeros_like(np.asarray(y, dtype=np.float64), dtype=np.float64)

        return AnalyticSurface(
            preset=preset_n,
            description="Flat analytic plane with exact continuous area equal to planar area.",
            parameters={"base_height_m": z0},
            evaluate=f,
            gradient_x=gx,
            gradient_y=gy,
        )

    if preset_n == "analytic_tilted_plane":
        z0 = 120.0
        slope_x = 0.08 * relief_f
        slope_y = -0.05 * relief_f

        def f(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            xx = np.asarray(x, dtype=np.float64)
            yy = np.asarray(y, dtype=np.float64)
            return z0 + slope_x * (xx - cx) + slope_y * (yy - cy)

        def gx(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            return np.full_like(np.asarray(x, dtype=np.float64), slope_x, dtype=np.float64)

        def gy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            return np.full_like(np.asarray(y, dtype=np.float64), slope_y, dtype=np.float64)

        return AnalyticSurface(
            preset=preset_n,
            description="Tilted analytic plane with exact constant-slope surface area.",
            parameters={"base_height_m": z0, "slope_x": slope_x, "slope_y": slope_y},
            evaluate=f,
            gradient_x=gx,
            gradient_y=gy,
        )

    if preset_n == "analytic_sinusoidal":
        z0 = 100.0
        amplitude = max(2.0, 10.0 * relief_f + 1.5 * roughness_f)
        wavelength_x = max(width / 2.75, 4.0)
        wavelength_y = max(height / 3.25, 4.0)
        phase_x = float(rng.uniform(0.0, 2.0 * math.pi))
        phase_y = float(rng.uniform(0.0, 2.0 * math.pi))
        kx = 2.0 * math.pi / wavelength_x
        ky = 2.0 * math.pi / wavelength_y

        def f(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            xx = np.asarray(x, dtype=np.float64)
            yy = np.asarray(y, dtype=np.float64)
            return z0 + amplitude * np.sin(kx * xx + phase_x) * np.cos(ky * yy + phase_y)

        def gx(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            xx = np.asarray(x, dtype=np.float64)
            yy = np.asarray(y, dtype=np.float64)
            return amplitude * kx * np.cos(kx * xx + phase_x) * np.cos(ky * yy + phase_y)

        def gy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            xx = np.asarray(x, dtype=np.float64)
            yy = np.asarray(y, dtype=np.float64)
            return -amplitude * ky * np.sin(kx * xx + phase_x) * np.sin(ky * yy + phase_y)

        return AnalyticSurface(
            preset=preset_n,
            description="Smooth sinusoidal benchmark with periodic multi-axis gradients.",
            parameters={
                "base_height_m": z0,
                "amplitude_m": amplitude,
                "wavelength_x_m": wavelength_x,
                "wavelength_y_m": wavelength_y,
                "phase_x_rad": phase_x,
                "phase_y_rad": phase_y,
            },
            evaluate=f,
            gradient_x=gx,
            gradient_y=gy,
        )

    if preset_n == "analytic_gaussian_hill":
        z0 = 95.0
        amplitude = max(3.0, 28.0 * relief_f + 1.25 * roughness_f)
        sigma_x = max(width / 5.5, 1.0)
        sigma_y = max(height / 5.0, 1.0)

        def _exp_term(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            xx = np.asarray(x, dtype=np.float64)
            yy = np.asarray(y, dtype=np.float64)
            return np.exp(-0.5 * (((xx - cx) / sigma_x) ** 2 + ((yy - cy) / sigma_y) ** 2))

        def f(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            return z0 + amplitude * _exp_term(x, y)

        def gx(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            xx = np.asarray(x, dtype=np.float64)
            return amplitude * _exp_term(x, y) * (-(xx - cx) / (sigma_x * sigma_x))

        def gy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            yy = np.asarray(y, dtype=np.float64)
            return amplitude * _exp_term(x, y) * (-(yy - cy) / (sigma_y * sigma_y))

        return AnalyticSurface(
            preset=preset_n,
            description="Single Gaussian hill with smooth radial decay.",
            parameters={
                "base_height_m": z0,
                "amplitude_m": amplitude,
                "center_x_m": cx,
                "center_y_m": cy,
                "sigma_x_m": sigma_x,
                "sigma_y_m": sigma_y,
            },
            evaluate=f,
            gradient_x=gx,
            gradient_y=gy,
        )

    if preset_n == "analytic_multi_gaussian":
        z0 = 110.0
        n_components = 4
        components: list[dict[str, float]] = []
        for idx in range(n_components):
            amp_sign = -1.0 if idx == 1 else 1.0
            components.append(
                {
                    "amplitude_m": amp_sign * float(rng.uniform(8.0, 24.0)) * max(relief_f, 0.25),
                    "center_x_m": float(rng.uniform(0.15 * width, 0.85 * width)),
                    "center_y_m": float(rng.uniform(0.15 * height, 0.85 * height)),
                    "sigma_x_m": float(rng.uniform(0.08 * width, 0.22 * width)),
                    "sigma_y_m": float(rng.uniform(0.08 * height, 0.22 * height)),
                }
            )

        def _component_sum(x: np.ndarray, y: np.ndarray, *, derivative: str | None) -> np.ndarray:
            xx = np.asarray(x, dtype=np.float64)
            yy = np.asarray(y, dtype=np.float64)
            acc = np.zeros_like(xx, dtype=np.float64)
            for comp in components:
                amp = float(comp["amplitude_m"])
                sx = float(comp["sigma_x_m"])
                sy = float(comp["sigma_y_m"])
                dx_n = (xx - float(comp["center_x_m"])) / sx
                dy_n = (yy - float(comp["center_y_m"])) / sy
                g = np.exp(-0.5 * (dx_n * dx_n + dy_n * dy_n))
                if derivative is None:
                    acc += amp * g
                elif derivative == "x":
                    acc += amp * g * (-(xx - float(comp["center_x_m"])) / (sx * sx))
                elif derivative == "y":
                    acc += amp * g * (-(yy - float(comp["center_y_m"])) / (sy * sy))
            return acc

        def f(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            return z0 + _component_sum(x, y, derivative=None)

        def gx(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            return _component_sum(x, y, derivative="x")

        def gy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            return _component_sum(x, y, derivative="y")

        return AnalyticSurface(
            preset=preset_n,
            description="Multiple analytic Gaussian hills and depressions with known continuous derivatives.",
            parameters={"base_height_m": z0, "components": components},
            evaluate=f,
            gradient_x=gx,
            gradient_y=gy,
        )

    if preset_n == "analytic_saddle":
        z0 = 105.0
        amplitude = max(4.0, 18.0 * relief_f)
        scale_x = max(width / 3.2, 1.0)
        scale_y = max(height / 3.2, 1.0)

        def f(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            xx = np.asarray(x, dtype=np.float64)
            yy = np.asarray(y, dtype=np.float64)
            return z0 + amplitude * (((xx - cx) / scale_x) ** 2 - ((yy - cy) / scale_y) ** 2)

        def gx(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            xx = np.asarray(x, dtype=np.float64)
            return amplitude * (2.0 * (xx - cx) / (scale_x * scale_x))

        def gy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            yy = np.asarray(y, dtype=np.float64)
            return amplitude * (-2.0 * (yy - cy) / (scale_y * scale_y))

        return AnalyticSurface(
            preset=preset_n,
            description="Hyperbolic saddle surface with opposing principal curvatures.",
            parameters={
                "base_height_m": z0,
                "amplitude_m": amplitude,
                "scale_x_m": scale_x,
                "scale_y_m": scale_y,
            },
            evaluate=f,
            gradient_x=gx,
            gradient_y=gy,
        )

    if preset_n == "analytic_dome":
        z0 = 90.0
        amplitude = max(3.0, 26.0 * relief_f)
        radius = 0.42 * min(width, height)

        def _inside(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            xx = np.asarray(x, dtype=np.float64)
            yy = np.asarray(y, dtype=np.float64)
            q = 1.0 - (((xx - cx) ** 2 + (yy - cy) ** 2) / (radius * radius))
            return np.clip(q, 0.0, None)

        def f(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            inside = _inside(x, y)
            return z0 + amplitude * inside * inside

        def gx(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            xx = np.asarray(x, dtype=np.float64)
            inside = _inside(x, y)
            return np.where(inside > 0.0, -4.0 * amplitude * inside * (xx - cx) / (radius * radius), 0.0)

        def gy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            yy = np.asarray(y, dtype=np.float64)
            inside = _inside(x, y)
            return np.where(inside > 0.0, -4.0 * amplitude * inside * (yy - cy) / (radius * radius), 0.0)

        return AnalyticSurface(
            preset=preset_n,
            description="Compact smooth dome with finite support and continuous slopes at the rim.",
            parameters={
                "base_height_m": z0,
                "amplitude_m": amplitude,
                "center_x_m": cx,
                "center_y_m": cy,
                "radius_m": radius,
            },
            evaluate=f,
            gradient_x=gx,
            gradient_y=gy,
        )

    if preset_n == "analytic_hybrid_multiscale":
        z0 = 100.0
        broad_amp = max(2.0, 8.0 * relief_f)
        rough_amp = max(1.5, 3.0 * relief_f + 2.0 * roughness_f)
        broad_wlx = max(width / 1.8, 4.0)
        broad_wly = max(height / 2.1, 4.0)
        patch_cx = float(rng.uniform(0.6 * width, 0.8 * width))
        patch_cy = float(rng.uniform(0.25 * height, 0.45 * height))
        patch_sx = max(0.09 * width, 1.0)
        patch_sy = max(0.11 * height, 1.0)
        rough_l1 = max(min(width, height) / 18.0, 1.5)
        rough_l2 = max(min(width, height) / 24.0, 1.2)
        rough_l3 = max(min(width, height) / 32.0, 1.0)
        phase1 = float(rng.uniform(0.0, 2.0 * math.pi))
        phase2 = float(rng.uniform(0.0, 2.0 * math.pi))
        phase3 = float(rng.uniform(0.0, 2.0 * math.pi))
        gaussian_amp = max(1.5, 10.0 * relief_f)
        gaussian_sigma = max(0.18 * min(width, height), 1.0)

        def _rough_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            xx = np.asarray(x, dtype=np.float64)
            yy = np.asarray(y, dtype=np.float64)
            return np.exp(-0.5 * (((xx - patch_cx) / patch_sx) ** 2 + ((yy - patch_cy) / patch_sy) ** 2))

        def _rough_mask_dx(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            xx = np.asarray(x, dtype=np.float64)
            mask = _rough_mask(x, y)
            return mask * (-(xx - patch_cx) / (patch_sx * patch_sx))

        def _rough_mask_dy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            yy = np.asarray(y, dtype=np.float64)
            mask = _rough_mask(x, y)
            return mask * (-(yy - patch_cy) / (patch_sy * patch_sy))

        def _broad_base(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            xx = np.asarray(x, dtype=np.float64)
            yy = np.asarray(y, dtype=np.float64)
            return broad_amp * np.sin(2.0 * math.pi * xx / broad_wlx) * np.cos(2.0 * math.pi * yy / broad_wly)

        def _broad_dx(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            xx = np.asarray(x, dtype=np.float64)
            yy = np.asarray(y, dtype=np.float64)
            return (
                broad_amp
                * (2.0 * math.pi / broad_wlx)
                * np.cos(2.0 * math.pi * xx / broad_wlx)
                * np.cos(2.0 * math.pi * yy / broad_wly)
            )

        def _broad_dy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            xx = np.asarray(x, dtype=np.float64)
            yy = np.asarray(y, dtype=np.float64)
            return (
                -broad_amp
                * (2.0 * math.pi / broad_wly)
                * np.sin(2.0 * math.pi * xx / broad_wlx)
                * np.sin(2.0 * math.pi * yy / broad_wly)
            )

        def _gaussian(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            xx = np.asarray(x, dtype=np.float64)
            yy = np.asarray(y, dtype=np.float64)
            return np.exp(-0.5 * (((xx - 0.33 * width) / gaussian_sigma) ** 2 + ((yy - 0.7 * height) / gaussian_sigma) ** 2))

        def _gaussian_dx(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            xx = np.asarray(x, dtype=np.float64)
            return _gaussian(x, y) * (-(xx - 0.33 * width) / (gaussian_sigma * gaussian_sigma))

        def _gaussian_dy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            yy = np.asarray(y, dtype=np.float64)
            return _gaussian(x, y) * (-(yy - 0.7 * height) / (gaussian_sigma * gaussian_sigma))

        def _rough_signal(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            xx = np.asarray(x, dtype=np.float64)
            yy = np.asarray(y, dtype=np.float64)
            return (
                np.sin(2.0 * math.pi * xx / rough_l1 + phase1)
                + 0.6 * np.cos(2.0 * math.pi * yy / rough_l2 + phase2)
                + 0.35 * np.sin(2.0 * math.pi * (xx + yy) / rough_l3 + phase3)
            )

        def _rough_signal_dx(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            xx = np.asarray(x, dtype=np.float64)
            yy = np.asarray(y, dtype=np.float64)
            return (
                (2.0 * math.pi / rough_l1) * np.cos(2.0 * math.pi * xx / rough_l1 + phase1)
                + 0.35 * (2.0 * math.pi / rough_l3) * np.cos(2.0 * math.pi * (xx + yy) / rough_l3 + phase3)
            )

        def _rough_signal_dy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            xx = np.asarray(x, dtype=np.float64)
            yy = np.asarray(y, dtype=np.float64)
            return (
                -0.6 * (2.0 * math.pi / rough_l2) * np.sin(2.0 * math.pi * yy / rough_l2 + phase2)
                + 0.35 * (2.0 * math.pi / rough_l3) * np.cos(2.0 * math.pi * (xx + yy) / rough_l3 + phase3)
            )

        def f(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            mask = _rough_mask(x, y)
            return z0 + _broad_base(x, y) + gaussian_amp * _gaussian(x, y) + rough_amp * mask * _rough_signal(x, y)

        def gx(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            mask = _rough_mask(x, y)
            return (
                _broad_dx(x, y)
                + gaussian_amp * _gaussian_dx(x, y)
                + rough_amp * (_rough_mask_dx(x, y) * _rough_signal(x, y) + mask * _rough_signal_dx(x, y))
            )

        def gy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            mask = _rough_mask(x, y)
            return (
                _broad_dy(x, y)
                + gaussian_amp * _gaussian_dy(x, y)
                + rough_amp * (_rough_mask_dy(x, y) * _rough_signal(x, y) + mask * _rough_signal_dy(x, y))
            )

        return AnalyticSurface(
            preset=preset_n,
            description="Hybrid analytic terrain with smooth background and a localized rough patch for adaptive refinement tests.",
            parameters={
                "base_height_m": z0,
                "broad_amplitude_m": broad_amp,
                "rough_amplitude_m": rough_amp,
                "broad_wavelength_x_m": broad_wlx,
                "broad_wavelength_y_m": broad_wly,
                "rough_patch_center_x_m": patch_cx,
                "rough_patch_center_y_m": patch_cy,
                "rough_patch_sigma_x_m": patch_sx,
                "rough_patch_sigma_y_m": patch_sy,
                "rough_wavelength_1_m": rough_l1,
                "rough_wavelength_2_m": rough_l2,
                "rough_wavelength_3_m": rough_l3,
                "gaussian_amplitude_m": gaussian_amp,
                "gaussian_sigma_m": gaussian_sigma,
            },
            evaluate=f,
            gradient_x=gx,
            gradient_y=gy,
        )

    raise AssertionError(f"Unhandled analytic preset: {preset_n}")


def compute_continuous_surface_reference(
    surface: AnalyticSurface,
    *,
    extent_width_m: float,
    extent_height_m: float,
    holes: list[CircularHoleSpec] | tuple[CircularHoleSpec, ...] | None = None,
    rel_tol: float = 1e-7,
    abs_tol: float = 1e-6,
    base_samples: int = 129,
    max_levels: int = 5,
) -> ContinuousSurfaceReference:
    width = float(extent_width_m)
    height = float(extent_height_m)
    if width <= 0 or height <= 0:
        raise ValueError("extent_width_m and extent_height_m must be > 0")
    if rel_tol < 0 or abs_tol < 0:
        raise ValueError("rel_tol and abs_tol must be >= 0")
    if int(base_samples) < 5:
        raise ValueError("base_samples must be >= 5")
    if int(max_levels) < 0:
        raise ValueError("max_levels must be >= 0")

    holes_list = list(holes or [])
    if not holes_list and surface.preset in {"analytic_plane", "analytic_tilted_plane"}:
        planar_area = width * height
        if surface.preset == "analytic_plane":
            surface_area = planar_area
        else:
            slope_x = float(surface.parameters["slope_x"])
            slope_y = float(surface.parameters["slope_y"])
            surface_area = planar_area * math.sqrt(1.0 + slope_x * slope_x + slope_y * slope_y)
        return ContinuousSurfaceReference(
            planar_area_m2=planar_area,
            surface_area_m2=surface_area,
            surface_ratio=surface_area / planar_area if planar_area > 0 else 1.0,
            integration_method="exact_closed_form",
            samples_x=0,
            samples_y=0,
            levels=0,
            rel_tol=float(rel_tol),
            abs_tol=float(abs_tol),
            masked_fraction=0.0,
        )

    from scipy.integrate import simpson

    nx, ny = _initial_sample_shape(width=width, height=height, base_samples=int(base_samples))
    prev_surface: float | None = None
    prev_planar: float | None = None
    final_surface = 0.0
    final_planar = 0.0
    levels_run = 0

    for level in range(int(max_levels) + 1):
        xs = np.linspace(0.0, width, nx, dtype=np.float64)
        ys = np.linspace(0.0, height, ny, dtype=np.float64)
        xg, yg = np.meshgrid(xs, ys)

        grad_x = surface.gradient_x(xg, yg).astype(np.float64, copy=False)
        grad_y = surface.gradient_y(xg, yg).astype(np.float64, copy=False)
        integrand = np.sqrt(1.0 + grad_x * grad_x + grad_y * grad_y, dtype=np.float64)
        planar_density = np.ones_like(integrand, dtype=np.float64)

        if holes_list:
            mask = circular_hole_mask(xg, yg, holes_list)
            integrand = np.where(mask, 0.0, integrand)
            planar_density = np.where(mask, 0.0, planar_density)

        final_surface = float(simpson(simpson(integrand, x=xs, axis=1), x=ys, axis=0))
        final_planar = float(simpson(simpson(planar_density, x=xs, axis=1), x=ys, axis=0))
        levels_run = level + 1

        if prev_surface is not None and prev_planar is not None:
            if _within_tolerance(final_surface, prev_surface, rel_tol=rel_tol, abs_tol=abs_tol) and _within_tolerance(
                final_planar,
                prev_planar,
                rel_tol=rel_tol,
                abs_tol=abs_tol,
            ):
                break

        prev_surface = final_surface
        prev_planar = final_planar
        nx = 2 * nx - 1
        ny = 2 * ny - 1

    masked_fraction = 0.0
    full_planar = width * height
    if full_planar > 0:
        masked_fraction = max(0.0, min(1.0, 1.0 - (final_planar / full_planar)))

    return ContinuousSurfaceReference(
        planar_area_m2=final_planar,
        surface_area_m2=final_surface,
        surface_ratio=final_surface / final_planar if final_planar > 0 else 1.0,
        integration_method="adaptive_simpson_grid",
        samples_x=nx,
        samples_y=ny,
        levels=levels_run,
        rel_tol=float(rel_tol),
        abs_tol=float(abs_tol),
        masked_fraction=masked_fraction,
    )


def _initial_sample_shape(*, width: float, height: float, base_samples: int) -> tuple[int, int]:
    longest = max(width, height)
    if longest <= 0:
        return 5, 5
    nx = _as_odd(max(5, int(round(base_samples * width / longest))))
    ny = _as_odd(max(5, int(round(base_samples * height / longest))))
    return nx, ny


def _as_odd(value: int) -> int:
    return value if value % 2 == 1 else value + 1


def _within_tolerance(current: float, previous: float, *, rel_tol: float, abs_tol: float) -> bool:
    delta = abs(float(current) - float(previous))
    scale = max(abs(float(current)), abs(float(previous)), 1.0)
    return delta <= max(float(abs_tol), float(rel_tol) * scale)
