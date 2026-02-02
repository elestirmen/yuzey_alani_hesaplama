"""Synthetic DEM generators and reference (high-resolution) area integrator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


SurfaceFunc = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True, slots=True)
class SyntheticGrid:
    rows: int
    cols: int
    dx: float
    dy: float

    @property
    def width(self) -> float:
        return float(self.cols) * float(self.dx)

    @property
    def height(self) -> float:
        return float(self.rows) * float(self.dy)


def grid_centers(rows: int, cols: int, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    """Return meshgrid of cell-center coordinates (x, y)."""
    xs = (np.arange(cols, dtype=np.float64) + 0.5) * float(dx)
    ys = (np.arange(rows, dtype=np.float64) + 0.5) * float(dy)
    x, y = np.meshgrid(xs, ys)
    return x, y


def plane(a: float, b: float, c: float = 0.0) -> SurfaceFunc:
    """z = a*x + b*y + c"""

    def f(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return a * x + b * y + c

    return f


def sinusoid(amplitude: float, kx: float, ky: float) -> SurfaceFunc:
    """z = A*sin(kx*x)*sin(ky*y)"""

    def f(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return amplitude * np.sin(kx * x) * np.sin(ky * y)

    return f


def paraboloid(scale: float, *, x0: float = 0.0, y0: float = 0.0) -> SurfaceFunc:
    """z = ((x-x0)^2 + (y-y0)^2) / scale"""

    def f(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return ((x - x0) ** 2 + (y - y0) ** 2) / float(scale)

    return f


def sample_surface_centers(grid: SyntheticGrid, func: SurfaceFunc) -> np.ndarray:
    x, y = grid_centers(grid.rows, grid.cols, grid.dx, grid.dy)
    return func(x, y).astype(np.float64, copy=False)


def reference_area_two_triangles(
    *,
    func: SurfaceFunc,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    fine_dx: float,
    fine_dy: float,
) -> float:
    """Reference surface area by fine sampling and 2-triangle per fine cell.

    The surface is evaluated on a regular grid of (fine_dx, fine_dy) corner nodes, then each
    fine cell is split into two triangles and summed with the 3D cross-product formula.
    """
    width = float(x1) - float(x0)
    height = float(y1) - float(y0)
    if width <= 0 or height <= 0:
        raise ValueError("Invalid extent for reference area")
    if fine_dx <= 0 or fine_dy <= 0:
        raise ValueError("fine_dx and fine_dy must be > 0")

    ncols = int(round(width / float(fine_dx)))
    nrows = int(round(height / float(fine_dy)))
    if not np.isclose(ncols * fine_dx, width) or not np.isclose(nrows * fine_dy, height):
        raise ValueError("Extent must be divisible by fine_dx/fine_dy for reference integration")

    xs = float(x0) + np.arange(ncols + 1, dtype=np.float64) * float(fine_dx)
    ys = float(y0) + np.arange(nrows + 1, dtype=np.float64) * float(fine_dy)
    X, Y = np.meshgrid(xs, ys)
    Z = func(X, Y).astype(np.float64, copy=False)

    p00 = Z[:-1, :-1]
    p10 = Z[:-1, 1:]
    p01 = Z[1:, :-1]
    p11 = Z[1:, 1:]

    dx = float(fine_dx)
    dy = float(fine_dy)

    dz_b = p10 - p00
    dz_c = p11 - p00
    mag1 = np.sqrt((dz_b * dy) ** 2 + (dx * (dz_b - dz_c)) ** 2 + (dx * dy) ** 2)

    dz_b2 = p11 - p00
    dz_c2 = p01 - p00
    mag2 = np.sqrt((dy * (dz_c2 - dz_b2)) ** 2 + (dx * dz_c2) ** 2 + (dx * dy) ** 2)

    area = 0.5 * (mag1 + mag2)
    return float(area.sum(dtype=np.float64))

