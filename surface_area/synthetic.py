"""Synthetic DEM generators and reference (high-resolution) area integrator.

Bu modül, yüzey alanı hesaplama algoritmalarını test etmek için gerçekçi
sentetik arazi modelleri üretir.

GERÇEKÇI ARAZİ TİPLERİ:
----------------------
- mountain:     Dağlık arazi (fBm noise + sırtlar)
- valley:       Vadi ve akarsu yatakları
- hills:        Yumuşak tepeler (rolling hills)
- coastal:      Kıyı şeridi (deniz-kara geçişi)
- plateau:      Yüksek plato ve yamaçlar
- canyon:       Kanyon/boğaz yapıları
- volcanic:     Volkanik arazi (kraterler, lav akışı)
- glacial:      Buzul vadisi (U-şekilli)
- karst:        Karstik arazi (düdenler, mağaralar)
- alluvial:     Alüvyal ova/delta

TEST PATTERNLERİ:
----------------
- plane:        Düz eğimli yüzey
- waves:        Sinüzoidal dalgalar
- crater_field: Krater alanı
- terraced:     Teraslı arazi
- patchwork:    Karışık yüzeyler
- mixed:        Maksimum çeşitlilik
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np


SurfaceFunc = Callable[[np.ndarray, np.ndarray], np.ndarray]

# Test pattern preset'leri (eski)
_TEST_PRESETS = [
    "plane",
    "waves",
    "crater_field",
    "terraced",
    "patchwork",
    "mixed",
]

# Gerçekçi arazi preset'leri (yeni)
_REALISTIC_PRESETS = [
    "mountain",
    "valley",
    "hills",
    "coastal",
    "plateau",
    "canyon",
    "volcanic",
    "glacial",
    "karst",
    "alluvial",
]

# Tüm preset'ler
SYNTHETIC_PRESETS = _TEST_PRESETS + _REALISTIC_PRESETS


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
    """Reference surface area by fine sampling and 2-triangle per fine cell."""
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


@dataclass(frozen=True, slots=True)
class SurfaceAreaResult:
    """Yüzey alanı hesaplama sonucu.

    Attributes:
        surface_area_m2: Gerçek 3D yüzey alanı (metrekare)
        planar_area_m2: Düzlemsel (2D projeksiyon) alan (metrekare)
        surface_ratio: Yüzey alanı / düzlemsel alan oranı (>=1.0)
        rows: Satır sayısı
        cols: Sütun sayısı
        dx: X piksel boyutu (metre)
        dy: Y piksel boyutu (metre)
        valid_cells: Geçerli (nodata olmayan) hücre sayısı
        nodata_cells: Nodata hücre sayısı
    """
    surface_area_m2: float
    planar_area_m2: float
    surface_ratio: float
    rows: int
    cols: int
    dx: float
    dy: float
    valid_cells: int
    nodata_cells: int

    @property
    def surface_area_km2(self) -> float:
        """Yüzey alanı km² cinsinden."""
        return self.surface_area_m2 / 1e6

    @property
    def planar_area_km2(self) -> float:
        """Düzlemsel alan km² cinsinden."""
        return self.planar_area_m2 / 1e6

    @property
    def surface_area_ha(self) -> float:
        """Yüzey alanı hektar cinsinden."""
        return self.surface_area_m2 / 1e4

    @property
    def planar_area_ha(self) -> float:
        """Düzlemsel alan hektar cinsinden."""
        return self.planar_area_m2 / 1e4


def compute_reference_surface_area(
    z: np.ndarray,
    *,
    dx: float,
    dy: float,
    nodata_value: float | None = None,
) -> SurfaceAreaResult:
    """Bir yükseklik dizisinin GERÇEK (referans) yüzey alanını hesaplar.

    Her hücre iki üçgene bölünerek 3D yüzey alanı hesaplanır.
    Bu, sentetik verilerin benchmark olarak kullanılması için
    "ground truth" değerini sağlar.

    Yöntem:
    -------
    Her piksel hücresinin 4 köşe noktası alınır (z değerleri enterpolasyonla).
    Hücre iki üçgene bölünür ve her üçgenin 3D alanı cross-product ile hesaplanır.

    Args:
        z: 2D yükseklik dizisi (rows x cols)
        dx: X yönünde piksel boyutu (metre)
        dy: Y yönünde piksel boyutu (metre)
        nodata_value: Nodata değeri (None ise tüm hücreler geçerli kabul edilir)

    Returns:
        SurfaceAreaResult: Detaylı yüzey alanı bilgisi

    Örnek:
        >>> z = generate_synthetic_dsm(rows=1000, cols=1000, dx=1.0, preset="mountain")
        >>> result = compute_reference_surface_area(z, dx=1.0, dy=1.0)
        >>> print(f"Gerçek yüzey alanı: {result.surface_area_m2:.2f} m²")
        >>> print(f"Yüzey/Düzlem oranı: {result.surface_ratio:.4f}")
    """
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError("z must be a 2D array")

    rows, cols = z.shape
    if rows < 2 or cols < 2:
        raise ValueError("z must have at least 2 rows and 2 columns")

    dx = float(dx)
    dy = float(dy)
    if dx <= 0 or dy <= 0:
        raise ValueError("dx and dy must be positive")

    # Nodata mask oluştur
    if nodata_value is not None:
        nodata_mask = np.isclose(z, float(nodata_value)) | np.isnan(z)
    else:
        nodata_mask = np.isnan(z)

    valid_mask = ~nodata_mask
    nodata_cells = int(nodata_mask.sum())
    valid_cells = int(valid_mask.sum())

    # Hücre köşe noktalarındaki z değerlerini al
    # Her hücre (i,j) için 4 köşe: (i,j), (i,j+1), (i+1,j), (i+1,j+1)
    # Bunlar için z değerlerini hücre merkezlerinden enterpolasyon yaparak hesapla

    # Basit yaklaşım: Hücre merkezlerini köşe noktaları gibi kullan
    # Bu, piksel sayısı yeterince büyükse iyi bir yaklaşım
    z00 = z[:-1, :-1]  # Sol üst
    z10 = z[:-1, 1:]   # Sağ üst
    z01 = z[1:, :-1]   # Sol alt
    z11 = z[1:, 1:]    # Sağ alt

    # Herhangi bir köşesi nodata olan hücreleri atla
    if nodata_value is not None:
        nv = float(nodata_value)
        cell_valid = (
            ~np.isclose(z00, nv) & ~np.isnan(z00) &
            ~np.isclose(z10, nv) & ~np.isnan(z10) &
            ~np.isclose(z01, nv) & ~np.isnan(z01) &
            ~np.isclose(z11, nv) & ~np.isnan(z11)
        )
    else:
        cell_valid = ~np.isnan(z00) & ~np.isnan(z10) & ~np.isnan(z01) & ~np.isnan(z11)

    # Üçgen 1: (0,0) -> (1,0) -> (1,1)
    # Vektörler: A = (dx, 0, z10-z00), B = (dx, dy, z11-z00)
    # Alan = 0.5 * |A x B|

    # Üçgen 1 için vektörler
    # P0 = (0, 0, z00)
    # P1 = (dx, 0, z10)
    # P2 = (dx, dy, z11)
    # A = P1 - P0 = (dx, 0, z10-z00)
    # B = P2 - P0 = (dx, dy, z11-z00)

    dz_a = z10 - z00
    dz_b = z11 - z00

    # Cross product: A x B
    # |i    j    k   |
    # |dx   0    dz_a|
    # |dx   dy   dz_b|
    # = i(0*dz_b - dz_a*dy) - j(dx*dz_b - dz_a*dx) + k(dx*dy - 0*dx)
    # = (-dz_a*dy, -dx*(dz_b - dz_a), dx*dy)

    cross1_x = -dz_a * dy
    cross1_y = -dx * (dz_b - dz_a)
    cross1_z = dx * dy
    mag1 = np.sqrt(cross1_x**2 + cross1_y**2 + cross1_z**2)

    # Üçgen 2: (0,0) -> (1,1) -> (0,1)
    # Vektörler: A = (dx, dy, z11-z00), B = (0, dy, z01-z00)

    dz_c = z01 - z00

    # Cross product
    # |i    j    k   |
    # |dx   dy   dz_b|
    # |0    dy   dz_c|
    # = i(dy*dz_c - dz_b*dy) - j(dx*dz_c - dz_b*0) + k(dx*dy - dy*0)
    # = (dy*(dz_c - dz_b), -dx*dz_c, dx*dy)

    cross2_x = dy * (dz_c - dz_b)
    cross2_y = -dx * dz_c
    cross2_z = dx * dy
    mag2 = np.sqrt(cross2_x**2 + cross2_y**2 + cross2_z**2)

    # Her hücrenin alanı = 0.5 * (|cross1| + |cross2|)
    cell_areas = 0.5 * (mag1 + mag2)

    # Geçersiz hücreleri sıfırla
    cell_areas = np.where(cell_valid, cell_areas, 0.0)

    # Toplam yüzey alanı
    surface_area = float(cell_areas.sum(dtype=np.float64))

    # Düzlemsel alan (geçerli hücreler için)
    planar_cell_area = dx * dy
    valid_cell_count = int(cell_valid.sum())
    planar_area = valid_cell_count * planar_cell_area

    # Oran
    if planar_area > 0:
        surface_ratio = surface_area / planar_area
    else:
        surface_ratio = 1.0

    return SurfaceAreaResult(
        surface_area_m2=surface_area,
        planar_area_m2=planar_area,
        surface_ratio=surface_ratio,
        rows=rows,
        cols=cols,
        dx=dx,
        dy=dy,
        valid_cells=valid_cells,
        nodata_cells=nodata_cells,
    )


# =============================================================================
# YARDIMCI FONKSİYONLAR
# =============================================================================

def _smoothstep01(t: np.ndarray) -> np.ndarray:
    """Smoothstep on [0,1] with clamping."""
    tt = np.clip(t, 0.0, 1.0)
    return tt * tt * (3.0 - 2.0 * tt)


def _quintic_smoothstep(t: np.ndarray) -> np.ndarray:
    """Quintic (6t^5 - 15t^4 + 10t^3) smoothstep - daha pürüzsüz geçiş."""
    tt = np.clip(t, 0.0, 1.0)
    return tt * tt * tt * (tt * (tt * 6.0 - 15.0) + 10.0)


def _fractal_gaussian_noise(
    *,
    rng: np.random.Generator,
    rows: int,
    cols: int,
    sigmas_px: list[float],
    amps: list[float],
) -> np.ndarray:
    """Fractal-ish noise: sum of Gaussian-smoothed white noise fields."""
    if rows <= 0 or cols <= 0:
        raise ValueError("rows/cols must be > 0")
    if len(sigmas_px) != len(amps):
        raise ValueError("sigmas_px and amps must have same length")
    if not sigmas_px:
        return np.zeros((rows, cols), dtype=np.float64)

    try:
        from scipy.ndimage import gaussian_filter
    except Exception as e:
        raise RuntimeError("scipy is required for synthetic roughness generation") from e

    acc = np.zeros((rows, cols), dtype=np.float64)
    for sigma, amp in zip(sigmas_px, amps, strict=True):
        s = float(sigma)
        a = float(amp)
        if s <= 0 or a == 0:
            continue
        white = rng.standard_normal((rows, cols)).astype(np.float64, copy=False)
        sm = gaussian_filter(white, sigma=s, mode="reflect")
        sm = sm - float(sm.mean(dtype=np.float64))
        std = float(sm.std(dtype=np.float64))
        if std > 0:
            sm = sm / std
        acc += a * sm
    return acc


def _add_gaussian_bumps(
    z: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    rng: np.random.Generator,
    count: int,
    amp_range: tuple[float, float],
    sigma_m_range: tuple[float, float],
) -> np.ndarray:
    """Rastgele Gaussian çıkıntılar/çukurlar ekler."""
    if count <= 0:
        return z
    z2 = z.copy()
    rows, cols = z.shape
    if cols > 1:
        dx = float(x[0, 1] - x[0, 0])
    else:
        dx = float(x[0, 0]) * 2.0
    if rows > 1:
        dy = float(y[1, 0] - y[0, 0])
    else:
        dy = float(y[0, 0]) * 2.0
    width = float(cols) * dx
    height = float(rows) * dy

    a0, a1 = float(amp_range[0]), float(amp_range[1])
    s0, s1 = float(sigma_m_range[0]), float(sigma_m_range[1])
    if s0 <= 0 or s1 <= 0:
        raise ValueError("sigma_m_range must be > 0")

    for _ in range(int(count)):
        amp = float(rng.uniform(a0, a1))
        x0 = float(rng.uniform(0.0, width))
        y0 = float(rng.uniform(0.0, height))
        sx = float(rng.uniform(s0, s1))
        sy = float(rng.uniform(s0, s1))
        g = np.exp(-0.5 * (((x - x0) / sx) ** 2 + ((y - y0) / sy) ** 2))
        z2 = z2 + amp * g
    return z2


# =============================================================================
# FRACTAL BROWNIAN MOTION (fBm) - Gerçekçi Arazi Üretimi
# =============================================================================

def _fbm_noise(
    *,
    rng: np.random.Generator,
    rows: int,
    cols: int,
    dx: float,
    dy: float,
    octaves: int = 6,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    base_wavelength_m: float = 500.0,
) -> np.ndarray:
    """Fractal Brownian Motion (fBm) noise - doğal arazi için temel.

    fBm, farklı frekanslarda (oktavlarda) noise katmanlarını birleştirir.
    Doğadaki arazi, dağlar, bulutlar gibi yapıları taklit eder.

    Args:
        rng: Rastgele sayı üreteci
        rows, cols: Çıktı boyutu
        dx, dy: Piksel boyutu (metre)
        octaves: Oktav sayısı (detay seviyesi, 1-8)
        persistence: Her oktavda genlik azalması (0.3-0.7)
        lacunarity: Frekans artış oranı (genellikle 2.0)
        base_wavelength_m: Temel dalga boyu (metre)

    Returns:
        Normalize edilmiş fBm noise dizisi
    """
    try:
        from scipy.ndimage import gaussian_filter
    except Exception as e:
        raise RuntimeError("scipy is required for fBm noise generation") from e

    px = 0.5 * (dx + dy)  # Ortalama piksel boyutu
    acc = np.zeros((rows, cols), dtype=np.float64)
    amplitude = 1.0
    total_amplitude = 0.0

    for i in range(octaves):
        # Bu oktav için dalga boyu ve sigma
        wavelength = base_wavelength_m / (lacunarity ** i)
        sigma_px = wavelength / px

        # Minimum sigma kontrolü
        if sigma_px < 0.5:
            break

        # Beyaz gürültü üret ve yumuşat
        white = rng.standard_normal((rows, cols)).astype(np.float64, copy=False)
        smoothed = gaussian_filter(white, sigma=sigma_px, mode="wrap")

        # Normalize et
        std = float(smoothed.std(dtype=np.float64))
        if std > 0:
            smoothed = smoothed / std

        acc += amplitude * smoothed
        total_amplitude += amplitude
        amplitude *= persistence

    # Toplam genliğe göre normalize et
    if total_amplitude > 0:
        acc = acc / total_amplitude

    return acc


def _ridge_noise(
    *,
    rng: np.random.Generator,
    rows: int,
    cols: int,
    dx: float,
    dy: float,
    octaves: int = 5,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    base_wavelength_m: float = 400.0,
    ridge_sharpness: float = 2.0,
) -> np.ndarray:
    """Ridge (sırt) noise - dağ sırtları ve keskin tepeler için.

    Standart fBm'in mutlak değerini alıp ters çevirerek
    keskin sırtlar oluşturur.

    Args:
        ridge_sharpness: Sırt keskinliği (1.0-4.0)
    """
    # Temel fBm üret
    fbm = _fbm_noise(
        rng=rng,
        rows=rows,
        cols=cols,
        dx=dx,
        dy=dy,
        octaves=octaves,
        persistence=persistence,
        lacunarity=lacunarity,
        base_wavelength_m=base_wavelength_m,
    )

    # Ridge dönüşümü: 1 - |noise| ^ sharpness
    ridge = 1.0 - np.abs(fbm) ** (1.0 / ridge_sharpness)

    return ridge


def _turbulence_noise(
    *,
    rng: np.random.Generator,
    rows: int,
    cols: int,
    dx: float,
    dy: float,
    octaves: int = 6,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    base_wavelength_m: float = 300.0,
) -> np.ndarray:
    """Turbulence noise - mutlak değerli fBm (daha kaotik)."""
    try:
        from scipy.ndimage import gaussian_filter
    except Exception as e:
        raise RuntimeError("scipy is required") from e

    px = 0.5 * (dx + dy)
    acc = np.zeros((rows, cols), dtype=np.float64)
    amplitude = 1.0
    total_amplitude = 0.0

    for i in range(octaves):
        wavelength = base_wavelength_m / (lacunarity ** i)
        sigma_px = wavelength / px

        if sigma_px < 0.5:
            break

        white = rng.standard_normal((rows, cols)).astype(np.float64, copy=False)
        smoothed = gaussian_filter(white, sigma=sigma_px, mode="wrap")

        std = float(smoothed.std(dtype=np.float64))
        if std > 0:
            smoothed = smoothed / std

        # Mutlak değer al (turbulence)
        acc += amplitude * np.abs(smoothed)
        total_amplitude += amplitude
        amplitude *= persistence

    if total_amplitude > 0:
        acc = acc / total_amplitude

    return acc


# =============================================================================
# EROZYON SİMÜLASYONU
# =============================================================================

def _simple_hydraulic_erosion(
    z: np.ndarray,
    *,
    iterations: int = 50,
    erosion_rate: float = 0.05,
    deposition_rate: float = 0.03,
    evaporation_rate: float = 0.02,
) -> np.ndarray:
    """Basitleştirilmiş hidrolik erozyon simülasyonu.

    Su akışını simüle ederek doğal vadi ve akarsu yatakları oluşturur.
    """
    try:
        from scipy.ndimage import gaussian_filter
    except Exception as e:
        raise RuntimeError("scipy is required for erosion simulation") from e

    z_eroded = z.copy()
    rows, cols = z.shape

    for _ in range(iterations):
        # Gradyan hesapla (su akış yönü)
        gy, gx = np.gradient(z_eroded)
        gradient_mag = np.sqrt(gx**2 + gy**2)

        # Su birikimi simülasyonu (yumuşatılmış gradyan)
        water = gaussian_filter(gradient_mag, sigma=3.0)

        # Erozyon: dik yamaçlarda daha fazla
        erosion = erosion_rate * water * gradient_mag
        z_eroded -= erosion

        # Tortu birikimi: düz alanlarda
        flatness = np.exp(-gradient_mag * 10)
        deposition = deposition_rate * water * flatness
        z_eroded += deposition

        # Yumuşatma (doğal difüzyon)
        z_eroded = gaussian_filter(z_eroded, sigma=0.5)

    return z_eroded


def _thermal_erosion(
    z: np.ndarray,
    *,
    iterations: int = 30,
    talus_angle: float = 0.5,  # radyan cinsinden maksimum eğim
    erosion_amount: float = 0.1,
) -> np.ndarray:
    """Termal erozyon - aşırı dik yamaçları yumuşatır."""
    z_eroded = z.copy()

    for _ in range(iterations):
        # 4 yönde gradyan
        d_north = np.roll(z_eroded, -1, axis=0) - z_eroded
        d_south = np.roll(z_eroded, 1, axis=0) - z_eroded
        d_east = np.roll(z_eroded, -1, axis=1) - z_eroded
        d_west = np.roll(z_eroded, 1, axis=1) - z_eroded

        # Talus açısını aşan yerlerde malzeme transferi
        for d, roll_axis, roll_dir in [
            (d_north, 0, -1),
            (d_south, 0, 1),
            (d_east, 1, -1),
            (d_west, 1, 1),
        ]:
            mask = d < -talus_angle
            transfer = np.where(mask, erosion_amount * (np.abs(d) - talus_angle), 0.0)
            z_eroded -= transfer
            z_eroded += np.roll(transfer, roll_dir, axis=roll_axis)

    return z_eroded


# =============================================================================
# GERÇEKÇİ ARAZİ ÜRETİCİLERİ
# =============================================================================

def _generate_mountain(
    x: np.ndarray,
    y: np.ndarray,
    grid: SyntheticGrid,
    *,
    rng: np.random.Generator,
    relief: float,
    base_elevation: float = 800.0,
) -> np.ndarray:
    """Dağlık arazi üretir.

    Özellikleri:
    - fBm tabanlı ana topografi
    - Ridge noise ile keskin sırtlar
    - Zirve noktaları
    - Erozyon ile doğal vadiler
    """
    rows, cols = x.shape
    dx, dy = grid.dx, grid.dy

    # Ana fBm topografisi
    fbm = _fbm_noise(
        rng=rng,
        rows=rows,
        cols=cols,
        dx=dx,
        dy=dy,
        octaves=7,
        persistence=0.55,
        lacunarity=2.1,
        base_wavelength_m=grid.width * 0.4,
    )

    # Ridge noise (sırtlar)
    ridges = _ridge_noise(
        rng=np.random.default_rng(rng.integers(0, 2**31)),
        rows=rows,
        cols=cols,
        dx=dx,
        dy=dy,
        octaves=5,
        persistence=0.5,
        base_wavelength_m=grid.width * 0.25,
        ridge_sharpness=2.5,
    )

    # Birleştir
    z = base_elevation + relief * (
        80.0 * fbm +          # Ana topografi
        40.0 * ridges +       # Sırtlar
        15.0 * fbm * ridges   # Etkileşim
    )

    # Birkaç ana zirve ekle
    n_peaks = max(2, int(grid.width * grid.height / 1e7))
    for _ in range(n_peaks):
        px = float(rng.uniform(0.1 * grid.width, 0.9 * grid.width))
        py = float(rng.uniform(0.1 * grid.height, 0.9 * grid.height))
        peak_h = float(rng.uniform(30.0, 80.0)) * relief
        sigma_x = float(rng.uniform(grid.width * 0.05, grid.width * 0.15))
        sigma_y = float(rng.uniform(grid.height * 0.05, grid.height * 0.15))
        z += peak_h * np.exp(-0.5 * (((x - px) / sigma_x) ** 2 + ((y - py) / sigma_y) ** 2))

    # Hafif erozyon uygula
    z = _simple_hydraulic_erosion(z, iterations=20, erosion_rate=0.03)

    return z


def _generate_valley(
    x: np.ndarray,
    y: np.ndarray,
    grid: SyntheticGrid,
    *,
    rng: np.random.Generator,
    relief: float,
    base_elevation: float = 200.0,
) -> np.ndarray:
    """Vadi ve akarsu yatağı üretir.

    Özellikleri:
    - V veya U şekilli ana vadi
    - Yan vadiler
    - Kıvrımlı akarsu yatağı
    - Taşkın ovası
    """
    rows, cols = x.shape
    dx, dy = grid.dx, grid.dy

    # Ana vadi ekseni (hafif kıvrımlı)
    valley_center_y = grid.height * 0.5
    valley_amplitude = grid.height * 0.15
    valley_wavelength = grid.width * 0.4

    # Kıvrımlı vadi merkezi
    valley_offset = valley_amplitude * np.sin(2.0 * math.pi * x / valley_wavelength)
    dist_from_valley = np.abs(y - valley_center_y - valley_offset)

    # Vadi kesiti (V-şekilli)
    valley_width = grid.height * 0.3
    valley_depth = 60.0 * relief
    v_profile = valley_depth * (dist_from_valley / valley_width)
    v_profile = np.clip(v_profile, 0.0, valley_depth)

    # Tepe topografisi (vadi kenarları)
    fbm = _fbm_noise(
        rng=rng,
        rows=rows,
        cols=cols,
        dx=dx,
        dy=dy,
        octaves=6,
        persistence=0.5,
        base_wavelength_m=grid.width * 0.3,
    )

    # Yükseklik: kenarlar yüksek, vadi tabanı düşük
    z = base_elevation + v_profile + relief * 25.0 * fbm

    # Akarsu yatağı (düz alan)
    river_width = grid.height * 0.03
    river_mask = dist_from_valley < river_width
    z = np.where(river_mask, base_elevation - 5.0 * relief, z)

    # Taşkın ovası (vadi tabanı)
    floodplain_width = grid.height * 0.08
    floodplain_mask = dist_from_valley < floodplain_width
    floodplain_noise = _fbm_noise(
        rng=np.random.default_rng(rng.integers(0, 2**31)),
        rows=rows,
        cols=cols,
        dx=dx,
        dy=dy,
        octaves=4,
        persistence=0.4,
        base_wavelength_m=grid.width * 0.1,
    )
    z = np.where(
        floodplain_mask & ~river_mask,
        base_elevation + relief * 3.0 * floodplain_noise,
        z
    )

    # Erozyon
    z = _simple_hydraulic_erosion(z, iterations=30, erosion_rate=0.04)

    return z


def _generate_hills(
    x: np.ndarray,
    y: np.ndarray,
    grid: SyntheticGrid,
    *,
    rng: np.random.Generator,
    relief: float,
    base_elevation: float = 150.0,
) -> np.ndarray:
    """Yumuşak tepeler (rolling hills) üretir.

    Özellikleri:
    - Düşük frekanslı fBm (geniş tepeler)
    - Yumuşak geçişler
    - Çayırlar için uygun topografi
    """
    rows, cols = x.shape
    dx, dy = grid.dx, grid.dy

    # Düşük frekanslı, yumuşak fBm
    fbm = _fbm_noise(
        rng=rng,
        rows=rows,
        cols=cols,
        dx=dx,
        dy=dy,
        octaves=5,
        persistence=0.4,  # Düşük persistence = daha yumuşak
        lacunarity=1.8,
        base_wavelength_m=grid.width * 0.5,  # Geniş dalgalar
    )

    # İkinci katman (orta ölçek)
    fbm2 = _fbm_noise(
        rng=np.random.default_rng(rng.integers(0, 2**31)),
        rows=rows,
        cols=cols,
        dx=dx,
        dy=dy,
        octaves=4,
        persistence=0.35,
        lacunarity=2.0,
        base_wavelength_m=grid.width * 0.2,
    )

    # Yumuşak eğim (genel yön)
    slope = 0.005 * (x - 0.5 * grid.width) + 0.003 * (y - 0.5 * grid.height)

    z = base_elevation + slope + relief * (
        35.0 * fbm +    # Ana tepeler
        12.0 * fbm2     # Detay
    )

    # Çok hafif termal erozyon (yumuşatma)
    z = _thermal_erosion(z, iterations=10, talus_angle=0.6, erosion_amount=0.05)

    return z


def _generate_coastal(
    x: np.ndarray,
    y: np.ndarray,
    grid: SyntheticGrid,
    *,
    rng: np.random.Generator,
    relief: float,
    sea_level: float = 0.0,
) -> np.ndarray:
    """Kıyı şeridi üretir.

    Özellikleri:
    - Deniz-kara geçişi
    - Kumsal / falezler
    - Kıyı çizgisi düzensizliği
    - Arkadaki yükseklik
    """
    rows, cols = x.shape
    dx, dy = grid.dx, grid.dy

    # Kıyı çizgisi (düzensiz)
    coastline_base = grid.width * 0.3
    coastline_noise = _fbm_noise(
        rng=rng,
        rows=rows,
        cols=cols,
        dx=dx,
        dy=dy,
        octaves=4,
        persistence=0.5,
        base_wavelength_m=grid.height * 0.3,
    )

    # Kıyı çizgisi x konumu (y'ye bağlı değişim)
    coastline_variation = grid.width * 0.1 * coastline_noise
    coastline_x = coastline_base + coastline_variation

    # Denizden uzaklık
    dist_from_coast = x - coastline_x

    # Kara topografisi
    land_fbm = _fbm_noise(
        rng=np.random.default_rng(rng.integers(0, 2**31)),
        rows=rows,
        cols=cols,
        dx=dx,
        dy=dy,
        octaves=6,
        persistence=0.5,
        base_wavelength_m=grid.width * 0.25,
    )

    # Yükseklik profili
    # Deniz: sabit seviye
    # Kıyı: hızlı yükseliş
    # İç kısım: tepeler
    coastal_slope = 0.1 * relief  # Kıyı eğimi
    inland_base = 30.0 * relief

    z = np.where(
        dist_from_coast < 0,
        sea_level - 5.0,  # Deniz tabanı
        sea_level + np.minimum(dist_from_coast * coastal_slope, inland_base) +
        relief * 20.0 * land_fbm * _smoothstep01(dist_from_coast / (grid.width * 0.2))
    )

    # Plaj/kumsal (dar şerit)
    beach_width = grid.width * 0.02
    beach_mask = (dist_from_coast >= 0) & (dist_from_coast < beach_width)
    z = np.where(beach_mask, sea_level + 2.0 + relief * 0.5 * np.abs(coastline_noise), z)

    return z


def _generate_plateau(
    x: np.ndarray,
    y: np.ndarray,
    grid: SyntheticGrid,
    *,
    rng: np.random.Generator,
    relief: float,
    base_elevation: float = 500.0,
) -> np.ndarray:
    """Yüksek plato üretir.

    Özellikleri:
    - Düz üst yüzey
    - Dik yamaçlar (escarpment)
    - Plato kenarında vadiler
    """
    rows, cols = x.shape
    dx, dy = grid.dx, grid.dy

    # Plato sınırı (düzensiz)
    boundary_noise = _fbm_noise(
        rng=rng,
        rows=rows,
        cols=cols,
        dx=dx,
        dy=dy,
        octaves=4,
        persistence=0.5,
        base_wavelength_m=grid.width * 0.2,
    )

    # Merkeze uzaklık
    cx, cy = grid.width * 0.5, grid.height * 0.5
    dist_from_center = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    plateau_radius = min(grid.width, grid.height) * 0.35

    # Düzensiz sınır
    effective_dist = dist_from_center - boundary_noise * grid.width * 0.05

    # Plato profili (sigmoid)
    plateau_edge_width = grid.width * 0.05
    plateau_mask = 1.0 - _quintic_smoothstep(
        (effective_dist - plateau_radius) / plateau_edge_width
    )

    # Plato üst yüzeyi (hafif dalgalı)
    top_surface = _fbm_noise(
        rng=np.random.default_rng(rng.integers(0, 2**31)),
        rows=rows,
        cols=cols,
        dx=dx,
        dy=dy,
        octaves=4,
        persistence=0.3,
        base_wavelength_m=grid.width * 0.3,
    )

    # Alt seviye (plato dışı)
    lower_surface = _fbm_noise(
        rng=np.random.default_rng(rng.integers(0, 2**31)),
        rows=rows,
        cols=cols,
        dx=dx,
        dy=dy,
        octaves=5,
        persistence=0.5,
        base_wavelength_m=grid.width * 0.2,
    )

    plateau_height = 80.0 * relief
    z = (
        base_elevation +
        plateau_mask * (plateau_height + relief * 8.0 * top_surface) +
        (1.0 - plateau_mask) * relief * 15.0 * lower_surface
    )

    # Yamaçlarda erozyon
    z = _simple_hydraulic_erosion(z, iterations=15, erosion_rate=0.02)

    return z


def _generate_canyon(
    x: np.ndarray,
    y: np.ndarray,
    grid: SyntheticGrid,
    *,
    rng: np.random.Generator,
    relief: float,
    base_elevation: float = 400.0,
) -> np.ndarray:
    """Kanyon/boğaz üretir.

    Özellikleri:
    - Derin, dar vadi
    - Dik yamaçlar
    - Kıvrımlı rota
    - Tabakalı duvarlar
    """
    rows, cols = x.shape
    dx, dy = grid.dx, grid.dy

    # Ana plato yüzeyi
    plateau = _fbm_noise(
        rng=rng,
        rows=rows,
        cols=cols,
        dx=dx,
        dy=dy,
        octaves=5,
        persistence=0.45,
        base_wavelength_m=grid.width * 0.4,
    )

    z = base_elevation + relief * 15.0 * plateau

    # Kanyon rotası (kıvrımlı)
    canyon_center_y = grid.height * 0.5
    canyon_wavelength = grid.width * 0.5
    num_meanders = 3

    for i in range(num_meanders):
        phase = float(rng.uniform(0, 2 * math.pi))
        amp = grid.height * float(rng.uniform(0.1, 0.2))
        wl = canyon_wavelength / (i + 1)
        canyon_center_y = canyon_center_y + amp * np.sin(2.0 * math.pi * x / wl + phase)

    # Kanyondan uzaklık
    dist = np.abs(y - canyon_center_y)

    # Kanyon kesiti (V şekilli ama dik kenarlı)
    canyon_width = grid.height * 0.08
    canyon_depth = 100.0 * relief

    # Dik kenar profili
    canyon_profile = canyon_depth * (1.0 - _quintic_smoothstep(dist / canyon_width))

    # Tabaka efekti (basamaklı duvarlar)
    layer_height = 15.0 * relief
    num_layers = int(canyon_depth / layer_height)
    for i in range(num_layers):
        layer_dist = canyon_width * (1.0 - i / num_layers)
        layer_mask = (dist > layer_dist * 0.95) & (dist < layer_dist * 1.05)
        canyon_profile = np.where(layer_mask, canyon_profile - 3.0 * relief, canyon_profile)

    z = z - canyon_profile

    # Kanyon tabanı (düz ve dar)
    canyon_floor_width = canyon_width * 0.15
    floor_mask = dist < canyon_floor_width
    floor_noise = _fbm_noise(
        rng=np.random.default_rng(rng.integers(0, 2**31)),
        rows=rows,
        cols=cols,
        dx=dx,
        dy=dy,
        octaves=3,
        persistence=0.4,
        base_wavelength_m=grid.width * 0.1,
    )
    z = np.where(floor_mask, base_elevation - canyon_depth + relief * 2.0 * floor_noise, z)

    return z


def _generate_volcanic(
    x: np.ndarray,
    y: np.ndarray,
    grid: SyntheticGrid,
    *,
    rng: np.random.Generator,
    relief: float,
    base_elevation: float = 300.0,
) -> np.ndarray:
    """Volkanik arazi üretir.

    Özellikleri:
    - Ana volkan konisi
    - Krater
    - Yan koniler
    - Lav akış kanalları
    """
    rows, cols = x.shape
    dx, dy = grid.dx, grid.dy

    # Ana volkan merkezi
    volcano_x = grid.width * float(rng.uniform(0.4, 0.6))
    volcano_y = grid.height * float(rng.uniform(0.4, 0.6))
    volcano_height = 150.0 * relief
    volcano_radius = min(grid.width, grid.height) * 0.35

    # Volkan konisi (konkav profil)
    dist = np.sqrt((x - volcano_x) ** 2 + (y - volcano_y) ** 2)
    cone = volcano_height * (1.0 - (dist / volcano_radius) ** 0.7)
    cone = np.maximum(cone, 0.0)

    # Krater
    crater_radius = volcano_radius * 0.15
    crater_depth = volcano_height * 0.3
    crater_mask = dist < crater_radius
    crater = np.where(
        crater_mask,
        -crater_depth * (1.0 - (dist / crater_radius) ** 2),
        0.0
    )

    z = base_elevation + cone + crater

    # Yan koniler
    n_parasitic = int(rng.integers(3, 7))
    for _ in range(n_parasitic):
        px = float(rng.uniform(0.2 * grid.width, 0.8 * grid.width))
        py = float(rng.uniform(0.2 * grid.height, 0.8 * grid.height))
        ph = float(rng.uniform(20.0, 50.0)) * relief
        pr = float(rng.uniform(0.03, 0.08)) * min(grid.width, grid.height)

        p_dist = np.sqrt((x - px) ** 2 + (y - py) ** 2)
        p_cone = ph * (1.0 - (p_dist / pr) ** 0.8)
        z += np.maximum(p_cone, 0.0)

    # Lav akış kanalları (radyal)
    n_flows = int(rng.integers(3, 6))
    for _ in range(n_flows):
        angle = float(rng.uniform(0, 2 * math.pi))
        flow_width = float(rng.uniform(0.02, 0.04)) * grid.width
        flow_length = float(rng.uniform(0.5, 0.9)) * volcano_radius

        # Akış merkez çizgisi
        flow_x = volcano_x + np.cos(angle) * dist
        flow_y = volcano_y + np.sin(angle) * dist
        flow_dist = np.abs(
            (x - volcano_x) * np.sin(angle) - (y - volcano_y) * np.cos(angle)
        )

        # Sadece volkan yamacında
        radial_mask = (dist > crater_radius) & (dist < flow_length)
        flow_mask = (flow_dist < flow_width) & radial_mask

        z = np.where(flow_mask, z - 5.0 * relief, z)

    # Pürüzlülük
    roughness = _fbm_noise(
        rng=np.random.default_rng(rng.integers(0, 2**31)),
        rows=rows,
        cols=cols,
        dx=dx,
        dy=dy,
        octaves=5,
        persistence=0.5,
        base_wavelength_m=grid.width * 0.1,
    )
    z += relief * 5.0 * roughness

    return z


def _generate_glacial(
    x: np.ndarray,
    y: np.ndarray,
    grid: SyntheticGrid,
    *,
    rng: np.random.Generator,
    relief: float,
    base_elevation: float = 400.0,
) -> np.ndarray:
    """Buzul vadisi (U-şekilli) üretir.

    Özellikleri:
    - U-şekilli ana vadi
    - Asılı vadiler
    - Sirk (cirque)
    - Morenler
    """
    rows, cols = x.shape
    dx, dy = grid.dx, grid.dy

    # Dağ topografisi
    mountain = _fbm_noise(
        rng=rng,
        rows=rows,
        cols=cols,
        dx=dx,
        dy=dy,
        octaves=6,
        persistence=0.55,
        base_wavelength_m=grid.width * 0.35,
    )

    z = base_elevation + relief * 60.0 * mountain + relief * 40.0

    # U-şekilli ana vadi
    valley_center_y = grid.height * 0.5
    valley_width = grid.height * 0.25
    valley_depth = 80.0 * relief

    dist = np.abs(y - valley_center_y)

    # U profili: düz taban + dik yamaçlar
    u_profile = valley_depth * _smoothstep01((dist - valley_width * 0.3) / (valley_width * 0.7))

    z = z - (valley_depth - u_profile)

    # Vadi tabanı düzleştir
    floor_width = valley_width * 0.3
    floor_mask = dist < floor_width
    floor_noise = _fbm_noise(
        rng=np.random.default_rng(rng.integers(0, 2**31)),
        rows=rows,
        cols=cols,
        dx=dx,
        dy=dy,
        octaves=3,
        persistence=0.3,
        base_wavelength_m=grid.width * 0.15,
    )
    z = np.where(floor_mask, base_elevation - valley_depth * 0.9 + relief * 3.0 * floor_noise, z)

    # Sirk (vadi başlangıcı)
    cirque_x = grid.width * 0.1
    cirque_radius = grid.width * 0.15
    cirque_dist = np.sqrt((x - cirque_x) ** 2 + (y - valley_center_y) ** 2)
    cirque_mask = cirque_dist < cirque_radius
    cirque_depth = valley_depth * 0.5
    cirque_profile = cirque_depth * (1.0 - (cirque_dist / cirque_radius) ** 2)
    z = np.where(cirque_mask, np.minimum(z, base_elevation - cirque_profile), z)

    # Yan morenler (vadi kenarı tepecikleri)
    for side in [-1, 1]:
        moraine_y = valley_center_y + side * valley_width * 0.9
        moraine_width = valley_width * 0.1
        moraine_dist = np.abs(y - moraine_y)
        moraine_mask = moraine_dist < moraine_width
        moraine_height = 10.0 * relief * (1.0 - moraine_dist / moraine_width)
        z = np.where(moraine_mask, z + moraine_height, z)

    return z


def _generate_karst(
    x: np.ndarray,
    y: np.ndarray,
    grid: SyntheticGrid,
    *,
    rng: np.random.Generator,
    relief: float,
    base_elevation: float = 250.0,
) -> np.ndarray:
    """Karstik arazi üretir.

    Özellikleri:
    - Düdenler (sinkholes)
    - Kokpit (yuvarlak çukurlar)
    - Hum'lar (koni tepeler)
    - Düzensiz yüzey
    """
    rows, cols = x.shape
    dx, dy = grid.dx, grid.dy

    # Temel topografi
    base = _fbm_noise(
        rng=rng,
        rows=rows,
        cols=cols,
        dx=dx,
        dy=dy,
        octaves=5,
        persistence=0.45,
        base_wavelength_m=grid.width * 0.3,
    )

    z = base_elevation + relief * 20.0 * base

    # Düdenler (sinkholes)
    n_sinkholes = max(10, int(grid.width * grid.height / 5e5))
    for _ in range(n_sinkholes):
        sx = float(rng.uniform(0.05 * grid.width, 0.95 * grid.width))
        sy = float(rng.uniform(0.05 * grid.height, 0.95 * grid.height))
        sr = float(rng.uniform(5.0, 30.0))  # Yarıçap (metre)
        sd = float(rng.uniform(5.0, 25.0)) * relief  # Derinlik

        s_dist = np.sqrt((x - sx) ** 2 + (y - sy) ** 2)
        sinkhole = sd * np.exp(-0.5 * (s_dist / sr) ** 2)
        z = z - sinkhole

    # Hum'lar (koni tepeler)
    n_hums = max(5, int(grid.width * grid.height / 1e6))
    for _ in range(n_hums):
        hx = float(rng.uniform(0.1 * grid.width, 0.9 * grid.width))
        hy = float(rng.uniform(0.1 * grid.height, 0.9 * grid.height))
        hr = float(rng.uniform(20.0, 60.0))  # Taban yarıçapı
        hh = float(rng.uniform(15.0, 40.0)) * relief  # Yükseklik

        h_dist = np.sqrt((x - hx) ** 2 + (y - hy) ** 2)
        hum = hh * np.maximum(0.0, 1.0 - h_dist / hr)
        z = z + hum

    # Yüzey pürüzlülüğü (çözünme dokusu)
    roughness = _turbulence_noise(
        rng=np.random.default_rng(rng.integers(0, 2**31)),
        rows=rows,
        cols=cols,
        dx=dx,
        dy=dy,
        octaves=5,
        persistence=0.6,
        base_wavelength_m=grid.width * 0.05,
    )
    z += relief * 5.0 * roughness

    return z


def _generate_alluvial(
    x: np.ndarray,
    y: np.ndarray,
    grid: SyntheticGrid,
    *,
    rng: np.random.Generator,
    relief: float,
    base_elevation: float = 50.0,
) -> np.ndarray:
    """Alüvyal ova/delta üretir.

    Özellikleri:
    - Düz genel topografi
    - Menderesli akarsu kanalları
    - Terkedilmiş kanallar (oxbow)
    - Hafif mikro-rölyef
    """
    rows, cols = x.shape
    dx, dy = grid.dx, grid.dy

    # Çok düz temel yüzey
    z = np.full((rows, cols), base_elevation, dtype=np.float64)

    # Hafif genel eğim
    slope = 0.0005 * relief * (grid.width - x)  # Batıya doğru alçalma
    z = z + slope

    # Mikro-rölyef
    micro = _fbm_noise(
        rng=rng,
        rows=rows,
        cols=cols,
        dx=dx,
        dy=dy,
        octaves=4,
        persistence=0.35,
        base_wavelength_m=grid.width * 0.15,
    )
    z = z + relief * 3.0 * micro

    # Ana akarsu kanalı (menderesli)
    channel_y = grid.height * 0.5
    channel_width = grid.height * 0.02
    channel_depth = 3.0 * relief

    # Menderes
    n_meanders = 5
    for i in range(n_meanders):
        amp = grid.height * float(rng.uniform(0.05, 0.15))
        wl = grid.width / (float(rng.uniform(1.5, 3.0)))
        phase = float(rng.uniform(0, 2 * math.pi))
        channel_y = channel_y + amp * np.sin(2.0 * math.pi * x / wl + phase)

    # Kanal kazısı
    channel_dist = np.abs(y - channel_y)
    channel_mask = channel_dist < channel_width
    z = np.where(channel_mask, z - channel_depth, z)

    # Doğal setler (kanal kenarı yükselmesi)
    levee_width = channel_width * 3
    levee_mask = (channel_dist >= channel_width) & (channel_dist < levee_width)
    levee_height = relief * 1.5 * (1.0 - (channel_dist - channel_width) / (levee_width - channel_width))
    z = np.where(levee_mask, z + levee_height, z)

    # Oxbow gölleri (terkedilmiş menderes)
    n_oxbows = int(rng.integers(2, 5))
    for _ in range(n_oxbows):
        ox = float(rng.uniform(0.2 * grid.width, 0.8 * grid.width))
        oy = float(rng.uniform(0.3 * grid.height, 0.7 * grid.height))
        o_radius = float(rng.uniform(0.02, 0.05)) * grid.height
        o_depth = relief * 2.0

        o_dist = np.sqrt((x - ox) ** 2 + (y - oy) ** 2)
        # Yarım ay şekli
        angle = np.arctan2(y - oy, x - ox)
        arc_mask = (o_dist < o_radius * 1.2) & (o_dist > o_radius * 0.8) & (np.cos(angle) > 0)
        z = np.where(arc_mask, z - o_depth, z)

    return z


# =============================================================================
# ANA ÜRETİCİ FONKSİYON
# =============================================================================

def generate_synthetic_dsm(
    *,
    rows: int,
    cols: int,
    dx: float,
    dy: float | None = None,
    preset: str = "mountain",
    seed: int = 0,
    relief: float = 1.0,
    roughness_m: float = 0.75,
    nodata_value: float | None = None,
    nodata_holes: int = 0,
    nodata_radius_m: float = 12.0,
) -> np.ndarray:
    """Sentetik DSM (float32) üretir.

    Args:
        rows, cols: Raster boyutu
        dx: X piksel boyutu (metre)
        dy: Y piksel boyutu (None ise dx kullanılır)
        preset: Arazi tipi (SYNTHETIC_PRESETS listesinden)
        seed: Rastgele sayı tohumu
        relief: Rölyef çarpanı (1.0 = normal)
        roughness_m: Mikro pürüzlülük genliği (metre)
        nodata_value: Nodata değeri
        nodata_holes: Nodata delik sayısı
        nodata_radius_m: Nodata delik yarıçapı

    Returns:
        float32 numpy dizisi (yükseklik değerleri metre cinsinden)

    Gerçekçi Preset'ler:
        mountain:  Dağlık arazi (fBm + sırtlar + zirveler)
        valley:    Vadi ve akarsu yatakları
        hills:     Yumuşak tepeler (rolling hills)
        coastal:   Kıyı şeridi (deniz-kara geçişi)
        plateau:   Yüksek plato
        canyon:    Kanyon/boğaz
        volcanic:  Volkanik arazi
        glacial:   Buzul vadisi (U-şekilli)
        karst:     Karstik arazi (düdenler)
        alluvial:  Alüvyal ova/delta

    Test Preset'leri:
        plane, waves, crater_field, terraced, patchwork, mixed
    """
    if rows <= 0 or cols <= 0:
        raise ValueError("rows/cols must be > 0")
    dx = float(dx)
    dy = float(dx if dy is None else dy)
    if dx <= 0 or dy <= 0:
        raise ValueError("dx/dy must be > 0")

    preset_n = preset.strip().lower()
    if preset_n not in set(SYNTHETIC_PRESETS):
        raise ValueError(f"Unknown synthetic preset: {preset!r}. Choices: {SYNTHETIC_PRESETS}")

    seed_i = int(seed)
    rng_main = np.random.default_rng(seed_i)
    rng_noise = np.random.default_rng(seed_i + 100)
    rng_holes = np.random.default_rng(seed_i + 200)

    grid = SyntheticGrid(rows=rows, cols=cols, dx=dx, dy=dy)
    x, y = grid_centers(grid.rows, grid.cols, grid.dx, grid.dy)

    # ==========================================================================
    # GERÇEKÇİ ARAZİ TİPLERİ
    # ==========================================================================
    if preset_n == "mountain":
        z = _generate_mountain(x, y, grid, rng=rng_main, relief=relief)
    elif preset_n == "valley":
        z = _generate_valley(x, y, grid, rng=rng_main, relief=relief)
    elif preset_n == "hills":
        z = _generate_hills(x, y, grid, rng=rng_main, relief=relief)
    elif preset_n == "coastal":
        z = _generate_coastal(x, y, grid, rng=rng_main, relief=relief)
    elif preset_n == "plateau":
        z = _generate_plateau(x, y, grid, rng=rng_main, relief=relief)
    elif preset_n == "canyon":
        z = _generate_canyon(x, y, grid, rng=rng_main, relief=relief)
    elif preset_n == "volcanic":
        z = _generate_volcanic(x, y, grid, rng=rng_main, relief=relief)
    elif preset_n == "glacial":
        z = _generate_glacial(x, y, grid, rng=rng_main, relief=relief)
    elif preset_n == "karst":
        z = _generate_karst(x, y, grid, rng=rng_main, relief=relief)
    elif preset_n == "alluvial":
        z = _generate_alluvial(x, y, grid, rng=rng_main, relief=relief)

    # ==========================================================================
    # TEST PATTERNLERİ (ESKİ)
    # ==========================================================================
    elif preset_n in _TEST_PRESETS:
        # Eski test pattern'leri için mevcut kodu kullan
        rng_craters = np.random.default_rng(seed_i + 1)
        rng_terr = np.random.default_rng(seed_i + 2)
        rng_mixed = np.random.default_rng(seed_i + 5)

        base = 100.0 + 0.02 * (x - 0.5 * grid.width) - 0.015 * (y - 0.5 * grid.height)

        def waves() -> np.ndarray:
            wl1 = 80.0
            wl2 = 35.0
            k1 = 2.0 * math.pi / wl1
            k2 = 2.0 * math.pi / wl2
            return base + (5.0 * relief) * (np.sin(k1 * x) * np.cos(0.7 * k1 * y) + 0.35 * np.sin(k2 * (x + 0.4 * y)))

        def crater_field() -> np.ndarray:
            z_cf = base.copy()
            z_cf = _add_gaussian_bumps(
                z_cf, x, y,
                rng=rng_craters,
                count=40,
                amp_range=(-18.0 * relief, 18.0 * relief),
                sigma_m_range=(10.0, 45.0),
            )
            theta = math.radians(25.0)
            xr = x * math.cos(theta) + y * math.sin(theta)
            z_cf += (3.0 * relief) * np.sin(2.0 * math.pi * xr / 120.0)
            return z_cf

        def terraced() -> np.ndarray:
            z_t = base + (0.0025 * relief) * ((x - 0.5 * grid.width) ** 2 - 0.6 * (y - 0.5 * grid.height) ** 2)
            step_h = 2.0
            zq = step_h * np.round(z_t / step_h)
            blend = _smoothstep01((x - 0.1 * grid.width) / (0.8 * grid.width))
            zt = (1.0 - blend) * z_t + blend * zq
            for _ in range(8):
                w = float(rng_terr.uniform(12.0, 40.0))
                h = float(rng_terr.uniform(12.0, 40.0))
                x0 = float(rng_terr.uniform(0.55 * grid.width, 0.95 * grid.width))
                y0 = float(rng_terr.uniform(0.55 * grid.height, 0.95 * grid.height))
                mask = (np.abs(x - x0) < 0.5 * w) & (np.abs(y - y0) < 0.5 * h)
                zt = np.where(mask, zt.mean(dtype=np.float64) + float(rng_terr.uniform(4.0, 12.0)), zt)
            return zt

        def plane_only() -> np.ndarray:
            return base

        if preset_n == "plane":
            z = plane_only()
        elif preset_n == "waves":
            z = waves()
        elif preset_n == "crater_field":
            z = crater_field()
        elif preset_n == "terraced":
            z = terraced()
        else:  # patchwork or mixed
            z_plane = plane_only()
            z_waves = waves()
            z_craters = crater_field()
            z_terr = terraced()

            blend_px = 12.0
            wx = _smoothstep01((x - 0.5 * grid.width) / (blend_px * dx) + 0.5)
            wy = _smoothstep01((y - 0.5 * grid.height) / (blend_px * dy) + 0.5)
            w_left = 1.0 - wx
            w_right = wx
            w_top = 1.0 - wy
            w_bottom = wy
            w_tl = w_left * w_top
            w_tr = w_right * w_top
            w_bl = w_left * w_bottom
            w_br = w_right * w_bottom
            z = w_tl * z_plane + w_tr * z_waves + w_bl * z_craters + w_br * z_terr

            if preset_n == "mixed":
                r2 = ((x - 0.35 * grid.width) / (0.28 * grid.width)) ** 2 + ((y - 0.4 * grid.height) / (0.22 * grid.height)) ** 2
                z += (10.0 * relief) * np.exp(-0.5 * r2)
                z = _add_gaussian_bumps(
                    z, x, y,
                    rng=rng_mixed,
                    count=20,
                    amp_range=(-10.0 * relief, 10.0 * relief),
                    sigma_m_range=(6.0, 22.0),
                )
    else:
        raise ValueError(f"Unhandled preset: {preset_n}")

    # ==========================================================================
    # MİKRO PÜRÜZLÜLÜK (tüm preset'ler için)
    # ==========================================================================
    if roughness_m > 0:
        px = 0.5 * (dx + dy)
        sigmas_px = [0.9 / px, 2.0 / px, 5.0 / px]
        amps = [1.0, 0.55, 0.25]
        z += float(roughness_m) * _fractal_gaussian_noise(
            rng=rng_noise, rows=rows, cols=cols, sigmas_px=sigmas_px, amps=amps
        )

    # ==========================================================================
    # NODATA DELİKLERİ
    # ==========================================================================
    if nodata_value is not None and int(nodata_holes) > 0:
        r0 = float(nodata_radius_m)
        if r0 <= 0:
            raise ValueError("nodata_radius_m must be > 0")
        holes_mask = np.zeros((rows, cols), dtype=bool)
        for _ in range(int(nodata_holes)):
            x0 = float(rng_holes.uniform(0.0, grid.width))
            y0 = float(rng_holes.uniform(0.0, grid.height))
            r = float(rng_holes.uniform(0.7 * r0, 1.4 * r0))
            holes_mask |= (x - x0) ** 2 + (y - y0) ** 2 <= r * r
        z = z.copy()
        z[holes_mask] = float(nodata_value)

    return z.astype(np.float32, copy=False)
