#!/usr/bin/env python3
"""
Sentetik DSM/DEM GeoTIFF Üretici
================================

Bu script, yüzey alanı hesaplama yöntemlerini test etmek ve karşılaştırmak için
sentetik sayısal yükseklik modeli (DSM - Digital Surface Model) verileri üretir.

KULLANIM AMACI:
--------------
- Yüzey alanı hesaplama algoritmalarının doğruluğunu test etme
- Farklı arazi tiplerinde performans karşılaştırma
- Benchmark testleri için kontrollü veri üretme
- Algoritma validasyonu için bilinen özelliklere sahip sentetik veri oluşturma

GERÇEKÇİ ARAZİ TİPLERİ (Yeni):
------------------------------
- mountain:     Dağlık arazi - fBm noise, sırtlar, zirveler, erozyon
- valley:       Vadi ve akarsu - V/U şekilli vadi, menderes, taşkın ovası
- hills:        Yumuşak tepeler - düşük frekanslı rolling hills
- coastal:      Kıyı şeridi - deniz-kara geçişi, plaj, falezler
- plateau:      Yüksek plato - düz üst yüzey, dik yamaçlar
- canyon:       Kanyon/boğaz - derin dar vadi, tabakalı duvarlar
- volcanic:     Volkanik arazi - koni, krater, lav akışları
- glacial:      Buzul vadisi - U-şekilli vadi, sirk, morenler
- karst:        Karstik arazi - düdenler, hum'lar, mağara çökmeleri
- alluvial:     Alüvyal ova - düz delta, menderesli kanallar

TEST PATTERNLERİ (Eski):
-----------------------
- plane:        Düz eğimli yüzey (basit doğrulama için)
- waves:        Sinüzoidal dalgalar (pürüzlü yüzey testi)
- crater_field: Krater/çukur alanları
- terraced:     Teraslı/basamaklı arazi
- patchwork:    Test tiplerinin karışımı
- mixed:        Patchwork + ekstra çeşitlilik

ÇIKTI DOSYASI:
-------------
- Format: GeoTIFF (Float32)
- Projeksiyon: Varsayılan EPSG:32636 (UTM Zone 36N)
- Nodata değeri: Varsayılan -9999

ÖRNEK KULLANIM:
--------------
    # Gerçekçi dağlık arazi (varsayılan)
    python generate_synthetic_tif.py

    # Vadi ve akarsu yatağı
    python generate_synthetic_tif.py --target valley --rows 5000 --cols 5000

    # Yüksek çözünürlüklü kıyı şeridi
    python generate_synthetic_tif.py --target coastal --rows 8000 --cols 8000 --dx 0.5

    # Volkanik arazi
    python generate_synthetic_tif.py --target volcanic --relief 1.5

    # Buzul vadisi
    python generate_synthetic_tif.py --target glacial --rows 6000 --cols 6000

PERFORMANS NOTLARI:
------------------
- 10000x10000 piksel ≈ 400 MB bellek kullanımı
- Gerçekçi preset'ler (mountain, valley, vb.) daha fazla işlem gücü gerektirir
- scipy.ndimage modülü gereklidir

Yazar: Surface Area Calculator Project
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rasterio

from surface_area.analytic_surfaces import (
    ContinuousSurfaceReference,
    build_analytic_surface,
    circular_hole_mask_for_grid,
    compute_continuous_surface_reference,
    generate_circular_holes,
    is_analytic_preset,
)
from surface_area.io import parse_resampling, resample_dem, safe_gsd_tag, write_dem_float32_geotiff
from surface_area.progress import ProgressPrinter
from surface_area.synthetic import (
    ANALYTIC_PRESETS,
    RASTER_FIRST_PRESETS,
    SYNTHETIC_PRESETS,
    SurfaceAreaResult,
    compute_reference_surface_area,
    generate_synthetic_dsm,
)
from surface_area.terrain_complexity import compute_complexity_descriptors, summarize_complexity_descriptors


# =============================================================================
# TEK NOKTADAN DÜZENLENEBİLEN VARSAYILAN CONFIG
# =============================================================================

TARGET_GROUP_CHOICES = ("raster_first", "analytic", "all")
TARGET_CHOICES = tuple(SYNTHETIC_PRESETS) + TARGET_GROUP_CHOICES

config: dict[str, object] = {
    # Çıktı GeoTIFF yolu.
    # Şablonlar: {preset}, {rows}, {cols}, {dx}, {dy}, {seed}, {timestamp}
    "out": "out_synth/synth_{preset}_{rows}x{cols}_dx{dx:g}_seed{seed}_{timestamp}.tif",
    # Tek seçim parametresi:
    # - Bir preset adı verirsen yalnızca o yüzey üretilir.
    #   Örnekler: mountain, valley, coastal, analytic_gaussian_hill
    # - Bir grup adı verirsen o gruba ait tüm yüzey tipleri üretilir.
    #   raster_first -> gerçekçi arazi + klasik raster-first test preset'leri
    #   analytic     -> sürekli ground truth üretebilen analitik benchmark preset'leri
    #   all          -> tanımlı tüm preset'ler
    # Bu alan, eski preset + benchmark_family + all_presets kombinasyonunun
    # yerine geçen tek giriş noktasıdır.
    "target": "all",
    # Raster boyutu
    "rows": 8192,
    "cols": 8192,
    # Piksel boyutu (metre)
    "dx": 0.05,
    "dy": None,
    # Analitik preset'lerde istenirse fiziksel extent doğrudan metre cinsinden verilebilir.
    # Bu durumda rows/cols, extent ve native dx/dy'den türetilir.
    "extent_width": None,
    "extent_height": None,
    # Rastgele tohum. None olursa her çalıştırmada farklı seed üretilir.
    "seed": 0,
    # Yüzey karakteri
    "relief": 1.0,
    "roughness_m": 0.75,
    # fBm/turbulence kullanan preset'lerde oktav bazlı worker sayısı (1 = kapalı).
    "fbm_workers": 8,
    # GeoTIFF coğrafi bilgileri
    "crs": "EPSG:32636",
    "origin_x": 500_000.0,
    "origin_y": 4_500_000.0,
    # Nodata ayarları
    "nodata": -9999.0,
    "nodata_holes": 0,
    "nodata_radius_m": 12.0,
    # Analitik benchmark'larda aynı fiziksel extent üzerinde ek değerlendirme GSD'leri üret.
    "eval_gsd": [],
    "resampling": "bilinear",
    # Sürekli yüzey alanı integrasyonu için ayarlar.
    "continuous_rel_tol": 1e-7,
    "continuous_abs_tol": 1e-6,
    "continuous_base_samples": 129,
    "continuous_max_levels": 5,
    # İsteğe bağlı yerel karmaşıklık özetleri / rasterları.
    "complexity": False,
    "write_complexity_rasters": False,
    "complexity_window": 5,
    # Referans olsun diye burada tutuluyor; parser bunu otomatik kullanıyor.
    "target_choices": list(TARGET_CHOICES),
}


# =============================================================================
# PARAMETRELER İÇİN SINIRLAR VE VARSAYILANLAR
# =============================================================================

# Satır/sütun limitleri
MIN_ROWS = 2
MIN_COLS = 2
MAX_ROWS = 100_000  # 100k satır (bellek sınırı)
MAX_COLS = 100_000  # 100k sütun (bellek sınırı)

# Piksel boyutu limitleri (metre)
MIN_PIXEL_SIZE = 0.001  # 1 mm
MAX_PIXEL_SIZE = 1000.0  # 1 km

# Relief ve roughness limitleri
MIN_RELIEF = 0.0
MAX_RELIEF = 1000.0
MIN_ROUGHNESS = 0.0
MAX_ROUGHNESS = 100.0
MIN_FBM_WORKERS = 1

# Nodata hole limitleri
MAX_NODATA_HOLES = 1000
MIN_NODATA_RADIUS = 0.1  # 10 cm
MAX_NODATA_RADIUS = 1000.0  # 1 km

# Bellek tahmini için float32 boyutu
BYTES_PER_PIXEL = 4  # float32

BENCHMARK_FAMILY_CHOICES = TARGET_GROUP_CHOICES


@dataclass(frozen=True, slots=True)
class ResolvedGridGeometry:
    rows: int
    cols: int
    dx: float
    dy: float
    extent_width_m: float
    extent_height_m: float
    mode: str


@dataclass(frozen=True, slots=True)
class ResolutionReferenceRecord:
    label: str
    tif_file: str
    dx: float
    dy: float
    is_native: bool
    resampling: str
    reference: SurfaceAreaResult


# =============================================================================
# YAPILANDIRMA SINIFI
# =============================================================================

@dataclass(frozen=True, slots=True)
class SynthConfig:
    """Sentetik DSM üretimi için yapılandırma parametreleri.

    Bu sınıf, script'i IDE'den doğrudan çalıştırırken kullanılacak
    varsayılan değerleri tanımlar. Varsayılanları tek yerden değiştirmek için
    dosya başındaki ``config`` sözlüğünü düzenleyin.

    Attributes:
        out: Çıktı GeoTIFF dosya yolu (şablon destekler: {preset}, {rows}, vb.)
        target: Tek seçim parametresi.
                Bir preset adı verilirse yalnızca o yüzey üretilir.
                Örnekler: mountain, valley, coastal, analytic_gaussian_hill
                Bir grup adı verilirse script o gruba ait tüm yüzeyleri üretir.
                raster_first = gerçekçi arazi + klasik raster-first test preset'leri
                analytic     = sürekli ground truth üreten tüm analitik benchmark'lar
                all          = tanımlı tüm preset'ler
                Böylece kullanıcı tek raster ile toplu benchmark modu arasında
                aynı parametre üzerinden geçiş yapabilir.
        rows: Raster satır sayısı
        cols: Raster sütun sayısı
        dx: X yönünde piksel boyutu (metre)
        dy: Y yönünde piksel boyutu (metre, None ise dx kullanılır)
        seed: Rastgele sayı üreteci tohumu (tekrarlanabilirlik için)
        relief: Makro rölyef çarpanı (0=düz, 1=normal, >1=abartılı)
        roughness_m: Mikro pürüzlülük genliği (metre)
        fbm_workers: fBm/turbulence kullanan preset'lerde oktav bazlı worker sayısı
        crs: Koordinat referans sistemi (örn: EPSG:32636)
        origin_x: Sol üst köşe X koordinatı
        origin_y: Sol üst köşe Y koordinatı
        nodata: Nodata değeri (None ile devre dışı)
        nodata_holes: Eklenecek dairesel nodata delik sayısı
        nodata_radius_m: Nodata delikleri için taban yarıçap (metre)
    """

    out: str = field(
        default=str(config["out"]),
        metadata={"help": "Çıktı GeoTIFF yolu ({preset}, {rows}, {cols}, {dx}, {seed}, {timestamp} şablonları desteklenir)"},
    )
    target: str = field(
        default=str(config["target"]),
        metadata={
            "help": (
                "Tek seçim parametresi. Bir preset adı verirsen yalnızca o yüzeyi üretir "
                "(örn: mountain, valley, analytic_gaussian_hill). "
                "Bir grup adı verirsen toplu üretim yapar: "
                "raster_first = gerçekçi + klasik raster-first preset'ler, "
                "analytic = tüm analitik benchmark'lar, "
                "all = tanımlı tüm preset'ler. "
                f"Geçerli seçenekler: {', '.join(TARGET_CHOICES)}"
            )
        },
    )
    rows: int = field(
        default=int(config["rows"]),
        metadata={"help": f"Raster satır sayısı ({MIN_ROWS}-{MAX_ROWS})"},
    )
    cols: int = field(
        default=int(config["cols"]),
        metadata={"help": f"Raster sütun sayısı ({MIN_COLS}-{MAX_COLS})"},
    )
    dx: float = field(
        default=float(config["dx"]),
        metadata={"help": f"X piksel boyutu metre ({MIN_PIXEL_SIZE}-{MAX_PIXEL_SIZE})"},
    )
    dy: float | None = field(
        default=config["dy"],
        metadata={"help": "Y piksel boyutu metre (None ise dx kullanılır)"},
    )
    extent_width: float | None = field(
        default=config["extent_width"],
        metadata={"help": "Analitik preset'lerde fiziksel extent genişliği metre (rows/cols yerine kullanılabilir)"},
    )
    extent_height: float | None = field(
        default=config["extent_height"],
        metadata={"help": "Analitik preset'lerde fiziksel extent yüksekliği metre (rows/cols yerine kullanılabilir)"},
    )
    seed: int | None = field(
        default=config["seed"],
        metadata={"help": "Rastgele sayı tohumu (sabit değer = tekrarlanabilir, None = her seferinde farklı)"},
    )
    relief: float = field(
        default=float(config["relief"]),
        metadata={"help": f"Makro rölyef çarpanı ({MIN_RELIEF}-{MAX_RELIEF})"},
    )
    roughness_m: float = field(
        default=float(config["roughness_m"]),
        metadata={"help": f"Mikro pürüzlülük genliği metre ({MIN_ROUGHNESS}-{MAX_ROUGHNESS})"},
    )
    fbm_workers: int = field(
        default=int(config["fbm_workers"]),
        metadata={"help": f"fBm/turbulence kullanan preset'lerde worker sayısı ({MIN_FBM_WORKERS}+; 1=kapalı)"},
    )
    crs: str = field(
        default=str(config["crs"]),
        metadata={"help": "CRS string (örn: EPSG:32636 = UTM Zone 36N)"},
    )
    origin_x: float = field(
        default=float(config["origin_x"]),
        metadata={"help": "Sol üst köşe X koordinatı (metre)"},
    )
    origin_y: float = field(
        default=float(config["origin_y"]),
        metadata={"help": "Sol üst köşe Y koordinatı (metre)"},
    )
    nodata: float | None = field(
        default=config["nodata"],
        metadata={"help": "Nodata değeri (None ile devre dışı bırakılır)"},
    )
    nodata_holes: int = field(
        default=int(config["nodata_holes"]),
        metadata={"help": f"Eklenecek dairesel nodata delik sayısı (0-{MAX_NODATA_HOLES})"},
    )
    nodata_radius_m: float = field(
        default=float(config["nodata_radius_m"]),
        metadata={"help": f"Nodata delikleri için taban yarıçap metre ({MIN_NODATA_RADIUS}-{MAX_NODATA_RADIUS})"},
    )
    eval_gsd: list[float] = field(
        default_factory=lambda: list(config["eval_gsd"]),
        metadata={"help": "Analitik benchmark için aynı extent üzerinde üretilecek ek değerlendirme GSD listesi"},
    )
    resampling: str = field(
        default=str(config["resampling"]),
        metadata={"help": "Analitik multi-resolution çıktılar için yeniden örnekleme yöntemi: bilinear | nearest | cubic"},
    )
    continuous_rel_tol: float = field(
        default=float(config["continuous_rel_tol"]),
        metadata={"help": "Sürekli yüzey alanı integrasyonu için göreli tolerans"},
    )
    continuous_abs_tol: float = field(
        default=float(config["continuous_abs_tol"]),
        metadata={"help": "Sürekli yüzey alanı integrasyonu için mutlak tolerans"},
    )
    continuous_base_samples: int = field(
        default=int(config["continuous_base_samples"]),
        metadata={"help": "Sürekli integrasyon için başlangıç örnekleme yoğunluğu (tek sayı tercih edilir)"},
    )
    continuous_max_levels: int = field(
        default=int(config["continuous_max_levels"]),
        metadata={"help": "Sürekli integrasyonda en fazla kaç adaptif rafine seviye deneneceği"},
    )
    complexity: bool = field(
        default=bool(config["complexity"]),
        metadata={"help": "Analitik native grid için yerel karmaşıklık özetlerini hesapla"},
    )
    write_complexity_rasters: bool = field(
        default=bool(config["write_complexity_rasters"]),
        metadata={"help": "Yerel karmaşıklık descriptor rasterlarını ayrı GeoTIFF olarak yaz"},
    )
    complexity_window: int = field(
        default=int(config["complexity_window"]),
        metadata={"help": "Karmaşıklık descriptorları için pencere boyutu (piksel)"},
    )


def _config_synth_kwargs(config_map: dict[str, object]) -> dict[str, object]:
    valid_keys = set(SynthConfig.__dataclass_fields__.keys())  # type: ignore[attr-defined]
    return {k: v for k, v in config_map.items() if k in valid_keys}


# IDE'den çalıştırırken kullanılacak varsayılan yapılandırma.
# Bu değerleri değiştirmek için dosya başındaki config sözlüğünü düzenleyin.
DEFAULT_SYNTH_CONFIG = SynthConfig(**_config_synth_kwargs(config))


# =============================================================================
# PARAMETRE DOĞRULAMA
# =============================================================================

class ValidationError(ValueError):
    """Parametre doğrulama hatası."""
    pass


def validate_parameters(
    *,
    rows: int,
    cols: int,
    dx: float,
    dy: float | None,
    preset: str,
    relief: float,
    roughness_m: float,
    fbm_workers: int,
    nodata_holes: int,
    nodata_radius_m: float,
    extent_width: float | None = None,
    extent_height: float | None = None,
    eval_gsd: list[float] | None = None,
    resampling: str = "bilinear",
    continuous_rel_tol: float = 1e-7,
    continuous_abs_tol: float = 1e-6,
    continuous_base_samples: int = 129,
    continuous_max_levels: int = 5,
    complexity_window: int = 5,
) -> list[str]:
    """Tüm parametreleri doğrular ve hata mesajlarını döndürür.

    Args:
        rows: Satır sayısı
        cols: Sütun sayısı
        dx: X piksel boyutu
        dy: Y piksel boyutu (None olabilir)
        preset: Arazi tipi
        relief: Rölyef çarpanı
        roughness_m: Pürüzlülük değeri
        nodata_holes: Nodata delik sayısı
        nodata_radius_m: Nodata yarıçapı

    Returns:
        Hata mesajları listesi. Boş liste = tüm parametreler geçerli.
    """
    errors: list[str] = []
    eval_gsd_values = list(eval_gsd or [])
    analytic = is_analytic_preset(str(preset))

    # Satır/sütun kontrolü
    if not isinstance(rows, int) or rows < MIN_ROWS:
        errors.append(f"rows en az {MIN_ROWS} olmalı, verilen: {rows}")
    elif rows > MAX_ROWS:
        errors.append(f"rows en fazla {MAX_ROWS} olabilir, verilen: {rows}")

    if not isinstance(cols, int) or cols < MIN_COLS:
        errors.append(f"cols en az {MIN_COLS} olmalı, verilen: {cols}")
    elif cols > MAX_COLS:
        errors.append(f"cols en fazla {MAX_COLS} olabilir, verilen: {cols}")

    # Piksel boyutu kontrolü
    if dx <= 0:
        errors.append(f"dx pozitif olmalı, verilen: {dx}")
    elif dx < MIN_PIXEL_SIZE:
        errors.append(f"dx en az {MIN_PIXEL_SIZE} olmalı, verilen: {dx}")
    elif dx > MAX_PIXEL_SIZE:
        errors.append(f"dx en fazla {MAX_PIXEL_SIZE} olabilir, verilen: {dx}")

    if dy is not None:
        if dy <= 0:
            errors.append(f"dy pozitif olmalı, verilen: {dy}")
        elif dy < MIN_PIXEL_SIZE:
            errors.append(f"dy en az {MIN_PIXEL_SIZE} olmalı, verilen: {dy}")
        elif dy > MAX_PIXEL_SIZE:
            errors.append(f"dy en fazla {MAX_PIXEL_SIZE} olabilir, verilen: {dy}")

    # Preset kontrolü
    if preset not in SYNTHETIC_PRESETS:
        errors.append(f"Geçersiz preset: '{preset}'. Geçerli seçenekler: {', '.join(SYNTHETIC_PRESETS)}")
    # Relief kontrolü
    if relief < MIN_RELIEF:
        errors.append(f"relief en az {MIN_RELIEF} olmalı, verilen: {relief}")
    elif relief > MAX_RELIEF:
        errors.append(f"relief en fazla {MAX_RELIEF} olabilir, verilen: {relief}")

    # Roughness kontrolü
    if roughness_m < MIN_ROUGHNESS:
        errors.append(f"roughness_m en az {MIN_ROUGHNESS} olmalı, verilen: {roughness_m}")
    elif roughness_m > MAX_ROUGHNESS:
        errors.append(f"roughness_m en fazla {MAX_ROUGHNESS} olabilir, verilen: {roughness_m}")

    if int(fbm_workers) < MIN_FBM_WORKERS:
        errors.append(f"fbm_workers en az {MIN_FBM_WORKERS} olmalı, verilen: {fbm_workers}")

    # Nodata holes kontrolü
    if nodata_holes < 0:
        errors.append(f"nodata_holes negatif olamaz, verilen: {nodata_holes}")
    elif nodata_holes > MAX_NODATA_HOLES:
        errors.append(f"nodata_holes en fazla {MAX_NODATA_HOLES} olabilir, verilen: {nodata_holes}")

    # Nodata radius kontrolü (sadece holes > 0 ise)
    if nodata_holes > 0:
        if nodata_radius_m < MIN_NODATA_RADIUS:
            errors.append(f"nodata_radius_m en az {MIN_NODATA_RADIUS} olmalı, verilen: {nodata_radius_m}")
        elif nodata_radius_m > MAX_NODATA_RADIUS:
            errors.append(f"nodata_radius_m en fazla {MAX_NODATA_RADIUS} olabilir, verilen: {nodata_radius_m}")

    if analytic:
        if (extent_width is None) != (extent_height is None):
            errors.append("Analitik preset'lerde extent_width ve extent_height birlikte verilmelidir.")
        if extent_width is not None and extent_width <= 0:
            errors.append(f"extent_width pozitif olmalı, verilen: {extent_width}")
        if extent_height is not None and extent_height <= 0:
            errors.append(f"extent_height pozitif olmalı, verilen: {extent_height}")
        native_min_gsd = min(float(dx), float(dx if dy is None else dy))
        for value in eval_gsd_values:
            if float(value) <= 0:
                errors.append(f"eval_gsd değerleri pozitif olmalı, verilen: {value}")
            elif float(value) + 1e-12 < native_min_gsd:
                errors.append(
                    f"Analitik eval_gsd native çözünürlükten daha ince olamaz. native={native_min_gsd:g}, verilen={value}"
                )
    else:
        if extent_width is not None or extent_height is not None:
            errors.append("extent_width/extent_height sadece analitik preset'lerde kullanılabilir.")
        if eval_gsd_values:
            errors.append("eval_gsd şu anda sadece analitik preset'lerde desteklenir.")

    if resampling not in {"nearest", "bilinear", "cubic"}:
        errors.append(f"resampling geçersiz: '{resampling}'. Seçenekler: nearest, bilinear, cubic")
    if continuous_rel_tol < 0:
        errors.append(f"continuous_rel_tol >= 0 olmalı, verilen: {continuous_rel_tol}")
    if continuous_abs_tol < 0:
        errors.append(f"continuous_abs_tol >= 0 olmalı, verilen: {continuous_abs_tol}")
    if int(continuous_base_samples) < 5:
        errors.append(f"continuous_base_samples en az 5 olmalı, verilen: {continuous_base_samples}")
    if int(continuous_max_levels) < 0:
        errors.append(f"continuous_max_levels negatif olamaz, verilen: {continuous_max_levels}")
    if int(complexity_window) < 3:
        errors.append(f"complexity_window en az 3 olmalı, verilen: {complexity_window}")

    # Bellek uyarısı (hata değil, uyarı)
    estimated_memory_mb = (rows * cols * BYTES_PER_PIXEL * 3) / (1024 * 1024)  # ~3x for processing
    if estimated_memory_mb > 4000:  # 4 GB
        # Bu bir uyarı, hata değil - errors listesine eklenmez
        pass

    return errors


def estimate_memory_usage(rows: int, cols: int) -> tuple[float, str]:
    """Tahmini bellek kullanımını hesaplar.

    Args:
        rows: Satır sayısı
        cols: Sütun sayısı

    Returns:
        (bellek_mb, formatlanmış_string) tuple'ı
    """
    # Ana dizi + işleme sırasında geçici diziler
    base_memory = rows * cols * BYTES_PER_PIXEL
    processing_overhead = base_memory * 2.5  # Gaussian filter vb. için
    total_bytes = base_memory + processing_overhead

    mb = total_bytes / (1024 * 1024)
    if mb < 1024:
        return mb, f"{mb:.1f} MB"
    else:
        gb = mb / 1024
        return mb, f"{gb:.2f} GB"


def estimate_file_size(rows: int, cols: int) -> tuple[float, str]:
    """Tahmini dosya boyutunu hesaplar.

    Args:
        rows: Satır sayısı
        cols: Sütun sayısı

    Returns:
        (boyut_mb, formatlanmış_string) tuple'ı
    """
    # GeoTIFF float32, sıkıştırmasız
    size_bytes = rows * cols * BYTES_PER_PIXEL
    mb = size_bytes / (1024 * 1024)
    if mb < 1024:
        return mb, f"{mb:.1f} MB"
    else:
        gb = mb / 1024
        return mb, f"{gb:.2f} GB"


def _normalize_eval_gsd_values(values: list[float] | None) -> list[float]:
    normalized: list[float] = []
    seen: set[float] = set()
    for value in values or []:
        vv = float(value)
        key = round(vv, 12)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(vv)
    return normalized


def _resolve_grid_geometry(args: argparse.Namespace) -> ResolvedGridGeometry:
    dx = float(args.dx)
    dy = float(args.dx if args.dy is None else args.dy)

    if is_analytic_preset(str(args.preset)) and args.extent_width is not None and args.extent_height is not None:
        extent_width = float(args.extent_width)
        extent_height = float(args.extent_height)
        cols_f = extent_width / dx
        rows_f = extent_height / dy
        cols = int(round(cols_f))
        rows = int(round(rows_f))
        if not math.isclose(cols * dx, extent_width, rel_tol=0.0, abs_tol=1e-9):
            raise ValidationError(f"extent_width={extent_width} değeri dx={dx} ile tam bölünmüyor.")
        if not math.isclose(rows * dy, extent_height, rel_tol=0.0, abs_tol=1e-9):
            raise ValidationError(f"extent_height={extent_height} değeri dy={dy} ile tam bölünmüyor.")
        return ResolvedGridGeometry(
            rows=rows,
            cols=cols,
            dx=dx,
            dy=dy,
            extent_width_m=extent_width,
            extent_height_m=extent_height,
            mode="extent_driven",
        )

    rows = int(args.rows)
    cols = int(args.cols)
    return ResolvedGridGeometry(
        rows=rows,
        cols=cols,
        dx=dx,
        dy=dy,
        extent_width_m=float(cols) * dx,
        extent_height_m=float(rows) * dy,
        mode="grid_driven",
    )


def _normalize_target_argument(args: argparse.Namespace) -> str:
    """Resolve the single public target from hidden legacy CLI arguments if needed."""
    legacy_preset = getattr(args, "legacy_preset", None)
    legacy_family = getattr(args, "legacy_benchmark_family", None)
    legacy_all = getattr(args, "legacy_all_presets", None)

    if legacy_all is True:
        return str(legacy_family or "all")
    if legacy_preset is not None:
        return str(legacy_preset)
    if legacy_family is not None:
        return str(legacy_family)
    return str(args.target)


def _resolve_target_presets(target: str) -> list[str]:
    if target == "raster_first":
        return list(RASTER_FIRST_PRESETS)
    if target == "analytic":
        return list(ANALYTIC_PRESETS)
    if target == "all":
        return list(SYNTHETIC_PRESETS)
    return [str(target)]


def _surface_area_result_to_dict(ref_area: SurfaceAreaResult) -> dict[str, float | int | None]:
    return {
        "planar_area_m2": ref_area.planar_area_m2,
        "planar_area_ha": ref_area.planar_area_ha,
        "planar_area_km2": ref_area.planar_area_km2,
        "surface_area_m2": ref_area.surface_area_m2,
        "surface_area_ha": ref_area.surface_area_ha,
        "surface_area_km2": ref_area.surface_area_km2,
        "surface_ratio": ref_area.surface_ratio,
        "increase_percent": (ref_area.surface_ratio - 1.0) * 100,
    }


def _continuous_reference_to_dict(ref_area: ContinuousSurfaceReference) -> dict[str, float | int]:
    return {
        "planar_area_m2": ref_area.planar_area_m2,
        "planar_area_ha": ref_area.planar_area_ha,
        "planar_area_km2": ref_area.planar_area_km2,
        "surface_area_m2": ref_area.surface_area_m2,
        "surface_area_ha": ref_area.surface_area_ha,
        "surface_area_km2": ref_area.surface_area_km2,
        "surface_ratio": ref_area.surface_ratio,
        "increase_percent": (ref_area.surface_ratio - 1.0) * 100,
        "integration_method": ref_area.integration_method,
        "samples_x": ref_area.samples_x,
        "samples_y": ref_area.samples_y,
        "levels": ref_area.levels,
        "rel_tol": ref_area.rel_tol,
        "abs_tol": ref_area.abs_tol,
        "masked_fraction": ref_area.masked_fraction,
    }


def _grid_info_from_reference(ref_area: SurfaceAreaResult) -> dict[str, float | int | None]:
    return {
        "rows": ref_area.rows,
        "cols": ref_area.cols,
        "dx": ref_area.dx,
        "dy": ref_area.dy,
        "valid_cells": ref_area.valid_cells,
        "nodata_cells": ref_area.nodata_cells,
        "valid_samples": ref_area.valid_samples,
        "nodata_samples": ref_area.nodata_samples,
    }


def _read_reference_from_raster(path: Path, *, dx: float, dy: float, nodata: float | None) -> SurfaceAreaResult:
    with rasterio.open(path) as ds:
        z = ds.read(1).astype("float64", copy=False)
        nodata_value = float(nodata) if nodata is not None else ds.nodata
    return compute_reference_surface_area(z, dx=dx, dy=dy, nodata_value=nodata_value)


def _write_resolution_manifest_csv(csv_path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# YARDIMCI FONKSİYONLAR
# =============================================================================

def _format_out_path(out: Path, *, params: dict[str, object]) -> Path:
    """Çıktı yolundaki şablonları doldurur.

    Desteklenen şablonlar: {preset}, {rows}, {cols}, {dx}, {dy}, {seed}, {timestamp}

    Args:
        out: Şablon içerebilen çıktı yolu
        params: Şablonu doldurmak için kullanılacak parametreler

    Returns:
        Doldurulmuş Path nesnesi

    Raises:
        ValueError: Geçersiz şablon veya bilinmeyen placeholder
    """
    s = str(out)
    if "{" not in s:
        return out
    try:
        return Path(s.format(**params))
    except KeyError as e:
        raise ValueError(f"--out şablonunda bilinmeyen placeholder: {e.args[0]!r}") from e
    except Exception as e:
        raise ValueError(f"Geçersiz --out şablonu: {s!r} ({e})") from e


def _print_header() -> None:
    """Başlık bilgisini yazdırır."""
    print("=" * 60)
    print("  SENTETIK DSM/DEM GEOTIFF ÜRETİCİ")
    print("  Yüzey Alanı Hesaplama Projesi")
    print("=" * 60)
    print()


def _print_parameters(
    args: argparse.Namespace,
    geometry: ResolvedGridGeometry,
    memory_str: str,
    file_size_str: str,
    actual_seed: int,
) -> None:
    """Kullanılacak parametreleri yazdırır."""
    print("PARAMETRELER:")
    print("-" * 40)
    terrain_family = "analytic" if is_analytic_preset(str(args.preset)) else "raster_first"
    print(f"  Arazi tipi (preset):    {args.preset}")
    print(f"  Benchmark ailesi:       {terrain_family}")
    print(f"  Boyut:                  {geometry.rows} x {geometry.cols} piksel")
    print(f"  Piksel boyutu:          dx={geometry.dx:g}m, dy={geometry.dy:g}m")
    print(f"  Fiziksel extent:        {geometry.extent_height_m:.1f}m x {geometry.extent_width_m:.1f}m")
    if terrain_family == "analytic":
        print(f"  Grid çözümleme modu:    {geometry.mode}")
    seed_info = f"{actual_seed}" + (" (rastgele)" if args.seed is None else " (kullanıcı belirli)")
    print(f"  Seed:                   {seed_info}")
    print(f"  Relief çarpanı:         {args.relief}")
    print(f"  Roughness:              {args.roughness_m}m")
    print(f"  fBm workers:            {args.fbm_workers}")
    if args.extent_width is not None and args.extent_height is not None:
        print(f"  İstenen extent:         {float(args.extent_height):.1f}m x {float(args.extent_width):.1f}m")
    if args.eval_gsd:
        print(f"  Ek eval GSD:            {', '.join(f'{float(v):g}' for v in args.eval_gsd)}")
        print(f"  Resampling:             {args.resampling}")
    print(f"  CRS:                    {args.crs}")
    print(f"  Origin:                 ({args.origin_x}, {args.origin_y})")
    print(f"  Nodata:                 {args.nodata}")
    if args.nodata_holes > 0:
        print(f"  Nodata delikleri:       {args.nodata_holes} adet (r={args.nodata_radius_m}m)")
    print("-" * 40)
    print(f"  Tahmini bellek:         {memory_str}")
    print(f"  Tahmini dosya boyutu:   {file_size_str}")
    print()


def _print_preset_info(preset: str) -> None:
    """Seçilen preset hakkında bilgi yazdırır."""
    info = {
        # Gerçekçi arazi tipleri
        "mountain": "Dağlık arazi - fBm noise, keskin sırtlar, zirveler ve erozyon vadileri",
        "valley": "Vadi ve akarsu - V/U şekilli ana vadi, kıvrımlı akarsu, taşkın ovası",
        "hills": "Yumuşak tepeler - düşük frekanslı rolling hills, çayırlar için uygun",
        "coastal": "Kıyı şeridi - deniz-kara geçişi, kumsal, falezler, iç kısım tepeleri",
        "plateau": "Yüksek plato - düz üst yüzey, dik yamaçlar (escarpment)",
        "canyon": "Kanyon/boğaz - derin dar vadi, kıvrımlı rota, tabakalı duvarlar",
        "volcanic": "Volkanik arazi - ana koni, krater, yan koniler, lav kanalları",
        "glacial": "Buzul vadisi - U-şekilli vadi, sirk, yan/son morenler",
        "karst": "Karstik arazi - düdenler (sinkholes), hum'lar (koni tepeler)",
        "alluvial": "Alüvyal ova - düz delta, menderesli kanallar, oxbow gölleri",
        # Test pattern'leri
        "plane": "Düz eğimli yüzey - basit doğrulama testleri için",
        "waves": "Sinüzoidal dalgalı yüzey - pürüzlü alan hesaplama testi",
        "crater_field": "Krater/çukur alanı - Gauss çıkıntıları ile",
        "terraced": "Teraslı arazi - keskin yükseklik geçişleri",
        "patchwork": "Test tiplerinin karışımı - genel performans testi",
        "mixed": "Patchwork + ekstra tepeler - maksimum çeşitlilik",
        # Analitik benchmark'lar
        "analytic_plane": "Analitik düzlem - sürekli ground truth planar alana eşittir",
        "analytic_tilted_plane": "Analitik eğimli düzlem - sabit eğimli kapalı form referans",
        "analytic_sinusoidal": "Analitik sinüzoidal yüzey - periyodik ve sürekli türevli",
        "analytic_gaussian_hill": "Analitik Gaussian tepe - lokal pürüzsüz zirve",
        "analytic_multi_gaussian": "Birden çok analitik Gaussian tepe/çukur kombinasyonu",
        "analytic_saddle": "Analitik saddle - ters işaretli ana eğrilikler",
        "analytic_dome": "Analitik kubbe - sonlu destekli yumuşak tepe",
        "analytic_hybrid_multiscale": "Analitik hibrit yüzey - yumuşak bölgeler + lokal kaba patch",
    }
    description = info.get(preset, "Bilinmeyen preset")
    if is_analytic_preset(preset):
        category = "ANALITIK BENCHMARK"
    else:
        is_realistic = preset in ["mountain", "valley", "hills", "coastal", "plateau", "canyon", "volcanic", "glacial", "karst", "alluvial"]
        category = "GERÇEKÇİ ARAZİ" if is_realistic else "TEST PATTERNİ"
    print(f"PRESET BİLGİSİ [{category}]:")
    print(f"  {preset}: {description}")
    print()


def _valid_stats(
    z: object,
    *,
    nodata_value: float | None,
) -> tuple[float, float, float] | None:
    """Geçerli (nodata olmayan) yükseklikler için min/max/mean döndürür."""
    import numpy as np

    z_arr = np.asarray(z, dtype=np.float64)
    valid = np.isfinite(z_arr)
    if nodata_value is not None and np.isfinite(float(nodata_value)):
        valid &= ~np.isclose(z_arr, float(nodata_value))

    if not np.any(valid):
        return None

    zz = z_arr[valid]
    return float(zz.min()), float(zz.max()), float(zz.mean(dtype=np.float64))


def _build_generation_parameters(
    args: argparse.Namespace,
    *,
    geometry: ResolvedGridGeometry,
    terrain_family: str,
    actual_seed: int,
    analytic_parameters: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "target": getattr(args, "target", args.preset),
        "preset": args.preset,
        "benchmark_family": terrain_family,
        "rows": geometry.rows,
        "cols": geometry.cols,
        "dx": geometry.dx,
        "dy": geometry.dy,
        "seed": actual_seed,
        "relief": args.relief,
        "roughness_m": args.roughness_m,
        "fbm_workers": args.fbm_workers,
        "crs": args.crs,
        "origin_x": args.origin_x,
        "origin_y": args.origin_y,
        "nodata": args.nodata,
        "nodata_holes": args.nodata_holes,
        "nodata_radius_m": args.nodata_radius_m,
        "physical_extent": {
            "width_m": geometry.extent_width_m,
            "height_m": geometry.extent_height_m,
            "mode": geometry.mode,
        },
        "native_gsd": {
            "dx_m": geometry.dx,
            "dy_m": geometry.dy,
        },
        "eval_gsd": [float(v) for v in _normalize_eval_gsd_values(args.eval_gsd)],
        "resampling": args.resampling,
        "analytic_parameters": analytic_parameters,
    }


def _write_complexity_rasters(
    base_tif: Path,
    *,
    descriptors: dict[str, object],
    dx: float,
    dy: float,
    crs: str,
    origin_x: float,
    origin_y: float,
    nodata: float | None,
) -> list[str]:
    outputs: list[str] = []
    outdir = base_tif.parent / f"{base_tif.stem}_complexity"
    for name, arr in descriptors.items():
        out_path = outdir / f"{base_tif.stem}_{name}.tif"
        write_dem_float32_geotiff(
            path=out_path,
            z=arr,
            dx=dx,
            dy=dy,
            crs=crs,
            nodata=nodata,
            origin_x=origin_x,
            origin_y=origin_y,
        )
        outputs.append(str(out_path.resolve()))
    return outputs


# =============================================================================
# ARGÜMAN PARSER
# =============================================================================

def build_parser(*, defaults: SynthConfig = DEFAULT_SYNTH_CONFIG) -> argparse.ArgumentParser:
    """Komut satırı argüman parser'ını oluşturur.

    Args:
        defaults: Varsayılan değerler için SynthConfig nesnesi

    Returns:
        Yapılandırılmış ArgumentParser
    """
    def _help(name: str) -> str:
        """Dataclass field'ından help metnini alır."""
        return str(SynthConfig.__dataclass_fields__[name].metadata.get("help", ""))

    p = argparse.ArgumentParser(
        prog="python generate_synthetic_tif.py",
        description="""
Yüzey alanı hesaplama yöntemlerini test etmek için sentetik DSM/DEM GeoTIFF üretir.

GERÇEKÇİ ARAZİ TİPLERİ:
  mountain      - Dağlık arazi (fBm noise, sırtlar, zirveler)
  valley        - Vadi ve akarsu yatağı (V/U şekilli)
  hills         - Yumuşak tepeler (rolling hills)
  coastal       - Kıyı şeridi (deniz-kara geçişi)
  plateau       - Yüksek plato (düz üst, dik yamaç)
  canyon        - Kanyon/boğaz (derin dar vadi)
  volcanic      - Volkanik arazi (koni, krater)
  glacial       - Buzul vadisi (U-şekilli)
  karst         - Karstik arazi (düdenler)
  alluvial      - Alüvyal ova (delta, menderes)

TEST PATTERNLERİ:
  plane         - Düz eğimli yüzey
  waves         - Sinüzoidal dalgalar
  crater_field  - Krater/çukur alanları
  terraced      - Teraslı/basamaklı arazi
  patchwork     - Test tiplerinin karışımı
  mixed         - Patchwork + ekstra çeşitlilik
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
ÖRNEKLER:
  # Gerçekçi dağlık arazi (varsayılan)
  python generate_synthetic_tif.py

  # Vadi ve akarsu yatağı
  python generate_synthetic_tif.py --target valley --rows 5000 --cols 5000

  # Yüksek çözünürlüklü kıyı şeridi
  python generate_synthetic_tif.py --target coastal --rows 8000 --cols 8000 --dx 0.5

  # Volkanik arazi (abartılı rölyef)
  python generate_synthetic_tif.py --target volcanic --relief 1.5

  # Buzul vadisi
  python generate_synthetic_tif.py --target glacial

  # Karstik arazi (düdenler ve hum'lar)
  python generate_synthetic_tif.py --target karst --rows 4000 --cols 4000

  # Nodata delikleri ile
  python generate_synthetic_tif.py --target mountain --nodata_holes 20

  # Raster-first ailesindeki tüm yüzeyleri üret
  python generate_synthetic_tif.py --target raster_first

  # Analitik Gaussian hill, fiziksel extent ile native GSD tanımı
  python generate_synthetic_tif.py --target analytic_gaussian_hill --dx 0.05 --extent_width 40 --extent_height 30

  # Analitik hibrit yüzey + çoklu çözünürlük benchmark çıktıları
  python generate_synthetic_tif.py --target analytic_hybrid_multiscale --dx 0.1 --extent_width 60 --extent_height 60 --eval_gsd 0.5 1 2 5
        """,
    )

    # Çıktı dosyası
    p.add_argument(
        "--out", "-o",
        type=Path,
        default=Path(defaults.out),
        help=_help("out"),
    )

    # Temel parametreler
    p.add_argument(
        "--target", "-t",
        choices=TARGET_CHOICES,
        default=defaults.target,
        help=_help("target"),
    )
    # Geriye dönük uyumluluk: eski CLI seçenekleri gizli tutulur.
    p.add_argument("--preset", "-p", dest="legacy_preset", choices=SYNTHETIC_PRESETS, help=argparse.SUPPRESS)
    p.add_argument(
        "--benchmark_family",
        dest="legacy_benchmark_family",
        choices=BENCHMARK_FAMILY_CHOICES,
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--all-presets",
        dest="legacy_all_presets",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--rows", "-r",
        type=int,
        default=defaults.rows,
        help=_help("rows"),
    )
    p.add_argument(
        "--cols", "-c",
        type=int,
        default=defaults.cols,
        help=_help("cols"),
    )
    p.add_argument(
        "--dx",
        type=float,
        default=defaults.dx,
        help=_help("dx"),
    )
    p.add_argument(
        "--dy",
        type=float,
        default=defaults.dy,
        help=_help("dy"),
    )
    p.add_argument(
        "--extent_width",
        type=float,
        default=defaults.extent_width,
        help=_help("extent_width"),
    )
    p.add_argument(
        "--extent_height",
        type=float,
        default=defaults.extent_height,
        help=_help("extent_height"),
    )
    p.add_argument(
        "--seed", "-s",
        type=int,
        default=defaults.seed,
        help=_help("seed"),
    )

    # Yüzey özellikleri
    p.add_argument(
        "--relief",
        type=float,
        default=defaults.relief,
        help=_help("relief"),
    )
    p.add_argument(
        "--roughness_m",
        type=float,
        default=defaults.roughness_m,
        help=_help("roughness_m"),
    )
    p.add_argument(
        "--fbm-workers",
        type=int,
        default=defaults.fbm_workers,
        help=_help("fbm_workers"),
    )

    # Coğrafi parametreler
    p.add_argument(
        "--crs",
        type=str,
        default=defaults.crs,
        help=_help("crs"),
    )
    p.add_argument(
        "--origin_x",
        type=float,
        default=defaults.origin_x,
        help=_help("origin_x"),
    )
    p.add_argument(
        "--origin_y",
        type=float,
        default=defaults.origin_y,
        help=_help("origin_y"),
    )

    # Nodata parametreleri
    p.add_argument(
        "--nodata",
        type=float,
        default=defaults.nodata,
        help=_help("nodata"),
    )
    p.add_argument(
        "--nodata_holes",
        type=int,
        default=defaults.nodata_holes,
        help=_help("nodata_holes"),
    )
    p.add_argument(
        "--nodata_radius_m",
        type=float,
        default=defaults.nodata_radius_m,
        help=_help("nodata_radius_m"),
    )
    p.add_argument(
        "--eval_gsd",
        type=float,
        nargs="*",
        default=list(defaults.eval_gsd),
        help=_help("eval_gsd"),
    )
    p.add_argument(
        "--resampling",
        choices=["nearest", "bilinear", "cubic"],
        default=defaults.resampling,
        help=_help("resampling"),
    )
    p.add_argument(
        "--continuous_rel_tol",
        type=float,
        default=defaults.continuous_rel_tol,
        help=_help("continuous_rel_tol"),
    )
    p.add_argument(
        "--continuous_abs_tol",
        type=float,
        default=defaults.continuous_abs_tol,
        help=_help("continuous_abs_tol"),
    )
    p.add_argument(
        "--continuous_base_samples",
        type=int,
        default=defaults.continuous_base_samples,
        help=_help("continuous_base_samples"),
    )
    p.add_argument(
        "--continuous_max_levels",
        type=int,
        default=defaults.continuous_max_levels,
        help=_help("continuous_max_levels"),
    )
    p.add_argument(
        "--complexity",
        action=argparse.BooleanOptionalAction,
        default=defaults.complexity,
        help=_help("complexity"),
    )
    p.add_argument(
        "--write_complexity_rasters",
        action=argparse.BooleanOptionalAction,
        default=defaults.write_complexity_rasters,
        help=_help("write_complexity_rasters"),
    )
    p.add_argument(
        "--complexity_window",
        type=int,
        default=defaults.complexity_window,
        help=_help("complexity_window"),
    )

    # Ek seçenekler
    p.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Sessiz mod - sadece hataları ve sonucu göster",
    )

    return p


# =============================================================================
# ÇALIŞTIRMA YARDIMCILARI
# =============================================================================

def _resolve_actual_seed(seed: int | None) -> int:
    """Kullanılacak gerçek seed değerini belirler."""
    import random

    if seed is None:
        return random.randint(0, 2**31 - 1)
    return int(seed)


def _resolve_generation_out_path(
    out_template: Path,
    *,
    preset: str,
    rows: int,
    cols: int,
    dx: float,
    dy: float,
    actual_seed: int,
    timestamp: str,
) -> Path:
    """Tek bir preset üretimi için çıktı yolunu çözer."""
    return _format_out_path(
        out_template,
        params={
            "preset": preset,
            "rows": rows,
            "cols": cols,
            "dx": dx,
            "dy": dy,
            "seed": actual_seed,
            "timestamp": timestamp,
        },
    )


def _print_reference_section(title: str, ref_area: SurfaceAreaResult | ContinuousSurfaceReference) -> None:
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)
    print(f"  Düzlemsel Alan (2D):     {ref_area.planar_area_m2:,.2f} m²")
    print(f"                           {ref_area.planar_area_ha:,.4f} ha")
    print(f"                           {ref_area.planar_area_km2:,.6f} km²")
    print()
    print(f"  Yüzey Alanı (3D):        {ref_area.surface_area_m2:,.2f} m²")
    print(f"                           {ref_area.surface_area_ha:,.4f} ha")
    print(f"                           {ref_area.surface_area_km2:,.6f} km²")
    print()
    print(f"  Yüzey/Düzlem Oranı:      {ref_area.surface_ratio:.6f}")
    print(f"  Artış Yüzdesi:           {(ref_area.surface_ratio - 1.0) * 100:.4f}%")


def _run_raster_first_generation(
    args: argparse.Namespace,
    *,
    geometry: ResolvedGridGeometry,
    actual_seed: int,
    out: Path,
    quiet: bool,
) -> int:
    if not quiet:
        print("Sentetik DSM üretiliyor...")
        print(f"  Preset: {args.preset}")
        print(f"  Boyut: {geometry.rows} x {geometry.cols}")

    progress = None if quiet else ProgressPrinter()
    try:
        z = generate_synthetic_dsm(
            rows=int(geometry.rows),
            cols=int(geometry.cols),
            dx=float(geometry.dx),
            dy=float(geometry.dy),
            preset=str(args.preset),
            seed=actual_seed,
            relief=float(args.relief),
            roughness_m=float(args.roughness_m),
            fbm_workers=int(args.fbm_workers),
            nodata_value=float(args.nodata) if args.nodata is not None else None,
            nodata_holes=int(args.nodata_holes),
            nodata_radius_m=float(args.nodata_radius_m),
            progress=progress,
        )
    except Exception as e:
        print(f"❌ DSM üretim hatası: {e}")
        return 1
    if progress is not None:
        progress.finish()

    if not quiet:
        stats = _valid_stats(z, nodata_value=float(args.nodata) if args.nodata is not None else None)
        if stats is None:
            print("⚠️  DSM üretildi ancak geçerli hücre bulunamadı.\n")
        else:
            z_min, z_max, z_mean = stats
            print(f"✓ DSM üretildi: min={z_min:.2f}m, max={z_max:.2f}m, mean={z_mean:.2f}m\n")

    try:
        ref_area = compute_reference_surface_area(
            z,
            dx=float(geometry.dx),
            dy=float(geometry.dy),
            nodata_value=float(args.nodata) if args.nodata is not None else None,
        )
    except Exception as e:
        print(f"⚠️  Yüzey alanı hesaplama hatası: {e}")
        ref_area = None

    if not quiet:
        print(f"GeoTIFF yazılıyor: {out}")

    try:
        info = write_dem_float32_geotiff(
            path=out,
            z=z,
            dx=float(geometry.dx),
            dy=float(geometry.dy),
            crs=str(args.crs),
            nodata=float(args.nodata) if args.nodata is not None else None,
            origin_x=float(args.origin_x),
            origin_y=float(args.origin_y),
        )
    except Exception as e:
        print(f"❌ GeoTIFF yazım hatası: {e}")
        return 1

    if not quiet:
        print()
        print("=" * 60)
        print("✓ BAŞARILI!")
        print("=" * 60)

    print(f"Dosya: {out}")
    print(f"  Boyut: {info.width} x {info.height} piksel")
    print(f"  Piksel: dx={info.dx:g}m, dy={info.dy:g}m")
    print(f"  Preset: {args.preset}, Seed: {actual_seed}")
    if not quiet:
        print(f"  Dosya boyutu: {out.stat().st_size / (1024 * 1024):.1f} MB")

    if ref_area is not None:
        _print_reference_section("NATIVE-GRID REFERANS YÜZEY ALANI (Benchmark)", ref_area)
        if ref_area.nodata_cells > 0:
            print()
            print(f"  Geçerli hücreler:        {ref_area.valid_cells:,}")
            print(f"  Nodata hücreler:         {ref_area.nodata_cells:,}")

        print()
        print("-" * 60)
        print("Bu değerler raster-first benchmark için native-grid referanstır.")
        print("Analitik ground truth değildir.")

        generation_parameters = _build_generation_parameters(
            args,
            geometry=geometry,
            terrain_family="raster_first",
            actual_seed=actual_seed,
            analytic_parameters=None,
        )
        json_file = out.with_suffix(".reference.json")
        _write_reference_json(
            json_file,
            generation_parameters=generation_parameters,
            native_grid_reference=ref_area,
            continuous_ground_truth=None,
            tif_path=out,
            resolution_records=[
                ResolutionReferenceRecord(
                    label="native",
                    tif_file=str(out.resolve()),
                    dx=float(geometry.dx),
                    dy=float(geometry.dy),
                    is_native=True,
                    resampling="native",
                    reference=ref_area,
                )
            ],
            complexity_summary=None,
            complexity_files=None,
        )
        print(f"\nJSON formatında kaydedildi: {json_file}")

    if not quiet:
        print()
        print("Bu dosyayı yüzey alanı hesaplama ile test etmek için:")
        suggested_outdir = out.parent / "out_run"
        print(f'  python main.py run --dem "{out}" --outdir "{suggested_outdir}"')

    return 0


def _run_analytic_generation(
    args: argparse.Namespace,
    *,
    geometry: ResolvedGridGeometry,
    actual_seed: int,
    out: Path,
    quiet: bool,
) -> int:
    if not quiet:
        print("Analitik benchmark yüzeyi oluşturuluyor...")
        print(f"  Preset: {args.preset}")
        print(f"  Native grid: {geometry.rows} x {geometry.cols}")

    analytic_surface = build_analytic_surface(
        str(args.preset),
        extent_width_m=float(geometry.extent_width_m),
        extent_height_m=float(geometry.extent_height_m),
        relief=float(args.relief),
        roughness_m=float(args.roughness_m),
        seed=actual_seed,
    )

    xs = (np.arange(geometry.cols, dtype=np.float64) + 0.5) * float(geometry.dx)
    ys = (np.arange(geometry.rows, dtype=np.float64) + 0.5) * float(geometry.dy)
    xg, yg = np.meshgrid(xs, ys)
    z_native = analytic_surface.evaluate(xg, yg).astype(np.float64, copy=False)

    nodata_value = float(args.nodata) if args.nodata is not None else None
    holes = []
    hole_mask = np.zeros((geometry.rows, geometry.cols), dtype=bool)
    if nodata_value is not None and int(args.nodata_holes) > 0:
        holes = generate_circular_holes(
            rng=np.random.default_rng(actual_seed + 200),
            count=int(args.nodata_holes),
            base_radius_m=float(args.nodata_radius_m),
            width_m=float(geometry.extent_width_m),
            height_m=float(geometry.extent_height_m),
        )
        hole_mask = circular_hole_mask_for_grid(
            rows=geometry.rows,
            cols=geometry.cols,
            dx=float(geometry.dx),
            dy=float(geometry.dy),
            holes=holes,
        )

    z_export = z_native.copy()
    if nodata_value is not None and np.any(hole_mask):
        z_export[hole_mask] = nodata_value

    if not quiet:
        stats = _valid_stats(z_export, nodata_value=nodata_value)
        if stats is None:
            print("⚠️  Analitik yüzey üretildi ancak geçerli hücre bulunamadı.\n")
        else:
            z_min, z_max, z_mean = stats
            print(f"✓ Analitik yüzey örneklendi: min={z_min:.2f}m, max={z_max:.2f}m, mean={z_mean:.2f}m\n")

    if not quiet:
        print("Continuous ground truth hesaplanıyor...")
    continuous_ground_truth = compute_continuous_surface_reference(
        analytic_surface,
        extent_width_m=float(geometry.extent_width_m),
        extent_height_m=float(geometry.extent_height_m),
        holes=holes if nodata_value is not None else None,
        rel_tol=float(args.continuous_rel_tol),
        abs_tol=float(args.continuous_abs_tol),
        base_samples=int(args.continuous_base_samples),
        max_levels=int(args.continuous_max_levels),
    )

    native_grid_reference = compute_reference_surface_area(
        z_export,
        dx=float(geometry.dx),
        dy=float(geometry.dy),
        nodata_value=nodata_value,
    )

    if not quiet:
        print(f"GeoTIFF yazılıyor: {out}")

    info = write_dem_float32_geotiff(
        path=out,
        z=z_export,
        dx=float(geometry.dx),
        dy=float(geometry.dy),
        crs=str(args.crs),
        nodata=nodata_value,
        origin_x=float(args.origin_x),
        origin_y=float(args.origin_y),
    )

    complexity_summary: dict[str, dict[str, float | int]] | None = None
    complexity_files: list[str] | None = None
    if bool(args.complexity) or bool(args.write_complexity_rasters):
        descriptor_arrays = compute_complexity_descriptors(
            z_native,
            dx=float(geometry.dx),
            dy=float(geometry.dy),
            window_size=int(args.complexity_window),
        )
        complexity_summary = summarize_complexity_descriptors(descriptor_arrays, mask=hole_mask if np.any(hole_mask) else None)
        if nodata_value is not None and np.any(hole_mask):
            masked_arrays = {
                name: np.where(hole_mask, nodata_value, values).astype(np.float64, copy=False)
                for name, values in descriptor_arrays.items()
            }
        else:
            masked_arrays = descriptor_arrays
        if bool(args.write_complexity_rasters):
            complexity_files = _write_complexity_rasters(
                out,
                descriptors=masked_arrays,
                dx=float(geometry.dx),
                dy=float(geometry.dy),
                crs=str(args.crs),
                origin_x=float(args.origin_x),
                origin_y=float(args.origin_y),
                nodata=nodata_value,
            )

    resolution_records: list[ResolutionReferenceRecord] = [
        ResolutionReferenceRecord(
            label="native",
            tif_file=str(out.resolve()),
            dx=float(geometry.dx),
            dy=float(geometry.dy),
            is_native=True,
            resampling="native",
            reference=native_grid_reference,
        )
    ]

    eval_gsd_values = _normalize_eval_gsd_values(args.eval_gsd)
    if eval_gsd_values:
        resampled_dir = out.parent / f"{out.stem}_resampled"
        rs = parse_resampling(str(args.resampling))
        for gsd in eval_gsd_values:
            if math.isclose(float(gsd), float(geometry.dx), rel_tol=0.0, abs_tol=1e-12) and math.isclose(
                float(geometry.dx),
                float(geometry.dy),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                continue
            resampled_path = resampled_dir / f"{out.stem}_gsd{safe_gsd_tag(float(gsd))}_{rs.name}.tif"
            res_info = resample_dem(
                src_path=out,
                dst_path=resampled_path,
                target_gsd_m=float(gsd),
                resampling=rs,
                nodata=nodata_value,
            )
            resampled_reference = _read_reference_from_raster(
                resampled_path,
                dx=float(res_info.dx),
                dy=float(res_info.dy),
                nodata=nodata_value,
            )
            resolution_records.append(
                ResolutionReferenceRecord(
                    label=f"gsd_{float(gsd):g}",
                    tif_file=str(resampled_path.resolve()),
                    dx=float(res_info.dx),
                    dy=float(res_info.dy),
                    is_native=False,
                    resampling=rs.name,
                    reference=resampled_reference,
                )
            )

    if not quiet:
        print()
        print("=" * 60)
        print("✓ BAŞARILI!")
        print("=" * 60)

    print(f"Dosya: {out}")
    print(f"  Boyut: {info.width} x {info.height} piksel")
    print(f"  Piksel: dx={info.dx:g}m, dy={info.dy:g}m")
    print(f"  Preset: {args.preset}, Seed: {actual_seed}")
    print(f"  Fiziksel extent: {geometry.extent_width_m:g}m x {geometry.extent_height_m:g}m")
    if not quiet:
        print(f"  Dosya boyutu: {out.stat().st_size / (1024 * 1024):.1f} MB")

    _print_reference_section("CONTINUOUS GROUND TRUTH", continuous_ground_truth)
    print()
    print("-" * 60)
    print("Bu değerler sürekli z=f(x,y) yüzeyinden türetilmiştir.")
    print("Native raster örneklemesinden ayrı tutulur.")

    _print_reference_section("NATIVE-GRID RASTER REFERENCE", native_grid_reference)
    if native_grid_reference.nodata_cells > 0:
        print()
        print(f"  Geçerli hücreler:        {native_grid_reference.valid_cells:,}")
        print(f"  Nodata hücreler:         {native_grid_reference.nodata_cells:,}")

    generation_parameters = _build_generation_parameters(
        args,
        geometry=geometry,
        terrain_family="analytic",
        actual_seed=actual_seed,
        analytic_parameters=analytic_surface.parameters,
    )
    json_file = out.with_suffix(".reference.json")
    _write_reference_json(
        json_file,
        generation_parameters=generation_parameters,
        native_grid_reference=native_grid_reference,
        continuous_ground_truth=continuous_ground_truth,
        tif_path=out,
        resolution_records=resolution_records,
        complexity_summary=complexity_summary,
        complexity_files=complexity_files,
    )
    print(f"\nJSON formatında kaydedildi: {json_file}")

    manifest_rows: list[dict[str, object]] = []
    for record in resolution_records:
        manifest_rows.append(
            {
                "terrain_family": "analytic",
                "preset": args.preset,
                "label": record.label,
                "is_native": record.is_native,
                "tif_file": record.tif_file,
                "native_dx_m": geometry.dx,
                "native_dy_m": geometry.dy,
                "evaluated_dx_m": record.dx,
                "evaluated_dy_m": record.dy,
                "evaluated_gsd_m": record.dx if math.isclose(record.dx, record.dy, rel_tol=0.0, abs_tol=1e-12) else "",
                "extent_width_m": geometry.extent_width_m,
                "extent_height_m": geometry.extent_height_m,
                "resampling": record.resampling,
                "continuous_planar_area_m2": continuous_ground_truth.planar_area_m2,
                "continuous_surface_area_m2": continuous_ground_truth.surface_area_m2,
                "continuous_surface_ratio": continuous_ground_truth.surface_ratio,
                "native_planar_area_m2": record.reference.planar_area_m2,
                "native_surface_area_m2": record.reference.surface_area_m2,
                "native_surface_ratio": record.reference.surface_ratio,
                "valid_cells": record.reference.valid_cells,
                "nodata_cells": record.reference.nodata_cells,
            }
        )
    manifest_csv = out.with_suffix(".reference_levels.csv")
    _write_resolution_manifest_csv(manifest_csv, manifest_rows)
    print(f"CSV manifest kaydedildi: {manifest_csv}")

    if complexity_summary is not None:
        print("Karmaşıklık özetleri JSON içine eklendi.")
    if complexity_files:
        print(f"Karmaşıklık rasterları yazıldı: {len(complexity_files)} adet")

    if not quiet:
        print()
        print("Bu dosyayı yüzey alanı hesaplama ile test etmek için:")
        suggested_outdir = out.parent / "out_run"
        print(f'  python main.py run --dem "{out}" --outdir "{suggested_outdir}"')

    return 0


def _run_single_generation(
    args: argparse.Namespace,
    *,
    actual_seed: int,
    out: Path,
    quiet: bool,
    batch_index: int,
    batch_total: int,
) -> int:
    """Tek bir preset için üretim akışını çalıştırır."""
    if not quiet:
        if batch_total > 1:
            print()
            print("=" * 60)
            print(f"[{batch_index}/{batch_total}] PRESET: {args.preset}")
            print("=" * 60)
        print("Parametreler doğrulanıyor...")

    try:
        geometry = _resolve_grid_geometry(args)
    except ValidationError as e:
        print(f"\n❌ PARAMETRE HATASI: {e}")
        print("\nKullanım bilgisi için: python generate_synthetic_tif.py --help")
        return 1

    validation_errors = validate_parameters(
        rows=geometry.rows,
        cols=geometry.cols,
        dx=geometry.dx,
        dy=geometry.dy,
        preset=args.preset,
        relief=args.relief,
        roughness_m=args.roughness_m,
        fbm_workers=args.fbm_workers,
        nodata_holes=args.nodata_holes,
        nodata_radius_m=args.nodata_radius_m,
        extent_width=args.extent_width,
        extent_height=args.extent_height,
        eval_gsd=_normalize_eval_gsd_values(args.eval_gsd),
        resampling=args.resampling,
        continuous_rel_tol=args.continuous_rel_tol,
        continuous_abs_tol=args.continuous_abs_tol,
        continuous_base_samples=args.continuous_base_samples,
        continuous_max_levels=args.continuous_max_levels,
        complexity_window=args.complexity_window,
    )

    if validation_errors:
        print("\n❌ PARAMETRE HATALARI:")
        for err in validation_errors:
            print(f"   • {err}")
        print("\nKullanım bilgisi için: python generate_synthetic_tif.py --help")
        return 1

    if not quiet:
        print("✓ Tüm parametreler geçerli.\n")

    memory_mb, memory_str = estimate_memory_usage(geometry.rows, geometry.cols)
    _file_mb, file_size_str = estimate_file_size(geometry.rows, geometry.cols)
    if memory_mb > 4000 and not quiet:
        print("⚠️  UYARI: Tahmini bellek kullanımı 4 GB'ı aşıyor!")
        print("   Sisteminizde yeterli RAM olduğundan emin olun.\n")

    if not quiet:
        _print_parameters(args, geometry, memory_str, file_size_str, actual_seed)
        _print_preset_info(args.preset)

    out.parent.mkdir(parents=True, exist_ok=True)

    if is_analytic_preset(str(args.preset)):
        return _run_analytic_generation(args, geometry=geometry, actual_seed=actual_seed, out=out, quiet=quiet)
    return _run_raster_first_generation(args, geometry=geometry, actual_seed=actual_seed, out=out, quiet=quiet)


def main(argv: list[str] | None = None, *, defaults: SynthConfig = DEFAULT_SYNTH_CONFIG) -> int:
    """Script'in ana giriş noktası.

    Args:
        argv: Komut satırı argümanları (None ise sys.argv kullanılır)
        defaults: Varsayılan yapılandırma

    Returns:
        Çıkış kodu (0=başarılı, 1=hata)
    """
    args = build_parser(defaults=defaults).parse_args(argv)
    args.target = _normalize_target_argument(args)
    quiet = args.quiet

    if not quiet:
        _print_header()

    target_presets = _resolve_target_presets(str(args.target))
    effective_all = len(target_presets) > 1
    if not quiet and effective_all:
        print(f"Toplu hedef modu etkin ({args.target}). {len(target_presets)} adet yüzey tipi üretilecek.")

    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    planned_runs: list[tuple[argparse.Namespace, int, Path]] = []
    seen_outputs: dict[Path, str] = {}

    for preset in target_presets:
        run_args = argparse.Namespace(**vars(args))
        run_args.preset = preset
        actual_seed = _resolve_actual_seed(run_args.seed)
        try:
            geometry = _resolve_grid_geometry(run_args)
        except ValidationError as e:
            print(f"❌ Parametre hatası: {e}")
            return 1
        run_args.rows = geometry.rows
        run_args.cols = geometry.cols
        run_args.dy = geometry.dy

        try:
            out = _resolve_generation_out_path(
                Path(run_args.out),
                preset=str(run_args.preset),
                rows=int(geometry.rows),
                cols=int(geometry.cols),
                dx=float(geometry.dx),
                dy=float(geometry.dy),
                actual_seed=actual_seed,
                timestamp=timestamp,
            )
        except ValueError as e:
            print(f"❌ Çıktı yolu hatası: {e}")
            return 1

        existing_preset = seen_outputs.get(out)
        if existing_preset is not None:
            print("❌ Çıktı yolu çakışması tespit edildi.")
            print(f"   {existing_preset!r} ve {preset!r} aynı dosyaya yazıyor: {out}")
            print("   Tüm presetler modunda --out şablonunda {preset} veya başka ayırt edici alanlar kullanın.")
            return 1

        seen_outputs[out] = preset
        planned_runs.append((run_args, actual_seed, out))

    generated_outputs: list[Path] = []
    total_runs = len(planned_runs)
    for batch_index, (run_args, actual_seed, out) in enumerate(planned_runs, start=1):
        rc = _run_single_generation(
            run_args,
            actual_seed=actual_seed,
            out=out,
            quiet=quiet,
            batch_index=batch_index,
            batch_total=total_runs,
        )
        if rc != 0:
            return rc
        generated_outputs.append(out)

    if not quiet and total_runs > 1:
        print()
        print("=" * 60)
        print("TOPLU ÜRETİM TAMAMLANDI")
        print("=" * 60)
        for out in generated_outputs:
            print(f"  - {out}")

    return 0


def _write_reference_json(
    json_path: Path,
    *,
    generation_parameters: dict[str, object],
    native_grid_reference: SurfaceAreaResult,
    continuous_ground_truth: ContinuousSurfaceReference | None,
    tif_path: Path,
    resolution_records: list[ResolutionReferenceRecord],
    complexity_summary: dict[str, dict[str, float | int]] | None,
    complexity_files: list[str] | None,
) -> None:
    """Referans yüzey alanı bilgisini JSON olarak kaydeder."""
    from datetime import datetime, timezone

    terrain_family = str(generation_parameters.get("benchmark_family", "raster_first"))
    native_grid_reference_payload = _surface_area_result_to_dict(native_grid_reference)
    resolution_payload = [
        {
            "label": record.label,
            "tif_file": record.tif_file,
            "is_native": record.is_native,
            "resampling": record.resampling,
            "grid_info": _grid_info_from_reference(record.reference),
            "native_grid_reference": _surface_area_result_to_dict(record.reference),
        }
        for record in resolution_records
    ]
    data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tif_file": str(tif_path.resolve()),
        "terrain_family": terrain_family,
        "reference_method": "native_grid_two_triangle",
        "reference_limitations": [
            "native_grid_reference raster çözünürlüğüne bağlıdır.",
            "continuous_ground_truth yalnızca analitik benchmark ailesinde bulunur.",
        ],
        "parameters": generation_parameters,
        "generation_parameters": generation_parameters,
        "reference_surface_area": native_grid_reference_payload,
        "native_grid_reference": native_grid_reference_payload,
        "continuous_ground_truth": None
        if continuous_ground_truth is None
        else _continuous_reference_to_dict(continuous_ground_truth),
        "grid_info": _grid_info_from_reference(native_grid_reference),
        "multi_resolution": resolution_payload,
        "complexity_summary": complexity_summary,
        "complexity_files": complexity_files,
        "description": (
            "Bu dosya sentetik benchmark için hem native-grid raster referansını hem de "
            "varsa continuous ground truth bilgisini ayrı alanlarda saklar."
        ),
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# =============================================================================
# SCRIPT GİRİŞ NOKTASI
# =============================================================================

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
