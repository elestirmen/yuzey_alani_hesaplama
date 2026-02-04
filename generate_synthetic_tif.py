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
    python generate_synthetic_tif.py --preset valley --rows 5000 --cols 5000

    # Yüksek çözünürlüklü kıyı şeridi
    python generate_synthetic_tif.py --preset coastal --rows 8000 --cols 8000 --dx 0.5

    # Volkanik arazi
    python generate_synthetic_tif.py --preset volcanic --relief 1.5

    # Buzul vadisi
    python generate_synthetic_tif.py --preset glacial --rows 6000 --cols 6000

PERFORMANS NOTLARI:
------------------
- 10000x10000 piksel ≈ 400 MB bellek kullanımı
- Gerçekçi preset'ler (mountain, valley, vb.) daha fazla işlem gücü gerektirir
- scipy.ndimage modülü gereklidir

Yazar: Surface Area Calculator Project
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import NoReturn

from surface_area.io import write_dem_float32_geotiff
from surface_area.synthetic import (
    SYNTHETIC_PRESETS,
    generate_synthetic_dsm,
    compute_reference_surface_area,
    SurfaceAreaResult,
)


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

# Nodata hole limitleri
MAX_NODATA_HOLES = 1000
MIN_NODATA_RADIUS = 0.1  # 10 cm
MAX_NODATA_RADIUS = 1000.0  # 1 km

# Bellek tahmini için float32 boyutu
BYTES_PER_PIXEL = 4  # float32


# =============================================================================
# YAPILANDIRMA SINIFI
# =============================================================================

@dataclass(frozen=True, slots=True)
class SynthConfig:
    """Sentetik DSM üretimi için yapılandırma parametreleri.

    Bu sınıf, script'i IDE'den doğrudan çalıştırırken kullanılacak
    varsayılan değerleri tanımlar. Değerleri değiştirmek için
    DEFAULT_SYNTH_CONFIG'u düzenleyin.

    Attributes:
        out: Çıktı GeoTIFF dosya yolu (şablon destekler: {preset}, {rows}, vb.)
        preset: Arazi tipi - Gerçekçi: mountain, valley, hills, coastal, plateau,
                canyon, volcanic, glacial, karst, alluvial
                Test: plane, waves, crater_field, terraced, patchwork, mixed
        rows: Raster satır sayısı
        cols: Raster sütun sayısı
        dx: X yönünde piksel boyutu (metre)
        dy: Y yönünde piksel boyutu (metre, None ise dx kullanılır)
        seed: Rastgele sayı üreteci tohumu (tekrarlanabilirlik için)
        relief: Makro rölyef çarpanı (0=düz, 1=normal, >1=abartılı)
        roughness_m: Mikro pürüzlülük genliği (metre)
        crs: Koordinat referans sistemi (örn: EPSG:32636)
        origin_x: Sol üst köşe X koordinatı
        origin_y: Sol üst köşe Y koordinatı
        nodata: Nodata değeri (None ile devre dışı)
        nodata_holes: Eklenecek dairesel nodata delik sayısı
        nodata_radius_m: Nodata delikleri için taban yarıçap (metre)
    """

    out: str = field(
        default="out_synth/synth_{preset}_{rows}x{cols}_dx{dx:g}_seed{seed}_{timestamp}.tif",
        metadata={"help": "Çıktı GeoTIFF yolu ({preset}, {rows}, {cols}, {dx}, {seed}, {timestamp} şablonları desteklenir)"},
    )
    preset: str = field(
        default="mountain",   #"mountain",  # Varsayılan olarak gerçekçi dağlık arazi
        metadata={"help": f"Arazi tipi. Seçenekler: {', '.join(SYNTHETIC_PRESETS)}"},
    )
    rows: int = field(
        default=4096,
        metadata={"help": f"Raster satır sayısı ({MIN_ROWS}-{MAX_ROWS})"},
    )
    cols: int = field(
        default=4096,
        metadata={"help": f"Raster sütun sayısı ({MIN_COLS}-{MAX_COLS})"},
    )
    dx: float = field(
        default=1.0,
        metadata={"help": f"X piksel boyutu metre ({MIN_PIXEL_SIZE}-{MAX_PIXEL_SIZE})"},
    )
    dy: float | None = field(
        default=None,
        metadata={"help": "Y piksel boyutu metre (None ise dx kullanılır)"},
    )
    seed: int | None = field(
        default=None,  # None = her seferinde farklı rastgele seed
        metadata={"help": "Rastgele sayı tohumu (None = her seferinde farklı, sabit değer = tekrarlanabilir)"},
    )
    relief: float = field(
        default=1.0,
        metadata={"help": f"Makro rölyef çarpanı ({MIN_RELIEF}-{MAX_RELIEF})"},
    )
    roughness_m: float = field(
        default=0.75,
        metadata={"help": f"Mikro pürüzlülük genliği metre ({MIN_ROUGHNESS}-{MAX_ROUGHNESS})"},
    )
    crs: str = field(
        default="EPSG:32636",
        metadata={"help": "CRS string (örn: EPSG:32636 = UTM Zone 36N)"},
    )
    origin_x: float = field(
        default=500_000.0,
        metadata={"help": "Sol üst köşe X koordinatı (metre)"},
    )
    origin_y: float = field(
        default=4_500_000.0,
        metadata={"help": "Sol üst köşe Y koordinatı (metre)"},
    )
    nodata: float | None = field(
        default=-9999.0,
        metadata={"help": "Nodata değeri (None ile devre dışı bırakılır)"},
    )
    nodata_holes: int = field(
        default=0,
        metadata={"help": f"Eklenecek dairesel nodata delik sayısı (0-{MAX_NODATA_HOLES})"},
    )
    nodata_radius_m: float = field(
        default=12.0,
        metadata={"help": f"Nodata delikleri için taban yarıçap metre ({MIN_NODATA_RADIUS}-{MAX_NODATA_RADIUS})"},
    )


# IDE'den çalıştırırken kullanılacak varsayılan yapılandırma.
# Bu değerleri değiştirerek farklı sentetik veriler üretebilirsiniz.
DEFAULT_SYNTH_CONFIG = SynthConfig()


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
    nodata_holes: int,
    nodata_radius_m: float,
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


def _print_parameters(args: argparse.Namespace, dy: float, memory_str: str, file_size_str: str, actual_seed: int) -> None:
    """Kullanılacak parametreleri yazdırır."""
    print("PARAMETRELER:")
    print("-" * 40)
    print(f"  Arazi tipi (preset):    {args.preset}")
    print(f"  Boyut:                  {args.rows} x {args.cols} piksel")
    print(f"  Piksel boyutu:          dx={args.dx:g}m, dy={dy:g}m")
    print(f"  Gerçek boyut:           {args.rows * dy:.1f}m x {args.cols * args.dx:.1f}m")
    seed_info = f"{actual_seed}" + (" (rastgele)" if args.seed is None else " (kullanıcı belirli)")
    print(f"  Seed:                   {seed_info}")
    print(f"  Relief çarpanı:         {args.relief}")
    print(f"  Roughness:              {args.roughness_m}m")
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
    }
    description = info.get(preset, "Bilinmeyen preset")
    is_realistic = preset in ["mountain", "valley", "hills", "coastal", "plateau",
                               "canyon", "volcanic", "glacial", "karst", "alluvial"]
    category = "GERÇEKÇİ ARAZİ" if is_realistic else "TEST PATTERNİ"
    print(f"PRESET BİLGİSİ [{category}]:")
    print(f"  {preset}: {description}")
    print()


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
  python generate_synthetic_tif.py --preset valley --rows 5000 --cols 5000

  # Yüksek çözünürlüklü kıyı şeridi
  python generate_synthetic_tif.py --preset coastal --rows 8000 --cols 8000 --dx 0.5

  # Volkanik arazi (abartılı rölyef)
  python generate_synthetic_tif.py --preset volcanic --relief 1.5

  # Buzul vadisi
  python generate_synthetic_tif.py --preset glacial

  # Karstik arazi (düdenler ve hum'lar)
  python generate_synthetic_tif.py --preset karst --rows 4000 --cols 4000

  # Nodata delikleri ile
  python generate_synthetic_tif.py --preset mountain --nodata_holes 20
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
        "--preset", "-p",
        choices=SYNTHETIC_PRESETS,
        default=defaults.preset,
        help=_help("preset"),
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
        "--seed", "-s",
        type=int,
        default=None,  # None = rastgele seed
        help="Rastgele sayı tohumu. Belirtilmezse her seferinde farklı seed kullanılır. Aynı deseni tekrar üretmek için sabit bir değer verin.",
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

    # Ek seçenekler
    p.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Sessiz mod - sadece hataları ve sonucu göster",
    )

    return p


# =============================================================================
# ANA FONKSİYON
# =============================================================================

def main(argv: list[str] | None = None, *, defaults: SynthConfig = DEFAULT_SYNTH_CONFIG) -> int:
    """Script'in ana giriş noktası.

    Args:
        argv: Komut satırı argümanları (None ise sys.argv kullanılır)
        defaults: Varsayılan yapılandırma

    Returns:
        Çıkış kodu (0=başarılı, 1=hata)
    """
    args = build_parser(defaults=defaults).parse_args(argv)
    quiet = args.quiet

    # dy değerini belirle
    dy = float(args.dx if args.dy is None else args.dy)

    # Seed değerini belirle (None ise rastgele üret)
    import random
    if args.seed is None:
        # Rastgele seed üret (0 ile 2^31-1 arası)
        actual_seed = random.randint(0, 2**31 - 1)
    else:
        actual_seed = int(args.seed)

    # =========================================================================
    # PARAMETRE DOĞRULAMA (Kodun başında)
    # =========================================================================
    if not quiet:
        _print_header()
        print("Parametreler doğrulanıyor...")

    validation_errors = validate_parameters(
        rows=args.rows,
        cols=args.cols,
        dx=args.dx,
        dy=args.dy,
        preset=args.preset,
        relief=args.relief,
        roughness_m=args.roughness_m,
        nodata_holes=args.nodata_holes,
        nodata_radius_m=args.nodata_radius_m,
    )

    if validation_errors:
        print("\n❌ PARAMETRE HATALARI:")
        for err in validation_errors:
            print(f"   • {err}")
        print("\nKullanım bilgisi için: python generate_synthetic_tif.py --help")
        return 1

    if not quiet:
        print("✓ Tüm parametreler geçerli.\n")

    # Bellek ve dosya boyutu tahmini
    memory_mb, memory_str = estimate_memory_usage(args.rows, args.cols)
    file_mb, file_size_str = estimate_file_size(args.rows, args.cols)

    # Büyük dosya uyarısı
    if memory_mb > 4000 and not quiet:
        print("⚠️  UYARI: Tahmini bellek kullanımı 4 GB'ı aşıyor!")
        print("   Sisteminizde yeterli RAM olduğundan emin olun.\n")

    # =========================================================================
    # PARAMETRE BİLGİLERİNİ GÖSTER
    # =========================================================================
    if not quiet:
        _print_parameters(args, dy, memory_str, file_size_str, actual_seed)
        _print_preset_info(args.preset)

    # =========================================================================
    # ÇIKTI YOLUNU HAZIRLA
    # =========================================================================
    # Zaman damgası oluştur (her çalıştırmada farklı dosya adı için)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        out: Path = _format_out_path(
            args.out,
            params={
                "preset": str(args.preset),
                "rows": int(args.rows),
                "cols": int(args.cols),
                "dx": float(args.dx),
                "dy": dy,
                "seed": actual_seed,
                "timestamp": timestamp,
            },
        )
    except ValueError as e:
        print(f"❌ Çıktı yolu hatası: {e}")
        return 1

    out.parent.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # SENTETİK DSM ÜRETİMİ
    # =========================================================================
    if not quiet:
        print("Sentetik DSM üretiliyor...")
        print(f"  Preset: {args.preset}")
        print(f"  Boyut: {args.rows} x {args.cols}")

    try:
        z = generate_synthetic_dsm(
            rows=int(args.rows),
            cols=int(args.cols),
            dx=float(args.dx),
            dy=None if args.dy is None else float(args.dy),
            preset=str(args.preset),
            seed=actual_seed,
            relief=float(args.relief),
            roughness_m=float(args.roughness_m),
            nodata_value=float(args.nodata) if args.nodata is not None else None,
            nodata_holes=int(args.nodata_holes),
            nodata_radius_m=float(args.nodata_radius_m),
        )
    except Exception as e:
        print(f"❌ DSM üretim hatası: {e}")
        return 1

    if not quiet:
        print(f"✓ DSM üretildi: min={z.min():.2f}m, max={z.max():.2f}m, mean={z.mean():.2f}m\n")

    # =========================================================================
    # REFERANS YÜZEY ALANI HESAPLAMA (BENCHMARK İÇİN)
    # =========================================================================
    if not quiet:
        print("Referans yüzey alanı hesaplanıyor (benchmark için)...")

    try:
        ref_area: SurfaceAreaResult = compute_reference_surface_area(
            z,
            dx=float(args.dx),
            dy=dy,
            nodata_value=float(args.nodata) if args.nodata is not None else None,
        )
    except Exception as e:
        print(f"⚠️  Yüzey alanı hesaplama hatası: {e}")
        ref_area = None

    if ref_area is not None and not quiet:
        print(f"✓ Referans yüzey alanı hesaplandı.\n")

    # =========================================================================
    # GEOTIFF YAZIMI
    # =========================================================================
    if not quiet:
        print(f"GeoTIFF yazılıyor: {out}")

    try:
        info = write_dem_float32_geotiff(
            path=out,
            z=z,
            dx=float(args.dx),
            dy=dy,
            crs=str(args.crs),
            nodata=float(args.nodata) if args.nodata is not None else None,
            origin_x=float(args.origin_x),
            origin_y=float(args.origin_y),
        )
    except Exception as e:
        print(f"❌ GeoTIFF yazım hatası: {e}")
        return 1

    # =========================================================================
    # SONUÇ RAPORU
    # =========================================================================
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
        actual_size_mb = out.stat().st_size / (1024 * 1024)
        print(f"  Dosya boyutu: {actual_size_mb:.1f} MB")

    # =========================================================================
    # REFERANS YÜZEY ALANI BİLGİSİ (BENCHMARK)
    # =========================================================================
    if ref_area is not None:
        print()
        print("=" * 60)
        print("REFERANS YÜZEY ALANI (Benchmark için Ground Truth)")
        print("=" * 60)
        print(f"  Düzlemsel Alan (2D):     {ref_area.planar_area_m2:,.2f} m²")
        print(f"                           {ref_area.planar_area_ha:,.4f} ha")
        print(f"                           {ref_area.planar_area_km2:,.6f} km²")
        print()
        print(f"  Gerçek Yüzey Alanı (3D): {ref_area.surface_area_m2:,.2f} m²")
        print(f"                           {ref_area.surface_area_ha:,.4f} ha")
        print(f"                           {ref_area.surface_area_km2:,.6f} km²")
        print()
        print(f"  Yüzey/Düzlem Oranı:      {ref_area.surface_ratio:.6f}")
        print(f"  Artış Yüzdesi:           {(ref_area.surface_ratio - 1.0) * 100:.4f}%")

        if ref_area.nodata_cells > 0:
            print()
            print(f"  Geçerli hücreler:        {ref_area.valid_cells:,}")
            print(f"  Nodata hücreler:         {ref_area.nodata_cells:,}")

        print()
        print("-" * 60)
        print("Bu değerleri kendi yöntemlerinizle karşılaştırabilirsiniz.")
        print("Yöntemlerinizin doğruluğu = Hesaplanan / Referans")

        # JSON formatında da çıktı ver (programatik kullanım için)
        json_file = out.with_suffix(".reference.json")
        _write_reference_json(json_file, args, ref_area, out, actual_seed)
        print(f"\nJSON formatında kaydedildi: {json_file}")

    if not quiet:
        print()
        print("Bu dosyayı yüzey alanı hesaplama ile test etmek için:")
        print(f'  python main.py "{out}"')

    return 0


def _write_reference_json(
    json_path: Path,
    args: argparse.Namespace,
    ref_area: SurfaceAreaResult,
    tif_path: Path,
    actual_seed: int,
) -> None:
    """Referans yüzey alanı bilgisini JSON olarak kaydeder."""
    import json
    from datetime import datetime

    data = {
        "generated_at": datetime.now().isoformat(),
        "tif_file": str(tif_path.resolve()),
        "parameters": {
            "preset": args.preset,
            "rows": args.rows,
            "cols": args.cols,
            "dx": args.dx,
            "dy": args.dy if args.dy is not None else args.dx,
            "seed": actual_seed,
            "relief": args.relief,
            "roughness_m": args.roughness_m,
            "crs": args.crs,
            "nodata": args.nodata,
            "nodata_holes": args.nodata_holes,
        },
        "reference_surface_area": {
            "planar_area_m2": ref_area.planar_area_m2,
            "planar_area_ha": ref_area.planar_area_ha,
            "planar_area_km2": ref_area.planar_area_km2,
            "surface_area_m2": ref_area.surface_area_m2,
            "surface_area_ha": ref_area.surface_area_ha,
            "surface_area_km2": ref_area.surface_area_km2,
            "surface_ratio": ref_area.surface_ratio,
            "increase_percent": (ref_area.surface_ratio - 1.0) * 100,
        },
        "grid_info": {
            "rows": ref_area.rows,
            "cols": ref_area.cols,
            "dx": ref_area.dx,
            "dy": ref_area.dy,
            "valid_cells": ref_area.valid_cells,
            "nodata_cells": ref_area.nodata_cells,
        },
        "description": (
            "Bu dosya, sentetik DSM'nin GERÇEK (referans) yüzey alanını içerir. "
            "Bu değer, yüzey alanı hesaplama yöntemlerinin doğruluğunu test etmek için "
            "ground truth olarak kullanılabilir."
        ),
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# =============================================================================
# SCRIPT GİRİŞ NOKTASI
# =============================================================================

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
