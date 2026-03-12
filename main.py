from __future__ import annotations

from dataclasses import dataclass, field
import sys
from pathlib import Path

import surface_area.cli as surface_area_cli


AVAILABLE_METHODS = tuple(surface_area_cli.METHOD_CHOICES)
METHOD_PRESETS: dict[str, list[str]] = {
    "default": list(surface_area_cli.DEFAULT_METHODS),
    "fast": [
        "gradient_multiplier",
        "tin_2tri_cell",
        "jenness_window_8tri",
    ],
    "balanced": [
        "gradient_multiplier",
        "tin_2tri_cell",
        "jenness_window_8tri",
        "sector_adaptive_jenness_integral",
        "bilinear_patch_integral",
    ],
    "jenness_focus": [
        "jenness_window_8tri",
        "sector_adaptive_jenness_integral",
    ],
    "full": list(AVAILABLE_METHODS),
}
METHOD_PRESET_NOTES: dict[str, str] = {
    "default": (
        "Ana karsilastirma grubu. Klasik taban cizgisini korurken sector-adaptive Jenness onerisine "
        "odaklanir; bilinear integral metodlarini varsayilan sete katmaz."
    ),
    "fast": "En hizli grup. Raster buyukse veya once kaba bir fikir edinmek istiyorsan bunu sec.",
    "balanced": (
        "Ana calisma disinda ek bir integral benchmark da gormek istersen bunu sec. "
        "Bilinear patch'i bilerek acikca ekler."
    ),
    "jenness_focus": "Jenness ailesi uzerinde calisiyorsan en anlamli grup budur. Klasik Jenness ile sector-adaptive Jenness'i yan yana kosar.",
    "full": "Tum metodlari calistirir. En kapsamli secimdir ama sure en cok bunda uzar.",
}


def _normalize_method_list(methods: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for method in methods:
        method_name = method.strip().lower()
        if not method_name or method_name in seen:
            continue
        normalized.append(method_name)
        seen.add(method_name)
    return normalized


def _resolve_methods(method_choice: str, methods: list[str] | None) -> list[str]:
    if methods is not None:
        resolved = _normalize_method_list(methods)
        if not resolved:
            raise ValueError("methods list is empty after normalization")
        invalid = sorted(set(resolved) - set(AVAILABLE_METHODS))
        if invalid:
            raise ValueError(f"Invalid methods: {invalid}. Choices: {list(AVAILABLE_METHODS)}")
        return resolved

    preset_key = method_choice.strip().lower()
    if preset_key not in METHOD_PRESETS:
        raise ValueError(f"method_choice must be one of: {sorted(METHOD_PRESETS)}")
    return list(METHOD_PRESETS[preset_key])


def _normalize_gsd_values(gsd_values: list[float | str]) -> list[float | str]:
    normalized: list[float | str] = []
    for value in gsd_values:
        if isinstance(value, str):
            token = value.strip().lower()
            if token != surface_area_cli.GSD_NATIVE_TOKEN:
                raise ValueError(
                    f"gsd string values must be '{surface_area_cli.GSD_NATIVE_TOKEN}', got: {value!r}"
                )
            normalized.append(token)
            continue
        if not isinstance(value, (int, float)):
            raise ValueError(
                f"gsd values must be numbers or '{surface_area_cli.GSD_NATIVE_TOKEN}', got: {gsd_values!r}"
            )
        numeric = float(value)
        if numeric <= 0:
            raise ValueError(f"gsd numeric values must be > 0, got: {gsd_values!r}")
        normalized.append(numeric)
    return normalized


config: dict[str, object] = {
    # ============================================================
    # 1) TEMEL DOSYALAR
    # ============================================================
    # Analiz edilecek DEM/DSM dosyasi.
    # Buraya GeoTIFF yolunu yaziyorsun.
    # Ornek: "vadi_dsm.tif"
    "dem": "vadi_dsm.tif",
    # Sonuclarin yazilacagi klasor.
    # CSV dosyalari, grafikler ve gecici ciktilar burada olusur.
    # Ornek: "out_vadi"
    "outdir": "out_vadi",
    # ============================================================
    # 2) HANGI COZUMLERDE CALISACAK?
    # ============================================================
    # GSD = piksel boyutu / hedef cozum.
    # "native" yazarsan goruntunun kendi piksel boyutunda calisir; yeniden ornekleme yapmaz.
    # Listedeki sayisal her deger icin raster yeniden orneklenir ve secili tum metodlar tekrar calisir.
    # Yani liste ne kadar uzunsa toplam sure de o kadar uzar.
    # Kisa bir test icin [1, 2, 5] gibi kucuk bir liste daha pratiktir.
    "gsd": ["native", 0.1, 0.5, 1, 2, 5, 10, 20, 50],
    # ============================================================
    # 3) HANGI METOTLAR CALISACAK?
    # ============================================================
    # method_choice:
    # Hazir bir metot grubu secmek icin kullanilir.
    # Cogu kullanici icin en rahat yol budur.
    #
    # methods:
    # Hazir grup yerine metotlari tek tek kendin yazmak istersen bunu kullanirsin.
    #
    # ONEMLI:
    # methods = None ise method_choice devrededir.
    # methods dolu bir liste ise method_choice artik kullanilmaz.
    #
    # Ornek 1:
    #   "method_choice": "jenness_focus",
    #   "methods": None
    # -> sadece Jenness ile ilgili iki metot calisir.
    #
    # Ornek 2:
    #   "method_choice": "default",
    #   "methods": ["gradient_multiplier", "tin_2tri_cell"]
    # -> method_choice yok sayilir, sadece bu iki metot calisir.
    #
    # Hazir seceneklerin anlami:
    #   default       -> ana karsilastirma grubu, sector-adaptive odakli
    #   fast          -> hizli grup, buyuk veri icin iyi
    #   balanced      -> default + bilinear integral benchmark
    #   jenness_focus -> Jenness gelistirirken en uygun grup
    #   full          -> tum metotlar, en yavas secim
    "method_choice": "default",
    "methods": None,
    # Asagidaki iki alan sadece referans olsun diye burada duruyor.
    # Program bu listeleri otomatik uretiyor; normalde degistirmen gerekmez.
    "method_choices": list(AVAILABLE_METHODS),
    "method_presets": {name: list(methods) for name, methods in METHOD_PRESETS.items()},
    # ============================================================
    # 4) YENIDEN ORNEKLEME VE ORTAK AYARLAR
    # ============================================================
    # Raster daha kaba veya daha ince bir GSD'ye gecirilirken kullanilan yontem.
    # bilinear: genel kullanim icin en guvenli secim
    # nearest : orijinal piksel degerlerini en sert haliyle korur
    # cubic   : daha yumusak sonuc verir, bazen biraz daha pahali olabilir
    "resampling": "bilinear",
    # Bir hucrenin egimini hesaplarken hangi komsu piksellerin kullanilacagini belirler.
    # Yani "egim" degeri hangi stencil / kernel ile turetilsin sorusunun cevabidir.
    # horn -> 3x3 pencerede 8 komsuyu agirlikli kullanir; genelde daha dengeli varsayilandir.
    # zt   -> sadece N, S, E, W komsularina bakar; daha sade ve bazen daha hizlidir.
    # Bu ayar sadece gradientten egim ureten metodlari etkiler:
    # - gradient_multiplier
    # - multiscale_decomposed_area
    # Ornegin tin_2tri_cell veya jenness_window_8tri sonucunu degistirmez.
    "slope_method": "horn",
    # Rasterin icindeki nodata degeri yanlissa burada elle verebilirsin.
    # None birakirsan dosyanin kendi nodata bilgisi kullanilir.
    "nodata": None,
    # ============================================================
    # 5) YONTEME OZEL AYARLAR
    # ============================================================
    # Sadece jenness_window_8tri metodunu etkiler.
    # Genelde varsayilan deger yeterlidir.
    "jenness_weight": 0.25,
    # Sadece bilinear_patch_integral icin kullanilir.
    # N buyurse hucre icini daha kucuk parcalarla hesaplar.
    # Bu genelde daha ince hesap demektir ama maliyet hizla artar.
    # Hizli deneme icin 3 veya 5, daha ayrintili deneme icin daha buyuk degerler dusunulebilir.
    "integral_N": 5,
    # Asagidaki 4 alan sadece sector_adaptive_jenness_integral icin kullanilir.
    #
    # rel_tol:
    # Hata toleransi gibi dusunebilirsin.
    # Kucuk olursa algoritma "daha emin olayim" deyip daha fazla refine eder.
    # Sonuc genelde daha dikkatli olur ama sure uzar.
    #
    # abs_tol:
    # Ek bir mutlak tolerans siniri.
    # Cogu durumda 0.0 birakmak yeterlidir.
    #
    # max_level:
    # Ucgenleri en fazla kac seviye bolmesine izin verdigini belirler.
    # Buyuk olursa zor hucrelerde daha derine iner, bu da daha yavas olabilir.
    #
    # min_samples:
    # Bir ucgen icinde kac quadrature ornegi kullanilacagina etki eder.
    # Dusurmek her zaman hiz kazandirmayabilir.
    "sector_jenness_rel_tol": 1e-4,
    "sector_jenness_abs_tol": 0.0,
    "sector_jenness_max_level": 5,
    "sector_jenness_min_samples": 3,
    # Asagidaki 2 alan sadece multiscale_decomposed_area icin anlamlidir.
    #
    # sigma_mode:
    # mult -> sigma_m listesindeki degerler GSD'nin kati gibi okunur.
    #         Ornek: GSD=2 ise sigma=2.0 degeri 4 metre olur.
    # m    -> sigma_m listesindeki degerler dogrudan metre kabul edilir.
    #
    # sigma_m:
    # Kullanilacak sigma listesi.
    "sigma_mode": "mult",
    "sigma_m": [2.0, 5.0],
    # Hesap bittiginde grafik PNG'leri de olussun mu?
    "plots": True,
    # Her GSD icin olusan ara GeoTIFF dosyalari diskte kalsin mi?
    # False ise is bitince silinir.
    "keep_resampled": False,
    # Kac worker process kullanilsin?
    # Buyuk sayi her zaman daha hizli olmaz; raster I/O bazen darboaz olur.
    # Emin degilsen once 1, 2, 4 gibi degerlerle deneme yapmak daha dogrudur.
    "workers": 8,
}


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Settings for running the CLI from main.py."""

    dem: str = field(
        default="vadi_dsm.tif",
        metadata={"help": "Analiz edilecek DEM/DSM GeoTIFF dosyasi. Goreli veya tam yol verebilirsin."},
    )
    outdir: str = field(
        default="out_vadi",
        metadata={"help": "Sonuclarin yazilacagi klasor. Yoksa otomatik olusturulur."},
    )
    gsd: list[float | str] = field(
        default_factory=lambda: ["native", 0.06, 0.1, 0.5, 1, 2, 5, 10, 20, 50],
        metadata={
            "help": (
                "Hedef GSD listesi. 'native' yazarsan rasteri kendi cozumunde kullanir; "
                "sayisal degerlerde ise her hedef GSD icin yeniden ornekleme yapip secili tum metotlari tekrar calistirir. "
                "Bu yuzden liste uzadikca toplam sure artar. Hizli deneme icin ['native', 1, 2, 5] gibi kisa bir liste iyidir."
            )
        },
    )
    method_choice: str = field(
        default="default",
        metadata={
            "help": (
                f"Hazir metot grubu. Secenekler: {', '.join(METHOD_PRESETS)}. "
                "methods=None ise bu alan kullanilir. Elle tek tek metot yazmak istemiyorsan en kolay secim budur."
            )
        },
    )
    methods: list[str] | None = field(
        default_factory=lambda: None,
        metadata={
            "help": (
                f"Elle metot listesi. Doluysa method_choice tamamen devre disi kalir ve sadece buradaki metotlar calisir. "
                f"Kullanilabilir adlar: {', '.join(AVAILABLE_METHODS)}"
            )
        },
    )
    resampling: str = field(
        default="bilinear",
        metadata={
            "help": (
                "Rasteri yeni GSD'ye gecirirken kullanilan yeniden ornekleme yontemi. "
                "bilinear genel kullanim icin iyi varsayilandir; nearest daha serttir; cubic daha yumusak sonuc verebilir."
            )
        },
    )
    slope_method: str = field(
        default="horn",
        metadata={
            "help": (
                "Bir hucrenin egimini hesaplarken hangi komsu piksellerin kullanilacagini belirler: "
                "horn 3x3 / 8 komsulu agirlikli kernel, zt ise N-S-E-W yonlerinde daha sade fark kernelidir. "
                "Bu ayar sadece gradient_multiplier ve multiscale_decomposed_area gibi gradient tabanli metodlari etkiler."
            )
        },
    )
    nodata: float | None = field(
        default=None,
        metadata={
            "help": (
                "Nodata degerini elle vermek icin kullanilir. Raster dosyasinin nodata bilgisi yanlissa burada duzeltebilirsin. "
                "None ise dosyanin icindeki nodata kullanilir."
            )
        },
    )
    jenness_weight: float = field(
        default=0.25,
        metadata={
            "help": (
                "Bu parametre sadece jenness_window_8tri metodunu etkiler. "
                "Genelde varsayilan deger yeterlidir; ancak bu metodun merkez agirligini degistirmek istersen buraya dokunursun."
            )
        },
    )
    integral_N: int = field(
        default=5,
        metadata={
            "help": (
                "Bu parametre sadece bilinear_patch_integral icin kullanilir. "
                "Hucre ici integrali kac alt parcaya bolerek hesaplayacagini belirler. "
                "N buyudukce hesap daha ince olur ama sure de belirgin sekilde artar."
            )
        },
    )
    sector_jenness_rel_tol: float = field(
        default=1e-4,
        metadata={
            "help": (
                "Bu parametre sadece sector_adaptive_jenness_integral icin kullanilir. "
                "Algoritmanin ne kadar kolay duracagini belirleyen goreli toleranstir. "
                "Deger kuculdukce daha fazla refine yapar; bu genelde daha pahali ama daha dikkatli hesap demektir."
            )
        },
    )
    sector_jenness_abs_tol: float = field(
        default=0.0,
        metadata={
            "help": (
                "Bu parametre sadece sector_adaptive_jenness_integral icin kullanilir. "
                "Mutlak toleranstir. Cogu durumda 0.0 olarak birakmak yeterlidir."
            )
        },
    )
    sector_jenness_max_level: int = field(
        default=5,
        metadata={
            "help": (
                "Bu parametre sadece sector_adaptive_jenness_integral icin kullanilir. "
                "Adaptif bolmenin en fazla kac seviye derine inebilecegini belirler. "
                "Buyuk deger zor hucrelerde daha ayrintili hesap demektir, ama runtime da artabilir."
            )
        },
    )
    sector_jenness_min_samples: int = field(
        default=3,
        metadata={
            "help": (
                "Bu parametre sadece sector_adaptive_jenness_integral icin kullanilir. "
                "Ucgen icindeki quadrature ornekleme yogunlugunu etkiler. "
                "Dusuk deger her zaman daha hizli sonuc vermeyebilir."
            )
        },
    )
    sigma_mode: str = field(
        default="mult",
        metadata={
            "help": (
                "Bu parametre sadece multiscale_decomposed_area icin anlamlidir. "
                "'mult' secilirse sigma_m degerleri GSD'nin kati gibi yorumlanir; 'm' secilirse dogrudan metre kabul edilir."
            )
        },
    )
    sigma_m: list[float] = field(
        default_factory=lambda: [2.0, 5.0],
        metadata={
            "help": (
                "Bu parametre sadece multiscale_decomposed_area icin kullanilir. "
                "Kullanilacak sigma degerlerinin listesidir; anlamini sigma_mode belirler."
            )
        },
    )
    plots: bool = field(
        default=True,
        metadata={"help": "True ise hesap bittikten sonra grafik PNG dosyalari da uretir."},
    )
    keep_resampled: bool = field(
        default=False,
        metadata={
            "help": "True ise her GSD icin olusan ara GeoTIFF dosyalari saklanir. Debug icin yararlidir, disk kullanimini artirir."
        },
    )
    workers: int = field(
        default=1,
        metadata={
            "help": (
                "Paralel worker process sayisi. Daha buyuk deger bazen hizlandirir ama her veri setinde ayni etkiyi vermez. "
                "Emin degilsen once 1, 2 veya 4 ile deneme yapmak daha guvenlidir."
            )
        },
    )

    def resolved_methods(self) -> list[str]:
        return _resolve_methods(self.method_choice, self.methods)

    def validate(self) -> None:
        dem_path = Path(self.dem)
        if not dem_path.exists():
            raise ValueError(f"DEM not found: {dem_path}")
        if dem_path.is_dir():
            raise ValueError(f"DEM must be a file, got directory: {dem_path}")

        outdir_path = Path(self.outdir)
        if outdir_path.exists() and not outdir_path.is_dir():
            raise ValueError(f"outdir must be a directory path, got file: {outdir_path}")

        if not self.gsd:
            raise ValueError("gsd list must not be empty")
        _ = _normalize_gsd_values(self.gsd)

        _ = self.resolved_methods()

        if self.resampling not in {"bilinear", "nearest", "cubic"}:
            raise ValueError("resampling must be one of: bilinear, nearest, cubic")
        if self.slope_method not in {"horn", "zt"}:
            raise ValueError("slope_method must be one of: horn, zt")

        if self.nodata is not None and not isinstance(self.nodata, (int, float)):
            raise ValueError("nodata must be a number or null")
        if float(self.jenness_weight) <= 0:
            raise ValueError("jenness_weight must be > 0")
        if int(self.integral_N) <= 0:
            raise ValueError("integral_N must be > 0")
        if float(self.sector_jenness_rel_tol) < 0:
            raise ValueError("sector_jenness_rel_tol must be >= 0")
        if float(self.sector_jenness_abs_tol) < 0:
            raise ValueError("sector_jenness_abs_tol must be >= 0")
        if int(self.sector_jenness_max_level) < 0:
            raise ValueError("sector_jenness_max_level must be >= 0")
        if int(self.sector_jenness_min_samples) <= 0:
            raise ValueError("sector_jenness_min_samples must be > 0")
        if self.sigma_mode not in {"mult", "m"}:
            raise ValueError("sigma_mode must be 'mult' or 'm'")
        if not self.sigma_m:
            raise ValueError("sigma_m list must not be empty")
        if any(float(v) <= 0 for v in self.sigma_m):
            raise ValueError(f"sigma_m values must be > 0, got: {self.sigma_m!r}")
        if int(self.workers) <= 0:
            raise ValueError("workers must be > 0")

    def to_argv(self) -> list[str]:
        self.validate()
        resolved_methods = self.resolved_methods()
        normalized_gsd = _normalize_gsd_values(self.gsd)

        argv: list[str] = [
            "run",
            "--dem",
            self.dem,
            "--outdir",
            self.outdir,
            "--gsd",
            *[
                surface_area_cli.GSD_NATIVE_TOKEN if isinstance(v, str) else f"{float(v):g}"
                for v in normalized_gsd
            ],
            "--resampling",
            self.resampling,
            "--slope_method",
            self.slope_method,
            "--jenness_weight",
            f"{float(self.jenness_weight):g}",
            "--integral_N",
            str(int(self.integral_N)),
            "--sector_jenness_rel_tol",
            f"{float(self.sector_jenness_rel_tol):g}",
            "--sector_jenness_abs_tol",
            f"{float(self.sector_jenness_abs_tol):g}",
            "--sector_jenness_max_level",
            str(int(self.sector_jenness_max_level)),
            "--sector_jenness_min_samples",
            str(int(self.sector_jenness_min_samples)),
            "--sigma_mode",
            self.sigma_mode,
            "--sigma_m",
            *[f"{float(v):g}" for v in self.sigma_m],
            "--workers",
            str(int(self.workers)),
            "--methods",
            *resolved_methods,
        ]

        if self.nodata is not None:
            argv.extend(["--nodata", f"{float(self.nodata):g}"])
        if self.plots:
            argv.append("--plots")
        if self.keep_resampled:
            argv.append("--keep_resampled")
        return argv


def _config_run_kwargs(config_map: dict[str, object]) -> dict[str, object]:
    valid_keys = set(RunConfig.__dataclass_fields__.keys())  # type: ignore[attr-defined]
    return {k: v for k, v in config_map.items() if k in valid_keys}


DEFAULT_RUN_CONFIG = RunConfig(**_config_run_kwargs(config))


def _print_main_help() -> None:
    print("Usage:")
    print("  python main.py run --dem <path> --outdir <dir> [--gsd ...] [--methods ...] [--plots]")
    print("  python main.py              # runs with the top-level config mapping")
    print("  python main.py --help")
    print("")
    print("config keys you can edit in main.py:")
    for f in RunConfig.__dataclass_fields__.values():  # type: ignore[attr-defined]
        help_text = (f.metadata or {}).get("help", "")
        default_repr = config.get(f.name, None)
        print(f"  - {f.name}: {help_text} (current: {default_repr})")
    print(f"  - method_choices: available methods you can place into config['methods'] (current: {config['method_choices']})")
    print("  - method_presets: ready-made method groups:")
    for preset_name, methods in METHOD_PRESETS.items():
        note = METHOD_PRESET_NOTES[preset_name]
        print(f"      {preset_name}: {methods} | {note}")
    try:
        resolved = DEFAULT_RUN_CONFIG.resolved_methods()
        print(f"  - effective methods with current config: {resolved}")
    except ValueError as e:
        print(f"  - effective methods with current config: INVALID ({e})")


def main() -> int:
    argv = sys.argv[1:]
    if len(argv) == 1 and argv[0] in {"-h", "--help", "help"}:
        _print_main_help()
        return 0
    if argv:
        return int(surface_area_cli.main(argv))

    try:
        run_config = RunConfig(**_config_run_kwargs(config))
        return int(surface_area_cli.main(run_config.to_argv()))
    except TypeError as e:
        print(f"Invalid main.py config mapping: {e}", file=sys.stderr)
        return 2
    except ValueError as e:
        print(f"Invalid main.py config values: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
