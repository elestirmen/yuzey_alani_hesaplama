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
        "bilinear_patch_integral",
    ],
    "jenness_focus": [
        "jenness_window_8tri",
        "sector_adaptive_jenness_integral",
    ],
    "full": list(AVAILABLE_METHODS),
}
METHOD_PRESET_NOTES: dict[str, str] = {
    "default": "Standart secim. Cogu durumda buradan baslanir.",
    "fast": "En hizli secim. Once hizli test icin uygun.",
    "balanced": "Hiz ve kalite arasinda orta yol.",
    "jenness_focus": "Sadece Jenness ile ilgili iki metot.",
    "full": "Tum metotlar. En yavas secim.",
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


config: dict[str, object] = {
    # Girdi ve cikti klasoru.
    "dem": "vadi_dsm.tif",
    "outdir": "out_vadi",
    # Hangi cozumlerde calisacagi.
    # Liste uzarsa toplam sure de uzar.
    "gsd": [0.06, 0.1, 0.5, 1, 2, 5, 10, 20, 50],
    # Metot secimi:
    # - method_choice = hazir grup sec
    # - methods = elle liste yaz
    # methods doluysa method_choice dikkate alinmaz.
    # Pratik ozet:
    #   default       -> standart grup
    #   fast          -> hizli grup
    #   balanced      -> orta yol
    #   jenness_focus -> sadece Jenness metotlari
    #   full          -> tum metotlar
    "method_choice": "default",
    "methods": None,
    "method_choices": list(AVAILABLE_METHODS),
    "method_presets": {name: list(methods) for name, methods in METHOD_PRESETS.items()},
    # Rasteri yeni GSD'ye indirirken kullanilan yontem.
    # Genelde bilinear yeterli olur.
    "resampling": "bilinear",
    # Eğim hesabinin tipi.
    # Sadece gradient tabanli metotlari etkiler.
    "slope_method": "horn",
    # None ise rasterin kendi nodata degeri kullanilir.
    "nodata": None,
    # Sadece jenness_window_8tri icin.
    "jenness_weight": 0.25,
    # Sadece bilinear_patch_integral icin.
    # Buyurse daha yavas ama daha ince hesap yapar.
    "integral_N": 5,
    # Sadece sector_adaptive_jenness_integral icin.
    # rel_tol kuculurse daha dikkatli hesap yapar ama yavaslar.
    # max_level buyurse daha derine boler ama yavaslar.
    # min_samples ucgen icindeki ornekleme yogunlugudur.
    "sector_jenness_rel_tol": 1e-4,
    "sector_jenness_abs_tol": 0.0,
    "sector_jenness_max_level": 5,
    "sector_jenness_min_samples": 3,
    # Sadece multiscale_decomposed_area icin.
    # mult = GSD kati, m = dogrudan metre.
    "sigma_mode": "mult",
    "sigma_m": [2.0, 5.0],
    # Grafik uret.
    "plots": True,
    # Ara GeoTIFF dosyalarini sakla.
    "keep_resampled": False,
    # Paralel worker sayisi.
    "workers": 8,
}


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Settings for running the CLI from main.py."""

    dem: str = field(
        default="vadi_dsm.tif",
        metadata={"help": "Input DEM/DSM GeoTIFF path (relative to workspace or absolute path)."},
    )
    outdir: str = field(
        default="out_vadi",
        metadata={"help": "Output directory path (created if needed)."},
    )
    gsd: list[float] = field(
        default_factory=lambda: [0.06, 0.1, 0.5, 1, 2, 5, 10, 20, 50],
        metadata={
            "help": "Calisacagi GSD listesi. Liste uzarsa sure uzar. Ornek: [0.5, 1, 2, 5]."
        },
    )
    method_choice: str = field(
        default="default",
        metadata={
            "help": f"Hazir metot grubu. Secenekler: {', '.join(METHOD_PRESETS)}. methods doluysa bu kullanilmaz."
        },
    )
    methods: list[str] | None = field(
        default_factory=lambda: None,
        metadata={
            "help": f"Elle metot listesi. Doluysa method_choice yerine bu calisir. Secenekler: {', '.join(AVAILABLE_METHODS)}"
        },
    )
    resampling: str = field(
        default="bilinear",
        metadata={
            "help": "Yeniden ornekleme tipi: bilinear, nearest, cubic. Genelde bilinear yeterli."
        },
    )
    slope_method: str = field(
        default="horn",
        metadata={
            "help": "Egim hesabi tipi: horn veya zt. Sadece gradient tabanli metotlari etkiler."
        },
    )
    nodata: float | None = field(
        default=None,
        metadata={"help": "Nodata override. None ise rasterin kendi nodata degeri kullanilir."},
    )
    jenness_weight: float = field(
        default=0.25,
        metadata={"help": "Sadece jenness_window_8tri icin kullanilir."},
    )
    integral_N: int = field(
        default=5,
        metadata={
            "help": "Sadece bilinear_patch_integral icin. Buyurse daha ince ama daha yavas hesap yapar."
        },
    )
    sector_jenness_rel_tol: float = field(
        default=1e-4,
        metadata={"help": "Sadece sector_adaptive_jenness_integral icin. Kuculurse daha yavas ama daha dikkatli hesaplar."},
    )
    sector_jenness_abs_tol: float = field(
        default=0.0,
        metadata={"help": "Sadece sector_adaptive_jenness_integral icin. Genelde 0.0 birakilir."},
    )
    sector_jenness_max_level: int = field(
        default=5,
        metadata={
            "help": "Sadece sector_adaptive_jenness_integral icin. Buyurse daha derine boler ve yavaslar."
        },
    )
    sector_jenness_min_samples: int = field(
        default=3,
        metadata={"help": "Sadece sector_adaptive_jenness_integral icin. Ucgen icindeki ornekleme yogunlugu."},
    )
    sigma_mode: str = field(
        default="mult",
        metadata={"help": "Sadece multiscale_decomposed_area icin. mult = GSD kati, m = metre."},
    )
    sigma_m: list[float] = field(
        default_factory=lambda: [2.0, 5.0],
        metadata={"help": "Sadece multiscale_decomposed_area icin sigma listesi."},
    )
    plots: bool = field(
        default=True,
        metadata={"help": "True ise grafik uretir."},
    )
    keep_resampled: bool = field(
        default=False,
        metadata={"help": "True ise ara GeoTIFF dosyalari silinmez."},
    )
    workers: int = field(
        default=1,
        metadata={"help": "Paralel worker sayisi."},
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
        if any((not isinstance(v, (int, float))) for v in self.gsd):
            raise ValueError(f"gsd values must be numbers, got: {self.gsd!r}")
        if any(float(v) <= 0 for v in self.gsd):
            raise ValueError(f"gsd values must be > 0, got: {self.gsd!r}")

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

        argv: list[str] = [
            "run",
            "--dem",
            self.dem,
            "--outdir",
            self.outdir,
            "--gsd",
            *[f"{float(v):g}" for v in self.gsd],
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
