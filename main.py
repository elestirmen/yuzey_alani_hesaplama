from __future__ import annotations

from dataclasses import MISSING, dataclass, field
import sys
from pathlib import Path

import surface_area.cli as surface_area_cli


@dataclass(frozen=True, slots=True)
class RunConfig:
    """main.py üzerinden çalıştırma ayarları.

    Bu sınıf IDE'de `main.py` dosyasını "Program" olarak çalıştırırken,
    parametreleri tek bir yerde kontrol edebilmeniz için var.

    Not: Buradaki alanlar, `python -m surface_area run ...` CLI argümanlarına çevrilir.
    """

    dem: str = field(
        default="vadi_dsm.tif",
        metadata={"help": "Girdi DEM/DSM GeoTIFF dosya yolu (workspace göreli veya tam yol)."},
    )
    outdir: str = field(
        default="out_vadi",
        metadata={"help": "Çıktı klasörü yolu (oluşturulur)."},
    )
    gsd: list[float] = field(
        default_factory=lambda: [0.06, 0.1,0.5,1,2,5,10,20,50],
        metadata={"help": "Hedef çözünürlük listesi (metre). Örn: [2, 5, 10]."},
    )
    methods: list[str] | None = field(
        default_factory=lambda: ["gradient_multiplier"],
        metadata={
            "help": (
                "Çalıştırılacak yöntemler. Varsayılan: ['gradient_multiplier'] (hızlı). "
                "Tümü için None verin. "
                f"Seçenekler: {', '.join(surface_area_cli.METHOD_CHOICES)}"
            )
        },
    )
    resampling: str = field(
        default="bilinear",
        metadata={"help": "Resampling yöntemi: bilinear | nearest | cubic."},
    )
    slope_method: str = field(
        default="horn",
        metadata={"help": "Eğim/gradient kernel: horn | zt."},
    )
    nodata: float | None = field(
        default=None,
        metadata={"help": "Nodata override. None => dataset nodata değeri kullanılır."},
    )
    jenness_weight: float = field(
        default=0.25,
        metadata={"help": "Jenness yöntemi ağırlık katsayısı (varsayılan 0.25)."},
    )
    integral_N: int = field(
        default=5,
        metadata={"help": "Bilinear integral alt bölme sayısı (NxN). Örn: 5."},
    )
    sigma_mode: str = field(
        default="mult",
        metadata={"help": "Multiscale sigma yorumu: mult (GSD çarpanı) | m (metre)."},
    )
    sigma_m: list[float] = field(
        default_factory=lambda: [2.0, 5.0],
        metadata={"help": "Multiscale sigma listesi (sigma_mode'a göre). Örn: [2, 5]."},
    )
    plots: bool = field(
        default=True,
        metadata={"help": "True ise PNG grafikler üretir (CLI: --plots)."},
    )
    keep_resampled: bool = field(
        default=False,
        metadata={"help": "True ise resample edilmiş GeoTIFF'leri saklar (CLI: --keep_resampled)."},
    )

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

        if self.methods is not None:
            invalid = sorted(set(self.methods) - set(surface_area_cli.METHOD_CHOICES))
            if invalid:
                raise ValueError(f"Invalid methods: {invalid}. Choices: {surface_area_cli.METHOD_CHOICES}")

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
        if self.sigma_mode not in {"mult", "m"}:
            raise ValueError("sigma_mode must be 'mult' or 'm'")
        if not self.sigma_m:
            raise ValueError("sigma_m list must not be empty")
        if any(float(v) <= 0 for v in self.sigma_m):
            raise ValueError(f"sigma_m values must be > 0, got: {self.sigma_m!r}")

    def to_argv(self) -> list[str]:
        self.validate()

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
            "--sigma_mode",
            self.sigma_mode,
            "--sigma_m",
            *[f"{float(v):g}" for v in self.sigma_m],
        ]

        if self.methods is not None:
            argv.extend(["--methods", *self.methods])
        if self.nodata is not None:
            argv.extend(["--nodata", f"{float(self.nodata):g}"])
        if self.plots:
            argv.append("--plots")
        if self.keep_resampled:
            argv.append("--keep_resampled")
        return argv


DEFAULT_RUN_CONFIG = RunConfig()


def _print_main_help() -> None:
    print("Usage:")
    print("  python main.py run --dem <path> --outdir <dir> [--gsd ...] [--methods ...] [--plots]")
    print("  python main.py              # DEFAULT_RUN_CONFIG ile (IDE için önerilir)")
    print("  python main.py --help")
    print("")
    print("RunConfig parametreleri (main.py içinden ayarlayabilirsiniz):")
    for f in RunConfig.__dataclass_fields__.values():  # type: ignore[attr-defined]
        help_text = (f.metadata or {}).get("help", "")
        if f.default is not MISSING:
            default_repr = f.default
        elif f.default_factory is not MISSING:
            default_repr = "<factory>"
        else:
            default_repr = None
        print(f"  - {f.name}: {help_text} (default: {default_repr})")

def main() -> int:
    argv = sys.argv[1:]
    if len(argv) == 1 and argv[0] in {"-h", "--help", "help"}:
        _print_main_help()
        return 0
    if argv:
        return int(surface_area_cli.main(argv))

    try:
        return int(surface_area_cli.main(DEFAULT_RUN_CONFIG.to_argv()))
    except ValueError as e:
        print(f"Invalid main.py defaults: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
