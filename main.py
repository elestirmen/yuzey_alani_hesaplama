from __future__ import annotations

from dataclasses import dataclass, field
import sys
from pathlib import Path

import surface_area.cli as surface_area_cli


config: dict[str, object] = {
    "dem": "vadi_dsm.tif",
    "outdir": "out_vadi",
    "gsd": [0.06, 0.1, 0.5, 1, 2, 5, 10, 20, 50],
    "method_choices": list(surface_area_cli.METHOD_CHOICES),
    "methods": None,
    "resampling": "bilinear",
    "slope_method": "horn",
    "nodata": None,
    "jenness_weight": 0.25,
    "integral_N": 5,
    "sector_jenness_rel_tol": 1e-4,
    "sector_jenness_abs_tol": 0.0,
    "sector_jenness_max_level": 5,
    "sector_jenness_min_samples": 3,
    "sigma_mode": "mult",
    "sigma_m": [2.0, 5.0],
    "plots": True,
    "keep_resampled": False,
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
        metadata={"help": "Target resolution list in meters. Example: [2, 5, 10]."},
    )
    methods: list[str] | None = field(
        default_factory=lambda: None,
        metadata={
            "help": (
                "Methods to run. If None, CLI defaults are used. "
                f"Choices: {', '.join(surface_area_cli.METHOD_CHOICES)}"
            )
        },
    )
    resampling: str = field(
        default="bilinear",
        metadata={"help": "Resampling method: bilinear | nearest | cubic."},
    )
    slope_method: str = field(
        default="horn",
        metadata={"help": "Slope/gradient kernel: horn | zt."},
    )
    nodata: float | None = field(
        default=None,
        metadata={"help": "Nodata override. None uses dataset nodata."},
    )
    jenness_weight: float = field(
        default=0.25,
        metadata={"help": "Jenness method weight coefficient."},
    )
    integral_N: int = field(
        default=5,
        metadata={"help": "Bilinear integral subdivision count (NxN)."},
    )
    sector_jenness_rel_tol: float = field(
        default=1e-4,
        metadata={"help": "Sector Jenness relative tolerance."},
    )
    sector_jenness_abs_tol: float = field(
        default=0.0,
        metadata={"help": "Sector Jenness absolute tolerance."},
    )
    sector_jenness_max_level: int = field(
        default=5,
        metadata={"help": "Sector Jenness maximum adaptive level."},
    )
    sector_jenness_min_samples: int = field(
        default=3,
        metadata={"help": "Sector Jenness minimum quadrature samples per triangle."},
    )
    sigma_mode: str = field(
        default="mult",
        metadata={"help": "Multiscale sigma interpretation: mult (GSD multiple) | m (meters)."},
    )
    sigma_m: list[float] = field(
        default_factory=lambda: [2.0, 5.0],
        metadata={"help": "Multiscale sigma list, interpreted by sigma_mode."},
    )
    plots: bool = field(
        default=True,
        metadata={"help": "If True, generate PNG plots (CLI: --plots)."},
    )
    keep_resampled: bool = field(
        default=False,
        metadata={"help": "If True, keep resampled GeoTIFF files (CLI: --keep_resampled)."},
    )
    workers: int = field(
        default=1,
        metadata={"help": "Blockwise raster compute worker process count (CLI: --workers)."},
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
