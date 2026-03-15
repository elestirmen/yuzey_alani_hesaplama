from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def test_generate_synthetic_tif_script(tmp_path) -> None:
    import generate_synthetic_tif

    out = tmp_path / "bench_synth.tif"
    rc = generate_synthetic_tif.main(
        [
            "--out",
            str(out),
            "--target",
            "mixed",
            "--rows",
            "80",
            "--cols",
            "64",
            "--dx",
            "2",
            "--seed",
            "7",
            "--nodata_holes",
            "2",
        ]
    )
    assert rc == 0

    import rasterio

    with rasterio.open(out) as ds:
        assert ds.width == 64
        assert ds.height == 80
        assert ds.dtypes[0] == "float32"
        assert ds.nodata == -9999.0
        arr = ds.read(1)

    nodata = -9999.0
    valid = arr != nodata
    assert int((~valid).sum()) > 0

    v = arr[valid]
    assert v.size > 0
    assert np.isfinite(v).all()

    ref_path = out.with_suffix(".reference.json")
    payload = json.loads(ref_path.read_text(encoding="utf-8"))

    assert payload["reference_method"] == "native_grid_two_triangle"
    assert payload["generated_at"].endswith("+00:00")

    grid = payload["grid_info"]
    ref = payload["reference_surface_area"]
    params = payload["parameters"]
    assert "valid_samples" in grid
    assert "nodata_samples" in grid
    assert abs(float(ref["planar_area_m2"]) - float(grid["valid_cells"]) * float(params["dx"]) * float(params["dy"])) < 1e-9

    manifest_path = tmp_path / generate_synthetic_tif.LATEST_GENERATED_DEM_LIST_NAME
    assert manifest_path.exists()
    manifest_lines = [
        line.strip()
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert manifest_lines == [str(out.resolve())]


def test_generate_synthetic_tif_script_supports_analytic_ground_truth_and_multiresolution(tmp_path) -> None:
    import generate_synthetic_tif

    out = tmp_path / "analytic_gaussian.tif"
    rc = generate_synthetic_tif.main(
        [
            "--out",
            str(out),
            "--target",
            "analytic_gaussian_hill",
            "--dx",
            "1",
            "--extent_width",
            "32",
            "--extent_height",
            "24",
            "--seed",
            "11",
            "--eval_gsd",
            "2",
            "4",
            "--continuous_base_samples",
            "33",
            "--continuous_max_levels",
            "2",
            "--complexity",
            "--write_complexity_rasters",
            "--quiet",
        ]
    )
    assert rc == 0

    import rasterio

    with rasterio.open(out) as ds:
        assert ds.width == 32
        assert ds.height == 24
        assert ds.dtypes[0] == "float32"

    payload = json.loads(out.with_suffix(".reference.json").read_text(encoding="utf-8"))
    assert payload["terrain_family"] == "analytic"
    assert payload["continuous_ground_truth"] is not None
    assert payload["native_grid_reference"]["surface_area_m2"] >= payload["native_grid_reference"]["planar_area_m2"]
    assert payload["generation_parameters"]["physical_extent"]["width_m"] == 32.0
    assert payload["generation_parameters"]["physical_extent"]["height_m"] == 24.0
    assert len(payload["multi_resolution"]) == 3
    assert payload["complexity_summary"] is not None
    assert len(payload["complexity_files"]) == 5

    manifest_csv = out.with_suffix(".reference_levels.csv")
    assert manifest_csv.exists()

    for row in payload["multi_resolution"]:
        assert Path(row["tif_file"]).exists()


def test_generate_synthetic_parser_respects_config_seed_default() -> None:
    import generate_synthetic_tif

    parser = generate_synthetic_tif.build_parser(defaults=generate_synthetic_tif.SynthConfig(seed=123))
    args = parser.parse_args([])
    assert args.seed == 123


def test_generate_synthetic_parser_supports_target_group_default() -> None:
    import generate_synthetic_tif

    parser = generate_synthetic_tif.build_parser(defaults=generate_synthetic_tif.SynthConfig(target="all"))

    args = parser.parse_args([])
    assert args.target == "all"

    args = parser.parse_args(["--target", "mountain"])
    assert args.target == "mountain"


def test_default_synth_config_uses_single_preset_target() -> None:
    import generate_synthetic_tif

    assert generate_synthetic_tif.DEFAULT_SYNTH_CONFIG.target == "mountain"
