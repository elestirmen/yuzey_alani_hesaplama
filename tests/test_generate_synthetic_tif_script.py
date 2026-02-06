from __future__ import annotations

import json

import numpy as np


def test_generate_synthetic_tif_script(tmp_path) -> None:
    import generate_synthetic_tif

    out = tmp_path / "bench_synth.tif"
    rc = generate_synthetic_tif.main(
        [
            "--out",
            str(out),
            "--preset",
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


def test_generate_synthetic_parser_respects_config_seed_default() -> None:
    import generate_synthetic_tif

    parser = generate_synthetic_tif.build_parser(defaults=generate_synthetic_tif.SynthConfig(seed=123))
    args = parser.parse_args([])
    assert args.seed == 123
