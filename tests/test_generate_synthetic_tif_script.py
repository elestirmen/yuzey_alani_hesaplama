from __future__ import annotations

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

