from __future__ import annotations

import numpy as np


def test_synth_command_writes_geotiff(tmp_path) -> None:
    from surface_area.cli import main as cli_main

    out = tmp_path / "synthetic_patchwork.tif"
    rc = cli_main(
        [
            "synth",
            "--out",
            str(out),
            "--preset",
            "patchwork",
            "--rows",
            "64",
            "--cols",
            "80",
            "--dx",
            "2",
            "--seed",
            "123",
            "--nodata_holes",
            "3",
        ]
    )
    assert rc == 0

    import rasterio

    with rasterio.open(out) as ds:
        assert ds.width == 80
        assert ds.height == 64
        assert ds.count == 1
        assert ds.dtypes[0] == "float32"
        assert ds.nodata == -9999.0
        assert ds.crs is not None
        assert abs(float(ds.transform.a) - 2.0) < 1e-9
        assert abs(float(ds.transform.e) + 2.0) < 1e-9
        arr = ds.read(1)

    nodata = -9999.0
    valid = arr != nodata
    assert int((~valid).sum()) > 0

    v = arr[valid]
    assert v.size > 0
    assert np.isfinite(v).all()
    assert float(v.max() - v.min()) > 5.0

