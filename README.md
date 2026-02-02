# DEM 3B Yüzey Alanı (Surface Area) Hesaplama

Bu proje, bir DEM/DSM GeoTIFF rasterını farklı hedef GSD değerlerine yeniden örnekleyip (resample) birden fazla yöntemle 3B yüzey alanını (A3D) hesaplar; sonuçları CSV (long + wide) ve grafikler olarak üretir. Büyük rasterlar için `rasterio.block_windows` ile blok-blok işlenir.

## Kurulum

Python 3.10+ (test edildi: 3.12).

```bash
pip install -r requirements.txt
```

## CLI Kullanımı

Komut:

```bash
python -m surface_area run --dem dag_dsm.tif --outdir out --plots
```

Not: Çok büyük DEM’lerde kaynak çözünürlükten daha küçük GSD’ye (upsample) gitmek çıktı rasterını çok büyütebilir. Önce daha kaba GSD’lerle (örn. 2–50 m) denemek genellikle daha pratiktir.

Örnek (istenen formata yakın):

```bash
python -m surface_area run ^
  --dem dag_dsm.tif ^
  --outdir out ^
  --gsd 0.5 1 2 5 10 ^
  --methods jenness_window_8tri tin_2tri_cell gradient_multiplier bilinear_patch_integral multiscale_decomposed_area ^
  --resampling bilinear ^
  --slope_method horn ^
  --jenness_weight 0.25 ^
  --integral_N 5 ^
  --sigma_mode mult ^
  --sigma_m 2 5 ^
  --plots
```

### Önemli seçenekler

- `--gsd ...`: hedef GSD listesi (metre / DEM CRS linear unit).
- `--methods ...`: çalıştırılacak yöntemler.
- `--resampling bilinear|nearest|cubic`: yeniden örnekleme.
- `--nodata <value>`: DEM nodata değeri dataset’te tanımlı değilse override edin.
- `--slope_method horn|zt`: gradient tabanlı yöntemlerde türev kernel seçimi.
- `--sigma_mode mult|m` ve `--sigma_m ...`: multiscale için sigma (mult=GSD çarpanı, m=metre).
- `--plots`: PNG grafik üret.
- `--keep_resampled`: resample edilmiş GeoTIFF’leri disk üzerinde sakla (varsayılan: silinir).

## Çıktılar

`--outdir` altında:

- `results_long.csv`: satır = (gsd_m, method)
- `results_wide.csv`: satır = gsd_m, sütunlar = `{method}_{metric}` (örn. `gradient_multiplier_A3D`)
- `run_info.json`: sürüm bilgisi + sabit parametreler + DEM metadata
- Grafikler (opsiyonel):
  - `A3D_vs_GSD.png`
  - `ratio_vs_GSD.png`
  - `micro_ratio_vs_GSD.png` (multiscale)

Kolonlar (long):

- `A2D`: planimetrik alan (valid_cells * dx * dy)
- `A3D`: 3B yüzey alanı
- `ratio`: A3D / A2D
- `runtime_sec`: yöntem hesaplama süresi (compute-only; IO hariç)
- `note`: parametre özeti

## Yöntemler (özet)

1) `jenness_window_8tri`: 3x3 komşulukta merkez hücre etrafında 8 üçgen; Heron formülü. `--jenness_weight` (varsayılan 0.25).

2) `tin_2tri_cell`: köşe yüksekliklerini komşu hücre merkezlerinden (count==4) türetip her hücreyi 2 üçgen ile alan.

3) `gradient_multiplier`: `sqrt(1+p^2+q^2)` çarpanı ile alan; `--slope_method horn|zt`.

4) `bilinear_patch_integral`: hücreyi bilinear yüzey kabul edip NxN alt bölme ile sayısal integrasyon (`--integral_N`).

5) `multiscale_decomposed_area`: Gaussian low-pass ile `zL`; `A_total` ve `A_topo` gradient tabanlı hesaplanır; `A_micro = A_total - A_topo`.

## Nodata ve Kenar Yönetimi

- Nodata hücreleri maskelenir; nodata içeren bölgeler hesaplamaya dahil edilmez.
- Stencil gerektiren yöntemlerde (Horn/ZT, Jenness) tam stencil valid değilse hücre atlanır.
- Köşe-tabanlı yöntemlerde köşeler yalnızca 4 geçerli hücre merkezinden türetildiğinde kabul edilir (count==4). Bu, rasterın dış 1 hücre sınırını ve nodata komşuluğunu otomatik olarak dışlar.

## CRS ve Birimler

Alan ve GSD hesapları DEM’in CRS linear biriminde yapılır. CRS metre değilse CLI uyarı verir; bu durumda GSD değerleri/metrikler derece vb. birimlerde anlamsız olabilir.

## Testler

```bash
pytest -q
```

`tests/test_synthetic.py` sentetik yüzeyler (plane/sinusoid/paraboloid) ile yöntemlerin göreli hatasını doğrular.
