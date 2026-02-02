# DEM 3D Yüzey Alanı Hesaplama Aracı

> DEM/DSM GeoTIFF verilerinden 3D yüzey alanı (A3D) hesaplama için kapsamlı Python kütüphanesi

---

## Genel Bakış

Bu proje, Sayısal Yükseklik Modeli (DEM) veya Sayısal Yüzey Modeli (DSM) GeoTIFF raster verilerini kullanarak **3 boyutlu yüzey alanı (A3D)** hesaplamalarını gerçekleştirir.

### Temel Özellikler

- **Çoklu Yöntem Desteği**: 5 farklı yüzey alanı hesaplama algoritması
- **Çoklu Çözünürlük Analizi**: Farklı GSD (Ground Sample Distance) değerlerinde yeniden örnekleme
- **Büyük Dosya Desteği**: `rasterio.block_windows` ile bellek-etkin blok işleme
- **Multiscale Analiz**: Gaussian alçak geçiren filtre ile topoğrafik/mikro alan ayrıştırması
- **Zengin Çıktılar**: CSV (long + wide format), JSON metadata ve PNG grafikler
- **Nodata Yönetimi**: Otomatik nodata maskeleme ve kenar hücre kontrolü

---

## Kurulum

### Gereksinimler

- **Python**: 3.10+ (test edildi: 3.12)
- **İşletim Sistemi**: Windows, Linux, macOS

### Bağımlılıklar

```
numpy       - Sayısal hesaplamalar
rasterio    - GeoTIFF okuma/yazma
scipy       - Gaussian filtre işlemleri
pandas      - Veri çerçevesi işlemleri
matplotlib  - Grafik oluşturma
pytest      - Test çerçevesi
```

### Kurulum Adımları

```bash
# 1. Repoyu klonlayın
git clone <repo-url>
cd yuzey_alani_hesaplama

# 2. Sanal ortam oluşturun (önerilir)
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux/macOS

# 3. Bağımlılıkları yükleyin
pip install -r requirements.txt
```

---

## Kullanım

### Temel Kullanım

```bash
python -m surface_area run --dem <DEM_DOSYASI> --outdir <CIKTI_KLASORU>
```

### Tam Özellikli Örnek

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

### CLI Parametreleri

| Parametre | Açıklama | Varsayılan |
|-----------|----------|------------|
| `--dem` | Girdi DEM/DSM GeoTIFF dosyası **(zorunlu)** | - |
| `--outdir` | Çıktı dizini **(zorunlu)** | - |
| `--gsd` | Hedef GSD listesi (metre) | `0.1, 0.5, 1, 2, 5, 10, 20, 50` |
| `--methods` | Çalıştırılacak yöntemler | Tümü |
| `--resampling` | Yeniden örnekleme metodu: `bilinear`, `nearest`, `cubic` | `bilinear` |
| `--nodata` | Nodata değeri (dataset'te tanımlı değilse) | Otomatik |
| `--slope_method` | Gradient kernel: `horn` veya `zt` | `horn` |
| `--jenness_weight` | Jenness yöntemi ağırlık katsayısı | `0.25` |
| `--integral_N` | Bilinear integral alt bölme sayısı (NxN) | `5` |
| `--sigma_mode` | Sigma modu: `mult` (GSD çarpanı) veya `m` (metre) | `mult` |
| `--sigma_m` | Multiscale sigma değerleri | `2.0, 5.0` |
| `--plots` | PNG grafik üretimi | Kapalı |
| `--keep_resampled` | Resample edilmiş GeoTIFF'leri sakla | Kapalı |
| `--reference_csv` | Karşılaştırma için referans CSV | - |

---

## Yöntemler

### 1. Jenness Window 8-Triangle (`jenness_window_8tri`)

3x3 komşuluk penceresinde merkez hücre etrafında **8 üçgen** oluşturur. Her üçgenin alanı **Heron formülü** ile hesaplanır.

```
  NW --- N --- NE
   |  \  |  /  |
   |   \ | /   |
  W ----[C]---- E    C = Merkez hücre
   |   / | \   |     8 üçgen: C-N-NE, C-NE-E, C-E-SE, ...
   |  /  |  \  |
  SW --- S --- SE
```

**Formül:**
```
A_cell = weight × Σ(Heron üçgen alanları)
Heron: A = √[s(s-a)(s-b)(s-c)]  where s = (a+b+c)/2
```

**Parametre:** `--jenness_weight` (varsayılan: 0.25)

---

### 2. TIN 2-Triangle Cell (`tin_2tri_cell`)

Her hücreyi **köşe noktaları** ile tanımlanan **2 üçgen** olarak modeller. Köşe yükseklikleri, komşu 4 hücre merkezinin ortalamasından türetilir.

```
  p00 -------- p10        Her hücre 2 üçgene bölünür:
   |  \        |          △1: p00-p10-p11
   |    \      |          △2: p00-p11-p01
   |      \    |
   |        \  |
  p01 -------- p11
```

**Formül (Cross Product):**
```
A = 0.5 × |v1 × v2|
```

---

### 3. Gradient Multiplier (`gradient_multiplier`)

Yerel eğim gradyanlarını (∂z/∂x, ∂z/∂y) kullanarak **alan çarpanı** hesaplar.

**Formül:**
```
A_cell = dx × dy × √(1 + p² + q²)

p = ∂z/∂x,  q = ∂z/∂y
```

**Gradient Kernelleri:**

| Kernel | Açıklama | Stencil |
|--------|----------|---------|
| **Horn** | 8 komşu ağırlıklı ortalama | 3x3 (tüm komşular) |
| **Zevenbergen-Thorne (ZT)** | 4 komşu basit fark | Cross (N,S,E,W) |

**Horn Kernel:**
```
∂z/∂x = [(NE + 2E + SE) - (NW + 2W + SW)] / (8×dx)
∂z/∂y = [(SW + 2S + SE) - (NW + 2N + NE)] / (8×dy)
```

---

### 4. Bilinear Patch Integral (`bilinear_patch_integral`)

Her hücreyi **bilinear yüzey** olarak modeller ve **NxN alt bölme** ile sayısal integrasyon yapar.

```
  +-------+-------+
  |       |       |     N=2 örneği:
  |   △   |   △   |     4 alt hücre × 2 üçgen = 8 üçgen
  +-------+-------+
  |       |       |
  |   △   |   △   |
  +-------+-------+
```

**Bilinear İnterpolasyon:**
```
z(u,v) = (1-u)(1-v)×z00 + u(1-v)×z10 + (1-u)v×z01 + uv×z11
```

**Parametre:** `--integral_N` (varsayılan: 5, yani 5×5=25 alt hücre)

---

### 5. Multiscale Decomposed Area (`multiscale_decomposed_area`)

**Gaussian alçak geçiren filtre** ile yüzey alanını **topoğrafik** ve **mikro-pürüzlülük** bileşenlerine ayırır.

```
A_total = A_topo + A_micro

A_total : Toplam 3D yüzey alanı (gradient multiplier)
A_topo  : Düzleştirilmiş (low-pass) yüzeyin alanı
A_micro : Mikro-pürüzlülük katkısı
```

**Düzleştirme:**
- Nodata-aware **normalized convolution** kullanılır
- `σ` (sigma) parametresi düzleştirme ölçeğini kontrol eder

**Parametreler:**
- `--sigma_mode mult`: Sigma = GSD × değer
- `--sigma_mode m`: Sigma = mutlak metre değeri
- `--sigma_m`: Sigma değerleri listesi

---

## Çıktılar

`--outdir` altında oluşturulan dosyalar:

### CSV Dosyaları

#### `results_long.csv`
Her satır bir (GSD, method) kombinasyonunu temsil eder.

| Kolon | Açıklama |
|-------|----------|
| `gsd_m` | Hedef GSD (metre) |
| `dx`, `dy` | Gerçek piksel boyutları |
| `method` | Hesaplama yöntemi |
| `A2D` | Planimetrik alan (m²) = valid_cells × dx × dy |
| `A3D` | 3D yüzey alanı (m²) |
| `ratio` | Alan oranı = A3D / A2D |
| `valid_cells` | Geçerli hücre sayısı |
| `runtime_sec` | Hesaplama süresi (saniye, IO hariç) |
| `note` | Parametre özeti |

**Multiscale için ek kolonlar:**
| Kolon | Açıklama |
|-------|----------|
| `a_topo` | Topoğrafik alan bileşeni |
| `a_micro` | Mikro-pürüzlülük bileşeni |
| `micro_ratio` | A_micro / A_total |
| `sigma_m` | Kullanılan sigma değeri (metre) |

#### `results_wide.csv`
Satır = GSD, Sütunlar = `{method}_{metric}` formatında pivot tablo.

### Metadata

#### `run_info.json`
```json
{
  "timestamp_utc": "2024-01-15T10:30:00+00:00",
  "dem": "dag_dsm.tif",
  "dem_info": {
    "path": "dag_dsm.tif",
    "crs": "EPSG:32636",
    "width": 1000,
    "height": 800,
    "nodata": -9999.0,
    "dx": 0.5,
    "dy": 0.5
  },
  "versions": {
    "python": "3.12.0",
    "surface_area": "0.1.0",
    "numpy": "1.26.0",
    "rasterio": "1.3.9"
  },
  "params": { ... }
}
```

### Grafikler (`--plots`)

| Dosya | Açıklama |
|-------|----------|
| `A3D_vs_GSD.png` | 3D yüzey alanı vs GSD (log ölçek) |
| `ratio_vs_GSD.png` | A3D/A2D oranı vs GSD |
| `micro_ratio_vs_GSD.png` | Mikro oran vs GSD (multiscale) |

---

## Teknik Detaylar

### Nodata ve Kenar Yönetimi

| Durum | Davranış |
|-------|----------|
| Nodata hücreler | Maskelenir, hesaplamaya dahil edilmez |
| Stencil tabanlı yöntemler (Horn/ZT, Jenness) | Tam stencil valid değilse hücre atlanır |
| Köşe tabanlı yöntemler (TIN, Bilinear) | 4 geçerli hücre merkezinden türetilmediğinde köşe atlanır |
| Raster kenarları | Dış 1 hücre sınırı otomatik olarak dışlanır |

### CRS ve Birim Uyarıları

- Tüm hesaplamalar DEM'in CRS linear biriminde yapılır
- CRS metre değilse CLI uyarı verir
- Derece bazlı CRS'lerde GSD ve alan değerleri anlamsız olabilir

### Bellek Yönetimi

- Büyük rasterlar `rasterio.block_windows` ile blok-blok işlenir
- Her blok için overlap (örtüşme) hesaplanır
- Multiscale için overlap = `ceil(4 × max_sigma_px) + 1`

---

## Proje Yapısı

```
yuzey_alani_hesaplama/
├── surface_area/
│   ├── __init__.py      # Paket tanımı, versiyon
│   ├── __main__.py      # Entry point
│   ├── cli.py           # Komut satırı arayüzü
│   ├── io.py            # Raster I/O işlemleri
│   ├── methods.py       # Yüzey alanı algoritmaları
│   ├── multiscale.py    # Multiscale ayrıştırma
│   ├── plotting.py      # Grafik fonksiyonları
│   └── synthetic.py     # Sentetik test yüzeyleri
├── tests/
│   ├── conftest.py      # Test konfigürasyonu
│   └── test_synthetic.py # Birim testleri
├── .githooks/
│   └── pre-commit       # Git hook'ları
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Testler

### Test Çalıştırma

```bash
# Tüm testleri çalıştır
pytest -q

# Detaylı çıktı
pytest -v

# Belirli bir test
pytest tests/test_synthetic.py::test_plane_all_methods_high_accuracy
```

### Test Kapsamı

`test_synthetic.py` sentetik yüzeyler üzerinde yöntemlerin doğruluğunu test eder:

| Yüzey | Açıklama | Tolerans |
|-------|----------|----------|
| **Plane** | z = ax + by + c | < 0.1% hata |
| **Sinusoid** | z = A×sin(kx×x)×sin(ky×y) | < 5% hata |
| **Paraboloid** | z = (x² + y²) / scale | < 5% hata |

**Referans alan hesabı:** Yüksek çözünürlüklü (dx/10, dy/10) iki-üçgen integrasyon

---

## Performans İpuçları

1. **Büyük DEM'ler için**: Önce daha kaba GSD'lerle (2-50m) test edin
2. **Upsample dikkat**: Kaynak çözünürlükten daha küçük GSD çıktıyı çok büyütebilir
3. **Multiscale**: `--sigma_mode mult` genellikle daha tutarlı sonuç verir
4. **Bellek**: `--keep_resampled` kapalı tutun (varsayılan)

---

## Sürüm Geçmişi

- **v0.1.0** - İlk sürüm
  - 5 yüzey alanı hesaplama yöntemi
  - Multiscale ayrıştırma
  - CLI arayüzü
  - CSV/JSON/PNG çıktıları

---

## Lisans

Bu proje açık kaynak olarak sunulmaktadır.

---

## Kaynaklar

- Jenness, J. S. (2004). Calculating landscape surface area from digital elevation models. *Wildlife Society Bulletin*, 32(3), 829-839.
- Horn, B. K. (1981). Hill shading and the reflectance map. *Proceedings of the IEEE*, 69(1), 14-47.
- Zevenbergen, L. W., & Thorne, C. R. (1987). Quantitative analysis of land surface topography. *Earth Surface Processes and Landforms*, 12(1), 47-56.
