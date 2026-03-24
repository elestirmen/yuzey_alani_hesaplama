<p align="center">
  <img src="https://img.icons8.com/fluency/96/mountain.png" alt="DEM 3D Surface" width="80"/>
</p>

<h1 align="center">DEM 3D Yüzey Alanı Hesaplama Aracı</h1>

<p align="center">
  <strong>DEM/DSM GeoTIFF verilerinden 3D yüzey alanı (A3D) hesaplama için kapsamlı Python kütüphanesi</strong>
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>
  <a href="#lisans"><img src="https://img.shields.io/badge/Lisans-Açık%20Kaynak-28A745?style=for-the-badge" alt="License"></a>
  <a href="#"><img src="https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-blue?style=for-the-badge" alt="Platform"></a>
</p>

<p align="center">
  <a href="#-hızlı-başlangıç">Hızlı Başlangıç</a> •
  <a href="#-özellikler">Özellikler</a> •
  <a href="#-kurulum">Kurulum</a> •
  <a href="#-kullanım">Kullanım</a> •
  <a href="#toplu-işlem-batch">Toplu işlem</a> •
  <a href="#-yöntemler">Yöntemler</a> •
  <a href="#-çıktılar">Çıktılar</a>
</p>

---

## 📋 İçindekiler

- [🚀 Hızlı Başlangıç](#-hızlı-başlangıç)
- [✨ Özellikler](#-özellikler)
- [🔄 İş Akışı](#-iş-akışı)
- [📦 Kurulum](#-kurulum)
  - [Gereksinimler](#gereksinimler)
  - [Adım Adım Kurulum](#adım-adım-kurulum)
  - [Bağımlılıklar](#bağımlılıklar)
  - [Kurulum Sorun Giderme](#kurulum-sorun-giderme)
- [💻 Kullanım](#-kullanım)
  - [IDE / `main.py` yapılandırması](#yöntem-1-idemainpy-yapılandırması-önerilen)
  - [Komut satırı](#yöntem-2-komut-satırı)
  - [Toplu işlem (batch)](#toplu-işlem-batch)
  - [Sentetik DSM](#sentetik-dsm-metot-kıyaslama)
  - [VS Code ile çalıştırma](#yöntem-3-vs-code-ile-çalıştırma)
- [⚙️ Parametreler](#️-parametreler)
  - [Zorunlu Parametreler](#zorunlu-parametreler)
  - [İsteğe Bağlı Parametreler](#isteğe-bağlı-parametreler)
  - [Parametre Detayları](#parametre-detayları)
- [🔬 Yöntemler](#-yöntemler)
- [📊 Çıktılar](#-çıktılar)
- [🛠️ Teknik Detaylar](#️-teknik-detaylar)
- [🧪 Testler](#-testler)
- [⚡ Performans İpuçları](#-performans-ipuçları)
- [📚 Kaynaklar](#-kaynaklar)
- [📄 Lisans](#-lisans)

---

## 🚀 Hızlı Başlangıç

En basit kullanım için:

```bash
# 1. Depoyu klonlayın
git clone <repo-url>
cd yuzey_alani_hesaplama

# 2. Bağımlılıkları yükleyin
pip install -r requirements.txt

# 3. Tek bir GeoTIFF için
python main.py run --dem dem_dosyam.tif --outdir sonuclar
```

Eşdeğer olarak paket üzerinden: `python -m surface_area run --dem ... --outdir ...`

`main.py` üstündeki **`config`** sözlüğünü düzenleyip parametreleri tek yerden yönetebilir, ardından argümansız çalıştırabilirsiniz:

```bash
python main.py
```

---

## ✨ Özellikler

<table>
<tr>
<td width="50%">

### 🎯 Hesaplama
- **7 farklı algoritma** ile yüzey alanı hesaplama
- **Çoklu çözünürlük analizi** (GSD desteği)
- **Multiscale ayrıştırma** (topoğrafik + mikro)
- **ROI/Parsel desteği** (GeoJSON/Shapefile)

</td>
<td width="50%">

### 📈 Çıktılar
- **Excel (`.xlsx`)** formatında sonuçlar (sheet'ler halinde)
- **JSON** metadata ve run bilgisi
- **PNG** grafikler (A3D vs GSD, ratio vb.)
- **Ground truth** referans değerler

</td>
</tr>
<tr>
<td>

### ⚡ Performans
- **Blok işleme** ile büyük dosya desteği (`rasterio.block_windows`)
- **Bellek-etkin** raster işleme
- **Nodata yönetimi** ve kenar hücre kontrolü

</td>
<td>

### 🛠️ Sentetik DSM üretimi
- **Raster tabanlı preset'ler** (10 gerçekçi arazi + 6 test pattern) ve **analitik benchmark yüzeyleri** (`analytic_*` ailesi)
- **fBm / ridge / turbulence** gürültüleri ve **erozyon** (hidrolik + termal)
- Yan dosya **`.reference.json`**: native-grid ve (uygun preset'lerde) sürekli analitik referans alanlar

</td>
</tr>
<tr>
<td colspan="2">

### 📦 Toplu çalıştırma ve performans
- **Toplu girdi:** klasördeki tüm `.tif/.tiff` dosyaları veya `.demlist` listesi; her DEM için alt klasör + isteğe bağlı `batch_summary.xlsx`
- **Paralel blok işleme:** `--workers` (CLI varsayılanı `1`; `main.py` içinde `config["workers"]` ile yükseltilebilir)
- **Native ızgara:** `--gsd native` ile yeniden örnekleme yapmadan kaynak piksel boyutunda hesap
- **İstem dışı upsample:** varsayılan olarak kaynak ızgaradan **daha ince** hedef GSD reddedilir; `--allow-upsample` ile açılır

</td>
</tr>
</table>

---

## 🔄 İş Akışı

### Ana İşlem Akışı

```mermaid
flowchart TB
    subgraph INPUT["📥 Girdi"]
        DEM[("🗺️ DEM/DSM<br/>GeoTIFF")]
        SYNTH["🏔️ Sentetik DSM<br/>(Opsiyonel)"]
        ROI["📍 ROI<br/>GeoJSON/Shapefile"]
    end

    subgraph PROCESS["⚙️ İşleme"]
        direction TB
        RESAMPLE["📐 Yeniden Örnekleme<br/>(Hedef GSD'lere)"]
        METHODS["🔬 Yöntemler"]
        
        subgraph METHODS_DETAIL["Hesaplama Yöntemleri"]
            M1["Gradient Multiplier"]
            M2["TIN 2-Triangle"]
            M3["Jenness 8-Triangle"]
            M4["Sector Adaptive Jenness"]
            M5["Bilinear Integral"]
            M6["Adaptive Bilinear"]
            M7["Multiscale Decomposed"]
        end
    end

    subgraph OUTPUT["📤 Çıktı"]
        XLSX["📊 Excel<br/>results.xlsx (+ batch_summary)"]
        JSON["📋 JSON<br/>run_info.json"]
        PLOTS["📈 PNG<br/>A3D, oran, runtime, hata…"]
        ROI_OUT["📍 ROI<br/>results_roi_long"]
    end

    DEM --> RESAMPLE
    SYNTH -.-> DEM
    RESAMPLE --> METHODS
    METHODS --> METHODS_DETAIL
    ROI -.-> METHODS
    
    METHODS_DETAIL --> XLSX
    METHODS_DETAIL --> JSON
    METHODS_DETAIL --> PLOTS
    ROI --> ROI_OUT

    style INPUT fill:#e1f5fe
    style PROCESS fill:#fff3e0
    style OUTPUT fill:#e8f5e9
    style METHODS_DETAIL fill:#fce4ec
```

### Sentetik DSM Üretim Akışı

```mermaid
flowchart LR
    subgraph PRESETS["🎨 Preset seçimi"]
        direction TB
        REAL["🏔️ Gerçekçi arazi<br/>10 preset"]
        TEST["🔬 Test pattern<br/>6 preset"]
        ANA["📐 Analitik benchmark<br/>analytic_* · 8 preset"]
    end

    subgraph GENERATION["⚙️ Üretim Aşamaları"]
        direction TB
        FBM["1️⃣ fBm Noise<br/>(Temel topografya)"]
        RIDGE["2️⃣ Ridge Noise<br/>(Sırtlar ve tepeler)"]
        EROSION["3️⃣ Erozyon Sim.<br/>(Vadiler)"]
        BUMPS["4️⃣ Gaussian Bumps<br/>(Detaylar)"]
    end

    subgraph OUTPUTS["📤 Çıktı"]
        TIFF["🗺️ GeoTIFF"]
        REF["📊 .reference.json<br/>native + sürekli GT"]
    end

    ANA -.-> REF
    PRESETS --> GENERATION
    FBM --> RIDGE --> EROSION --> BUMPS
    GENERATION --> OUTPUTS

    style PRESETS fill:#e3f2fd
    style GENERATION fill:#fff8e1
    style OUTPUTS fill:#e8f5e9
```

---

## 📦 Kurulum

### Gereksinimler

| Gereksinim | Minimum | Önerilen |
|:----------:|:-------:|:--------:|
| 🐍 **Python** | 3.10 | 3.12 |
| 💾 **RAM** | 4 GB | 8+ GB (büyük DEM'ler için) |
| 💿 **Disk** | 100 MB | Veri boyutuna bağlı |
| 🖥️ **İşletim Sistemi** | Windows 10, Linux (Ubuntu 20.04+), macOS 11+ | - |

### Adım Adım Kurulum

#### 1️⃣ Repository'yi Klonlayın

```bash
git clone <repo-url>
cd yuzey_alani_hesaplama
```

#### 2️⃣ Sanal Ortam Oluşturun (Önerilir)

**🪟 Windows (PowerShell):**
```powershell
# Sanal ortam oluştur
python -m venv .venv

# Sanal ortamı etkinleştir
.venv\Scripts\Activate.ps1

# Not: ExecutionPolicy hatası alırsanız önce şunu çalıştırın:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**🪟 Windows (CMD):**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**🐧 Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### 3️⃣ Bağımlılıkları Yükleyin

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 4️⃣ Kurulumu Doğrulayın

```bash
# Paket sürümü
python -c "import surface_area; print(f'surface_area v{surface_area.__version__}')"

# main.py üst düzey yardım (config anahtarları + metot ön ayarları)
python main.py --help

# Tam CLI (run / synth alt komutları)
python -m surface_area --help
python -m surface_area run --help
```

### Bağımlılıklar

`requirements.txt` içeriği (sürüm alt sınırları):

| Paket | Amaç |
|:-----:|:-----|
| `numpy` | Sayısal diziler ve vektörize hesap |
| `numba` | JIT ile hızlandırılmış çekirdekler |
| `rasterio` | GeoTIFF okuma/yazma, CRS, blok pencereleri |
| `scipy` | Gaussian filtre, erozyon ve bilimsel yardımcılar |
| `pandas` | Sonuç tabloları |
| `openpyxl` | Excel (`.xlsx`) üretimi |
| `matplotlib` | PNG grafikleri |
| `shapely` | ROI geometrisi (GeoJSON yolu) |

**Testler** için ayrıca `pytest` kurulmalıdır (`pip install pytest`).

**Manuel kurulum (tek satır):**
```bash
pip install "numpy>=2.0" "numba>=0.61" "rasterio>=1.4" "scipy>=1.14" "pandas>=2.2" "openpyxl>=3.1" "matplotlib>=3.9" "shapely>=2.0"
```

> **Not (ROI Shapefile):** `.shp` ve benzeri OGR formatları için ortamınızda `fiona` / `geopandas` gerekir. Yalnızca GeoJSON kullanıyorsanız `shapely` yeterlidir.

### Kurulum Sorun Giderme

#### 🔴 Windows'ta rasterio Kurulum Hatası

Rasterio, GDAL kütüphanesine bağımlıdır. Windows'ta sorun yaşarsanız:

```powershell
# Conda ile kurulum (önerilir)
conda install -c conda-forge rasterio

# veya wheel dosyasından kurulum
pip install --find-links=https://github.com/cgohlke/geospatial-wheels/releases rasterio
```

#### 🔴 Permission Hatası (Linux/macOS)

```bash
# --user flag'i ile kur
pip install --user -r requirements.txt

# veya sudo ile (önerilmez)
sudo pip install -r requirements.txt
```

#### 🔴 SSL/Network Hatası

```bash
# Trusted host ekleyerek kur
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
```

---

## 💻 Kullanım

### Yöntem 1: IDE / `main.py` yapılandırması (önerilen)

`main.py` dosyasının üstündeki **`config`** sözlüğünü düzenleyin. Bu sözlük `RunConfig` ile birleştirilir; `method_choice` + `methods`, `gsd` listesi (`"native"` veya sayılar), `allow_upsample`, `workers` vb. buradan yönetilir.

Özet alanlar:

| Alan | Anlamı |
|:-----|:-------|
| `dem` | Tek GeoTIFF, `.demlist` dosyası veya `.tif` içeren klasör |
| `outdir` | Çıktı kökü (toplu girdide her DEM alt klasör) |
| `gsd` | Örn. `["native", 1, 2, 5]` — `native` kaynak ızgarada hesap |
| `method_choice` | `default`, `fast`, `balanced`, `jenness_focus`, `full` |
| `methods` | `None` ise `method_choice` kullanılır; dolu liste verilirse preset yok sayılır |
| `allow_upsample` | Kaynak çözünürlükten ince hedef GSD’lere izin |
| `workers` | Blok hesap için işlem sayısı |

Ardından:

```bash
python main.py
```

### Yöntem 2: Komut satırı

```bash
python main.py run ^
  --dem dag_dsm.tif ^
  --outdir out ^
  --gsd native 0.5 1 2 5 10 ^
  --methods gradient_multiplier tin_2tri_cell ^
  --plots ^
  --workers 4
```

`--gsd` verilmezse CLI, `native` + çoklu metre değerlerini varsayılan listeyle doldurur. Kaynak pikselden **daha ince** hedefler için `--allow-upsample` gerekir (`--no-allow-upsample` varsayılan).

### Toplu işlem (batch)

Aynı parametrelerle birden fazla DEM çalıştırmak için:

1. **Klasör:** `--dem` olarak yalnızca `.tif` / `.tiff` içeren bir dizin verin. Dosyalar ada göre sıralanır; her biri için `outdir/<stem>/` altına sonuç yazılır (aynı kök ada sahip birden fazla dosya varsa çakışmayı önlemek için sonek eklenir).
2. **Liste dosyası:** `--dem path/to/liste.demlist` — satır başına bir GeoTIFF yolu (`#` ile yorum, boş satırlar atlanır).

İki veya daha fazla DEM işlendiğinde, `outdir` köküne ek olarak **`batch_summary.xlsx`** üretilir (tüm çalıştırmaların özet karşılaştırması).

```bash
python main.py run --dem ./dem_klasoru --outdir ./sonuclar --gsd native 1 2 --plots
```

### Sentetik DSM (Metot Kıyaslama)

Yöntemleri gerçek DEM'lere geçmeden önce **kontrollü** yüzeyler üzerinde doğrulamak/kıyaslamak için sentetik DSM/DEM üretebilirsiniz. İki üretim yolu vardır:

1. **Hızlı üretim (CLI):** `python -m surface_area synth ...` sadece GeoTIFF üretir.
2. **Benchmark / native-grid referans:** `generate_synthetic_tif.py` GeoTIFF + **native çözünürlükte** referans A2D/A3D hesaplar ve `.reference.json` üretir.

#### 1️⃣ Hızlı Üretim: `surface_area synth`

```bash
# 1) Sentetik DSM üret (patchwork = farklı desenlerin karışımı)
python -m surface_area synth \
  --out synthetic_patchwork.tif \
  --preset patchwork \
  --rows 512 --cols 512 \
  --dx 1 --seed 0 \
  --nodata_holes 5

# 2) Metotları çalıştır
python -m surface_area run \
  --dem synthetic_patchwork.tif \
  --outdir out_synth \
  --gsd 1 2 5 10 \
  --methods jenness_window_8tri sector_adaptive_jenness_integral tin_2tri_cell gradient_multiplier bilinear_patch_integral adaptive_bilinear_patch_integral multiscale_decomposed_area \
  --plots
```

#### 2️⃣ Preset'ler

CLI `synth --preset` ve `generate_synthetic_tif.py --preset` için tüm seçenekler `surface_area.synthetic.SYNTHETIC_PRESETS` içinde birleştirilir: **16 raster-öncelikli** preset + **8 analitik** preset.

**🗺️ Gerçekçi arazi** (saha benzeri testler):

| Preset | Açıklama |
|:------:|:---------|
| `mountain` | Dağlık arazi (fBm noise + sırtlar, erozyon) |
| `valley` | Vadi ve akarsu yatakları |
| `hills` | Yumuşak tepeler (rolling hills) |
| `coastal` | Kıyı şeridi (deniz-kara geçişi) |
| `plateau` | Yüksek plato ve yamaçlar |
| `canyon` | Kanyon/boğaz yapıları |
| `volcanic` | Volkanik arazi (kraterler, lav akışı) |
| `glacial` | Buzul vadisi (U-şekilli) |
| `karst` | Karstik arazi (düdenler, mağaralar) |
| `alluvial` | Alüvyal ova/delta |

**🔬 Test pattern'leri** (hızlı görsel/karşılaştırma):

| Preset | Açıklama |
|:------:|:---------|
| `plane` | Düz eğimli yüzey |
| `waves` | Sinüzoidal dalgalar |
| `crater_field` | Krater alanı |
| `terraced` | Teraslı arazi |
| `patchwork` | Karışık yüzeyler |
| `mixed` | Maksimum çeşitlilik |

**📐 Analitik benchmark'lar** (`.reference.json` içinde sürekli yüzey üzerinde **continuous ground truth** alanı; metot hataları için ideal):

| Preset | Açıklama |
|:------:|:---------|
| `analytic_plane` | Düzlem |
| `analytic_tilted_plane` | Eğik düzlem |
| `analytic_sinusoidal` | Sinüs yüzeyi |
| `analytic_gaussian_hill` | Tek Gauss tepesi |
| `analytic_multi_gaussian` | Çoklu Gauss |
| `analytic_saddle` | Eyer yüzeyi |
| `analytic_dome` | Kubbe |
| `analytic_hybrid_multiscale` | Çok ölçekli karma |

#### 3️⃣ Önemli Parametreler

| Parametre | Açıklama |
|:---------:|:---------|
| `--dx`, `--dy` | Piksel boyutu (metre). Daha küçük değer = daha fazla detay + daha büyük dosya/bellek. |
| `--relief` | Makro rölyef çarpanı (yamaçları/zirveleri büyütür). |
| `--roughness_m` | Mikro pürüzlülük genliği (metre). Mikro-topografya etkisini artırır. |
| `--seed` | Tekrarlanabilirlik için sabit seed. |
| `--nodata_holes`, `--nodata_radius_m` | Nodata delikleri oluşturarak nodata/kenar davranışını test eder. |

#### 4️⃣ Native-Grid Referans Alan: `generate_synthetic_tif.py`

`generate_synthetic_tif.py`, aynı sentetik yüzeyi üretip GeoTIFF'in yanına `.reference.json` yazar: **native-grid** referans alanlar, çoklu çözünürlük özetleri ve (analitik preset'lerde) **sürekli yüzey üzerinde integral** ile elde edilen `continuous_ground_truth` alanları.

`--out` parametresi `{preset}`, `{rows}`, `{cols}`, `{dx}`, `{seed}`, `{timestamp}` gibi şablonları destekler.

> Raster tabanlı üretimlerde native-grid referansı çözünürlüğe bağlıdır; analitik preset'lerde ise sürekli GT GSD'den bağımsızdır.

```bash
# Sentetik DSM + native-grid referans alan
python generate_synthetic_tif.py \
  --out out_synth/synth_mountain_dx1_seed42.tif \
  --preset mountain \
  --rows 2048 --cols 2048 \
  --dx 1 --seed 42 \
  --nodata_holes 20

# Referans: out_synth/synth_mountain_dx1_seed42.reference.json
# İçerik: surface_area_m2, planar_area_m2, surface_ratio, valid_cells, nodata_cells

# Kıyas: gsd=1 ile çalıştırıp results.xlsx içindeki results_long sayfasındaki A3D'yi JSON'daki surface_area_m2 ile karşılaştırın.
python -m surface_area run \
  --dem out_synth/synth_mountain_dx1_seed42.tif \
  --outdir out_synth_run \
  --gsd 1 \
  --methods jenness_window_8tri sector_adaptive_jenness_integral tin_2tri_cell gradient_multiplier bilinear_patch_integral adaptive_bilinear_patch_integral \
  --plots
```

> 💡 **Not:** Aynı komutları `python main.py synth ...` / `python main.py run ...` şeklinde de kullanabilirsiniz.

### Yöntem 3: VS Code ile Çalıştırma

Bu repoda hazır **VS Code** çalıştırma ayarları bulunur:

1. `Run and Debug (Ctrl+Shift+D)` → **SurfaceArea: main.py** seçin
2. Python interpreter olarak `.venv` seçin (`Ctrl+Shift+P` → `Python: Select Interpreter`)
3. `F5` ile çalıştırın

Bağımlılıkları kurmak için: `Terminal → Run Task…` → **SurfaceArea: install deps (venv)**

### Yardım Mesajı

```bash
python main.py --help
```

---

## ⚙️ Parametreler

### Zorunlu Parametreler

| Parametre | Kısa | Tip | Açıklama |
|:---------:|:----:|:---:|:---------|
| `--dem` | - | `str` | Girdi DEM/DSM GeoTIFF dosya yolu. Mutlak veya göreli yol olabilir. |
| `--outdir` | - | `str` | Çıktı dizini. Mevcut değilse otomatik oluşturulur. |

**Örnek:**
```bash
python main.py run --dem C:\data\dem.tif --outdir C:\results
```

### İsteğe Bağlı Parametreler

| Parametre | Tip | Varsayılan | Açıklama |
|:---------:|:---:|:----------:|:---------|
| `--gsd` | `list` | `native`, sonra `0.1 … 50` (sayılar metre) | Hedef GSD listesi; `native` = kaynak piksel boyutunda hesap (yeniden örnekleme yok) |
| `--allow-upsample` | bool | `False` | `True` ise kaynak ızgaradan **daha ince** hedef GSD’lere izin |
| `--methods` | `list[str]` | `jenness_window_8tri`, `sector_adaptive_jenness_integral`, `tin_2tri_cell`, `gradient_multiplier` | Çalıştırılacak hesaplama yöntemleri |
| `--workers` | `int` | `1` | Blok bazlı hesap için paralel işlem sayısı |
| `--resampling` | `str` | `bilinear` | Yeniden örnekleme algoritması |
| `--nodata` | `float` | Otomatik | Nodata değeri (dataset'te tanımlı değilse) |
| `--slope_method` | `str` | `horn` | Gradient/eğim hesaplama kerneli |
| `--jenness_weight` | `float` | `0.25` | Jenness yöntemi ağırlık katsayısı |
| `--integral_N` | `int` | `5` | Bilinear integral alt bölme sayısı |
| `--adaptive_rel_tol` | `float` | `1e-4` | Adaptive bilinear bağıl tolerans |
| `--adaptive_abs_tol` | `float` | `0.0` | Adaptive bilinear mutlak tolerans |
| `--adaptive_max_level` | `int` | `5` | Adaptive bilinear maksimum inceltme seviyesi |
| `--adaptive_min_N` | `int` | `2` | Adaptive bilinear başlangıç N |
| `--adaptive_roughness_fastpath` | `bool` | Açık | Düz/planar patch fast-path |
| `--adaptive_roughness_threshold` | `float` | Otomatik | Fast-path eşiği (opsiyonel) |
| `--sector_jenness_rel_tol` | `float` | `1e-4` | Sector adaptive Jenness bağıl tolerans |
| `--sector_jenness_abs_tol` | `float` | `0.0` | Sector adaptive Jenness mutlak tolerans |
| `--sector_jenness_max_level` | `int` | `5` | Sector adaptive Jenness maksimum inceltme seviyesi (`level-0`=bölmesiz, `level-1`=4 alt üçgen, `level-2`=16, ...) |
| `--sector_jenness_min_samples` | `int` | `3` | Sector adaptive Jenness için minimum triangle quadrature eşiği |
| `--sigma_mode` | `str` | `mult` | Multiscale sigma yorumlama modu |
| `--sigma_m` | `list[float]` | `2.0, 5.0` | Multiscale sigma değerleri |
| `--roi` | `str` | - | ROI polygon yolu (GeoJSON veya Shapefile) |
| `--roi_id_field` | `str` | Otomatik | ROI id alanı |
| `--roi_mode` | `str` | `mask` | ROI modu: `mask` veya `fraction` |
| `--roi_all_touched` | `flag` | Kapalı | `mask` modunda all_touched rasterize |
| `--roi_only` | `flag` | Kapalı | Sadece ROI çıktısını üret |
| `--plots` | `flag` | Kapalı | PNG grafik üretimini etkinleştirir |
| `--keep_resampled` | `flag` | Kapalı | Resample edilmiş GeoTIFF'leri saklar |
| `--reference_csv` | `str` | - | Karşılaştırma için referans CSV dosyası |

### Parametre Detayları

#### 📏 `--gsd` (Ground Sample Distance)

Hedef çözünürlükleri belirler. Liste **hem** özel anahtar kelime **`native`** **hem** metre cinsinden pozitif sayılar içerebilir. Her sayısal GSD için raster yeniden örneklenir ve seçilen tüm metotlar o hedef için tekrar çalıştırılır; `native` ile yalnızca kaynak ızgarada hesap yapılır.

| Değer | Açıklama | Kullanım senaryosu |
|:-----:|:---------|:-------------------|
| `native` | Yeniden örnekleme yok; DEM’in gerçek `dx`/`dy` değerleri | Hızlı referans, “orijinal grid” karşılaştırması |
| `< 1` m | Alt-metre çözünürlük | Mikro-topografya (çoğu zaman upsample gerektirir) |
| `1–5` m | Yüksek çözünürlük | Tipik DEM analizi |
| `5–20` m | Orta çözünürlük | Bölgesel özet |
| `> 20` m | Düşük çözünürlük | Geniş alan, trend |

```bash
# Native ızgara + birkaç metre hedefi
--gsd native 0.5 1 2 5 10

# Sadece tek hedef
--gsd 1
```

> **Upsample:** Kaynak piksel boyutundan **daha ince** hedef GSD, varsayılan olarak **reddedilir**; bilinçli upsample için `--allow-upsample` kullanın.

---

#### 🔬 `--methods` (Hesaplama Yöntemleri)

Kullanılabilir yöntemler:

| Yöntem | Açıklama | Hız | Doğruluk |
|:-------|:---------|:---:|:--------:|
| `gradient_multiplier` | Gradient tabanlı alan çarpanı | ⚡⚡⚡ Çok hızlı | Yüksek |
| `tin_2tri_cell` | Her hücre 2 üçgen olarak modellenir | ⚡⚡ Hızlı | Yüksek |
| `jenness_window_8tri` | 3x3 pencerede 8 üçgen | ⚡⚡ Hızlı | Çok yüksek |
| `sector_adaptive_jenness_integral` | 3x3 quadratic fit + hücre içi 8 sektör integrali | ⚡ Orta | Çok yüksek |
| `bilinear_patch_integral` | Bilinear yüzey integrasyonu | ⚡ Yavaş | En yüksek |
| `adaptive_bilinear_patch_integral` | Bilinear integral (adaptif inceltme) | ⚡ Yavaş | En yüksek |
| `multiscale_decomposed_area` | Çok ölçekli ayrıştırma | ⚡ Yavaş | Özel |

```bash
# Tek yöntem
--methods gradient_multiplier

# Çoklu yöntem
--methods gradient_multiplier tin_2tri_cell jenness_window_8tri sector_adaptive_jenness_integral

# Varsayılan yöntem seti (--methods belirtilmezse)
# jenness_window_8tri, sector_adaptive_jenness_integral, tin_2tri_cell, gradient_multiplier
```

> Not: `multiscale_decomposed_area` artık varsayılan sette değildir.
> Topo/mikro ayrıştırma gerektiğinde `--methods multiscale_decomposed_area` ile açıkça ekleyin.
> `bilinear_patch_integral` ve `adaptive_bilinear_patch_integral` artık varsayılan sette değildir.
> Bunları ek benchmark / appendix karşılaştırması için açıkça seçin.

---

#### 🔄 `--resampling` (Yeniden Örnekleme)

DEM'i farklı çözünürlüklere dönüştürürken kullanılan interpolasyon yöntemi:

| Değer | Açıklama | Önerilen Kullanım |
|:-----:|:---------|:------------------|
| `bilinear` | Bilinear interpolasyon (4 komşu) | **Varsayılan**, çoğu durum için ideal |
| `nearest` | En yakın komşu (interpolasyon yok) | Kategorik veriler, tam değer koruma |
| `cubic` | Kübik konvolüsyon (16 komşu) | Yumuşak geçişler, görsel kalite |

```bash
--resampling bilinear
--resampling nearest
--resampling cubic
```

---

#### 📐 `--slope_method` (Eğim Hesaplama Kerneli)

Gradient/eğim hesaplaması için kullanılan kernel:

| Değer | Tam Adı | Stencil | Açıklama |
|:-----:|:--------|:-------:|:---------|
| `horn` | Horn (1981) | 3x3 (8 komşu) | Ağırlıklı ortalama, gürültüye dayanıklı |
| `zt` | Zevenbergen-Thorne (1987) | Cross (4 komşu) | Basit fark, daha hızlı |

**Horn Kernel Formülü:**
```
∂z/∂x = [(NE + 2E + SE) - (NW + 2W + SW)] / (8×dx)
∂z/∂y = [(SW + 2S + SE) - (NW + 2N + NE)] / (8×dy)
```

**Zevenbergen-Thorne Formülü:**
```
∂z/∂x = (E - W) / (2×dx)
∂z/∂y = (S - N) / (2×dy)
```

```bash
--slope_method horn   # Varsayılan, önerilen
--slope_method zt     # Daha hızlı alternatif
```

---

#### ⚖️ `--jenness_weight` (Jenness Ağırlık Katsayısı)

`jenness_window_8tri` yöntemi için üçgen alanlarının toplama katsayısı.

| Değer | Açıklama |
|:-----:|:---------|
| `0.25` | **Varsayılan** - Her üçgenin 1/4'ü merkez hücreye atanır |
| `0.125` | Daha konservatif hesaplama |
| `0.5` | Daha agresif hesaplama |

```bash
--jenness_weight 0.25
```

> 💡 **Not:** Literatürde yaygın olarak 0.25 değeri kullanılır (Jenness, 2004).

---

#### 🔢 `--integral_N` (Bilinear Alt Bölme)

`bilinear_patch_integral` yöntemi için her hücrenin kaç alt hücreye bölüneceğini belirler.

| Değer | Alt Hücre | Üçgen Sayısı | Performans |
|:-----:|:---------:|:------------:|:-----------|
| `1` | 1×1 = 1 | 2 | Çok hızlı, düşük doğruluk |
| `5` | 5×5 = 25 | 50 | **Varsayılan**, dengeli |
| `10` | 10×10 = 100 | 200 | Yüksek doğruluk, yavaş |
| `20` | 20×20 = 400 | 800 | Çok yüksek doğruluk, çok yavaş |

```bash
--integral_N 5   # Varsayılan
--integral_N 10  # Daha hassas
```

---

#### 🎚️ Adaptive Bilinear Parametreleri

`adaptive_bilinear_patch_integral`, `bilinear_patch_integral` ile aynı bilinear patch modelini kullanır; ancak her hücre için alt-bölme sayısını (N) tolerans kontrollü olarak artırır.

| Parametre | Varsayılan | Açıklama |
|:---------:|:----------:|:---------|
| `--adaptive_rel_tol` | `1e-4` | Bağıl tolerans |
| `--adaptive_abs_tol` | `0.0` | Mutlak tolerans |
| `--adaptive_max_level` | `5` | Maksimum inceltme seviyesi (N -> 2N -> 4N ...) |
| `--adaptive_min_N` | `2` | Başlangıç alt-bölme sayısı |
| `--adaptive_roughness_fastpath` | Açık | Düz/planar hücrelerde hızlı yol |
| `--adaptive_roughness_threshold` | Otomatik | Hızlı-yol eşiği (opsiyonel) |

```bash
--methods adaptive_bilinear_patch_integral \
--adaptive_rel_tol 1e-4 \
--adaptive_max_level 5 \
--adaptive_min_N 2
```

---

#### 🧭 Sector Adaptive Jenness Parametreleri

`sector_adaptive_jenness_integral`, klasik `jenness_window_8tri` yönteminin yerine geçmek için değil,
aynı 3x3 komşuluk fikrini ve 8-sektör topolojisini sürekli bir yüzey modeliyle genişletmek için
eklenmiştir.

| Parametre | Varsayılan | Açıklama |
|:---------:|:----------:|:---------|
| `--sector_jenness_rel_tol` | `1e-4` | Bağıl tolerans |
| `--sector_jenness_abs_tol` | `0.0` | Mutlak tolerans |
| `--sector_jenness_max_level` | `5` | Maksimum recursive subdivision seviyesi |
| `--sector_jenness_min_samples` | `3` | Üçgen quadrature örnek sayısı eşiği |

Seviye yorumu:
- `level-0`: sektör tek üçgen olarak hesaplanır.
- `level-1`: aynı üçgen 4 alt üçgene bölünür.
- `level-2`: toplam 16 alt üçgen değerlendirilir.
- `--sector_jenness_max_level`, bu derinlik için üst sınırdır.

```bash
--methods sector_adaptive_jenness_integral \
--sector_jenness_rel_tol 1e-4 \
--sector_jenness_max_level 5 \
--sector_jenness_min_samples 3
```

Notlar:
- İntegrasyon hücre içi 8 sektör üzerinde yapılır; eski Jenness'teki `weight=0.25` yaklaşımını kullanmaz.
- Yerel yüzey modeli 4 köşe değerinden türetilen bilinear patch değildir; 3x3 neighborhood üzerinden fit edilen quadratic yüzeydir.
- Düz veya neredeyse düz hücrelerde analytic plane fast-path kullanılır.
- Varsayılan yöntem setine dahildir; `--methods` verilmezse çalıştırılır.

---

#### 📊 `--sigma_mode` ve `--sigma_m` (Multiscale Parametreleri)

`multiscale_decomposed_area` yöntemi için Gaussian filtre ayarları.

**`--sigma_mode`:** Sigma değerlerinin nasıl yorumlanacağı

| Değer | Açıklama | Örnek |
|:-----:|:---------|:------|
| `mult` | GSD çarpanı olarak | σ = 2 × GSD (GSD=5m ise σ=10m) |
| `m` | Mutlak metre olarak | σ = 2m (sabit) |

**`--sigma_m`:** Sigma değerleri listesi

```bash
# GSD'nin 2 ve 5 katı sigma değerleri
--sigma_mode mult --sigma_m 2 5

# Sabit 10 ve 25 metre sigma
--sigma_mode m --sigma_m 10 25
```

**Sigma değeri ne anlama gelir?**

| Sigma | Etki |
|:-----:|:-----|
| Küçük (1-3) | Daha az düzleştirme, mikro detaylar korunur |
| Orta (3-10) | Dengeli ayrıştırma |
| Büyük (10+) | Güçlü düzleştirme, sadece makro topografya kalır |

---

#### 🎯 `--roi` (ROI / Parcel Bazlı Alanlar)

İsteğe bağlı olarak polygon ROI (GeoJSON veya Shapefile) verip her ROI için A2D/A3D hesaplayabilirsiniz.

> 🌍 **CRS Notu:** GeoJSON dosyalarında CRS belirtilmezse EPSG:4326 (lon/lat) varsayılır. ROI geometrileri DEM CRS'ine dönüştürülerek hesaplanır.

| Parametre | Açıklama |
|:---------:|:---------|
| `--roi <path>` | GeoJSON veya Shapefile yolu |
| `--roi_id_field <field>` | ROI id alanı (varsayılan: `id` varsa `id`, yoksa ilk alan) |
| `--roi_mode mask\|fraction` | `mask`: Hızlı. `fraction`: Daha hassas. |
| `--roi_all_touched` | `mask` modunda all_touched rasterize |
| `--roi_only` | Sadece ROI çıktısını yaz |

**Örnek (mask):**
```bash
python -m surface_area run \
  --dem dem.tif \
  --outdir out \
  --gsd 1 \
  --methods gradient_multiplier \
  --roi parcels.geojson \
  --roi_mode mask
```

**Örnek (fraction):**
```bash
python -m surface_area run \
  --dem dem.tif \
  --outdir out \
  --gsd 1 \
  --methods adaptive_bilinear_patch_integral \
  --roi parcels.geojson \
  --roi_mode fraction
```

---

#### 📈 `--plots` (Grafik Üretimi)

Etkinleştirildiğinde `outdir` altında PNG çizimleri üretilir. Veriye bağlı olarak bazı dosyalar atlanabilir (ör. hata dağılımı için referans yoksa).

| Dosya | Açıklama |
|:------|:---------|
| `A3D_vs_GSD.png` | 3D yüzey alanı vs GSD (log-x); varsa native-grid ve sürekli GT referans çizgileri |
| `ratio_vs_GSD.png` | A3D/A2D vs GSD |
| `surface_excess_vs_GSD.png` | (A3D/A2D − 1) yüzde olarak |
| `continuous_gt_rel_err_vs_GSD.png` | Sürekli GT’ye göre bağıl hata (`.reference.json` uygunsa) |
| `native_grid_ref_rel_err_vs_GSD.png` | Native-grid referansa göre bağıl hata (`.reference.json` uygunsa) |
| `runtime_vs_GSD.png` | Hesaplama + yeniden örnekleme süresi vs GSD |
| `error_vs_runtime.png` | Hata–süre trade-off (GT veya native ref hatası varsa) |
| `micro_ratio_vs_GSD.png` | Mikro oran vs GSD (yalnızca `multiscale_decomposed_area`) |

```bash
--plots
```

---

#### 💾 `--keep_resampled` (Ara Dosyaları Sakla)

Bu flag etkinleştirildiğinde, her GSD için oluşturulan resample edilmiş GeoTIFF dosyaları saklanır.

```bash
--keep_resampled  # Ara dosyaları sakla
```

> ⚠️ **Dikkat:** Çok sayıda GSD değeri için bu seçenek disk alanını önemli ölçüde kullanabilir.

---

### Tam Örnek Komut

```bash
python main.py run ^
  --dem dag_dsm.tif ^
  --outdir out ^
  --gsd 0.5 1 2 5 10 ^
  --methods jenness_window_8tri sector_adaptive_jenness_integral tin_2tri_cell gradient_multiplier bilinear_patch_integral multiscale_decomposed_area ^
  --resampling bilinear ^
  --slope_method horn ^
  --jenness_weight 0.25 ^
  --integral_N 5 ^
  --sector_jenness_rel_tol 1e-4 ^
  --sector_jenness_max_level 5 ^
  --sector_jenness_min_samples 3 ^
  --sigma_mode mult ^
  --sigma_m 2 5 ^
  --plots
```

---

## 🔬 Yöntemler

### Yöntem Karşılaştırma Tablosu

| Yöntem | Hız | Doğruluk | Açıklama |
|:-------|:---:|:--------:|:---------|
| `gradient_multiplier` | ⚡⚡⚡ Çok hızlı | Yüksek | Gradient tabanlı alan çarpanı |
| `tin_2tri_cell` | ⚡⚡ Hızlı | Yüksek | Her hücre 2 üçgen |
| `jenness_window_8tri` | ⚡⚡ Hızlı | Çok yüksek | 3x3 pencerede 8 üçgen |
| `sector_adaptive_jenness_integral` | ⚡ Orta | Çok yüksek | 3x3 quadratic fit + hücre içi 8 sektör integrali |
| `bilinear_patch_integral` | ⚡ Yavaş | En yüksek | Bilinear yüzey integrasyonu |
| `adaptive_bilinear_patch_integral` | ⚡ Yavaş | En yüksek | Adaptif bilinear integral |
| `multiscale_decomposed_area` | ⚡ Yavaş | Özel | Topo + mikro ayrıştırma |

---

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

### 2. Sector Adaptive Jenness Integral (`sector_adaptive_jenness_integral`)

Bu yöntem, `jenness_window_8tri` için bir **replacement** değildir; aynı 3x3 neighborhood
fikrini ve 8-sektör topolojisini koruyan **continuous-surface, sector-based** bir uzantıdır.

Özet akış:

1. Mevcut hücrenin etrafındaki tam 3x3 merkez örneklerinden yerel quadratic yüzey fit edilir.
2. Yüzey modeli:

```text
z(x, y) = a*x^2 + b*y^2 + c*x*y + d*x + e*y + f
```

3. Hücrenin kendi footprint'i, merkez nokta ile 4 kenar orta noktası ve 4 köşe arasında
   oluşan 8 sektör içine bölünür. Toplam 2D alan tam olarak `dx * dy` olur.
4. Her sektör için alan integrali analitik türevlerle hesaplanır:

```text
dz/dx = 2*a*x + c*y + d
dz/dy = 2*b*y + c*x + e

A_cell = sum_over_8_sectors integral integral sqrt(1 + (dz/dx)^2 + (dz/dy)^2) dA
```

5. İntegrasyon adaptif triangle subdivision ile yapılır. Quadratic terimler ihmal edilebilir
   düzeydeyse yöntem recursive çözüm yerine analytic plane fast-path kullanır.

Refinement level ne demektir?

```text
level-0: sektör tek üçgen olarak hesaplanır
level-1: aynı üçgen 4 alt üçgene bölünür
level-2: her alt üçgen tekrar 4'e bölünür -> toplam 16 alt üçgen
level-3: toplam 64 alt üçgen
...
```

- `level-0`, hiç bölmeden yapılan ilk kaba değerlendirmedir.
- `level-1`, ilk adaptif bölme seviyesidir; pratikte birçok hücre burada yeterince iyi yakınsar.
- `--sector_jenness_max_level`, bu subdivision derinliği için üst sınırı belirler.
- `results_long` sayfasındaki `sector_jenness_avg_level` ve `sector_jenness_max_level_used` bu hiyerarşiyi raporlar.

Klasik Jenness'ten farkı:
- Komşu hücre merkezlerinden kurulan 3B triangle fan kullanmaz.
- `weight=0.25` gibi sabit bir paylaştırma katsayısına ihtiyaç duymaz.
- 8-sektör ayrımı hücrenin kendi 2D footprint'i içinde yapılır.

Bilinear/adaptive bilinear yöntemlerinden farkı:
- Hücre köşe değerlerinden bilinear patch kurmaz.
- 3x3 neighborhood üzerinden 6-parametreli local quadratic model fit eder.

Neden sadece "Jenness + integral" değildir:
- Jenness'ten sadece yerel 3x3 destek ve 8-sektör fikrini korur.
- Geometri artık komşu-merkez üçgenleri değil, hücre footprint'i içindeki sektörlerdir.
- Yüzey artık parça-parça düz triangle fan değil, sürekli quadratic lokal modeldir.
- Entegrasyon 3B kenar uzunluklarından değil, analitik türevli yüzey integrand'inden gelir.

Yeni CLI bayrakları:
- `--sector_jenness_rel_tol`
- `--sector_jenness_abs_tol`
- `--sector_jenness_max_level`
- `--sector_jenness_min_samples`

Ek sonuç kolonları:
- `sector_jenness_avg_level`
- `sector_jenness_max_level_used`
- `sector_jenness_refined_fraction`

Beklenen güçlü yönler:
- Yerel eğriliği klasik Jenness'ten daha iyi temsil eder.
- Düz planlarda fast-path sayesinde gereksiz recursion yapmaz.
- Cell footprint tam olarak korunduğu için Jenness weight heuristiği ortadan kalkar.

Sınırlamalar:
- İlk sürüm tam 3x3 geçerli stencil ister; nodata'ya karşı konservatiftir.
- Python-level adaptif recursion, `gradient_multiplier` kadar hızlı değildir.
- Local quadratic fit çok keskin kırıkları birebir temsil etmeyebilir.

### 3. TIN 2-Triangle Cell (`tin_2tri_cell`)

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

### 4. Gradient Multiplier (`gradient_multiplier`)

Yerel eğim gradyanlarını (∂z/∂x, ∂z/∂y) kullanarak **alan çarpanı** hesaplar.

**Formül:**
```
A_cell = dx × dy × √(1 + p² + q²)

p = ∂z/∂x,  q = ∂z/∂y
```

**Gradient Kernelleri:**

| Kernel | Açıklama | Stencil |
|:------:|:---------|:-------:|
| **Horn** | 8 komşu ağırlıklı ortalama | 3x3 (tüm komşular) |
| **Zevenbergen-Thorne (ZT)** | 4 komşu basit fark | Cross (N,S,E,W) |

**Horn Kernel:**
```
∂z/∂x = [(NE + 2E + SE) - (NW + 2W + SW)] / (8×dx)
∂z/∂y = [(SW + 2S + SE) - (NW + 2N + NE)] / (8×dy)
```

---

### 5. Bilinear Patch Integral (`bilinear_patch_integral`)

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

### 6. Adaptive Bilinear Patch Integral (`adaptive_bilinear_patch_integral`)

`bilinear_patch_integral` yönteminin tolerans kontrollü adaptif sürümüdür.

- Her hücrede N önce `--adaptive_min_N` ile başlar.
- N ikiye katlanarak artırılır (N → 2N → 4N ...).
- Ardışık iki seviye arasındaki fark tolerans altına düşünce durur ve son (fine) seviye alanı döner.

**Ne zaman kullanılır?**
- Düz alanlar: düşük seviye, hızlı.
- Engebeli alanlar: daha fazla inceltme, daha yüksek doğruluk.

**Ek çıktı kolonları (results_long sayfasında):**
- `adaptive_avg_level`: Ortalama adaptif seviye
- `adaptive_max_level_used`: Kullanılan maksimum seviye
- `adaptive_refined_cell_fraction`: Seviye > 1 olan hücre oranı
- `adaptive_total_subcells_evaluated`: Toplam değerlendirilen alt-hücre sayısı

---

### 7. Multiscale Decomposed Area (`multiscale_decomposed_area`)

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

## 📊 Çıktılar

`--outdir` altında oluşturulan dosyalar:

### Excel Dosyası

#### `results.xlsx` / `results_long`

Her satır bir (GSD, method) kombinasyonunu temsil eder.

| Kolon | Tip | Açıklama |
|:------|:---:|:---------|
| `gsd_m` | float | Hedef GSD (metre) |
| `dx`, `dy` | float | Gerçek piksel boyutları |
| `method` | str | Hesaplama yöntemi |
| `A2D` | float | Planimetrik alan (m²) = valid_cells × dx × dy |
| `A3D` | float | 3D yüzey alanı (m²) |
| `ratio` | float | Alan oranı = A3D / A2D |
| `valid_cells` | int | Geçerli hücre sayısı |
| `runtime_sec` | float | Yüzey alanı hesabı süresi (saniye; blok işleme) |
| `resample_runtime_sec` | float | DEM’i ilgili GSD’ye yeniden örnekleme süresi (saniye); `native` satırında genelde yok veya 0 |
| `note` | str | Parametre özeti |

**Adaptive bilinear için ek kolonlar (results_long sayfasında):**

| Kolon | Tip | Açıklama |
|:------|:---:|:---------|
| `adaptive_avg_level` | float | Ortalama adaptif seviye |
| `adaptive_max_level_used` | int | Kullanılan maksimum seviye |
| `adaptive_refined_cell_fraction` | float | Seviye > 1 olan hücre oranı |
| `adaptive_total_subcells_evaluated` | int | Toplam değerlendirilen alt-hücre sayısı |

**Sector adaptive Jenness için ek kolonlar (results_long sayfasında):**

| Kolon | Tip | Açıklama |
|:------|:---:|:---------|
| `sector_jenness_avg_level` | float | Hücre başına kullanılan ortalama maksimum refinement seviyesi |
| `sector_jenness_max_level_used` | int | Kullanılan maksimum refinement seviyesi |
| `sector_jenness_refined_fraction` | float | En az bir recursive refinement alan hücre oranı |

Buradaki seviye, sector adaptive subdivision seviyesidir:
- `0` = bölmesiz ilk değerlendirme
- `1` = ilk 4'lü bölme
- `2` = ikinci bölme (16 alt üçgen)

**Multiscale için ek kolonlar:**

| Kolon | Tip | Açıklama |
|:------|:---:|:---------|
| `a_topo` | float | Topoğrafik alan bileşeni |
| `a_micro` | float | Mikro-pürüzlülük bileşeni |
| `a_total` | float | Toplam çok ölçekli yüzey alanı (topo + mikro ile tutarlı) |
| `micro_ratio` | float | A_micro / A_total |
| `sigma_m` | float | Kullanılan sigma değeri (metre) |

#### `results.xlsx` / `results_wide`

Satir = GSD, sutunlar = sadece hesaplanan yontemlerin `A3D` alan kolonlari (`{method}_A3D`).

#### `results.xlsx` / `results_roi_long` (ROI verilirse)

Her satır bir (GSD, ROI, method) kombinasyonunu temsil eder.

| Kolon | Tip | Açıklama |
|:------|:---:|:---------|
| `gsd_m` | float | Hedef GSD |
| `roi_id` | str | ROI/parsel kimliği |
| `method` | str | Yöntem |
| `A2D`, `A3D`, `ratio` | float | ROI bazlı alanlar ve oran |
| `valid_cells` | int | ROI ile kesişen geçerli hücre sayısı |
| `runtime_sec` | float | Yaklaşık hesaplama süresi |
| `resample_runtime_sec` | float | Yeniden örnekleme süresi |
| `note` | str | ROI modu ve notlar |

Yan dosya **`.reference.json`** bulunduğunda `results_long` satırlarına `continuous_gt_*`, `synthetic_native_ref_*` ve türev hata kolonları eklenebilir (sentetik/analitik kıyaslar için).

#### `batch_summary.xlsx` (toplu çalıştırma)

Birden fazla DEM işlendiğinde `outdir` kökünde üretilir; `batch_summary` sayfasında tüm DEM’lerin özet metriklerini karşılaştırabilirsiniz.

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

Ayrıntılı dosya listesi için yukarıdaki **📈 `--plots` (Grafik üretimi)** tablosuna bakın.

---

## 🛠️ Teknik Detaylar

### Nodata ve Kenar Yönetimi

| Durum | Davranış |
|:------|:---------|
| Nodata hücreler | Maskelenir, hesaplamaya dahil edilmez |
| Stencil tabanlı yöntemler (Horn/ZT, Jenness, Sector Adaptive Jenness) | Tam stencil valid değilse hücre atlanır |
| Köşe tabanlı yöntemler (TIN, Bilinear, Adaptive Bilinear) | 4 geçerli hücre merkezinden türetilmediğinde köşe atlanır |
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

## 📁 Proje Yapısı

```
yuzey_alani_hesaplama/
├── 📂 surface_area/
│   ├── __init__.py          # Paket tanımı, sürüm
│   ├── __main__.py          # python -m surface_area girişi
│   ├── cli.py               # run / synth alt komutları, Excel, batch özeti
│   ├── io.py                # Raster I/O, blok pencereleri, yeniden örnekleme
│   ├── methods.py           # Yedi yüzey alanı algoritması (numba hızlandırmalı çekirdekler)
│   ├── multiscale.py        # Gaussian ayrıştırma
│   ├── plotting.py          # PNG grafikleri
│   ├── progress.py          # İlerleme çıktısı
│   ├── roi.py               # ROI (GeoJSON / Shapefile)
│   ├── synthetic.py         # Sentetik DSM + preset'ler
│   └── analytic_surfaces.py # Analitik benchmark yüzeyleri ve sürekli GT integrali
├── 📂 tests/
│   ├── conftest.py
│   ├── test_synthetic.py
│   ├── test_cli_synth.py
│   ├── test_generate_synthetic_tif_script.py
│   ├── test_adaptive_and_roi.py
│   ├── test_sector_adaptive_jenness.py
│   ├── test_main_batch.py       # main.py toplu işlem
│   ├── test_parallel_cli.py     # --workers davranışı
│   └── test_plotting.py         # Grafik üretimi
├── 📂 .githooks/
│   └── pre-commit            # Git hook'ları
├── 📂 .vscode/
│   ├── launch.json           # Debug konfigürasyonu
│   └── tasks.json            # Task tanımları
├── .gitignore
├── generate_synthetic_tif.py # Sentetik DSM üretimi (+ referans alan)
├── main.py                   # Ana çalıştırma dosyası
├── requirements.txt          # Bağımlılıklar
└── README.md                 # Bu dosya
```

---

## 🧪 Testler

### Test Çalıştırma

```bash
# Tüm testleri çalıştır
pytest -q

# Detaylı çıktı
pytest -v

# Belirli bir test
pytest tests/test_synthetic.py::test_plane_all_methods_high_accuracy

# Coverage raporu
pytest --cov=surface_area --cov-report=html
```

### Test Kapsamı

| Test Dosyası | Açıklama |
|:-------------|:---------|
| `tests/test_synthetic.py` | Sentetik (analitik) yüzeyler üzerinde yöntemlerin doğruluğunu test eder |
| `tests/test_cli_synth.py` | `python -m surface_area synth` komutunun ürettiği GeoTIFF/metadata kontrolleri |
| `tests/test_generate_synthetic_tif_script.py` | `generate_synthetic_tif.py` script'inin GeoTIFF + `.reference.json` üretimi |
| `tests/test_adaptive_and_roi.py` | Adaptif integral diagnostikleri ve ROI (mask/fraction) akışı |
| `tests/test_sector_adaptive_jenness.py` | Sector adaptive Jenness analitik ve CLI testleri |
| `tests/test_main_batch.py` | Klasör / `.demlist` ile toplu çalıştırma |
| `tests/test_parallel_cli.py` | Çok işlemci blok hesaplama |
| `tests/test_plotting.py` | PNG çizimleri |

### Test Yüzeyleri

`tests/test_synthetic.py` içinde kullanılan bazı yüzeyler:

| Yüzey | Açıklama | Tolerans |
|:------|:---------|:--------:|
| **Plane** | z = ax + by + c | < 0.1% hata |
| **Sinusoid** | z = A×sin(kx×x)×sin(ky×y) | < 5% hata |
| **Paraboloid** | z = (x² + y²) / scale | < 5% hata |

**Referans alan hesabı:** Yüksek çözünürlüklü (dx/10, dy/10) iki-üçgen integrasyon

---

## ⚡ Performans İpuçları

| 💡 İpucu | Açıklama |
|:---------|:---------|
| **Büyük DEM'ler** | Önce daha kaba GSD'lerle (2-50m) test edin |
| **Upsample dikkat** | Kaynak çözünürlükten daha küçük GSD çıktıyı çok büyütebilir |
| **Multiscale** | `--sigma_mode mult` genellikle daha tutarlı sonuç verir |
| **Bellek** | `--keep_resampled` kapalı tutun (varsayılan) |
| **Hızlı sonuç** | `method_choice="fast"` veya yalnızca `gradient_multiplier` |
| **Paralellik** | `--workers` değerini I/O sınırına göre artırın (ör. 2–8); her ortamda linear hızlanma beklemeyin |

---

## 📜 Sürüm Geçmişi

### v0.1.0 (güncel kod tabanı)

- 7 yüzey alanı yöntemi; `main.py` üzerinde **metot ön ayarları** (`default`, `fast`, `balanced`, …)
- **Native GSD** (`native`), **upsample kontrolü** (`--allow-upsample`), **çok işlemci** blok hesaplama (`--workers`)
- **Toplu işlem:** DEM klasörü / `.demlist` + `batch_summary.xlsx`
- Sentetik üretim: raster preset'leri + **analitik** `analytic_*` yüzeyleri; `generate_synthetic_tif.py` ile zengin `.reference.json`
- ROI (mask / fraction), Excel + JSON + genişletilmiş PNG seti (`runtime`, hata eğrileri, vb.)
- Bağımlılıklar: `numpy` 2.x, `numba`, `openpyxl`, … (`requirements.txt`)

---

## 📚 Kaynaklar

- Jenness, J. S. (2004). *Calculating landscape surface area from digital elevation models.* Wildlife Society Bulletin, 32(3), 829-839.
- Horn, B. K. (1981). *Hill shading and the reflectance map.* Proceedings of the IEEE, 69(1), 14-47.
- Zevenbergen, L. W., & Thorne, C. R. (1987). *Quantitative analysis of land surface topography.* Earth Surface Processes and Landforms, 12(1), 47-56.

---

## 📄 Lisans

Bu proje açık kaynak olarak sunulmaktadır.

---

## 🤝 Katkıda Bulunma

Katkılarınızı bekliyoruz! Lütfen:

1. 🍴 Projeyi fork edin
2. 🌿 Yeni bir branch oluşturun (`git checkout -b feature/yenilik`)
3. 💾 Değişikliklerinizi commit edin (`git commit -m 'Yeni özellik eklendi'`)
4. 📤 Branch'inizi push edin (`git push origin feature/yenilik`)
5. 🔃 Pull Request açın

---

## 💬 Destek

Sorularınız veya önerileriniz için issue açabilirsiniz.

---

<p align="center">
  Made with ❤️ for GIS and Remote Sensing
</p>
