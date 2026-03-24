# DEM/DSM Tabanlı 3B Yüzey Alanı Hesabında Klasik ve Sektör-Adaptif Jenness Yaklaşımlarının Çok Ölçekli Karşılaştırılması

**Yazar(lar):** Ad Soyad1, Ad Soyad2  
**Kurum(lar):** Kurum bilgileri buraya eklenecek  
**E-posta:** e-posta adresleri buraya eklenecek

## Öz

Bu çalışma, DEM/DSM tabanlı üç boyutlu yüzey alanı hesabında kullanılan farklı ayrıklaştırma yaklaşımlarını çok ölçekli bir benchmark düzeni içinde karşılaştırmak ve bu kapsamda önerilen yeni sektör-adaptif Jenness türevi yöntemin pratik katkısını değerlendirmek amacıyla gerçekleştirilmiştir. Çalışmada 24 farklı sentetik yüzey senaryosu, 9 farklı mekânsal çözünürlük düzeyi ve 4 aktif yüzey alanı hesabı yöntemi kullanılmıştır. Karşılaştırmalar, uygun referansın mevcut olduğu 88 alan-GSD kombinasyonu üzerinden yürütülmüştür. Elde edilen sonuçlar, klasik Jenness 8-Triangle yaklaşımının hem genel doğruluk hem de doğruluk-hız dengesi açısından en güçlü yöntem olduğunu göstermiştir. Native çözünürlükte klasik Jenness yönteminin medyan mutlak göreli hatası %0,0191 iken, önerilen sektör-adaptif Jenness türevi yöntemde bu değer %0,0617 olarak bulunmuştur. Buna karşılık önerilen yöntem, klasik Jenness'e göre native çözünürlükte yaklaşık 36 kat daha yüksek çalışma süresi gerektirmiştir. Yeni yaklaşım `patchwork` ve `mixed` gibi karma yüzeylerde anlamlı kazanımlar üretmiş; buna karşılık `canyon` ve `terraced` gibi keskin morfolojilerde klasik Jenness'in belirgin biçimde gerisinde kalmıştır. Sonuçlar, önerilen yöntemin klasik Jenness'in genel amaçlı yerine geçen yeni bir varsayılan çözümden çok, belirli karma yüzey tiplerinde seçmeli olarak değerlendirilebilecek uzmanlaşmış bir varyant olarak ele alınmasının daha uygun olduğunu göstermektedir.

**Anahtar Kelimeler:** DEM, DSM, 3B yüzey alanı, Jenness, yüzey pürüzlülüğü, çok ölçekli analiz

## Abstract

This study aims to compare different discretization-based surface area estimation approaches for DEM/DSM-derived three-dimensional surface area analysis under a multi-scale benchmark framework, and to evaluate the practical contribution of a newly proposed sector-adaptive Jenness variant. The experiments were conducted on 24 synthetic surface scenarios, 9 spatial resolution levels, and 4 active surface-area estimation methods. The comparisons were performed on 88 area-GSD combinations for which an appropriate reference was available. The results show that the classical Jenness 8-Triangle approach provides the strongest overall performance in terms of both accuracy and the accuracy-runtime trade-off. At native resolution, the median absolute relative error of classical Jenness was 0.0191%, whereas the proposed sector-adaptive Jenness variant yielded 0.0617%. However, the proposed method required approximately 36 times higher runtime than classical Jenness at native resolution. The new approach produced meaningful gains on mixed composite surfaces such as `patchwork` and `mixed`, but performed substantially worse than classical Jenness on sharp morphologies such as `canyon` and `terraced`. These findings suggest that the proposed method should be positioned not as a general replacement for classical Jenness, but rather as a specialized adaptive extension for selected mixed-surface cases.

**Keywords:** DEM, DSM, 3D surface area, Jenness, terrain roughness, multi-scale analysis

## 1. Giriş

Sayısal yükseklik modelleri üzerinden üç boyutlu yüzey alanı hesabı, jeomorfoloji, arazi pürüzlülüğü analizi, ekolojik modelleme, mühendislik planlama ve yüksek çözünürlüklü topografik karakterizasyon gibi birçok uygulama alanı için temel bir problemdir. Özellikle yüksek mekânsal çözünürlüklü DEM/DSM verilerinin yaygınlaşmasıyla birlikte, yüzeyin planimetrik alanı ile gerçek üç boyutlu alanı arasındaki farkın güvenilir biçimde hesaplanması daha kritik hale gelmiştir. Ancak bu problem, yalnızca veri çözünürlüğüne değil, aynı zamanda yüzeyin nasıl ayrıklaştırıldığına, hücre içi yüzeyin hangi matematiksel modelle temsil edildiğine ve kullanılan integrasyon yaklaşımına da güçlü biçimde bağlıdır.

Bu bağlamda literatürde ve uygulamada en yaygın yaklaşım aileleri; gradyan tabanlı alan çarpanı yöntemleri, TIN tabanlı hücre ayrıklaştırmaları ve komşuluk penceresi üzerinde daha zengin yerel yüzey tanımlayan pencere tabanlı yöntemlerdir. Klasik Jenness yaklaşımı, 3x3 komşuluk içindeki üçgensel ayrıklaştırma mantığı sayesinde özellikle pratik uygulamalarda güçlü bir referans yöntem olarak öne çıkmaktadır. Bununla birlikte, karma topografik paternlerin tek bir hücre çevresinde birlikte bulunduğu durumlarda daha zengin yerel yüzey temsilleri ve hücre içi adaptif integrasyon şemaları kullanmanın ek doğruluk sağlayıp sağlamayacağı açık bir araştırma sorusu olmaya devam etmektedir.

Bu çalışmanın çıkış noktası da bu sorudur. Çalışmada, klasik Jenness yaklaşımını sektör temelli adaptif integral mantığı ile genişleten yeni bir Jenness türevi yöntem önerilmiş ve bu yeni yaklaşımın gerçekten pratik bir kazanım üretip üretmediği sistematik olarak test edilmiştir. Bu amaçla yalnızca tek bir yüzey tipi veya tek bir çözünürlük düzeyi üzerinden değil, çok sayıda sentetik senaryo ve çok sayıda çözünürlük düzeyi üzerinden bir benchmark düzeni kurulmuştur. Böylece hem yöntemlerin genel davranışı hem de önerilen yaklaşımın hangi yüzey tiplerinde avantaj ya da dezavantaj ürettiği daha güvenilir biçimde gözlenebilmiştir.

Bu bildirinin temel katkıları üç başlık altında özetlenebilir. İlk olarak, DEM/DSM tabanlı 3B yüzey alanı hesabı için çok ölçekli ve çok senaryolu bir karşılaştırma düzeni sunulmuştur. İkinci olarak, klasik Jenness yaklaşımını genişleten yeni sektör-adaptif bir varyant önerilmiş ve bu yaklaşım kontrollü benchmark koşullarında sınanmıştır. Üçüncü olarak, sonuçlar yalnızca genel ortalamalar üzerinden değil; yüzey ailesi, çözünürlük rejimi ve belirli zor senaryolar temelinde tartışılarak önerilen yaklaşımın genel amaçlı bir çözüm mü yoksa uzmanlaşmış bir araç mı olduğu sorgulanmıştır.

## 2. Materyal ve Yöntem

### 2.1. Çalışma çerçevesi

Çalışmada karşılaştırmalar Python tabanlı bir hesaplama çerçevesi içinde yürütülmüştür. Bu çerçeve, büyük rasterlar üzerinde blok bazlı işlemeyi, çoklu GSD analizini, yöntem bazlı çalışma sürelerinin kaydını, grafik ve Excel çıktılarının üretilmesini ve sentetik benchmark verilerinin tekrar üretilebilir biçimde oluşturulmasını desteklemektedir. Tüm aktif deneylerde bilinear yeniden örnekleme, Horn türevi gradyan kestirimi ve ortak bir GSD listesi kullanılmıştır. Kullanılan aktif yöntemler `gradient_multiplier`, `tin_2tri_cell`, `jenness_window_8tri` ve bu çalışmada önerilen `sector_adaptive_jenness_integral` yaklaşımıdır.

### 2.2. Karşılaştırılan yöntemler

`gradient_multiplier` yöntemi, gradyan tabanlı alan çarpanı mantığına dayalı, hesaplama maliyeti düşük bir temel çizgi yöntemidir. `tin_2tri_cell` yaklaşımı, her hücreyi iki üçgen olarak modelleyerek daha geometrik bir ayrıklaştırma sunar. `jenness_window_8tri` yöntemi ise 3x3 komşuluk üzerinde sekiz üçgenli bir pencere ayrıklaştırması kullanır ve bu nedenle hem yerel bağlamı dikkate alır hem de pratik olarak güçlü doğruluk sağlar.

Bu çalışmada önerilen `sector_adaptive_jenness_integral` yöntemi, klasik Jenness mantığını doğrudan kopyalamak yerine aynı 3x3 komşuluk fikrini daha zengin bir yerel yüzey modeli ile genişletmektedir. Bu yaklaşımda hücre çevresindeki yükseklik örneklerinden daha süreklilikçi bir yüzey yaklaşımı elde edilmekte, ardından hücre içi alan hesabı sektör bazında adaptif integral mantığı ile gerçekleştirilmektedir. Beklenti, özellikle karma ve heterojen mikro-topografyanın aynı hücre çevresinde bulunduğu durumlarda bu yaklaşımın klasik ayrıklaştırmalara göre ek doğruluk üretmesidir.

### 2.3. Benchmark veri seti

Karşılaştırmalar 24 sentetik yüzey senaryosu üzerinde gerçekleştirilmiştir. Bu set; 8 analitik yüzey, 10 gerçekçi arazi tipi ve 6 test pattern içermektedir. Analitik yüzeyler sürekli ground truth üretmeye uygundur ve bu nedenle özellikle çözünürlük etkisini kontrollü biçimde izlemek için kullanılmıştır. Gerçekçi yüzey grubu dağlık, kıyısal, alüvyal, karstik, buzul, kanyon ve benzeri morfolojileri temsil etmektedir. Test pattern grubu ise `plane`, `waves`, `crater_field`, `terraced`, `patchwork` ve `mixed` senaryolarından oluşmaktadır. Bu senaryoların amacı, yöntemleri hem pürüzsüz hem kırıklı hem de çok bileşenli yüzeylerde sınamaktır.

Tüm ana batch seti 0,05 m native piksel boyutuna sahip 16384x16384 hücrelik rasterlar üzerinden kurulmuştur. Deneyler `native`, 0,1, 0,5, 1, 2, 5, 10, 20 ve 50 m olmak üzere 9 farklı GSD düzeyinde yürütülmüştür. Tüm 24 senaryo için native-grid referansı mevcuttur. Buna ek olarak 8 analitik yüzey için sürekli ground truth temelli çözünürlükler arası karşılaştırma da yapılabilmiştir. Sonuç olarak bu çalışma 864 yöntem-sonuç satırı ve 160 referans satırı içeren kapsamlı bir değerlendirme kümesi üretmiştir.

### 2.4. Değerlendirme ölçütleri

Yöntemler mutlak göreli hata, medyan mutlak göreli hata, ortalama mutlak göreli hata ve çalışma süresi bakımından değerlendirilmiştir. Çalışmada özellikle medyan hata ölçütü öne çıkarılmıştır; çünkü bazı yüzeylerin belirgin şekilde daha zor olması nedeniyle ortalama değerler tek başına yöntemin tipik davranışını temsil etmekte yetersiz kalabilmektedir. Ayrıca yalnızca genel ortalamalara bakmak yerine, yöntemlerin alan bazlı “birincilik” sayıları, eşik bazlı göreli iyileşme sayıları ve yüzey ailesine göre davranışları da ayrıca analiz edilmiştir. Bu yaklaşım, önerilen yeni yöntemin gerçekten hangi koşullarda değer ürettiğini daha açık biçimde göstermektedir.

## 3. Bulgular

### 3.1. Genel performans

Sonuçların ilk ve en net çıktısı, klasik `jenness_window_8tri` yönteminin genel lider olduğudur. Referanslı 88 karşılaştırmanın 77'sinde en düşük hata bu yöntem tarafından üretilmiştir. Native çözünürlükte medyan mutlak göreli hata klasik Jenness için %0,0191, `tin_2tri_cell` için %0,0526, önerilen sektör-adaptif Jenness türevi için %0,0617 ve `gradient_multiplier` için %0,0667 düzeyindedir. Bu tablo, klasik Jenness yaklaşımının yalnızca iyi bir referans yöntem değil, aynı zamanda mevcut benchmark setinde en başarılı genel çözüm olduğunu göstermektedir.

Çalışma süreleri dikkate alındığında ise `gradient_multiplier` yöntemi 6,21 saniyelik native medyan runtime ile en hızlı yöntemdir. `tin_2tri_cell` yaklaşık 15,30 saniye ile ikinci sırada yer alırken, klasik Jenness yaklaşık 60,38 saniye medyan çalışma süresine sahiptir. Önerilen sektör-adaptif Jenness türevi yöntemin native medyan çalışma süresi ise 2173,11 saniyedir. Dolayısıyla bu yöntem native çözünürlükte klasik Jenness'ten yaklaşık 36 kat, `gradient_multiplier` yönteminden ise yaklaşık 350 kat daha yavaştır. Bu nedenle yeni yaklaşımın değerlendirilmesinde yalnızca hata düzeyi değil, hata başına ödenen ek hesaplama maliyeti de kritik hale gelmektedir.

### 3.2. Önerilen yeni Jenness türevi yöntemin değerlendirilmesi

Bu çalışmanın temel sorusu, önerilen sektör-adaptif Jenness türevi yöntemin klasik Jenness karşısında gerçek ve sistematik bir üstünlük sağlayıp sağlamadığıdır. Elde edilen bulgular, bu soruya temkinli bir cevap verilmesi gerektiğini göstermektedir. Yeni yöntem, klasik Jenness'e karşı yalnızca 88 karşılaştırmanın 6'sında daha düşük hata üretmiştir. Üstelik bu 6 durumun yalnızca 2 tanesinde iyileşme %5'in üzerine çıkmaktadır. Buna karşılık yeni yöntem, klasik Jenness'ten 21 durumda %1'den fazla, 14 durumda ise %5'ten fazla daha kötü sonuç vermiştir. Bu sonuçlar, yeni yaklaşımın genel amaçlı bir replacement olarak değerlendirilemeyeceğini açık biçimde göstermektedir.

Bununla birlikte yeni yöntemin tamamen başarısız olduğu da söylenemez. Native çözünürlükte `patchwork` yüzeyinde klasik Jenness'e göre yaklaşık %43,8, `mixed` yüzeyinde ise yaklaşık %32,9 oranında göreli hata iyileşmesi elde edilmiştir. Bu iki senaryo, farklı pürüzlülük bileşenlerinin ve farklı yüzey desenlerinin aynı sahnede birlikte bulunduğu kompozit yüzeylerdir. Bu nedenle önerilen sektör-adaptif integral yaklaşımının en çok bu tür yüzeylerde değer üretiyor olması anlamlıdır. Ancak bu kazanımların genelleştirilebilir olduğunu söylemek için mevcut veri henüz yeterli değildir; çünkü aynı yöntem `canyon` ve `terraced` gibi keskin morfolojilerde dramatik biçimde kötüleşmektedir.

Özellikle `canyon` yüzeyinde klasik Jenness yaklaşık %0,0978 hata üretirken, yeni yöntemin hatası %4,6389 düzeyine yükselmiştir. Benzer biçimde `terraced` senaryosunda klasik Jenness yaklaşık %0,0140 hata üretirken, yeni yöntem yaklaşık %4,4346 hataya çıkmıştır. Bu iki örnek, önerilen yaklaşımın keskin kenarlar, ani yükseklik geçişleri ve basamaklı geometri içeren yüzeylerde ciddi bir model uyumsuzluğu yaşayabileceğini düşündürmektedir.

### 3.3. Yüzey ailesi ve çözünürlük etkisi

Analitik yüzeyler üzerinde yeni yöntem ile klasik Jenness arasındaki fark son derece küçüktür. Native çözünürlükte analitik grup içinde yeni yöntem hiçbir durumda klasik Jenness'i geçememiştir; ancak farklar çoğu durumda pratik olarak ihmal edilebilecek düzeydedir. Bu bulgu, yeni yaklaşımın pürüzsüz ve sürekli yüzeylerde temel tutarlılık gösterdiğini; fakat klasik Jenness'e göre somut bir ek kazanç üretmediğini göstermektedir.

Gerçekçi yüzey grubunda sonuçlar yeni yöntem aleyhine daha nettir. Dağlık, kıyısal, karstik, buzul ve alüvyal senaryolar dahil olmak üzere gerçekçi yüzeylerin hiçbirinde native çözünürlükte klasik Jenness aşılmamıştır. Üstelik bu grupta yeni yöntemin medyan hata farkı klasik Jenness aleyhine yaklaşık 0,099 yüzde puandır. Test pattern grubunda ise iki yönlü bir davranış ortaya çıkmıştır. `patchwork` ve `mixed` yüzeylerinde anlamlı kazanımlar gözlenirken, `terraced`, `waves`, `crater_field` ve kısmen `plane` yüzeylerinde yeni yaklaşımın performansı zayıflamıştır.

GSD etkisi incelendiğinde, önerilen yöntemin avantajının çözünürlük azaldıkça neredeyse tamamen kaybolduğu görülmektedir. Native çözünürlükte 24 senaryonun yalnızca 2'sinde üstünlük söz konusu iken, 0,1-2 m aralığında görülen az sayıdaki üstünlükler esas olarak analitik ve birbirine çok yakın hata düzeylerinden kaynaklanmaktadır. Bu çözünürlüklerde yeni yöntemin klasik Jenness'e göre medyan hata oranı 1,000009 ile 1,000072 aralığında değişmekte, yani pratikte neredeyse eşit kalmaktadır. Buna karşılık runtime farkı devam etmektedir. 5 m ve daha kaba çözünürlüklerde ise yeni yöntem hiçbir GSD düzeyinde klasik Jenness'i geçememektedir. Bu durum, çözünürlük düştükçe yeniden örnekleme ve bilgi kaybının belirleyici hale geldiğini, dolayısıyla daha karmaşık hücre içi integrasyonun beklenen ek faydayı üretmediğini düşündürmektedir.

## 4. Tartışma

Elde edilen bulgular, daha karmaşık yerel yüzey modellemesinin her durumda daha iyi sonuç üretmediğini açık biçimde göstermektedir. Önerilen sektör-adaptif Jenness türevi yaklaşımının temel mantığı, 3x3 komşuluk içinde daha zengin bir yerel yüzey temsili kurmak ve hücre içi alan hesabını daha ayrıntılı bir integral şeması ile gerçekleştirmektir. Teorik olarak bu yaklaşımın, tek bir hücre çevresinde farklı pürüzlülük bileşenlerinin bir arada bulunduğu heterojen yüzeylerde avantaj sağlaması beklenmektedir. `patchwork` ve `mixed` senaryolarında elde edilen sonuçlar bu beklentiyi kısmen doğrulamaktadır.

Ancak aynı yaklaşımın keskin morfolojilerde başarısız olması, yöntemin yerel yüzey modelinin belirli durumlarda aşırı yumuşatıcı davrandığını düşündürmektedir. Özellikle `terraced` ve `canyon` yüzeyleri, yüksek frekanslı keskin geçişlerin ve kırıklı geometrinin baskın olduğu örneklerdir. Bu tür durumlarda komşuluk tabanlı daha süreklilikçi bir yaklaşım, gerçek yüzey yapısını temsil etmek yerine onu yumuşatarak sistematik hataya yol açıyor olabilir. Bu yorum veri üzerinden türetilmiş bir çıkarımdır ve ileriki aşamalarda ayrıca sınanmalıdır; ancak mevcut sonuçlarla en tutarlı açıklama budur.

Bu bağlamda bildirinin en önemli yorumu, önerilen yöntemin nasıl konumlandırılması gerektiğidir. Mevcut veri, yöntemin klasik Jenness'in yerine geçen yeni genel çözüm olarak sunulmasını desteklememektedir. Buna karşılık yöntem, karma kompozit yüzeyler için uzmanlaşmış bir adaptif varyant olarak sunulabilir. Böyle bir konumlandırma, hem yöntemin gerçekten değer ürettiği örnekleri görünür kılar hem de hangi morfolojik durumlarda zayıfladığını dürüst biçimde tartışma olanağı sağlar. Bu yaklaşım, bildirinin metodolojik güvenilirliğini de artıracaktır.

Bir başka önemli tartışma noktası, doğruluk-hız dengesi bakımından pratik kullanımdır. Eğer uygulama bağlamında sınırlı hesaplama bütçesi varsa `gradient_multiplier` hızlı bir ön değerlendirme yöntemi olarak kullanılabilir. Eğer temel hedef genel doğruluk ise klasik Jenness en güçlü ana adaydır. Eğer özel olarak karma ve kompozit yüzeylerin seçmeli analizi hedefleniyorsa, önerilen sektör-adaptif Jenness türevi yaklaşım ileri düzey bir seçenek olarak değerlendirilebilir. Bu çok katmanlı kullanım senaryosu, yöntemlerin birbirinin yerine değil, farklı amaçlar için tamamlayıcı araçlar olarak düşünülmesini önermektedir.

## 5. Sonuç

Bu çalışma, DEM/DSM tabanlı 3B yüzey alanı hesabında kullanılan ayrıklaştırma temelli yöntemlerin çok ölçekli ve çok senaryolu bir benchmark düzeni içinde karşılaştırılmasını sağlamıştır. Sonuçlar, klasik `jenness_window_8tri` yönteminin hem genel doğruluk hem de doğruluk-hız dengesi bakımından en başarılı yöntem olduğunu göstermektedir. Önerilen sektör-adaptif Jenness türevi yaklaşım ise özellikle `patchwork` ve `mixed` gibi karma yüzeylerde anlamlı kazanımlar üretmiş, ancak gerçekçi keskin morfolojilerde ve kaba çözünürlük rejimlerinde klasik Jenness'in gerisinde kalmıştır.

Bu nedenle mevcut veri temelinde iki temel sonuç önerilmektedir. Birincisi, klasik Jenness yaklaşımı çalışmanın ana referans ve varsayılan yöntemi olarak korunmalıdır. İkincisi, önerilen yeni yöntem genel amaçlı bir replacement olarak değil, seçili karma yüzey tiplerinde kullanılabilecek uzmanlaşmış bir adaptif genişletme olarak değerlendirilmelidir. Gelecek çalışmalarda `patchwork` ve `mixed` benzeri yüzeylerin çoğaltılması, `canyon` ve `terraced` gibi keskin morfolojilerde gözlenen başarısızlığın nedenlerinin incelenmesi ve çoklu seed tabanlı tekrar deneyleriyle istatistiksel güven aralıklarının raporlanması önerilmektedir.

## 6. Teşekkür

Bu bölüm gerekli ise kurum, proje veya laboratuvar desteği için düzenlenecektir.

## 7. Kaynaklar

Bu taslakta kaynaklar bölümü bilerek yer tutucu olarak bırakılmıştır. Bildiri gönderiminden önce aşağıdaki başlıklara ait gerçek bibliyografik kayıtlar eklenmelidir:

1. Jenness yönteminin orijinal çalışması.
2. DEM/DSM tabanlı yüzey alanı hesabı üzerine temel yöntem makaleleri.
3. TIN ve gradient tabanlı yüzey alanı kestirimi çalışmaları.
4. Yüksek çözünürlüklü topografya ve arazi pürüzlülüğü analizi literatürü.
5. Gerekirse sentetik yüzey üretimi ve analitik benchmark tasarımı ile ilgili çalışmalar.
