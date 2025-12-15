# COMPAS Algoritmik Yanlılık (Bias) Analiz Raporu

**Tarih:** 2025-12-15  
**Veri Seti:** COMPAS (ProPublica Analysis)  
**Analiz Eden Script:** `egitim/bias.py`

Bu rapor, ABD ceza adalet sisteminde kullanılan **COMPAS** (Correctional Offender Management Profiling for Alternative Sanctions) risk değerlendirme algoritmasındaki olası ırksal yanlılıkları (racial bias) analiz etmek amacıyla hazırlanmıştır. Analiz, `bias.py` scripti kullanılarak gerçekleştirilmiş ve istatistiksel testlerle desteklenmiştir.

---

## 1. Yönetici Özeti

Yapılan kapsamlı veri analizi sonucunda, COMPAS algoritmasının **African-American** (Afro-Amerikan) bireyler aleyhine sistematik bir yanlılık (bias) içerdiği yönünde güçlü kanıtlar elde edilmiştir.

*   **Daha Yüksek Risk Skorları:** African-American bireylerin ortalama risk skoru (**5.90**), Caucasian (Beyaz) bireylere (**4.24**) göre istatistiksel olarak anlamlı düzeyde daha yüksektir.
*   **Adaletsiz Hata Dağılımı:** Algoritma hata yaptığında, bu hatalar ırklara göre farklılık göstermektedir. Suç tekrarlamayan African-American bireylerin "Yüksek Riskli" olarak yanlış etiketlenme olasılığı (**%45.28**), aynı durumdaki Caucasian bireylerin maruz kaldığı oranın (**%21.75**) iki katından fazladır.
*   **İstatiksel Anlamlılık:** Irk ile risk skoru arasındaki ilişkinin tesadüfi olmadığı, yapılan T-Test ve Chi-Square testleri ile doğrulanmıştır (p < 0.05).

---

## 2. Metodoloji

Bu analizde aşağıdaki adımlar izlenmiştir:

1.  **Veri Toplama:** ProPublica tarafından paylaşılan ham COMPAS verileri kullanılmıştır.
2.  **Veri Temizleme:** Analiz için gerekli olan `race`, `age`, `priors_count` (önceki suçlar), `decile_score` (risk puanı) ve `is_recid` (suç tekrarı) değişkenleri filtrelenmiştir. Eksik veriler temizlenmiştir.
3.  **Metrik Hesaplama:** Her ırk grubu için *False Positive Rate (FPR)* ve *False Negative Rate (FNR)* hesaplanmıştır.
4.  **İstatistiksel Testler:** Gruplar arası farkların anlamlılığını test etmek için T-Test ve Chi-Square testi uygulanmıştır.
5.  **Özellik Önemi:** Random Forest algoritması kullanılarak, modelin karar verirken hangi değişkenlere ağırlık verdiği incelenmiştir.

---

## 3. Risk Skoru Analizi

Farklı ırk gruplarına verilen ortalama risk skorları (Decile Score 1-10 arası) aşağıdadır:

| Irk Grubu | Ortalama Risk Skoru |
| :--- | :--- |
| **African-American** | **5.90** |
| **Caucasian** | **4.24** |
| Hispanic | (Daha düşük ortalama) |
| Other | (Daha düşük ortalama) |

> **Bulgu:** African-American grubu, ortalamada diğer tüm gruplardan daha yüksek bir risk puanına sahiptir. Skor dağılım grafikleri incelendiğinde, African-American grubu için dağılımın sağa çarpık (yüksek risk), Caucasian grubu için ise sola çarpık (düşük risk) olduğu görülmektedir.

*İlgili Grafik:* `compass_data/bias_analysis_score_dist.png`

---

## 4. Hata Analizi (Confusion Metrics)

Bir algoritmanın adaleti, sadece doğruluğuyla değil, yaptığı hataların nasıl dağıldığıyla da ölçülür.

| Metrik | African-American | Caucasian | Açıklama |
| :--- | :--- | :--- | :--- |
| **False Positive Rate (FPR)** | **%45.28** | **%21.75** | **Masumu Suçlama:** Gelecekte suç işlemeyecek bireye "Yüksek Riskli" denmesi. |
| **False Negative Rate (FNR)** | %27.99 | %47.72 | **Suçluyu Kaçırma:** Gelecekte suç işleyecek bireye "Düşük Riskli" denmesi. |
| **Doğruluk (Accuracy)** | ~%63 | ~%67 | Genel model doğruluğu. |

> **Kritik Bulgu:** Algoritma, Caucasian bireylerde riski olduğundan düşük tahmin etme (False Negative) eğilimindeyken; African-American bireylerde riski olduğundan yüksek tahmin etme (False Positive) eğilimindedir. Bu durum, "False Positive Eşitsizliği" olarak adlandırılan bir adalet sorunudur.

*İlgili Grafik:* `compass_data/bias_analysis_metrics.png`

---

## 5. İstatistiksel Test Sonuçları

Gözlemlenen farkların şans eseri olup olmadığını belirlemek için yapılan testler:

### 5.1. T-Test (Ortamalar Arası Fark)
*   **Hipotez:** African-American ve Caucasian risk skoru ortalamaları eşittir.
*   **P-Value:** 0.000000
*   **Sonuç:** Hipotez reddedilmiştir. Ortalamalar arasındaki fark **istatistiksel olarak anlamlıdır.**

### 5.2. Chi-Square Test (Bağımsızlık)
*   **Hipotez:** Irk ile "Yüksek Risk" kategorisine girmek arasında bir ilişki yoktur.
*   **P-Value:** 0.000000
*   **Sonuç:** Hipotez reddedilmiştir. Irk ve risk kategorisi arasında **güçlü bir bağımlılık** vardır.

---

## 6. Model ve Özellik Önemi (Feature Importance)

Modelin risk skorunu belirlerken en çok hangi özelliklere ağırlık verdiğini anlamak için *Random Forest* modeli eğitilmiş ve değişken önem düzeyleri çıkarılmıştır:

| Özellik | Önem Düzeyi |
| :--- | :--- |
| **Age (Yaş)** | **0.4734** |
| **Priors Count (Önceki Suçlar)** | **0.2856** |
| C Charge Degree (Suç Derecesi) | 0.1203 |
| Race (Irk) | 0.0691 |
| Sex (Cinsiyet) | 0.0281 |

> **Analiz:** Modelde "Irk" değişkeninin doğrudan ağırlığı düşük (%6.9) görünse de, sonuçlarda ırksal yanlılık çıkması, diğer değişkenlerin (özellikle "Önceki Suçlar" ve sosyo-ekonomik faktörlerin) ırk ile korelasyon içinde olmasından ("Proxy Variable") kaynaklanmaktadır. Sistemik eşitsizlikler nedeniyle belirli grupların daha fazla tutuklanma geçmişine sahip olması, algoritmanın bu geçmişi "risk" olarak öğrenmesine ve yanlılığı yeniden üretmesine neden olmaktadır.

*İlgili Grafik:* `compass_data/bias_analysis_feature_imp.png`

---

## 7. Kontrol Değişkenleri Altında Analiz

Yanlılığın sadece yaş veya suç geçmişi farkından kaynaklanıp kaynaklanmadığını test etmek için, benzer özelliklere sahip bireyler karşılaştırılmıştır:

**Senaryo 1: Genç, Orta Düzey Suç Geçmişi (Yaş 25-35, Önceki Suç: 2-5)**
*   African-American Ortalama Skor: **5.65**
*   Caucasian Ortalama Skor: **4.94**

**Senaryo 2: Orta Yaş, Düşük Suç Geçmişi (Yaş 45+, Önceki Suç: 0-1)**
*   African-American Ortalama Skor: **2.54**
*   Caucasian Ortalama Skor: **1.67**

> **Sonuç:** Benzer yaş ve suç geçmişine sahip bireyler karşılaştırıldığında bile, African-American bireylerin ortalama risk skorları Caucasian bireylere göre daha yüksek çıkmaktadır.

---

## 8. Sonuç

Analizler, COMPAS risk değerlendirme algoritmasının ırksal olarak nötr olmadığını göstermektedir. Algoritma özellikle **False Positive (Yanlış Alarm)** oranlarında African-American bireyler aleyhine belirgin bir dengesizlik sergilemektedir. Bu durum, algoritmanın yargı kararlarında kullanımının, mevcut toplumsal eşitsizlikleri pekiştirme riski taşıdığını ortaya koymaktadır.

---
*Rapor Sonu*
