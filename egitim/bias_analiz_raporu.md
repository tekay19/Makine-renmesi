# COMPAS Bias Analiz Raporu

**Tarih:** 2025-12-15  
**Veri Seti:** COMPAS (ProPublica)

Bu rapor, COMPAS risk değerlendirme algoritmasındaki olası yanlılıkları (bias) analiz etmektedir.

---

## 1. Yönetici Özeti

Yapılan analiz sonucunda, COMPAS algoritmasının **African-American** bireyler aleyhine sistematik bir yanlılık (bias) içerdiği gözlemlenmiştir.

*   **Daha Yüksek Risk Skorları:** African-American bireylerin ortalama risk skoru (5.90), Caucasian bireylere (4.24) göre belirgin şekilde daha yüksektir.
*   **Yanlış Pozitif Oranı (Adaletsizlik):** Suç tekrarlamayan African-American bireylerin "Yüksek Riskli" olarak etiketlenme olasılığı (**%45.28**), aynı durumdaki Caucasian bireylerin iki katından fazladır (**%21.75**).
*   **İstatiksel Anlamlılık:** Irk ile risk skoru arasındaki ilişki istatistiksel olarak anlamlı bulunmuştur (p < 0.05).

---

## 2. Risk Skoru Analizi

Irk gruplarına göre ortalama risk skorları (Decile Score 1-10):

| Irk | Ortalama Skor |
| :--- | :--- |
| **African-American** | **5.90** |
| **Caucasian** | **4.24** |
| Hispanic | (Daha düşük) |
| Other | (Daha düşük) |

> **Bulgu:** African-American grubu ortalamada en yüksek risk puanına sahiptir.

*İlgili Grafik:* `compass_data/bias_analysis_score_dist.png`

---

## 3. Hata Analizi (Confusion Metrics)

Algoritmanın hata yapma türleri ırklara göre nasıl değişiyor?

| Metrik | African-American | Caucasian | Anlamı |
| :--- | :--- | :--- | :--- |
| **False Positive Rate (FPR)** | **%45.28** | **%21.75** | Suçsuz olduğu halde "Suçlu" damgası yeme oranı. |
| **False Negative Rate (FNR)** | %27.99 | %47.72 | Suçlu olduğu halde "Suçsuz" sanılma oranı. |

> **Bulgu:** Algoritma, Caucasian bireylerde "suçluyu kaçırma" (False Negative) eğilimindeyken, African-American bireylerde "masumu suçlama" (False Positive) eğilimindedir.

*İlgili Grafik:* `compass_data/bias_analysis_metrics.png`

---

## 4. İstatistiksel Testler

Yapılan testler, gözlemlenen farkların tesadüfi olmadığını kanıtlamaktadır.

### T-Test (Ortalama Farkı)
*   **Karşılaştırma:** African-American vs Caucasian
*   **P-Value:** 0.000000
*   **Sonuç:** ✅ İstatistiksel olarak anlamlı fark var.

### Chi-Square Test (Bağımsızlık Testi)
*   **İlişki:** Irk vs Yüksek Risk Kategorisi
*   **P-Value:** 0.000000
*   **Sonuç:** ✅ Irk ve risk kategorisi arasında güçlü bir bağımlılık var.

---

## 5. Model ve Özellik Önemi

Modelin karar verirken en çok hangi özelliklere baktığı (Random Forest Feature Importance):

| Özellik | Önem Düzeyi |
| :--- | :--- |
| **Age (Yaş)** | **0.4734** |
| **Priors Count (Önceki Suçlar)** | **0.2856** |
| C Charge Degree (Suç Derecesi) | 0.1203 |
| Race (Irk) | 0.0691 |
| Sex (Cinsiyet) | 0.0281 |

> **Analiz:** Model doğrudan "Irk" özelliğine (Race) çok düşük bir ağırlık verse de (%6.9), sonuçlarda ırksal yanlılık çıkması, diğer değişkenlerin (özellikle "Önceki Suçlar" ve demografik dağılımın) ırk ile **proxy (vekil)** ilişkisi içinde olduğunu düşündürmektedir.

*İlgili Grafik:* `compass_data/bias_analysis_feature_imp.png`

---

## 6. Kontrol Değişkenleri ile Analiz (Örneklem)

Benzer yaş ve benzer suç geçmişine sahip bireyler karşılaştırıldığında bile farklar devam etmektedir.

**Örnek: Yaş 25-35, Önceki Suç Sayısı 2-5 Arası**
*   African-American Ortalama Skor: **5.65**
*   Caucasian Ortalama Skor: **4.94**

**Örnek: Yaş 45+, Önceki Suç Sayısı 0-1 (Düşük Risk Grubu)**
*   African-American Ortalama Skor: **2.54**
*   Caucasian Ortalama Skor: **1.67**

---
*Rapor Sonu*
