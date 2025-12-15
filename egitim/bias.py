import kagglehub
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import shutil

# Mevcut çalışma dizini
current_dir = os.path.dirname(os.path.abspath(__file__))
# Hedef data klasörü
target_dir = os.path.join(current_dir, 'compass_data')

# Download latest version
print("Dataset indiriliyor/kontrol ediliyor...")
path = kagglehub.dataset_download("danofer/compass")
print("KaggleHub Path:", path)

# Hedef klasör yoksa oluştur
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# Dosyaları kopyala
print(f"Dosyalar {target_dir} klasörüne kopyalanıyor...")
copied_files = []
for file_name in os.listdir(path):
    full_file_name = os.path.join(path, file_name)
    if os.path.isfile(full_file_name):
        shutil.copy(full_file_name, target_dir)
        copied_files.append(file_name)
        print(f"Kopyalandı: {file_name}")

print("\n--- İşlem Tamamlandı ---")
print(f"Veri seti dosyaları şu klasörde: {target_dir}")
print("Dosyalar:", copied_files)
# ========================================
# ADIM 0: VERİ YÜKLEME
# ========================================
csv_path = os.path.join(target_dir, 'cox-violent-parsed.csv')
df = pd.read_csv(csv_path)

# Gerekli sütunları seç
cols_to_keep = ['sex', 'age', 'age_cat', 'race',
                'juv_fel_count', 'juv_misd_count', 'juv_other_count',
                'priors_count', 'c_charge_degree',
                'days_b_screening_arrest',
                'decile_score', 'score_text', 
                'is_recid']

df = df[cols_to_keep].dropna()

print(f"Veri boyutu: {df.shape}")
print(f"\nIrk dağılımı:\n{df['race'].value_counts()}")

# ========================================
# ADIM 1: Skor Dağılımı (Hangi ırk ne alıyor?)
# ========================================
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
df.groupby('race')['decile_score'].mean().plot(kind='bar')
plt.title('Ortalama Risk Skoru (Irklar Arası)')
plt.ylabel('Ortalama Skor')

plt.subplot(1, 2, 2)
sns.boxplot(data=df, x='race', y='decile_score')
plt.title('Skor Dağılımı')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(target_dir, 'bias_analysis_score_dist.png'))
print(f"Grafik kaydedildi: {os.path.join(target_dir, 'bias_analysis_score_dist.png')}")
# plt.show()

# ========================================
# ADIM 2: False Positive/Negative Rate
# ========================================
def calculate_confusion_metrics(data, race_name):
    """Belirli bir ırk için confusion metrics hesapla"""
    race_data = data[data['race'] == race_name].copy()
    
    # Yüksek risk: decile_score > 5
    race_data['predicted_high_risk'] = (race_data['decile_score'] > 5).astype(int)
    
    # Gerçek sonuç: is_recid
    actual = race_data['is_recid']
    predicted = race_data['predicted_high_risk']
    
    # True Positive: Yüksek risk dedi, gerçekten suç işledi
    tp = ((predicted == 1) & (actual == 1)).sum()
    # False Positive: Yüksek risk dedi, suç işlemedi ❌
    fp = ((predicted == 1) & (actual == 0)).sum()
    # True Negative: Düşük risk dedi, suç işlemedi
    tn = ((predicted == 0) & (actual == 0)).sum()
    # False Negative: Düşük risk dedi, suç işledi ❌
    fn = ((predicted == 0) & (actual == 1)).sum()
    
    # Metrikler
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    return {
        'race': race_name,
        'FPR': fpr,  # Suçsuz olana "suçlu" deme oranı
        'FNR': fnr,  # Suçlu olana "suçsuz" deme oranı
        'Accuracy': accuracy,
        'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
        'Total': tp + fp + tn + fn
    }

# Tüm ırklar için hesapla
races = df['race'].unique()
results = []

for race in races:
    results.append(calculate_confusion_metrics(df, race))

results_df = pd.DataFrame(results)
print("\n=== CONFUSION METRICS (IRKLAR ARASI) ===")
print(results_df[['race', 'FPR', 'FNR', 'Accuracy', 'Total']])

# Görselleştir
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
x = np.arange(len(results_df))
width = 0.35

plt.bar(x - width/2, results_df['FPR'], width, label='False Positive Rate', color='red', alpha=0.7)
plt.bar(x + width/2, results_df['FNR'], width, label='False Negative Rate', color='blue', alpha=0.7)

plt.xlabel('Irk')
plt.ylabel('Oran')
plt.title('Hata Oranları Karşılaştırması')
plt.xticks(x, results_df['race'], rotation=45, ha='right')
plt.legend()

plt.subplot(1, 2, 2)
results_df.plot(x='race', y='Accuracy', kind='bar', ax=plt.gca(), color='green', alpha=0.7)
plt.title('Model Doğruluğu (Irklar Arası)')
plt.ylabel('Accuracy')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(target_dir, 'bias_analysis_metrics.png'))
print(f"Grafik kaydedildi: {os.path.join(target_dir, 'bias_analysis_metrics.png')}")
# plt.show()

# ========================================
# ADIM 3: Kontrol Değişkenleri ile Analiz
# ========================================
age_groups = [(18, 25), (25, 35), (35, 45), (45, 100)]
priors_groups = [(0, 1), (2, 5), (6, 100)]

print("\n=== KONTROL DEĞİŞKENLERİ ANALİZİ ===")
for age_min, age_max in age_groups:
    for priors_min, priors_max in priors_groups:
        subset = df[
            (df['age'] >= age_min) & (df['age'] < age_max) &
            (df['priors_count'] >= priors_min) & (df['priors_count'] <= priors_max)
        ]
        
        if len(subset) > 50:  # Yeterli veri varsa
            print(f"\nYaş {age_min}-{age_max}, Önceki Suç {priors_min}-{priors_max} (N={len(subset)}):")
            race_scores = subset.groupby('race')['decile_score'].mean()
            print(race_scores)

# ========================================
# ADIM 4: İSTATİSTİKSEL TESTLER
# ========================================
print("\n=== İSTATİSTİKSEL TESTLER ===")

# African-American vs Caucasian karşılaştırması
aa_scores = df[df['race'] == 'African-American']['decile_score']
cauc_scores = df[df['race'] == 'Caucasian']['decile_score']

# T-test
t_stat, p_value = stats.ttest_ind(aa_scores, cauc_scores)
print(f"\nT-Test (African-American vs Caucasian):")
print(f"  Ortalama fark: {aa_scores.mean() - cauc_scores.mean():.2f}")
print(f"  T-statistic: {t_stat:.4f}")
print(f"  P-value: {p_value:.6f}")
print(f"  Anlamlı mı? {'✅ EVET (p < 0.05)' if p_value < 0.05 else '❌ HAYIR'}")

# Chi-square test (Yüksek risk oranları)
df['high_risk'] = (df['decile_score'] > 5).astype(int)
contingency_table = pd.crosstab(df['race'], df['high_risk'])
chi2, p_chi, dof, expected = stats.chi2_contingency(contingency_table)

print(f"\nChi-Square Test (Irk vs Yüksek Risk):")
print(f"  Chi-square: {chi2:.4f}")
print(f"  P-value: {p_chi:.6f}")
print(f"  Anlamlı mı? {'✅ EVET (p < 0.05)' if p_chi < 0.05 else '❌ HAYIR'}")

# ========================================
# ADIM 5: ÖZELLİK ÖNEMİ ANALİZİ
# ========================================
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

print("\n=== ÖZELLİK ÖNEMİ ANALİZİ ===")

# Kategorik değişkenleri encode et
df_encoded = df.copy()
le_sex = LabelEncoder()
le_race = LabelEncoder()
le_charge = LabelEncoder()

df_encoded['sex'] = le_sex.fit_transform(df_encoded['sex'])
df_encoded['race'] = le_race.fit_transform(df_encoded['race'])
df_encoded['c_charge_degree'] = le_charge.fit_transform(df_encoded['c_charge_degree'])

# Model oluştur
features = ['age', 'sex', 'race', 'priors_count', 'juv_fel_count', 'c_charge_degree']
X = df_encoded[features]
y = df_encoded['is_recid']

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# Özellik önemleri
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nÖzellik Önem Sıralaması:")
print(feature_importance)

# Görselleştir
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Önem Skoru')
plt.title('Hangi Özellik Tahmine Ne Kadar Etki Ediyor?')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(target_dir, 'bias_analysis_feature_imp.png'))
print(f"Grafik kaydedildi: {os.path.join(target_dir, 'bias_analysis_feature_imp.png')}")
# plt.show()

# ========================================
# ÖZET RAPOR
# ========================================
print("\n" + "="*60)
print("ÖZET RAPOR: COMPAS BİAS ANALİZİ")
print("="*60)

print(f"\n1. SKOR ORTALAMARI:")
for race in ['African-American', 'Caucasian']:
    if race in df['race'].values:
        avg_score = df[df['race'] == race]['decile_score'].mean()
        print(f"   {race}: {avg_score:.2f}")

print(f"\n2. FALSE POSITIVE RATES:")
for _, row in results_df.iterrows():
    if row['race'] in ['African-American', 'Caucasian']:
        print(f"   {row['race']}: {row['FPR']:.2%}")

print(f"\n3. EN ÖNEMLİ ÖZELLİKLER:")
for _, row in feature_importance.head(3).iterrows():
    print(f"   {row['feature']}: {row['importance']:.4f}")

print(f"\n4. İSTATİSTİKSEL ANLAM:")
print(f"   Irk ve skor arasında anlamlı ilişki var mı? {'✅ EVET' if p_chi < 0.05 else '❌ HAYIR'}")