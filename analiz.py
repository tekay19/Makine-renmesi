from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the dataset
cancer = load_breast_cancer()

# Create a DataFrame
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df["target"] = cancer.target

# Define features and target
X = cancer.data
y = cancer.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
#Neden scale kullanıyoruz?KNN gibi mesafe tabanlı algoritmalarda eğer bir özellik çok büyük sayılar içeriyorsa mesafe hesaplamasında baskın hale gelir daha önemli gözükür.
#örneği bir boy 150 kilo 40 olsa boy değeri yüksek kilo değeri küçük kalıyor ve önem dengesi farklı oluyır.Biz modelin daha sağlıklı bir tahmin yapması için
#tüm sütunlar eşit önemde olmalı
scaler = StandardScaler()#özellikleri saynı öneme getirir
X_train = scaler.fit_transform(X_train)#fit_transform ile burdaki fit kullandık çünkü tüm eğitim verisinin standart sapmasını hesaplar.transform ise bu istatistiklere göre veriyi dönüştürür.(Ölçekler)
#fit_transform ile verinin standart sapmasını hesaplayıp bu istatistikelre göre veriyi dönüştürürüz.İkisi aynı olur
X_test = scaler.transform(X_test)


# Initialize and train the KNN model
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Doğruluk={accuracy}")

accuracy_values = []
k_values = []

for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_values.append(accuracy)
    k_values.append(k)
    print(f"K={k} Doğruluk={accuracy}")

plt.figure()
plt.scatter(k_values, accuracy_values, color='red')
plt.xlabel("K")
plt.ylabel("Doğruluk")
plt.title("Farklı K Değerlerine Göre Doğruluk Oranı")
plt.savefig("dogruluk_grafigi.png")
print("Grafik dogruluk_grafigi.png olarak kaydedildi.", flush=True)