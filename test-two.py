# ----------------------------
# Gerekli Kütüphanelerin Yüklenmesi
# ----------------------------
import numpy as np                             # Sayısal hesaplamalar ve array işlemleri
import pandas as pd                            # Veri manipülasyonu ve DataFrame kullanımı
from sklearn.preprocessing import LabelEncoder, StandardScaler
# LabelEncoder → kategorik hedefleri sayısala çevirir
# StandardScaler → özellikleri standart normal dağılıma getirir (ortalama=0, std=1)
from sklearn.model_selection import KFold, cross_val_score
# KFold → k-fold çapraz doğrulama için
# cross_val_score → modelin her fold üzerindeki doğruluğunu hesaplar
from scikeras.wrappers import KerasClassifier
# Keras modelini scikit-learn API'si ile kullanmak için
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# Sinir ağı modeli ve katmanları

# ----------------------------
# Veri Setinin Hazırlanması
# ----------------------------
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin",
           "BMI","DiabetesPedigreeFunction","Age","Outcome"]

# CSV dosyasını indirip pandas DataFrame olarak oku
veriler = pd.read_csv(url, names=columns)

# Outcome sütununu sayısala çevir (0/1)
label_encoder = LabelEncoder().fit(veriler.Outcome)
y = label_encoder.transform(veriler.Outcome)
# y → hedef değişken (target), shape=(768,) yani 1 boyutlu
# Not: cross_val_score ve sparse_categorical_crossentropy ile uyumlu

# Özellikleri ayır
x = veriler.drop(["Outcome"], axis=1)
# x → bağımsız değişkenler (features), shape=(768,8)

# Özellikleri standartlaştır
sc = StandardScaler()
x = sc.fit_transform(x)
# StandardScaler: her sütunun ortalamasını 0, standart sapmasını 1 yapar
# Sinir ağı eğitimini stabilize eder ve daha hızlı öğrenme sağlar

# ----------------------------
# Model Fonksiyonu
# ----------------------------
def create_model():
    # Sequential model: katmanları üst üste ekleme yöntemi
    model = Sequential()
    model.add(Dense(17, activation='relu', input_dim=8))
    # 1. gizli katman: 17 nöron
    # input_dim=8 → 8 özellik var
    # ReLU aktivasyonu → negatifleri 0 yapar, pozitifleri geçirir

    model.add(Dense(8, activation='relu'))
    # 2. gizli katman: 8 nöron, ReLU aktivasyonu

    model.add(Dense(2, activation='softmax'))
    # Çıkış katmanı: 2 nöron (2 sınıf için)
    # Softmax → her sınıfa ait olasılık üretir, toplamı 1 olur

    # Modeli derle
    model.compile(
        optimizer='adam', 
        loss='sparse_categorical_crossentropy',  
        # sparse_categorical_crossentropy → hedef 1 boyutlu iken kullanılabilir
        metrics=['accuracy']
    )
    return model

# Keras modelini scikit-learn wrapper ile sar
model = KerasClassifier(
    model=create_model,  # yukarıdaki fonksiyon ile model oluşturulur
    epochs=50,           # her fold için 50 epoch eğitim yapılacak
    batch_size=16,       # her seferde 16 örnek ile ağırlık güncelleme
    verbose=0            # eğitim sırasında konsola log yazma
)

# ----------------------------
# 5-Fold Çapraz Doğrulama
# ----------------------------
kfold = KFold(
    n_splits=5,        # veri 5 eşit parçaya bölünecek
    shuffle=True,      # bölmeden önce veriyi karıştır
    random_state=42    # rastgelelik kontrolü için sabit sayı
)

# cross_val_score → modelin her fold üzerindeki doğruluğunu hesaplar
# İşleyiş:
# 1. Veri 5 eşit parçaya bölünür
# 2. Her seferinde 1 parça test, kalan 4 parça eğitim için kullanılır
# 3. Toplam 5 kez model eğitilir ve doğruluklar kaydedilir
results = cross_val_score(model, x, y, cv=kfold)

# ----------------------------
# Sonuçların Yazdırılması
# ----------------------------
print("Her fold doğrulukları:", results)
# Her fold'un doğruluğunu gösterir
print("Ortalama doğruluk:", results.mean())
# 5 fold'un ortalama doğruluğu → modelin genel başarısı
