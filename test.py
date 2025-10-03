# ----------------------------
# Kütüphanelerin Yüklenmesi
# ----------------------------
import numpy as np                 # Sayısal hesaplamalar ve numpy array işlemleri için
import pandas as pd                # Veri manipülasyonu ve DataFrame işlemleri için
from sklearn.preprocessing import LabelEncoder  # Kategorik verileri sayısal hale çevirmek için

# ----------------------------
# Veri Setinin URL'si ve Sütun İsimleri
# ----------------------------
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin",
           "BMI","DiabetesPedigreeFunction","Age","Outcome"]  # Veri setindeki her sütunun adı

# ----------------------------
# Veri Setinin Pandas DataFrame olarak okunması
# ----------------------------
veriler = pd.read_csv(url, names=columns)  # CSV dosyasını indirip DataFrame oluştur

# ----------------------------
# LabelEncoder Oluşturulması ve Eğitilmesi
# ----------------------------
label_encoder = LabelEncoder().fit(veriler.Outcome)
# LabelEncoder: Kategorik verileri (burada Outcome: 0 veya 1) sayısal hale çevirir
# Aslında Outcome zaten 0 ve 1, ama encode edilmesi:
# - Kodlama standardı sağlar
# - Eğer veri 0 ve 1 dışında "Yes"/"No" gibi değerler içeriyorsa sayısala çevirir

# ----------------------------
# Verilerin Sayısala Çevrilmesi
# ----------------------------
labels = label_encoder.transform(veriler.Outcome)
# Outcome sütunundaki her gözlem sayısal değere çevrildi
# Örn: "Yes" → 1, "No" → 0

# ----------------------------
# Benzersiz Sınıfların Listelenmesi
# ----------------------------
classes = list(label_encoder.classes_)
print("Benzersiz Outcome değerleri:", classes)
print("Toplam sınıf sayısı:", len(classes))  # Bu veri için 2 sınıf: 0 ve 1

# ----------------------------
# Bağımsız ve bağımlı değişkenlerin ayrılması
# ----------------------------
x = veriler.drop(["Outcome"], axis=1)
# x → Modelin öğrenmesi için kullanılacak özellikler (features)
# Outcome sütunu tahmin edilmek istendiği için çıkarıldı
# axis=1 → sütun bazında sil

y = labels
# y → Hedef değişken (target) yani Outcome

# ----------------------------
# Özelliklerin Standardizasyonu
# ----------------------------
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)
# StandardScaler: tüm özellikleri standart normal dağılıma getirir
# Ortalama = 0, Standart sapma = 1
# Sinir ağı daha stabil ve hızlı öğrenir

# ----------------------------
# Eğitim ve Test Verilerinin Hazırlanması
# ----------------------------
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
# train_test_split:
# - Veriyi rastgele %80 eğitim, %20 test olarak ayırır
# - Eğitim sırasında model bu veriyi görür, test verisi ise final performansı ölçmek için saklanır

# ----------------------------
# Çıktı Değerlerinin One-Hot Kodlanması
# ----------------------------
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train, num_classes=len(classes))
y_test = to_categorical(y_test, num_classes=len(classes))
# One-hot encoding: Kategorik hedefleri vektör hâline çevirir
# Örn: 0 → [1,0], 1 → [0,1]
# Softmax çıkış katmanı ile sınıflandırma yaparken gereklidir

# ----------------------------
# Modelin Oluşturulması
# ----------------------------
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(17, activation='relu', input_dim=8))
# 1. gizli katman: 17 nöron
# input_dim=8 → x_train'in 8 sütunu var (özellik sayısı)
# activation='relu' → negatif değerler 0 olur, pozitifler geçer

model.add(Dense(8, activation='relu'))
# 2. gizli katman: 8 nöron, yine ReLU aktivasyonu

model.add(Dense(len(classes), activation='softmax'))
# Çıkış katmanı: sınıf sayısı kadar nöron (2)
# Softmax → olasılık değerleri üretir, toplamları 1 olur

# ----------------------------
# Model Özeti
# ----------------------------
model.summary()
# Modelin katman yapısı, nöron sayıları ve parametreleri gösterilir

# ----------------------------
# Modelin Derlenmesi
# ----------------------------
model.compile(
    optimizer='adam',                      # Ağırlıkları güncellemek için Adam optimizasyonu
    loss='categorical_crossentropy',       # Çok sınıflı (ya da binary one-hot) sınıflandırma kaybı
    metrics=['accuracy']                   # Eğitim ve test doğruluğunu takip et
)

# ----------------------------
# Modelin Eğitilmesi
# ----------------------------
history = model.fit(
    x_train, y_train,                      # Eğitim verisi
    epochs=50,                             # 50 kez tüm veri üzerinden geç
    batch_size=16,                         # Her seferde 16 örnek kullan
    validation_split=0.2,                  # Eğitim verisinin %20'si doğrulama için ayrılır
    verbose=1                              # Her epoch sonunda çıktı göster
)
# history: epoch başına kayıp ve doğruluk değerlerini tutar
# validation_split → overfitting kontrolü ve eğitim sırasında performans takibi

# ----------------------------
# Test Verisi Üzerinde Değerlendirme
# ----------------------------
loss, accuracy = model.evaluate(x_test, y_test)
print("Test Kaybı:", loss)
print("Test Doğruluğu:", accuracy)
# Daha önce görülmemiş test verisi üzerinde modelin performansını gösterir

# ----------------------------
# Eğitim ve Doğrulama Başarımlarının Görselleştirilmesi
# ----------------------------
import matplotlib.pyplot as plt

# Doğruluk grafiği
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Doğruluğu')
plt.ylabel('Doğruluk')
plt.xlabel('Epoch')
plt.legend(['Eğitim', 'Doğrulama'], loc='upper left')
plt.show()

# Kayıp grafiği
plt.plot(history.history['loss'])        # Eğitim kaybı
plt.plot(history.history['val_loss'])    # Doğrulama kaybı
plt.title('Model Kaybı')
plt.ylabel('Kayıp')
plt.xlabel('Epoch')
plt.legend(['Eğitim', 'Doğrulama'], loc='upper left')
plt.show()
