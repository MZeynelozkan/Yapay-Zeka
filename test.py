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
           "BMI","DiabetesPedigreeFunction","Age","Outcome"]  

# ----------------------------
# Veri Setinin Pandas DataFrame olarak okunması
# ----------------------------
veriler = pd.read_csv(url, names=columns)  

# ----------------------------
# LabelEncoder Oluşturulması ve Eğitilmesi
# ----------------------------
label_encoder = LabelEncoder().fit(veriler.Outcome)
# 'Pregnancies' sütunundaki tüm benzersiz değerleri öğrenir (fit)

# ----------------------------
# Verilerin Sayısala Çevrilmesi
# ----------------------------
labels = label_encoder.transform(veriler.Outcome)
# Her gözlemi sayısal değere dönüştürür

# ----------------------------
# Benzersiz Sınıfların Listelenmesi
# ----------------------------
classes = list(label_encoder.classes_)
# Sütundaki benzersiz değerleri listeler
print("Benzersiz Pregnancies değerleri:", classes)
print("Toplam sınıf sayısı:", len(classes))

# ----------------------------
# Bağımsız ve bağımlı değişkenlerin ayrılması
# ----------------------------
x = veriler.drop(["Outcome"], axis=1)  
# 'Pregnancies' sütunu çıkarıldı (tahmin edilecek hedef)

y = labels
# y → hedef değer (encoded Pregnancies)

# ----------------------------
# Özelliklerin Standardizasyonu
# ----------------------------
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)
# Tüm özellikler ortalaması 0, standart sapması 1 olacak şekilde normalize edildi

# ----------------------------
# Eğitim ve Test Verilerinin Hazırlanması
# ----------------------------
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
# %80 eğitim, %20 test olarak ayrıldı

# ----------------------------
# Çıktı Değerlerinin One-Hot Kodlanması
# ----------------------------
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train, num_classes=len(classes))
y_test = to_categorical(y_test, num_classes=len(classes))
# y değerlerini kategorik hale çeviriyoruz
# Örn: 0 → [1,0,0,...], 1 → [0,1,0,...] 
# Softmax ile tüm benzersiz sınıfları tahmin edebilmek için gerekli

# ----------------------------
# Modelin Oluşturulması
# ----------------------------
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(17, activation='relu', input_dim=8))
# 1. gizli katman, 17 nöron
# input_dim=8 → x_train'in 8 sütunu var

model.add(Dense(8, activation='relu'))
# 2. gizli katman, 8 nöron

model.add(Dense(len(classes), activation='softmax'))
# çıkış katmanı, sınıf sayısı kadar nöron
# softmax → olasılık değerleri üretir, toplamı 1 olur

# ----------------------------
# Model Özeti
# ----------------------------
model.summary()
# Katmanlar, nöron sayıları ve parametreleri gösterir


# ----------------------------
# Modelin Derlenmesi
# ----------------------------
model.compile(
    optimizer='adam',                      # Ağırlıkları güncellemek için Adam optimizasyonu
    loss='categorical_crossentropy',       # Çok sınıflı sınıflandırma için kayıp fonksiyonu
    metrics=['accuracy']                   # Eğitim ve test sırasında doğruluk ölçümü
)

# ----------------------------
# Modelin Eğitilmesi
# ----------------------------
history = model.fit(
    x_train, y_train,                      # Eğitim verisi
    epochs=50,                             # Tüm eğitim verisini 50 kez geç
    batch_size=16,                         # Her seferde 16 örnek ile güncelleme yapılır
    validation_split=0.2,                  # Eğitim verisinin %20'si doğrulama için ayrılır
    verbose=1                              # Her epoch sonunda çıktı göster
)
# history → kayıp ve doğruluk değerlerini içerir

# ----------------------------
# Test Verisi Üzerinde Değerlendirme
# ----------------------------
loss, accuracy = model.evaluate(x_test, y_test)
print("Test Kaybı:", loss)
print("Test Doğruluğu:", accuracy)
# Bu, modelin daha önce görmediği test verisi üzerindeki performansını gösterir
