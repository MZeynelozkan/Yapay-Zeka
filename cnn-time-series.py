# Gerekli kütüphanelerin yüklenmesi
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv1D, Dropout, MaxPooling1D, Flatten
import matplotlib.pyplot as plt

# =====================================
# VERİLERİN YÜKLENMESİ
# =====================================
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Hedef etiketin dönüştürülmesi
label_encoder = LabelEncoder().fit(train.species)
labels = label_encoder.transform(train.species)
classes = list(label_encoder.classes_)

# Gereksiz sütunların kaldırılması
train = train.drop(["id", "species"], axis=1)
test = test.drop("id", axis=1)

nb_features = train.shape[1]
nb_classes = len(classes)

# =====================================
# VERİLERİN ÖLÇEKLENMESİ
# =====================================
scaler = StandardScaler().fit(train.values)
train_scaled = scaler.transform(train.values)
test_scaled = scaler.transform(test.values)

# =====================================
# EĞİTİM VE DOĞRULAMA VERİSİNE AYIRMA
# =====================================
X_train, X_valid, y_train, y_valid = train_test_split(train_scaled, labels, test_size=0.1, random_state=42)

# =====================================
# KATEGORİK DÖNÜŞÜM
# =====================================
y_train = to_categorical(y_train, nb_classes)
y_valid = to_categorical(y_valid, nb_classes)

# CNN giriş şekline uygun hale getirme (örnek sayısı, özellik sayısı, 1 kanal)
X_train = X_train.reshape(X_train.shape[0], nb_features, 1)
X_valid = X_valid.reshape(X_valid.shape[0], nb_features, 1)

# =====================================
# MODELİN OLUŞTURULMASI
# =====================================
model = Sequential()
model.add(Conv1D(512, 1, input_shape=(nb_features, 1)))
model.add(Activation("relu"))
model.add(MaxPooling1D(2))
model.add(Conv1D(256, 1))
model.add(MaxPooling1D(2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(2048, activation="relu"))
model.add(Dense(1024, activation="relu"))
model.add(Dense(nb_classes, activation="softmax"))

model.summary()

# =====================================
# MODELİN DERLENMESİ VE EĞİTİLMESİ
# =====================================
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

history = model.fit(X_train, y_train, epochs=15, validation_data=(X_valid, y_valid))

# =====================================
# ORTALAMA DEĞERLERİN GÖSTERİLMESİ
# =====================================
print("Ortalama Eğitim Kaybı: ", np.mean(history.history["loss"]))
print("Ortalama Eğitim Başarımı: ", np.mean(history.history["accuracy"]))
print("Ortalama Doğrulama Kaybı: ", np.mean(history.history["val_loss"]))
print("Ortalama Doğrulama Başarımı: ", np.mean(history.history["val_accuracy"]))

# =====================================
# SONUÇLARIN GRAFİK ÜZERİNDE GÖSTERİLMESİ
# =====================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

# Kayıp grafiği
ax1.plot(history.history["loss"], color='g', label="Eğitim kaybı")
ax1.plot(history.history["val_loss"], color='y', label="Doğrulama kaybı")
ax1.set_title("Eğitim ve Doğrulama Kayıpları")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Kayıp")
ax1.legend()

# Başarım grafiği
ax2.plot(history.history["accuracy"], color='b', label="Eğitim başarımı")
ax2.plot(history.history["val_accuracy"], color='r', label="Doğrulama başarımı")
ax2.set_title("Eğitim ve Doğrulama Başarımları")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Başarım")
ax2.legend()

plt.tight_layout()
plt.show()
