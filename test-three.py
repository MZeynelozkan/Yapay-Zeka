# ----------------------------
# Kütüphaneler
# ----------------------------
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# ----------------------------
# Veri Setini Yükleme
# ----------------------------
veriler = pd.read_csv("data.csv")  # Aynı klasördeki dosya

# ----------------------------
# Özellikler ve Hedef
# ----------------------------
x = veriler.drop(["price_range"], axis=1)
y = veriler["price_range"]

# ----------------------------
# Standartlaştırma
# ----------------------------
sc = StandardScaler()
x = sc.fit_transform(x)

# ----------------------------
# Train/Test Split
# ----------------------------
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# ----------------------------
# One-Hot Encoding
# ----------------------------
classes = sorted(y.unique())
print(len(classes))
y_train = to_categorical(y_train, num_classes=len(classes))
y_test = to_categorical(y_test, num_classes=len(classes))

print("Benzersiz Price Range değerleri:", classes)
print("Toplam sınıf sayısı:", len(classes))

# ----------------------------
# Modelin Oluşturulması
# ----------------------------
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=x_train.shape[1]))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(classes), activation='softmax'))  # 4 sınıf

# ----------------------------
# Model Özeti
# ----------------------------
model.summary()

# ----------------------------
# Modelin Derlenmesi
# ----------------------------
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ----------------------------
# Modelin Eğitilmesi
# ----------------------------
history = model.fit(
    x_train, y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

# ----------------------------
# Test Verisi Üzerinde Değerlendirme
# ----------------------------
loss, accuracy = model.evaluate(x_test, y_test)
print("Test Kaybı:", loss)
print("Test Doğruluğu:", accuracy)

# ----------------------------
# Eğitim ve Doğrulama Grafikleri
# ----------------------------
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Doğruluğu')
plt.ylabel('Doğruluk')
plt.xlabel('Epoch')
plt.legend(['Eğitim', 'Doğrulama'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Kaybı')
plt.ylabel('Kayıp')
plt.xlabel('Epoch')
plt.legend(['Eğitim', 'Doğrulama'], loc='upper left')
plt.show()
