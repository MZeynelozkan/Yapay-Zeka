# ----------------------------
# Kütüphanelerin Yüklenmesi
# ----------------------------
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# ----------------------------
# Veri Setinin Yüklenmesi
# ----------------------------
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin",
           "BMI","DiabetesPedigreeFunction","Age","Outcome"]
veriler = pd.read_csv(url, names=columns)

# ----------------------------
# Label Encoding
# ----------------------------
label_encoder = LabelEncoder().fit(veriler.Outcome)
labels = label_encoder.transform(veriler.Outcome)
classes = list(label_encoder.classes_)

# ----------------------------
# Bağımsız ve bağımlı değişkenler
# ----------------------------
x = veriler.drop(["Outcome"], axis=1)
y = labels

# ----------------------------
# Standardizasyon
# ----------------------------
sc = StandardScaler()
x = sc.fit_transform(x)

# ----------------------------
# Eğitim/Test
# ----------------------------
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# ----------------------------
# One-Hot Encoding
# ----------------------------
y_train = to_categorical(y_train, num_classes=len(classes))
y_test = to_categorical(y_test, num_classes=len(classes))

# ----------------------------
# Model
# ----------------------------
model = Sequential()
model.add(Dense(17, activation='relu', input_dim=8))
model.add(Dense(8, activation='relu'))
model.add(Dense(len(classes), activation='softmax'))

# ----------------------------
# Model Derleme
# ----------------------------
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ----------------------------
# Model Eğitimi
# ----------------------------
history = model.fit(
    x_train, y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

# ----------------------------
# Test
# ----------------------------
loss, accuracy = model.evaluate(x_test, y_test)
print("Test Kaybı:", loss)
print("Test Doğruluğu:", accuracy)

# ----------------------------
# Karmaşıklık Matrisi
# ----------------------------
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)
cm = confusion_matrix(y_true, y_pred_classes)
print("Karmaşıklık Matrisi:\n", cm)

# ----------------------------
# Grafikler
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
