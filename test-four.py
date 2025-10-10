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
import pandas as pd
import simpsom as sps
from sklearn.cluster import KMeans
import numpy as np

# CSV dosya yolu
file_path = "C:/Users/Zeynel/Desktop/Yapay Zeka/airline-safety.csv"

# Veriyi oku
veri = pd.read_csv(file_path)
print(veri.head())
# => veri artık tüm sütunları ve her havayoluna ait istatistikleri içeriyor

# Analiz için kullanılacak sayısal özellikleri seçiyoruz
X = veri.drop(["airline", "avail_seat_km_per_week"], axis=1)
# => SOM sadece sayısal verilere ihtiyaç duyar, bu yüzden categorical veya string sütunları çıkarıyoruz

# SOM (Self-Organizing Map) ağı oluşturuyoruz
# 20x20 boyutunda, Periodic Boundary Conditions (PBC) aktif
# SOM ağı, girdiler arasındaki benzerlikleri öğrenmek ve 2D bir haritada gruplaşmalar oluşturmak için kullanılır
net = sps.SOMNet(20, 20, X.values, PBC=True)
# => net, SOM ağını temsil eder. Her node (düğüm) rastgele ağırlıklarla başlatılır

# SOM'u eğitiyoruz (Batch eğitim varsayılan)
net.train()
# => Bu adımda SOM, veri noktalarının birbirine yakın olduğu node’ları öğrenir
# => Amaç: benzer veriler harita üzerinde yakın node’lara yerleşir

# BMU (Best Matching Unit) nedir?
# => Her veri noktası için SOM üzerindeki en yakın node’a BMU denir
# => BMU, veri noktasının SOM haritasındaki temsilcisi gibi düşünebilirsin

# Her veri noktasının BMU'sunu buluyoruz
hrt = np.array([net.find_bmu_ix(x.reshape(1, -1))[0] for x in X.values])
# => x.reshape(1, -1) ile tek veri noktasını 2D array’e çeviriyoruz (SOM bunu istiyor)
# => find_bmu_ix() bize BMU node'unun indeksini döndürüyor
# => hrt, her satırın hangi SOM node'una en yakın olduğunu gösteren indeksler listesi

# BMU indekslerini kullanarak KMeans ile 3 kümeye ayırıyoruz
kmeans = KMeans(n_clusters=3, max_iter=300, random_state=0)
y_kmeans = kmeans.fit_predict(hrt.reshape(-1, 1))  # KMeans için 2D array
# => KMeans, BMU indekslerini kullanarak benzer node’ları gruplara ayırıyor
# => y_kmeans her satırın hangi kümeye ait olduğunu gösteriyor

# Küme etiketlerini kontrol ediyoruz
print("Küme etiketleri:", y_kmeans)
print("Küme etiketleri sayısı:", len(y_kmeans))

# Orijinal veri frame'ine küme etiketlerini ekliyoruz
veri["cluster"] = kmeans.labels_
# => veri artık hangi havayolunun hangi kümeye ait olduğunu içeriyor

# Küme bazlı örnek veri incelemesi
print("Cluster 0 ilk 5 satır:\n", veri[veri["cluster"] == 0].head(5))
print("Cluster 1 ilk 5 satır:\n", veri[veri["cluster"] == 1].head(5))
print("Cluster 2 ilk 5 satır:\n", veri[veri["cluster"] == 2].head(5))
