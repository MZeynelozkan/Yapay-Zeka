# =============================================
# GEREKLİ KÜTÜPHANELER
# =============================================
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Dropout, MaxPooling1D, Flatten, Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes

# =============================================
# 1️⃣ VERİ SETİNİN YAPAY OLARAK OLUŞTURULMASI (örnek)
# =============================================
np.random.seed(42)
num_samples = 1000
num_features = 90
num_classes = 5

train_data = pd.DataFrame(
    np.random.randn(num_samples, num_features),
    columns=[f"feature_{i}" for i in range(num_features)]
)
train_data["species"] = np.random.choice(
    ["class_A", "class_B", "class_C", "class_D", "class_E"],
    size=num_samples
)
train_data["id"] = np.arange(num_samples)

test_data = pd.DataFrame(
    np.random.randn(200, num_features),
    columns=[f"feature_{i}" for i in range(num_features)]
)
test_data["id"] = np.arange(200)

train_data.to_csv("train.csv", index=False)
test_data.to_csv("test.csv", index=False)

# =============================================
# 2️⃣ VERİLERİN YÜKLENMESİ VE HAZIRLANMASI
# =============================================
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

label_encoder = LabelEncoder().fit(train.species)
labels = label_encoder.transform(train.species)
classes = list(label_encoder.classes_)

train = train.drop(["id", "species"], axis=1)
test = test.drop("id", axis=1)

nb_features = train.shape[1]
nb_classes = len(classes)

scaler = StandardScaler().fit(train.values)
train_scaled = scaler.transform(train.values)
test_scaled = scaler.transform(test.values)

# =============================================
# 3️⃣ ÇAPRAZ DOĞRULAMA (K-Fold Cross Validation)
# =============================================
kf = KFold(n_splits=5, shuffle=True, random_state=42)
acc_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(train_scaled)):
    print(f"\nFold {fold+1} başlıyor...")
    
    X_train, X_val = train_scaled[train_idx], train_scaled[val_idx]
    y_train, y_val = labels[train_idx], labels[val_idx]
    
    y_train = to_categorical(y_train, nb_classes)
    y_val = to_categorical(y_val, nb_classes)
    
    X_train = X_train.reshape(X_train.shape[0], nb_features, 1)
    X_val = X_val.reshape(X_val.shape[0], nb_features, 1)
    
    model = Sequential([
        Conv1D(256, 1, activation='relu', input_shape=(nb_features, 1)),
        MaxPooling1D(2),
        Dropout(0.25),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(nb_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=10, verbose=0, validation_data=(X_val, y_val))
    
    acc = model.evaluate(X_val, y_val, verbose=0)[1]
    acc_scores.append(acc)
    print(f"Fold {fold+1} doğruluk: {acc:.4f}")

print("\n✅ Ortalama çapraz doğrulama doğruluğu:", np.mean(acc_scores))

# =============================================
# 4️⃣ F1, Precision, Recall, Specificity
# =============================================
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_val, axis=1)

print("\n🔍 Sınıflandırma Raporu:")
print(classification_report(y_true, y_pred_classes, digits=4))

# Confusion Matrix ve Specificity
cm = confusion_matrix(y_true, y_pred_classes)
TN = np.sum(cm) - (np.sum(cm, axis=0) + np.sum(cm, axis=1) - np.diag(cm)).sum()
FP = np.sum(cm) - TN - np.sum(np.diag(cm))
specificity = TN / (TN + FP)
print(f"Specificity: {specificity:.4f}")

# =============================================
# 5️⃣ DİYABET VERİSİ İLE 1D CNN MODELİ (1DESA)
# =============================================
print("\n🩸 Diyabet veri seti ile model eğitimi...")
diabetes = load_diabetes()
X = diabetes.data
y = (diabetes.target > diabetes.target.mean()).astype(int)

scaler = StandardScaler()
X = scaler.fit_transform(X)
X = X.reshape(X.shape[0], X.shape[1], 1)
y = to_categorical(y, 2)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model2 = Sequential([
    Conv1D(32, 2, activation='relu', input_shape=(X.shape[1], 1)),
    MaxPooling1D(2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')
])

model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model2.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val), verbose=0)

# Metrikler
y_pred = model2.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_val, axis=1)

f1 = f1_score(y_true, y_pred_classes)
precision = precision_score(y_true, y_pred_classes)
recall = recall_score(y_true, y_pred_classes)

print(f"\nDiyabet doğruluk: {model2.evaluate(X_val, y_val, verbose=0)[1]:.4f}")
print(f"F1 Skoru: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

# Grafikler
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Eğitim')
plt.plot(history.history['val_accuracy'], label='Doğrulama')
plt.title('Doğruluk')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.title('Kayıp')
plt.legend()
plt.show()

# =============================================
# 6️⃣ FARKLI OPTİMİZASYON ALGORİTMALARI KARŞILAŞTIRMASI
# =============================================
optimizers = {
    "Adam": Adam(),
    "SGD": SGD(),
    "RMSprop": RMSprop(),
    "Adagrad": Adagrad()
}

results = {}

for name, opt in optimizers.items():
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1], 1)),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=10, verbose=0, validation_data=(X_val, y_val))
    val_acc = np.mean(history.history['val_accuracy'])
    results[name] = val_acc
    print(f"{name} doğrulama başarımı: {val_acc:.4f}")

best = max(results, key=results.get)
print(f"\n🏆 En iyi optimizer: {best} ({results[best]:.4f})")
