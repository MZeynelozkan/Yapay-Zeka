# === GEREKLİ KÜTÜPHANELER ===
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import RMSprop

# === TEMEL PARAMETRELER ===
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_CHANNELS = 3
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
BATCH_SIZE = 32
EPOCHS = 10

# === VERİ SETİNİ OKU ===
# Eğer dosyalar tek klasördeyse (örneğin "dataset/"), klasör yolunu buraya yaz.
dataset_dir = "dataset"  # veya data/train gibi

# Dosya isimlerini al
filenames = []
categories = []

for category in ['cats', 'dogs']:
    path = os.path.join(dataset_dir, category)
    for file in os.listdir(path):
        filenames.append(os.path.join(path, file))
        categories.append(category)

df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})

# Veriyi train ve validation olarak ayır
train_df, validate_df = train_test_split(df, test_size=0.2, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)

total_train = train_df.shape[0]
total_validate = validate_df.shape[0]

print(f"Toplam eğitim görüntüsü: {total_train}")
print(f"Toplam doğrulama görüntüsü: {total_validate}")

# === GÖRSEL ÖNİŞLEME ===
train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=BATCH_SIZE
)

validation_generator = validation_datagen.flow_from_dataframe(
    validate_df,
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=BATCH_SIZE
)

# === MODEL TANIMI ===
model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(
    loss='categorical_crossentropy',
    optimizer=RMSprop(learning_rate=0.0001),
    metrics=['accuracy']
)

model.summary()

# === MODELİ EĞİT ===
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    steps_per_epoch=total_train // BATCH_SIZE,
    validation_steps=total_validate // BATCH_SIZE
)

# === MODELİ KAYDET ===
model.save('cats_dogs_model.h5')
print("✅ Model kaydedildi: cats_dogs_model.h5")

# === EĞİTİM GRAFİĞİ ===
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.legend()
plt.title('Eğitim Doğruluk Grafiği')
plt.show()
