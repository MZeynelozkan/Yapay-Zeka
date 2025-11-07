import os
import shutil
import random

# Mevcut dizin (cat ve dog klasörlerinin olduğu yer)
base_dir = "dataset"  # örnek: "dataset" veya "data" olabilir

# Yeni klasör yapısı
new_base_dir = "data"
train_dir = os.path.join(new_base_dir, "train")
val_dir = os.path.join(new_base_dir, "validation")

# Gerekli klasörleri oluştur
for folder in [train_dir, val_dir]:
    os.makedirs(os.path.join(folder, "cats"), exist_ok=True)
    os.makedirs(os.path.join(folder, "dogs"), exist_ok=True)

# Eğitim/validation oranı
split_ratio = 0.8

def split_and_copy(label):
    source_dir = os.path.join(base_dir, label + "s")  # cats veya dogs
    files = os.listdir(source_dir)
    random.shuffle(files)

    split = int(len(files) * split_ratio)
    train_files = files[:split]
    val_files = files[split:]

    for f in train_files:
        src = os.path.join(source_dir, f)
        dst = os.path.join(train_dir, label + "s", f"{label}_{f}")
        shutil.copy(src, dst)

    for f in val_files:
        src = os.path.join(source_dir, f)
        dst = os.path.join(val_dir, label + "s", f"{label}_{f}")
        shutil.copy(src, dst)

    print(f"{label.capitalize()} klasöründen {len(train_files)} eğitim, {len(val_files)} doğrulama resmi kopyalandı.")

# Çalıştır
split_and_copy("cat")
split_and_copy("dog")

print("\n✅ Dosyalar başarıyla ayrıldı ve isimler düzenlendi!")
