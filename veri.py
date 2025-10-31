import pandas as pd
import numpy as np

# --- Parametreler ---
num_train = 1000  # eğitim örneği sayısı
num_test = 300    # test örneği sayısı
num_features = 192  # özellik sayısı
num_classes = 10    # sınıf sayısı

np.random.seed(42)

# --- Özelliklerin oluşturulması ---
train_features = np.random.rand(num_train, num_features)
test_features = np.random.rand(num_test, num_features)

# --- Sınıf isimleri (örnek olarak species_0, species_1, ... ) ---
species_list = [f"species_{i}" for i in range(num_classes)]
train_species = np.random.choice(species_list, num_train)

# --- DataFrame oluştur ---
train_df = pd.DataFrame(train_features, columns=[f"feature_{i}" for i in range(num_features)])
train_df.insert(0, "id", np.arange(1, num_train + 1))
train_df["species"] = train_species  # hedef sütunu ekleniyor ✅

test_df = pd.DataFrame(test_features, columns=[f"feature_{i}" for i in range(num_features)])
test_df.insert(0, "id", np.arange(1, num_test + 1))

# --- Dosyaları kaydet ---
train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)

print("✅ Yapay veri seti başarıyla oluşturuldu:")
print(f"- train.csv (içinde {num_train} örnek, {num_features} özellik ve 'species' sütunu var)")
print(f"- test.csv (içinde {num_test} örnek, {num_features} özellik var)")
