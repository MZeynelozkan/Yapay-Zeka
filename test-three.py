# =============================================
#  Uygulama 5 – SOM ile Uçak Güvenliği Kümeleme (MiniSom + KaggleHub)
#  (Yalnızca terminal çıktısı)
# =============================================

# pip install minisom scikit-learn pandas kagglehub[pandas-datasets]

import pandas as pd
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import kagglehub
from kagglehub import KaggleDatasetAdapter

# 1️⃣ Kaggle'dan veri setini indir
print("📥 Veri Kaggle'dan indiriliyor...")
file_path = "airline-safety.csv"  # veri seti içindeki dosya adı

df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "fivethirtyeight/fivethirtyeight-airline-safety-dataset",
    file_path
)

print("✅ Veri başarıyla yüklendi!\n")

# 2️⃣ Sayısal veriyi ayır
airlines = df['airline']
X = df.drop(columns=['airline'])

# 3️⃣ Veriyi normalize et
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 4️⃣ SOM ağı oluştur
som = MiniSom(
    x=20, y=20,
    input_len=X_scaled.shape[1],
    sigma=1.0, learning_rate=0.01,
    neighborhood_function='gaussian',
    random_seed=42
)

# 5️⃣ Ağı eğit
print("🧠 SOM eğitiliyor... (biraz sürebilir)")
som.train_random(X_scaled, num_iteration=10000)
print("✅ Eğitim tamamlandı!\n")

# 6️⃣ SOM çıktısını K-Means ile kümele
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)
df['cluster'] = cluster_labels

# 7️⃣ Küme bazlı firmaları göster
print("📊 Kümeleme Sonuçları:")
for i in range(3):
    firmalar = df[df['cluster'] == i]['airline'].values
    print(f"\nKüme {i} ({len(firmalar)} firma):")
    for f in firmalar:
        print(f" - {f}")

# 8️⃣ Küme ortalamaları (istatistiksel özet)
print("\n📈 Küme Ortalamaları:")
means = df.groupby('cluster')[
    ['fatal_accidents_85_99', 'fatal_accidents_00_14',
     'fatalities_85_99', 'fatalities_00_14']
].mean()
print(means)

# 9️⃣ En güvenli kümeyi bul
best_cluster = means.mean(axis=1).idxmin()
print(f"\n✅ En güvenli havayolları kümesi: {best_cluster}")
print("✈️ Bu kümeye ait firmalar:")
for f in df[df['cluster'] == best_cluster]['airline'].values:
    print(f" - {f}")
