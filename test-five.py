import pandas as pd
import simpsom as sps
from sklearn.cluster import KMeans
import numpy as np

# CSV dosya yolu

file_path = "/Users/mehmetzeynelozkan/Desktop/Yapay-Zeka/Mall_Customers.csv"

# Veriyi oku
veri = pd.read_csv(file_path)
print(veri.head())
# => veri artık tüm sütunları ve her havayoluna ait istatistikleri içeriyor

# Analiz için kullanılacak sayısal özellikleri seçiyoruz
X = veri.drop(["Genre", "Spending Score (1-100)"], axis=1)
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
