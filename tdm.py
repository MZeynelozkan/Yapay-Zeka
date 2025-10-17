# -----------------------------
# 🧮 Term-Document Matrix (TDM)
# -----------------------------
# Bu kod: Belgelerdeki kelimelerin tekrar sayısını hesaplayıp
# bir matris şeklinde gösterir.
# -----------------------------

# 1️⃣ Belgeleri tanımla
docs = [
    "Ben elma yedim",
    "Elma çok güzel",
    "Ben güzel elmayı sevdim"
]

# 2️⃣ Belgeleri kelimelere ayır (Tokenization)
tokenized_docs = [doc.lower().split() for doc in docs]
# Çıktı örneği:
# [['ben', 'elma', 'yedim'], ['elma', 'çok', 'güzel'], ['ben', 'güzel', 'elmayı', 'sevdim']]

# 3️⃣ Kelime sözlüğünü (vocab) oluştur
vocab = sorted(set(word for doc in tokenized_docs for word in doc))
# Vocab örneği: ['ben', 'çok', 'elma', 'elmayı', 'güzel', 'sevdim', 'yedim']

# 4️⃣ Boş matris oluştur (belge sayısı x kelime sayısı)
matrix = [[0 for _ in range(len(vocab))] for _ in range(len(docs))]

# 5️⃣ Her kelimenin belgede kaç kez geçtiğini say (frekans)
for i, doc in enumerate(tokenized_docs):   # belge üzerinde dön
    for word in doc:                       # her kelimeyi kontrol et
        if word in vocab:                  # vocab içinde varsa
            j = vocab.index(word)          # sütun indeksini bul
            matrix[i][j] += 1              # frekansı 1 artır

# 6️⃣ Sonucu pandas DataFrame ile göster (isteğe bağlı)
import pandas as pd
tdm = pd.DataFrame(matrix, columns=vocab)
print("\n📊 Term-Document Matrix:\n")
print(tdm)
