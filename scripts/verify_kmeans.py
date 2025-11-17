import joblib

kmeans_path = "data/kmeans/km.bin"
print(f"🔹 Loading KMeans model from {kmeans_path} ...")

km = joblib.load(kmeans_path)
print(f"✅ KMeans OK — n_clusters={km.n_clusters}")
