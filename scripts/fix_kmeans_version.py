import joblib
from sklearn.cluster import MiniBatchKMeans

print("🔹 Reloading km.bin ...")
km = joblib.load("data/kmeans/km.bin")
joblib.dump(km, "data/kmeans/km_fixed.pkl")
print("✅ Re-saved as data/kmeans/km_fixed.pkl (current sklearn version compatible)")
