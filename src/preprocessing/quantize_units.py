import numpy as np
from sklearn.cluster import MiniBatchKMeans
from pathlib import Path
from tqdm import tqdm
import joblib


def train_kmeans(feature_dir, kmeans_out, n_clusters=100, batch_size=5000):
    """
    Train MiniBatchKMeans on sampled HuBERT features.
    """
    feature_dir = Path(feature_dir)
    all_features = []

    print("🧠 Collecting feature samples for KMeans training...")
    np.random.seed(42)

    # Sample a few frames per file to keep memory small
    for f in tqdm(list(feature_dir.rglob("*.npy"))[:500]):
        feats = np.load(f)
        if feats.ndim == 2:
            n = min(200, len(feats))  # sample up to 200 frames
            idx = np.random.choice(len(feats), n, replace=False)
            all_features.append(feats[idx])

    all_features = np.vstack(all_features)
    print(f"Collected {all_features.shape[0]} samples of dimension {all_features.shape[1]}.")

    print(f"⚙️ Training MiniBatchKMeans with {n_clusters} clusters...")
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, verbose=1)
    kmeans.fit(all_features)
    joblib.dump(kmeans, kmeans_out)
    print(f"✅ Saved KMeans model: {kmeans_out}")


def quantize_features(feature_dir, kmeans_model, output_dir, deduplicate=True):
    """
    Quantize HuBERT features into discrete unit IDs and save.
    """
    feature_dir = Path(feature_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    kmeans = joblib.load(kmeans_model)
    print(f"✅ Loaded KMeans model from {kmeans_model}")

    for f in tqdm(list(feature_dir.rglob("*.npy")), desc="Quantizing features"):
        feats = np.load(f)
        unit_ids = kmeans.predict(feats)

        if deduplicate:
            # remove consecutive duplicates
            unit_ids = [unit_ids[0]] + [u for i, u in enumerate(unit_ids[1:]) if u != unit_ids[i]]
            unit_ids = np.array(unit_ids)

        rel_path = f.relative_to(feature_dir)
        subdir = output_dir / rel_path.parent
        subdir.mkdir(parents=True, exist_ok=True)

        out_path = subdir / f"{f.stem.replace('_hubert','')}_units.npy"
        np.save(out_path, unit_ids)

    print(f"🎯 Quantized units saved to: {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_kmeans", action="store_true")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--feature_dir", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--kmeans_out", default="data/kmeans/kmeans_100.pkl")
    parser.add_argument("--kmeans_model", default=None)
    parser.add_argument("--n_clusters", type=int, default=100)
    args = parser.parse_args()

    if args.train_kmeans:
        train_kmeans(args.feature_dir, args.kmeans_out, args.n_clusters)
    elif args.quantize:
        if not args.kmeans_model:
            raise ValueError("--kmeans_model required for quantization")
        quantize_features(args.feature_dir, args.kmeans_model, args.output_dir)
