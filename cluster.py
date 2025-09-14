# cluster.py
from sklearn.cluster import KMeans
import numpy as np, joblib
emb = np.load("embeddings.npy")
kmeans = KMeans(n_clusters=8, random_state=42).fit(emb)
joblib.dump(kmeans, "kmeans_styles.pkl")
print("Saved kmeans_styles.pkl")
