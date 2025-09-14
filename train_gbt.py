# train_gbt.py
import numpy as np, pandas as pd, joblib, json
from sklearn.model_selection import train_test_split
import lightgbm as lgb

# Load embeddings + mean colors
emb = np.load("embeddings.npy")
mc = pd.read_csv("mean_colors.csv").values
X = np.hstack([emb, mc])

# Load label map from JSON
with open("label_map.json", "r") as f:
    label_map = json.load(f)

df = pd.read_csv("labels_clean.csv")
y = df["occasion"].map(lambda x: label_map[x]).values

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("🔹 Training LightGBM on", X_train.shape[0], "samples...")

# LightGBM model with accuracy metric
clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05)

clf.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="multi_logloss",
    callbacks=[lgb.early_stopping(30), lgb.log_evaluation(20)]
)

# Save model
joblib.dump(clf, "gbt_occ.pkl")
print("✅ Saved gbt_occ.pkl")
