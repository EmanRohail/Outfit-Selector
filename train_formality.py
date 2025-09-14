# train_formality.py
import numpy as np, pandas as pd, joblib
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

# Load features
emb = np.load("embeddings.npy")
mc = pd.read_csv("mean_colors.csv").values
X = np.hstack([emb, mc])  # embeddings + mean colors

# Load labels 
df = pd.read_csv("labels_clean.csv")
y = df["formality"].values  

#  Train / Test split 
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Model 
reg = lgb.LGBMRegressor(
    n_estimators=200,
    learning_rate=0.05,
    objective="regression"
)

print(f"Training LightGBM on {len(X_train)} samples...")
reg.fit(X_train, y_train)

# Evaluate 
y_pred = reg.predict(X_val)
from math import sqrt
rmse = sqrt(mean_squared_error(y_val, y_pred))

print(f"Validation RMSE: {rmse:.3f}")

# Save model
joblib.dump(reg, "formality_reg.pkl")
print("✅ Saved formality_reg.pkl")
