# predict.py
import os, sys, joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from train import SimpleCNN        # CNN model class
from predict_utils import get_embedding_from_pil  # embedding extractor

# -------------------
# 1. Load labels & mapping
# -------------------
df = pd.read_csv("labels_clean.csv")
if "occasion" not in df.columns or "formality" not in df.columns:
    raise SystemExit("labels_clean.csv must contain 'occasion' and 'formality' columns.")

label_list = list(df["occasion"].unique())
label_map = {label: idx for idx, label in enumerate(label_list)}
inv_label_map = {idx: label for label, idx in label_map.items()}

# Occasion → target formality
target_formality = df.groupby("occasion")["formality"].mean().to_dict()

# -------------------
# 2. Load CNN model
# -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes=len(label_map)).to(device)
model_path = "outfit_model.pth"
if not os.path.exists(model_path):
    raise SystemExit(f"Model file not found: {model_path}")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Transforms (must match training)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# -------------------
# 3. Load formality model if exists
# -------------------
formality_model = None
if os.path.exists("formality_reg.pkl"):
    formality_model = joblib.load("formality_reg.pkl")
    print("ℹ️ Formality model loaded.")

# -------------------
# 4. Prediction function
# -------------------
def predict_probs(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)
    
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    # CNN forward
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    probs_dict = {inv_label_map[i]: float(probs[i]) for i in range(len(probs))}

    # Adjust with formality model
    if formality_model:
        # Full features: CNN embedding + mean color
        embedding = get_embedding_from_pil(img)
        arr = np.array(img.resize((64, 64)))
        r, g, b = arr[:, :, 0].mean(), arr[:, :, 1].mean(), arr[:, :, 2].mean()
        mean_color = np.array([r, g, b])
        X_feat = np.hstack([embedding, mean_color]).reshape(1, -1)

        formality_pred = formality_model.predict(X_feat)[0]

        for occ in probs_dict:
            t_form = target_formality.get(occ, 2)
            penalty = 1 - abs(formality_pred - t_form)/4
            penalty = max(0, penalty)
            probs_dict[occ] *= penalty

        # Renormalize
        total = sum(probs_dict.values())
        if total > 0:
            probs_dict = {k: v / total for k, v in probs_dict.items()}

    return probs_dict

# -------------------
# 5. Sample test image if none provided
# -------------------
def sample_test_image():
    if "split" in df.columns and (df["split"] == "test").any():
        for p in df[df["split"] == "test"]["image_path"].tolist():
            if os.path.exists(p): return p
    for p in df["image_path"].tolist():
        if os.path.exists(p): return p
    return None

# -------------------
# 6. Main
# -------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        img_path = sample_test_image()
        if img_path is None:
            print("No valid image found in dataset.")
            sys.exit(1)

    print("Using image:", img_path)
    probs = predict_probs(img_path)

  # Show top-3 predictions in percentages
items = sorted(probs.items(), key=lambda x: x[1], reverse=True)
print("\nTop predictions (with formality adjustment if enabled):")
for label, p in items[:3]:
    print(f"  {label}: {p*100:.1f}%")
