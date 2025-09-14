# extract_embeddings.py
import pandas as pd, numpy as np, os
from PIL import Image
from predict_utils import get_embedding_from_pil
from tqdm import tqdm

df = pd.read_csv("labels_clean.csv")
embs = []
mean_colors = []

for i,row in tqdm(df.iterrows(), total=len(df)):
    path = row["image_path"]
    try:
        img = Image.open(path).convert("RGB")
    except:
        print("skip", path); embs.append(np.zeros(128)); mean_colors.append([0,0,0]); continue
    emb = get_embedding_from_pil(img)
    arr = np.array(img.resize((64,64)))  # small thumbnail
    r,g,b = arr[:,:,0].mean(), arr[:,:,1].mean(), arr[:,:,2].mean()
    embs.append(emb)
    mean_colors.append([r,g,b])

embs = np.vstack(embs)
mc = np.vstack(mean_colors)
np.save("embeddings.npy", embs)
pd.DataFrame(mc, columns=["mean_r","mean_g","mean_b"]).to_csv("mean_colors.csv", index=False)
print("Saved embeddings.npy and mean_colors.csv")
