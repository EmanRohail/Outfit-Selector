# save_label_map.py
import pandas as pd, json
df = pd.read_csv("labels_clean.csv")
label_map = {label: idx for idx, label in enumerate(df["occasion"].unique())}
with open("label_map.json","w") as f:
    json.dump(label_map, f)
print("label_map.json saved:", label_map)

