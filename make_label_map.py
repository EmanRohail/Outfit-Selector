import pandas as pd
import json

df = pd.read_csv("labels_clean.csv")

# Unique occasions (casual, office, party)
label_map = {label: idx for idx, label in enumerate(df["occasion"].unique())}

with open("label_map.json", "w") as f:
    json.dump(label_map, f)

print("✅ label_map.json created:", label_map)
