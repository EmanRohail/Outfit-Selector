import os
import pandas as pd

# Paths (use the full path to your dataset)
base_dir = r"C:\Users\hp\Downloads\clothing-dataset-small-master\clothing-dataset-small-master"

splits = ["train", "test", "validation"]

data = []

# Walk through each split
for split in splits:
    split_dir = os.path.join(base_dir, split)
    for category in os.listdir(split_dir):
        category_dir = os.path.join(split_dir, category)
        if os.path.isdir(category_dir):
            for img in os.listdir(category_dir):
                img_path = os.path.join(split_dir, category, img)

                # Map category -> occasion
                if category in ["tshirt", "jeans"]:
                    occasion, formality = "casual", 1
                elif category in ["shirt", "blazer"]:
                    occasion, formality = "office", 3
                elif category in ["dress"]:
                    occasion, formality = "party", 4
                else:
                    occasion, formality = "casual", 2  # fallback

                data.append([img_path, category, split, occasion, formality])

# Save CSV
df = pd.DataFrame(data, columns=["image_path", "category", "split", "occasion", "formality"])
df.to_csv("labels_clean.csv", index=False)

print("✅ labels_clean.csv created with", len(df), "rows")
