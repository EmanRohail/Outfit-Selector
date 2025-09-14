import os

base_dir = r"C:\Users\hp\Downloads\clothing-dataset-small-master\clothing-dataset-small-master"
print("Looking in:", base_dir)
print("Exists? ", os.path.exists(base_dir))

if os.path.exists(base_dir):
    print("Contents:", os.listdir(base_dir))
