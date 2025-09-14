# predict_utils.py
import json, torch, numpy as np, pandas as pd
from PIL import Image
from torchvision import transforms
from train import SimpleCNN   # uses the model class you trained

# load label map
with open("label_map.json","r") as f:
    label_map = json.load(f)
inv_label_map = {int(v): k for k,v in label_map.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes=len(label_map)).to(device)
model.load_state_dict(torch.load("outfit_model.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
])

import torch.nn.functional as F
def predict_probs_from_pil(pil_img):
    img = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(img)                # raw logits
        probs = F.softmax(out, dim=1).cpu().numpy()[0]
    # return dict label->prob
    return {inv_label_map[i]: float(probs[i]) for i in range(len(probs))}

# For embeddings (needed for GBT / clustering)
def get_embedding_from_pil(pil_img):
    img = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        # replicate SimpleCNN steps up to fc1 to get feature vector
        x = model.pool(model.relu(model.conv1(img)))
        x = model.pool(model.relu(model.conv2(x)))
        x = x.view(x.size(0), -1)
        emb = model.relu(model.fc1(x))
    return emb.cpu().numpy()[0]  # 1D numpy array
