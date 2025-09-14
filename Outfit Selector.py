import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import pandas as pd
import joblib
import os

from train import SimpleCNN  #  CNN class

# load labels and mapping
df = pd.read_csv("labels_clean.csv")
label_list = list(df["occasion"].unique())
label_map = {label: idx for idx, label in enumerate(label_list)}
inv_label_map = {idx: label for label, idx in label_map.items()}

#load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cnn_model = SimpleCNN(num_classes=len(label_map)).to(device)
cnn_model.load_state_dict(torch.load("outfit_model.pth", map_location=device))
cnn_model.eval()

# load GBT classifier 
gbt_model = None
if os.path.exists("gbt_occ.pkl"):
    gbt_model = joblib.load("gbt_occ.pkl")

# load formality regressor 
formality_model = None
if os.path.exists("formality_reg.pkl"):
    formality_model = joblib.load("formality_reg.pkl")

#transforms 
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# prediction function
def predict_probs(img: Image.Image):
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = cnn_model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    return {inv_label_map[i]: float(probs[i]) for i in range(len(probs))}

# Streamlit app 
st.title("👗 Outfit Selector")
st.write("Upload an outfit photo and get how much it suits an occasion.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
occasion_choice = st.selectbox("Select your target occasion", label_list)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Outfit", use_container_width=True)

    probs = predict_probs(img)
    top_preds = sorted(probs.items(), key=lambda x: x[1], reverse=True)

    st.subheader("Prediction Results")
    for label, p in top_preds:
        st.write(f"**{label}**: {p:.2%}")

    # when user picks an occasion, show the match %
    target_prob = probs.get(occasion_choice, 0)
    st.subheader("🎯 Occasion Match")
    st.write(f"This outfit suits **{occasion_choice}** with probability: **{target_prob:.2%}**")
