import os
import requests
import streamlit as st
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

# Constants
MODEL_URL = "https://github.com/lokeshcse078/Foot_Ulcer_Detection/releases/tag/v1.0/model.pth"  # Update with actual URL
MODEL_PATH = "model.pth"
IMG_SIZE = (224, 224)
BACKGROUND_IMAGE_URL = "https://github.com/lokeshcse078/Foot_Ulcer_Detection/blob/main/bg.jpg?raw=true"

# Download model if not exists
@st.cache_resource
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("🔄 Downloading model..."):
            response = requests.get(MODEL_URL)
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
    model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    model.eval()
    return model

model = download_model()

# Preprocess function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    return transform(image).unsqueeze(0)

# UI Styling
st.markdown(f"""
    <style>
    .stApp {{
        background-image: url('{BACKGROUND_IMAGE_URL}');
        background-size: cover;
        background-position: center;
    }}
    </style>
""", unsafe_allow_html=True)

st.title("Thermal Foot Ulcer Detection")

uploaded_file = st.file_uploader("Upload thermal image...", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    processed = preprocess_image(image)
    with torch.no_grad():
        output = model(processed)
        prediction = torch.sigmoid(output).item()

    st.subheader("Prediction Result")
    if prediction > 0.5:
        st.error("Prediction: Foot ulcer detected! \u26A0\uFE0F")
    else:
        st.write(f"Confidence: {(1 - prediction)*100:.2f}%")
        st.success("Prediction: No foot ulcer detected \u2705")
