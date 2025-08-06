import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import requests
import os

# --------------------
# Configuration
# --------------------
MODEL_URL = "https://github.com/yourusername/yourrepo/raw/main/dfu_model.pth"  # Replace with your actual URL
MODEL_PATH = "dfu_model.pth"
IMG_SIZE = (224, 224)

# --------------------
# Download the model if not present
# --------------------
@st.cache_resource
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model from GitHub..."):
            response = requests.get(MODEL_URL)
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)

# --------------------
# Image Preprocessing
# --------------------
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# --------------------
# Main App
# --------------------
st.title("🦶 DFU Detection from Thermal Image")
st.write("Upload a thermal image of a foot to check for possible Diabetic Foot Ulcers.")

# Load model
download_model()
model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
model.eval()

# Image Upload
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(input_tensor).item()

    # Display result
    st.subheader("🧠 Prediction Result")
    if output > 0.5:
        st.error(f"⚠ Foot ulcer detected (Confidence: {output*100:.2f}%)")
    else:
        st.success(f"✅ No foot ulcer detected (Confidence: {(1-output)*100:.2f}%)")
