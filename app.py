import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import requests
import os

# --------------------
# Configuration
# --------------------
MODEL_URL = "https://github.com/lokeshcse078/Foot_Ulcer_Detection/releases/download/v1.0/trained_model.pth"
MODEL_PATH = "trained_model.pth"
IMG_SIZE = (224, 224)

# --------------------
# Download the model
# --------------------
@st.cache_resource
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            response = requests.get(MODEL_URL)
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)

# --------------------
# Model architecture must match the training code
# --------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.net(x)

# --------------------
# Image Preprocessing
# --------------------
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# --------------------
# Main App
# --------------------
st.title("🦶 DFU Detection from Thermal Image")
st.write("Upload a thermal image of a foot to check for Diabetic Foot Ulcers.")

# Load model
download_model()
model = SimpleCNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

# Image Upload
uploaded_file = st.file_uploader("Upload a foot image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_tensor = preprocess_image(image)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[0][predicted_class].item()

    st.subheader("🔍 Prediction Result")
    if predicted_class == 1:
        st.error(f"⚠ Foot ulcer detected (Confidence: {confidence * 100:.2f}%)")
    else:
        st.success(f"✅ No foot ulcer detected (Confidence: {confidence * 100:.2f}%)")
