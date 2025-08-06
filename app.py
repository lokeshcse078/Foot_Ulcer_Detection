import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import streamlit as st

# Define your model architecture here
class FootUlcerModel(nn.Module):
    def __init__(self):
        super(FootUlcerModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(32 * 64 * 64, 2)  # adjust according to your image size
        )

    def forward(self, x):
        return self.model(x)

# Load the model
MODEL_PATH = "model.pth"  # Replace with actual path

model = FootUlcerModel()
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# Define preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Streamlit UI
st.title("Foot Ulcer Detection from Thermal Image")

uploaded_file = st.file_uploader("Upload Thermal Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)

    label = "Ulcer Detected" if predicted.item() == 1 else "Healthy Foot"
    st.success(f"Prediction: {label}")
