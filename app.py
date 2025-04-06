import streamlit as st
import requests
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# Constants
MODEL_URL = "https://github.com/lokeshcse078/Foot_Ulcer_Detection/releases/download/v1.0/model.h5"
MODEL_PATH = "model.h5"
IMG_SIZE = (224, 224)
BG_IMAGE_URL = "https://raw.githubusercontent.com/lokeshcse078/Foot_Ulcer_Detection/main/bg.jpg"

# Background CSS
st.markdown(
    f"""
    <style>
    html, body, .stApp {{
        background-image: url("{BG_IMAGE_URL}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: white;
    }}

    .menu-btn {{
        position: fixed;
        top: 15px;
        left: 15px;
        z-index: 9999;
        background-color: rgba(0,0,0,0.6);
        color: white;
        border: none;
        padding: 10px 15px;
        border-radius: 8px;
        font-size: 18px;
        cursor: pointer;
    }}

    .menu-btn:hover {{
        background-color: rgba(255,255,255,0.2);
    }}
    </style>

    <script>
    function triggerSidebar() {{
        window.parent.postMessage({{ type: 'streamlit:toggleSidebar' }}, '*');
    }}
    </script>

    <button class="menu-btn" onclick="triggerSidebar()">☰</button>
    """,
    unsafe_allow_html=True
)

# Sidebar content (acts like popup)
with st.sidebar:
    st.header("📬 Help Menu")
    st.markdown("Need assistance?")
    st.markdown("📧 **Email:** [sit23cs078@sairamtap.edu.in](mailto:sit23cs078@sairamtap.edu.in)")

# Title
st.title("🦶 Thermal Ulcer Detection")
st.write("Upload a thermal image to check if a foot ulcer is detected.")

# Download the model if not already present
@st.cache_resource
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("🔄 Downloading model..."):
            response = requests.get(MODEL_URL)
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

# Load model
model = download_model()

# Image Preprocessing Function
def preprocess_image(image):
    image = image.resize(IMG_SIZE)
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# File upload
uploaded_file = st.file_uploader("📁 Choose a thermal image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = load_img(uploaded_file)
    st.image(image, caption="🖼 Uploaded Image", width=300)

    with st.spinner("🔍 Analyzing the image..."):
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)[0][0]

    # Display result
    if prediction > 0.5:
        st.error("⚠️ Prediction: Foot ulcer detected!")
    else:
        st.success("✅ Prediction: No foot ulcer detected!")
