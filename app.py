import streamlit as st
import requests
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# URL to the model hosted externally (update this with your actual model URL)
MODEL_URL = "https://github.com/lokeshcse078/Foot_Ulcer_Detection/releases/download/v1.0/model.h5"
MODEL_PATH = "model.h5"
IMG_SIZE = (224, 224)

# Download the model if it doesn't exist locally
if not os.path.exists(MODEL_PATH):
    st.write("Downloading model...")
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, 'wb') as f:
        f.write(response.content)
    st.write("Model downloaded successfully!")

# Load the model
model = tf.keras.models.load_model(MODEL_PATH)

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = image.resize(IMG_SIZE)
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Streamlit interface
st.title("Thermal Ulcer Detection")
st.write("Upload a thermal image to predict whether it has a foot ulcer or not.")

uploaded_file = st.file_uploader("Choose a thermal image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = load_img(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)[0][0]

    if prediction > 0.5:
        st.error("Prediction: Foot ulcer detected! ⚠️")
    else:
        st.success("Prediction: No foot ulcer detected! ✅")
