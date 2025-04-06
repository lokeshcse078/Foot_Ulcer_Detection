import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# Load trained model
MODEL_PATH = "thermal_ulcer_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Define image size
IMG_SIZE = (224, 224)

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize(IMG_SIZE)  # Resize to match model input size
    image = img_to_array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit UI
st.title("Thermal Ulcer Detection")
st.write("Upload a thermal image to predict whether it has a foot ulcer or not.")

uploaded_file = st.file_uploader("Choose a thermal image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = load_img(uploaded_file)  # Load image
    st.image(image, caption="Uploaded Image", use_container_width=True)
    

    processed_image = preprocess_image(image)  # Preprocess
    prediction = model.predict(processed_image)[0][0]  # Predict
    
    # Display result
    if prediction > 0.5:
        st.error("Prediction: Foot ulcer detected! ⚠️")
    else:
        st.success("Prediction: No foot ulcer detected! ✅")
