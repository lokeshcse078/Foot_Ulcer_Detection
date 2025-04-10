import os
import requests
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
import random
import sqlite3

# Constants
MODEL_URL = "https://github.com/lokeshcse078/Foot_Ulcer_Detection/releases/download/v1.0/model.h5"
MODEL_PATH = "model.h5"
IMG_SIZE = (224, 224)
DB_FILE = "otp_store.db"

# Background image URL
BACKGROUND_IMAGE_URL = "https://github.com/lokeshcse078/Foot_Ulcer_Detection/blob/main/bg.jpg"

# Email credentials
EMAIL_USER = "lokeshkumar.cse.078@gmail.com"
EMAIL_PASS = "wwpo fizj fhxp wbbp"

@st.cache_resource
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("🔄 Downloading model..."):
            response = requests.get(MODEL_URL)
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = download_model()

def preprocess_image(image):
    image = image.resize(IMG_SIZE)
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Setup OTP table
def create_otp_table():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS otp_codes (
            email TEXT,
            otp TEXT,
            expiry TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

create_otp_table()

def send_otp(email):
    otp = str(random.randint(100000, 999999))
    expiry = datetime.now() + timedelta(minutes=5)
    expiry_str = expiry.strftime("%Y-%m-%d %H:%M:%S")

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO otp_codes (email, otp, expiry) VALUES (?, ?, ?)", (email, otp, expiry_str))
    conn.commit()
    conn.close()

    subject = "Your OTP Code"
    body = f"Your OTP code is {otp}. It is valid for 5 minutes."
    msg = MIMEMultipart()
    msg["From"] = EMAIL_USER
    msg["To"] = email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASS)
        server.sendmail(EMAIL_USER, email, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        st.error(f"Error sending OTP: {e}")
        return False

def verify_otp(email, otp):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT otp, expiry FROM otp_codes WHERE email = ? ORDER BY expiry DESC LIMIT 1", (email,))
    result = c.fetchone()
    conn.close()
    if result:
        stored_otp, expiry = result
        expiry_datetime = datetime.strptime(expiry, "%Y-%m-%d %H:%M:%S")
        if stored_otp == otp and datetime.now() < expiry_datetime:
            return True
    return False

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

# Session State Management
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "email" not in st.session_state:
    st.session_state.email = ""

# OTP Login Flow
if not st.session_state.logged_in:
    if not st.session_state.email:
        email = st.text_input("Enter your email")
        if st.button("Send OTP"):
            if "@" in email and "." in email:
                if send_otp(email):
                    st.session_state.email = email
                    st.success("OTP sent to your email.")
                    st.rerun()
            else:
                st.warning("Please enter a valid email address.")
    else:
        otp = st.text_input("Enter the OTP sent to your email")
        if st.button("Verify OTP"):
            if verify_otp(st.session_state.email, otp):
                st.success("Login successful!")
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid OTP or OTP expired.")
        if st.button("Resend OTP"):
            send_otp(st.session_state.email)
            st.success("OTP resent.")
else:
    st.success(f"Welcome, {st.session_state.email}!")
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.email = ""
        st.rerun()

    uploaded_file = st.file_uploader("Upload thermal image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = load_img(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        processed = preprocess_image(image)
        prediction = model.predict(processed)[0][0]

        st.subheader("Prediction Result")
        if prediction > 0.5:
            st.error("Prediction: Foot ulcer detected! ⚠️")
        else:
            st.success("Prediction: No foot ulcer detected ✅")
