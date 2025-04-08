import os
import requests
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import smtplib
import random
import bcrypt
import sqlite3
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta

# Constants
MODEL_URL = "https://github.com/lokeshcse078/Foot_Ulcer_Detection/releases/download/v1.0/model.h5"
MODEL_PATH = "model.h5"
IMG_SIZE = (224, 224)
DB_URL = "https://raw.githubusercontent.com/your-username/your-repository/main/users.db"  # Update with the correct URL
DB_PATH = "users.db"

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

# Preprocess image
def preprocess_image(image):
    image = image.resize(IMG_SIZE)
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Function to download the database if not already present
def download_db():
    if not os.path.exists(DB_PATH):
        with st.spinner("🔄 Downloading database..."):
            response = requests.get(DB_URL)
            with open(DB_PATH, "wb") as f:
                f.write(response.content)

# Download the DB at the start of the app
download_db()

# Database setup
def create_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE,
            password TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS otp_codes (
            email TEXT,
            otp TEXT,
            expiry TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

create_db()

# Email credentials (replace with env vars ideally)
EMAIL_USER = "lokeshkumar.cse.078@gmail.com"
EMAIL_PASS = "wwpo fizj fhxp wbbp"  # Replace this with a secure method

def send_otp(email):
    otp = str(random.randint(100000, 999999))
    expiry = datetime.now() + timedelta(minutes=5)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO otp_codes (email, otp, expiry) VALUES (?, ?, ?)", (email, otp, expiry))
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

def register_user(email, password):
    hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (email, password) VALUES (?, ?)", (email, hashed_pw))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def verify_user(email, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE email = ?", (email,))
    result = c.fetchone()
    conn.close()
    if result:
        return bcrypt.checkpw(password.encode(), result[0])
    return False

def verify_otp(email, otp):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT otp, expiry FROM otp_codes WHERE email = ? ORDER BY expiry DESC LIMIT 1", (email,))
    result = c.fetchone()
    conn.close()

    if result:
        stored_otp, expiry = result
        try:
            expiry_datetime = datetime.strptime(expiry, "%Y-%m-%d %H:%M:%S.%f")
        except ValueError:
            expiry_datetime = datetime.strptime(expiry, "%Y-%m-%d %H:%M:%S")
        if stored_otp == otp and datetime.now() < expiry_datetime:
            return True
    return False

# UI
st.title("Thermal Foot Ulcer Detection")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "registering" not in st.session_state:
    st.session_state.registering = False

if not st.session_state.logged_in:
    if not st.session_state.registering:
        auth_option = st.radio("Select option:", ["Register", "Login"])

        if auth_option == "Register":
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            if st.button("Send OTP"):
                send_otp(email)
                st.session_state.registering = True
                st.session_state.temp_email = email
                st.session_state.temp_password = password
                st.rerun()

        elif auth_option == "Login":
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            if st.button("Login"):
                if verify_user(email, password):
                    st.session_state.logged_in = True
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid credentials.")

    else:
        email = st.session_state.get("temp_email", "")
        otp = st.text_input("Enter OTP")
        if st.button("Verify OTP"):
            if verify_otp(email, otp):
                register_user(email, st.session_state.temp_password)
                st.session_state.logged_in = True
                st.success("Registration & login successful!")
                st.session_state.registering = False
                st.rerun()
            else:
                st.error("Invalid OTP or expired.")
else:
    st.success("Logged in successfully!")
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.experimental_rerun()

    uploaded_file = st.file_uploader("Upload thermal image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = load_img(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        processed = preprocess_image(image)
        prediction = model.predict(processed)[0][0]  # Binary output

        st.subheader("Prediction Result")
        if prediction > 0.5:
            st.error("Prediction: Foot ulcer detected! ⚠️")
        else:
            st.success("Prediction: No foot ulcer detected ✅")
