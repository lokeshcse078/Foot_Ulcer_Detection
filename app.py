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
import hashlib
from supabase import create_client, Client

# Supabase credentials
SUPABASE_URL = st.secrets["supabase"]["url"]
SUPABASE_KEY = st.secrets["supabase"]["key"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Model & constants
MODEL_URL = st.secrets["model"]["url"]
MODEL_PATH = st.secrets["model"]["path"]
IMG_SIZE = (224, 224)

# Email credentials
EMAIL_USER = st.secrets["email"]["user"]
EMAIL_PASS = st.secrets["email"]["pass"]

# Background image
BACKGROUND_IMAGE_URL = "https://github.com/lokeshcse078/Foot_Ulcer_Detection/blob/main/bg.jpg?raw=true"

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

# OTP Handling
if "otp_store" not in st.session_state:
    st.session_state.otp_store = {}

def send_otp(email):
    otp = str(random.randint(100000, 999999))
    expiry = datetime.now() + timedelta(minutes=5)
    st.session_state.otp_store[email] = (otp, expiry)

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
    if email in st.session_state.otp_store:
        stored_otp, expiry = st.session_state.otp_store[email]
        if stored_otp == otp and datetime.now() < expiry:
            return True
    return False

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

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
if "otp_sent" not in st.session_state:
    st.session_state.otp_sent = False
if "otp_verified" not in st.session_state:
    st.session_state.otp_verified = False
if "user_name" not in st.session_state:
    st.session_state.user_name = ""

# Authentication
if not st.session_state.logged_in:
    auth_mode = st.radio("Select Mode", ["Login", "Register"])
    email = st.text_input("Email")

    if auth_mode == "Login":
        if email:
            user_query = supabase.table("user").select("*").eq("email", email).execute()
            user_exists = len(user_query.data) > 0

            if user_exists:
                password = st.text_input("Password", type="password")
                if st.button("Login"):
                    user = user_query.data[0]
                    if hash_password(password) == user["password"]:
                        st.session_state.logged_in = True
                        st.session_state.email = email
                        st.session_state.user_name = user.get("name", "")
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Incorrect password.")
            else:
                st.warning("No account found. Please register first.")

    elif auth_mode == "Register":
        if email:
            if not st.session_state.otp_sent:
                if st.button("Send OTP"):
                    if "@" in email and "." in email:
                        if send_otp(email):
                            st.session_state.email = email
                            st.session_state.otp_sent = True
                            st.success("OTP sent to your email.")
                            st.rerun()
                    else:
                        st.warning("Please enter a valid email address.")

        if st.session_state.otp_sent and not st.session_state.otp_verified:
            otp = st.text_input("Enter OTP sent to your email")
            if st.button("Verify OTP"):
                if verify_otp(st.session_state.email, otp):
                    st.session_state.otp_verified = True
                    st.success("OTP verified. Please set a password and name.")
                    st.rerun()
                else:
                    st.error("Invalid or expired OTP.")

        if st.session_state.otp_verified:
            name = st.text_input("Full Name")
            password = st.text_input("Set Password", type="password")
            
            if st.button("Register"):
                if name and password:
                    # Check if user already exists before inserting
                    existing = supabase.table("user").select("email").eq("email", st.session_state.email).execute()
                    if existing.data:
                        st.error("User already exists. Please login.")
                        st.session_state.otp_sent = False
                        st.session_state.otp_verified = False
                        st.rerun()
                    else:
                        try:
                            supabase.table("user").insert({
                                "email": st.session_state.email,
                                "name": name,
                                "password": hash_password(password),
                                "created_at": datetime.now().isoformat()
                            }).execute()
                            st.success("User registered successfully!")
                            st.session_state.logged_in = True
                            st.session_state.user_name = name
                            st.rerun()
                        except Exception as e:
                            st.error(f"Registration failed: {e}")
                else:
                    st.warning("Please fill in all fields.")


else:
    # Main App
    st.success(f"Welcome, {st.session_state.user_name or st.session_state.email}!")
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.email = ""
        st.session_state.otp_sent = False
        st.session_state.otp_verified = False
        st.session_state.user_name = ""
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
            st.write(f"Confidence: {prediction*100:.2f}%")
            st.success("Prediction: No foot ulcer detected ✅")
