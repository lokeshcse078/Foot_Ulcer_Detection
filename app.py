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
import base64

# Constants
MODEL_URL = "https://github.com/lokeshcse078/Foot_Ulcer_Detection/releases/download/v1.0/model.h5"
MODEL_PATH = "model.h5"
IMG_SIZE = (224, 224)
DB_URL = "https://raw.githubusercontent.com/your-username/your-repository/main/users.db"  # Update with the correct URL
DB_PATH = "users.db"

# Background image URL (change this to the URL of your background image)
BACKGROUND_IMAGE_URL = "https://github.com/lokeshcse078/Foot_Ulcer_Detection/blob/main/bg.jpg"  # Update with the actual image URL

# GitHub token (use environment variable for security)
GITHUB_TOKEN = "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"
# Email credentials (replace with env vars ideally)
EMAIL_USER = "lokeshkumar.cse.078@gmail.com"
EMAIL_PASS = "wwpo fizj fhxp wbbp"  # Replace this with a secure method


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
            response = requests.get(DB_URL, headers={'Authorization': f'token {GITHUB_TOKEN}'})
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

# Function to verify password from GitHub DB
def verify_user_from_github(email, password):
    headers = {
        'Authorization': f'token {GITHUB_TOKEN}',
        'Accept': 'application/vnd.github.v3+json'
    }

    # Get the content of the DB file from GitHub
    response = requests.get(DB_URL, headers=headers)

    if response.status_code == 200:
        try:
            # Decode the file content from base64
            response_json = response.json()
            content_b64 = response_json.get('content', '')
            decoded_content = base64.b64decode(content_b64).decode('utf-8')

            # Parse the decoded content to search for the user data
            users = decoded_content.split('\n')  # Assuming each user is on a new line
            for user in users:
                user_data = user.strip()
                if user_data:
                    stored_email, stored_hashed_password = user_data.split(',')  # Assuming CSV format
                    if stored_email == email and bcrypt.checkpw(password.encode(), stored_hashed_password.encode()):
                        return True  # Email and password match
            return False  # No match found
        except Exception as e:
            st.error(f"Error processing database: {e}")
            return False
    else:
        st.error(f"GitHub API Error: {response.status_code} - {response.text}")
        return False


def send_otp(email):
    otp = str(random.randint(100000, 999999))
    expiry = datetime.now() + timedelta(minutes=5)
    
    # Convert expiry to string in a format compatible with SQLite
    expiry_str = expiry.strftime("%Y-%m-%d %H:%M:%S")

    conn = sqlite3.connect("users.db")
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

def update_github_db(email, hashed_password):
    headers = {
        'Authorization': f'token {GITHUB_TOKEN}',
        'Accept': 'application/vnd.github.v3+json'
    }

    # Get the current file content (required for sha)
    response = requests.get(DB_URL, headers=headers)

    if response.status_code == 200:
        try:
            # Decode the file content from base64
            response_json = response.json()
            file_sha = response_json.get('sha')
            content_b64 = response_json.get('content', '')
            decoded_content = base64.b64decode(content_b64).decode('utf-8')

            # Append new user to the DB content
            new_user_entry = f"{email},{hashed_password.decode()}"
            updated_content = decoded_content + "\n" + new_user_entry

            # Re-encode the content to base64
            updated_content_b64 = base64.b64encode(updated_content.encode()).decode()

            # Prepare the data for the GitHub API
            data = {
                "message": "Register new user",
                "committer": {
                    "name": "Your Name",
                    "email": "your-email@example.com"
                },
                "content": updated_content_b64,
                "sha": file_sha
            }

            # Send the PUT request to update the file on GitHub
            update_response = requests.put(DB_URL, json=data, headers=headers)

            if update_response.status_code == 200:
                st.success("User registered and database updated on GitHub!")
            else:
                st.error(f"Error updating GitHub DB: {update_response.text}")
        except Exception as e:
            st.error(f"Error processing database: {e}")
    else:
        st.error(f"GitHub API Error: {response.status_code} - {response.text}")


# Inject custom CSS for background image
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("{BACKGROUND_IMAGE_URL}");
        background-size: cover;
        background-position: center;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

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
    
            # Check if email format is valid (basic check)
            if "@" not in email or "." not in email:
                st.warning("Please enter a valid email address.")
        
            if st.button("Send OTP"):
                # Ensure that email is not empty and is valid
                if email and "@" in email and "." in email:
                    # Call the send_otp function
                    if send_otp(email):
                        # Store email and password temporarily in session state for later use
                        st.session_state.registering = True
                        st.session_state.temp_email = email
                        st.session_state.temp_password = password
                        st.success("OTP has been sent to your email!")
                        st.rerun()  # Re-run the app to allow OTP input
                    else:
                        st.error("Failed to send OTP. Please try again.")
                else:
                    st.warning("Please enter a valid email address.")        
        elif auth_option == "Login":
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            if st.button("Login"):
                if verify_user_from_github(email, password):
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
                hashed_password = bcrypt.hashpw(st.session_state.temp_password.encode(), bcrypt.gensalt())
                register_user(email, st.session_state.temp_password)
                update_github_db(email, hashed_password)  # Update on GitHub
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
        st.rerun()

    uploaded_file = st.file_uploader("Upload thermal image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = load_img(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        processed = preprocess_image(image)
        prediction = model.predict(processed)[0][0]  # Binary output

        st.subheader("Prediction Result")
        if prediction > 0.5:
            st.error("Prediction: Foot ulcer detected! ⚠️")
        else:
            st.success("Prediction: No foot ulcer detected ✅")
