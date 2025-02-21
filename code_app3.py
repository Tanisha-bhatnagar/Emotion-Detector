import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Emotion labels
emotion_labels = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Sad',
    5: 'Surprise',
    6: 'Neutral'
}

# Load the trained model
model = load_model('emotion_detector_model.h5')

# Custom CSS for better UI
st.markdown("""
    <style>
        .stApp {
            background-color: #0d1b2a;
            color: white;
        }
        .main-title {
            font-size: 48px;
            color: #e0e1dd;
            font-weight: bold;
            text-align: center;
            margin-bottom: 20px;
        }
        .sub-title {
            font-size: 20px;
            color: #778da9;
            text-align: center;
            margin-bottom: 30px;
        }
        .stFileUploader {
            margin-top: 20px;
            background-color: rgba(255, 255, 255, 0.1);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }
        .stImage {
            border-radius: 15px;
            box-shadow: 0px 4px 8px rgba(255, 255, 255, 0.2);
        }
        .emotion-box {
            background-color: #415a77;
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            color: #ffffff;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-title">Emotion Detector</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Upload an image to detect the emotion.</p>', unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"], help="Upload a clear facial image for better accuracy.")

if uploaded_file is not None:
    # Load and preprocess image
    img = image.load_img(uploaded_file, target_size=(48, 48), color_mode='grayscale')
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict emotion
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=-1)
    predicted_emotion = emotion_labels[predicted_class[0]]
    
    # Display uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True, output_format='auto', channels='RGB')
    
    # Display prediction
    st.markdown(f'<div class="emotion-box">Prediction: {predicted_emotion}</div>', unsafe_allow_html=True)
