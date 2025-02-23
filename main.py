import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image, UnidentifiedImageError
import os

def set_background():
    st.markdown(
        """
        <style>
        .stApp {
            background: url("https://img.freepik.com/free-photo/fedora-hat-studio-with-copy-space_23-2150737081.jpg") no-repeat center center fixed;
            background-size: cover;
        }
        /* Remove overlay if not needed */
        /* .stApp::before {
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(255, 255, 255, 0.2); 
            z-index: -1;
        } */
        
        /* Make all text black */
        h1, h2, h3, h4, h5, h6, p, .stMarkdown, .stTextInput label, .stFileUploader label {
            color: black !important;
        }

        /* Ensure buttons and inputs are visible */
        .stButton>button {
            background-color: #fff !important;
            color: black !important;
            border: 1px solid black !important;
        }

        </style>
        """,
        unsafe_allow_html=True
    )

# Set background image
set_background()

# Load model safely
MODEL_PATH = "main\mobilenetv2_finetuned_model.h5"

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found! Check the file path.")
else:
    loaded_model = tf.keras.models.load_model(MODEL_PATH)

st.title('Hat Image Classification')

# Image upload options
genre = st.radio("How You Want To Upload Your Image", ('Browse Photos', 'Camera'))

if genre == 'Camera':
    ImagePath = st.camera_input("Take a picture")
else:
    ImagePath = st.file_uploader("Choose a file", type=['jpeg', 'jpg', 'png'])

if ImagePath is not None:
    try:
        # Open image with PIL
        image_ = Image.open(ImagePath)
        st.image(image_, width=250, caption="Uploaded Image")

        # Process and predict when button is clicked
        if st.button('Predict'):
            loaded_single_image = image_.resize((224, 224))  # Resize for model input
            test_image = np.array(loaded_single_image) / 255.0  # Normalize
            test_image = np.expand_dims(test_image, axis=0)  # Add batch dimension

            # Model prediction
            logits = loaded_model.predict(test_image)
            softmax = tf.nn.softmax(logits)
            predict_output = tf.argmax(logits, -1).numpy()[0]

            # CIFAR-10 class labels
            classes = [
                    "Ascot Cap", "Baseball Cap", "Beret", "Bicorne", "Boater", "Bowler", "Deerstalker",
                    "Fedora", "Fez", "Football Helmet", "Garrison Cap", "Hard Hat", "Military Helmet",
                    "Mortarboard", "Pith Helmet", "Pork Pie", "Sombrero", "Southwester", "Top Hat", "Zucchetto"
                ]
            predicted_class = classes[predict_output]
            probability = softmax.numpy()[0][predict_output] * 100

            # Display result
            st.header(f"Prediction: {predicted_class}")
            st.subheader(f"Probability: {probability:.2f}%")

    except UnidentifiedImageError:
        st.error('Invalid image format! Please upload a valid JPEG, JPG, or PNG file.')
